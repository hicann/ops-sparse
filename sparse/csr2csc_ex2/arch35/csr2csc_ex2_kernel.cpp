/**
 * ----------------------------------------------------------------------------------------------------------
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ----------------------------------------------------------------------------------------------------------
 */

/*!
 * \file csr2csc_ex2_kernel.cpp
 * \brief csr2csc_ex2 kernel 实现（SIMD 与 SIMT 混合编程）。
 *
 * 五个 kernel 的层次结构：
 *   Layer 1: __simt_vf__ 计算函数（线程级并行，grid-stride loop）
 *   Layer 2: Dispatcher 类（管理 GM 指针 + asc_vf_call 分发）
 *   Layer 3: kernel_do 启动器（<<<>>> 语法，Host 调用入口）
 *
 * Kernel 1 (CountCols):     遍历 csrColInd，asc_atomic_add 统计每 stripe 列直方图
 *                           （K1 优化：不再写 colCount，消除跨 block cache thrashing）
 * Kernel 1.5 (SumStripeHist): 对 stripeHist 按列求和重建 colCount（多线程并行）
 * Kernel 2 (PrefixSum):     单 warp(32线程) 并行 Blelloch scan -> cscColPtr
 *                           （K2 优化：单线程 O(n) -> 32 线程并行，shuffle 同步）
 * Kernel 3 (StripeBase):    stripe 直方图按列前缀和 -> 每 stripe 写游标基址（原地转换）
 * Kernel 4 (Scatter):       每 block 单线程顺序 scatter，游标私有（无原子竞争），
 *                           按 k 升序写保证列内行号有序（与 golden 逐位一致）。
 */

#include "kernel_operator.h"
#include "simt_api/asc_simt.h"
#include "simt_api/device_atomic_functions.h"
#include "csr2csc_ex2_kernel.h"

namespace {

// SIMT VF 线程数：min(work, kCsr2CscThreadsPerBlock)，非零保护，向上 warp 对齐
__aicore__ inline uint32_t ComputeSimtThreadNum(int32_t work)
{
    uint32_t simtThreadNum = kCsr2CscThreadsPerBlock;
    if (static_cast<uint32_t>(work) < simtThreadNum) {
        simtThreadNum = static_cast<uint32_t>(work);
    }
    if (simtThreadNum == 0) {
        simtThreadNum = 1;
    }
    simtThreadNum = (simtThreadNum + kCsr2CscWarpSize - 1u) &
        ~(kCsr2CscWarpSize - 1u);
    return simtThreadNum;
}

}  // namespace

// ---------------------------------------------------------------------------
// 防溢出说明（本文件多处 int64_t 转换的统一依据）：
//   stripe 索引与 (n+1) 的乘积可达 stripeCount * (n+1)，在 stripeCount 与 n 均较大时
//   会溢出 int32_t（如 stripeCount=448, n=4096 → 1.8M > 2^20，仍安全；但 stripeCount
//   与 n 同时取上限时 448*2^31 远超 int32）。因此所有 stripe * (n+1) 形式的地址偏移
//   统一用 int64_t 计算。同理 (stripe+1)*stripeSize 也用 int64_t 防 int32 溢出。
//   以下各处仅以"防溢出"简注，详细依据见本段说明。
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Kernel 1: CountCols — 遍历 csrColInd，原子加统计每 stripe 列直方图
//   依赖：stripeHist 由 host 侧 aclrtMemsetAsync 清零
//        （host.cpp SetupCsr2cscEx2Workspace），atomic_add 累加结果正确。
//
//   K1 性能优化：不再写 colCount（原每元素 2 次 GM atomic -> 1 次），
//   消除 colCount 的跨 block cache thrashing。colCount 由 Kernel 1.5
//   SumStripeHist 从 stripeHist 按列求和重建，结果等价（atomic_add 交换律）。
// ---------------------------------------------------------------------------

__simt_vf__ __aicore__ __launch_bounds__(kCsr2CscThreadsPerBlock) inline void
Csr2CscCountColsSimtCompute(
    __gm__ const int32_t *csrColInd,
    __gm__ int32_t *stripeHist,
    int32_t nnz, int32_t n, int32_t idxBase,
    int32_t stripeCount, int32_t stripeSize,
    int32_t numBlocks)
{
    int32_t globalTid = static_cast<int32_t>(threadIdx.x) +
        static_cast<int32_t>(blockIdx.x) * static_cast<int32_t>(blockDim.x);
    int32_t gridStride = static_cast<int32_t>(blockDim.x) * numBlocks;
    int32_t stripeStride = n + 1;

    // 循环前一次除法定位初始 stripe 与下一 stripe 边界，循环内以
    // "边界比较 + 加法推进" 增量维护 stripe（idx 单调递增，等价于 idx / stripeSize），
    // 避免热循环内每个元素执行一次运行时除法（stripeSize >= 1，除法安全）。
    int32_t stripe = globalTid / stripeSize;
    int64_t nextStripeBegin = static_cast<int64_t>(stripe + 1) * stripeSize; // 防溢出

    for (int32_t idx = globalTid; idx < nnz; idx += gridStride) {
        int32_t col = csrColInd[idx] - idxBase;
        while (static_cast<int64_t>(idx) >= nextStripeBegin) {
            stripe++;
            nextStripeBegin += stripeSize;
        }
        if (stripe >= stripeCount) {
            stripe = stripeCount - 1;
        }
        // 防溢出：stripe * stripeStride 用 int64_t
        asc_atomic_add(&stripeHist[static_cast<int64_t>(stripe) * stripeStride + col], 1);
    }
}

class Csr2CscCountColsDispatcher {
public:
    __aicore__ inline void Init(
        GM_ADDR gmColInd, GM_ADDR gmStripeHist,
        int32_t nnz, int32_t n, int32_t idxBase,
        int32_t stripeCount, int32_t stripeSize, int32_t numBlocks)
    {
        csrColInd_ = (__gm__ const int32_t *)gmColInd;
        stripeHist_ = (__gm__ int32_t *)gmStripeHist;
        nnz_ = nnz;
        n_ = n;
        idxBase_ = idxBase;
        stripeCount_ = stripeCount;
        stripeSize_ = stripeSize;
        numBlocks_ = numBlocks;
    }

    __aicore__ inline void Process()
    {
        uint32_t simtThreadNum = ComputeSimtThreadNum(nnz_);

        asc_vf_call<Csr2CscCountColsSimtCompute>(dim3{simtThreadNum},
            csrColInd_, stripeHist_, nnz_, n_, idxBase_,
            stripeCount_, stripeSize_, numBlocks_);
    }

private:
    __gm__ const int32_t *csrColInd_{nullptr};
    __gm__ int32_t *stripeHist_{nullptr};
    int32_t nnz_{0};
    int32_t n_{0};
    int32_t idxBase_{0};
    int32_t stripeCount_{0};
    int32_t stripeSize_{0};
    int32_t numBlocks_{0};
};

extern "C" __global__ __aicore__ void csr2csc_count_kernel(
    GM_ADDR csrColInd, GM_ADDR stripeHist,
    int32_t nnz, int32_t n, int32_t idxBase,
    int32_t stripeCount, int32_t stripeSize, int32_t numBlocks)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    Csr2CscCountColsDispatcher launcher;
    launcher.Init(csrColInd, stripeHist, nnz, n, idxBase,
                  stripeCount, stripeSize, numBlocks);
    launcher.Process();
}

extern "C" void csr2csc_count_kernel_do(
    GM_ADDR csrColInd,
    GM_ADDR stripeHist,
    const Csr2CscCountTilingData &tiling,
    uint32_t numBlocks,
    void *stream)
{
    csr2csc_count_kernel<<<numBlocks, 0, stream>>>(
        csrColInd, stripeHist,
        tiling.nnz, tiling.n, tiling.idxBase,
        tiling.stripeCount, tiling.stripeSize,
        static_cast<int32_t>(numBlocks));
}

// ---------------------------------------------------------------------------
// Kernel 1.5: SumStripeHist — 对 stripeHist 按列求和重建 colCount
//   colCount[j] = sum_{t=0}^{stripeCount-1} stripeHist[t*(n+1)+j]
//   多线程并行（grid-stride loop 按列分配），每线程处理若干列。
//   正确性：sum_t stripeHist[t][j] 等于原 CountCols 中 colCount[j] 的累加结果
//   （两者对同一元素集计数，atomic_add 满足交换律）。
// ---------------------------------------------------------------------------

__simt_vf__ __aicore__ __launch_bounds__(kCsr2CscThreadsPerBlock) inline void
Csr2CscSumStripeHistSimtCompute(
    __gm__ const int32_t *stripeHist,
    __gm__ int32_t *colCount,
    int32_t n, int32_t stripeCount, int32_t numBlocks)
{
    int32_t globalTid = static_cast<int32_t>(threadIdx.x) +
        static_cast<int32_t>(blockIdx.x) * static_cast<int32_t>(blockDim.x);
    int32_t gridStride = static_cast<int32_t>(blockDim.x) * numBlocks;
    int32_t stripeStride = n + 1;

    for (int32_t j = globalTid; j < n; j += gridStride) {
        int32_t sum = 0;
        for (int32_t t = 0; t < stripeCount; t++) {
            // 防溢出：t * stripeStride 用 int64_t
            sum += stripeHist[static_cast<int64_t>(t) * stripeStride + j];
        }
        colCount[j] = sum;
    }
}

class Csr2CscSumStripeHistDispatcher {
public:
    __aicore__ inline void Init(
        GM_ADDR gmStripeHist, GM_ADDR gmWorkspace,
        int32_t n, int32_t stripeCount, int32_t numBlocks)
    {
        stripeHist_ = (__gm__ const int32_t *)gmStripeHist;
        colCount_ = (__gm__ int32_t *)gmWorkspace;
        n_ = n;
        stripeCount_ = stripeCount;
        numBlocks_ = numBlocks;
    }

    __aicore__ inline void Process()
    {
        uint32_t simtThreadNum = ComputeSimtThreadNum(n_);

        asc_vf_call<Csr2CscSumStripeHistSimtCompute>(dim3{simtThreadNum},
            stripeHist_, colCount_, n_, stripeCount_, numBlocks_);
    }

private:
    __gm__ const int32_t *stripeHist_{nullptr};
    __gm__ int32_t *colCount_{nullptr};
    int32_t n_{0};
    int32_t stripeCount_{0};
    int32_t numBlocks_{0};
};

extern "C" __global__ __aicore__ void csr2csc_sum_stripe_hist_kernel(
    GM_ADDR stripeHist, GM_ADDR workspace,
    int32_t n, int32_t stripeCount, int32_t numBlocks)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    Csr2CscSumStripeHistDispatcher launcher;
    launcher.Init(stripeHist, workspace, n, stripeCount, numBlocks);
    launcher.Process();
}

extern "C" void csr2csc_sum_stripe_hist_kernel_do(
    GM_ADDR stripeHist,
    GM_ADDR workspace,
    const Csr2CscSumStripeHistTilingData &tiling,
    uint32_t numBlocks,
    void *stream)
{
    csr2csc_sum_stripe_hist_kernel<<<numBlocks, 0, stream>>>(
        stripeHist, workspace,
        tiling.n, tiling.stripeCount,
        static_cast<int32_t>(numBlocks));
}

// ---------------------------------------------------------------------------
// Kernel 2: PrefixSum — exclusive prefix sum -> cscColPtr
//
//   K2 性能优化：单线程 O(n) -> 单 warp(32线程) 并行 Blelloch scan。
//   3-phase 算法（全 warp shuffle 同步，无需 asc_syncthreads）：
//     Phase 1: 每 thread 连续分块串行求和 colCount[chunkStart..chunkEnd)，得 localSum
//     Phase 2: warp 内 asc_shfl_up 做 Hillis-Steele exclusive scan，得 thread 间前缀 prefix
//     Phase 3: 每 thread 重新遍历 chunk，写 cscColPtr[j+1] = prefix + running + idxBase
//
//   正确性：cscColPtr[j+1] = idxBase + sum_{k=0}^{j} colCount[k]（inclusive prefix + idxBase）。
//   Phase 3 的 running 从 prefix（= 之前所有 thread 的 localSum 之和 = colCount[0..chunkStart-1] 之和）
//   开始累加 colCount[chunkStart..j]，故 cscColPtr[j+1] = idxBase + sum_{k=0}^{j} colCount[k]。
//   边界：cscColPtr[0] = idxBase（lane 0 写）；cscColPtr[n] = idxBase + nnz（自动正确）。
//
//   [defense-in-depth] running 最终等于 nnz，而 idxBase ∈ {0,1}。
//   host 侧 aclsparseCsr2cscEx2 已校验：if (nnz > INT32_MAX - baseVal) return INVALID_VALUE;
//   因此 running + idxBase 不会溢出 int32_t。此处无额外运行时断言，以 host 保证为准。
// ---------------------------------------------------------------------------

__simt_vf__ __aicore__ __launch_bounds__(kCsr2CscWarpSize) inline void
Csr2CscPrefixSumSimtCompute(
    __gm__ const int32_t *colCount,
    __gm__ int32_t *cscColPtr,
    int32_t n, int32_t idxBase)
{
    constexpr int32_t kWarpScanThreads = static_cast<int32_t>(kCsr2CscWarpSize);
    int32_t laneId = static_cast<int32_t>(threadIdx.x);

    // 连续分块：thread j 处理 [j*chunkSize, min((j+1)*chunkSize, n))。
    // chunkSize = ceil(n / 32)，n >= 1（n==0 由 host 快路径处理，不进入此 kernel）。
    int32_t chunkSize = (n + kWarpScanThreads - 1) / kWarpScanThreads;
    int32_t chunkStart = laneId * chunkSize;
    int32_t chunkEnd = chunkStart + chunkSize;
    if (chunkEnd > n) {
        chunkEnd = n;
    }

    // Phase 1: 串行求和本 thread 的 chunk，得 localSum
    int32_t localSum = 0;
    for (int32_t j = chunkStart; j < chunkEnd; j++) {
        localSum += colCount[j];
    }

    // Phase 2: warp 内 Hillis-Steele inclusive scan（5 步 shfl_up），再转 exclusive
    int32_t v = localSum;
    for (int32_t offset = 1; offset < kWarpScanThreads; offset *= 2) {
        int32_t up = asc_shfl_up(v, offset);
        if (laneId >= offset) {
            v = up + v;
        }
    }
    // v = inclusive scan；exclusive = v - localSum（本 thread 之前所有 thread 的 localSum 之和）
    int32_t prefix = v - localSum;

    // Phase 3: 写最终 cscColPtr[j+1] = prefix + running + idxBase
    // running 从 prefix 开始累加 colCount[chunkStart..j]
    int32_t running = prefix + idxBase;
    for (int32_t j = chunkStart; j < chunkEnd; j++) {
        running += colCount[j];
        cscColPtr[j + 1] = running;
    }

    // cscColPtr[0] = idxBase
    if (laneId == 0) {
        cscColPtr[0] = idxBase;
    }
}

class Csr2CscPrefixSumDispatcher {
public:
    __aicore__ inline void Init(
        GM_ADDR gmWorkspace, GM_ADDR gmCscColPtr,
        int32_t n, int32_t idxBase)
    {
        colCount_ = (__gm__ const int32_t *)gmWorkspace;
        cscColPtr_ = (__gm__ int32_t *)gmCscColPtr;
        n_ = n;
        idxBase_ = idxBase;
    }

    __aicore__ inline void Process()
    {
        // 单 warp（32 线程）并行 scan：warp 内 shuffle 同步，无需跨 warp 通信
        asc_vf_call<Csr2CscPrefixSumSimtCompute>(dim3{kCsr2CscWarpSize},
            colCount_, cscColPtr_, n_, idxBase_);
    }

private:
    __gm__ const int32_t *colCount_{nullptr};
    __gm__ int32_t *cscColPtr_{nullptr};
    int32_t n_{0};
    int32_t idxBase_{0};
};

extern "C" __global__ __aicore__ void csr2csc_prefixsum_kernel(
    GM_ADDR workspace, GM_ADDR cscColPtr,
    int32_t n, int32_t idxBase)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    Csr2CscPrefixSumDispatcher launcher;
    launcher.Init(workspace, cscColPtr, n, idxBase);
    launcher.Process();
}

extern "C" void csr2csc_prefixsum_kernel_do(
    GM_ADDR workspace,
    GM_ADDR cscColPtr,
    const Csr2CscPrefixSumTilingData &tiling,
    void *stream)
{
    csr2csc_prefixsum_kernel<<<1, 0, stream>>>(
        workspace, cscColPtr, tiling.n, tiling.idxBase);
}

// ---------------------------------------------------------------------------
// Kernel 3: StripeBase — stripe 直方图按列前缀和 -> 每 stripe 写游标基址
// 对每列 j：base[t][j] = colStart[j] + sum_{t'<t} hist[t'][j]，
// 原地将 stripeHist 区转换为 scatter 写游标（单线程独占一列，无竞争）。
// ---------------------------------------------------------------------------

__simt_vf__ __aicore__ __launch_bounds__(kCsr2CscThreadsPerBlock) inline void
Csr2CscStripeBaseSimtCompute(
    __gm__ int32_t *stripeBase,
    __gm__ const int32_t *cscColPtr,
    int32_t n, int32_t idxBase,
    int32_t stripeCount, int32_t numBlocks)
{
    int32_t globalTid = static_cast<int32_t>(threadIdx.x) +
        static_cast<int32_t>(blockIdx.x) * static_cast<int32_t>(blockDim.x);
    int32_t gridStride = static_cast<int32_t>(blockDim.x) * numBlocks;
    int32_t stripeStride = n + 1;

    for (int32_t col = globalTid; col < n; col += gridStride) {
        // colStart = cscColPtr[col] 的 0-based 偏移（与 golden 的 colOffset 约定一致）
        int32_t running = cscColPtr[col] - idxBase;
        for (int32_t t = 0; t < stripeCount; t++) {
            // 防溢出：t * stripeStride 用 int64_t
            int64_t offset = static_cast<int64_t>(t) * stripeStride + col;
            int32_t cnt = stripeBase[offset];
            stripeBase[offset] = running;
            running += cnt;
        }
    }
}

class Csr2CscStripeBaseDispatcher {
public:
    __aicore__ inline void Init(
        GM_ADDR gmStripeBase, GM_ADDR gmCscColPtr,
        int32_t n, int32_t idxBase, int32_t stripeCount, int32_t numBlocks)
    {
        stripeBase_ = (__gm__ int32_t *)gmStripeBase;
        cscColPtr_ = (__gm__ const int32_t *)gmCscColPtr;
        n_ = n;
        idxBase_ = idxBase;
        stripeCount_ = stripeCount;
        numBlocks_ = numBlocks;
    }

    __aicore__ inline void Process()
    {
        uint32_t simtThreadNum = ComputeSimtThreadNum(n_);

        asc_vf_call<Csr2CscStripeBaseSimtCompute>(dim3{simtThreadNum},
            stripeBase_, cscColPtr_, n_, idxBase_, stripeCount_, numBlocks_);
    }

private:
    __gm__ int32_t *stripeBase_{nullptr};
    __gm__ const int32_t *cscColPtr_{nullptr};
    int32_t n_{0};
    int32_t idxBase_{0};
    int32_t stripeCount_{0};
    int32_t numBlocks_{0};
};

extern "C" __global__ __aicore__ void csr2csc_stripebase_kernel(
    GM_ADDR stripeBase, GM_ADDR cscColPtr,
    int32_t n, int32_t idxBase, int32_t stripeCount, int32_t numBlocks)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    Csr2CscStripeBaseDispatcher launcher;
    launcher.Init(stripeBase, cscColPtr, n, idxBase, stripeCount, numBlocks);
    launcher.Process();
}

extern "C" void csr2csc_stripebase_kernel_do(
    GM_ADDR stripeBase,
    GM_ADDR cscColPtr,
    const Csr2CscStripeBaseTilingData &tiling,
    uint32_t numBlocks,
    void *stream)
{
    csr2csc_stripebase_kernel<<<numBlocks, 0, stream>>>(
        stripeBase, cscColPtr,
        tiling.n, tiling.idxBase, tiling.stripeCount,
        static_cast<int32_t>(numBlocks));
}

// ---------------------------------------------------------------------------
// Kernel 4: Scatter — 每 block 负责一个 stripe（连续 k 区间），单线程顺序 scatter
//
// 正确性设计（与 golden 逐位一致）：
//   golden 按 CSR 行优先（即 k 升序）依次将元素放入目标列的 CSC 段内。
//   本 kernel 将 [0, nnz) 划分为 stripeCount 个连续 k 区间（stripe），
//   列 j 的 CSC 段被划分为各 stripe 的子段 [base[t][j], base[t][j]+hist[t][j])，
//   子段之间按 stripe 序排列且互不重叠（Kernel 3 前缀和保证）；
//   每个 stripe 由单个线程按 k 升序顺序写入（游标私有，无原子竞争），
//   因此列 j 内全局写入顺序 = k 升序 = golden 行优先顺序，结果逐位一致。
//
// 依赖：stripeSize 由 host 保证 ≥ 1（host.cpp LaunchCsr2cscEx2Kernel 中
//   stripeSize = CeilDiv(nnz, numBlocks)，numBlocks ≥ 1 故 stripeSize ≥ 1；
//   且 host 对 stripeSize == 0 做兜底设为 1），kernel 内 stripeSize 用作乘法因子。
// ---------------------------------------------------------------------------

/// 二分查找 kFirst 所属行：最小 r 满足 csrRowPtr[r+1] - idxBase > kFirst。
/// 从 Csr2CscScatterSimtCompute 拆分以控制 NBNC。
/// kFirst < nnz <= INT32_MAX（nnz 为 int32 且 host 已校验非负），截断安全。
__simt_callee__ __aicore__ inline int32_t
FindScatterStartRow(
    __gm__ const int32_t *csrRowPtr, int32_t m, int32_t idxBase, int32_t kFirst)
{
    int32_t lo = 0;
    int32_t hi = m - 1;
    while (lo < hi) {
        int32_t mid = lo + (hi - lo) / 2;
        if (csrRowPtr[mid + 1] - idxBase > kFirst) {
            hi = mid;
        } else {
            lo = mid + 1;
        }
    }
    return lo;
}

template <typename ValT>
__simt_vf__ __aicore__ __launch_bounds__(1) inline void
Csr2CscScatterSimtCompute(
    __gm__ const int32_t *csrRowPtr,
    __gm__ const int32_t *csrColInd,
    __gm__ const ValT *csrVal,
    __gm__ int32_t *cscRowInd,
    __gm__ ValT *cscVal,
    __gm__ int32_t *stripeCursor,
    int32_t m, int32_t n, int32_t nnz, int32_t idxBase,
    int32_t copyValues, int32_t stripeSize)
{
    int32_t stripe = static_cast<int32_t>(blockIdx.x);
    int64_t kStart = static_cast<int64_t>(stripe) * stripeSize;
    if (kStart >= nnz) {
        return;
    }
    int64_t kEnd = kStart + stripeSize;
    if (kEnd > nnz) {
        kEnd = nnz;
    }

    int32_t kFirst = static_cast<int32_t>(kStart);
    int32_t row = FindScatterStartRow(csrRowPtr, m, idxBase, kFirst);

    // 防溢出：stripe * (n+1) 用 int64_t
    __gm__ int32_t *cursor = stripeCursor + static_cast<int64_t>(stripe) * (n + 1);
    // 缓存当前行结束位置，row 前进时才重新从 GM 加载，避免内层循环每轮重读 csrRowPtr[row+1]
    int32_t rowEnd = csrRowPtr[row + 1] - idxBase;

    // 单线程按 k 升序顺序写入（游标私有，无原子竞争），保证列内行号升序
    for (int32_t k = kFirst; k < static_cast<int32_t>(kEnd); k++) {
        // 行单调前进，跳过空行（k 必属于某非空行）
        while (row + 1 < m && k >= rowEnd) {
            row++;
            rowEnd = csrRowPtr[row + 1] - idxBase;
        }
        int32_t col = csrColInd[k] - idxBase;
        int32_t pos = cursor[col];
        cursor[col] = pos + 1;

        cscRowInd[pos] = row + idxBase;
        if (copyValues == kCsr2CscNumeric) {
            cscVal[pos] = csrVal[k];
        }
    }
}

// 模板特化：按 valType 分发
template <typename ValT>
class Csr2CscScatterDispatcher {
public:
    __aicore__ inline void Init(
        GM_ADDR gmRowPtr, GM_ADDR gmColInd, GM_ADDR gmCsrVal,
        GM_ADDR gmCscRowInd, GM_ADDR gmCscVal, GM_ADDR gmStripeCursor,
        int32_t m, int32_t n, int32_t nnz, int32_t idxBase,
        int32_t copyValues, int32_t stripeSize)
    {
        csrRowPtr_ = (__gm__ const int32_t *)gmRowPtr;
        csrColInd_ = (__gm__ const int32_t *)gmColInd;
        csrVal_ = (__gm__ const ValT *)gmCsrVal;
        cscRowInd_ = (__gm__ int32_t *)gmCscRowInd;
        cscVal_ = (__gm__ ValT *)gmCscVal;
        stripeCursor_ = (__gm__ int32_t *)gmStripeCursor;
        m_ = m;
        n_ = n;
        nnz_ = nnz;
        idxBase_ = idxBase;
        copyValues_ = copyValues;
        stripeSize_ = stripeSize;
    }

    __aicore__ inline void Process()
    {
        // 每 block 一个线程处理一个 stripe（stripe 间游标私有、无竞争）
        asc_vf_call<Csr2CscScatterSimtCompute<ValT>>(dim3{1},
            csrRowPtr_, csrColInd_, csrVal_,
            cscRowInd_, cscVal_, stripeCursor_,
            m_, n_, nnz_, idxBase_, copyValues_, stripeSize_);
    }

private:
    __gm__ const int32_t *csrRowPtr_{nullptr};
    __gm__ const int32_t *csrColInd_{nullptr};
    __gm__ const ValT *csrVal_{nullptr};
    __gm__ int32_t *cscRowInd_{nullptr};
    __gm__ ValT *cscVal_{nullptr};
    __gm__ int32_t *stripeCursor_{nullptr};
    int32_t m_{0};
    int32_t n_{0};
    int32_t nnz_{0};
    int32_t idxBase_{0};
    int32_t copyValues_{0};
    int32_t stripeSize_{0};
};

extern "C" __global__ __aicore__ void csr2csc_scatter_kernel(
    GM_ADDR csrRowPtr, GM_ADDR csrColInd, GM_ADDR csrVal,
    GM_ADDR cscRowInd, GM_ADDR cscVal, GM_ADDR stripeCursor,
    int32_t m, int32_t n, int32_t nnz, int32_t idxBase,
    int32_t copyValues, int32_t stripeSize, uint32_t valSize)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

    if (valSize == sizeof(int8_t)) {
        Csr2CscScatterDispatcher<int8_t> launcher;
        launcher.Init(csrRowPtr, csrColInd, csrVal,
                      cscRowInd, cscVal, stripeCursor,
                      m, n, nnz, idxBase, copyValues, stripeSize);
        launcher.Process();
    } else if (valSize == sizeof(uint16_t)) {
        Csr2CscScatterDispatcher<uint16_t> launcher;
        launcher.Init(csrRowPtr, csrColInd, csrVal,
                      cscRowInd, cscVal, stripeCursor,
                      m, n, nnz, idxBase, copyValues, stripeSize);
        launcher.Process();
    } else if (valSize == sizeof(float)) {
        Csr2CscScatterDispatcher<float> launcher;
        launcher.Init(csrRowPtr, csrColInd, csrVal,
                      cscRowInd, cscVal, stripeCursor,
                      m, n, nnz, idxBase, copyValues, stripeSize);
        launcher.Process();
    } else {
        // host 侧 GetValSize 已校验 valType 仅支持 1/2/4 字节，此分支不可达
    }
}

extern "C" void csr2csc_scatter_kernel_do(
    GM_ADDR csrRowPtr,
    GM_ADDR csrColInd,
    GM_ADDR csrVal,
    GM_ADDR cscRowInd,
    GM_ADDR cscVal,
    GM_ADDR stripeCursor,
    const Csr2CscScatterTilingData &tiling,
    uint32_t numBlocks,
    void *stream)
{
    // numBlocks = stripeCount，每 block 单线程处理一个 stripe
    csr2csc_scatter_kernel<<<numBlocks, 0, stream>>>(
        csrRowPtr, csrColInd, csrVal,
        cscRowInd, cscVal, stripeCursor,
        tiling.m, tiling.n, tiling.nnz, tiling.idxBase,
        tiling.copyValues, tiling.stripeSize, tiling.valSize);
}
