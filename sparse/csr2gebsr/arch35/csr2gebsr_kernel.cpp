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
 * \file csr2gebsr_kernel.cpp
 * \brief csr2gebsr kernel 实现（SIMT，仅 arch35 可用）。
 *
 * 三层结构（class-based Dispatcher + asc_vf_call 混合编程模式）：
 *   Layer 1: __simt_vf__ 计算函数（线程级并行，grid-stride loop）
 *   Layer 2: __global__ 调度器（读 tiling → asc_vf_call 分发）
 *   Layer 3: _kernel_do 启动器（<<<>>> 语法）
 *
 * 三个 kernel：
 *   Kernel 1 (Nnz CountBlocksPerRow): 逐块行统计非零块数
 *   Kernel 2 (PrefixSum):             device 侧 exclusive prefix sum
 *                - 退化路径（useBlocks==1）: 单 kernel 串行 O(mb)
 *                - 并行路径（useBlocks>1）: 融合单 kernel，Phase1/2/3 + SyncAll
 *                （原三阶段 p1/p2/p3 kernel 保留，host 侧不再调用）
 *   Kernel 3 (Convert):               填充 bsrColIndC + bsrValC
 */

#include <cstdint>
#include "kernel_operator.h"
#include "simt_api/asc_simt.h"
#include "csr2gebsr_kernel.h"

// ===========================================================================
// Common dispatcher base
// ===========================================================================

/// 公共基类：封装 block-id 到 blockRow 范围的映射和 SIMT 线程数计算
struct Csr2gebsrDispatcherBase {
    int32_t mb_{0};
    uint32_t blockRowsPerCore_{0};

    /// 计算本 core 负责的 [blockRowStart, blockRowEnd) 范围和 SIMT 线程数
    __aicore__ inline void ComputeBlockRowRangeAndThreads(
        int32_t &blockRowStart, int32_t &blockRowEnd, uint32_t &simtThreadNum)
    {
        int32_t outerId = static_cast<int32_t>(AscendC::GetBlockIdx());
        blockRowStart = outerId * static_cast<int32_t>(blockRowsPerCore_);
        blockRowEnd = blockRowStart + static_cast<int32_t>(blockRowsPerCore_);
        if (blockRowEnd > mb_) {
            blockRowEnd = mb_;
        }
        int32_t coreRangeLen = blockRowEnd - blockRowStart;

        simtThreadNum = kCsr2gebsrMaxThreadsPerBlock;
        if (static_cast<uint32_t>(coreRangeLen) < simtThreadNum) {
            simtThreadNum = static_cast<uint32_t>(coreRangeLen);
        }
        if (simtThreadNum == 0) {
            simtThreadNum = 1;
        }
        simtThreadNum = (simtThreadNum + kCsr2gebsrWarpSize - 1u) & ~(kCsr2gebsrWarpSize - 1u);
    }
};

// ===========================================================================
// Kernel 1: CountBlocksPerRow (Nnz 计数)
// ===========================================================================

/// 前向声明：标记非零块函数（Nnz 和 Convert 共用，定义在 BinarySearchCsrRow 之前）
__simt_callee__ __aicore__ inline int32_t MarkNonzeroBlocks(
    __gm__ const int32_t *csrRowPtrA,
    __gm__ const int32_t *csrColIndA,
    __gm__ int32_t *marker,
    int32_t rowStart, int32_t rowEnd, int32_t n,
    int32_t colBlockDim, float invColBlockDim,
    int32_t baseA, int32_t actualBr, int64_t markerBase);

/// 计算块行索引和行范围（Nnz 和 Convert 共用）
struct BlockRowInfo {
    int32_t actualBr;
    int32_t rowStart;
    int32_t rowEnd;
    int64_t markerBase;
};

__simt_callee__ __aicore__ inline BlockRowInfo ComputeBlockRowInfo(
    int32_t br, int32_t blockRowStart,
    int32_t rowBlockDim, int32_t m, int32_t nb)
{
    BlockRowInfo info{};
    info.actualBr = blockRowStart + br;
    info.rowStart = info.actualBr * rowBlockDim;
    info.rowEnd = info.rowStart + rowBlockDim;
    if (info.rowEnd > m) { info.rowEnd = m; }
    info.markerBase = static_cast<int64_t>(info.actualBr) * nb;
    return info;
}

/// SIMT VF 计算函数 - 逐块行统计非零块数
///
/// 对每个块行 br，遍历其包含的 CSR 行，用 marker 数组去重统计非零列块数。
/// marker[colBlock] != br 表示该列块未被当前块行标记过。
///
/// 注意：dav-3510 上 __simt_vf__ 内的 blockDim.x 返回外层 grid 的 blockDim
/// （即 numBlocks），禁止使用。改用显式传入的 threadsPerCore。
__simt_vf__ __aicore__ __launch_bounds__(kCsr2gebsrMaxThreadsPerBlock) inline void
Csr2gebsrNnzSimtCompute(
    __gm__ const int32_t *csrRowPtrA,
    __gm__ const int32_t *csrColIndA,
    __gm__ int32_t *nnzBlocksPerRow,
    __gm__ int32_t *marker,
    int32_t m, int32_t n, int32_t nb, int32_t rowBlockDim, int32_t colBlockDim,
    int32_t baseA, float invColBlockDim,
    int32_t blockRowStart, int32_t blockRowEnd,
    int32_t threadsPerCore)
{
    if (colBlockDim < 1) { return; }
    for (int32_t br = static_cast<int32_t>(threadIdx.x);
         br < (blockRowEnd - blockRowStart);
         br += threadsPerCore) {
        auto info = ComputeBlockRowInfo(br, blockRowStart, rowBlockDim, m, nb);
        int32_t count = MarkNonzeroBlocks(
            csrRowPtrA, csrColIndA, marker,
            info.rowStart, info.rowEnd, n, colBlockDim, invColBlockDim,
            baseA, info.actualBr, info.markerBase);
        nnzBlocksPerRow[info.actualBr] = count;
    }
}

/// Dispatcher 类：管理 GM 指针 + 调用 asc_vf_call
class Csr2gebsrNnzDispatcher : public Csr2gebsrDispatcherBase {
public:
    __aicore__ inline void Init(
        GM_ADDR gmCsrRowPtrA, GM_ADDR gmCsrColIndA,
        GM_ADDR gmNnzBlocksPerRow, GM_ADDR gmMarker,
        const Csr2gebsrNnzTilingData *tiling)
    {
        csrRowPtrA_ = (__gm__ const int32_t *)gmCsrRowPtrA;
        csrColIndA_ = (__gm__ const int32_t *)gmCsrColIndA;
        nnzBlocksPerRow_ = (__gm__ int32_t *)gmNnzBlocksPerRow;
        marker_ = (__gm__ int32_t *)gmMarker;
        mb_ = tiling->mb;
        m_ = tiling->m;
        n_ = tiling->n;
        nb_ = tiling->nb;
        rowBlockDim_ = tiling->rowBlockDim;
        colBlockDim_ = tiling->colBlockDim;
        baseA_ = tiling->baseA;
        invColBlockDim_ = tiling->invColBlockDim;
        blockRowsPerCore_ = tiling->blockRowsPerCore;
    }

    __aicore__ inline void Process()
    {
        int32_t blockRowStart{};
        int32_t blockRowEnd{};
        uint32_t simtThreadNum{};
        ComputeBlockRowRangeAndThreads(blockRowStart, blockRowEnd, simtThreadNum);

        asc_vf_call<Csr2gebsrNnzSimtCompute>(
            dim3{simtThreadNum},
            csrRowPtrA_, csrColIndA_, nnzBlocksPerRow_, marker_,
            m_, n_, nb_, rowBlockDim_, colBlockDim_, baseA_, invColBlockDim_,
            blockRowStart, blockRowEnd,
            static_cast<int32_t>(simtThreadNum));
    }

private:
    __gm__ const int32_t *csrRowPtrA_{nullptr};
    __gm__ const int32_t *csrColIndA_{nullptr};
    __gm__ int32_t *nnzBlocksPerRow_{nullptr};
    __gm__ int32_t *marker_{nullptr};
    int32_t m_{0};
    int32_t n_{0};
    int32_t nb_{0};
    int32_t rowBlockDim_{0};
    int32_t colBlockDim_{0};
    int32_t baseA_{0};
    float invColBlockDim_{1.0f};
};

/// __global__ 调度器（Kernel 1）
extern "C" __global__ __aicore__ void csr2gebsr_nnz_kernel(
    GM_ADDR gmCsrRowPtrA, GM_ADDR gmCsrColIndA,
    GM_ADDR gmNnzBlocksPerRow, GM_ADDR gmMarker,
    const Csr2gebsrNnzTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    Csr2gebsrNnzDispatcher dispatcher;
    dispatcher.Init(gmCsrRowPtrA, gmCsrColIndA,
                    gmNnzBlocksPerRow, gmMarker, &tiling);
    dispatcher.Process();
}

/// kernel_do 启动器（Kernel 1: Nnz 计数）
extern "C" void csr2gebsr_nnz_kernel_do(
    GM_ADDR csrRowPtrA, GM_ADDR csrColIndA,
    GM_ADDR nnzBlocksPerRow, GM_ADDR marker,
    const Csr2gebsrNnzTilingData &tiling,
    uint32_t numBlocks,
    void *stream)
{
    csr2gebsr_nnz_kernel<<<numBlocks, nullptr, stream>>>(
        csrRowPtrA, csrColIndA, nnzBlocksPerRow, marker, tiling);
}

// ===========================================================================
// Kernel 2: PrefixSum (device 侧 exclusive prefix sum，单 block)
// ===========================================================================

/// SIMT VF 计算函数 - exclusive prefix sum
///
/// 读 nnzBlocksPerRow[0..mb-1]，写 bsrRowPtrC[0..mb]，写 nnzbDev[0]。
/// 单线程执行（dim3{1}），O(mb) 顺序操作。
__simt_vf__ __aicore__ inline void
Csr2gebsrPrefixSumSimtCompute(
    __gm__ const int32_t *nnzBlocksPerRow,
    __gm__ int32_t *bsrRowPtrC,
    __gm__ int32_t *nnzbDev,
    int32_t mb,
    int32_t baseC)
{
    int32_t runningSum = 0;
    bsrRowPtrC[0] = baseC;
    for (int32_t i = 0; i < mb; i++) {
        runningSum += nnzBlocksPerRow[i];
        bsrRowPtrC[i + 1] = runningSum + baseC;
    }
    nnzbDev[0] = runningSum;
}

/// Dispatcher 类：管理 GM 指针 + 调用 asc_vf_call（单线程）
class Csr2gebsrPrefixSumDispatcher {
public:
    __aicore__ inline void Init(
        GM_ADDR gmNnzBlocksPerRow, GM_ADDR gmBsrRowPtrC, GM_ADDR gmNnzbDev,
        const Csr2gebsrPrefixSumTilingData *tiling)
    {
        nnzBlocksPerRow_ = (__gm__ const int32_t *)gmNnzBlocksPerRow;
        bsrRowPtrC_ = (__gm__ int32_t *)gmBsrRowPtrC;
        nnzbDev_ = (__gm__ int32_t *)gmNnzbDev;
        mb_ = tiling->mb;
        baseC_ = tiling->baseC;
    }

    __aicore__ inline void Process()
    {
        asc_vf_call<Csr2gebsrPrefixSumSimtCompute>(
            dim3{1u},
            nnzBlocksPerRow_, bsrRowPtrC_, nnzbDev_,
            mb_, baseC_);
    }

private:
    __gm__ const int32_t *nnzBlocksPerRow_{nullptr};
    __gm__ int32_t *bsrRowPtrC_{nullptr};
    __gm__ int32_t *nnzbDev_{nullptr};
    int32_t mb_{0};
    int32_t baseC_{0};
};

/// __global__ 调度器（Kernel 2: Prefix Sum，单 block）
extern "C" __global__ __aicore__ void csr2gebsr_prefixsum_kernel(
    GM_ADDR gmNnzBlocksPerRow, GM_ADDR gmBsrRowPtrC, GM_ADDR gmNnzbDev,
    const Csr2gebsrPrefixSumTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    Csr2gebsrPrefixSumDispatcher dispatcher;
    dispatcher.Init(gmNnzBlocksPerRow, gmBsrRowPtrC, gmNnzbDev, &tiling);
    dispatcher.Process();
}

/// kernel_do 启动器（Kernel 2: Prefix Sum，单 block）
/// 退化路径：useBlocks == 1 时由 host 侧调用此函数（原串行 O(mb) 实现）
extern "C" void csr2gebsr_prefixsum_kernel_do(
    GM_ADDR nnzBlocksPerRow, GM_ADDR bsrRowPtrC, GM_ADDR nnzbDev,
    const Csr2gebsrPrefixSumTilingData &tiling,
    void *stream)
{
    csr2gebsr_prefixsum_kernel<<<1, nullptr, stream>>>(
        nnzBlocksPerRow, bsrRowPtrC, nnzbDev, tiling);
}

// ===========================================================================
// Kernel 2 融合路径使用的 Phase1/2/3 SIMT VF 计算函数
//
// Phase 1: 段内 inclusive prefix sum + 写 segSum（每核单线程）
// Phase 2: 段间 exclusive scan + 写 nnzbDev（单线程）
// Phase 3: 加 segOffset + baseC（每核多线程 grid-stride）
// ===========================================================================

/// Phase 1: 段内 inclusive prefix sum + 写 segSum
__simt_vf__ __aicore__ inline void
Csr2gebsrPrefixSumPhase1SimtCompute(
    __gm__ const int32_t *nnzBlocksPerRow,
    __gm__ int32_t *bsrRowPtrC,
    __gm__ int32_t *segSum,
    int32_t blockRowStart, int32_t blockRowEnd,
    int32_t coreId)
{
    int32_t runningSum = 0;
    for (int32_t i = blockRowStart; i < blockRowEnd; i++) {
        runningSum += nnzBlocksPerRow[i];
        bsrRowPtrC[i + 1] = runningSum;
    }
    segSum[coreId] = runningSum;
}

/// Phase 2: 段间 exclusive scan + 写 nnzbDev
__simt_vf__ __aicore__ inline void
Csr2gebsrPrefixSumPhase2SimtCompute(
    __gm__ int32_t *segSum,
    __gm__ int32_t *nnzbDev,
    uint32_t useBlocks)
{
    int32_t offset = 0;
    for (uint32_t c = 0; c < useBlocks; c++) {
        int32_t segSumOld = segSum[c];
        segSum[c] = offset;
        offset += segSumOld;
    }
    nnzbDev[0] = offset;
}

/// Phase 3: 加 segOffset + baseC（grid-stride 并行）
__simt_vf__ __aicore__ __launch_bounds__(kCsr2gebsrMaxThreadsPerBlock) inline void
Csr2gebsrPrefixSumPhase3SimtCompute(
    __gm__ int32_t *bsrRowPtrC,
    __gm__ const int32_t *segSum,
    int32_t baseC,
    int32_t blockRowStart, int32_t blockRowEnd,
    int32_t coreId,
    int32_t threadsPerCore)
{
    int32_t myOffset = segSum[coreId] + baseC;
    int32_t segLen = blockRowEnd - blockRowStart;

    if (coreId == 0 && static_cast<int32_t>(threadIdx.x) == 0) {
        bsrRowPtrC[0] = baseC;
    }

    for (int32_t j = static_cast<int32_t>(threadIdx.x); j < segLen; j += threadsPerCore) {
        bsrRowPtrC[blockRowStart + 1 + j] += myOffset;
    }
}

// ===========================================================================
// Kernel 2 融合路径（PrefixSum 三阶段融合为单 kernel + SyncAll）
//
// 将 Phase1/2/3 融合为单个 __global__ kernel，用 AscendC::SyncAll<true>() 替代
// 跨 kernel launch 保序，消除 2 次额外 launch 开销。
//   Phase1 VF_CALL → SyncAll（等所有核段内 scan 完成）
//   → Phase2 VF_CALL（仅核 0）→ SyncAll（等核 0 段间 scan 完成）
//   → Phase3 VF_CALL
//
// SyncAll 内部调用 PipeBarrier<PIPE_ALL>() + ffts_cross_core_sync，保证所有核
// 的 GM 写入对后续阶段可见。arch35 (DAV_3510) 支持无参数 SyncAll<true>()。
//
// 退化路径（useBlocks==1）仍走原 csr2gebsr_prefixsum_kernel_do（串行）。
// 原三阶段 p1/p2/p3 kernel 保留不删除，host 侧不再调用（回退安全）。
// ===========================================================================

/// 融合 Dispatcher：单 kernel 内 Phase1→SyncAll→Phase2→SyncAll→Phase3
///
/// 复用已有 Csr2gebsrPrefixSumPhase1/2/3SimtCompute 函数，仅在 VF_CALL 之间
//  插入全核同步。
class Csr2gebsrPrefixSumFusedDispatcher : public Csr2gebsrDispatcherBase {
public:
    __aicore__ inline void Init(
        GM_ADDR gmNnzBlocksPerRow, GM_ADDR gmBsrRowPtrC,
        GM_ADDR gmNnzbDev, GM_ADDR gmSegSum,
        const Csr2gebsrPrefixSumTilingData *tiling)
    {
        nnzBlocksPerRow_ = (__gm__ const int32_t *)gmNnzBlocksPerRow;
        bsrRowPtrC_ = (__gm__ int32_t *)gmBsrRowPtrC;
        nnzbDev_ = (__gm__ int32_t *)gmNnzbDev;
        segSum_ = (__gm__ int32_t *)gmSegSum;
        mb_ = tiling->mb;
        baseC_ = tiling->baseC;
        useBlocks_ = tiling->useBlocks;
        blockRowsPerCore_ = tiling->blockRowsPerCore;
    }

    __aicore__ inline void Process()
    {
        int32_t blockRowStart{};
        int32_t blockRowEnd{};
        uint32_t simtThreadNum{};
        ComputeBlockRowRangeAndThreads(blockRowStart, blockRowEnd, simtThreadNum);
        int32_t coreId = static_cast<int32_t>(AscendC::GetBlockIdx());

        // Phase 1: 段内 inclusive prefix sum + 写 segSum（每核单线程）
        asc_vf_call<Csr2gebsrPrefixSumPhase1SimtCompute>(
            dim3{1u},
            nnzBlocksPerRow_, bsrRowPtrC_, segSum_,
            blockRowStart, blockRowEnd, coreId);

        // 全核同步：等所有核 Phase1 完成（segSum 写入对 Phase2 可见）
        AscendC::SyncAll<true>();

        // Phase 2: 段间 exclusive scan + 写 nnzbDev（仅核 0）
        if (coreId == 0) {
            asc_vf_call<Csr2gebsrPrefixSumPhase2SimtCompute>(
                dim3{1u},
                segSum_, nnzbDev_, useBlocks_);
        }

        // 全核同步：等核 0 Phase2 完成（segSum 已 in-place 写为 segOffset，对 Phase3 可见）
        AscendC::SyncAll<true>();

        // Phase 3: 加 segOffset + baseC（每核多线程 grid-stride）
        asc_vf_call<Csr2gebsrPrefixSumPhase3SimtCompute>(
            dim3{simtThreadNum},
            bsrRowPtrC_, segSum_,
            baseC_, blockRowStart, blockRowEnd, coreId,
            static_cast<int32_t>(simtThreadNum));
    }

private:
    __gm__ const int32_t *nnzBlocksPerRow_{nullptr};
    __gm__ int32_t *bsrRowPtrC_{nullptr};
    __gm__ int32_t *nnzbDev_{nullptr};
    __gm__ int32_t *segSum_{nullptr};
    int32_t baseC_{0};
    uint32_t useBlocks_{0};
};

/// __global__ 调度器（Kernel 2 融合路径：单 kernel 三阶段 + SyncAll）
extern "C" __global__ __aicore__ void csr2gebsr_prefixsum_fused_kernel(
    GM_ADDR gmNnzBlocksPerRow, GM_ADDR gmBsrRowPtrC,
    GM_ADDR gmNnzbDev, GM_ADDR gmSegSum,
    const Csr2gebsrPrefixSumTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    Csr2gebsrPrefixSumFusedDispatcher dispatcher;
    dispatcher.Init(gmNnzBlocksPerRow, gmBsrRowPtrC,
                    gmNnzbDev, gmSegSum, &tiling);
    dispatcher.Process();
}

/// kernel_do 启动器（Kernel 2 融合路径: <<<useBlocks>>> 单次 launch）
extern "C" void csr2gebsr_prefixsum_fused_kernel_do(
    GM_ADDR nnzBlocksPerRow, GM_ADDR bsrRowPtrC,
    GM_ADDR nnzbDev, GM_ADDR segSum,
    const Csr2gebsrPrefixSumTilingData &tiling,
    uint32_t numBlocks,
    void *stream)
{
    csr2gebsr_prefixsum_fused_kernel<<<numBlocks, nullptr, stream>>>(
        nnzBlocksPerRow, bsrRowPtrC, nnzbDev, segSum, tiling);
}

// ===========================================================================
// Kernel 3: Convert (填充 bsrColIndC + bsrValC)
// ===========================================================================

/// 标记本块行中存在的列块（marker trick），返回非零块数
/// Nnz 和 Convert 的 Pass1 共用此函数
__simt_callee__ __aicore__ inline int32_t MarkNonzeroBlocks(
    __gm__ const int32_t *csrRowPtrA,
    __gm__ const int32_t *csrColIndA,
    __gm__ int32_t *marker,
    int32_t rowStart, int32_t rowEnd, int32_t n,
    int32_t colBlockDim, float invColBlockDim,
    int32_t baseA, int32_t actualBr, int64_t markerBase)
{
    int32_t count = 0;
    for (int32_t r = rowStart; r < rowEnd; r++) {
        int32_t pA = csrRowPtrA[r] - baseA;
        if (pA < 0) { pA = 0; }
        int32_t qA = csrRowPtrA[r + 1] - baseA;
        for (int32_t j = pA; j < qA; j++) {
            int32_t col = csrColIndA[j] - baseA;
            if (col < 0 || col >= n) {
                continue;
            }
            int32_t colBlock = static_cast<int32_t>(
                static_cast<float>(col) * invColBlockDim);
            if (marker[markerBase + colBlock] != actualBr) {
                marker[markerBase + colBlock] = actualBr;
                count++;
            }
        }
    }
    return count;
}

/// 二分查找辅助函数：在 CSR 行 r 中查找列索引 targetCol
/// 若找到返回对应值，否则返回 0
template <typename VAL_T>
__simt_callee__ __aicore__ inline VAL_T BinarySearchCsrRow(
    __gm__ const VAL_T *csrValA,
    __gm__ const int32_t *csrColIndA,
    int32_t rowStart, int32_t rowEnd,
    int32_t targetCol, int32_t baseA)
{
    // rowStart<0 防御，防止负索引越界读
    if (rowStart < 0) { rowStart = 0; }
    int32_t lo = rowStart;
    int32_t hi = rowEnd - 1;
    while (lo <= hi) {
        int32_t mid = lo + (hi - lo) / 2;
        int32_t midCol = csrColIndA[mid] - baseA;
        if (midCol == targetCol) {
            return csrValA[mid];
        }
        if (midCol < targetCol) {
            lo = mid + 1;
        } else {
            hi = mid - 1;
        }
    }
    return static_cast<VAL_T>(0);
}

/// 块值填充：遍历块内 (inRow, inCol)，按方向 layout 写入 bsrValC[blockOffset + writeIdx]。
/// blockOffset 用 int64_t 承载，避免 (blockStart+blockIdx)*blockSize 在
/// nnzb*blockSize > INT32_MAX 时溢出为负导致 bsrValC 越界写。
/// writeIdx 保持 int32_t（块内偏移，max = rowBlockDim*colBlockDim，小值）。
template <typename VAL_T>
__simt_callee__ __aicore__ inline void FillBlockValues(
    __gm__ const VAL_T *csrValA,
    __gm__ const int32_t *csrRowPtrA,
    __gm__ const int32_t *csrColIndA,
    __gm__ VAL_T *bsrValC,
    int32_t rowStart, int32_t m, int32_t n,
    int32_t rowBlockDim, int32_t colBlockDim,
    int32_t colBlock, int32_t baseA, int32_t dir,
    int64_t blockOffset)
{
    for (int32_t inRow = 0; inRow < rowBlockDim; inRow++) {
        int32_t r = rowStart + inRow;
        for (int32_t inCol = 0; inCol < colBlockDim; inCol++) {
            int32_t col = colBlock * colBlockDim + inCol;
            VAL_T value = static_cast<VAL_T>(0);
            if (r < m && col < n) {
                int32_t pA = csrRowPtrA[r] - baseA;
                // pA<0 防御，防止负索引越界读
                if (pA < 0) { pA = 0; }
                int32_t qA = csrRowPtrA[r + 1] - baseA;
                value = BinarySearchCsrRow(
                    csrValA, csrColIndA, pA, qA, col, baseA);
            }
            int32_t writeIdx;
            if (dir == 0) {  // ROW
                writeIdx = inRow * colBlockDim + inCol;
            } else {  // COLUMN
                writeIdx = inCol * rowBlockDim + inRow;
            }
            bsrValC[blockOffset + writeIdx] = value;
        }
    }
}

/// SIMT VF 计算函数 - 模板化 Convert（按值类型大小特化）
///
/// 对每个块行 br：
///   Pass 1: 扫描 CSR 非零元，标记存在的列块（marker trick）
///   Pass 2: 按列块顺序输出 bsrColIndC + bsrValC（块值填充委托 FillBlockValues）
template <typename VAL_T>
__simt_vf__ __aicore__ __launch_bounds__(kCsr2gebsrMaxThreadsPerBlock) inline void
Csr2gebsrConvertSimtCompute(
    __gm__ const VAL_T *csrValA,
    __gm__ const int32_t *csrRowPtrA,
    __gm__ const int32_t *csrColIndA,
    __gm__ const int32_t *bsrRowPtrC,
    __gm__ int32_t *bsrColIndC,
    __gm__ VAL_T *bsrValC,
    __gm__ int32_t *marker,
    int32_t m, int32_t n, int32_t nb,
    int32_t rowBlockDim, int32_t colBlockDim,
    int32_t baseA, int32_t baseC, int32_t dir, float invColBlockDim,
    int32_t blockRowStart, int32_t blockRowEnd,
    int32_t threadsPerCore)
{
    if (colBlockDim < 1) { return; }
    for (int32_t br = static_cast<int32_t>(threadIdx.x);
         br < (blockRowEnd - blockRowStart);
         br += threadsPerCore) {
        auto info = ComputeBlockRowInfo(br, blockRowStart, rowBlockDim, m, nb);
        int32_t blockStart = bsrRowPtrC[info.actualBr] - baseC;

        MarkNonzeroBlocks(
            csrRowPtrA, csrColIndA, marker,
            info.rowStart, info.rowEnd, n, colBlockDim, invColBlockDim,
            baseA, info.actualBr, info.markerBase);

        int32_t blockIdx = 0;
        int64_t blockSize = static_cast<int64_t>(rowBlockDim) * colBlockDim;
        for (int32_t colBlock = 0; colBlock < nb; colBlock++) {
            if (marker[info.markerBase + colBlock] != info.actualBr) {
                continue;
            }

            bsrColIndC[blockStart + blockIdx] = colBlock + baseC;
            int64_t blockOffset = static_cast<int64_t>(blockStart + blockIdx) * blockSize;
            FillBlockValues<VAL_T>(
                csrValA, csrRowPtrA, csrColIndA, bsrValC,
                info.rowStart, m, n, rowBlockDim, colBlockDim,
                colBlock, baseA, dir, blockOffset);
            blockIdx++;
        }
    }
}

/// Dispatcher 类：管理 GM 指针 + 按 valSize 分发模板特化
class Csr2gebsrConvertDispatcher : public Csr2gebsrDispatcherBase {
public:
    __aicore__ inline void Init(
        GM_ADDR gmCsrValA, GM_ADDR gmCsrRowPtrA, GM_ADDR gmCsrColIndA,
        GM_ADDR gmBsrRowPtrC, GM_ADDR gmBsrColIndC, GM_ADDR gmBsrValC,
        GM_ADDR gmMarker,
        const Csr2gebsrConvertTilingData *tiling)
    {
        csrValA_ = gmCsrValA;
        csrRowPtrA_ = (__gm__ const int32_t *)gmCsrRowPtrA;
        csrColIndA_ = (__gm__ const int32_t *)gmCsrColIndA;
        bsrRowPtrC_ = (__gm__ const int32_t *)gmBsrRowPtrC;
        bsrColIndC_ = (__gm__ int32_t *)gmBsrColIndC;
        bsrValC_ = gmBsrValC;
        marker_ = (__gm__ int32_t *)gmMarker;
        mb_ = tiling->mb;
        m_ = tiling->m;
        n_ = tiling->n;
        nb_ = tiling->nb;
        rowBlockDim_ = tiling->rowBlockDim;
        colBlockDim_ = tiling->colBlockDim;
        baseA_ = tiling->baseA;
        baseC_ = tiling->baseC;
        dir_ = tiling->dir;
        valSize_ = tiling->valSize;
        invColBlockDim_ = tiling->invColBlockDim;
        blockRowsPerCore_ = tiling->blockRowsPerCore;
    }

    __aicore__ inline void Process()
    {
        int32_t blockRowStart{};
        int32_t blockRowEnd{};
        uint32_t simtThreadNum{};
        ComputeBlockRowRangeAndThreads(blockRowStart, blockRowEnd, simtThreadNum);

        if (valSize_ == 4) {
            asc_vf_call<Csr2gebsrConvertSimtCompute<int32_t>>(
                dim3{simtThreadNum},
                (__gm__ const int32_t *)csrValA_, csrRowPtrA_, csrColIndA_,
                bsrRowPtrC_, bsrColIndC_, (__gm__ int32_t *)bsrValC_,
                marker_,
                m_, n_, nb_, rowBlockDim_, colBlockDim_,
                baseA_, baseC_, dir_, invColBlockDim_,
                blockRowStart, blockRowEnd,
                static_cast<int32_t>(simtThreadNum));
        } else if (valSize_ == 2) {
            asc_vf_call<Csr2gebsrConvertSimtCompute<int16_t>>(
                dim3{simtThreadNum},
                (__gm__ const int16_t *)csrValA_, csrRowPtrA_, csrColIndA_,
                bsrRowPtrC_, bsrColIndC_, (__gm__ int16_t *)bsrValC_,
                marker_,
                m_, n_, nb_, rowBlockDim_, colBlockDim_,
                baseA_, baseC_, dir_, invColBlockDim_,
                blockRowStart, blockRowEnd,
                static_cast<int32_t>(simtThreadNum));
        } else {
            return;
        }
    }

private:
    GM_ADDR csrValA_{nullptr};
    __gm__ const int32_t *csrRowPtrA_{nullptr};
    __gm__ const int32_t *csrColIndA_{nullptr};
    __gm__ const int32_t *bsrRowPtrC_{nullptr};
    __gm__ int32_t *bsrColIndC_{nullptr};
    GM_ADDR bsrValC_{nullptr};
    __gm__ int32_t *marker_{nullptr};
    int32_t m_{0};
    int32_t n_{0};
    int32_t nb_{0};
    int32_t rowBlockDim_{0};
    int32_t colBlockDim_{0};
    int32_t baseA_{0};
    int32_t baseC_{0};
    int32_t dir_{0};
    uint32_t valSize_{4};
    float invColBlockDim_{1.0f};
};

/// __global__ 调度器（Kernel 3: Convert）
extern "C" __global__ __aicore__ void csr2gebsr_convert_kernel(
    GM_ADDR gmCsrValA, GM_ADDR gmCsrRowPtrA, GM_ADDR gmCsrColIndA,
    GM_ADDR gmBsrRowPtrC, GM_ADDR gmBsrColIndC, GM_ADDR gmBsrValC,
    GM_ADDR gmMarker,
    const Csr2gebsrConvertTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    Csr2gebsrConvertDispatcher dispatcher;
    dispatcher.Init(gmCsrValA, gmCsrRowPtrA, gmCsrColIndA,
                    gmBsrRowPtrC, gmBsrColIndC, gmBsrValC,
                    gmMarker, &tiling);
    dispatcher.Process();
}

/// kernel_do 启动器（Kernel 3: Convert）
extern "C" void csr2gebsr_convert_kernel_do(
    GM_ADDR csrValA, GM_ADDR csrRowPtrA, GM_ADDR csrColIndA,
    GM_ADDR bsrRowPtrC, GM_ADDR bsrColIndC, GM_ADDR bsrValC,
    GM_ADDR marker,
    const Csr2gebsrConvertTilingData &tiling,
    uint32_t numBlocks,
    void *stream)
{
    csr2gebsr_convert_kernel<<<numBlocks, nullptr, stream>>>(
        csrValA, csrRowPtrA, csrColIndA,
        bsrRowPtrC, bsrColIndC, bsrValC,
        marker, tiling);
}
