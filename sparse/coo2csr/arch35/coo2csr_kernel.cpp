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
 * \file coo2csr_kernel.cpp
 * \brief coo2csr kernel 实现（SIMT 编程模型）。
 *
 * 代码层次结构：
 *   Layer 1: __simt_vf__ 计算函数（线程级并行）
 *   Layer 2: Dispatcher 类（管理 GM 指针 + asc_vf_call 分发）
 *   Layer 3: kernel_do 启动器（<<<>>> 语法，Host 调用入口）
 *
 * Kernel 1 (CountRows):           遍历 cooRowInd，RLE + asc_atomic_add 统计每行 nnz
 * Kernel 2 (PrefixSum, 单核):     exclusive prefix sum -> csrRowPtr
 * Kernel 2 Phase A (Local):       各 block 局部前缀和
 * Kernel 2 Phase B (Blocks):      block 总和排他前缀和
 * Kernel 2 Phase C (Correct):     偏移修正
 * Kernel Fused (Count+PrefixSum): 单 block 内完成 CountRows + PrefixSum（小规模 nnz 路径）
 */

#include "kernel_operator.h"
#include "simt_api/asc_simt.h"
#include "simt_api/device_atomic_functions.h"
#include "simt_api/device_sync_functions.h"
#include "simt_api/device_warp_functions.h"
#include "coo2csr_kernel.h"

// ===========================================================================
// 公共：自适应线程数计算
//
// nnz < kCoo2CsrThreadsPerBlock 时缩减线程数，
// 向上取整到 warp 倍数确保 warp 内线程均为 active 状态。
// ===========================================================================
__aicore__ inline uint32_t Coo2CsrComputeThreadNum(int32_t nnz)
{
    uint32_t simtThreadNum = kCoo2CsrThreadsPerBlock;
    if (static_cast<uint32_t>(nnz) < simtThreadNum) {
        simtThreadNum = static_cast<uint32_t>(nnz);
    }
    if (simtThreadNum == 0) {
        simtThreadNum = 1;
    }
    simtThreadNum = (simtThreadNum + kCoo2CsrWarpSize - 1u) &
        ~(kCoo2CsrWarpSize - 1u);
    return simtThreadNum;
}

// ===========================================================================
// 公共：RLE 计数 — 遍历 cooRowInd 连续段，合并相同行索引后原子加
//
// 对 cooRowInd[start..end-1] 做 RLE，每次行变化时 asc_atomic_add。
// 包含值域边界检查：curRow 超出 [0, m-1] 时跳过，防止 GM 越界写。
// ===========================================================================
__simt_callee__ __aicore__ inline void
Coo2CsrRleCount(
    __gm__ const int32_t *cooRowInd,
    __gm__ int32_t *rowCount,
    int32_t start, int32_t end,
    int32_t idxBase, int32_t m)
{
    if (start >= end) {
        return;
    }
    int32_t curRow = static_cast<int32_t>(static_cast<uint32_t>(cooRowInd[start]) - static_cast<uint32_t>(idxBase));
    int32_t count = 1;
    for (int32_t i = start + 1; i < end; i++) {
        int32_t row = static_cast<int32_t>(static_cast<uint32_t>(cooRowInd[i]) - static_cast<uint32_t>(idxBase));
        if (row == curRow) {
            count++;
        } else {
            if (curRow >= 0 && curRow < m) {
                (void)asc_atomic_add(&rowCount[curRow], count);
            }
            curRow = row;
            count = 1;
        }
    }
    if (curRow >= 0 && curRow < m) {
        (void)asc_atomic_add(&rowCount[curRow], count);
    }
}

// ===========================================================================
// 公共：排他前缀和 — rowCount[0..m-1] -> csrRowPtr[0..m]
//
// 使用 int64_t 做 runningSum 防止 nnz 接近 INT32_MAX 时溢出。
// ===========================================================================
__simt_callee__ __aicore__ inline void
Coo2CsrExclusivePrefixSum(
    __gm__ const int32_t *rowCount,
    __gm__ int32_t *csrRowPtr,
    int32_t m, int32_t idxBase)
{
    int64_t runningSum = 0;
    csrRowPtr[0] = idxBase;

    for (int32_t j = 0; j < m; j++) {
        runningSum += rowCount[j];
        csrRowPtr[j + 1] = static_cast<int32_t>(runningSum + idxBase);
    }
}

// ===========================================================================
// Kernel 1: CountRows — 连续 chunk + RLE，原子加统计每行 nnz
// ===========================================================================

__simt_vf__ __aicore__ __launch_bounds__(kCoo2CsrThreadsPerBlock) inline void
Coo2CsrCountRowsSimtCompute(
    __gm__ const int32_t *cooRowInd,
    __gm__ int32_t *rowCount,
    int32_t nnz, int32_t idxBase, int32_t m,
    int32_t numBlocks)
{
    int32_t globalTid = static_cast<int32_t>(threadIdx.x) +
        static_cast<int32_t>(blockIdx.x) * static_cast<int32_t>(blockDim.x);
    int32_t totalThreads = static_cast<int32_t>(blockDim.x) * numBlocks;

    // 连续 chunk 划分（int64_t 防溢出）
    int64_t chunkSize64 = (static_cast<int64_t>(nnz) + totalThreads - 1) / totalThreads;
    int32_t chunkSize = static_cast<int32_t>(chunkSize64);
    int64_t start64 = static_cast<int64_t>(globalTid) * chunkSize;
    if (start64 >= static_cast<int64_t>(nnz)) {
        return;
    }
    int32_t start = static_cast<int32_t>(start64);
    int64_t end64 = start64 + chunkSize64;
    if (end64 > static_cast<int64_t>(nnz)) {
        end64 = static_cast<int64_t>(nnz);
    }
    int32_t end = static_cast<int32_t>(end64);

    Coo2CsrRleCount(cooRowInd, rowCount, start, end, idxBase, m);
}

class Coo2CsrCountRowsDispatcher {
public:
    __aicore__ inline void Init(
        GM_ADDR gmRowInd, GM_ADDR gmWorkspace,
        int32_t nnz, int32_t idxBase, int32_t numBlocks, int32_t m)
    {
        cooRowInd_ = reinterpret_cast<__gm__ const int32_t *>(gmRowInd);
        rowCount_ = reinterpret_cast<__gm__ int32_t *>(gmWorkspace);
        nnz_ = nnz;
        idxBase_ = idxBase;
        numBlocks_ = numBlocks;
        m_ = m;
    }

    __aicore__ inline void Process()
    {
        uint32_t simtThreadNum = Coo2CsrComputeThreadNum(nnz_);

        asc_vf_call<Coo2CsrCountRowsSimtCompute>(dim3{simtThreadNum},
            cooRowInd_, rowCount_, nnz_, idxBase_, m_, numBlocks_);
    }

private:
    __gm__ const int32_t *cooRowInd_{nullptr};
    __gm__ int32_t *rowCount_{nullptr};
    int32_t nnz_{0};
    int32_t idxBase_{0};
    int32_t numBlocks_{0};
    int32_t m_{0};
};

extern "C" __global__ __aicore__ void coo2csr_count_kernel(
    GM_ADDR cooRowInd, GM_ADDR workspace,
    int32_t nnz, int32_t idxBase, int32_t numBlocks, int32_t m)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    Coo2CsrCountRowsDispatcher launcher;
    launcher.Init(cooRowInd, workspace, nnz, idxBase, numBlocks, m);
    launcher.Process();
}

extern "C" void coo2csr_count_kernel_do(
    GM_ADDR cooRowInd,
    GM_ADDR workspace,
    const Coo2CsrCountTilingData &tiling,
    uint32_t numBlocks,
    void *stream)
{
    coo2csr_count_kernel<<<numBlocks, 0, stream>>>(
        cooRowInd, workspace,
        tiling.nnz, tiling.idxBase, static_cast<int32_t>(numBlocks),
        tiling.m);
}

// ===========================================================================
// Kernel 2 (单核): PrefixSum — exclusive prefix sum -> csrRowPtr
// ===========================================================================

__simt_vf__ __aicore__ __launch_bounds__(1) inline void
Coo2CsrPrefixSumSimtCompute(
    __gm__ const int32_t *rowCount,
    __gm__ int32_t *csrRowPtr,
    int32_t m, int32_t idxBase)
{
    Coo2CsrExclusivePrefixSum(rowCount, csrRowPtr, m, idxBase);
}

class Coo2CsrPrefixSumDispatcher {
public:
    __aicore__ inline void Init(
        GM_ADDR gmWorkspace, GM_ADDR gmCsrRowPtr,
        int32_t m, int32_t idxBase)
    {
        rowCount_ = reinterpret_cast<__gm__ const int32_t *>(gmWorkspace);
        csrRowPtr_ = reinterpret_cast<__gm__ int32_t *>(gmCsrRowPtr);
        m_ = m;
        idxBase_ = idxBase;
    }

    __aicore__ inline void Process()
    {
        asc_vf_call<Coo2CsrPrefixSumSimtCompute>(dim3{1},
            rowCount_, csrRowPtr_, m_, idxBase_);
    }

private:
    __gm__ const int32_t *rowCount_{nullptr};
    __gm__ int32_t *csrRowPtr_{nullptr};
    int32_t m_{0};
    int32_t idxBase_{0};
};

extern "C" __global__ __aicore__ void coo2csr_prefixsum_kernel(
    GM_ADDR workspace, GM_ADDR csrRowPtr,
    int32_t m, int32_t idxBase)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    Coo2CsrPrefixSumDispatcher launcher;
    launcher.Init(workspace, csrRowPtr, m, idxBase);
    launcher.Process();
}

extern "C" void coo2csr_prefixsum_kernel_do(
    GM_ADDR workspace,
    GM_ADDR csrRowPtr,
    const Coo2CsrPrefixSumTilingData &tiling,
    void *stream)
{
    coo2csr_prefixsum_kernel<<<1, 0, stream>>>(
        workspace, csrRowPtr, tiling.m, tiling.idxBase);
}

// ===========================================================================
// 公共：warp inclusive scan — 子段串行前缀和 + warp shuffle scan + 偏移修正
//
// 供 Phase A (Coo2CsrPrefixSumLocalSimtCompute) 和 Fused
// (Coo2CsrFusedSimtCompute) 共用。各线程先对子段 [myStart,myEnd) 做串行
// 前缀和（相对子段起点），再用 asc_shfl_up 在 warp 内做 inclusive scan，
// 导出 exclusive offset 修正 csrRowPtr 为全局值。返回 warp 总和。
// ===========================================================================
__simt_callee__ __aicore__ inline int32_t
Coo2CsrWarpInclusiveScan(
    __gm__ const int32_t *rowCount, __gm__ int32_t *csrRowPtr,
    int32_t myStart, int32_t myEnd, int32_t lane, int32_t idxBase)
{
    constexpr int32_t kScanWidth = static_cast<int32_t>(kCoo2CsrWarpSize);
    int64_t localSum = 0;
    for (int32_t j = myStart; j < myEnd; j++) {
        localSum += rowCount[j];
        csrRowPtr[j + 1] = static_cast<int32_t>(localSum + idxBase);
    }
    int32_t val = static_cast<int32_t>(localSum);
    for (int32_t offset = 1; offset < kScanWidth; offset <<= 1) {
        int32_t n = asc_shfl_up(val, static_cast<uint32_t>(offset));
        if (lane >= offset) {
            val += n;
        }
    }
    int32_t total = asc_shfl(val, kScanWidth - 1);
    int32_t shifted = asc_shfl_up(val, 1);
    int32_t exclusiveOffset = (lane == 0) ? 0 : shifted;
    for (int32_t j = myStart; j < myEnd; j++) {
        int64_t v = static_cast<int64_t>(csrRowPtr[j + 1]) + exclusiveOffset;
        csrRowPtr[j + 1] = static_cast<int32_t>(v);
    }
    return total;
}

// ===========================================================================
// Kernel 2 Phase A (PrefixSum Local): 各 block 计算局部排他前缀和
//
// 每 block 启用 kCoo2CsrWarpSize(32) 线程，将 chunkSize 切分为 32 个子段，
// 用 warp shuffle scan 完成线程间交换，无需 UB。
// ===========================================================================

__simt_vf__ __aicore__ __launch_bounds__(kCoo2CsrWarpSize) inline void
Coo2CsrPrefixSumLocalSimtCompute(
    __gm__ const int32_t *rowCount,
    __gm__ int32_t *csrRowPtr,
    __gm__ int32_t *blockTotals,
    int32_t m, int32_t idxBase,
    int32_t chunkSize)
{
    constexpr int32_t kScanWidth = static_cast<int32_t>(kCoo2CsrWarpSize);
    int32_t blockId = static_cast<int32_t>(blockIdx.x);
    int64_t start64 = static_cast<int64_t>(blockId) * chunkSize;
    int64_t end64 = start64 + chunkSize;
    if (end64 > static_cast<int64_t>(m)) {
        end64 = static_cast<int64_t>(m);
    }
    int32_t start = static_cast<int32_t>(start64);
    int32_t end = static_cast<int32_t>(end64);
    int32_t len = end - start;

    int32_t lane = static_cast<int32_t>(threadIdx.x);
    int32_t elemPerThread = (len + kScanWidth - 1) / kScanWidth;
    int32_t myStart = start + lane * elemPerThread;
    int32_t myEnd = myStart + elemPerThread;
    if (myEnd > end) {
        myEnd = end;
    }
    if (myStart > end) {
        myStart = end;
    }

    int32_t total = Coo2CsrWarpInclusiveScan(
        rowCount, csrRowPtr, myStart, myEnd, lane, idxBase);

    if (lane == 0) {
        blockTotals[blockId] = total;
        if (blockId == 0) {
            csrRowPtr[0] = idxBase;
        }
    }
}

class Coo2CsrPrefixSumLocalDispatcher {
public:
    __aicore__ inline void Init(
        GM_ADDR gmWorkspace, GM_ADDR gmCsrRowPtr, GM_ADDR gmBlockTotals,
        int32_t m, int32_t idxBase,
        int32_t chunkSize)
    {
        rowCount_ = reinterpret_cast<__gm__ const int32_t *>(gmWorkspace);
        csrRowPtr_ = reinterpret_cast<__gm__ int32_t *>(gmCsrRowPtr);
        blockTotals_ = reinterpret_cast<__gm__ int32_t *>(gmBlockTotals);
        m_ = m;
        idxBase_ = idxBase;
        chunkSize_ = chunkSize;
    }

    __aicore__ inline void Process()
    {
        asc_vf_call<Coo2CsrPrefixSumLocalSimtCompute>(dim3{kCoo2CsrWarpSize},
            rowCount_, csrRowPtr_, blockTotals_,
            m_, idxBase_, chunkSize_);
    }

private:
    __gm__ const int32_t *rowCount_{nullptr};
    __gm__ int32_t *csrRowPtr_{nullptr};
    __gm__ int32_t *blockTotals_{nullptr};
    int32_t m_{0};
    int32_t idxBase_{0};
    int32_t chunkSize_{0};
};

extern "C" __global__ __aicore__ void coo2csr_prefixsum_local_kernel(
    GM_ADDR workspace, GM_ADDR csrRowPtr, GM_ADDR blockTotals,
    int32_t m, int32_t idxBase,
    int32_t chunkSize)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    Coo2CsrPrefixSumLocalDispatcher launcher;
    launcher.Init(workspace, csrRowPtr, blockTotals,
                  m, idxBase, chunkSize);
    launcher.Process();
}

extern "C" void coo2csr_prefixsum_local_kernel_do(
    GM_ADDR workspace,
    GM_ADDR csrRowPtr,
    GM_ADDR blockTotals,
    const Coo2CsrPrefixSumLocalTilingData &tiling,
    uint32_t numBlocks,
    void *stream)
{
    coo2csr_prefixsum_local_kernel<<<numBlocks, 0, stream>>>(
        workspace, csrRowPtr, blockTotals,
        tiling.m, tiling.idxBase, tiling.chunkSize);
}

// ===========================================================================
// Kernel 2 Phase B (PrefixSum Blocks): 单 block 对 blockTotals 排他前缀和
// ===========================================================================

__simt_vf__ __aicore__ __launch_bounds__(1) inline void
Coo2CsrPrefixSumBlocksSimtCompute(
    __gm__ int32_t *blockTotals,
    uint32_t numBlocks)
{
    int64_t running = 0;
    for (uint32_t b = 0; b < numBlocks; b++) {
        int32_t temp = blockTotals[b];
        blockTotals[b] = static_cast<int32_t>(running);
        running += temp;
    }
}

class Coo2CsrPrefixSumBlocksDispatcher {
public:
    __aicore__ inline void Init(
        GM_ADDR gmBlockTotals, uint32_t numBlocks)
    {
        blockTotals_ = reinterpret_cast<__gm__ int32_t *>(gmBlockTotals);
        numBlocks_ = numBlocks;
    }

    __aicore__ inline void Process()
    {
        asc_vf_call<Coo2CsrPrefixSumBlocksSimtCompute>(dim3{1},
            blockTotals_, numBlocks_);
    }

private:
    __gm__ int32_t *blockTotals_{nullptr};
    uint32_t numBlocks_{0};
};

extern "C" __global__ __aicore__ void coo2csr_prefixsum_blocks_kernel(
    GM_ADDR blockTotals, uint32_t numBlocks)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    Coo2CsrPrefixSumBlocksDispatcher launcher;
    launcher.Init(blockTotals, numBlocks);
    launcher.Process();
}

extern "C" void coo2csr_prefixsum_blocks_kernel_do(
    GM_ADDR blockTotals,
    const Coo2CsrPrefixSumBlocksTilingData &tiling,
    void *stream)
{
    coo2csr_prefixsum_blocks_kernel<<<1, 0, stream>>>(
        blockTotals, tiling.numBlocks);
}

// ===========================================================================
// Kernel 2 Phase C (PrefixSum Correct): 各 block 将 block 偏移加到 csrRowPtr
//
// 偏移修正无数据依赖，各元素独立加 offset，用 grid-stride loop 多线程并行。
// ===========================================================================

__simt_vf__ __aicore__ __launch_bounds__(kCoo2CsrWarpSize) inline void
Coo2CsrPrefixSumCorrectSimtCompute(
    __gm__ int32_t *csrRowPtr,
    __gm__ const int32_t *blockTotals,
    int32_t m,
    int32_t chunkSize)
{
    int32_t blockId = static_cast<int32_t>(blockIdx.x);
    int64_t start64 = static_cast<int64_t>(blockId) * chunkSize;
    int64_t end64 = start64 + chunkSize;
    if (end64 > static_cast<int64_t>(m)) {
        end64 = static_cast<int64_t>(m);
    }
    int32_t start = static_cast<int32_t>(start64);
    int32_t end = static_cast<int32_t>(end64);

    int32_t offset = blockTotals[blockId];
    int32_t tid = static_cast<int32_t>(threadIdx.x);
    int32_t stride = static_cast<int32_t>(blockDim.x);
    for (int32_t j = start + tid; j < end; j += stride) {
        int64_t val = static_cast<int64_t>(csrRowPtr[j + 1]) + offset;
        csrRowPtr[j + 1] = static_cast<int32_t>(val);
    }
}

class Coo2CsrPrefixSumCorrectDispatcher {
public:
    __aicore__ inline void Init(
        GM_ADDR gmCsrRowPtr, GM_ADDR gmBlockTotals,
        int32_t m, int32_t chunkSize)
    {
        csrRowPtr_ = reinterpret_cast<__gm__ int32_t *>(gmCsrRowPtr);
        blockTotals_ = reinterpret_cast<__gm__ const int32_t *>(gmBlockTotals);
        m_ = m;
        chunkSize_ = chunkSize;
    }

    __aicore__ inline void Process()
    {
        asc_vf_call<Coo2CsrPrefixSumCorrectSimtCompute>(dim3{kCoo2CsrWarpSize},
            csrRowPtr_, blockTotals_, m_, chunkSize_);
    }

private:
    __gm__ int32_t *csrRowPtr_{nullptr};
    __gm__ const int32_t *blockTotals_{nullptr};
    int32_t m_{0};
    int32_t chunkSize_{0};
};

extern "C" __global__ __aicore__ void coo2csr_prefixsum_correct_kernel(
    GM_ADDR csrRowPtr, GM_ADDR blockTotals,
    int32_t m, int32_t chunkSize)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    Coo2CsrPrefixSumCorrectDispatcher launcher;
    launcher.Init(csrRowPtr, blockTotals, m, chunkSize);
    launcher.Process();
}

extern "C" void coo2csr_prefixsum_correct_kernel_do(
    GM_ADDR csrRowPtr,
    GM_ADDR gmBlockTotals,
    const Coo2CsrPrefixSumCorrectTilingData &tiling,
    uint32_t numBlocks,
    void *stream)
{
    coo2csr_prefixsum_correct_kernel<<<numBlocks, 0, stream>>>(
        csrRowPtr, gmBlockTotals, tiling.m, tiling.chunkSize);
}

// ===========================================================================
// Kernel Fused: 单 block 内完成 CountRows + PrefixSum
//
// 小规模 nnz 路径：线程内 RLE + asc_atomic_add 统计每行 nnz，
// asc_syncthreads() 同步后，warp shuffle scan 做排他前缀和 -> csrRowPtr。
// ===========================================================================

__simt_vf__ __aicore__ __launch_bounds__(kCoo2CsrThreadsPerBlock) inline void
Coo2CsrFusedSimtCompute(
    __gm__ const int32_t *cooRowInd,
    __gm__ int32_t *rowCount,
    __gm__ int32_t *csrRowPtr,
    int32_t nnz, int32_t m, int32_t idxBase)
{
    if (nnz > 0) {
        int32_t tid = static_cast<int32_t>(threadIdx.x);
        int32_t totalThreads = static_cast<int32_t>(blockDim.x);
        // int64_t 防溢出
        int64_t chunkSize64 = (static_cast<int64_t>(nnz) + totalThreads - 1) / totalThreads;
        int32_t chunkSize = static_cast<int32_t>(chunkSize64);
        int64_t start64 = static_cast<int64_t>(tid) * chunkSize;
        if (start64 < static_cast<int64_t>(nnz)) {
            int64_t end64 = start64 + chunkSize64;
            if (end64 > static_cast<int64_t>(nnz)) {
                end64 = static_cast<int64_t>(nnz);
            }
            int32_t start = static_cast<int32_t>(start64);
            int32_t end = static_cast<int32_t>(end64);
            Coo2CsrRleCount(cooRowInd, rowCount, start, end, idxBase, m);
        }
    }

    asc_syncthreads();

    if (nnz == 0) {
        int32_t tid = static_cast<int32_t>(threadIdx.x);
        int32_t totalThreads = static_cast<int32_t>(blockDim.x);
        for (int32_t j = tid; j <= m; j += totalThreads) {
            csrRowPtr[j] = idxBase;
        }
        return;
    }

    // 前缀和阶段：warp shuffle scan 并行（32 线程），复用 Coo2CsrWarpInclusiveScan。
    if (threadIdx.x < kCoo2CsrWarpSize) {
        constexpr int32_t kScanWidth = static_cast<int32_t>(kCoo2CsrWarpSize);
        int32_t lane = static_cast<int32_t>(threadIdx.x);
        int32_t elemPerThread = (m + kScanWidth - 1) / kScanWidth;
        int32_t myStart = lane * elemPerThread;
        int32_t myEnd = myStart + elemPerThread;
        if (myEnd > m) {
            myEnd = m;
        }
        if (myStart > m) {
            myStart = m;
        }
        (void)Coo2CsrWarpInclusiveScan(
            rowCount, csrRowPtr, myStart, myEnd, lane, idxBase);
        if (lane == 0) {
            csrRowPtr[0] = idxBase;
        }
    }
}

class Coo2CsrFusedDispatcher {
public:
    __aicore__ inline void Init(
        GM_ADDR gmRowInd, GM_ADDR gmWorkspace, GM_ADDR gmCsrRowPtr,
        int32_t nnz, int32_t m, int32_t idxBase)
    {
        cooRowInd_ = reinterpret_cast<__gm__ const int32_t *>(gmRowInd);
        rowCount_ = reinterpret_cast<__gm__ int32_t *>(gmWorkspace);
        csrRowPtr_ = reinterpret_cast<__gm__ int32_t *>(gmCsrRowPtr);
        nnz_ = nnz;
        m_ = m;
        idxBase_ = idxBase;
    }

    __aicore__ inline void Process()
    {
        uint32_t simtThreadNum = Coo2CsrComputeThreadNum(nnz_);

        asc_vf_call<Coo2CsrFusedSimtCompute>(dim3{simtThreadNum},
            cooRowInd_, rowCount_, csrRowPtr_, nnz_, m_, idxBase_);
    }

private:
    __gm__ const int32_t *cooRowInd_{nullptr};
    __gm__ int32_t *rowCount_{nullptr};
    __gm__ int32_t *csrRowPtr_{nullptr};
    int32_t nnz_{0};
    int32_t m_{0};
    int32_t idxBase_{0};
};

extern "C" __global__ __aicore__ void coo2csr_fused_kernel(
    GM_ADDR cooRowInd, GM_ADDR workspace, GM_ADDR csrRowPtr,
    int32_t nnz, int32_t m, int32_t idxBase)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    Coo2CsrFusedDispatcher launcher;
    launcher.Init(cooRowInd, workspace, csrRowPtr, nnz, m, idxBase);
    launcher.Process();
}

extern "C" void coo2csr_fused_kernel_do(
    GM_ADDR cooRowInd,
    GM_ADDR workspace,
    GM_ADDR csrRowPtr,
    const Coo2CsrFusedTilingData &tiling,
    void *stream)
{
    coo2csr_fused_kernel<<<1, 0, stream>>>(
        cooRowInd, workspace, csrRowPtr, tiling.nnz, tiling.m, tiling.idxBase);
}
