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
 * \file csrgeam2_kernel.cpp
 * \brief csrgeam2 kernel 实现（SIMT，仅 arch35 可用）。
 *
 * 三层结构（参考 nnz_kernel.cpp 的 class-based 混合编程模式）：
 *   Layer 1: __simt_vf__ 计算函数（线程级并行，grid-stride loop）
 *   Layer 2: __global__ 调度器（读 tiling → asc_vf_call 分发）
 *   Layer 3: _kernel_do 启动器（<<<>>> 语法）
 *
 * 三个 kernel：
 *   Kernel 1 (Nnz):        逐行双指针归并计数 → nnzPerRow
 *   Kernel 2 (Compute):    逐行双指针归并计算 → csrColIndC + csrValC
 *   Kernel 3 (Prefix Sum): device 侧 exclusive prefix sum → rowPtrC + nnzC
 */

#include <cstdint>
#include "kernel_operator.h"
#include "simt_api/asc_simt.h"
#include "csrgeam2_kernel.h"

// ===========================================================================
// Common helpers (shared across Kernel 1 / Kernel 2)
// ===========================================================================

// Row iterator: holds the current row index and 0-based A/B bound offsets.
// Populated by Csrgeam2LoadRow() to avoid duplicating row-bound reads across
// the Nnz and Compute VF functions.
struct Csrgeam2RowIter {
    int32_t r;    // actual row index (rowStart + row)
    int32_t pA;   // A row start (0-based)
    int32_t qA;   // A row end   (0-based)
    int32_t pB;   // B row start (0-based)
    int32_t qB;   // B row end   (0-based)
};

// Fetch A/B row bounds for a given (rowStart + row) and convert to 0-based.
// Marked __simt_callee__ so it can be called from __simt_vf__ functions.
__simt_callee__ __aicore__ inline Csrgeam2RowIter Csrgeam2LoadRow(
    __gm__ const int32_t *rowPtrA,
    __gm__ const int32_t *rowPtrB,
    int32_t rowStart, int32_t row,
    int32_t baseA, int32_t baseB)
{
    Csrgeam2RowIter it;
    it.r = rowStart + row;
    it.pA = rowPtrA[it.r] - baseA;
    it.qA = rowPtrA[it.r + 1] - baseA;
    it.pB = rowPtrB[it.r] - baseB;
    it.qB = rowPtrB[it.r + 1] - baseB;
    return it;
}

// Common dispatcher base: encapsulates block-id-to-row-range mapping and
// SIMT thread-count computation, shared by NnzDispatcher and ComputeDispatcher.
struct Csrgeam2DispatcherBase {
    int32_t m_{0};
    uint32_t rowsPerBlock_{0};

    // Compute the [rowStart, rowEnd) range for this core and the aligned
    // SIMT thread count.  Called inside Process() of each derived dispatcher.
    __aicore__ inline void ComputeRowRangeAndThreads(
        int32_t &rowStart, int32_t &rowEnd, uint32_t &simtThreadNum)
    {
        int32_t outerId = static_cast<int32_t>(AscendC::GetBlockIdx());
        rowStart = outerId * static_cast<int32_t>(rowsPerBlock_);
        rowEnd = rowStart + static_cast<int32_t>(rowsPerBlock_);
        if (rowEnd > m_) {
            rowEnd = m_;
        }
        int32_t coreRangeLen = rowEnd - rowStart;

        // SIMT thread count: clamp to core range, round up to warp multiple
        simtThreadNum = kCsrgeam2MaxThreadsPerBlock;
        if (static_cast<uint32_t>(coreRangeLen) < simtThreadNum) {
            simtThreadNum = static_cast<uint32_t>(coreRangeLen);
        }
        // Defensive: ComputeBlockSplits guarantees useBlocks <= ceil(m/maxThreadsPerBlock),
        // so the last core's range is >= 1 row. Guard retained for safety (#26).
        if (simtThreadNum == 0) {
            simtThreadNum = 1;
        }
        simtThreadNum = (simtThreadNum + kCsrgeam2WarpSize - 1u) & ~(kCsrgeam2WarpSize - 1u);
    }
};

// ===========================================================================
// Kernel 1: Nnz 计数
// ===========================================================================

// SIMT VF 计算函数 - 逐行双指针归并计数 A∪B 的非零元个数
//
// 参数说明：
//   rowStart, rowEnd: 本 core 负责的行范围（由 dispatcher 根据 GetBlockIdx() 计算）
//   threadsPerCore:   本 core 的 SIMT 线程数（用于 grid-stride 步长）
//
// 注意：dav-3510 上 __simt_vf__ 内的 blockDim.x 返回外层 grid 的 blockDim
// （即 numBlocks），而非 asc_vf_call 的 dim3 参数。因此禁止使用
// blockDim.x / blockIdx.x，改用显式传入的 threadsPerCore 和 threadIdx.x。
__simt_vf__ __aicore__ __launch_bounds__(kCsrgeam2MaxThreadsPerBlock) inline void
Csrgeam2NnzSimtCompute(
    __gm__ const int32_t *rowPtrA,
    __gm__ const int32_t *colIndA,
    __gm__ const int32_t *rowPtrB,
    __gm__ const int32_t *colIndB,
    __gm__ int32_t *nnzPerRow,
    int32_t baseA, int32_t baseB,
    int32_t rowStart, int32_t rowEnd,
    int32_t threadsPerCore)
{
    const int32_t rangeLen = rowEnd - rowStart;

    for (int32_t row = static_cast<int32_t>(threadIdx.x);
         row < rangeLen;
         row += threadsPerCore) {
        Csrgeam2RowIter it = Csrgeam2LoadRow(rowPtrA, rowPtrB, rowStart, row, baseA, baseB);

        int32_t count = 0;
        // 双指针有序归并
        while (it.pA < it.qA && it.pB < it.qB) {
            int32_t colA0 = colIndA[it.pA] - baseA;
            int32_t colB0 = colIndB[it.pB] - baseB;
            if (colA0 < colB0) {
                it.pA++;
            } else if (colA0 > colB0) {
                it.pB++;
            } else {
                // 列索引相同：A 和 B 都有此位置，归并为 1
                it.pA++;
                it.pB++;
            }
            count++;
        }
        // 残余元素
        count += (it.qA - it.pA) + (it.qB - it.pB);
        nnzPerRow[it.r] = count;
    }
}

// SIMD 侧封装类：管理 GM 指针 + 调用 asc_vf_call
class Csrgeam2NnzDispatcher : public Csrgeam2DispatcherBase {
public:
    __aicore__ inline void Init(
        GM_ADDR gmRowPtrA, GM_ADDR gmColIndA,
        GM_ADDR gmRowPtrB, GM_ADDR gmColIndB,
        GM_ADDR gmNnzPerRow,
        const Csrgeam2NnzTilingData *tiling)
    {
        rowPtrA_ = (__gm__ const int32_t *)gmRowPtrA;
        colIndA_ = (__gm__ const int32_t *)gmColIndA;
        rowPtrB_ = (__gm__ const int32_t *)gmRowPtrB;
        colIndB_ = (__gm__ const int32_t *)gmColIndB;
        nnzPerRow_ = (__gm__ int32_t *)gmNnzPerRow;
        m_ = tiling->m;
        baseA_ = tiling->baseA;
        baseB_ = tiling->baseB;
        rowsPerBlock_ = tiling->rowsPerBlock;
    }

    __aicore__ inline void Process()
    {
        int32_t rowStart{};
        int32_t rowEnd{};
        uint32_t simtThreadNum{};
        ComputeRowRangeAndThreads(rowStart, rowEnd, simtThreadNum);

        asc_vf_call<Csrgeam2NnzSimtCompute>(
            dim3{simtThreadNum},
            rowPtrA_, colIndA_, rowPtrB_, colIndB_, nnzPerRow_,
            baseA_, baseB_,
            rowStart, rowEnd,
            static_cast<int32_t>(simtThreadNum));
    }

private:
    __gm__ const int32_t *rowPtrA_{nullptr};
    __gm__ const int32_t *colIndA_{nullptr};
    __gm__ const int32_t *rowPtrB_{nullptr};
    __gm__ const int32_t *colIndB_{nullptr};
    __gm__ int32_t *nnzPerRow_{nullptr};
    int32_t baseA_{0};
    int32_t baseB_{0};
};

// __global__ 调度器（Kernel 1）
extern "C" __global__ __aicore__ void csrgeam2_nnz_kernel(
    GM_ADDR gmRowPtrA, GM_ADDR gmColIndA,
    GM_ADDR gmRowPtrB, GM_ADDR gmColIndB,
    GM_ADDR gmNnzPerRow,
    const Csrgeam2NnzTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    Csrgeam2NnzDispatcher dispatcher;
    dispatcher.Init(gmRowPtrA, gmColIndA, gmRowPtrB, gmColIndB,
                    gmNnzPerRow, &tiling);
    dispatcher.Process();
}

// kernel_do 启动器（Kernel 1: Nnz 计数）
extern "C" void csrgeam2_nnz_kernel_do(
    GM_ADDR rowPtrA, GM_ADDR colIndA,
    GM_ADDR rowPtrB, GM_ADDR colIndB,
    GM_ADDR nnzPerRow,
    const Csrgeam2NnzTilingData &tiling,
    uint32_t numBlocks,
    void *stream)
{
    csrgeam2_nnz_kernel<<<numBlocks, nullptr, stream>>>(
        rowPtrA, colIndA, rowPtrB, colIndB, nnzPerRow, tiling);
}

// ===========================================================================
// Kernel 2: 主计算（逐行合并）
// ===========================================================================

// SIMT VF 计算函数 - 逐行有序列归并 C = α·A + β·B
//
// 参数说明：
//   rowStart, rowEnd: 本 core 负责的行范围（由 dispatcher 根据 GetBlockIdx() 计算）
//   threadsPerCore:   本 core 的 SIMT 线程数（用于 grid-stride 步长）
//
// 参见 Csrgeam2NnzSimtCompute 注释：禁止使用 blockDim.x / blockIdx.x。
__simt_vf__ __aicore__ __launch_bounds__(kCsrgeam2MaxThreadsPerBlock) inline void
Csrgeam2ComputeSimtCompute(
    __gm__ const int32_t *rowPtrA,
    __gm__ const int32_t *colIndA,
    __gm__ const float *valA,
    __gm__ const int32_t *rowPtrB,
    __gm__ const int32_t *colIndB,
    __gm__ const float *valB,
    __gm__ const int32_t *rowPtrC,
    __gm__ int32_t *colIndC,
    __gm__ float *valC,
    int32_t baseA, int32_t baseB, int32_t baseC,
    float alpha, float beta,
    int32_t rowStart, int32_t rowEnd,
    int32_t threadsPerCore)
{
    const int32_t rangeLen = rowEnd - rowStart;

    for (int32_t row = static_cast<int32_t>(threadIdx.x);
         row < rangeLen;
         row += threadsPerCore) {
        Csrgeam2RowIter it = Csrgeam2LoadRow(rowPtrA, rowPtrB, rowStart, row, baseA, baseB);
        // C 的写入起始位置（已由 Nnz 阶段填充）
        int32_t pC = rowPtrC[it.r] - baseC;

        // 双指针有序归并
        while (it.pA < it.qA && it.pB < it.qB) {
            int32_t colA0 = colIndA[it.pA] - baseA;
            int32_t colB0 = colIndB[it.pB] - baseB;

            if (colA0 < colB0) {
                colIndC[pC] = colA0 + baseC;
                valC[pC] = alpha * valA[it.pA];
                it.pA++;
                pC++;
            } else if (colA0 > colB0) {
                colIndC[pC] = colB0 + baseC;
                valC[pC] = beta * valB[it.pB];
                it.pB++;
                pC++;
            } else {
                // 列索引相同: C = α·A + β·B
                colIndC[pC] = colA0 + baseC;
                valC[pC] = alpha * valA[it.pA] + beta * valB[it.pB];
                it.pA++;
                it.pB++;
                pC++;
            }
        }

        // 残余 A
        while (it.pA < it.qA) {
            int32_t colA0 = colIndA[it.pA] - baseA;
            colIndC[pC] = colA0 + baseC;
            valC[pC] = alpha * valA[it.pA];
            it.pA++;
            pC++;
        }

        // 残余 B
        while (it.pB < it.qB) {
            int32_t colB0 = colIndB[it.pB] - baseB;
            colIndC[pC] = colB0 + baseC;
            valC[pC] = beta * valB[it.pB];
            it.pB++;
            pC++;
        }
    }
}

// SIMD 侧封装类：管理 GM 指针 + 调用 asc_vf_call
class Csrgeam2ComputeDispatcher : public Csrgeam2DispatcherBase {
public:
    __aicore__ inline void Init(
        GM_ADDR gmRowPtrA, GM_ADDR gmColIndA, GM_ADDR gmValA,
        GM_ADDR gmRowPtrB, GM_ADDR gmColIndB, GM_ADDR gmValB,
        GM_ADDR gmRowPtrC, GM_ADDR gmColIndC, GM_ADDR gmValC,
        const Csrgeam2TilingData *tiling)
    {
        rowPtrA_ = (__gm__ const int32_t *)gmRowPtrA;
        colIndA_ = (__gm__ const int32_t *)gmColIndA;
        valA_ = (__gm__ const float *)gmValA;
        rowPtrB_ = (__gm__ const int32_t *)gmRowPtrB;
        colIndB_ = (__gm__ const int32_t *)gmColIndB;
        valB_ = (__gm__ const float *)gmValB;
        rowPtrC_ = (__gm__ const int32_t *)gmRowPtrC;
        colIndC_ = (__gm__ int32_t *)gmColIndC;
        valC_ = (__gm__ float *)gmValC;
        m_ = tiling->m;
        baseA_ = tiling->baseA;
        baseB_ = tiling->baseB;
        baseC_ = tiling->baseC;
        // DEVICE mode: alpha/beta 是 device 指针，kernel 直接从 device 内存读取
        if (tiling->alphaPtr != 0ULL) {
            alpha_ = *(__gm__ const float *)tiling->alphaPtr;
            beta_ = *(__gm__ const float *)tiling->betaPtr;
        } else {
            alpha_ = tiling->alpha;
            beta_ = tiling->beta;
        }
        rowsPerBlock_ = tiling->rowsPerBlock;
    }

    __aicore__ inline void Process()
    {
        int32_t rowStart{};
        int32_t rowEnd{};
        uint32_t simtThreadNum{};
        ComputeRowRangeAndThreads(rowStart, rowEnd, simtThreadNum);

        asc_vf_call<Csrgeam2ComputeSimtCompute>(
            dim3{simtThreadNum},
            rowPtrA_, colIndA_, valA_,
            rowPtrB_, colIndB_, valB_,
            rowPtrC_, colIndC_, valC_,
            baseA_, baseB_, baseC_,
            alpha_, beta_,
            rowStart, rowEnd,
            static_cast<int32_t>(simtThreadNum));
    }

private:
    __gm__ const int32_t *rowPtrA_{nullptr};
    __gm__ const int32_t *colIndA_{nullptr};
    __gm__ const float *valA_{nullptr};
    __gm__ const int32_t *rowPtrB_{nullptr};
    __gm__ const int32_t *colIndB_{nullptr};
    __gm__ const float *valB_{nullptr};
    __gm__ const int32_t *rowPtrC_{nullptr};
    __gm__ int32_t *colIndC_{nullptr};
    __gm__ float *valC_{nullptr};
    int32_t baseA_{0};
    int32_t baseB_{0};
    int32_t baseC_{0};
    float alpha_{0.0f};
    float beta_{0.0f};
};

// __global__ 调度器（Kernel 2: 主计算）
extern "C" __global__ __aicore__ void csrgeam2_compute_kernel(
    GM_ADDR gmRowPtrA, GM_ADDR gmColIndA, GM_ADDR gmValA,
    GM_ADDR gmRowPtrB, GM_ADDR gmColIndB, GM_ADDR gmValB,
    GM_ADDR gmRowPtrC, GM_ADDR gmColIndC, GM_ADDR gmValC,
    const Csrgeam2TilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    Csrgeam2ComputeDispatcher dispatcher;
    dispatcher.Init(gmRowPtrA, gmColIndA, gmValA,
                    gmRowPtrB, gmColIndB, gmValB,
                    gmRowPtrC, gmColIndC, gmValC,
                    &tiling);
    dispatcher.Process();
}

// kernel_do 启动器（Kernel 2: 主计算）
extern "C" void csrgeam2_compute_kernel_do(
    GM_ADDR rowPtrA, GM_ADDR colIndA, GM_ADDR valA,
    GM_ADDR rowPtrB, GM_ADDR colIndB, GM_ADDR valB,
    GM_ADDR rowPtrC, GM_ADDR colIndC, GM_ADDR valC,
    const Csrgeam2TilingData &tiling,
    uint32_t numBlocks,
    void *stream)
{
    csrgeam2_compute_kernel<<<numBlocks, nullptr, stream>>>(
        rowPtrA, colIndA, valA,
        rowPtrB, colIndB, valB,
        rowPtrC, colIndC, valC,
        tiling);
}

// ===========================================================================
// Kernel 3: Prefix Sum（device 侧 exclusive prefix sum，单 block）
// ===========================================================================

// SIMT VF 计算函数 - exclusive prefix sum
//
// 约束：__simt_vf__ 只能由 thread 0 执行，因此本函数为纯顺序 O(m) 操作。
// 不使用 blockDim.x / blockIdx.x（dav-3510 bug），dim3 固定为 {1}。
//
// 读 nnzPerRow[0..m-1]，写 rowPtrC[0..m]，写 nnzCDev[0]。
__simt_vf__ __aicore__ inline void
Csrgeam2PrefixSumSimtCompute(
    __gm__ const int32_t *nnzPerRow,
    __gm__ int32_t *rowPtrC,
    __gm__ int32_t *nnzCDev,
    int32_t m,
    int32_t baseC)
{
    int32_t runningSum = 0;
    rowPtrC[0] = baseC;
    for (int32_t i = 0; i < m; i++) {
        runningSum += nnzPerRow[i];
        rowPtrC[i + 1] = runningSum + baseC;
    }
    nnzCDev[0] = runningSum;
}

// SIMD 侧封装类：管理 GM 指针 + 调用 asc_vf_call（单线程）
class Csrgeam2PrefixSumDispatcher {
public:
    __aicore__ inline void Init(
        GM_ADDR gmNnzPerRow, GM_ADDR gmRowPtrC, GM_ADDR gmNnzCDev,
        const Csrgeam2PrefixSumTilingData *tiling)
    {
        nnzPerRow_ = (__gm__ const int32_t *)gmNnzPerRow;
        rowPtrC_   = (__gm__ int32_t *)gmRowPtrC;
        nnzCDev_   = (__gm__ int32_t *)gmNnzCDev;
        m_         = tiling->m;
        baseC_     = tiling->baseC;
    }

    __aicore__ inline void Process()
    {
        // 单线程执行：dim3{1}，threadIdx.x == 0
        asc_vf_call<Csrgeam2PrefixSumSimtCompute>(
            dim3{1u},
            nnzPerRow_, rowPtrC_, nnzCDev_,
            m_, baseC_);
    }

private:
    __gm__ const int32_t *nnzPerRow_{nullptr};
    __gm__ int32_t *rowPtrC_{nullptr};
    __gm__ int32_t *nnzCDev_{nullptr};
    int32_t m_{0};
    int32_t baseC_{0};
};

// __global__ 调度器（Kernel 3: Prefix Sum，单 block）
extern "C" __global__ __aicore__ void csrgeam2_prefixsum_kernel(
    GM_ADDR gmNnzPerRow, GM_ADDR gmRowPtrC, GM_ADDR gmNnzCDev,
    const Csrgeam2PrefixSumTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    Csrgeam2PrefixSumDispatcher dispatcher;
    dispatcher.Init(gmNnzPerRow, gmRowPtrC, gmNnzCDev, &tiling);
    dispatcher.Process();
}

// kernel_do 启动器（Kernel 3: Prefix Sum，单 block）
extern "C" void csrgeam2_prefixsum_kernel_do(
    GM_ADDR buffer, GM_ADDR rowPtrC, GM_ADDR nnzCDev,
    const Csrgeam2PrefixSumTilingData &tiling,
    void *stream)
{
    csrgeam2_prefixsum_kernel<<<1, nullptr, stream>>>(
        buffer, rowPtrC, nnzCDev, tiling);
}
