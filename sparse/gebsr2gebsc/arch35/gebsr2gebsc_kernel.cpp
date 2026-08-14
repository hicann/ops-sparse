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
 * \file gebsr2gebsc_kernel.cpp
 * \brief gebsr2gebsc kernel 实现（SIMD 与 SIMT 混合编程）。
 *
 * 块级结构转换与 csr2csc_ex2 完全一致（将每个 block 视为标量），
 * 区别仅在于 Scatter kernel 中块值的拷贝：
 *   - Direct copy (rC==rA && cC==cA): 逐字节拷贝整个块
 *   - Block transpose (rC==cA && cC==rA): 块内元素转置
 *
 * 五个 kernel 的层次结构与 csr2csc_ex2 相同：
 *   Layer 1: __simt_vf__ 计算函数（线程级并行，grid-stride loop）
 *   Layer 2: Dispatcher 类（管理 GM 指针 + asc_vf_call 分发）
 *   Layer 3: kernel_do 启动器（<<<>>> 语法，Host 调用入口）
 */

#include "kernel_operator.h"
#include "simt_api/asc_simt.h"
#include "simt_api/device_atomic_functions.h"
#include "gebsr2gebsc_kernel.h"

namespace {

__aicore__ inline uint32_t ComputeSimtThreadNum(int32_t work)
{
    uint32_t simtThreadNum = kGebsr2GebscThreadsPerBlock;
    if (static_cast<uint32_t>(work) < simtThreadNum) {
        simtThreadNum = static_cast<uint32_t>(work);
    }
    if (simtThreadNum == 0) {
        simtThreadNum = 1;
    }
    simtThreadNum = (simtThreadNum + kGebsr2GebscWarpSize - 1u) &
        ~(kGebsr2GebscWarpSize - 1u);
    return simtThreadNum;
}

}  // namespace

// ---------------------------------------------------------------------------
// Kernel 1: CountCols — 遍历 bsrColIndA，原子加统计每 stripe 列直方图
// ---------------------------------------------------------------------------

__simt_vf__ __aicore__ __launch_bounds__(kGebsr2GebscThreadsPerBlock) inline void
Gebsr2GebscCountColsSimtCompute(
    __gm__ const int32_t *bsrColIndA,
    __gm__ int32_t *stripeHist,
    int32_t nnzb, int32_t nb, int32_t idxBase,
    int32_t stripeCount, int32_t stripeSize,
    int32_t numBlocks)
{
    int32_t globalTid = static_cast<int32_t>(threadIdx.x) +
        static_cast<int32_t>(blockIdx.x) * static_cast<int32_t>(blockDim.x);
    int32_t gridStride = static_cast<int32_t>(blockDim.x) * numBlocks;
    int32_t stripeStride = nb + 1;

    int32_t stripe = globalTid / stripeSize;
    int64_t nextStripeBegin = static_cast<int64_t>(stripe + 1) * stripeSize;

    for (int32_t idx = globalTid; idx < nnzb; idx += gridStride) {
        int32_t col = bsrColIndA[idx] - idxBase;
        while (static_cast<int64_t>(idx) >= nextStripeBegin) {
            stripe++;
            nextStripeBegin += stripeSize;
        }
        if (stripe >= stripeCount) {
            stripe = stripeCount - 1;
        }
        asc_atomic_add(&stripeHist[static_cast<int64_t>(stripe) * stripeStride + col], 1);
    }
}

class Gebsr2GebscCountColsDispatcher {
public:
    __aicore__ inline void Init(
        GM_ADDR gmColInd, GM_ADDR gmStripeHist,
        int32_t nnzb, int32_t nb, int32_t idxBase,
        int32_t stripeCount, int32_t stripeSize, int32_t numBlocks)
    {
        bsrColIndA_ = (__gm__ const int32_t *)gmColInd;
        stripeHist_ = (__gm__ int32_t *)gmStripeHist;
        nnzb_ = nnzb;
        nb_ = nb;
        idxBase_ = idxBase;
        stripeCount_ = stripeCount;
        stripeSize_ = stripeSize;
        numBlocks_ = numBlocks;
    }

    __aicore__ inline void Process()
    {
        uint32_t simtThreadNum = ComputeSimtThreadNum(nnzb_);
        asc_vf_call<Gebsr2GebscCountColsSimtCompute>(dim3{simtThreadNum},
            bsrColIndA_, stripeHist_, nnzb_, nb_, idxBase_,
            stripeCount_, stripeSize_, numBlocks_);
    }

private:
    __gm__ const int32_t *bsrColIndA_{nullptr};
    __gm__ int32_t *stripeHist_{nullptr};
    int32_t nnzb_{0};
    int32_t nb_{0};
    int32_t idxBase_{0};
    int32_t stripeCount_{0};
    int32_t stripeSize_{0};
    int32_t numBlocks_{0};
};

extern "C" __global__ __aicore__ void gebsr2gebsc_count_kernel(
    GM_ADDR bsrColIndA, GM_ADDR stripeHist,
    int32_t nnzb, int32_t nb, int32_t idxBase,
    int32_t stripeCount, int32_t stripeSize, int32_t numBlocks)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    Gebsr2GebscCountColsDispatcher launcher;
    launcher.Init(bsrColIndA, stripeHist, nnzb, nb, idxBase,
                  stripeCount, stripeSize, numBlocks);
    launcher.Process();
}

extern "C" void gebsr2gebsc_count_kernel_do(
    GM_ADDR bsrColIndA,
    GM_ADDR stripeHist,
    const Gebsr2GebscCountTilingData &tiling,
    uint32_t numBlocks,
    void *stream)
{
    gebsr2gebsc_count_kernel<<<numBlocks, 0, stream>>>(
        bsrColIndA, stripeHist,
        tiling.nnzb, tiling.nb, tiling.idxBase,
        tiling.stripeCount, tiling.stripeSize,
        static_cast<int32_t>(numBlocks));
}

// ---------------------------------------------------------------------------
// Kernel 1.5: SumStripeHist — 对 stripeHist 按列求和重建 colCount
// ---------------------------------------------------------------------------

__simt_vf__ __aicore__ __launch_bounds__(kGebsr2GebscThreadsPerBlock) inline void
Gebsr2GebscSumStripeHistSimtCompute(
    __gm__ const int32_t *stripeHist,
    __gm__ int32_t *colCount,
    int32_t nb, int32_t stripeCount, int32_t numBlocks)
{
    int32_t globalTid = static_cast<int32_t>(threadIdx.x) +
        static_cast<int32_t>(blockIdx.x) * static_cast<int32_t>(blockDim.x);
    int32_t gridStride = static_cast<int32_t>(blockDim.x) * numBlocks;
    int32_t stripeStride = nb + 1;

    for (int32_t j = globalTid; j < nb; j += gridStride) {
        int32_t sum = 0;
        for (int32_t t = 0; t < stripeCount; t++) {
            sum += stripeHist[static_cast<int64_t>(t) * stripeStride + j];
        }
        colCount[j] = sum;
    }
}

class Gebsr2GebscSumStripeHistDispatcher {
public:
    __aicore__ inline void Init(
        GM_ADDR gmStripeHist, GM_ADDR gmWorkspace,
        int32_t nb, int32_t stripeCount, int32_t numBlocks)
    {
        stripeHist_ = (__gm__ const int32_t *)gmStripeHist;
        colCount_ = (__gm__ int32_t *)gmWorkspace;
        nb_ = nb;
        stripeCount_ = stripeCount;
        numBlocks_ = numBlocks;
    }

    __aicore__ inline void Process()
    {
        uint32_t simtThreadNum = ComputeSimtThreadNum(nb_);
        asc_vf_call<Gebsr2GebscSumStripeHistSimtCompute>(dim3{simtThreadNum},
            stripeHist_, colCount_, nb_, stripeCount_, numBlocks_);
    }

private:
    __gm__ const int32_t *stripeHist_{nullptr};
    __gm__ int32_t *colCount_{nullptr};
    int32_t nb_{0};
    int32_t stripeCount_{0};
    int32_t numBlocks_{0};
};

extern "C" __global__ __aicore__ void gebsr2gebsc_sum_stripe_hist_kernel(
    GM_ADDR stripeHist, GM_ADDR workspace,
    int32_t nb, int32_t stripeCount, int32_t numBlocks)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    Gebsr2GebscSumStripeHistDispatcher launcher;
    launcher.Init(stripeHist, workspace, nb, stripeCount, numBlocks);
    launcher.Process();
}

extern "C" void gebsr2gebsc_sum_stripe_hist_kernel_do(
    GM_ADDR stripeHist,
    GM_ADDR workspace,
    const Gebsr2GebscSumStripeHistTilingData &tiling,
    uint32_t numBlocks,
    void *stream)
{
    gebsr2gebsc_sum_stripe_hist_kernel<<<numBlocks, 0, stream>>>(
        stripeHist, workspace,
        tiling.nb, tiling.stripeCount,
        static_cast<int32_t>(numBlocks));
}

// ---------------------------------------------------------------------------
// Kernel 2: PrefixSum — exclusive prefix sum -> bscColPtr（单 warp 并行 scan）
// ---------------------------------------------------------------------------

__simt_vf__ __aicore__ __launch_bounds__(kGebsr2GebscWarpSize) inline void
Gebsr2GebscPrefixSumSimtCompute(
    __gm__ const int32_t *colCount,
    __gm__ int32_t *bscColPtr,
    int32_t nb, int32_t idxBase)
{
    constexpr int32_t kWarpScanThreads = static_cast<int32_t>(kGebsr2GebscWarpSize);
    int32_t laneId = static_cast<int32_t>(threadIdx.x);

    int32_t chunkSize = (nb + kWarpScanThreads - 1) / kWarpScanThreads;
    int32_t chunkStart = laneId * chunkSize;
    int32_t chunkEnd = chunkStart + chunkSize;
    if (chunkEnd > nb) {
        chunkEnd = nb;
    }

    int32_t localSum = 0;
    for (int32_t j = chunkStart; j < chunkEnd; j++) {
        localSum += colCount[j];
    }

    int32_t v = localSum;
    for (int32_t offset = 1; offset < kWarpScanThreads; offset *= 2) {
        int32_t up = asc_shfl_up(v, offset);
        if (laneId >= offset) {
            v = up + v;
        }
    }
    int32_t prefix = v - localSum;

    int32_t running = prefix + idxBase;
    for (int32_t j = chunkStart; j < chunkEnd; j++) {
        running += colCount[j];
        bscColPtr[j + 1] = running;
    }

    if (laneId == 0) {
        bscColPtr[0] = idxBase;
    }
}

class Gebsr2GebscPrefixSumDispatcher {
public:
    __aicore__ inline void Init(
        GM_ADDR gmWorkspace, GM_ADDR gmBscColPtr,
        int32_t nb, int32_t idxBase)
    {
        colCount_ = (__gm__ const int32_t *)gmWorkspace;
        bscColPtr_ = (__gm__ int32_t *)gmBscColPtr;
        nb_ = nb;
        idxBase_ = idxBase;
    }

    __aicore__ inline void Process()
    {
        asc_vf_call<Gebsr2GebscPrefixSumSimtCompute>(dim3{kGebsr2GebscWarpSize},
            colCount_, bscColPtr_, nb_, idxBase_);
    }

private:
    __gm__ const int32_t *colCount_{nullptr};
    __gm__ int32_t *bscColPtr_{nullptr};
    int32_t nb_{0};
    int32_t idxBase_{0};
};

extern "C" __global__ __aicore__ void gebsr2gebsc_prefixsum_kernel(
    GM_ADDR workspace, GM_ADDR bscColPtr,
    int32_t nb, int32_t idxBase)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    Gebsr2GebscPrefixSumDispatcher launcher;
    launcher.Init(workspace, bscColPtr, nb, idxBase);
    launcher.Process();
}

extern "C" void gebsr2gebsc_prefixsum_kernel_do(
    GM_ADDR workspace,
    GM_ADDR bscColPtr,
    const Gebsr2GebscPrefixSumTilingData &tiling,
    void *stream)
{
    gebsr2gebsc_prefixsum_kernel<<<1, 0, stream>>>(
        workspace, bscColPtr, tiling.nb, tiling.idxBase);
}

// ---------------------------------------------------------------------------
// Kernel 3: StripeBase — stripe 直方图按列前缀和 -> 每 stripe 写游标基址
// ---------------------------------------------------------------------------

__simt_vf__ __aicore__ __launch_bounds__(kGebsr2GebscThreadsPerBlock) inline void
Gebsr2GebscStripeBaseSimtCompute(
    __gm__ int32_t *stripeBase,
    __gm__ const int32_t *bscColPtr,
    int32_t nb, int32_t idxBase,
    int32_t stripeCount, int32_t numBlocks)
{
    int32_t globalTid = static_cast<int32_t>(threadIdx.x) +
        static_cast<int32_t>(blockIdx.x) * static_cast<int32_t>(blockDim.x);
    int32_t gridStride = static_cast<int32_t>(blockDim.x) * numBlocks;
    int32_t stripeStride = nb + 1;

    for (int32_t col = globalTid; col < nb; col += gridStride) {
        int32_t running = bscColPtr[col] - idxBase;
        for (int32_t t = 0; t < stripeCount; t++) {
            int64_t offset = static_cast<int64_t>(t) * stripeStride + col;
            int32_t cnt = stripeBase[offset];
            stripeBase[offset] = running;
            running += cnt;
        }
    }
}

class Gebsr2GebscStripeBaseDispatcher {
public:
    __aicore__ inline void Init(
        GM_ADDR gmStripeBase, GM_ADDR gmBscColPtr,
        int32_t nb, int32_t idxBase, int32_t stripeCount, int32_t numBlocks)
    {
        stripeBase_ = (__gm__ int32_t *)gmStripeBase;
        bscColPtr_ = (__gm__ const int32_t *)gmBscColPtr;
        nb_ = nb;
        idxBase_ = idxBase;
        stripeCount_ = stripeCount;
        numBlocks_ = numBlocks;
    }

    __aicore__ inline void Process()
    {
        uint32_t simtThreadNum = ComputeSimtThreadNum(nb_);
        asc_vf_call<Gebsr2GebscStripeBaseSimtCompute>(dim3{simtThreadNum},
            stripeBase_, bscColPtr_, nb_, idxBase_, stripeCount_, numBlocks_);
    }

private:
    __gm__ int32_t *stripeBase_{nullptr};
    __gm__ const int32_t *bscColPtr_{nullptr};
    int32_t nb_{0};
    int32_t idxBase_{0};
    int32_t stripeCount_{0};
    int32_t numBlocks_{0};
};

extern "C" __global__ __aicore__ void gebsr2gebsc_stripebase_kernel(
    GM_ADDR stripeBase, GM_ADDR bscColPtr,
    int32_t nb, int32_t idxBase, int32_t stripeCount, int32_t numBlocks)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    Gebsr2GebscStripeBaseDispatcher launcher;
    launcher.Init(stripeBase, bscColPtr, nb, idxBase, stripeCount, numBlocks);
    launcher.Process();
}

extern "C" void gebsr2gebsc_stripebase_kernel_do(
    GM_ADDR stripeBase,
    GM_ADDR bscColPtr,
    const Gebsr2GebscStripeBaseTilingData &tiling,
    uint32_t numBlocks,
    void *stream)
{
    gebsr2gebsc_stripebase_kernel<<<numBlocks, 0, stream>>>(
        stripeBase, bscColPtr,
        tiling.nb, tiling.idxBase, tiling.stripeCount,
        static_cast<int32_t>(numBlocks));
}

// ---------------------------------------------------------------------------
// Kernel 4: Scatter — 块索引结构写入 + 块值拷贝（direct copy / transpose）
// ---------------------------------------------------------------------------

__simt_callee__ __aicore__ inline int32_t
FindScatterStartRow(
    __gm__ const int32_t *bsrRowPtrA, int32_t mb, int32_t idxBase, int32_t kFirst)
{
    int32_t lo = 0;
    int32_t hi = mb - 1;
    while (lo < hi) {
        int32_t mid = lo + (hi - lo) / 2;
        if (bsrRowPtrA[mid + 1] - idxBase > kFirst) {
            hi = mid;
        } else {
            lo = mid + 1;
        }
    }
    return lo;
}

__simt_callee__ __aicore__ inline void
ScatterBlockDirect(
    __gm__ const uint8_t *bsrValA, __gm__ uint8_t *bscVal,
    int64_t srcBase, int64_t dstBase, int64_t blockBytesA)
{
    for (int64_t b = 0; b < blockBytesA; b++) {
        bscVal[dstBase + b] = bsrValA[srcBase + b];
    }
}

__simt_callee__ __aicore__ inline void
ScatterBlockTranspose(
    __gm__ const uint8_t *bsrValA, __gm__ uint8_t *bscVal,
    int64_t srcBase, int64_t dstBase,
    int32_t rowBlockDimA, int32_t colBlockDimA,
    int32_t dirA, int32_t blockElemBytes, uint32_t valSize)
{
    for (int32_t i = 0; i < rowBlockDimA; i++) {
        for (int32_t j = 0; j < colBlockDimA; j++) {
            int64_t srcOff, dstOff;
            if (dirA == kGebsr2GebscDirRow) {
                srcOff = static_cast<int64_t>(i * colBlockDimA + j) * blockElemBytes;
                dstOff = static_cast<int64_t>(j * rowBlockDimA + i) * blockElemBytes;
            } else {
                srcOff = static_cast<int64_t>(j * rowBlockDimA + i) * blockElemBytes;
                dstOff = static_cast<int64_t>(i * colBlockDimA + j) * blockElemBytes;
            }
            for (uint32_t b = 0; b < valSize; b++) {
                bscVal[dstBase + dstOff + b] = bsrValA[srcBase + srcOff + b];
            }
        }
    }
}

__simt_vf__ __aicore__ __launch_bounds__(1) inline void
Gebsr2GebscScatterSimtCompute(
    __gm__ const int32_t *bsrRowPtrA,
    __gm__ const int32_t *bsrColIndA,
    __gm__ const uint8_t *bsrValA,
    __gm__ int32_t *bscRowInd,
    __gm__ uint8_t *bscVal,
    __gm__ int32_t *stripeCursor,
    int32_t mb, int32_t nb, int32_t nnzb, int32_t idxBase,
    int32_t copyValues, int32_t stripeSize,
    int32_t rowBlockDimA, int32_t colBlockDimA,
    int32_t rowBlockDimC, int32_t colBlockDimC,
    int32_t copyMode, int32_t dirA, uint32_t valSize)
{
    (void)nb;
    int32_t stripe = static_cast<int32_t>(blockIdx.x);
    int64_t kStart = static_cast<int64_t>(stripe) * stripeSize;
    if (kStart >= nnzb) {
        return;
    }
    int64_t kEnd = kStart + stripeSize;
    if (kEnd > nnzb) {
        kEnd = nnzb;
    }

    int32_t kFirst = static_cast<int32_t>(kStart);
    int32_t row = FindScatterStartRow(bsrRowPtrA, mb, idxBase, kFirst);

    __gm__ int32_t *cursor = stripeCursor + static_cast<int64_t>(stripe) * (nb + 1);
    int32_t rowEnd = bsrRowPtrA[row + 1] - idxBase;

    int64_t blockBytesA = static_cast<int64_t>(rowBlockDimA * colBlockDimA) * valSize;
    int64_t blockBytesC = static_cast<int64_t>(rowBlockDimC * colBlockDimC) * valSize;
    int32_t blockElemBytes = static_cast<int32_t>(valSize);

    for (int32_t k = kFirst; k < static_cast<int32_t>(kEnd); k++) {
        while (row + 1 < mb && k >= rowEnd) {
            row++;
            rowEnd = bsrRowPtrA[row + 1] - idxBase;
        }
        int32_t col = bsrColIndA[k] - idxBase;
        int32_t pos = cursor[col];
        cursor[col] = pos + 1;

        bscRowInd[pos] = row + idxBase;

        if (copyValues == kGebsr2GebscNumeric) {
            int64_t srcBase = static_cast<int64_t>(k) * blockBytesA;
            int64_t dstBase = static_cast<int64_t>(pos) * blockBytesC;
            if (copyMode == kGebsr2GebscBlockDirectCopy) {
                ScatterBlockDirect(bsrValA, bscVal, srcBase, dstBase, blockBytesA);
            } else {
                ScatterBlockTranspose(bsrValA, bscVal, srcBase, dstBase,
                    rowBlockDimA, colBlockDimA, dirA, blockElemBytes, valSize);
            }
        }
    }
}

class Gebsr2GebscScatterDispatcher {
public:
    __aicore__ inline void Init(
        GM_ADDR gmRowPtr, GM_ADDR gmColInd, GM_ADDR gmBsrVal,
        GM_ADDR gmBscRowInd, GM_ADDR gmBscVal, GM_ADDR gmStripeCursor,
        int32_t mb, int32_t nb, int32_t nnzb, int32_t idxBase,
        int32_t copyValues, int32_t stripeSize,
        int32_t rowBlockDimA, int32_t colBlockDimA,
        int32_t rowBlockDimC, int32_t colBlockDimC,
        int32_t copyMode, int32_t dirA, uint32_t valSize)
    {
        bsrRowPtrA_ = (__gm__ const int32_t *)gmRowPtr;
        bsrColIndA_ = (__gm__ const int32_t *)gmColInd;
        bsrValA_ = (__gm__ const uint8_t *)gmBsrVal;
        bscRowInd_ = (__gm__ int32_t *)gmBscRowInd;
        bscVal_ = (__gm__ uint8_t *)gmBscVal;
        stripeCursor_ = (__gm__ int32_t *)gmStripeCursor;
        mb_ = mb;
        nb_ = nb;
        nnzb_ = nnzb;
        idxBase_ = idxBase;
        copyValues_ = copyValues;
        stripeSize_ = stripeSize;
        rowBlockDimA_ = rowBlockDimA;
        colBlockDimA_ = colBlockDimA;
        rowBlockDimC_ = rowBlockDimC;
        colBlockDimC_ = colBlockDimC;
        copyMode_ = copyMode;
        dirA_ = dirA;
        valSize_ = valSize;
    }

    __aicore__ inline void Process()
    {
        asc_vf_call<Gebsr2GebscScatterSimtCompute>(dim3{1},
            bsrRowPtrA_, bsrColIndA_, bsrValA_,
            bscRowInd_, bscVal_, stripeCursor_,
            mb_, nb_, nnzb_, idxBase_, copyValues_, stripeSize_,
            rowBlockDimA_, colBlockDimA_, rowBlockDimC_, colBlockDimC_,
            copyMode_, dirA_, valSize_);
    }

private:
    __gm__ const int32_t *bsrRowPtrA_{nullptr};
    __gm__ const int32_t *bsrColIndA_{nullptr};
    __gm__ const uint8_t *bsrValA_{nullptr};
    __gm__ int32_t *bscRowInd_{nullptr};
    __gm__ uint8_t *bscVal_{nullptr};
    __gm__ int32_t *stripeCursor_{nullptr};
    int32_t mb_{0};
    int32_t nb_{0};
    int32_t nnzb_{0};
    int32_t idxBase_{0};
    int32_t copyValues_{0};
    int32_t stripeSize_{0};
    int32_t rowBlockDimA_{0};
    int32_t colBlockDimA_{0};
    int32_t rowBlockDimC_{0};
    int32_t colBlockDimC_{0};
    int32_t copyMode_{0};
    int32_t dirA_{0};
    uint32_t valSize_{0};
};

extern "C" __global__ __aicore__ void gebsr2gebsc_scatter_kernel(
    GM_ADDR bsrRowPtrA, GM_ADDR bsrColIndA, GM_ADDR bsrValA,
    GM_ADDR bscRowInd, GM_ADDR bscVal, GM_ADDR stripeCursor,
    int32_t mb, int32_t nb, int32_t nnzb, int32_t idxBase,
    int32_t copyValues, int32_t stripeSize, uint32_t valSize,
    int32_t rowBlockDimA, int32_t colBlockDimA,
    int32_t rowBlockDimC, int32_t colBlockDimC,
    int32_t copyMode, int32_t dirA)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    Gebsr2GebscScatterDispatcher launcher;
    launcher.Init(bsrRowPtrA, bsrColIndA, bsrValA,
                  bscRowInd, bscVal, stripeCursor,
                  mb, nb, nnzb, idxBase, copyValues, stripeSize,
                  rowBlockDimA, colBlockDimA, rowBlockDimC, colBlockDimC,
                  copyMode, dirA, valSize);
    launcher.Process();
}

extern "C" void gebsr2gebsc_scatter_kernel_do(
    GM_ADDR bsrRowPtrA,
    GM_ADDR bsrColIndA,
    GM_ADDR bsrValA,
    GM_ADDR bscRowInd,
    GM_ADDR bscVal,
    GM_ADDR stripeCursor,
    const Gebsr2GebscScatterTilingData &tiling,
    uint32_t numBlocks,
    void *stream)
{
    gebsr2gebsc_scatter_kernel<<<numBlocks, 0, stream>>>(
        bsrRowPtrA, bsrColIndA, bsrValA,
        bscRowInd, bscVal, stripeCursor,
        tiling.mb, tiling.nb, tiling.nnzb, tiling.idxBase,
        tiling.copyValues, tiling.stripeSize, tiling.valSize,
        tiling.rowBlockDimA, tiling.colBlockDimA,
        tiling.rowBlockDimC, tiling.colBlockDimC,
        tiling.copyMode, tiling.dirA);
}
