/**
 * ----------------------------------------------------------------------------------------------------------
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ----------------------------------------------------------------------------------------------------------
 */

/*!
 * \file cscsort_kernel.cpp
 * \brief aclsparseXcscsort SIMD/SIMT mixed kernel (arch35 / ascend950).
 *
 * Short columns are sorted in UB with AscendC Sort. Long columns are sorted
 * directly in GM by a bottom-up SIMT merge-path implementation.
 * Each SIMT thread owns one output diagonal and moves rowInd/P as a bound pair.
 *
 * 以累计 nnz 为权重把完整列区间分配给各核；列边界通过 cscColPtr 二分定位，
 * 不拆分单列。各核只访问自己列区间对应的 rowInd/P/workspace，无需核间同步。
 * 每列 segment [begin, end) 内对 cscRowInd 做单键稳定升序排序，P 同步重排。
 */

#include <cstdint>
#include "kernel_operator.h"
#include "cscsort_kernel.h"
#include "simt_api/asc_simt.h"
#include "simt_api/device_sync_functions.h"

using namespace AscendC;

namespace {
constexpr uint32_t kOneBlkSize = 32U;

template <typename T>
__aicore__ inline uint32_t AlignTo32B(uint32_t count)
{
    constexpr uint32_t elemsPerBlock = kOneBlkSize / sizeof(T);
    return (count + elemsPerBlock - 1U) & ~(elemsPerBlock - 1U);
}
}  // namespace

__simt_callee__ inline uint32_t CscsortMergePartition(__gm__ const int32_t *a,
                                                      uint32_t aLen,
                                                      __gm__ const int32_t *b,
                                                      uint32_t bLen,
                                                      uint32_t diagonal)
{
    uint32_t low = (diagonal > bLen) ? diagonal - bLen : 0U;
    uint32_t high = (diagonal < aLen) ? diagonal : aLen;
    while (low <= high) {
        uint32_t aTake = low + (high - low) / 2U;
        uint32_t bTake = diagonal - aTake;
        if (aTake > 0U && bTake < bLen && a[aTake - 1U] > b[bTake]) {
            high = aTake - 1U;
        } else if (bTake > 0U && aTake < aLen && a[aTake] <= b[bTake - 1U]) {
            low = aTake + 1U;
        } else {
            return aTake;
        }
    }
    return low;
}

__simt_callee__ inline void
CscsortMergeRound(__gm__ const int32_t *srcRowInd, __gm__ const int32_t *srcP,
                  __gm__ int32_t *dstRowInd, __gm__ int32_t *dstP, uint32_t begin,
                  uint32_t len, uint32_t width)
{
    uint32_t pairSpan = width * 2U;
    for (uint32_t pairBase = 0U; pairBase < len; pairBase += pairSpan) {
        uint32_t aLen = (pairBase + width < len) ? width : len - pairBase;
        uint32_t bBase = pairBase + aLen;
        uint32_t bLen =
            (bBase < len) ? ((bBase + width < len) ? width : len - bBase) : 0U;
        __gm__ const int32_t *a = srcRowInd + begin + pairBase;
        __gm__ const int32_t *b = srcRowInd + begin + bBase;
        uint32_t pairLen = aLen + bLen;
        for (uint32_t diagonal = threadIdx.x; diagonal < pairLen;
             diagonal += blockDim.x) {
            uint32_t aTake = CscsortMergePartition(a, aLen, b, bLen, diagonal);
            uint32_t bTake = diagonal - aTake;
            bool takeA = bTake >= bLen || (aTake < aLen && a[aTake] <= b[bTake]);
            uint32_t source = begin + (takeA ? pairBase + aTake : bBase + bTake);
            dstRowInd[begin + pairBase + diagonal] = srcRowInd[source];
            dstP[begin + pairBase + diagonal] = srcP[source];
        }
    }
    asc_syncthreads();
}

__simt_callee__ inline void CscsortCopyCol(__gm__ const int32_t *srcRowInd,
                                           __gm__ const int32_t *srcP,
                                           __gm__ int32_t *dstRowInd,
                                           __gm__ int32_t *dstP, uint32_t begin,
                                           uint32_t len)
{
    for (uint32_t offset = threadIdx.x; offset < len; offset += blockDim.x) {
        dstRowInd[begin + offset] = srcRowInd[begin + offset];
        dstP[begin + offset] = srcP[begin + offset];
    }
    asc_syncthreads();
}

__simt_vf__ __aicore__
__launch_bounds__(kCscsortSimtMaxThreads) inline void CscsortMergeSimt(
    __gm__ const int32_t *colPtr, __gm__ int32_t *rowInd, __gm__ int32_t *p,
    __gm__ int32_t *scratchRowInd, __gm__ int32_t *scratchP, uint32_t colBegin,
    uint32_t colEnd, uint32_t indexBase, uint32_t runSize)
{
    for (uint32_t col = colBegin; col < colEnd; col++) {
        uint32_t begin =
            static_cast<uint32_t>(colPtr[col] - static_cast<int32_t>(indexBase));
        uint32_t end = static_cast<uint32_t>(colPtr[col + 1U] -
                                             static_cast<int32_t>(indexBase));
        uint32_t len = end - begin;
        if (len <= runSize) {
            continue;
        }
        __gm__ int32_t *srcRowInd = rowInd;
        __gm__ int32_t *srcP = p;
        __gm__ int32_t *dstRowInd = scratchRowInd;
        __gm__ int32_t *dstP = scratchP;
        bool resultInScratch = false;
        for (uint32_t width = 1U; width < len; width *= 2U) {
            CscsortMergeRound(srcRowInd, srcP, dstRowInd, dstP, begin, len, width);
            __gm__ int32_t *swapRowInd = srcRowInd;
            __gm__ int32_t *swapP = srcP;
            srcRowInd = dstRowInd;
            srcP = dstP;
            dstRowInd = swapRowInd;
            dstP = swapP;
            resultInScratch = !resultInScratch;
        }
        if (resultInScratch) {
            CscsortCopyCol(scratchRowInd, scratchP, rowInd, p, begin, len);
        }
    }
}

class CscsortKernel {
public:
    __aicore__ inline CscsortKernel() {}

    __aicore__ inline void Init(GM_ADDR gmColPtr, GM_ADDR gmRowInd, GM_ADDR gmP, GM_ADDR gmWs,
                                const CscsortTilingData &td, TPipe *pipe)
    {
        n_ = td.n;
        nnz_ = td.nnz;
        indexBase_ = td.indexBase;
        runSize_ = td.runSize;
        sortTmpBytes_ = td.sortTmpBytes;
        simtThreads_ = td.simtThreads;
        coreNum_ = td.coreNum;
        id_ = GetBlockIdx();
        pipe_ = pipe;

        colPtr_ = reinterpret_cast<__gm__ int32_t *>(gmColPtr);
        rowInd_ = reinterpret_cast<__gm__ int32_t *>(gmRowInd);
        p_ = reinterpret_cast<__gm__ int32_t *>(gmP);
        scratchRowInd_ = reinterpret_cast<__gm__ int32_t *>(gmWs);
        scratchP_ = scratchRowInd_ + td.nnz;
        colPtrGM_.SetGlobalBuffer(colPtr_);
        rowIndGM_.SetGlobalBuffer(rowInd_);
        pGM_.SetGlobalBuffer(p_);
    }

    __aicore__ inline void Process()
    {
        if (coreNum_ == 0U || id_ >= coreNum_) {
            return;
        }
        uint64_t elemBegin = static_cast<uint64_t>(nnz_) * id_ / coreNum_;
        uint64_t elemEnd = static_cast<uint64_t>(nnz_) * (id_ + 1U) / coreNum_;
        uint32_t colBegin = FindColBoundary(elemBegin);
        uint32_t colEnd = FindColBoundary(elemEnd);
        SortShortCols(colBegin, colEnd);
        auto mte3v =
            static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
        SetFlag<HardEvent::MTE3_V>(mte3v);
        WaitFlag<HardEvent::MTE3_V>(mte3v);
        asc_vf_call<CscsortMergeSimt>(dim3{simtThreads_}, colPtr_, rowInd_, p_,
                                      scratchRowInd_, scratchP_, colBegin, colEnd,
                                      indexBase_, runSize_);
    }

private:
    __aicore__ inline uint32_t FindColBoundary(uint64_t elementOffset)
    {
        if (elementOffset == 0U) {
            return 0U;
        }
        if (elementOffset >= nnz_) {
            return n_;
        }
        int64_t target = static_cast<int64_t>(elementOffset) + static_cast<int64_t>(indexBase_);
        uint32_t low = 0U;
        uint32_t high = n_;
        while (low < high) {
            uint32_t mid = low + (high - low) / 2U;
            if (static_cast<int64_t>(colPtrGM_.GetValue(mid)) < target) {
                low = mid + 1U;
            } else {
                high = mid;
            }
        }
        return low;
    }

    __aicore__ inline void SortShortCols(uint32_t colBegin, uint32_t colEnd)
    {
        for (uint32_t col = colBegin; col < colEnd; col++) {
            int32_t begin =
                colPtrGM_.GetValue(col) - static_cast<int32_t>(indexBase_);
            int32_t end =
                colPtrGM_.GetValue(col + 1U) - static_cast<int32_t>(indexBase_);
            uint32_t len = static_cast<uint32_t>(end - begin);
            if (len <= 1U) {
                continue;
            }
            if (len <= runSize_) {
                SortSingleRun(static_cast<int64_t>(begin), len);
            }
        }
    }

    __aicore__ inline void SortSingleRun(int64_t begin, uint32_t len)
    {
        pipe_->Reset();
        InitSortBuffers(len);
        RunSort(rowIndGM_, pGM_, rowIndGM_, pGM_, begin, len);
    }

    __aicore__ inline void InitSortBuffers(uint32_t len)
    {
        uint32_t bufferLen = AlignTo32B<int32_t>(len);
        pipe_->InitBuffer(rowIndInBuf_, bufferLen * sizeof(int32_t));
        pipe_->InitBuffer(pInBuf_, bufferLen * sizeof(int32_t));
        pipe_->InitBuffer(dstKeyBuf_, bufferLen * sizeof(int32_t));
        pipe_->InitBuffer(dstIdxBuf_, bufferLen * sizeof(uint32_t));
        pipe_->InitBuffer(outPBuf_, bufferLen * sizeof(int32_t));
        pipe_->InitBuffer(sharedTmpBuf_, sortTmpBytes_);
    }

    __aicore__ inline void RunSort(GlobalTensor<int32_t> &srcRowInd,
                                   GlobalTensor<int32_t> &srcP,
                                   GlobalTensor<int32_t> &dstRowInd,
                                   GlobalTensor<int32_t> &dstP, int64_t off,
                                   uint32_t len)
    {
        DataCopyExtParams cp{1, static_cast<uint32_t>(len * sizeof(int32_t)), 0, 0, 0};
        DataCopyPadExtParams<int32_t> pad{false, 0, 0, 0};

        LocalTensor<int32_t> rowIndIn = rowIndInBuf_.Get<int32_t>();
        LocalTensor<int32_t> pIn = pInBuf_.Get<int32_t>();
        DataCopyPad(rowIndIn, srcRowInd[off], cp, pad);
        DataCopyPad(pIn, srcP[off], cp, pad);
        auto mte2v = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(mte2v);
        WaitFlag<HardEvent::MTE2_V>(mte2v);

        LocalTensor<int32_t> dstKey = dstKeyBuf_.Get<int32_t>();
        LocalTensor<uint32_t> dstIdx = dstIdxBuf_.Get<uint32_t>();
        LocalTensor<uint8_t> sharedTmp = sharedTmpBuf_.Get<uint8_t>();
        static constexpr SortConfig sc{SortType::RADIX_SORT, false};
        Sort<int32_t, false, sc>(dstKey, dstIdx, rowIndIn, sharedTmp, len);
        ShiftLeft<uint32_t>(dstIdx, dstIdx, 2U, static_cast<int32_t>(len));
        LocalTensor<int32_t> outP = outPBuf_.Get<int32_t>();
        Gather<int32_t>(outP, pIn, dstIdx, 0U, len);

        auto vMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(vMte3);
        WaitFlag<HardEvent::V_MTE3>(vMte3);
        DataCopyPad(dstRowInd[off], dstKey, cp);
        DataCopyPad(dstP[off], outP, cp);
        auto mte3Mte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
        SetFlag<HardEvent::MTE3_MTE2>(mte3Mte2);
        WaitFlag<HardEvent::MTE3_MTE2>(mte3Mte2);
    }

private:
    TPipe *pipe_{nullptr};
    uint32_t id_{0U};
    uint32_t coreNum_{1U};
    uint32_t n_{0U};
    uint32_t nnz_{0U};
    uint32_t indexBase_{0U};
    uint32_t runSize_{0U};
    uint32_t sortTmpBytes_{0U};
    uint32_t simtThreads_{32U};
    __gm__ int32_t *colPtr_{nullptr};
    __gm__ int32_t *rowInd_{nullptr};
    __gm__ int32_t *p_{nullptr};
    __gm__ int32_t *scratchRowInd_{nullptr};
    __gm__ int32_t *scratchP_{nullptr};
    GlobalTensor<int32_t> colPtrGM_;
    GlobalTensor<int32_t> rowIndGM_;
    GlobalTensor<int32_t> pGM_;
    TBuf<TPosition::VECCALC> rowIndInBuf_;
    TBuf<TPosition::VECCALC> pInBuf_;
    TBuf<TPosition::VECCALC> dstKeyBuf_;
    TBuf<TPosition::VECCALC> dstIdxBuf_;
    TBuf<TPosition::VECCALC> outPBuf_;
    TBuf<TPosition::VECCALC> sharedTmpBuf_;
};

extern "C" __global__ __aicore__ void cscsort_kernel(
    GM_ADDR cscColPtr, GM_ADDR cscRowInd, GM_ADDR P, GM_ADDR workspace, const CscsortTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    CscsortKernel op;
    TPipe pipe;
    op.Init(cscColPtr, cscRowInd, P, workspace, tiling, &pipe);
    op.Process();
}

void cscsort_kernel_do(
    GM_ADDR cscColPtr, GM_ADDR cscRowInd, GM_ADDR P, GM_ADDR workspace,
    const CscsortTilingData &tiling, uint32_t numBlocks, void *stream)
{
    cscsort_kernel<<<numBlocks, nullptr, stream>>>(cscColPtr, cscRowInd, P, workspace, tiling);
}
