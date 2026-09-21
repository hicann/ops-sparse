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

#include <cstdint>
#include "kernel_operator.h"
#include "simt_api/asc_simt.h"
#include "simt_api/common_functions.h"
#include "simt_api/device_warp_functions.h"
#include "spmv.h"

namespace {

template <typename A, typename B>
struct SpmvIsSame {
    static constexpr bool value = false;
};

template <typename A>
struct SpmvIsSame<A, A> {
    static constexpr bool value = true;
};

__aicore__ inline uint32_t AlignThreadCount(uint32_t count)
{
    if (count == 0u) {
        count = 1u;
    }
    count = (count + SPMV_SIMT_WARP_SIZE - 1u) & ~(SPMV_SIMT_WARP_SIZE - 1u);
    return count > SPMV_MAX_SIMT_THREADS ? SPMV_MAX_SIMT_THREADS : count;
}

template <typename AccT>
__aicore__ inline AccT LoadAlpha(const SpmvTilingData& tiling)
{
    if constexpr (SpmvIsSame<AccT, int32_t>::value) {
        if (tiling.alphaDevicePtr != 0u) {
            return *reinterpret_cast<__gm__ const int32_t*>(tiling.alphaDevicePtr);
        }
        return tiling.alphaInt;
    } else {
        if (tiling.alphaDevicePtr != 0u) {
            return *reinterpret_cast<__gm__ const float*>(tiling.alphaDevicePtr);
        }
        return tiling.alphaFloat;
    }
}

template <typename AccT>
__aicore__ inline AccT LoadBeta(const SpmvTilingData& tiling)
{
    if constexpr (SpmvIsSame<AccT, int32_t>::value) {
        if (tiling.betaDevicePtr != 0u) {
            return *reinterpret_cast<__gm__ const int32_t*>(tiling.betaDevicePtr);
        }
        return tiling.betaInt;
    } else {
        if (tiling.betaDevicePtr != 0u) {
            return *reinterpret_cast<__gm__ const float*>(tiling.betaDevicePtr);
        }
        return tiling.betaFloat;
    }
}

__aicore__ __simt_callee__ inline int32_t ClampInt32(int64_t value)
{
    constexpr int64_t kInt32Max = 2147483647LL;
    constexpr int64_t kInt32Min = -2147483648LL;
    if (value > kInt32Max) {
        return static_cast<int32_t>(kInt32Max);
    }
    if (value < kInt32Min) {
        return static_cast<int32_t>(kInt32Min);
    }
    return static_cast<int32_t>(value);
}

__aicore__ __simt_callee__ inline int32_t SpmvInt32Result(int64_t sum, int32_t alpha, int32_t beta, int32_t oldY)
{
    constexpr int64_t kInt32Max = 2147483647LL;
    constexpr int64_t kInt32Min = -2147483648LL;
    constexpr int64_t kInt64Max = 9223372036854775807LL;
    const int64_t betaTerm = static_cast<int64_t>(beta) * oldY;
    if (alpha == 0) {
        return ClampInt32(betaTerm);
    }
    // INT8 dot products contain at most INT32_MAX entries, so |sum| < 2^45.
    // Widen alpha before negating to handle INT32_MIN without overflow.
    const int64_t magnitude = sum < 0 ? -sum : sum;
    const int64_t scale = alpha < 0 ? -static_cast<int64_t>(alpha) : static_cast<int64_t>(alpha);
    const bool negative = (sum < 0) != (alpha < 0);
    if (magnitude > kInt64Max / scale) {
        // |beta * oldY| <= 2^62 cannot cancel an overflowing product back
        // into INT32 range. This also covers a negative product of -2^63.
        return static_cast<int32_t>(negative ? kInt32Min : kInt32Max);
    }
    const int64_t product = negative ? -(magnitude * scale) : magnitude * scale;
    // Shift the final saturation bounds instead of adding two INT64 terms.
    // These bounds fit INT64; cancellation is preserved before saturation.
    if (product > kInt32Max - betaTerm) {
        return static_cast<int32_t>(kInt32Max);
    }
    if (product < kInt32Min - betaTerm) {
        return static_cast<int32_t>(kInt32Min);
    }
    return static_cast<int32_t>(product + betaTerm);
}

template <bool Strided>
__aicore__ __simt_callee__ inline int64_t PhysicalIndex(int32_t logicalIndex, int64_t stride)
{
    if constexpr (Strided) {
        return static_cast<int64_t>(logicalIndex) * stride;
    }
    return static_cast<int64_t>(logicalIndex);
}

template <typename ValT, bool Strided>
__aicore__ __simt_callee__ inline float SpmvFloatDotCsr(
    __gm__ const int32_t* colInd, __gm__ const ValT* values, __gm__ const ValT* xVec, int32_t begin, int32_t end,
    int64_t colIndStride, int64_t valuesStride, int64_t xStride)
{
    float sum = 0.0f;
    for (int32_t p = begin; p < end; ++p) {
        const int32_t xIndex = colInd[PhysicalIndex<Strided>(p, colIndStride)];
        sum += static_cast<float>(values[PhysicalIndex<Strided>(p, valuesStride)]) *
               static_cast<float>(xVec[PhysicalIndex<Strided>(xIndex, xStride)]);
    }
    return sum;
}

template <typename ValT, bool Strided>
__aicore__ __simt_callee__ inline float SpmvFloatDotCsc(
    __gm__ const int32_t* rowIndices, __gm__ const int32_t* permutation, __gm__ const ValT* values,
    __gm__ const ValT* xVec, int32_t begin, int32_t end, int64_t valuesStride, int64_t xStride)
{
    float sum = 0.0f;
    for (int32_t pos = begin; pos < end; ++pos) {
        const int32_t p = permutation[pos];
        sum += static_cast<float>(values[PhysicalIndex<Strided>(p, valuesStride)]) *
               static_cast<float>(xVec[PhysicalIndex<Strided>(rowIndices[pos], xStride)]);
    }
    return sum;
}

template <typename ValT, typename OutT, typename AccT, bool Transpose, bool Strided>
__simt_vf__ __aicore__ __launch_bounds__(SPMV_MAX_SIMT_THREADS) inline void SpmvSimtCompute(
    __gm__ const int32_t* rowOffsets, __gm__ const int32_t* colInd, __gm__ const ValT* values, __gm__ const ValT* xVec,
    __gm__ OutT* yVec, __gm__ const int32_t* colOffsets, __gm__ const int32_t* rowIndices,
    __gm__ const int32_t* permutation, int32_t outputBegin, int32_t outputEnd, int32_t threadsPerCore, AccT alpha,
    AccT beta, int32_t nnz, int64_t rowOffsetsStride, int64_t colIndStride, int64_t valuesStride, int64_t xStride,
    int64_t yStride)
{
    const int32_t range = outputEnd - outputBegin;
    for (int32_t local = static_cast<int32_t>(threadIdx.x); local < range; local += threadsPerCore) {
        const int32_t output = outputBegin + local;
        if constexpr (SpmvIsSame<AccT, int32_t>::value) {
            int64_t sum = 0;
            if (nnz > 0) {
                const int32_t begin =
                    Transpose ? colOffsets[output] : rowOffsets[PhysicalIndex<Strided>(output, rowOffsetsStride)];
                const int32_t end = Transpose ? colOffsets[output + 1] :
                                                rowOffsets[PhysicalIndex<Strided>(output + 1, rowOffsetsStride)];
                for (int32_t pos = begin; pos < end; ++pos) {
                    const int32_t p = Transpose ? permutation[pos] : pos;
                    const int32_t xIndex =
                        Transpose ? rowIndices[pos] : colInd[PhysicalIndex<Strided>(p, colIndStride)];
                    sum += static_cast<int64_t>(values[PhysicalIndex<Strided>(p, valuesStride)]) *
                           static_cast<int64_t>(xVec[PhysicalIndex<Strided>(xIndex, xStride)]);
                }
            }
            const int64_t yIndex = PhysicalIndex<Strided>(output, yStride);
            const int32_t oldY = beta == 0 ? 0 : yVec[yIndex];
            yVec[yIndex] = SpmvInt32Result(sum, alpha, beta, oldY);
        } else {
            float sum = 0.0f;
            if (nnz > 0) {
                const int32_t begin =
                    Transpose ? colOffsets[output] : rowOffsets[PhysicalIndex<Strided>(output, rowOffsetsStride)];
                const int32_t end = Transpose ? colOffsets[output + 1] :
                                                rowOffsets[PhysicalIndex<Strided>(output + 1, rowOffsetsStride)];
                sum = Transpose ? SpmvFloatDotCsc<ValT, Strided>(
                                      rowIndices, permutation, values, xVec, begin, end, valuesStride, xStride) :
                                  SpmvFloatDotCsr<ValT, Strided>(
                                      colInd, values, xVec, begin, end, colIndStride, valuesStride, xStride);
            }
            const int64_t yIndex = PhysicalIndex<Strided>(output, yStride);
            const float oldY = (beta == 0.0f) ? 0.0f : static_cast<float>(yVec[yIndex]);
            const float result = alpha * sum + beta * oldY;
            yVec[yIndex] = static_cast<OutT>(result);
        }
    }
}

template <typename OutT, bool Strided>
__aicore__ __simt_callee__ inline void SpmvStoreWarpResult(
    __gm__ OutT* yVec, int32_t output, int32_t lane, float sum, float alpha, float beta, int64_t yStride)
{
    constexpr int32_t kThreadsPerRow = 4;
    for (int32_t offset = kThreadsPerRow / 2; offset > 0; offset >>= 1) {
        sum += asc_shfl_down(sum, static_cast<uint32_t>(offset), kThreadsPerRow);
    }
    if (lane == 0) {
        const int64_t yIndex = static_cast<int64_t>(output) * (Strided ? yStride : 1);
        const float oldY = (beta == 0.0f) ? 0.0f : static_cast<float>(yVec[yIndex]);
        yVec[yIndex] = static_cast<OutT>(alpha * sum + beta * oldY);
    }
}

template <typename ValT, typename OutT, bool Transpose, bool Strided>
__simt_vf__ __aicore__ __launch_bounds__(SPMV_MAX_SIMT_THREADS) inline void SpmvSimtComputeWarp(
    __gm__ const int32_t* rowOffsets, __gm__ const int32_t* colInd, __gm__ const ValT* values, __gm__ const ValT* xVec,
    __gm__ OutT* yVec, __gm__ const int32_t* colOffsets, __gm__ const int32_t* rowIndices,
    __gm__ const int32_t* permutation, int32_t outputBegin, int32_t outputEnd, int32_t threadsPerCore, float alpha,
    float beta, int32_t nnz, int64_t rowOffsetsStride, int64_t colIndStride, int64_t valuesStride, int64_t xStride,
    int64_t yStride)
{
    constexpr int32_t kThreadsPerRow = 4;
    const int32_t lane = static_cast<int32_t>(threadIdx.x) & (kThreadsPerRow - 1);
    const int32_t group = static_cast<int32_t>(threadIdx.x) / kThreadsPerRow;
    const auto* xIndices = Transpose ? rowIndices : colInd;
    const int64_t indexStep = Transpose ? 1 : (Strided ? colIndStride : 1);
    const int64_t valueStep = Strided ? valuesStride : 1;
    const int64_t xStep = Strided ? xStride : 1;
    const int64_t rowStep = Strided ? rowOffsetsStride : 1;
    for (int32_t local = group; local < outputEnd - outputBegin; local += threadsPerCore / kThreadsPerRow) {
        const int32_t output = outputBegin + local;
        const int32_t begin = nnz == 0 ? 0 : (Transpose ? colOffsets[output] : rowOffsets[output * rowStep]);
        const int32_t end = nnz == 0 ? 0 : (Transpose ? colOffsets[output + 1] : rowOffsets[(output + 1) * rowStep]);
        // Constant accumulator indices retain four independent FMA chains.
        float sums[4]{};
        int32_t pos = begin + lane;
        for (; pos + 3 * kThreadsPerRow < end; pos += 4 * kThreadsPerRow) {
            const int32_t p0 = Transpose ? permutation[pos] : pos;
            const int32_t p1 = Transpose ? permutation[pos + kThreadsPerRow] : pos + kThreadsPerRow;
            const int32_t p2 = Transpose ? permutation[pos + 2 * kThreadsPerRow] : pos + 2 * kThreadsPerRow;
            const int32_t p3 = Transpose ? permutation[pos + 3 * kThreadsPerRow] : pos + 3 * kThreadsPerRow;
            const int32_t x0 = xIndices[(pos) * indexStep];
            const int32_t x1 = xIndices[(pos + kThreadsPerRow) * indexStep];
            const int32_t x2 = xIndices[(pos + 2 * kThreadsPerRow) * indexStep];
            const int32_t x3 = xIndices[(pos + 3 * kThreadsPerRow) * indexStep];
            sums[0] = __fmaf_rn(
                static_cast<float>(values[p0 * valueStep]), static_cast<float>(xVec[x0 * xStep]), sums[0]);
            sums[1] = __fmaf_rn(
                static_cast<float>(values[p1 * valueStep]), static_cast<float>(xVec[x1 * xStep]), sums[1]);
            sums[2] = __fmaf_rn(
                static_cast<float>(values[p2 * valueStep]), static_cast<float>(xVec[x2 * xStep]), sums[2]);
            sums[3] = __fmaf_rn(
                static_cast<float>(values[p3 * valueStep]), static_cast<float>(xVec[x3 * xStep]), sums[3]);
        }
        for (; pos < end; pos += kThreadsPerRow) {
            const int32_t p = Transpose ? permutation[pos] : pos;
            const int32_t xIndex = xIndices[pos * indexStep];
            sums[0] = __fmaf_rn(
                static_cast<float>(values[p * valueStep]), static_cast<float>(xVec[xIndex * xStep]), sums[0]);
        }
        SpmvStoreWarpResult<OutT, Strided>(
            yVec, output, lane, (sums[0] + sums[1]) + (sums[2] + sums[3]), alpha, beta, yStride);
    }
}

template <typename ValT>
__aicore__ __simt_callee__ inline float SpmvFloatDotCsrContiguous(
    __gm__ const int32_t* colInd, __gm__ const ValT* values, __gm__ const ValT* xVec, int32_t begin, int32_t end)
{
    float sum = 0.0f;
    for (int32_t p = begin; p < end; ++p) {
        sum += static_cast<float>(values[p]) * static_cast<float>(xVec[colInd[p]]);
    }
    return sum;
}

template <typename ValT>
__aicore__ __simt_callee__ inline float SpmvFloatDotCscContiguous(
    __gm__ const int32_t* rowIndices, __gm__ const int32_t* permutation, __gm__ const ValT* values,
    __gm__ const ValT* xVec, int32_t begin, int32_t end)
{
    float sum = 0.0f;
    for (int32_t pos = begin; pos < end; ++pos) {
        const int32_t p = permutation[pos];
        sum += static_cast<float>(values[p]) * static_cast<float>(xVec[rowIndices[pos]]);
    }
    return sum;
}

template <typename ValT, typename OutT, typename AccT, bool Transpose>
__simt_vf__ __aicore__ __launch_bounds__(SPMV_MAX_SIMT_THREADS) inline void SpmvSimtComputeContiguous(
    __gm__ const int32_t* rowOffsets, __gm__ const int32_t* colInd, __gm__ const ValT* values, __gm__ const ValT* xVec,
    __gm__ OutT* yVec, __gm__ const int32_t* colOffsets, __gm__ const int32_t* rowIndices,
    __gm__ const int32_t* permutation, int32_t outputBegin, int32_t outputEnd, int32_t threadsPerCore, AccT alpha,
    AccT beta, int32_t nnz)
{
    const int32_t range = outputEnd - outputBegin;
    for (int32_t local = static_cast<int32_t>(threadIdx.x); local < range; local += threadsPerCore) {
        const int32_t output = outputBegin + local;
        if constexpr (SpmvIsSame<AccT, int32_t>::value) {
            int64_t sum = 0;
            if (nnz > 0) {
                const int32_t begin = Transpose ? colOffsets[output] : rowOffsets[output];
                const int32_t end = Transpose ? colOffsets[output + 1] : rowOffsets[output + 1];
                for (int32_t pos = begin; pos < end; ++pos) {
                    const int32_t p = Transpose ? permutation[pos] : pos;
                    const int32_t xIndex = Transpose ? rowIndices[pos] : colInd[p];
                    sum += static_cast<int64_t>(values[p]) * static_cast<int64_t>(xVec[xIndex]);
                }
            }
            const int32_t oldY = beta == 0 ? 0 : yVec[output];
            yVec[output] = SpmvInt32Result(sum, alpha, beta, oldY);
        } else {
            float sum = 0.0f;
            if (nnz > 0) {
                const int32_t begin = Transpose ? colOffsets[output] : rowOffsets[output];
                const int32_t end = Transpose ? colOffsets[output + 1] : rowOffsets[output + 1];
                sum = Transpose ? SpmvFloatDotCscContiguous(rowIndices, permutation, values, xVec, begin, end) :
                                  SpmvFloatDotCsrContiguous(colInd, values, xVec, begin, end);
            }
            const float oldY = (beta == 0.0f) ? 0.0f : static_cast<float>(yVec[output]);
            yVec[output] = static_cast<OutT>(alpha * sum + beta * oldY);
        }
    }
}

template <typename ValT, typename OutT, typename AccT>
class SpmvDispatcher {
public:
    __aicore__ inline void Init(
        GM_ADDR rowOffsets, GM_ADDR colInd, GM_ADDR values, GM_ADDR xVec, GM_ADDR yVec, GM_ADDR colOffsets,
        GM_ADDR rowIndices, GM_ADDR permutation, const SpmvTilingData& tiling)
    {
        rowOffsets_ = reinterpret_cast<__gm__ const int32_t*>(rowOffsets);
        colInd_ = reinterpret_cast<__gm__ const int32_t*>(colInd);
        values_ = reinterpret_cast<__gm__ const ValT*>(values);
        xVec_ = reinterpret_cast<__gm__ const ValT*>(xVec);
        yVec_ = reinterpret_cast<__gm__ OutT*>(yVec);
        colOffsets_ = reinterpret_cast<__gm__ const int32_t*>(colOffsets);
        rowIndices_ = reinterpret_cast<__gm__ const int32_t*>(rowIndices);
        permutation_ = reinterpret_cast<__gm__ const int32_t*>(permutation);
        tiling_ = tiling;
    }

    __aicore__ inline void Process()
    {
        const int32_t outputSize = tiling_.transpose != 0 ? tiling_.cols : tiling_.rows;
        const uint32_t blockId = AscendC::GetBlockIdx();
        const uint64_t limit = static_cast<uint64_t>(outputSize);
        const uint64_t blockBegin = static_cast<uint64_t>(blockId) * tiling_.workPerBlock;
        const uint64_t blockEnd = blockBegin + static_cast<uint64_t>(tiling_.workPerBlock);
        const AccT alpha = LoadAlpha<AccT>(tiling_);
        const AccT beta = LoadBeta<AccT>(tiling_);
        const bool strided = tiling_.rowOffsetsStride != 1 || tiling_.colIndStride != 1 ||
                             tiling_.valuesStride != 1 || tiling_.xStride != 1 || tiling_.yStride != 1;
        // Bound VF work to improve locality without launching more than the
        // physical AIV count. Padded blockEnd keeps the number of VF calls
        // uniform across all cores, including empty tail tiles.
        constexpr uint64_t kTileRows = 512u;
        for (uint64_t tile = blockBegin; tile < blockEnd; tile += kTileRows) {
            const uint64_t tileEnd = tile + kTileRows < blockEnd ? tile + kTileRows : blockEnd;
            const int32_t begin = static_cast<int32_t>(tile < limit ? tile : limit);
            const int32_t end = static_cast<int32_t>(tileEnd < limit ? tileEnd : limit);
            const uint32_t threadCount = AlignThreadCount(static_cast<uint32_t>(end - begin));
            if (strided) {
                Dispatch<true>(begin, end, threadCount, alpha, beta);
            } else {
                Dispatch<false>(begin, end, threadCount, alpha, beta);
            }
        }
    }

private:
    template <bool Transpose, bool Strided>
    __aicore__ inline void DispatchOperation(int32_t begin, int32_t end, uint32_t threadCount, AccT alpha, AccT beta)
    {
        auto* colOffsets = Transpose ? colOffsets_ : nullptr;
        auto* rowIndices = Transpose ? rowIndices_ : nullptr;
        auto* permutation = Transpose ? permutation_ : nullptr;
        if constexpr (Strided) {
            if constexpr (SpmvIsSame<AccT, float>::value) {
                asc_vf_call<SpmvSimtComputeWarp<ValT, OutT, Transpose, true>>(
                    dim3{threadCount}, rowOffsets_, colInd_, values_, xVec_, yVec_, colOffsets, rowIndices, permutation,
                    begin, end, static_cast<int32_t>(threadCount), alpha, beta, tiling_.nnz, tiling_.rowOffsetsStride,
                    tiling_.colIndStride, tiling_.valuesStride, tiling_.xStride, tiling_.yStride);
            } else {
                asc_vf_call<SpmvSimtCompute<ValT, OutT, AccT, Transpose, true>>(
                    dim3{threadCount}, rowOffsets_, colInd_, values_, xVec_, yVec_, colOffsets, rowIndices, permutation,
                    begin, end, static_cast<int32_t>(threadCount), alpha, beta, tiling_.nnz, tiling_.rowOffsetsStride,
                    tiling_.colIndStride, tiling_.valuesStride, tiling_.xStride, tiling_.yStride);
            }
        } else {
            if constexpr (SpmvIsSame<AccT, float>::value) {
                asc_vf_call<SpmvSimtComputeWarp<ValT, OutT, Transpose, false>>(
                    dim3{threadCount}, rowOffsets_, colInd_, values_, xVec_, yVec_, colOffsets, rowIndices, permutation,
                    begin, end, static_cast<int32_t>(threadCount), alpha, beta, tiling_.nnz, int64_t{1}, int64_t{1},
                    int64_t{1}, int64_t{1}, int64_t{1});
            } else {
                asc_vf_call<SpmvSimtComputeContiguous<ValT, OutT, AccT, Transpose>>(
                    dim3{threadCount}, rowOffsets_, colInd_, values_, xVec_, yVec_, colOffsets, rowIndices, permutation,
                    begin, end, static_cast<int32_t>(threadCount), alpha, beta, tiling_.nnz);
            }
        }
    }

    template <bool Strided>
    __aicore__ inline void Dispatch(int32_t begin, int32_t end, uint32_t threadCount, AccT alpha, AccT beta)
    {
        // Every launched outer core must issue the VF call, even for an empty
        // range, to keep DAV-3510 split AIC/AIV dispatch uniform.
        if (tiling_.transpose != 0) {
            DispatchOperation<true, Strided>(begin, end, threadCount, alpha, beta);
        } else {
            DispatchOperation<false, Strided>(begin, end, threadCount, alpha, beta);
        }
    }

    __gm__ const int32_t* rowOffsets_{nullptr};
    __gm__ const int32_t* colInd_{nullptr};
    __gm__ const ValT* values_{nullptr};
    __gm__ const ValT* xVec_{nullptr};
    __gm__ OutT* yVec_{nullptr};
    __gm__ const int32_t* colOffsets_{nullptr};
    __gm__ const int32_t* rowIndices_{nullptr};
    __gm__ const int32_t* permutation_{nullptr};
    SpmvTilingData tiling_{};
};

template <typename ValT, typename OutT, typename AccT>
__aicore__ inline void RunSpmvDispatcher(
    GM_ADDR rowOffsets, GM_ADDR colInd, GM_ADDR values, GM_ADDR xVec, GM_ADDR yVec, GM_ADDR colOffsets,
    GM_ADDR rowIndices, GM_ADDR permutation, const SpmvTilingData& tiling)
{
    SpmvDispatcher<ValT, OutT, AccT> dispatcher;
    dispatcher.Init(rowOffsets, colInd, values, xVec, yVec, colOffsets, rowIndices, permutation, tiling);
    dispatcher.Process();
}

} // namespace

extern "C" __global__ __aicore__ void spmv_arch35_kernel(
    GM_ADDR rowOffsets, GM_ADDR colInd, GM_ADDR values, GM_ADDR xVec, GM_ADDR yVec, GM_ADDR colOffsets,
    GM_ADDR rowIndices, GM_ADDR permutation, const SpmvTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    switch (tiling.dataType) {
        case SPMV_DTYPE_FP32_FP32:
            RunSpmvDispatcher<float, float, float>(
                rowOffsets, colInd, values, xVec, yVec, colOffsets, rowIndices, permutation, tiling);
            break;
        case SPMV_DTYPE_INT8_INT32:
            RunSpmvDispatcher<int8_t, int32_t, int32_t>(
                rowOffsets, colInd, values, xVec, yVec, colOffsets, rowIndices, permutation, tiling);
            break;
        case SPMV_DTYPE_INT8_FP32:
            RunSpmvDispatcher<int8_t, float, float>(
                rowOffsets, colInd, values, xVec, yVec, colOffsets, rowIndices, permutation, tiling);
            break;
        case SPMV_DTYPE_FP16_FP32:
            RunSpmvDispatcher<half, float, float>(
                rowOffsets, colInd, values, xVec, yVec, colOffsets, rowIndices, permutation, tiling);
            break;
        case SPMV_DTYPE_FP16_FP16:
            RunSpmvDispatcher<half, half, float>(
                rowOffsets, colInd, values, xVec, yVec, colOffsets, rowIndices, permutation, tiling);
            break;
        case SPMV_DTYPE_BF16_FP32:
            RunSpmvDispatcher<bfloat16_t, float, float>(
                rowOffsets, colInd, values, xVec, yVec, colOffsets, rowIndices, permutation, tiling);
            break;
        case SPMV_DTYPE_BF16_BF16:
            RunSpmvDispatcher<bfloat16_t, bfloat16_t, float>(
                rowOffsets, colInd, values, xVec, yVec, colOffsets, rowIndices, permutation, tiling);
            break;
        default:
            break;
    }
}

extern "C" __global__ __aicore__ void spmv_arch35_preprocess_kernel(
    GM_ADDR rowOffsetsAddr, GM_ADDR colIndAddr, GM_ADDR colOffsetsAddr, GM_ADDR nextAddr, GM_ADDR rowIndicesAddr,
    GM_ADDR permutationAddr, int32_t rows, int32_t cols, int32_t nnz, int64_t rowOffsetsStride, int64_t colIndStride)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    auto* rowOffsets = reinterpret_cast<__gm__ const int32_t*>(rowOffsetsAddr);
    auto* colInd = reinterpret_cast<__gm__ const int32_t*>(colIndAddr);
    auto* colOffsets = reinterpret_cast<__gm__ int32_t*>(colOffsetsAddr);
    auto* next = reinterpret_cast<__gm__ int32_t*>(nextAddr);
    auto* rowIndices = reinterpret_cast<__gm__ int32_t*>(rowIndicesAddr);
    auto* permutation = reinterpret_cast<__gm__ int32_t*>(permutationAddr);

    // Stable CSR-to-CSC metadata construction. The sequential order is
    // intentional: entries in every CSC column retain increasing CSR row/entry
    // order, so transpose accumulation has a fixed order and is bitwise
    // repeatable.
    for (int64_t col = 0; col <= static_cast<int64_t>(cols); ++col) {
        colOffsets[col] = 0;
    }
    for (int32_t p = 0; p < nnz; ++p) {
        const int32_t col = colInd[static_cast<int64_t>(p) * colIndStride];
        if (col >= 0 && col < cols) {
            colOffsets[col + 1] += 1;
        }
    }
    for (int32_t col = 0; col < cols; ++col) {
        colOffsets[col + 1] += colOffsets[col];
        next[col] = colOffsets[col];
    }
    for (int32_t row = 0; row < rows; ++row) {
        const int32_t begin = rowOffsets[static_cast<int64_t>(row) * rowOffsetsStride];
        const int32_t end = rowOffsets[static_cast<int64_t>(row + 1) * rowOffsetsStride];
        for (int32_t p = begin; p < end; ++p) {
            const int32_t col = colInd[static_cast<int64_t>(p) * colIndStride];
            if (col >= 0 && col < cols) {
                const int32_t pos = next[col]++;
                rowIndices[pos] = row;
                permutation[pos] = p;
            }
        }
    }
}

extern "C" void spmv_arch35_kernel_launch(
    const void* rowOffsets, const void* colInd, const void* values, const void* xVec, void* yVec, void* workspace,
    const SpmvWorkspaceLayout& layout, const SpmvTilingData& tiling, uint32_t blockDim, void* stream)
{
    GM_ADDR colOffsets = nullptr;
    GM_ADDR rowIndices = nullptr;
    GM_ADDR permutation = nullptr;
    if (workspace != nullptr && tiling.transpose != 0 && tiling.nnz > 0) {
        GM_ADDR base = (GM_ADDR)workspace;
        colOffsets = base + layout.colOffsetsOffset;
        rowIndices = base + layout.rowIndicesOffset;
        permutation = base + layout.permutationOffset;
    }
    spmv_arch35_kernel<<<blockDim, nullptr, stream>>>(
        (GM_ADDR)rowOffsets, (GM_ADDR)colInd, (GM_ADDR)values, (GM_ADDR)xVec, (GM_ADDR)yVec, colOffsets, rowIndices,
        permutation, tiling);
}

extern "C" void spmv_arch35_preprocess_launch(
    const void* rowOffsets, const void* colInd, void* workspace, const SpmvWorkspaceLayout& layout, int32_t rows,
    int32_t cols, int32_t nnz, int64_t rowOffsetsStride, int64_t colIndStride, void* stream)
{
    GM_ADDR base = (GM_ADDR)workspace;
    spmv_arch35_preprocess_kernel<<<1, nullptr, stream>>>(
        (GM_ADDR)rowOffsets, (GM_ADDR)colInd, base + layout.colOffsetsOffset, base + layout.nextOffset,
        base + layout.rowIndicesOffset, base + layout.permutationOffset, rows, cols, nnz, rowOffsetsStride,
        colIndStride);
}
