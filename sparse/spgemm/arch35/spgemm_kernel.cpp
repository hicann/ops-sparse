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
#include "simt_api/device_warp_functions.h"
#include "spgemm_kernel.h"

namespace {

struct SpGemmComplex64 {
    float real;
    float imag;
};

__simt_callee__ __aicore__ inline void SetError(__gm__ int32_t *error)
{
    *error = 1;
}

__simt_callee__ __aicore__ inline bool ValidRow(
    __gm__ const int32_t *rowPtr, int32_t row, int32_t nnz,
    int32_t &begin, int32_t &end)
{
    begin = rowPtr[row];
    end = rowPtr[row + 1];
    return begin >= 0 && begin <= end && end <= nnz;
}

__simt_callee__ __aicore__ inline void SetRegularOffset(
    __gm__ int64_t *regularOffsets, __gm__ int64_t *regularTotal,
    int32_t row, int32_t m, int32_t regularDegree)
{
    if (regularDegree <= 0) {
        return;
    }
    int64_t productsPerRow = static_cast<int64_t>(regularDegree) * regularDegree;
    regularOffsets[row] = static_cast<int64_t>(row) * productsPerRow;
    if (row + 1 == m) {
        regularOffsets[m] = static_cast<int64_t>(m) * productsPerRow;
        *regularTotal = static_cast<int64_t>(m) * productsPerRow;
    }
}

__simt_callee__ __aicore__ inline void ValidateARow(
    __gm__ const int32_t *rowPtrA, __gm__ const int32_t *colIndA,
    __gm__ int32_t *regular, __gm__ int32_t *error,
    int32_t row, int32_t k, int32_t n, int32_t nnzA, int32_t regularDegree)
{
    int32_t begin = 0;
    int32_t end = 0;
    if (!ValidRow(rowPtrA, row, nnzA, begin, end)) {
        SetError(error);
        return;
    }
    if (regularDegree > 0 && end - begin != regularDegree) {
        *regular = 0;
    }
    int32_t previous = -1;
    for (int32_t p = begin; p < end; ++p) {
        int32_t col = colIndA[p];
        if (col < 0 || col >= k || col < previous) {
            SetError(error);
        }
        if (regularDegree > 0 && col >= 0 && col < n) {
            int32_t delta = col - row;
            if (delta < 0) {
                delta += n;
            }
            if (delta >= regularDegree) {
                *regular = 0;
            }
        }
        previous = col;
    }
}

__simt_callee__ __aicore__ inline void ValidateBRow(
    __gm__ const int32_t *rowPtrB, __gm__ const int32_t *colIndB,
    __gm__ int32_t *regular, __gm__ int32_t *error,
    int32_t row, int32_t n, int32_t nnzB, int32_t regularDegree)
{
    int32_t begin = 0;
    int32_t end = 0;
    if (!ValidRow(rowPtrB, row, nnzB, begin, end)) {
        SetError(error);
        return;
    }
    if (regularDegree > 0 && end - begin != regularDegree) {
        *regular = 0;
    }
    int32_t previous = -1;
    for (int32_t p = begin; p < end; ++p) {
        int32_t col = colIndB[p];
        if (col < 0 || col >= n || col < previous) {
            SetError(error);
        }
        if (regularDegree > 0 && col >= 0 && col < n) {
            int32_t delta = col - row;
            if (delta < 0) {
                delta += n;
            }
            if (delta % regularDegree != 0 || delta / regularDegree >= regularDegree) {
                *regular = 0;
            }
        }
        previous = col;
    }
}

__simt_vf__ __aicore__ __launch_bounds__(kSpGemmThreads) inline void SpGemmValidateSimt(
    __gm__ const int32_t *rowPtrA, __gm__ const int32_t *colIndA,
    __gm__ const int32_t *rowPtrB, __gm__ const int32_t *colIndB,
    __gm__ int64_t *regularOffsets, __gm__ int64_t *regularTotal,
    __gm__ int32_t *regular, __gm__ int32_t *error,
    int32_t m, int32_t k, int32_t n,
    int32_t nnzA, int32_t nnzB, int32_t rowStart, int32_t rowEnd,
    int32_t threadCount, int32_t regularDegree)
{
    for (int32_t row = rowStart + static_cast<int32_t>(threadIdx.x);
        row < rowEnd; row += threadCount) {
        if (row < m) {
            SetRegularOffset(regularOffsets, regularTotal, row, m, regularDegree);
            ValidateARow(rowPtrA, colIndA, regular, error, row, k, n, nnzA, regularDegree);
        }
        if (row < k) {
            ValidateBRow(rowPtrB, colIndB, regular, error, row, n, nnzB, regularDegree);
        }
    }
}

__simt_vf__ __aicore__ __launch_bounds__(kSpGemmThreads) inline void SpGemmWorkSimt(
    __gm__ const int32_t *rowPtrA, __gm__ const int32_t *colIndA,
    __gm__ const int32_t *rowPtrB, __gm__ int64_t *productCounts,
    __gm__ int32_t *error, int32_t m, int32_t k, int32_t nnzA,
    int32_t nnzB, int32_t rowStart, int32_t rowEnd, int32_t threadCount)
{
    for (int32_t row = rowStart + static_cast<int32_t>(threadIdx.x);
         row < rowEnd; row += threadCount) {
        int32_t aBegin = 0;
        int32_t aEnd = 0;
        int64_t products = 0;
        if (!ValidRow(rowPtrA, row, nnzA, aBegin, aEnd)) {
            SetError(error);
            productCounts[row] = 0;
            continue;
        }
        for (int32_t p = aBegin; p < aEnd; ++p) {
            int32_t bRow = colIndA[p];
            if (bRow < 0 || bRow >= k) {
                SetError(error);
                continue;
            }
            int32_t bBegin = 0;
            int32_t bEnd = 0;
            if (!ValidRow(rowPtrB, bRow, nnzB, bBegin, bEnd)) {
                SetError(error);
                continue;
            }
            products += static_cast<int64_t>(bEnd - bBegin);
        }
        productCounts[row] = products;
    }
}

template <typename CountT, typename OffsetT>
__simt_vf__ __aicore__ __launch_bounds__(kSpGemmThreads) inline void SpGemmScanLocalSimt(
    __gm__ const CountT *counts, __gm__ OffsetT *offsets,
    __gm__ int64_t *blockSums, __gm__ int32_t *error,
    int32_t count, int32_t numChunks, int32_t chunkSize,
    int32_t outerId, int32_t outerBlocks, int32_t threadCount)
{
    int32_t chunk = outerId * threadCount + static_cast<int32_t>(threadIdx.x);
    int32_t stride = outerBlocks * threadCount;
    for (; chunk < numChunks; chunk += stride) {
        int32_t begin = chunk * chunkSize;
        int32_t end = begin + chunkSize;
        if (end > count) {
            end = count;
        }
        int64_t running = 0;
        for (int32_t i = begin; i < end; ++i) {
            offsets[i] = static_cast<OffsetT>(running);
            int64_t next = running + static_cast<int64_t>(counts[i]);
            if (next < running) {
                SetError(error);
            }
            running = next;
        }
        blockSums[chunk] = running;
    }
}

template <typename OffsetT>
__simt_vf__ __aicore__ __launch_bounds__(kSpGemmWarpSize) inline void SpGemmScanBlocksSimt(
    __gm__ OffsetT *offsets, __gm__ const int64_t *blockSums,
    __gm__ int64_t *blockOffsets, __gm__ int64_t *total,
    __gm__ int32_t *error, int32_t count, int32_t numChunks)
{
    int32_t lane = static_cast<int32_t>(threadIdx.x);
    int32_t chunksPerLane = (numChunks + static_cast<int32_t>(kSpGemmWarpSize) - 1) /
        static_cast<int32_t>(kSpGemmWarpSize);
    int32_t begin = lane * chunksPerLane;
    int32_t end = begin + chunksPerLane;
    if (end > numChunks) {
        end = numChunks;
    }
    int64_t laneSum = 0;
    for (int32_t chunk = begin; chunk < end; ++chunk) {
        blockOffsets[chunk] = laneSum;
        int64_t next = laneSum + blockSums[chunk];
        if (next < laneSum) {
            SetError(error);
        }
        laneSum = next;
    }

    int64_t inclusive = laneSum;
    for (int32_t offset = 1; offset < static_cast<int32_t>(kSpGemmWarpSize); offset <<= 1) {
        uint32_t low = static_cast<uint32_t>(inclusive);
        uint32_t high = static_cast<uint32_t>(static_cast<uint64_t>(inclusive) >> 32U);
        uint32_t previousLow = static_cast<uint32_t>(asc_shfl_up(static_cast<int32_t>(low), offset));
        uint32_t previousHigh = static_cast<uint32_t>(asc_shfl_up(static_cast<int32_t>(high), offset));
        if (lane >= offset) {
            inclusive += static_cast<int64_t>((static_cast<uint64_t>(previousHigh) << 32U) | previousLow);
        }
    }
    int64_t laneOffset = inclusive - laneSum;
    for (int32_t chunk = begin; chunk < end; ++chunk) {
        blockOffsets[chunk] += laneOffset;
    }
    if (lane == static_cast<int32_t>(kSpGemmWarpSize) - 1) {
        if (sizeof(OffsetT) == sizeof(int32_t) && inclusive > 0x7FFFFFFFLL) {
            SetError(error);
        }
        offsets[count] = static_cast<OffsetT>(inclusive);
        *total = inclusive;
    }
}

template <typename OffsetT>
__simt_vf__ __aicore__ __launch_bounds__(kSpGemmThreads) inline void SpGemmScanAddSimt(
    __gm__ OffsetT *offsets, __gm__ const int64_t *blockOffsets,
    __gm__ int32_t *error, int32_t count, int32_t chunkSize,
    int32_t rowStart, int32_t rowEnd, int32_t threadCount)
{
    for (int32_t i = rowStart + static_cast<int32_t>(threadIdx.x);
         i < rowEnd; i += threadCount) {
        int64_t value = static_cast<int64_t>(offsets[i]) + blockOffsets[i / chunkSize];
        if (sizeof(OffsetT) == sizeof(int32_t) && value > 0x7FFFFFFFLL) {
            SetError(error);
        }
        offsets[i] = static_cast<OffsetT>(value);
    }
}

template <typename T>
__simt_callee__ __aicore__ inline float SpGemmToFloat(T value)
{
    return static_cast<float>(value);
}

template <typename T>
__simt_callee__ __aicore__ inline T SpGemmFromFloat(float value)
{
    return static_cast<T>(value);
}

template <typename T>
__simt_callee__ __aicore__ inline float SpGemmReadRealScalar(
    uint64_t ptr, float hostValue)
{
    if (ptr == 0) {
        return hostValue;
    }
    __gm__ const T *deviceValue = reinterpret_cast<__gm__ const T *>(ptr);
    return SpGemmToFloat<T>(*deviceValue);
}

__simt_callee__ __aicore__ inline SpGemmComplex64 SpGemmReadComplexScalar(
    uint64_t ptr, float hostReal, float hostImag)
{
    if (ptr == 0) {
        return {hostReal, hostImag};
    }
    __gm__ const SpGemmComplex64 *deviceValue =
        reinterpret_cast<__gm__ const SpGemmComplex64 *>(ptr);
    SpGemmComplex64 value{deviceValue->real, deviceValue->imag};
    return value;
}

template <typename T>
__simt_callee__ __aicore__ inline float SpGemmAccumulateRealColumn(
    int32_t minCol, int32_t aBegin, int32_t aEnd,
    __gm__ const int32_t *colIndA, __gm__ const T *valA,
    __gm__ const int32_t *rowPtrB, __gm__ const int32_t *colIndB,
    __gm__ const T *valB, __gm__ int32_t *cursors)
{
    float sum = 0.0F;
    for (int32_t aPos = aBegin; aPos < aEnd; ++aPos) {
        int32_t cursor = cursors[aPos];
        int32_t bEnd = rowPtrB[colIndA[aPos] + 1];
        while (cursor < bEnd && colIndB[cursor] == minCol) {
            sum += SpGemmToFloat<T>(valA[aPos]) * SpGemmToFloat<T>(valB[cursor]);
            ++cursor;
        }
        cursors[aPos] = cursor;
    }
    return sum;
}

__simt_callee__ __aicore__ inline SpGemmComplex64 SpGemmAccumulateComplexColumn(
    int32_t minCol, int32_t aBegin, int32_t aEnd,
    __gm__ const int32_t *colIndA, __gm__ const SpGemmComplex64 *valA,
    __gm__ const int32_t *rowPtrB, __gm__ const int32_t *colIndB,
    __gm__ const SpGemmComplex64 *valB, __gm__ int32_t *cursors)
{
    SpGemmComplex64 sum{0.0F, 0.0F};
    for (int32_t aPos = aBegin; aPos < aEnd; ++aPos) {
        int32_t cursor = cursors[aPos];
        int32_t bEnd = rowPtrB[colIndA[aPos] + 1];
        while (cursor < bEnd && colIndB[cursor] == minCol) {
            SpGemmComplex64 lhs{valA[aPos].real, valA[aPos].imag};
            SpGemmComplex64 rhs{valB[cursor].real, valB[cursor].imag};
            sum.real += lhs.real * rhs.real - lhs.imag * rhs.imag;
            sum.imag += lhs.real * rhs.imag + lhs.imag * rhs.real;
            ++cursor;
        }
        cursors[aPos] = cursor;
    }
    return sum;
}

__simt_callee__ __aicore__ inline int32_t SpGemmFindMinColumn(
    int32_t aBegin, int32_t aEnd, __gm__ const int32_t *colIndA,
    __gm__ const int32_t *rowPtrB, __gm__ const int32_t *colIndB,
    __gm__ const int32_t *cursors)
{
    int32_t minCol = 0x7FFFFFFF;
    for (int32_t aPos = aBegin; aPos < aEnd; ++aPos) {
        int32_t cursor = cursors[aPos];
        int32_t bEnd = rowPtrB[colIndA[aPos] + 1];
        if (cursor < bEnd && colIndB[cursor] < minCol) {
            minCol = colIndB[cursor];
        }
    }
    return minCol;
}

__simt_callee__ __aicore__ inline int32_t SpGemmShufflePair(
    int32_t first, int32_t second, int32_t index)
{
    uint32_t sourceLane = static_cast<uint32_t>(index >> 1);
    return (index & 1) == 0 ? asc_shfl(first, sourceLane) : asc_shfl(second, sourceLane);
}

__simt_callee__ __aicore__ inline float SpGemmShufflePair(
    float first, float second, int32_t index)
{
    uint32_t sourceLane = static_cast<uint32_t>(index >> 1);
    return (index & 1) == 0 ? asc_shfl(first, sourceLane) : asc_shfl(second, sourceLane);
}

__simt_callee__ __aicore__ inline int32_t SpGemmSortWidth(int32_t count)
{
    if (count <= 1) { return 1; }
    if (count <= 2) { return 2; }
    if (count <= 4) { return 4; }
    if (count <= 8) { return 8; }
    if (count <= 16) { return 16; }
    if (count <= 32) { return 32; }
    return 64;
}

__simt_callee__ __aicore__ inline float SpGemmShuffleValue(
    float first, float second, int32_t index)
{
    return SpGemmShufflePair(first, second, index);
}

__simt_callee__ __aicore__ inline SpGemmComplex64 SpGemmShuffleValue(
    const SpGemmComplex64 &first, const SpGemmComplex64 &second, int32_t index)
{
    return {
        SpGemmShufflePair(first.real, second.real, index),
        SpGemmShufflePair(first.imag, second.imag, index)};
}

template <typename ValueT>
__simt_callee__ __aicore__ inline void SpGemmSortPair(
    int32_t lane, int32_t width,
    int32_t &col0, ValueT &value0, int32_t &col1, ValueT &value1)
{
    int32_t index0 = lane << 1;
    int32_t index1 = index0 + 1;
    for (int32_t size = 2; size <= width; size <<= 1) {
        for (int32_t stride = size >> 1; stride > 0; stride >>= 1) {
            int32_t oldCol0 = col0;
            int32_t oldCol1 = col1;
            ValueT oldValue0 = value0;
            ValueT oldValue1 = value1;
            int32_t peer0 = index0 ^ stride;
            int32_t peer1 = index1 ^ stride;
            int32_t peerCol0 = SpGemmShufflePair(oldCol0, oldCol1, peer0);
            int32_t peerCol1 = SpGemmShufflePair(oldCol0, oldCol1, peer1);
            ValueT peerValue0 = SpGemmShuffleValue(oldValue0, oldValue1, peer0);
            ValueT peerValue1 = SpGemmShuffleValue(oldValue0, oldValue1, peer1);
            bool wantMin0 = ((index0 & size) == 0) == ((index0 & stride) == 0);
            bool wantMin1 = ((index1 & size) == 0) == ((index1 & stride) == 0);
            bool takePeer0 = wantMin0 ? peerCol0 < oldCol0 : peerCol0 > oldCol0;
            bool takePeer1 = wantMin1 ? peerCol1 < oldCol1 : peerCol1 > oldCol1;
            col0 = takePeer0 ? peerCol0 : oldCol0;
            col1 = takePeer1 ? peerCol1 : oldCol1;
            value0 = takePeer0 ? peerValue0 : oldValue0;
            value1 = takePeer1 ? peerValue1 : oldValue1;
        }
    }
}

struct SpGemmProductPosition {
    int32_t aPos;
    int32_t bPos;
};

__simt_callee__ __aicore__ inline SpGemmProductPosition SpGemmLocateProduct(
    int32_t product, int32_t count, int32_t aBegin, int32_t aEnd,
    __gm__ const int32_t *colIndA, __gm__ const int32_t *rowPtrB)
{
    if (product >= count) {
        return {-1, -1};
    }
    int32_t remaining = product;
    for (int32_t aPos = aBegin; aPos < aEnd; ++aPos) {
        int32_t bBegin = rowPtrB[colIndA[aPos]];
        int32_t bEnd = rowPtrB[colIndA[aPos] + 1];
        int32_t length = bEnd - bBegin;
        if (remaining < length) {
            return {aPos, bBegin + remaining};
        }
        remaining -= length;
    }
    return {-1, -1};
}

template <typename T>
__simt_callee__ __aicore__ inline void SpGemmLoadRealProduct(
    int32_t product, int32_t count, int32_t aBegin, int32_t aEnd,
    __gm__ const int32_t *colIndA, __gm__ const T *valA,
    __gm__ const int32_t *rowPtrB, __gm__ const int32_t *colIndB,
    __gm__ const T *valB, float alpha, int32_t &col, float &value)
{
    col = 0x7FFFFFFF;
    value = 0.0F;
    SpGemmProductPosition position = SpGemmLocateProduct(
        product, count, aBegin, aEnd, colIndA, rowPtrB);
    if (position.aPos < 0) {
        return;
    }
    col = colIndB[position.bPos];
    value = alpha * SpGemmToFloat<T>(valA[position.aPos]) *
        SpGemmToFloat<T>(valB[position.bPos]);
}

__simt_callee__ __aicore__ inline void SpGemmLoadComplexProduct(
    int32_t product, int32_t count, int32_t aBegin, int32_t aEnd,
    __gm__ const int32_t *colIndA, __gm__ const SpGemmComplex64 *valA,
    __gm__ const int32_t *rowPtrB, __gm__ const int32_t *colIndB,
    __gm__ const SpGemmComplex64 *valB, SpGemmComplex64 alpha,
    int32_t &col, SpGemmComplex64 &value)
{
    col = 0x7FFFFFFF;
    value = {0.0F, 0.0F};
    SpGemmProductPosition position = SpGemmLocateProduct(
        product, count, aBegin, aEnd, colIndA, rowPtrB);
    if (position.aPos < 0) {
        return;
    }
    SpGemmComplex64 lhs{valA[position.aPos].real, valA[position.aPos].imag};
    SpGemmComplex64 rhs{valB[position.bPos].real, valB[position.bPos].imag};
    SpGemmComplex64 productValue{
        lhs.real * rhs.real - lhs.imag * rhs.imag,
        lhs.real * rhs.imag + lhs.imag * rhs.real};
    value = {
        alpha.real * productValue.real - alpha.imag * productValue.imag,
        alpha.real * productValue.imag + alpha.imag * productValue.real};
    col = colIndB[position.bPos];
}

__simt_callee__ __aicore__ inline void SpGemmSetEmitPositions(
    int32_t lane, int32_t count, int32_t col0, int32_t col1,
    int32_t &position0, int32_t &position1, bool &emit0, bool &emit1)
{
    int32_t index0 = lane << 1;
    int32_t index1 = index0 + 1;
    emit0 = index0 < count &&
        (index0 + 1 == count || col0 != SpGemmShufflePair(col0, col1, index0 + 1));
    emit1 = index1 < count &&
        (index1 + 1 == count || col1 != SpGemmShufflePair(col0, col1, index1 + 1));
    position0 = emit0 ? 1 : 0;
    position1 = emit1 ? 1 : 0;
    for (int32_t offset = 1; offset < 64; offset <<= 1) {
        int32_t oldPosition0 = position0;
        int32_t oldPosition1 = position1;
        int32_t previous0 = index0 - offset;
        int32_t previous1 = index1 - offset;
        if (previous0 >= 0) {
            position0 += SpGemmShufflePair(oldPosition0, oldPosition1, previous0);
        }
        if (previous1 >= 0) {
            position1 += SpGemmShufflePair(oldPosition0, oldPosition1, previous1);
        }
    }
}

__simt_callee__ __aicore__ inline void SpGemmMergeRealPair(
    int32_t lane, int32_t count, int32_t col0, int32_t col1,
    float &value0, float &value1, int32_t &position0, int32_t &position1,
    bool &emit0, bool &emit1)
{
    int32_t index0 = lane << 1;
    int32_t index1 = index0 + 1;
    for (int32_t offset = 1; offset < 64; offset <<= 1) {
        float oldValue0 = value0;
        float oldValue1 = value1;
        int32_t previous0 = index0 - offset;
        int32_t previous1 = index1 - offset;
        if (previous0 >= 0 && col0 == SpGemmShufflePair(col0, col1, previous0)) {
            value0 += SpGemmShufflePair(oldValue0, oldValue1, previous0);
        }
        if (previous1 >= 0 && col1 == SpGemmShufflePair(col0, col1, previous1)) {
            value1 += SpGemmShufflePair(oldValue0, oldValue1, previous1);
        }
    }
    SpGemmSetEmitPositions(
        lane, count, col0, col1, position0, position1, emit0, emit1);
}

__simt_callee__ __aicore__ inline void SpGemmMergeComplexPair(
    int32_t lane, int32_t count, int32_t col0, int32_t col1,
    SpGemmComplex64 &value0, SpGemmComplex64 &value1,
    int32_t &position0, int32_t &position1, bool &emit0, bool &emit1)
{
    int32_t index0 = lane << 1;
    int32_t index1 = index0 + 1;
    for (int32_t offset = 1; offset < 64; offset <<= 1) {
        SpGemmComplex64 oldValue0 = value0;
        SpGemmComplex64 oldValue1 = value1;
        int32_t previous0 = index0 - offset;
        int32_t previous1 = index1 - offset;
        if (previous0 >= 0 && col0 == SpGemmShufflePair(col0, col1, previous0)) {
            value0.real += SpGemmShufflePair(oldValue0.real, oldValue1.real, previous0);
            value0.imag += SpGemmShufflePair(oldValue0.imag, oldValue1.imag, previous0);
        }
        if (previous1 >= 0 && col1 == SpGemmShufflePair(col0, col1, previous1)) {
            value1.real += SpGemmShufflePair(oldValue0.real, oldValue1.real, previous1);
            value1.imag += SpGemmShufflePair(oldValue0.imag, oldValue1.imag, previous1);
        }
    }
    SpGemmSetEmitPositions(
        lane, count, col0, col1, position0, position1, emit0, emit1);
}

struct SpGemmMergedRowState {
    int32_t aBegin;
    int32_t aEnd;
    int32_t cPos;
    int32_t cEnd;
    int32_t count;
};

__simt_callee__ __aicore__ inline SpGemmMergedRowState SpGemmInitializeMergedRow(
    __gm__ const int32_t *rowPtrA, __gm__ const int32_t *colIndA,
    __gm__ const int32_t *rowPtrB, __gm__ const int32_t *oldRowPtrC,
    __gm__ int32_t *cursors, int32_t row, bool useC)
{
    SpGemmMergedRowState state{
        rowPtrA[row], rowPtrA[row + 1],
        useC ? oldRowPtrC[row] : 0, useC ? oldRowPtrC[row + 1] : 0, 0};
    for (int32_t aPos = state.aBegin; aPos < state.aEnd; ++aPos) {
        cursors[aPos] = rowPtrB[colIndA[aPos]];
    }
    return state;
}

template <typename T>
__simt_callee__ __aicore__ inline void SpGemmComputeRealMergedRow(
    __gm__ const int32_t *rowPtrA, __gm__ const int32_t *colIndA, __gm__ const T *valA,
    __gm__ const int32_t *rowPtrB, __gm__ const int32_t *colIndB, __gm__ const T *valB,
    __gm__ const int32_t *oldRowPtrC, __gm__ const int32_t *oldColIndC, __gm__ const T *oldValC,
    __gm__ int32_t *cursors, __gm__ int32_t *candidateCols,
    __gm__ T *candidateVals, __gm__ int32_t *uniqueCounts, __gm__ int32_t *error,
    int32_t row, int64_t outBase, float alpha, float beta, bool useC)
{
    SpGemmMergedRowState state = SpGemmInitializeMergedRow(
        rowPtrA, colIndA, rowPtrB, oldRowPtrC, cursors, row, useC);
    while (true) {
        int32_t minCol = SpGemmFindMinColumn(
            state.aBegin, state.aEnd, colIndA, rowPtrB, colIndB, cursors);
        if (minCol == 0x7FFFFFFF) {
            break;
        }
        float value = alpha * SpGemmAccumulateRealColumn<T>(
            minCol, state.aBegin, state.aEnd, colIndA, valA,
            rowPtrB, colIndB, valB, cursors);
        if (useC) {
            if (state.cPos >= state.cEnd || oldColIndC[state.cPos] != minCol) {
                SetError(error);
            } else {
                value += beta * SpGemmToFloat<T>(oldValC[state.cPos]);
                ++state.cPos;
            }
        }
        candidateCols[outBase + state.count] = minCol;
        candidateVals[outBase + state.count] = SpGemmFromFloat<T>(value);
        ++state.count;
    }
    if (useC && state.cPos != state.cEnd) {
        SetError(error);
    }
    uniqueCounts[row] = state.count;
}

__simt_callee__ __aicore__ inline void SpGemmAddOldComplex(
    __gm__ const int32_t *oldColIndC, __gm__ const SpGemmComplex64 *oldValC,
    __gm__ int32_t *error, int32_t minCol, int32_t cEnd,
    const SpGemmComplex64 &beta, SpGemmComplex64 &value, int32_t &cPos)
{
    if (cPos >= cEnd || oldColIndC[cPos] != minCol) {
        SetError(error);
        return;
    }
    SpGemmComplex64 old{oldValC[cPos].real, oldValC[cPos].imag};
    value.real += beta.real * old.real - beta.imag * old.imag;
    value.imag += beta.real * old.imag + beta.imag * old.real;
    ++cPos;
}

__simt_callee__ __aicore__ inline void SpGemmComputeComplexMergedRow(
    __gm__ const int32_t *rowPtrA, __gm__ const int32_t *colIndA,
    __gm__ const SpGemmComplex64 *valA, __gm__ const int32_t *rowPtrB,
    __gm__ const int32_t *colIndB, __gm__ const SpGemmComplex64 *valB,
    __gm__ const int32_t *oldRowPtrC, __gm__ const int32_t *oldColIndC,
    __gm__ const SpGemmComplex64 *oldValC, __gm__ int32_t *cursors,
    __gm__ int32_t *candidateCols, __gm__ SpGemmComplex64 *candidateVals,
    __gm__ int32_t *uniqueCounts, __gm__ int32_t *error, int32_t row,
    int64_t outBase, const SpGemmComplex64 &alpha,
    const SpGemmComplex64 &beta, bool useC)
{
    SpGemmMergedRowState state = SpGemmInitializeMergedRow(
        rowPtrA, colIndA, rowPtrB, oldRowPtrC, cursors, row, useC);
    while (true) {
        int32_t minCol = SpGemmFindMinColumn(
            state.aBegin, state.aEnd, colIndA, rowPtrB, colIndB, cursors);
        if (minCol == 0x7FFFFFFF) {
            break;
        }
        SpGemmComplex64 product = SpGemmAccumulateComplexColumn(
            minCol, state.aBegin, state.aEnd, colIndA, valA,
            rowPtrB, colIndB, valB, cursors);
        SpGemmComplex64 value{
            alpha.real * product.real - alpha.imag * product.imag,
            alpha.real * product.imag + alpha.imag * product.real};
        if (useC) {
            SpGemmAddOldComplex(oldColIndC, oldValC, error, minCol,
                state.cEnd, beta, value, state.cPos);
        }
        candidateCols[outBase + state.count] = minCol;
        candidateVals[outBase + state.count].real = value.real;
        candidateVals[outBase + state.count].imag = value.imag;
        ++state.count;
    }
    if (useC && state.cPos != state.cEnd) {
        SetError(error);
    }
    uniqueCounts[row] = state.count;
}

template <typename T>
__simt_callee__ __aicore__ inline void SpGemmComputeRealSmallRow(
    __gm__ const int32_t *rowPtrA, __gm__ const int32_t *colIndA, __gm__ const T *valA,
    __gm__ const int32_t *rowPtrB, __gm__ const int32_t *colIndB, __gm__ const T *valB,
    __gm__ int32_t *candidateCols, __gm__ T *candidateVals,
    __gm__ int32_t *uniqueCounts, int32_t row, int32_t lane,
    int32_t count, int64_t outBase, float alpha)
{
    int32_t aBegin = rowPtrA[row];
    int32_t aEnd = rowPtrA[row + 1];
    int32_t col0 = 0;
    int32_t col1 = 0;
    float value0 = 0.0F;
    float value1 = 0.0F;
    int32_t width = SpGemmSortWidth(count);
    SpGemmLoadRealProduct<T>(lane << 1, count, aBegin, aEnd,
        colIndA, valA, rowPtrB, colIndB, valB, alpha, col0, value0);
    SpGemmLoadRealProduct<T>((lane << 1) + 1, count, aBegin, aEnd,
        colIndA, valA, rowPtrB, colIndB, valB, alpha, col1, value1);
    SpGemmSortPair(lane, width, col0, value0, col1, value1);
    int32_t position0 = 0;
    int32_t position1 = 0;
    bool emit0 = false;
    bool emit1 = false;
    SpGemmMergeRealPair(lane, count, col0, col1, value0, value1,
        position0, position1, emit0, emit1);
    if (emit0) {
        candidateCols[outBase + position0 - 1] = col0;
        candidateVals[outBase + position0 - 1] = SpGemmFromFloat<T>(value0);
    }
    if (emit1) {
        candidateCols[outBase + position1 - 1] = col1;
        candidateVals[outBase + position1 - 1] = SpGemmFromFloat<T>(value1);
    }
    if (lane == 0) {
        uniqueCounts[row] = asc_shfl(position1, 31U);
    }
}

__simt_callee__ __aicore__ inline void SpGemmComputeComplexSmallRow(
    __gm__ const int32_t *rowPtrA, __gm__ const int32_t *colIndA,
    __gm__ const SpGemmComplex64 *valA, __gm__ const int32_t *rowPtrB,
    __gm__ const int32_t *colIndB, __gm__ const SpGemmComplex64 *valB,
    __gm__ int32_t *candidateCols, __gm__ SpGemmComplex64 *candidateVals,
    __gm__ int32_t *uniqueCounts, int32_t row, int32_t lane,
    int32_t count, int64_t outBase, const SpGemmComplex64 &alpha)
{
    int32_t aBegin = rowPtrA[row];
    int32_t aEnd = rowPtrA[row + 1];
    int32_t col0 = 0;
    int32_t col1 = 0;
    SpGemmComplex64 value0{0.0F, 0.0F};
    SpGemmComplex64 value1{0.0F, 0.0F};
    int32_t width = SpGemmSortWidth(count);
    SpGemmLoadComplexProduct(lane << 1, count, aBegin, aEnd,
        colIndA, valA, rowPtrB, colIndB, valB, alpha, col0, value0);
    SpGemmLoadComplexProduct((lane << 1) + 1, count, aBegin, aEnd,
        colIndA, valA, rowPtrB, colIndB, valB, alpha, col1, value1);
    SpGemmSortPair(lane, width, col0, value0, col1, value1);
    int32_t position0 = 0;
    int32_t position1 = 0;
    bool emit0 = false;
    bool emit1 = false;
    SpGemmMergeComplexPair(lane, count, col0, col1, value0, value1,
        position0, position1, emit0, emit1);
    if (emit0) {
        candidateCols[outBase + position0 - 1] = col0;
        candidateVals[outBase + position0 - 1].real = value0.real;
        candidateVals[outBase + position0 - 1].imag = value0.imag;
    }
    if (emit1) {
        candidateCols[outBase + position1 - 1] = col1;
        candidateVals[outBase + position1 - 1].real = value1.real;
        candidateVals[outBase + position1 - 1].imag = value1.imag;
    }
    if (lane == 0) {
        uniqueCounts[row] = asc_shfl(position1, 31U);
    }
}

template <typename T>
__simt_vf__ __aicore__ __launch_bounds__(kSpGemmThreads) inline void SpGemmComputeRealWarpSimt(
    __gm__ const int32_t *rowPtrA, __gm__ const int32_t *colIndA, __gm__ const T *valA,
    __gm__ const int32_t *rowPtrB, __gm__ const int32_t *colIndB, __gm__ const T *valB,
    __gm__ const int32_t *oldRowPtrC, __gm__ const int32_t *oldColIndC, __gm__ const T *oldValC,
    __gm__ const int64_t *productOffsets, __gm__ int32_t *cursors,
    __gm__ int32_t *candidateCols, __gm__ T *candidateVals, __gm__ int32_t *uniqueCounts,
    __gm__ int32_t *error, int32_t m, float alphaHost, float betaHost,
    uint64_t alphaPtr, uint64_t betaPtr, int32_t rowStart, int32_t rowEnd, int32_t threadCount)
{
    int32_t lane = static_cast<int32_t>(threadIdx.x) & 31;
    int32_t warp = static_cast<int32_t>(threadIdx.x) >> 5;
    int32_t warps = threadCount >> 5;
    float alpha = SpGemmReadRealScalar<T>(alphaPtr, alphaHost);
    float beta = SpGemmReadRealScalar<T>(betaPtr, betaHost);
    bool useC = beta != 0.0F;
    for (int32_t row = rowStart + warp; row < rowEnd && row < m; row += warps) {
        int64_t outBase = productOffsets[row];
        int64_t productCount64 = productOffsets[row + 1] - outBase;
        if (!useC && productCount64 <= 64) {
            int32_t count = static_cast<int32_t>(productCount64);
            SpGemmComputeRealSmallRow<T>(
                rowPtrA, colIndA, valA, rowPtrB, colIndB, valB,
                candidateCols, candidateVals, uniqueCounts,
                row, lane, count, outBase, alpha);
            continue;
        }
        if (lane != 0) {
            continue;
        }
        SpGemmComputeRealMergedRow<T>(
            rowPtrA, colIndA, valA, rowPtrB, colIndB, valB,
            oldRowPtrC, oldColIndC, oldValC, cursors, candidateCols,
            candidateVals, uniqueCounts, error, row, outBase, alpha, beta, useC);
    }
}

__simt_vf__ __aicore__ __launch_bounds__(kSpGemmThreads) inline void SpGemmComputeComplexWarpSimt(
    __gm__ const int32_t *rowPtrA, __gm__ const int32_t *colIndA,
    __gm__ const SpGemmComplex64 *valA, __gm__ const int32_t *rowPtrB,
    __gm__ const int32_t *colIndB, __gm__ const SpGemmComplex64 *valB,
    __gm__ const int32_t *oldRowPtrC, __gm__ const int32_t *oldColIndC,
    __gm__ const SpGemmComplex64 *oldValC, __gm__ const int64_t *productOffsets,
    __gm__ int32_t *cursors, __gm__ int32_t *candidateCols,
    __gm__ SpGemmComplex64 *candidateVals, __gm__ int32_t *uniqueCounts,
    __gm__ int32_t *error, int32_t m, float alphaReal, float alphaImag,
    float betaReal, float betaImag, uint64_t alphaPtr, uint64_t betaPtr,
    int32_t rowStart, int32_t rowEnd, int32_t threadCount)
{
    int32_t lane = static_cast<int32_t>(threadIdx.x) & 31;
    int32_t warp = static_cast<int32_t>(threadIdx.x) >> 5;
    int32_t warps = threadCount >> 5;
    SpGemmComplex64 alpha = SpGemmReadComplexScalar(alphaPtr, alphaReal, alphaImag);
    SpGemmComplex64 beta = SpGemmReadComplexScalar(betaPtr, betaReal, betaImag);
    bool useC = beta.real != 0.0F || beta.imag != 0.0F;
    for (int32_t row = rowStart + warp; row < rowEnd && row < m; row += warps) {
        int64_t outBase = productOffsets[row];
        int64_t productCount64 = productOffsets[row + 1] - outBase;
        if (!useC && productCount64 <= 64) {
            int32_t count = static_cast<int32_t>(productCount64);
            SpGemmComputeComplexSmallRow(
                rowPtrA, colIndA, valA, rowPtrB, colIndB, valB,
                candidateCols, candidateVals, uniqueCounts,
                row, lane, count, outBase, alpha);
            continue;
        }
        if (lane != 0) {
            continue;
        }
        SpGemmComputeComplexMergedRow(
            rowPtrA, colIndA, valA, rowPtrB, colIndB, valB,
            oldRowPtrC, oldColIndC, oldValC, cursors, candidateCols,
            candidateVals, uniqueCounts, error, row, outBase, alpha, beta, useC);
    }
}

struct SpGemmRegularPosition {
    int32_t aPos;
    int32_t bPos;
    int32_t col;
};

__simt_callee__ __aicore__ inline SpGemmRegularPosition GetRegularPosition(
    __gm__ const int32_t *rowPtrA, __gm__ const int32_t *rowPtrB,
    int32_t row, int32_t outputInRow, int32_t n,
    int32_t degree, int32_t productsPerRow)
{
    int32_t firstProduct = row + productsPerRow > n ? n - row : 0;
    int32_t product = firstProduct + outputInRow;
    if (product >= productsPerRow) {
        product -= productsPerRow;
    }
    int32_t aLogical = product % degree;
    int32_t bLogical = product / degree;
    int32_t firstA = row + degree > n ? n - row : 0;
    int32_t aSlot = aLogical - firstA;
    if (aSlot < 0) {
        aSlot += degree;
    }
    int32_t bRow = row + aLogical;
    if (bRow >= n) {
        bRow -= n;
    }
    int32_t lastBOffset = (degree - 1) * degree;
    int32_t firstB = bRow + lastBOffset >= n ?
        (n - bRow + degree - 1) / degree : 0;
    int32_t bSlot = bLogical - firstB;
    if (bSlot < 0) {
        bSlot += degree;
    }
    int32_t col = row + product;
    if (col >= n) {
        col -= n;
    }
    return {rowPtrA[row] + aSlot, rowPtrB[bRow] + bSlot, col};
}

__simt_callee__ __aicore__ inline void StoreRegularRowMetadata(
    __gm__ int32_t *uniqueCounts, __gm__ int32_t *rowPtrC,
    int32_t row, int32_t n, int32_t outputInRow,
    int32_t productsPerRow, int64_t output)
{
    if (outputInRow != 0) {
        return;
    }
    uniqueCounts[row] = productsPerRow;
    rowPtrC[row] = static_cast<int32_t>(output);
    if (row + 1 == n) {
        rowPtrC[n] = static_cast<int32_t>(output + productsPerRow);
    }
}

template <typename T>
__simt_vf__ __aicore__ __launch_bounds__(kSpGemmThreads) inline void SpGemmComputeRegularRealSimt(
    __gm__ const int32_t *rowPtrA, __gm__ const T *valA,
    __gm__ const int32_t *rowPtrB, __gm__ const T *valB,
    __gm__ const int64_t *productOffsets, __gm__ int32_t *candidateCols,
    __gm__ T *candidateVals, __gm__ int32_t *uniqueCounts,
    __gm__ int32_t *rowPtrC, __gm__ int32_t *directCols, __gm__ T *directVals,
    int32_t n, int32_t degree, float alphaHost, uint64_t alphaPtr,
    int32_t directOutput, int32_t rowStart, int32_t rowEnd, int32_t threadCount)
{
    float alpha = SpGemmReadRealScalar<T>(alphaPtr, alphaHost);
    int32_t productsPerRow = degree * degree;
    int32_t localProducts = (rowEnd - rowStart) * productsPerRow;
    for (int32_t index = static_cast<int32_t>(threadIdx.x);
         index < localProducts; index += threadCount) {
        int32_t row = rowStart + index / productsPerRow;
        int32_t outputInRow = index % productsPerRow;
        SpGemmRegularPosition position = GetRegularPosition(
            rowPtrA, rowPtrB, row, outputInRow, n, degree, productsPerRow);
        int64_t output = productOffsets[row] + outputInRow;
        float value = alpha * SpGemmToFloat<T>(valA[position.aPos]) *
            SpGemmToFloat<T>(valB[position.bPos]);
        if (directOutput != 0) {
            directCols[output] = position.col;
            directVals[output] = SpGemmFromFloat<T>(value);
        } else {
            candidateCols[output] = position.col;
            candidateVals[output] = SpGemmFromFloat<T>(value);
        }
        StoreRegularRowMetadata(
            uniqueCounts, rowPtrC, row, n, outputInRow, productsPerRow, output);
    }
}

__simt_vf__ __aicore__ __launch_bounds__(kSpGemmThreads) inline void SpGemmComputeRegularComplexSimt(
    __gm__ const int32_t *rowPtrA, __gm__ const SpGemmComplex64 *valA,
    __gm__ const int32_t *rowPtrB, __gm__ const SpGemmComplex64 *valB,
    __gm__ const int64_t *productOffsets, __gm__ int32_t *candidateCols,
    __gm__ SpGemmComplex64 *candidateVals, __gm__ int32_t *uniqueCounts,
    __gm__ int32_t *rowPtrC, __gm__ int32_t *directCols,
    __gm__ SpGemmComplex64 *directVals,
    int32_t n, int32_t degree, float alphaReal, float alphaImag,
    uint64_t alphaPtr, int32_t directOutput,
    int32_t rowStart, int32_t rowEnd, int32_t threadCount)
{
    SpGemmComplex64 alpha = SpGemmReadComplexScalar(alphaPtr, alphaReal, alphaImag);
    int32_t productsPerRow = degree * degree;
    int32_t localProducts = (rowEnd - rowStart) * productsPerRow;
    for (int32_t index = static_cast<int32_t>(threadIdx.x);
         index < localProducts; index += threadCount) {
        int32_t row = rowStart + index / productsPerRow;
        int32_t outputInRow = index % productsPerRow;
        SpGemmRegularPosition position = GetRegularPosition(
            rowPtrA, rowPtrB, row, outputInRow, n, degree, productsPerRow);
        SpGemmComplex64 lhs{valA[position.aPos].real, valA[position.aPos].imag};
        SpGemmComplex64 rhs{valB[position.bPos].real, valB[position.bPos].imag};
        SpGemmComplex64 productValue{
            lhs.real * rhs.real - lhs.imag * rhs.imag,
            lhs.real * rhs.imag + lhs.imag * rhs.real};
        SpGemmComplex64 value{
            alpha.real * productValue.real - alpha.imag * productValue.imag,
            alpha.real * productValue.imag + alpha.imag * productValue.real};
        int64_t output = productOffsets[row] + outputInRow;
        if (directOutput != 0) {
            directCols[output] = position.col;
            directVals[output].real = value.real;
            directVals[output].imag = value.imag;
        } else {
            candidateCols[output] = position.col;
            candidateVals[output].real = value.real;
            candidateVals[output].imag = value.imag;
        }
        StoreRegularRowMetadata(
            uniqueCounts, rowPtrC, row, n, outputInRow, productsPerRow, output);
    }
}

template <typename T>
__simt_vf__ __aicore__ __launch_bounds__(kSpGemmThreads) inline void SpGemmCopySimt(
    __gm__ const int64_t *productOffsets, __gm__ const int32_t *uniqueCounts,
    __gm__ const int32_t *candidateCols, __gm__ const T *candidateVals,
    __gm__ const int32_t *rowPtrC, __gm__ int32_t *colIndC, __gm__ T *valC,
    int32_t m, int32_t rowStart, int32_t rowEnd, int32_t threadCount)
{
    for (int32_t row = rowStart + static_cast<int32_t>(threadIdx.x);
         row < rowEnd && row < m; row += threadCount) {
        int64_t src = productOffsets[row];
        int32_t dst = rowPtrC[row];
        int32_t count = uniqueCounts[row];
        for (int32_t j = 0; j < count; ++j) {
            colIndC[dst + j] = candidateCols[src + j];
            valC[dst + j] = candidateVals[src + j];
        }
    }
}

template <typename T>
__simt_vf__ __aicore__ __launch_bounds__(kSpGemmThreads) inline void SpGemmCopyFlatSimt(
    __gm__ const int32_t *candidateCols, __gm__ const T *candidateVals,
    __gm__ int32_t *colIndC, __gm__ T *valC, int32_t count,
    int32_t block, int32_t blocks, int32_t threadCount)
{
    int32_t index = block * threadCount + static_cast<int32_t>(threadIdx.x);
    int32_t stride = blocks * threadCount;
    for (; index < count; index += stride) {
        colIndC[index] = candidateCols[index];
        valC[index] = candidateVals[index];
    }
}

__simt_vf__ __aicore__ __launch_bounds__(kSpGemmThreads) inline void SpGemmCopyComplexFlatSimt(
    __gm__ const int32_t *candidateCols, __gm__ const SpGemmComplex64 *candidateVals,
    __gm__ int32_t *colIndC, __gm__ SpGemmComplex64 *valC, int32_t count,
    int32_t block, int32_t blocks, int32_t threadCount)
{
    int32_t index = block * threadCount + static_cast<int32_t>(threadIdx.x);
    int32_t stride = blocks * threadCount;
    for (; index < count; index += stride) {
        colIndC[index] = candidateCols[index];
        valC[index].real = candidateVals[index].real;
        valC[index].imag = candidateVals[index].imag;
    }
}

__simt_vf__ __aicore__ __launch_bounds__(kSpGemmThreads) inline void SpGemmCopyComplexSimt(
    __gm__ const int64_t *productOffsets, __gm__ const int32_t *uniqueCounts,
    __gm__ const int32_t *candidateCols, __gm__ const SpGemmComplex64 *candidateVals,
    __gm__ const int32_t *rowPtrC, __gm__ int32_t *colIndC,
    __gm__ SpGemmComplex64 *valC, int32_t m, int32_t rowStart,
    int32_t rowEnd, int32_t threadCount)
{
    for (int32_t row = rowStart + static_cast<int32_t>(threadIdx.x);
         row < rowEnd && row < m; row += threadCount) {
        int64_t src = productOffsets[row];
        int32_t dst = rowPtrC[row];
        int32_t count = uniqueCounts[row];
        for (int32_t j = 0; j < count; ++j) {
            colIndC[dst + j] = candidateCols[src + j];
            valC[dst + j].real = candidateVals[src + j].real;
            valC[dst + j].imag = candidateVals[src + j].imag;
        }
    }
}

__aicore__ inline void SpGemmRowRange(
    int32_t totalRows, uint32_t rowsPerBlock,
    int32_t &rowStart, int32_t &rowEnd, uint32_t &threadCount)
{
    rowStart = static_cast<int32_t>(AscendC::GetBlockIdx()) * static_cast<int32_t>(rowsPerBlock);
    rowEnd = rowStart + static_cast<int32_t>(rowsPerBlock);
    if (rowEnd > totalRows) {
        rowEnd = totalRows;
    }
    int32_t range = rowEnd - rowStart;
    threadCount = range > 0 ? static_cast<uint32_t>(range) : 1U;
    if (threadCount > kSpGemmThreads) {
        threadCount = kSpGemmThreads;
    }
    threadCount = (threadCount + kSpGemmWarpSize - 1U) & ~(kSpGemmWarpSize - 1U);
}

template <typename T>
__aicore__ inline void SpGemmDispatchRealCompute(
    GM_ADDR rowPtrA, GM_ADDR colIndA, GM_ADDR valA,
    GM_ADDR rowPtrB, GM_ADDR colIndB, GM_ADDR valB,
    GM_ADDR oldRowPtrC, GM_ADDR oldColIndC, GM_ADDR oldValC,
    GM_ADDR productOffsets, GM_ADDR cursors, GM_ADDR candidateCols,
    GM_ADDR candidateVals, GM_ADDR uniqueCounts, GM_ADDR error,
    const SpGemmComputeTilingData &tiling,
    int32_t rowStart, int32_t rowEnd, uint32_t threads)
{
    if (tiling.regularDegree > 0) {
        asc_vf_call<SpGemmComputeRegularRealSimt<T>>(dim3{threads},
            (__gm__ const int32_t *)rowPtrA, (__gm__ const T *)valA,
            (__gm__ const int32_t *)rowPtrB, (__gm__ const T *)valB,
            (__gm__ const int64_t *)productOffsets, (__gm__ int32_t *)candidateCols,
            (__gm__ T *)candidateVals, (__gm__ int32_t *)uniqueCounts,
            (__gm__ int32_t *)oldRowPtrC, (__gm__ int32_t *)oldColIndC, (__gm__ T *)oldValC,
            tiling.n, tiling.regularDegree, tiling.alphaReal, tiling.alphaPtr,
            tiling.directOutput, rowStart, rowEnd, static_cast<int32_t>(threads));
        return;
    }
    asc_vf_call<SpGemmComputeRealWarpSimt<T>>(dim3{threads},
        (__gm__ const int32_t *)rowPtrA, (__gm__ const int32_t *)colIndA, (__gm__ const T *)valA,
        (__gm__ const int32_t *)rowPtrB, (__gm__ const int32_t *)colIndB, (__gm__ const T *)valB,
        (__gm__ const int32_t *)oldRowPtrC, (__gm__ const int32_t *)oldColIndC, (__gm__ const T *)oldValC,
        (__gm__ const int64_t *)productOffsets, (__gm__ int32_t *)cursors,
        (__gm__ int32_t *)candidateCols, (__gm__ T *)candidateVals, (__gm__ int32_t *)uniqueCounts,
        (__gm__ int32_t *)error, tiling.m, tiling.alphaReal, tiling.betaReal,
        tiling.alphaPtr, tiling.betaPtr, rowStart, rowEnd, static_cast<int32_t>(threads));
}

__aicore__ inline void SpGemmDispatchComplexCompute(
    GM_ADDR rowPtrA, GM_ADDR colIndA, GM_ADDR valA,
    GM_ADDR rowPtrB, GM_ADDR colIndB, GM_ADDR valB,
    GM_ADDR oldRowPtrC, GM_ADDR oldColIndC, GM_ADDR oldValC,
    GM_ADDR productOffsets, GM_ADDR cursors, GM_ADDR candidateCols,
    GM_ADDR candidateVals, GM_ADDR uniqueCounts, GM_ADDR error,
    const SpGemmComputeTilingData &tiling,
    int32_t rowStart, int32_t rowEnd, uint32_t threads)
{
    if (tiling.regularDegree > 0) {
        asc_vf_call<SpGemmComputeRegularComplexSimt>(dim3{threads},
            (__gm__ const int32_t *)rowPtrA, (__gm__ const SpGemmComplex64 *)valA,
            (__gm__ const int32_t *)rowPtrB, (__gm__ const SpGemmComplex64 *)valB,
            (__gm__ const int64_t *)productOffsets, (__gm__ int32_t *)candidateCols,
            (__gm__ SpGemmComplex64 *)candidateVals, (__gm__ int32_t *)uniqueCounts,
            (__gm__ int32_t *)oldRowPtrC, (__gm__ int32_t *)oldColIndC,
            (__gm__ SpGemmComplex64 *)oldValC,
            tiling.n, tiling.regularDegree, tiling.alphaReal, tiling.alphaImag,
            tiling.alphaPtr, tiling.directOutput,
            rowStart, rowEnd, static_cast<int32_t>(threads));
        return;
    }
    asc_vf_call<SpGemmComputeComplexWarpSimt>(dim3{threads},
        (__gm__ const int32_t *)rowPtrA, (__gm__ const int32_t *)colIndA,
        (__gm__ const SpGemmComplex64 *)valA, (__gm__ const int32_t *)rowPtrB,
        (__gm__ const int32_t *)colIndB, (__gm__ const SpGemmComplex64 *)valB,
        (__gm__ const int32_t *)oldRowPtrC, (__gm__ const int32_t *)oldColIndC,
        (__gm__ const SpGemmComplex64 *)oldValC, (__gm__ const int64_t *)productOffsets,
        (__gm__ int32_t *)cursors, (__gm__ int32_t *)candidateCols,
        (__gm__ SpGemmComplex64 *)candidateVals, (__gm__ int32_t *)uniqueCounts,
        (__gm__ int32_t *)error, tiling.m, tiling.alphaReal, tiling.alphaImag,
        tiling.betaReal, tiling.betaImag, tiling.alphaPtr, tiling.betaPtr,
        rowStart, rowEnd, static_cast<int32_t>(threads));
}

template <typename T>
__aicore__ inline void SpGemmDispatchRealCopy(
    GM_ADDR productOffsets, GM_ADDR uniqueCounts,
    GM_ADDR candidateCols, GM_ADDR candidateVals,
    GM_ADDR rowPtrC, GM_ADDR colIndC, GM_ADDR valC,
    const SpGemmCopyTilingData &tiling,
    int32_t rowStart, int32_t rowEnd, uint32_t threads)
{
    if (tiling.nnzC == tiling.numProducts) {
        asc_vf_call<SpGemmCopyFlatSimt<T>>(dim3{kSpGemmThreads},
            (__gm__ const int32_t *)candidateCols, (__gm__ const T *)candidateVals,
            (__gm__ int32_t *)colIndC, (__gm__ T *)valC, tiling.nnzC,
            static_cast<int32_t>(AscendC::GetBlockIdx()), static_cast<int32_t>(tiling.numBlocks),
            static_cast<int32_t>(kSpGemmThreads));
        return;
    }
    asc_vf_call<SpGemmCopySimt<T>>(dim3{threads},
        (__gm__ const int64_t *)productOffsets, (__gm__ const int32_t *)uniqueCounts,
        (__gm__ const int32_t *)candidateCols, (__gm__ const T *)candidateVals,
        (__gm__ const int32_t *)rowPtrC, (__gm__ int32_t *)colIndC, (__gm__ T *)valC,
        tiling.m, rowStart, rowEnd, static_cast<int32_t>(threads));
}

__aicore__ inline void SpGemmDispatchComplexCopy(
    GM_ADDR productOffsets, GM_ADDR uniqueCounts,
    GM_ADDR candidateCols, GM_ADDR candidateVals,
    GM_ADDR rowPtrC, GM_ADDR colIndC, GM_ADDR valC,
    const SpGemmCopyTilingData &tiling,
    int32_t rowStart, int32_t rowEnd, uint32_t threads)
{
    if (tiling.nnzC == tiling.numProducts) {
        asc_vf_call<SpGemmCopyComplexFlatSimt>(dim3{kSpGemmThreads},
            (__gm__ const int32_t *)candidateCols,
            (__gm__ const SpGemmComplex64 *)candidateVals,
            (__gm__ int32_t *)colIndC, (__gm__ SpGemmComplex64 *)valC,
            tiling.nnzC, static_cast<int32_t>(AscendC::GetBlockIdx()),
            static_cast<int32_t>(tiling.numBlocks), static_cast<int32_t>(kSpGemmThreads));
        return;
    }
    asc_vf_call<SpGemmCopyComplexSimt>(dim3{threads},
        (__gm__ const int64_t *)productOffsets, (__gm__ const int32_t *)uniqueCounts,
        (__gm__ const int32_t *)candidateCols, (__gm__ const SpGemmComplex64 *)candidateVals,
        (__gm__ const int32_t *)rowPtrC, (__gm__ int32_t *)colIndC,
        (__gm__ SpGemmComplex64 *)valC, tiling.m, rowStart, rowEnd,
        static_cast<int32_t>(threads));
}

}  // namespace

extern "C" __global__ __aicore__ void spgemm_validate_kernel(
    GM_ADDR rowPtrA, GM_ADDR colIndA, GM_ADDR rowPtrB, GM_ADDR colIndB,
    GM_ADDR regularOffsets, GM_ADDR regularTotal, GM_ADDR regular,
    GM_ADDR error, const SpGemmValidateTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    int32_t rows = tiling.m > tiling.k ? tiling.m : tiling.k;
    int32_t rowStart = 0;
    int32_t rowEnd = 0;
    uint32_t threads = 0;
    SpGemmRowRange(rows, tiling.rowsPerBlock, rowStart, rowEnd, threads);
    asc_vf_call<SpGemmValidateSimt>(dim3{threads},
        (__gm__ const int32_t *)rowPtrA, (__gm__ const int32_t *)colIndA,
        (__gm__ const int32_t *)rowPtrB, (__gm__ const int32_t *)colIndB,
        (__gm__ int64_t *)regularOffsets, (__gm__ int64_t *)regularTotal,
        (__gm__ int32_t *)regular, (__gm__ int32_t *)error,
        tiling.m, tiling.k, tiling.n,
        tiling.nnzA, tiling.nnzB, rowStart, rowEnd,
        static_cast<int32_t>(threads), tiling.regularDegree);
}

extern "C" __global__ __aicore__ void spgemm_work_kernel(
    GM_ADDR rowPtrA, GM_ADDR colIndA, GM_ADDR rowPtrB,
    GM_ADDR productCounts, GM_ADDR regular, GM_ADDR error,
    const SpGemmWorkTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if (*(__gm__ int32_t *)regular != 0) { return; }
    int32_t rowStart = 0;
    int32_t rowEnd = 0;
    uint32_t threads = 0;
    SpGemmRowRange(tiling.m, tiling.rowsPerBlock, rowStart, rowEnd, threads);
    asc_vf_call<SpGemmWorkSimt>(dim3{threads},
        (__gm__ const int32_t *)rowPtrA, (__gm__ const int32_t *)colIndA,
        (__gm__ const int32_t *)rowPtrB, (__gm__ int64_t *)productCounts,
        (__gm__ int32_t *)error, tiling.m, tiling.k, tiling.nnzA,
        tiling.nnzB, rowStart, rowEnd, static_cast<int32_t>(threads));
}

template <typename CountT, typename OffsetT>
__aicore__ inline void SpGemmScanLocalDispatch(
    GM_ADDR counts, GM_ADDR offsets, GM_ADDR blockSums, GM_ADDR error,
    const SpGemmScanTilingData &tiling)
{
    asc_vf_call<SpGemmScanLocalSimt<CountT, OffsetT>>(dim3{kSpGemmThreads},
        (__gm__ const CountT *)counts, (__gm__ OffsetT *)offsets,
        (__gm__ int64_t *)blockSums, (__gm__ int32_t *)error,
        tiling.count, tiling.numChunks, tiling.chunkSize,
        static_cast<int32_t>(AscendC::GetBlockIdx()),
        static_cast<int32_t>(tiling.outerBlocks), static_cast<int32_t>(kSpGemmThreads));
}

extern "C" __global__ __aicore__ void spgemm_scan_i64_local_kernel(
    GM_ADDR counts, GM_ADDR offsets, GM_ADDR blockSums, GM_ADDR regular, GM_ADDR error,
    const SpGemmScanTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if (*(__gm__ int32_t *)regular != 0) { return; }
    SpGemmScanLocalDispatch<int64_t, int64_t>(counts, offsets, blockSums, error, tiling);
}

extern "C" __global__ __aicore__ void spgemm_scan_i32_local_kernel(
    GM_ADDR counts, GM_ADDR offsets, GM_ADDR blockSums, GM_ADDR error,
    const SpGemmScanTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    SpGemmScanLocalDispatch<int32_t, int32_t>(counts, offsets, blockSums, error, tiling);
}

template <typename OffsetT>
__aicore__ inline void SpGemmScanBlocksDispatch(
    GM_ADDR offsets, GM_ADDR blockSums, GM_ADDR blockOffsets,
    GM_ADDR total, GM_ADDR error, const SpGemmScanTilingData &tiling)
{
    asc_vf_call<SpGemmScanBlocksSimt<OffsetT>>(dim3{kSpGemmWarpSize},
        (__gm__ OffsetT *)offsets, (__gm__ const int64_t *)blockSums,
        (__gm__ int64_t *)blockOffsets, (__gm__ int64_t *)total,
        (__gm__ int32_t *)error, tiling.count, tiling.numChunks);
}

extern "C" __global__ __aicore__ void spgemm_scan_i64_blocks_kernel(
    GM_ADDR offsets, GM_ADDR blockSums, GM_ADDR blockOffsets,
    GM_ADDR total, GM_ADDR regular, GM_ADDR error, const SpGemmScanTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if (*(__gm__ int32_t *)regular != 0) { return; }
    SpGemmScanBlocksDispatch<int64_t>(offsets, blockSums, blockOffsets, total, error, tiling);
}

extern "C" __global__ __aicore__ void spgemm_scan_i32_blocks_kernel(
    GM_ADDR offsets, GM_ADDR blockSums, GM_ADDR blockOffsets,
    GM_ADDR total, GM_ADDR error, const SpGemmScanTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    SpGemmScanBlocksDispatch<int32_t>(offsets, blockSums, blockOffsets, total, error, tiling);
}

template <typename OffsetT>
__aicore__ inline void SpGemmScanAddDispatch(
    GM_ADDR offsets, GM_ADDR blockOffsets, GM_ADDR error,
    const SpGemmScanTilingData &tiling)
{
    int32_t rowStart = 0;
    int32_t rowEnd = 0;
    uint32_t threads = 0;
    uint32_t rowsPerBlock = static_cast<uint32_t>((tiling.count + static_cast<int32_t>(tiling.outerBlocks) - 1) /
                                                   static_cast<int32_t>(tiling.outerBlocks));
    SpGemmRowRange(tiling.count, rowsPerBlock, rowStart, rowEnd, threads);
    asc_vf_call<SpGemmScanAddSimt<OffsetT>>(dim3{threads},
        (__gm__ OffsetT *)offsets, (__gm__ const int64_t *)blockOffsets,
        (__gm__ int32_t *)error, tiling.count, tiling.chunkSize,
        rowStart, rowEnd, static_cast<int32_t>(threads));
}

extern "C" __global__ __aicore__ void spgemm_scan_i64_add_kernel(
    GM_ADDR offsets, GM_ADDR blockOffsets, GM_ADDR regular, GM_ADDR error,
    const SpGemmScanTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if (*(__gm__ int32_t *)regular != 0) { return; }
    SpGemmScanAddDispatch<int64_t>(offsets, blockOffsets, error, tiling);
}

extern "C" __global__ __aicore__ void spgemm_scan_i32_add_kernel(
    GM_ADDR offsets, GM_ADDR blockOffsets, GM_ADDR error,
    const SpGemmScanTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    SpGemmScanAddDispatch<int32_t>(offsets, blockOffsets, error, tiling);
}

extern "C" __global__ __aicore__ void spgemm_compute_kernel(
    GM_ADDR rowPtrA, GM_ADDR colIndA, GM_ADDR valA,
    GM_ADDR rowPtrB, GM_ADDR colIndB, GM_ADDR valB,
    GM_ADDR oldRowPtrC, GM_ADDR oldColIndC, GM_ADDR oldValC,
    GM_ADDR productOffsets, GM_ADDR cursors, GM_ADDR candidateCols,
    GM_ADDR candidateVals, GM_ADDR uniqueCounts, GM_ADDR error,
    const SpGemmComputeTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    int32_t rowStart = 0;
    int32_t rowEnd = 0;
    uint32_t threads = 0;
    SpGemmRowRange(tiling.m, tiling.rowsPerBlock, rowStart, rowEnd, threads);
    if (tiling.valType == SPGEMM_VAL_FP16) {
        SpGemmDispatchRealCompute<half>(
            rowPtrA, colIndA, valA, rowPtrB, colIndB, valB,
            oldRowPtrC, oldColIndC, oldValC, productOffsets, cursors,
            candidateCols, candidateVals, uniqueCounts, error,
            tiling, rowStart, rowEnd, threads);
    } else if (tiling.valType == SPGEMM_VAL_BF16) {
        SpGemmDispatchRealCompute<bfloat16_t>(
            rowPtrA, colIndA, valA, rowPtrB, colIndB, valB,
            oldRowPtrC, oldColIndC, oldValC, productOffsets, cursors,
            candidateCols, candidateVals, uniqueCounts, error,
            tiling, rowStart, rowEnd, threads);
    } else if (tiling.valType == SPGEMM_VAL_FP32) {
        SpGemmDispatchRealCompute<float>(
            rowPtrA, colIndA, valA, rowPtrB, colIndB, valB,
            oldRowPtrC, oldColIndC, oldValC, productOffsets, cursors,
            candidateCols, candidateVals, uniqueCounts, error,
            tiling, rowStart, rowEnd, threads);
    } else {
        SpGemmDispatchComplexCompute(
            rowPtrA, colIndA, valA, rowPtrB, colIndB, valB,
            oldRowPtrC, oldColIndC, oldValC, productOffsets, cursors,
            candidateCols, candidateVals, uniqueCounts, error,
            tiling, rowStart, rowEnd, threads);
    }
}

extern "C" __global__ __aicore__ void spgemm_copy_kernel(
    GM_ADDR productOffsets, GM_ADDR uniqueCounts,
    GM_ADDR candidateCols, GM_ADDR candidateVals,
    GM_ADDR rowPtrC, GM_ADDR colIndC, GM_ADDR valC,
    const SpGemmCopyTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    int32_t rowStart = 0;
    int32_t rowEnd = 0;
    uint32_t threads = 0;
    SpGemmRowRange(tiling.m, tiling.rowsPerBlock, rowStart, rowEnd, threads);
    if (tiling.valType == SPGEMM_VAL_FP16) {
        SpGemmDispatchRealCopy<half>(
            productOffsets, uniqueCounts, candidateCols, candidateVals,
            rowPtrC, colIndC, valC, tiling, rowStart, rowEnd, threads);
    } else if (tiling.valType == SPGEMM_VAL_BF16) {
        SpGemmDispatchRealCopy<bfloat16_t>(
            productOffsets, uniqueCounts, candidateCols, candidateVals,
            rowPtrC, colIndC, valC, tiling, rowStart, rowEnd, threads);
    } else if (tiling.valType == SPGEMM_VAL_FP32) {
        SpGemmDispatchRealCopy<float>(
            productOffsets, uniqueCounts, candidateCols, candidateVals,
            rowPtrC, colIndC, valC, tiling, rowStart, rowEnd, threads);
    } else {
        SpGemmDispatchComplexCopy(
            productOffsets, uniqueCounts, candidateCols, candidateVals,
            rowPtrC, colIndC, valC, tiling, rowStart, rowEnd, threads);
    }
}

extern "C" void spgemm_validate_kernel_do(
    GM_ADDR rowPtrA, GM_ADDR colIndA, GM_ADDR rowPtrB, GM_ADDR colIndB,
    GM_ADDR regularOffsets, GM_ADDR regularTotal, GM_ADDR regular,
    GM_ADDR error, const SpGemmValidateTilingData &tiling,
    uint32_t numBlocks, void *stream)
{
    spgemm_validate_kernel<<<numBlocks, nullptr, stream>>>(
        rowPtrA, colIndA, rowPtrB, colIndB, regularOffsets,
        regularTotal, regular, error, tiling);
}

extern "C" void spgemm_work_kernel_do(
    GM_ADDR rowPtrA, GM_ADDR colIndA, GM_ADDR rowPtrB,
    GM_ADDR productCounts, GM_ADDR regular, GM_ADDR error,
    const SpGemmWorkTilingData &tiling, uint32_t numBlocks, void *stream)
{
    spgemm_work_kernel<<<numBlocks, nullptr, stream>>>(
        rowPtrA, colIndA, rowPtrB, productCounts, regular, error, tiling);
}

extern "C" void spgemm_scan_i64_kernel_do(
    GM_ADDR counts, GM_ADDR offsets, GM_ADDR blockSums,
    GM_ADDR blockOffsets, GM_ADDR total, GM_ADDR regular, GM_ADDR error,
    const SpGemmScanTilingData &tiling, uint32_t numBlocks, void *stream)
{
    spgemm_scan_i64_local_kernel<<<numBlocks, nullptr, stream>>>(
        counts, offsets, blockSums, regular, error, tiling);
    spgemm_scan_i64_blocks_kernel<<<1, nullptr, stream>>>(
        offsets, blockSums, blockOffsets, total, regular, error, tiling);
    spgemm_scan_i64_add_kernel<<<numBlocks, nullptr, stream>>>(
        offsets, blockOffsets, regular, error, tiling);
}

extern "C" void spgemm_compute_kernel_do(
    GM_ADDR rowPtrA, GM_ADDR colIndA, GM_ADDR valA,
    GM_ADDR rowPtrB, GM_ADDR colIndB, GM_ADDR valB,
    GM_ADDR oldRowPtrC, GM_ADDR oldColIndC, GM_ADDR oldValC,
    GM_ADDR productOffsets, GM_ADDR cursors,
    GM_ADDR candidateCols, GM_ADDR candidateVals, GM_ADDR uniqueCounts,
    GM_ADDR error, const SpGemmComputeTilingData &tiling,
    uint32_t numBlocks, void *stream)
{
    spgemm_compute_kernel<<<numBlocks, nullptr, stream>>>(
        rowPtrA, colIndA, valA, rowPtrB, colIndB, valB,
        oldRowPtrC, oldColIndC, oldValC, productOffsets, cursors,
        candidateCols, candidateVals, uniqueCounts, error, tiling);
}

extern "C" void spgemm_scan_i32_kernel_do(
    GM_ADDR counts, GM_ADDR offsets, GM_ADDR blockSums,
    GM_ADDR blockOffsets, GM_ADDR total, GM_ADDR error,
    const SpGemmScanTilingData &tiling, uint32_t numBlocks, void *stream)
{
    spgemm_scan_i32_local_kernel<<<numBlocks, nullptr, stream>>>(counts, offsets, blockSums, error, tiling);
    spgemm_scan_i32_blocks_kernel<<<1, nullptr, stream>>>(
        offsets, blockSums, blockOffsets, total, error, tiling);
    spgemm_scan_i32_add_kernel<<<numBlocks, nullptr, stream>>>(offsets, blockOffsets, error, tiling);
}

extern "C" void spgemm_copy_kernel_do(
    GM_ADDR productOffsets, GM_ADDR uniqueCounts,
    GM_ADDR candidateCols, GM_ADDR candidateVals,
    GM_ADDR rowPtrC, GM_ADDR colIndC, GM_ADDR valC,
    const SpGemmCopyTilingData &tiling, uint32_t numBlocks, void *stream)
{
    spgemm_copy_kernel<<<numBlocks, nullptr, stream>>>(
        productOffsets, uniqueCounts, candidateCols, candidateVals,
        rowPtrC, colIndC, valC, tiling);
}
