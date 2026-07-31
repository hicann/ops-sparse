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

#include "kernel_operator.h"
#include "simt_api/asc_simt.h"
#include "simt_api/common_functions.h"
#include "densetosparse_kernel.h"

namespace {

constexpr uint32_t kFormatCoo = 0;
constexpr uint32_t kFormatCsr = 1;
constexpr uint32_t kFormatBell = 3;
constexpr uint32_t kOrderRow = 0;
constexpr uint64_t kScanChunk = 1024;
// Maximum minor-dimension elements processed by one CSR/CSC count unit; this is
// a software scheduling granularity, not a hardware or UB capacity limit.
constexpr uint64_t kMajorChunk = 4096;
constexpr uint64_t kCooTile = 1024;
constexpr uint64_t kAlign = 32;
constexpr int32_t kDeviceStatusSuccess = 0;
constexpr int32_t kDeviceStatusInvalid = 1;

template <typename T>
__simt_callee__ __aicore__ inline bool IsNonzero(T bits)
{
    return bits != static_cast<T>(0);
}

template <>
__simt_callee__ __aicore__ inline bool IsNonzero<uint16_t>(uint16_t bits)
{
    return (bits & 0x7FFFU) != 0;
}

template <>
__simt_callee__ __aicore__ inline bool IsNonzero<uint32_t>(uint32_t bits)
{
    return (bits & 0x7FFFFFFFU) != 0;
}

__simt_callee__ __aicore__ inline uint64_t DensePosition(
    uint64_t row, uint64_t col, uint64_t ld, uint32_t order)
{
    return order == kOrderRow ? row * ld + col : col * ld + row;
}

template <typename ValueT>
__simt_vf__ __aicore__ __launch_bounds__(kDenseToSparseThreads) inline void
CountUnits(__gm__ const ValueT *dense, __gm__ uint64_t *counts,
    uint64_t rows, uint64_t cols, uint64_t ld, uint64_t unitCount,
    uint32_t format, uint32_t order, uint32_t numBlocks)
{
    const uint64_t tid = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const uint64_t stride = static_cast<uint64_t>(numBlocks) * blockDim.x;
    const uint64_t minorDim = format == kFormatCsr ? cols : rows;
    const uint64_t chunks = format == kFormatCoo ? 0 :
        (minorDim + kMajorChunk - 1) / kMajorChunk;
    for (uint64_t unit = tid; unit < unitCount; unit += stride) {
        uint64_t count = 0;
        uint64_t begin = 0;
        uint64_t end = 0;
        if (format == kFormatCoo) {
            begin = unit * kCooTile;
            end = begin + kCooTile;
            const uint64_t logical = rows * cols;
            if (end > logical) {
                end = logical;
            }
            for (uint64_t pos = begin; pos < end; ++pos) {
                const uint64_t row = pos / cols;
                const uint64_t col = pos - row * cols;
                count += IsNonzero<ValueT>(
                    dense[DensePosition(row, col, ld, order)]) ? 1 : 0;
            }
        } else {
            const uint64_t major = unit / chunks;
            const uint64_t chunk = unit - major * chunks;
            begin = chunk * kMajorChunk;
            end = begin + kMajorChunk;
            if (end > minorDim) {
                end = minorDim;
            }
            for (uint64_t minor = begin; minor < end; ++minor) {
                const uint64_t row = format == kFormatCsr ? major : minor;
                const uint64_t col = format == kFormatCsr ? minor : major;
                count += IsNonzero<ValueT>(
                    dense[DensePosition(row, col, ld, order)]) ? 1 : 0;
            }
        }
        counts[unit] = count;
    }
}

__simt_vf__ __aicore__ __launch_bounds__(kDenseToSparseThreads) inline void
ScanInclusive(__gm__ uint64_t *values, __gm__ uint64_t *chunkSums,
    uint64_t count, uint64_t chunkCount, uint32_t numBlocks)
{
    const uint64_t tid = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const uint64_t stride = static_cast<uint64_t>(numBlocks) * blockDim.x;
    for (uint64_t chunk = tid; chunk < chunkCount; chunk += stride) {
        const uint64_t begin = chunk * kScanChunk;
        uint64_t end = begin + kScanChunk;
        if (end > count) {
            end = count;
        }
        uint64_t sum = 0;
        for (uint64_t i = begin; i < end; ++i) {
            sum += values[i];
            values[i] = sum;
        }
        if (chunkSums != nullptr) {
            chunkSums[chunk] = sum;
        }
    }
}

__simt_vf__ __aicore__ __launch_bounds__(kDenseToSparseThreads) inline void
AddChunkBases(__gm__ uint64_t *values, __gm__ const uint64_t *chunkPrefix,
    uint64_t count, uint32_t numBlocks)
{
    const uint64_t tid = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const uint64_t stride = static_cast<uint64_t>(numBlocks) * blockDim.x;
    for (uint64_t i = tid; i < count; i += stride) {
        const uint64_t chunk = i / kScanChunk;
        if (chunk > 0) {
            values[i] += chunkPrefix[chunk - 1];
        }
    }
}

__simt_vf__ __aicore__ __launch_bounds__(kDenseToSparseThreads) inline void
InitializeHeader(__gm__ int32_t *status, __gm__ uint64_t *total)
{
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        status[0] = kDeviceStatusSuccess;
        total[0] = 0;
    }
}

__simt_vf__ __aicore__ __launch_bounds__(kDenseToSparseThreads) inline void
WriteTotal(__gm__ const uint64_t *prefix, __gm__ int32_t *status,
    __gm__ uint64_t *total, uint64_t count, uint64_t expectedNnz,
    uint32_t base, uint32_t offsetType, uint32_t validateExpected)
{
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        const uint64_t nnz = prefix[count - 1];
        total[0] = nnz;
        if ((offsetType == 0 && nnz + base > 0x7FFFFFFFULL) ||
            (validateExpected != 0 && nnz != expectedNnz)) {
            status[0] = kDeviceStatusInvalid;
        }
    }
}

template <typename OffsetT>
__simt_vf__ __aicore__ __launch_bounds__(kDenseToSparseThreads) inline void
WriteOffsets(__gm__ const uint64_t *prefix, __gm__ OffsetT *offsets,
    __gm__ int32_t *status, uint64_t majorDim, uint64_t chunks,
    uint32_t base, uint32_t numBlocks)
{
    const uint64_t tid = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const uint64_t stride = static_cast<uint64_t>(numBlocks) * blockDim.x;
    if (tid == 0) {
        offsets[0] = static_cast<OffsetT>(base);
    }
    for (uint64_t major = tid; major < majorDim; major += stride) {
        const uint64_t value = prefix[(major + 1) * chunks - 1] + base;
        if (sizeof(OffsetT) == sizeof(int32_t) && value > 0x7FFFFFFFULL) {
            status[0] = kDeviceStatusInvalid;
        } else {
            offsets[major + 1] = static_cast<OffsetT>(value);
        }
    }
}

template <typename OffsetT>
__simt_vf__ __aicore__ __launch_bounds__(kDenseToSparseThreads) inline void
FillEmptyOffsets(__gm__ OffsetT *offsets, uint64_t count, uint32_t base,
    uint32_t numBlocks)
{
    const uint64_t tid = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const uint64_t stride = static_cast<uint64_t>(numBlocks) * blockDim.x;
    for (uint64_t i = tid; i < count; i += stride) {
        offsets[i] = static_cast<OffsetT>(base);
    }
}

template <typename OffsetT>
__simt_vf__ __aicore__ __launch_bounds__(kDenseToSparseThreads) inline void
ValidateOffsets(__gm__ const uint64_t *prefix, __gm__ const OffsetT *offsets,
    __gm__ int32_t *status, uint64_t majorDim, uint64_t chunks,
    uint64_t nnz, uint32_t base, uint32_t numBlocks)
{
    const uint64_t tid = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const uint64_t stride = static_cast<uint64_t>(numBlocks) * blockDim.x;
    for (uint64_t major = tid; major < majorDim; major += stride) {
        const int64_t encodedStart = static_cast<int64_t>(offsets[major]);
        const int64_t encodedEnd = static_cast<int64_t>(offsets[major + 1]);
        const uint64_t expectedStart = major == 0 ?
            0 : prefix[major * chunks - 1];
        const uint64_t expectedEnd = prefix[(major + 1) * chunks - 1];
        if (encodedStart < static_cast<int64_t>(base) ||
            encodedEnd < encodedStart) {
            status[0] = kDeviceStatusInvalid;
            continue;
        }
        const uint64_t start = static_cast<uint64_t>(encodedStart - base);
        const uint64_t end = static_cast<uint64_t>(encodedEnd - base);
        if (start != expectedStart || end != expectedEnd || end > nnz) {
            status[0] = kDeviceStatusInvalid;
        }
    }
}

__simt_callee__ __aicore__ inline uint64_t GetUnitOutputPosition(
    __gm__ const uint64_t *prefix, GM_ADDR offsets, uint64_t unit,
    uint64_t major, uint64_t chunks, uint32_t format, uint32_t base,
    uint32_t offsetType)
{
    const uint64_t prefixBefore = unit == 0 ? 0 : prefix[unit - 1];
    if (format == kFormatCoo) {
        return prefixBefore;
    }
    const uint64_t majorPrefix = major == 0 ?
        0 : prefix[major * chunks - 1];
    const int64_t encodedOffset = offsetType == 0 ?
        static_cast<int64_t>(((__gm__ const int32_t *)offsets)[major]) :
        ((__gm__ const int64_t *)offsets)[major];
    return static_cast<uint64_t>(encodedOffset - base) +
        prefixBefore - majorPrefix;
}

template <typename ValueT, typename IndexT>
__simt_vf__ __aicore__ __launch_bounds__(kDenseToSparseThreads) inline void
ConvertUnits(__gm__ const ValueT *dense, __gm__ const uint64_t *prefix,
    __gm__ const int32_t *status, GM_ADDR offsets, __gm__ IndexT *indices,
    __gm__ IndexT *rowIndices, __gm__ IndexT *colIndices,
    __gm__ ValueT *values, uint64_t rows, uint64_t cols, uint64_t ld,
    uint64_t unitCount, uint32_t format, uint32_t order, uint32_t base,
    uint32_t offsetType, uint32_t numBlocks)
{
    if (status[0] != kDeviceStatusSuccess) {
        return;
    }
    const uint64_t tid = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const uint64_t stride = static_cast<uint64_t>(numBlocks) * blockDim.x;
    const uint64_t minorDim = format == kFormatCsr ? cols : rows;
    const uint64_t chunks = format == kFormatCoo ? 0 :
        (minorDim + kMajorChunk - 1) / kMajorChunk;
    for (uint64_t unit = tid; unit < unitCount; unit += stride) {
        const uint64_t major = format == kFormatCoo ? 0 : unit / chunks;
        uint64_t out = GetUnitOutputPosition(prefix, offsets, unit, major,
            chunks, format, base, offsetType);
        uint64_t begin = format == kFormatCoo ?
            unit * kCooTile : (unit % chunks) * kMajorChunk;
        uint64_t end = begin +
            (format == kFormatCoo ? kCooTile : kMajorChunk);
        const uint64_t limit = format == kFormatCoo ? rows * cols : minorDim;
        if (end > limit) {
            end = limit;
        }
        for (uint64_t item = begin; item < end; ++item) {
            const uint64_t row = format == kFormatCoo ? item / cols :
                (format == kFormatCsr ? major : item);
            const uint64_t col = format == kFormatCoo ? item - row * cols :
                (format == kFormatCsr ? item : major);
            const ValueT bits = dense[DensePosition(row, col, ld, order)];
            if (IsNonzero<ValueT>(bits)) {
                if (format == kFormatCoo) {
                    rowIndices[out] = static_cast<IndexT>(row + base);
                    colIndices[out] = static_cast<IndexT>(col + base);
                } else {
                    indices[out] = static_cast<IndexT>(item + base);
                }
                values[out] = bits;
                ++out;
            }
        }
    }
}

template <typename ValueT, typename IndexT>
__simt_vf__ __aicore__ __launch_bounds__(kDenseToSparseThreads) inline void
ConvertBell(__gm__ const ValueT *dense, __gm__ const IndexT *ellColInd,
    __gm__ ValueT *values, uint64_t rows, uint64_t cols, uint64_t ld,
    uint64_t blockSize, uint64_t ellCols, uint32_t order, uint32_t base,
    uint32_t numBlocks)
{
    const uint64_t blockRows = rows / blockSize;
    const uint64_t slots = ellCols / blockSize;
    const uint64_t tasks = blockRows * slots;
    const uint64_t tid = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const uint64_t stride = static_cast<uint64_t>(numBlocks) * blockDim.x;
    for (uint64_t task = tid; task < tasks; task += stride) {
        const uint64_t blockRow = task / slots;
        const int64_t encoded = static_cast<int64_t>(ellColInd[task]);
        const bool valid = encoded >= static_cast<int64_t>(base) &&
            static_cast<uint64_t>(encoded - base) < cols / blockSize;
        const uint64_t blockCol = valid ? static_cast<uint64_t>(encoded - base) : 0;
        for (uint64_t innerCol = 0; innerCol < blockSize; ++innerCol) {
            for (uint64_t innerRow = 0; innerRow < blockSize; ++innerRow) {
                const uint64_t row = blockRow * blockSize + innerRow;
                const uint64_t valuePos = task * blockSize * blockSize +
                    innerCol * blockSize + innerRow;
                const uint64_t col = blockCol * blockSize + innerCol;
                values[valuePos] = valid ?
                    dense[DensePosition(row, col, ld, order)] :
                    static_cast<ValueT>(0);
            }
        }
    }
}

template <typename ValueT>
__aicore__ inline void DispatchCount(
    GM_ADDR dense, GM_ADDR level0, const DenseToSparseTilingData &tiling)
{
    asc_vf_call<CountUnits<ValueT>>(dim3{kDenseToSparseThreads},
        (__gm__ const ValueT *)dense, (__gm__ uint64_t *)level0, tiling.rows,
        tiling.cols, tiling.ld, tiling.unitCount, tiling.format, tiling.order,
        tiling.numBlocks);
}

template <typename ValueT, typename IndexT>
__aicore__ inline void DispatchConvert(
    GM_ADDR dense, GM_ADDR prefix, GM_ADDR status, GM_ADDR offsets,
    GM_ADDR indices, GM_ADDR rowIndices, GM_ADDR colIndices, GM_ADDR values,
    const DenseToSparseTilingData &tiling)
{
    asc_vf_call<ConvertUnits<ValueT, IndexT>>(dim3{kDenseToSparseThreads},
        (__gm__ const ValueT *)dense, (__gm__ const uint64_t *)prefix,
        (__gm__ const int32_t *)status, offsets,
        (__gm__ IndexT *)indices, (__gm__ IndexT *)rowIndices,
        (__gm__ IndexT *)colIndices, (__gm__ ValueT *)values, tiling.rows,
        tiling.cols, tiling.ld, tiling.unitCount, tiling.format, tiling.order,
        tiling.base, tiling.offsetType, tiling.numBlocks);
}

template <typename ValueT, typename IndexT>
__aicore__ inline void DispatchBell(
    GM_ADDR dense, GM_ADDR ellColInd, GM_ADDR values,
    const DenseToSparseTilingData &tiling)
{
    asc_vf_call<ConvertBell<ValueT, IndexT>>(dim3{kDenseToSparseThreads},
        (__gm__ const ValueT *)dense, (__gm__ const IndexT *)ellColInd,
        (__gm__ ValueT *)values, tiling.rows, tiling.cols, tiling.ld,
        tiling.ellBlockSize, tiling.ellCols, tiling.order, tiling.base,
        tiling.numBlocks);
}

uint64_t Align32(uint64_t value)
{
    return (value + kAlign - 1) & ~(kAlign - 1);
}

uint64_t NextLevelCount(uint64_t count)
{
    return (count + kScanChunk - 1) / kScanChunk;
}

__aicore__ inline void GetCompressedLayout(
    const DenseToSparseTilingData &tiling, uint64_t &major, uint64_t &chunks)
{
    major = tiling.format == kFormatCsr ? tiling.rows : tiling.cols;
    const uint64_t minor =
        tiling.format == kFormatCsr ? tiling.cols : tiling.rows;
    chunks = (minor + kMajorChunk - 1) / kMajorChunk;
}

} // namespace

extern "C" __global__ __aicore__ void densetosparse_count_kernel(
    GM_ADDR dense, GM_ADDR level0, DenseToSparseTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if (tiling.elementBytes == 1) {
        DispatchCount<uint8_t>(dense, level0, tiling);
    } else if (tiling.elementBytes == 2) {
        DispatchCount<uint16_t>(dense, level0, tiling);
    } else {
        DispatchCount<uint32_t>(dense, level0, tiling);
    }
}

extern "C" __global__ __aicore__ void densetosparse_header_kernel(
    GM_ADDR status, GM_ADDR total)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    asc_vf_call<InitializeHeader>(dim3{kDenseToSparseThreads},
        (__gm__ int32_t *)status, (__gm__ uint64_t *)total);
}

extern "C" __global__ __aicore__ void densetosparse_scan_kernel(
    GM_ADDR values, GM_ADDR chunkSums, uint64_t count, uint64_t chunkCount,
    uint32_t numBlocks)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    asc_vf_call<ScanInclusive>(dim3{kDenseToSparseThreads},
        (__gm__ uint64_t *)values, (__gm__ uint64_t *)chunkSums, count,
        chunkCount, numBlocks);
}

extern "C" __global__ __aicore__ void densetosparse_add_base_kernel(
    GM_ADDR values, GM_ADDR chunkPrefix, uint64_t count, uint32_t numBlocks)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    asc_vf_call<AddChunkBases>(dim3{kDenseToSparseThreads},
        (__gm__ uint64_t *)values, (__gm__ const uint64_t *)chunkPrefix,
        count, numBlocks);
}

extern "C" __global__ __aicore__ void densetosparse_offsets_kernel(
    GM_ADDR prefix, GM_ADDR offsets, GM_ADDR status,
    DenseToSparseTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    uint64_t major = 0;
    uint64_t chunks = 0;
    GetCompressedLayout(tiling, major, chunks);
    if (tiling.offsetType == 0) {
        asc_vf_call<WriteOffsets<int32_t>>(dim3{kDenseToSparseThreads},
            (__gm__ const uint64_t *)prefix, (__gm__ int32_t *)offsets,
            (__gm__ int32_t *)status, major, chunks, tiling.base,
            tiling.numBlocks);
    } else {
        asc_vf_call<WriteOffsets<int64_t>>(dim3{kDenseToSparseThreads},
            (__gm__ const uint64_t *)prefix, (__gm__ int64_t *)offsets,
            (__gm__ int32_t *)status, major, chunks, tiling.base,
            tiling.numBlocks);
    }
}

extern "C" __global__ __aicore__ void densetosparse_total_kernel(
    GM_ADDR prefix, GM_ADDR status, GM_ADDR total, uint64_t count,
    uint64_t expectedNnz, uint32_t base, uint32_t offsetType,
    uint32_t validateExpected)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    asc_vf_call<WriteTotal>(dim3{kDenseToSparseThreads},
        (__gm__ const uint64_t *)prefix, (__gm__ int32_t *)status,
        (__gm__ uint64_t *)total, count, expectedNnz, base, offsetType,
        validateExpected);
}

extern "C" __global__ __aicore__ void densetosparse_validate_offsets_kernel(
    GM_ADDR prefix, GM_ADDR offsets, GM_ADDR status,
    DenseToSparseTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    uint64_t major = 0;
    uint64_t chunks = 0;
    GetCompressedLayout(tiling, major, chunks);
    if (tiling.offsetType == 0) {
        asc_vf_call<ValidateOffsets<int32_t>>(dim3{kDenseToSparseThreads},
            (__gm__ const uint64_t *)prefix, (__gm__ const int32_t *)offsets,
            (__gm__ int32_t *)status, major, chunks, tiling.nnz, tiling.base,
            tiling.numBlocks);
    } else {
        asc_vf_call<ValidateOffsets<int64_t>>(dim3{kDenseToSparseThreads},
            (__gm__ const uint64_t *)prefix, (__gm__ const int64_t *)offsets,
            (__gm__ int32_t *)status, major, chunks, tiling.nnz, tiling.base,
            tiling.numBlocks);
    }
}

extern "C" __global__ __aicore__ void densetosparse_empty_offsets_kernel(
    GM_ADDR offsets, DenseToSparseTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    const uint64_t major = tiling.format == kFormatCsr ?
        tiling.rows : tiling.cols;
    if (tiling.offsetType == 0) {
        asc_vf_call<FillEmptyOffsets<int32_t>>(dim3{kDenseToSparseThreads},
            (__gm__ int32_t *)offsets, major + 1, tiling.base,
            tiling.numBlocks);
    } else {
        asc_vf_call<FillEmptyOffsets<int64_t>>(dim3{kDenseToSparseThreads},
            (__gm__ int64_t *)offsets, major + 1, tiling.base,
            tiling.numBlocks);
    }
}

extern "C" __global__ __aicore__ void densetosparse_convert_kernel(
    GM_ADDR dense, GM_ADDR prefix, GM_ADDR status, GM_ADDR offsets,
    GM_ADDR indices, GM_ADDR rowIndices, GM_ADDR colIndices, GM_ADDR values,
    DenseToSparseTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if (tiling.elementBytes == 1 && tiling.indexType == 0) {
        DispatchConvert<uint8_t, int32_t>(dense, prefix, status, offsets,
            indices, rowIndices, colIndices, values, tiling);
    } else if (tiling.elementBytes == 1) {
        DispatchConvert<uint8_t, int64_t>(dense, prefix, status, offsets,
            indices, rowIndices, colIndices, values, tiling);
    } else if (tiling.elementBytes == 2 && tiling.indexType == 0) {
        DispatchConvert<uint16_t, int32_t>(dense, prefix, status, offsets,
            indices, rowIndices, colIndices, values, tiling);
    } else if (tiling.elementBytes == 2) {
        DispatchConvert<uint16_t, int64_t>(dense, prefix, status, offsets,
            indices, rowIndices, colIndices, values, tiling);
    } else if (tiling.indexType == 0) {
        DispatchConvert<uint32_t, int32_t>(dense, prefix, status, offsets,
            indices, rowIndices, colIndices, values, tiling);
    } else {
        DispatchConvert<uint32_t, int64_t>(dense, prefix, status, offsets,
            indices, rowIndices, colIndices, values, tiling);
    }
}

extern "C" __global__ __aicore__ void densetosparse_bell_kernel(
    GM_ADDR dense, GM_ADDR ellColInd, GM_ADDR values,
    DenseToSparseTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if (tiling.elementBytes == 1 && tiling.indexType == 0) {
        DispatchBell<uint8_t, int32_t>(dense, ellColInd, values, tiling);
    } else if (tiling.elementBytes == 1) {
        DispatchBell<uint8_t, int64_t>(dense, ellColInd, values, tiling);
    } else if (tiling.elementBytes == 2 && tiling.indexType == 0) {
        DispatchBell<uint16_t, int32_t>(dense, ellColInd, values, tiling);
    } else if (tiling.elementBytes == 2) {
        DispatchBell<uint16_t, int64_t>(dense, ellColInd, values, tiling);
    } else if (tiling.indexType == 0) {
        DispatchBell<uint32_t, int32_t>(dense, ellColInd, values, tiling);
    } else {
        DispatchBell<uint32_t, int64_t>(dense, ellColInd, values, tiling);
    }
}

static void LaunchCountAndScan(GM_ADDR dense, GM_ADDR workspace,
    uint32_t numBlocks, const DenseToSparseTilingData &tiling, void *stream)
{
    GM_ADDR status = workspace + tiling.statusOffset;
    GM_ADDR total = workspace + tiling.nnzOffset;
    densetosparse_header_kernel<<<1, nullptr, stream>>>(status, total);
    GM_ADDR level0 = workspace + tiling.level0Offset;
    densetosparse_count_kernel<<<numBlocks, nullptr, stream>>>(
        dense, level0, tiling);
    GM_ADDR levels[16] = {};
    uint64_t counts[16] = {};
    uint32_t levelCount = 1;
    levels[0] = level0;
    counts[0] = tiling.unitCount;
    uint64_t offset = tiling.level0Offset +
        Align32(tiling.unitCount * sizeof(uint64_t));
    while (counts[levelCount - 1] > kScanChunk && levelCount < 16) {
        counts[levelCount] = NextLevelCount(counts[levelCount - 1]);
        levels[levelCount] = workspace + offset;
        offset += Align32(counts[levelCount] * sizeof(uint64_t));
        ++levelCount;
    }
    for (uint32_t level = 0; level < levelCount; ++level) {
        const uint64_t chunks = NextLevelCount(counts[level]);
        GM_ADDR next = level + 1 < levelCount ? levels[level + 1] : nullptr;
        densetosparse_scan_kernel<<<numBlocks, nullptr, stream>>>(
            levels[level], next, counts[level], chunks, numBlocks);
    }
    for (uint32_t level = levelCount - 1; level > 0; --level) {
        densetosparse_add_base_kernel<<<numBlocks, nullptr, stream>>>(
            levels[level - 1], levels[level], counts[level - 1], numBlocks);
    }
}

void densetosparse_analysis_kernel_do(
    GM_ADDR dense, GM_ADDR offsets, GM_ADDR workspace, uint32_t numBlocks,
    const DenseToSparseTilingData &tiling, void *stream)
{
    if (tiling.unitCount == 0) {
        if (tiling.format != kFormatCoo) {
            densetosparse_empty_offsets_kernel<<<numBlocks, nullptr, stream>>>(
                offsets, tiling);
        }
        return;
    }
    LaunchCountAndScan(dense, workspace, numBlocks, tiling, stream);
    GM_ADDR level0 = workspace + tiling.level0Offset;
    GM_ADDR status = workspace + tiling.statusOffset;
    GM_ADDR total = workspace + tiling.nnzOffset;
    if (tiling.format != kFormatCoo) {
        densetosparse_offsets_kernel<<<numBlocks, nullptr, stream>>>(
            level0, offsets, status, tiling);
    }
    densetosparse_total_kernel<<<1, nullptr, stream>>>(
        level0, status, total, tiling.unitCount, 0, tiling.base,
        tiling.offsetType, 0);
}

void densetosparse_convert_kernel_do(
    GM_ADDR dense, GM_ADDR workspace, GM_ADDR offsets, GM_ADDR indices,
    GM_ADDR rowIndices, GM_ADDR colIndices, GM_ADDR values, GM_ADDR ellColInd,
    uint32_t numBlocks, const DenseToSparseTilingData &tiling, void *stream)
{
    if (tiling.format == kFormatBell) {
        densetosparse_bell_kernel<<<numBlocks, nullptr, stream>>>(
            dense, ellColInd, values, tiling);
        return;
    }
    LaunchCountAndScan(dense, workspace, numBlocks, tiling, stream);
    GM_ADDR status = workspace + tiling.statusOffset;
    GM_ADDR total = workspace + tiling.nnzOffset;
    GM_ADDR level0 = workspace + tiling.level0Offset;
    densetosparse_total_kernel<<<1, nullptr, stream>>>(
        level0, status, total, tiling.unitCount, tiling.nnz, tiling.base,
        tiling.offsetType, 1);
    if (tiling.format != kFormatCoo) {
        densetosparse_validate_offsets_kernel<<<numBlocks, nullptr, stream>>>(
            level0, offsets, status, tiling);
    }
    densetosparse_convert_kernel<<<numBlocks, nullptr, stream>>>(
        dense, level0, status, offsets, indices, rowIndices,
        colIndices, values, tiling);
}
