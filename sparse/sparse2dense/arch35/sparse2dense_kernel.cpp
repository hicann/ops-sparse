/**
 * ----------------------------------------------------------------------------------------------------------
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under
 * the terms and conditions of CANN Open Software License Agreement Version 2.0
 * (the "License"). Please refer to the License for details. You may not use
 * this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
 * AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
 * FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
 * for the full text of the License.
 * ----------------------------------------------------------------------------------------------------------
 */

/*!
 * \file sparse2dense_kernel.cpp
 * \brief SparseToDense arch35 SIMT（对标 gather/scatter/densetosparse）。
 *
 * Host 同 stream 先 launch zero kernel（uint64 grid-stride），再 launch scatter。
 * CSR/CSC/COO 均按 nnz grid-stride 写非零元（__gm__ 直写）；CSR/CSC 用 ptr
 * 二分定位所属行/列，避免中等密度时按行/列并行度不足。
 *
 * 本路径为 SIMT GM 直写，不走 DataCopyPad MTE2/MTE3 UB 中转，因此无需
 * SetFlag/WaitFlag HardEvent::MTE2_MTE3（该检视意见适用于 arch22 EmitRun）。
 */

#include <cstdint>
#include "kernel_operator.h"
#include "simt_api/asc_simt.h"
#include "sparse2dense_kernel.h"

namespace {

__simt_callee__ __aicore__ inline uint64_t DenseOff(
    int32_t r, int32_t c, int32_t ld, int32_t isColMajor)
{
    return isColMajor != 0
        ? static_cast<uint64_t>(c) * static_cast<uint64_t>(ld) + static_cast<uint64_t>(r)
        : static_cast<uint64_t>(r) * static_cast<uint64_t>(ld) + static_cast<uint64_t>(c);
}

/** Largest seg in [0, nSeg) with ptr[seg]-base <= p (nnz index). */
__simt_callee__ __aicore__ inline int32_t FindSegByNnz(
    __gm__ const int32_t *ptr, int32_t nSeg, int32_t base, int32_t p)
{
    int32_t lo = 0;
    int32_t hi = nSeg - 1;
    while (lo < hi) {
        // Avoid int32 overflow of (lo + hi + 1) when nSeg is large (e.g. m~1.5e9).
        const int32_t mid = lo + ((hi - lo + 1) >> 1);
        if (ptr[mid] - base <= p) {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    return lo;
}

__simt_vf__ __aicore__ __launch_bounds__(kSparse2DenseMaxThreadsPerBlock) inline void
Sparse2DenseZeroU64(
    __gm__ uint64_t *denseWords, uint64_t nWords, uint32_t numBlocks)
{
    const uint64_t tid =
        static_cast<uint64_t>(blockIdx.x) * static_cast<uint64_t>(blockDim.x) +
        static_cast<uint64_t>(threadIdx.x);
    const uint64_t stride =
        static_cast<uint64_t>(numBlocks) * static_cast<uint64_t>(blockDim.x);
    for (uint64_t i = tid; i < nWords; i += stride) {
        denseWords[i] = 0ULL;
    }
}

__simt_vf__ __aicore__ __launch_bounds__(kSparse2DenseMaxThreadsPerBlock) inline void
Sparse2DenseZeroU8(
    __gm__ uint8_t *denseBytes, uint64_t nBytes, uint32_t numBlocks)
{
    const uint64_t tid =
        static_cast<uint64_t>(blockIdx.x) * static_cast<uint64_t>(blockDim.x) +
        static_cast<uint64_t>(threadIdx.x);
    const uint64_t stride =
        static_cast<uint64_t>(numBlocks) * static_cast<uint64_t>(blockDim.x);
    for (uint64_t i = tid; i < nBytes; i += stride) {
        denseBytes[i] = 0;
    }
}

template <typename ValT>
__simt_vf__ __aicore__ __launch_bounds__(kSparse2DenseMaxThreadsPerBlock) inline void
Sparse2DenseCsrSimt(
    __gm__ const int32_t *rowPtr, __gm__ const int32_t *colInd,
    __gm__ const ValT *values, __gm__ ValT *dense,
    int32_t rows, uint64_t nnz, int32_t ld, int32_t indexBase, int32_t isColMajor,
    uint32_t numBlocks)
{
    const uint64_t tid =
        static_cast<uint64_t>(blockIdx.x) * static_cast<uint64_t>(blockDim.x) +
        static_cast<uint64_t>(threadIdx.x);
    const uint64_t stride =
        static_cast<uint64_t>(numBlocks) * static_cast<uint64_t>(blockDim.x);
    const int32_t base = indexBase;
    for (uint64_t p = tid; p < nnz; p += stride) {
        const int32_t row = FindSegByNnz(rowPtr, rows, base, static_cast<int32_t>(p));
        const int32_t c = colInd[p] - base;
        dense[DenseOff(row, c, ld, isColMajor)] = values[p];
    }
}

template <typename ValT>
__simt_vf__ __aicore__ __launch_bounds__(kSparse2DenseMaxThreadsPerBlock) inline void
Sparse2DenseCscSimt(
    __gm__ const int32_t *colPtr, __gm__ const int32_t *rowInd,
    __gm__ const ValT *values, __gm__ ValT *dense,
    int32_t cols, uint64_t nnz, int32_t ld, int32_t indexBase, int32_t isColMajor,
    uint32_t numBlocks)
{
    const uint64_t tid =
        static_cast<uint64_t>(blockIdx.x) * static_cast<uint64_t>(blockDim.x) +
        static_cast<uint64_t>(threadIdx.x);
    const uint64_t stride =
        static_cast<uint64_t>(numBlocks) * static_cast<uint64_t>(blockDim.x);
    const int32_t base = indexBase;
    for (uint64_t p = tid; p < nnz; p += stride) {
        const int32_t col = FindSegByNnz(colPtr, cols, base, static_cast<int32_t>(p));
        const int32_t r = rowInd[p] - base;
        dense[DenseOff(r, col, ld, isColMajor)] = values[p];
    }
}

// COO duplicates: element store is last-write-wins (cuSPARSE-compatible).
// No DUPLICATE status / host sync-back on the Generic execute path.
template <typename ValT>
__simt_vf__ __aicore__ __launch_bounds__(kSparse2DenseMaxThreadsPerBlock) inline void
Sparse2DenseCooSimt(
    __gm__ const int32_t *rowInd, __gm__ const int32_t *colInd,
    __gm__ const ValT *values, __gm__ ValT *dense,
    uint64_t nnz, int32_t ld, int32_t indexBase, int32_t isColMajor, uint32_t numBlocks)
{
    const uint64_t tid =
        static_cast<uint64_t>(blockIdx.x) * static_cast<uint64_t>(blockDim.x) +
        static_cast<uint64_t>(threadIdx.x);
    const uint64_t stride =
        static_cast<uint64_t>(numBlocks) * static_cast<uint64_t>(blockDim.x);
    const int32_t base = indexBase;
    for (uint64_t p = tid; p < nnz; p += stride) {
        const int32_t r = rowInd[p] - base;
        const int32_t c = colInd[p] - base;
        dense[DenseOff(r, c, ld, isColMajor)] = values[p];
    }
}

template <typename ValT>
__aicore__ inline void DispatchByFormat(
    GM_ADDR gmOffsets, GM_ADDR gmIndices, GM_ADDR gmValues, GM_ADDR gmDense,
    const Sparse2DenseTilingData &tiling)
{
    auto *offsets = reinterpret_cast<__gm__ const int32_t *>(gmOffsets);
    auto *indices = reinterpret_cast<__gm__ const int32_t *>(gmIndices);
    auto *values = reinterpret_cast<__gm__ const ValT *>(gmValues);
    auto *dense = reinterpret_cast<__gm__ ValT *>(gmDense);
    if (tiling.format == SPARSE2DENSE_FMT_CSC) {
        asc_vf_call<Sparse2DenseCscSimt<ValT>>(
            dim3{kSparse2DenseMaxThreadsPerBlock},
            offsets, indices, values, dense,
            tiling.n, tiling.nnz, tiling.ld, tiling.indexBase, tiling.isColMajor,
            tiling.numBlocks);
    } else if (tiling.format == SPARSE2DENSE_FMT_COO) {
        asc_vf_call<Sparse2DenseCooSimt<ValT>>(
            dim3{kSparse2DenseMaxThreadsPerBlock},
            offsets, indices, values, dense,
            tiling.nnz, tiling.ld, tiling.indexBase, tiling.isColMajor, tiling.numBlocks);
    } else {
        asc_vf_call<Sparse2DenseCsrSimt<ValT>>(
            dim3{kSparse2DenseMaxThreadsPerBlock},
            offsets, indices, values, dense,
            tiling.m, tiling.nnz, tiling.ld, tiling.indexBase, tiling.isColMajor,
            tiling.numBlocks);
    }
}

} // namespace

extern "C" __global__ __aicore__ void sparse2dense_zero_kernel(
    GM_ADDR gmDense, const Sparse2DenseTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    const uint64_t bytes = tiling.denseBytes;
    if (bytes == 0) {
        return;
    }
    const uint64_t nWords = bytes >> 3;
    if (nWords > 0) {
        asc_vf_call<Sparse2DenseZeroU64>(
            dim3{kSparse2DenseMaxThreadsPerBlock},
            reinterpret_cast<__gm__ uint64_t *>(gmDense), nWords, tiling.numBlocks);
    }
    const uint64_t remOff = nWords << 3;
    const uint64_t rem = bytes - remOff;
    if (rem > 0) {
        asc_vf_call<Sparse2DenseZeroU8>(
            dim3{kSparse2DenseMaxThreadsPerBlock},
            reinterpret_cast<__gm__ uint8_t *>(gmDense) + remOff, rem, tiling.numBlocks);
    }
}

extern "C" __global__ __aicore__ void sparse2dense_kernel(
    GM_ADDR gmOffsets, GM_ADDR gmIndices, GM_ADDR gmValues, GM_ADDR gmDense,
    const Sparse2DenseTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if (tiling.valueType == SPARSE2DENSE_VAL_F16) {
        DispatchByFormat<half>(gmOffsets, gmIndices, gmValues, gmDense, tiling);
    } else if (tiling.valueType == SPARSE2DENSE_VAL_BF16) {
        DispatchByFormat<bfloat16_t>(gmOffsets, gmIndices, gmValues, gmDense, tiling);
    } else if (tiling.valueType == SPARSE2DENSE_VAL_I32) {
        DispatchByFormat<int32_t>(gmOffsets, gmIndices, gmValues, gmDense, tiling);
    } else if (tiling.valueType == SPARSE2DENSE_VAL_I8) {
        DispatchByFormat<int8_t>(gmOffsets, gmIndices, gmValues, gmDense, tiling);
    } else if (tiling.valueType == SPARSE2DENSE_VAL_COMPLEX64) {
        DispatchByFormat<uint64_t>(gmOffsets, gmIndices, gmValues, gmDense, tiling);
    } else {
        DispatchByFormat<float>(gmOffsets, gmIndices, gmValues, gmDense, tiling);
    }
}

extern "C" void sparse2dense_kernel_do(
    GM_ADDR sparseOffsets, GM_ADDR sparseIndices, GM_ADDR sparseValues, GM_ADDR dense,
    const Sparse2DenseTilingData &tiling, uint32_t numBlocks, void *stream)
{
    // Same-stream ordering: all AIVs finish zero before any scatter starts.
    if (tiling.denseBytes > 0) {
        sparse2dense_zero_kernel<<<numBlocks, nullptr, stream>>>(dense, tiling);
    }
    sparse2dense_kernel<<<numBlocks, nullptr, stream>>>(
        sparseOffsets, sparseIndices, sparseValues, dense, tiling);
}
