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

#ifndef SPMV_ARCH35_H_
#define SPMV_ARCH35_H_

#include <cstddef>
#include <cstdint>

#include "cann_ops_sparse.h"

constexpr uint32_t SPMV_MAX_SIMT_THREADS = 512u;
constexpr uint32_t SPMV_SIMT_WARP_SIZE = 32u;
constexpr size_t SPMV_WS_ALIGNMENT = 64u;
constexpr size_t SPMV_WS_HEADER_BYTES = 256u;

enum SpmvDataType : int32_t
{
    SPMV_DTYPE_FP32_FP32 = 0,
    SPMV_DTYPE_INT8_INT32 = 1,
    SPMV_DTYPE_INT8_FP32 = 2,
    SPMV_DTYPE_FP16_FP32 = 3,
    SPMV_DTYPE_FP16_FP16 = 4,
    SPMV_DTYPE_BF16_FP32 = 5,
    SPMV_DTYPE_BF16_BF16 = 6,
};

struct SpmvTilingData {
    int32_t rows;
    int32_t cols;
    int32_t nnz;
    int32_t dataType;
    int32_t transpose;
    int32_t computeType;
    uint32_t workPerBlock;
    int64_t rowOffsetsStride;
    int64_t colIndStride;
    int64_t valuesStride;
    int64_t xStride;
    int64_t yStride;
    float alphaFloat;
    float betaFloat;
    int32_t alphaInt;
    int32_t betaInt;
    uint64_t alphaDevicePtr;
    uint64_t betaDevicePtr;
};

struct SpmvWorkspaceLayout {
    size_t colOffsetsOffset{0};
    size_t nextOffset{0};
    size_t rowIndicesOffset{0};
    size_t permutationOffset{0};
    size_t totalBytes{0};

    static inline size_t AlignUp(size_t value)
    {
        return (value + SPMV_WS_ALIGNMENT - 1u) & ~(SPMV_WS_ALIGNMENT - 1u);
    }

    inline void Compute(int32_t cols, int32_t nnz)
    {
        colOffsetsOffset = SPMV_WS_HEADER_BYTES;
        nextOffset = AlignUp(colOffsetsOffset + (static_cast<size_t>(cols) + 1u) * sizeof(int32_t));
        rowIndicesOffset = AlignUp(nextOffset + static_cast<size_t>(cols) * sizeof(int32_t));
        permutationOffset = AlignUp(rowIndicesOffset + static_cast<size_t>(nnz) * sizeof(int32_t));
        totalBytes = AlignUp(permutationOffset + static_cast<size_t>(nnz) * sizeof(int32_t));
    }
};

#endif // SPMV_ARCH35_H_
