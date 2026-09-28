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
 * \file sparse2dense_tiling_data.h
 * \brief SparseToDense TilingData（Host / Kernel 共用）— arch35 SIMT。
 */

#ifndef SPARSE2DENSE_TILING_DATA_H_
#define SPARSE2DENSE_TILING_DATA_H_

#include <cstdint>

// Prefer more threads per AIV to saturate HBM on dense zero + nnz scatter.
constexpr uint32_t kSparse2DenseMaxThreadsPerBlock = 1024;

constexpr int32_t SPARSE2DENSE_VAL_F32 = 0;
constexpr int32_t SPARSE2DENSE_VAL_F16 = 1;
constexpr int32_t SPARSE2DENSE_VAL_BF16 = 2;
constexpr int32_t SPARSE2DENSE_VAL_I32 = 3;
constexpr int32_t SPARSE2DENSE_VAL_I8 = 4;
constexpr int32_t SPARSE2DENSE_VAL_COMPLEX64 = 5; // bit-copy as uint64_t

constexpr int32_t SPARSE2DENSE_FMT_CSR = 0;
constexpr int32_t SPARSE2DENSE_FMT_CSC = 1;
constexpr int32_t SPARSE2DENSE_FMT_COO = 2;

struct Sparse2DenseTilingData {
    int32_t m;
    int32_t n;
    int32_t indexBase;
    int32_t valueType;
    int32_t isColMajor;
    int32_t ld;
    int32_t format;
    uint64_t nnz;
    uint64_t denseBytes; ///< full dense storage bytes (kernel zeros as uint64 words)
    uint32_t numBlocks;
};

#endif  // SPARSE2DENSE_TILING_DATA_H_
