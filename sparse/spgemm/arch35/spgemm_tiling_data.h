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

#ifndef SPGEMM_TILING_DATA_H_
#define SPGEMM_TILING_DATA_H_

#include <cstdint>

// A5 SIMT supports 256 threads per AIV block.  Eight warps per block keep the
// row-parallel small-product path saturated once LaunchBlocks reaches the AIV
// core-count cap on medium and large matrices.
constexpr uint32_t kSpGemmThreads = 256U;
constexpr uint32_t kSpGemmWarpSize = 32U;
constexpr uint32_t kSpGemmScanChunk = 1024U;

enum SpGemmValueType : int32_t {
    SPGEMM_VAL_FP16 = 0,
    SPGEMM_VAL_BF16,
    SPGEMM_VAL_FP32,
    SPGEMM_VAL_COMPLEX64
};

struct SpGemmValidateTilingData {
    int32_t m;
    int32_t k;
    int32_t n;
    int32_t nnzA;
    int32_t nnzB;
    int32_t regularDegree;
    uint32_t rowsPerBlock;
};

struct SpGemmWorkTilingData {
    int32_t m;
    int32_t k;
    int32_t nnzA;
    int32_t nnzB;
    uint32_t rowsPerBlock;
};

struct SpGemmScanTilingData {
    int32_t count;
    int32_t numChunks;
    int32_t chunkSize;
    uint32_t outerBlocks;
};

struct SpGemmComputeTilingData {
    int32_t m;
    int32_t k;
    int32_t n;
    int32_t nnzA;
    int32_t nnzB;
    int32_t valType;
    int32_t regularDegree;
    int32_t directOutput;
    uint32_t rowsPerBlock;
    float alphaReal;
    float alphaImag;
    float betaReal;
    float betaImag;
    uint64_t alphaPtr;
    uint64_t betaPtr;
};

struct SpGemmCopyTilingData {
    int32_t m;
    int32_t valType;
    int32_t numProducts;
    int32_t nnzC;
    uint32_t rowsPerBlock;
    uint32_t numBlocks;
};

#endif  // SPGEMM_TILING_DATA_H_
