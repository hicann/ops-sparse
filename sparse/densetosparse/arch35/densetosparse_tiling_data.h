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

#ifndef DENSETOSPARSE_TILING_DATA_H_
#define DENSETOSPARSE_TILING_DATA_H_

#include <cstdint>

constexpr uint32_t kDenseToSparseThreads = 256;

struct DenseToSparseTilingData {
    uint64_t rows;
    uint64_t cols;
    uint64_t ld;
    uint64_t nnz;
    uint64_t ellBlockSize;
    uint64_t ellCols;
    uint64_t unitCount;
    uint64_t statusOffset;
    uint64_t nnzOffset;
    uint64_t level0Offset;
    uint32_t format;
    uint32_t order;
    uint32_t base;
    uint32_t offsetType;
    uint32_t indexType;
    uint32_t elementBytes;
    uint32_t numBlocks;
};

#endif // DENSETOSPARSE_TILING_DATA_H_
