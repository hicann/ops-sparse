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
 * \file nnz_tiling_data.h
 * \brief aclsparseSnnz TilingData 结构体定义（Host / Kernel 共用）。
 */

#ifndef NNZ_TILING_DATA_H_
#define NNZ_TILING_DATA_H_

#include <cstdint>

constexpr uint32_t kNnzThreadsPerBlock = 64;

struct NnzTilingData {
    int32_t lda;         // leading dimension
    int32_t dirA;        // 0 = ROW, 1 = COLUMN
    int32_t totalUnits;  // 待处理 unit 总数：m (ROW) 或 n (COLUMN)
    int32_t unitSize;    // 每个 unit 的元素数：n (ROW) 或 m (COLUMN)
};

#endif // NNZ_TILING_DATA_H_
