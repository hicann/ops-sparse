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
 * \file gather_tiling_data.h
 * \brief gather TilingData 结构体定义（Host / Kernel 共用）。
 *
 * Gather:  X.values[i] = Y[X.indices[i] - idxBase]  for i = 0 .. nnz-1.
 */

#pragma once

#include <cstdint>
#include "acl/acl_base_rt.h"
#include "cann_ops_sparse.h"

// 每 block 最大线程数（SIMT VF launch_bounds）
constexpr uint32_t kGatherMaxThreadsPerBlock = 256;

// Gather TilingData（Host / Kernel 共用）
struct GatherTilingData {
    int64_t nnz; // 稀疏向量非零元个数
    uint32_t numBlocks;
    aclDataType valType;          // 值类型内部 ID
    aclsparseIndexType_t idxType; // 索引类型内部 ID
    aclsparseIndexBase_t idxBase; // 索引基址 (0 或 1)
};
