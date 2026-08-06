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

#ifndef SCATTER_H
#define SCATTER_H

#include "cann_ops_sparse.h"
#include "aclsparse_descr_internal.h"

#include <stdint.h>
#include <unistd.h>

#ifndef __gm__
#define __gm__
#endif

/// 将 const SpVec 描述符转为内部结构体指针（解除 const，仅读用途）。
/// vecX 按 aclsparseConstSpVecDescr_t 声明（const），const 解除统一在此完成，
/// 禁止在业务代码中直接 const_cast（与 arch35 风格一致）。
inline struct aclsparseSpVecDescr *ScatterToSpVecInner(aclsparseConstSpVecDescr_t desc)
{
    return const_cast<struct aclsparseSpVecDescr *>(
        reinterpret_cast<const struct aclsparseSpVecDescr *>(desc));
}

// Maximum number of cores for tiling
constexpr uint32_t SCATTER_MAX_CORE_NUM = 64;

// Maximum nnz elements per tile (UB capacity constraint)
constexpr uint32_t SCATTER_TILE_NN_MAX = 4096;

// Number of buffers per queue (2 = DoubleBuffer, overlap MTE2 with VEC+MTE3)
constexpr uint32_t SCATTER_BUFFER_NUM = 2;

// Density above this threshold triggers sort
constexpr float SCATTER_DENSITY_THRESHOLD = 0.3f;

// Tiling data structure passed to the kernel
struct ScatterTilingData {
    uint32_t nnz;                                      // total non-zero elements
    uint32_t blockNum;                                  // actual number of cores used
    uint32_t tileNn;                                    // elements per tile (always power of 2)
    uint32_t tileNnAligned8;                            // tileNn round up to 8 (float 32B align / DataCopyPad 8-elem block)
    uint32_t tileNnAligned4;                            // tileNn round up to 4 (int32_t 16B align)
    uint64_t yLen;                                      // Y vector length in bytes
    uint32_t coreNnzOffset[SCATTER_MAX_CORE_NUM];      // per-core start offset
    uint32_t coreNnzCount[SCATTER_MAX_CORE_NUM];       // per-core element count
};

#endif
