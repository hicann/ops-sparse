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
 * \file sparse2dense_kernel.h
 * \brief SparseToDense kernel_do 签名声明（Host / Kernel 共用）。
 *
 * 声明主转换 kernel 入口函数 sparse2dense_kernel_do。
 */

#ifndef SPARSE2DENSE_KERNEL_H_
#define SPARSE2DENSE_KERNEL_H_

#include <cstdint>
#include "sparse2dense_tiling_data.h"

// GM_ADDR: 由 Ascend C toolkit 标准定义（kernel_utils_macros.h）。
// NPU 侧为 __gm__ uint8_t*（携带全局内存地址空间属性），Host 侧为 uint8_t*。

extern "C" {

/// SparseToDense kernel：将稀疏矩阵的非零元 scatter 到稠密矩阵。
/// 输出稠密矩阵需在调用前由 host 侧 memset 为 0。
void sparse2dense_kernel_do(
    GM_ADDR sparseOffsets,  // CSR:rowOffsets / CSC:colOffsets / COO:cooRowInd (int32_t*)
    GM_ADDR sparseIndices,  // CSR:colInd / CSC:rowInd / COO:cooColInd (int32_t*)
    GM_ADDR sparseValues,   // 非零元值 (ValT*)
    GM_ADDR dense,          // 输出稠密矩阵 (ValT*)
    const Sparse2DenseTilingData &tiling,
    uint32_t numBlocks,
    void *stream);

}  // extern "C"

#endif  // SPARSE2DENSE_KERNEL_H_
