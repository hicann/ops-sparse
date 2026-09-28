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

// GM_ADDR: NPU 侧由 Ascend C toolkit（kernel_utils_macros.h）定义为 __gm__ uint8_t*。
// Host 侧不重定义宏，仅用类型别名回退，避免与 toolkit 抢 #define。
#ifndef GM_ADDR
using GM_ADDR = uint8_t *;
#endif

extern "C" {

/// SparseToDense：同 stream 先 SIMT 清零 dense，再 scatter 非零元。
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
