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
 * \file spmm_op_kernel.h
 * \brief spmm_op kernel_do 签名声明（Host / Kernel 共用）。
 *
 * 声明 2 个 kernel 入口函数：
 *   Kernel 1 (主计算):  C = alpha * op(A) * op(B) + beta * C
 *   Kernel 2 (BetaC):   nnz==0 时，C = beta * C
 *
 * ALG2 preprocess 在 host 侧完成（见 spmm_op_host.cpp 的 CreateDescrPreprocessAlg2），
 * 不使用 device-side kernel。
 */

#ifndef SPMM_OP_KERNEL_H_
#define SPMM_OP_KERNEL_H_

#include <cstdint>
#include "spmm_op_tiling_data.h"

// GM_ADDR: 由 Ascend C toolkit 标准定义（kernel_utils_macros.h）。
// NPU 侧为 __gm__ uint8_t*（携带全局内存地址空间属性），Host 侧为 uint8_t*。

extern "C" {
/// 主 SpMMOp kernel：C = alpha * op(A) * op(B) + beta * C
/// \param csrRowOffsets  CSR 行偏移 (int32_t* 或 int64_t*，由 tiling.rowOffsetType 决定)
/// \param csrColInd      CSR 列索引 (int32_t*)
/// \param csrValues      CSR 值 (float* 或 __fp16*，由 tiling.dtype 决定)
/// \param matB           输入稠密矩阵 B (float* 或 __fp16*)
/// \param matCOut       输入/输出稠密矩阵 C (float* 或 __fp16*，in-place)
/// \param reorder        ALG2 专用: workspace 内 reorder 表 (ALG1: nullptr)
/// \param binEdge        ALG2 专用: workspace 内 bin_edge[numBlocks+1] 表 (ALG1: nullptr)
/// \param tiling         Tiling 数据（const 引用传入，launch 时 by value 拷贝至 kernel 参数）
/// \param numBlocks      block 数
/// \param stream         ACL stream（异步执行）
void spmm_op_kernel_do(
    GM_ADDR csrRowOffsets,
    GM_ADDR csrColInd,
    GM_ADDR csrValues,
    GM_ADDR matB,
    GM_ADDR matCOut,
    GM_ADDR reorder,
    GM_ADDR binEdge,
    const SpmmOpTilingData &tiling,
    uint32_t numBlocks,
    void *stream);

/// 简化 kernel（nnz == 0 时使用）：C = beta * C
/// \param matCOut  输入/输出稠密矩阵 C (float* 或 __fp16*，in-place)
/// \param tiling    Tiling 数据
/// \param numBlocks block 数
/// \param stream    ACL stream（异步执行）
void spmm_op_beta_c_kernel_do(
    GM_ADDR matCOut,
    const SpmmOpTilingData &tiling,
    uint32_t numBlocks,
    void *stream);

}  // extern "C"

#endif  // SPMM_OP_KERNEL_H_
