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
 * \file spmv_op_kernel.h
 * \brief spmv_op kernel_do 签名声明（Host / Kernel 共用）。
 *
 * 声明主计算 kernel 入口函数 spmv_op_kernel_do，同时声明 nnz=0 时的
 * 简化 beta*y kernel 入口函数。
 */

#ifndef SPMV_OP_KERNEL_H_
#define SPMV_OP_KERNEL_H_

#include <cstdint>
#include "spmv_op_tiling_data.h"

// GM_ADDR: 由 Ascend C toolkit 标准定义（kernel_utils_macros.h）。
// NPU 侧为 __gm__ uint8_t*（携带全局内存地址空间属性），Host 侧为 uint8_t*。

extern "C" {

/// 主 SpMVOp kernel：Z = alpha * A * X + beta * Y，含 Priest 双重补偿求和
void spmv_op_kernel_do(
    GM_ADDR rowOffsets,     // CSR 行偏移 (int32_t* 或 int64_t*)
    GM_ADDR colInd,         // CSR 列索引 (int32_t*)
    GM_ADDR values,         // CSR 值 (float*)
    GM_ADDR xVec,           // 输入稠密向量 X (float*)
    GM_ADDR yVec,           // 输入稠密向量 Y (float*)
    GM_ADDR zVec,           // 输出稠密向量 Z (float*)
    GM_ADDR reorder,        // ALG2 专用: workspace 内 reorder 表 (ALG1: nullptr)
    GM_ADDR binEdge,        // ALG2 专用: workspace 内 bin_edge[numBlocks+1] 表 (ALG1: nullptr)
    const SpmvOpTilingData &tiling,
    uint32_t numBlocks,
    void *stream);

/// 简化 kernel（nnz == 0 时使用）：Z = beta * Y
void spmv_op_beta_y_kernel_do(
    GM_ADDR yVec,           // 输入稠密向量 Y (float*)
    GM_ADDR zVec,           // 输出稠密向量 Z (float*)
    const SpmvOpTilingData &tiling,
    uint32_t numBlocks,
    void *stream);

/// ALG2 预处理 kernel：在 device 上完成 rowOffsets 读取、按 nnz 降序排序和
/// bin_edge 负载均衡切分，将 reorder 和 bin_edge 写入 workspace。
/// 仅 launch 1 个 block（单线程串行完成 O(m log m) 归并排序）。
///
/// \param rowOffsets  CSR rowOffsets（int32_t* 或 int64_t*，由 rowOffsetType 决定）
/// \param reorder     输出: workspace 内 reorder[m] 缓冲区（int32_t*）
/// \param tmpReorder  输出: workspace 内 tmpReorder[m] 缓冲区（归并排序临时缓冲）
/// \param scratch     输出: workspace 内 scratch[m] 缓冲区（int32_t*）
/// \param tmpScratch  输出: workspace 内 tmpScratch[m] 缓冲区（归并排序临时缓冲）
/// \param binEdge     输出: workspace 内 bin_edge[numBlocks+1] 缓冲区（int32_t*）
/// \param m           矩阵行数
/// \param numBlocks   block 数（用于计算 bin_edge）
/// \param rowOffsetType  0 = int32，1 = int64（对应 SPMV_OP_IDX_RT_I32 / I64）
/// \param stream      ACL stream（异步执行）
void spmv_op_preprocess_kernel_do(
    GM_ADDR rowOffsets,
    GM_ADDR reorder,
    GM_ADDR tmpReorder,
    GM_ADDR scratch,
    GM_ADDR tmpScratch,
    GM_ADDR binEdge,
    int32_t m,
    uint32_t numBlocks,
    int32_t rowOffsetType,
    void *stream);

}  // extern "C"

#endif  // SPMV_OP_KERNEL_H_
