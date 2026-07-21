/**
 * ----------------------------------------------------------------------------------------------------------
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ----------------------------------------------------------------------------------------------------------
 */

/*!
 * \file coosort_kernel.h
 * \brief coosort kernel_do 签名声明（Host / Kernel 共用）。
 *
 * 同时声明单核直排 kernel 和多核多 run kernel。单个 run 均使用两趟
 * LSB radix 稳定排序（int32 signed Sort 原生支持负数）和向量化 Gather。
 */

#ifndef COOSORT_KERNEL_H_
#define COOSORT_KERNEL_H_

#include <cstdint>
#include "coosort_tiling_data.h"

#ifndef GM_ADDR
#define GM_ADDR uint8_t *
#endif

#ifdef __cplusplus
extern "C" {
#endif

/// 启动单核直排 kernel；该路径直接写回 row、col、P，不参与多核 Phase 1。
void coosort_kernel_do(
    GM_ADDR row, GM_ADDR col, GM_ADDR pOut, const CoosortTilingData &tiling, uint32_t numBlocks, void *stream);

/// 启动多核多 run kernel，依次执行 run 生成、树形归并和并行输出。
void coosort_multi_core_kernel_do(GM_ADDR row, GM_ADDR col, GM_ADDR pOut, GM_ADDR workspace,
    const CoosortTilingData &tiling, uint32_t numBlocks, void *stream);

#ifdef __cplusplus
}
#endif

#endif  // COOSORT_KERNEL_H_
