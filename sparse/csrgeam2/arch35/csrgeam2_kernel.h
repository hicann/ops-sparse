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
 * \file csrgeam2_kernel.h
 * \brief csrgeam2 kernel_do 签名声明（Host / Kernel 共用）。
 *
 * 声明三个 kernel_do 函数：
 *   - csrgeam2_nnz_kernel_do:       Kernel 1，逐行计数 A∪B 非零元
 *   - csrgeam2_prefixsum_kernel_do: Kernel 3，device 侧 exclusive prefix sum
 *   - csrgeam2_compute_kernel_do:   Kernel 2，逐行有序归并 C = α·A + β·B
 */

#ifndef CSRGEAM2_KERNEL_H_
#define CSRGEAM2_KERNEL_H_

#include <cstdint>
#include "csrgeam2_tiling_data.h"

#ifndef GM_ADDR
#define GM_ADDR uint8_t *
#endif

extern "C" {

/// Kernel 1 (Nnz 计数): 逐行双指针归并，计算每行 nnzC 并写入 nnzPerRow
void csrgeam2_nnz_kernel_do(
    GM_ADDR rowPtrA, GM_ADDR colIndA,
    GM_ADDR rowPtrB, GM_ADDR colIndB,
    GM_ADDR nnzPerRow,
    const Csrgeam2NnzTilingData &tiling,
    uint32_t numBlocks,
    void *stream);

/// Kernel 2 (主计算): 逐行双指针归并，输出 csrColIndC + csrValC
void csrgeam2_compute_kernel_do(
    GM_ADDR rowPtrA, GM_ADDR colIndA, GM_ADDR valA,
    GM_ADDR rowPtrB, GM_ADDR colIndB, GM_ADDR valB,
    GM_ADDR rowPtrC, GM_ADDR colIndC, GM_ADDR valC,
    const Csrgeam2TilingData &tiling,
    uint32_t numBlocks,
    void *stream);

/// Kernel 3 (Prefix Sum): device 侧 exclusive prefix sum + 写 nnzC
void csrgeam2_prefixsum_kernel_do(
    GM_ADDR buffer, GM_ADDR rowPtrC, GM_ADDR nnzCDev,
    const Csrgeam2PrefixSumTilingData &tiling,
    void *stream);

}  // extern "C"

#endif  // CSRGEAM2_KERNEL_H_
