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
 * \file coo2csr_kernel.h
 * \brief coo2csr kernel_do 签名声明（Host / Kernel 共用）。
 *
 * 声明各 kernel_do 函数：
 *   - coo2csr_count_kernel_do:               Kernel 1，行计数（RLE + 原子加）
 *   - coo2csr_prefixsum_kernel_do:           Kernel 2 单核，exclusive prefix sum
 *   - coo2csr_prefixsum_local_kernel_do:     Kernel 2 Phase A，局部前缀和
 *   - coo2csr_prefixsum_blocks_kernel_do:    Kernel 2 Phase B，block 总和前缀和
 *   - coo2csr_prefixsum_correct_kernel_do:   Kernel 2 Phase C，偏移修正
 *   - coo2csr_fused_kernel_do:               融合 kernel，单 block 完成 Count + PrefixSum
 */

#ifndef COO2CSR_KERNEL_H_
#define COO2CSR_KERNEL_H_

#include <cstdint>
#include "coo2csr_tiling_data.h"

#ifndef GM_ADDR
using GM_ADDR = uint8_t *;
#endif

extern "C" {

/// Kernel 1 (CountRows): 遍历 cooRowInd，RLE + asc_atomic_add 统计每行 nnz
void coo2csr_count_kernel_do(
    GM_ADDR cooRowInd,
    GM_ADDR workspace,
    const Coo2CsrCountTilingData &tiling,
    uint32_t numBlocks,
    void *stream);

/// Kernel 2 (PrefixSum, 单核): exclusive prefix sum -> csrRowPtr
void coo2csr_prefixsum_kernel_do(
    GM_ADDR workspace,
    GM_ADDR csrRowPtr,
    const Coo2CsrPrefixSumTilingData &tiling,
    void *stream);

/// Kernel 2 Phase A (PrefixSum Local): 各 block 计算局部前缀和，写出 blockTotals
void coo2csr_prefixsum_local_kernel_do(
    GM_ADDR workspace,
    GM_ADDR csrRowPtr,
    GM_ADDR blockTotals,
    const Coo2CsrPrefixSumLocalTilingData &tiling,
    uint32_t numBlocks,
    void *stream);

/// Kernel 2 Phase B (PrefixSum Blocks): 单 block 对 blockTotals 做排他前缀和
void coo2csr_prefixsum_blocks_kernel_do(
    GM_ADDR blockTotals,
    const Coo2CsrPrefixSumBlocksTilingData &tiling,
    void *stream);

/// Kernel 2 Phase C (PrefixSum Correct): 各 block 将 block 偏移加到 csrRowPtr
void coo2csr_prefixsum_correct_kernel_do(
    GM_ADDR csrRowPtr,
    GM_ADDR blockTotals,
    const Coo2CsrPrefixSumCorrectTilingData &tiling,
    uint32_t numBlocks,
    void *stream);

/// Fused Kernel: 单 block 内完成 CountRows + PrefixSum
void coo2csr_fused_kernel_do(
    GM_ADDR cooRowInd,
    GM_ADDR workspace,
    GM_ADDR csrRowPtr,
    const Coo2CsrFusedTilingData &tiling,
    void *stream);

}  // extern "C"

#endif  // COO2CSR_KERNEL_H_
