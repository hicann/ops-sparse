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
 * \file csr2csc_ex2_kernel.h
 * \brief csr2csc_ex2 kernel_do 签名声明（Host / Kernel 共用）。
 *
 * 声明五个 kernel_do 函数：
 *   - csr2csc_count_kernel_do:           Kernel 1，stripe 列直方图（原子加）
 *   - csr2csc_sum_stripe_hist_kernel_do: Kernel 1.5，stripeHist 按列求和 -> colCount
 *   - csr2csc_prefixsum_kernel_do:       Kernel 2，exclusive prefix sum（单 warp 并行 scan）
 *   - csr2csc_stripebase_kernel_do:      Kernel 3，stripe 直方图按列前缀和 -> 写游标基址
 *   - csr2csc_scatter_kernel_do:         Kernel 4，每 block 单线程顺序 scatter 写 cscRowInd / cscVal
 */

#ifndef CSR2CSC_EX2_KERNEL_H_
#define CSR2CSC_EX2_KERNEL_H_

#include <cstdint>
#include "csr2csc_ex2_tiling_data.h"

#ifndef GM_ADDR
#define GM_ADDR uint8_t *
#endif

extern "C" {

/// Kernel 1 (CountCols): 遍历 csrColInd，asc_atomic_add 统计每 stripe 列直方图。
/// 注意：不再写 colCount（K1 优化），colCount 由 Kernel 1.5 SumStripeHist 重建。
void csr2csc_count_kernel_do(
    GM_ADDR csrColInd,
    GM_ADDR stripeHist,
    const Csr2CscCountTilingData &tiling,
    uint32_t numBlocks,
    void *stream);

/// Kernel 1.5 (SumStripeHist): 对 stripeHist 按列求和 -> colCount（多线程并行）
void csr2csc_sum_stripe_hist_kernel_do(
    GM_ADDR stripeHist,
    GM_ADDR workspace,
    const Csr2CscSumStripeHistTilingData &tiling,
    uint32_t numBlocks,
    void *stream);

/// Kernel 2 (PrefixSum): exclusive prefix sum -> cscColPtr（单 warp 并行 scan）
void csr2csc_prefixsum_kernel_do(
    GM_ADDR workspace,
    GM_ADDR cscColPtr,
    const Csr2CscPrefixSumTilingData &tiling,
    void *stream);

/// Kernel 3 (StripeBase): stripe 直方图按列前缀和 -> 每 stripe 写游标基址（原地转换）
void csr2csc_stripebase_kernel_do(
    GM_ADDR stripeBase,
    GM_ADDR cscColPtr,
    const Csr2CscStripeBaseTilingData &tiling,
    uint32_t numBlocks,
    void *stream);

/// Kernel 4 (Scatter): 按 valType 分发，每 block 单线程顺序 scatter（保持 CSR 行优先顺序）
void csr2csc_scatter_kernel_do(
    GM_ADDR csrRowPtr,
    GM_ADDR csrColInd,
    GM_ADDR csrVal,
    GM_ADDR cscRowInd,
    GM_ADDR cscVal,
    GM_ADDR stripeCursor,
    const Csr2CscScatterTilingData &tiling,
    uint32_t numBlocks,
    void *stream);

}  // extern "C"

#endif  // CSR2CSC_EX2_KERNEL_H_
