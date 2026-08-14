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
 * \file gebsr2gebsc_kernel.h
 * \brief gebsr2gebsc kernel_do 签名声明（Host / Kernel 共用）。
 *
 * 声明五个 kernel_do 函数：
 *   - gebsr2gebsc_count_kernel_do:           Kernel 1，stripe 列直方图（原子加）
 *   - gebsr2gebsc_sum_stripe_hist_kernel_do: Kernel 1.5，stripeHist 按列求和 -> colCount
 *   - gebsr2gebsc_prefixsum_kernel_do:       Kernel 2，exclusive prefix sum（单 warp 并行 scan）
 *   - gebsr2gebsc_stripebase_kernel_do:      Kernel 3，stripe 直方图按列前缀和 -> 写游标基址
 *   - gebsr2gebsc_scatter_kernel_do:         Kernel 4，每 block 单线程顺序 scatter 写 bscRowInd / bscVal
 */

#ifndef GEBSR2GEBSC_KERNEL_H_
#define GEBSR2GEBSC_KERNEL_H_

#include <cstdint>
#include "gebsr2gebsc_tiling_data.h"

#ifndef GM_ADDR
#define GM_ADDR uint8_t *
#endif

extern "C" {

/// Kernel 1 (CountCols): 遍历 bsrColIndA，asc_atomic_add 统计每 stripe 列直方图。
/// 注意：不再写 colCount（K1 优化），colCount 由 Kernel 1.5 SumStripeHist 重建。
void gebsr2gebsc_count_kernel_do(
    GM_ADDR bsrColIndA,
    GM_ADDR stripeHist,
    const Gebsr2GebscCountTilingData &tiling,
    uint32_t numBlocks,
    void *stream);

/// Kernel 1.5 (SumStripeHist): 对 stripeHist 按列求和 -> colCount（多线程并行）
void gebsr2gebsc_sum_stripe_hist_kernel_do(
    GM_ADDR stripeHist,
    GM_ADDR workspace,
    const Gebsr2GebscSumStripeHistTilingData &tiling,
    uint32_t numBlocks,
    void *stream);

/// Kernel 2 (PrefixSum): exclusive prefix sum -> bscColPtr（单 warp 并行 scan）
void gebsr2gebsc_prefixsum_kernel_do(
    GM_ADDR workspace,
    GM_ADDR bscColPtr,
    const Gebsr2GebscPrefixSumTilingData &tiling,
    void *stream);

/// Kernel 3 (StripeBase): stripe 直方图按列前缀和 -> 每 stripe 写游标基址（原地转换）
void gebsr2gebsc_stripebase_kernel_do(
    GM_ADDR stripeBase,
    GM_ADDR bscColPtr,
    const Gebsr2GebscStripeBaseTilingData &tiling,
    uint32_t numBlocks,
    void *stream);

/// Kernel 4 (Scatter): 按 valType 分发，每 block 单线程顺序 scatter（保持 GEBSR 行优先顺序）
/// 含块值拷贝（direct copy / block transpose），游标私有（无原子竞争）
void gebsr2gebsc_scatter_kernel_do(
    GM_ADDR bsrRowPtrA,
    GM_ADDR bsrColIndA,
    GM_ADDR bsrValA,
    GM_ADDR bscRowInd,
    GM_ADDR bscVal,
    GM_ADDR stripeCursor,
    const Gebsr2GebscScatterTilingData &tiling,
    uint32_t numBlocks,
    void *stream);

}  // extern "C"

#endif  // GEBSR2GEBSC_KERNEL_H_
