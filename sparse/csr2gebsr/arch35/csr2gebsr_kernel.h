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
 * \file csr2gebsr_kernel.h
 * \brief csr2gebsr kernel_do 签名声明（Host / Kernel 共用）。
 *
 * 声明 kernel_do 函数：
 *   - csr2gebsr_nnz_kernel_do:            Kernel 1，逐块行计数非零块
 *   - csr2gebsr_prefixsum_kernel_do:      Kernel 2 退化路径，单 kernel 串行 prefix sum
 *   - csr2gebsr_prefixsum_fused_kernel_do: Kernel 2 融合路径，三阶段融合单 kernel + SyncAll
 *   - csr2gebsr_convert_kernel_do:        Kernel 3，填充 bsrColIndC + bsrValC
 */

#ifndef CSR2GEBSR_KERNEL_H_
#define CSR2GEBSR_KERNEL_H_

#include <cstdint>
#include "csr2gebsr_tiling_data.h"

// GM_ADDR 双定义说明（仓内约定，非缺陷）：
//   公共宏 kernel_utils_macros.h 将 GM_ADDR 定义为 __gm__ uint8_t*（含 __gm__，
//   供 kernel 侧使用）；本头文件被 host.cpp 包含时不引入该公共宏头，故用 #ifndef
//   守卫在此提供 uint8_t*（无 __gm__）定义，确保 host 侧 reinterpret_cast 编译通过。
//   kernel 编译单元中公共宏已定义，此处守卫不生效，无冲突。仓内多算子同此模式。
//   长期建议：引入公共 host/kernel 共用 GM_ADDR 定义头文件，消除各算子 kernel.h
//   的重复 #define。
#ifndef GM_ADDR
#define GM_ADDR uint8_t *
#endif

extern "C" {

/// Kernel 1 (Nnz CountBlocksPerRow): 逐块行统计非零块数
void csr2gebsr_nnz_kernel_do(
    GM_ADDR csrRowPtrA, GM_ADDR csrColIndA,
    GM_ADDR nnzBlocksPerRow, GM_ADDR marker,
    const Csr2gebsrNnzTilingData &tiling,
    uint32_t numBlocks,
    void *stream);

/// Kernel 2 (PrefixSum): device 侧 exclusive prefix sum + 写 nnzb
/// 退化路径（useBlocks == 1 时使用），单 block 单线程串行 O(mb)
void csr2gebsr_prefixsum_kernel_do(
    GM_ADDR nnzBlocksPerRow, GM_ADDR bsrRowPtrC, GM_ADDR nnzbDev,
    const Csr2gebsrPrefixSumTilingData &tiling,
    void *stream);

/// Kernel 2 融合路径 (PrefixSum 三阶段融合单 kernel + SyncAll)
/// <<<useBlocks>>>，单 kernel 内 Phase1→SyncAll→Phase2→SyncAll→Phase3
/// useBlocks > 1 时由 host 侧调用（消除 2 次额外 launch 开销）
void csr2gebsr_prefixsum_fused_kernel_do(
    GM_ADDR nnzBlocksPerRow, GM_ADDR bsrRowPtrC, GM_ADDR nnzbDev, GM_ADDR segSum,
    const Csr2gebsrPrefixSumTilingData &tiling,
    uint32_t numBlocks,
    void *stream);

/// Kernel 3 (Convert): 填充 bsrColIndC + bsrValC
void csr2gebsr_convert_kernel_do(
    GM_ADDR csrValA, GM_ADDR csrRowPtrA, GM_ADDR csrColIndA,
    GM_ADDR bsrRowPtrC, GM_ADDR bsrColIndC, GM_ADDR bsrValC,
    GM_ADDR marker,
    const Csr2gebsrConvertTilingData &tiling,
    uint32_t numBlocks,
    void *stream);

}  // extern "C"

#endif  // CSR2GEBSR_KERNEL_H_
