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
 * \file coo2csr_tiling_data.h
 * \brief coo2csr TilingData 结构体定义（Host / Kernel 共用）。
 *
 * coo2csr 包含多个 kernel：
 *   - Kernel 1 (CountRows):          遍历 cooRowInd，RLE + asc_atomic_add 统计每行 nnz
 *   - Kernel 2 (PrefixSum):          exclusive prefix sum -> csrRowPtr
 *     - 单核路径：单 block 串行前缀和
 *     - 多核路径：3 阶段并行前缀和（Local -> Blocks -> Correct）
 *   - Fused Kernel:                  单 block 内完成 CountRows + PrefixSum
 */

#ifndef COO2CSR_TILING_DATA_H_
#define COO2CSR_TILING_DATA_H_

#include <cstdint>

/// 每 block 最大 SIMT 线程数
constexpr uint32_t kCoo2CsrThreadsPerBlock = 1024;

/// Warp 大小（线程数向上取整对齐用）
constexpr uint32_t kCoo2CsrWarpSize = 32;
static_assert((kCoo2CsrWarpSize & (kCoo2CsrWarpSize - 1)) == 0, "warp size must be power of 2");

/// 多核并行前缀和阈值：m > 此值时多核并行前缀和收益超过 3-kernel launch 开销
constexpr int32_t kCoo2CsrParallelPrefixSumThreshold = 1024;

/// Kernel 1 (CountRows) TilingData
struct Coo2CsrCountTilingData {
    int32_t nnz = 0;         ///< 非零元素总数（cooRowInd 长度）
    int32_t idxBase = 0;     ///< 索引基值 (0 或 1)
    int32_t m = 0;           ///< 矩阵行数
};

/// Kernel 2 (PrefixSum, 单核) TilingData
struct Coo2CsrPrefixSumTilingData {
    int32_t m = 0;       ///< 矩阵行数
    int32_t idxBase = 0; ///< 索引基值 (0 或 1)
};

/// Kernel 2 Phase A (PrefixSum Local) TilingData
struct Coo2CsrPrefixSumLocalTilingData {
    int32_t m = 0;         ///< 矩阵行数
    int32_t idxBase = 0;   ///< 索引基值 (0 或 1)
    int32_t chunkSize = 0; ///< 每 block 处理的行数
};

/// Kernel 2 Phase B (PrefixSum Blocks) TilingData
struct Coo2CsrPrefixSumBlocksTilingData {
    uint32_t numBlocks = 0; ///< block 总数（blockTotals 数组长度）
};

/// Kernel 2 Phase C (PrefixSum Correct) TilingData
struct Coo2CsrPrefixSumCorrectTilingData {
    int32_t m = 0;         ///< 矩阵行数
    int32_t chunkSize = 0; ///< 每 block 处理的行数
};

/// Fused Kernel TilingData（单 block 内完成 CountRows + PrefixSum）
struct Coo2CsrFusedTilingData {
    int32_t nnz = 0;     ///< 非零元素总数
    int32_t m = 0;       ///< 矩阵行数
    int32_t idxBase = 0; ///< 索引基值 (0 或 1)
};

#endif  // COO2CSR_TILING_DATA_H_
