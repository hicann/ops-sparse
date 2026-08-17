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
 * \file csr2gebsr_tiling_data.h
 * \brief csr2gebsr TilingData 结构体定义（Host / Kernel 共用）。
 *
 * csr2gebsr 包含三个 kernel：
 *   - Kernel 1 (Nnz CountBlocksPerRow): 逐块行统计非零块数
 *   - Kernel 2 (PrefixSum): device 侧 exclusive prefix sum
 *                退化路径（useBlocks==1）单 kernel 串行 / 并行路径（useBlocks>1）三阶段
 *   - Kernel 3 (Convert): 填充 bsrColIndC + bsrValC
 * 各使用独立的 TilingData 结构体。
 */

#ifndef CSR2GEBSR_TILING_DATA_H_
#define CSR2GEBSR_TILING_DATA_H_

#include <cstdint>

/// 每 block 最大线程数（SIMT VF launch_bounds）
constexpr uint32_t kCsr2gebsrMaxThreadsPerBlock = 128;

/// Warp 大小（向上取整对齐用）
constexpr uint32_t kCsr2gebsrWarpSize = 32;

/// Nnz 阶段 Kernel 1 TilingData（逐块行计数非零块数）
struct Csr2gebsrNnzTilingData {
    int32_t m;              ///< 矩阵行数
    int32_t n;              ///< 矩阵列数
    int32_t mb;             ///< 块行数 = ceil(m / rowBlockDim)
    int32_t nb;             ///< 块列数 = ceil(n / colBlockDim)
    int32_t rowBlockDim;    ///< GEBSR 块行数
    int32_t colBlockDim;    ///< GEBSR 块列数
    int32_t baseA;          ///< CSR indexBase (0 或 1)
    uint32_t blockRowsPerCore; ///< 每 Core 处理的块行数
    float invColBlockDim;   ///< 1.0f / colBlockDim（预计算，kernel 用乘法替代除法）
};

/// Prefix Sum 阶段 Kernel 2 TilingData
///
/// useBlocks == 1 时走退化串行路径（原单 kernel）；useBlocks > 1 时走融合单 kernel 路径
/// （Phase1 段内 inclusive scan → SyncAll → Phase2 段间 scan → SyncAll → Phase3 加偏移）。
/// useBlocks / blockRowsPerCore 复用 ComputeBlockSplits 结果，与 Kernel 1 切分对齐。
struct Csr2gebsrPrefixSumTilingData {
    int32_t mb;             ///< 块行数
    int32_t baseC;          ///< 输出 indexBase (0 或 1)
    uint32_t useBlocks;         ///< Phase1/3 启动核数（= Kernel1 的 useBlocks）
    uint32_t blockRowsPerCore;  ///< 每核块行数 L = ceil(mb/useBlocks)
};

/// Convert 阶段 Kernel 3 TilingData
struct Csr2gebsrConvertTilingData {
    int32_t m;              ///< 矩阵行数
    int32_t n;              ///< 矩阵列数
    int32_t mb;             ///< 块行数
    int32_t nb;             ///< 块列数
    int32_t rowBlockDim;    ///< GEBSR 块行数
    int32_t colBlockDim;    ///< GEBSR 块列数
    int32_t baseA;          ///< CSR indexBase (0 或 1)
    int32_t baseC;          ///< 输出 indexBase (0 或 1)
    int32_t dir;            ///< 0 = ROW, 1 = COLUMN
    uint32_t valSize;       ///< 值类型字节数: 4 (FP32/INT32) 或 2 (FP16/BF16)
    uint32_t blockRowsPerCore; ///< 每 Core 处理的块行数
    float invColBlockDim;   ///< 1.0f / colBlockDim（预计算，kernel 用乘法替代除法）
};

#endif  // CSR2GEBSR_TILING_DATA_H_
