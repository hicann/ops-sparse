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
 * \file csrgeam2_tiling_data.h
 * \brief csrgeam2 TilingData 结构体定义（Host / Kernel 共用）。
 *
 * csrgeam2 包含三个 kernel：
 *   - Kernel 1 (Nnz):        逐行计数 A∪B 的非零元个数
 *   - Kernel 2 (Compute):    逐行有序归并计算 C = α·A + β·B
 *   - Kernel 3 (Prefix Sum): device 侧 exclusive prefix sum
 * 各使用独立的 TilingData 结构体。
 */

#ifndef CSRGEAM2_TILING_DATA_H_
#define CSRGEAM2_TILING_DATA_H_

#include <cstdint>

/// 每 block 最大线程数（SIMT VF launch_bounds）
/// 性能调优结果：128 线程平均最快（17837ms），平衡了并行度与寄存器分配。
constexpr uint32_t kCsrgeam2MaxThreadsPerBlock = 128;

/// Warp 大小（向上取整对齐用）
constexpr uint32_t kCsrgeam2WarpSize = 32;

/// Nnz 阶段 TilingData（Kernel 1: 逐行计数）
struct Csrgeam2NnzTilingData {
    int32_t m;              ///< 矩阵行数
    int32_t baseA;          ///< A 的 indexBase (0 或 1)
    int32_t baseB;          ///< B 的 indexBase (0 或 1)
    uint32_t rowsPerBlock;  ///< 每 Core 处理的行数
};

/// Prefix Sum 阶段 TilingData（Kernel 3: device 侧 exclusive prefix sum）
struct Csrgeam2PrefixSumTilingData {
    int32_t m;       ///< 矩阵行数
    int32_t baseC;   ///< C 的 indexBase (0 或 1)
};

/// 主计算阶段 TilingData（Kernel 2: 逐行合并）
/// pointerMode 区分 alpha/beta 来源：
///   HOST mode (alphaPtr==nullptr): kernel 从 tiling.alpha/beta 读值
///   DEVICE mode (alphaPtr!=nullptr): kernel 从 alphaPtr 指向的 device 内存读值
struct Csrgeam2TilingData {
    int32_t m;              ///< 矩阵行数
    int32_t n;              ///< 矩阵列数（预留字段，当前 kernel 未读取；未来可用于 colInd 越界校验）
    int32_t baseA;          ///< A 的 indexBase (0 或 1)
    int32_t baseB;          ///< B 的 indexBase (0 或 1)
    int32_t baseC;          ///< C 的 indexBase (0 或 1)
    float alpha;            ///< α 标量（HOST mode: 已解引用后的值）
    float beta;             ///< β 标量（HOST mode: 已解引用后的值）
    uint32_t rowsPerBlock;  ///< 每 Core 处理的行数
    uint64_t alphaPtr;      ///< DEVICE mode: α 的 device 地址（HOST mode: 0）
    uint64_t betaPtr;       ///< DEVICE mode: β 的 device 地址（HOST mode: 0）
};

#endif  // CSRGEAM2_TILING_DATA_H_
