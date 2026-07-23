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
 * \file spmv_op_tiling_data.h
 * \brief SpMVOp TilingData 结构体定义（Host / Kernel 共用）。
 *
 * 定义 SpmvOpTilingData 结构体，包含 SIMT kernel 执行所需的全部 tiling 参数。
 * Host 侧填充此结构体并通过 <<<>>> launch 传递至 kernel（by value）。
 */

#ifndef SPMV_OP_TILING_DATA_H_
#define SPMV_OP_TILING_DATA_H_

#include <cstdint>

// SIMT 线程块最大线程数（性能调优结果：128 线程平衡了并行度与寄存器分配）
constexpr uint32_t kSpmvOpMaxThreadsPerBlock = 128u;

// Warp 大小（向上取整对齐用）
constexpr uint32_t kSpmvOpWarpSize = 32u;

// Workspace 对齐要求（字节）
constexpr size_t kSpmvOpWsAlignment = 64u;

// ALG2 预处理数据在 workspace 内的固定起始偏移（保留给 header）
constexpr size_t kSpmvOpReorderOffset = 256u;

// 索引类型编码（kernel 内部分发）
constexpr int32_t SPMV_OP_IDX_RT_I32 = 0;  // rowOffsetsType = int32
constexpr int32_t SPMV_OP_IDX_RT_I64 = 1;  // rowOffsetsType = int64

// ALG 编码（kernel 内部分发）
constexpr int32_t SPMV_OP_ALG_TYPE_1 = 1;
constexpr int32_t SPMV_OP_ALG_TYPE_2 = 2;

/// Tiling 数据结构，host 侧构造并通过 <<<>>> launch 传入 kernel。
///
/// pointerMode 区分 alpha/beta 来源：
///   HOST mode   (alphaPtr == 0ULL): kernel 从 tiling.alpha / beta 读值
///   DEVICE mode (alphaPtr != 0ULL): kernel 从 alphaPtr / betaPtr 指向的 device 内存读值
///
/// algType 区分行映射模式（reorder / bin_edge 地址通过 kernel __gm__ 参数传入，不放入 tiling）：
///   ALG1 (kAlgType1): 使用原始行号，binEdge/reorder 均为 nullptr
///   ALG2 (kAlgType2): 使用 reorder[row] 作为原始行号，binEdge 用于负载均衡行范围切分
struct SpmvOpTilingData {
    int32_t m;              ///< 矩阵行数
    int32_t indexBase;      ///< CSR indexBase（0 或 1）
    int32_t rowOffsetType;  ///< SPMV_OP_IDX_RT_I32 或 SPMV_OP_IDX_RT_I64
    int32_t algType;        ///< SPMV_OP_ALG_TYPE_1 或 SPMV_OP_ALG_TYPE_2
    float alpha;            ///< α 标量（HOST mode：已解引用的值）
    float beta;             ///< β 标量（HOST mode：已解引用的值）
    uint32_t rowsPerBlock;  ///< 每 Core 处理的行数（ALG1 使用均匀切分；ALG2 时由 binEdge 覆盖，此字段仅供 ComputeRowRangeAndThreads 初始线程数估算）
    uint64_t alphaPtr;      ///< DEVICE mode: α 的 device 地址（HOST mode: 0）
    uint64_t betaPtr;       ///< DEVICE mode: β 的 device 地址（HOST mode: 0）
    // reorder / bin_edge 地址通过 kernel __gm__ 参数传入，不放入 tiling
};

#endif  // SPMV_OP_TILING_DATA_H_
