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
 * \file spmm_op_tiling_data.h
 * \brief SpMMOp TilingData 结构体定义（Host / Kernel 共用）。
 *
 * 定义 SpmmOpTilingData 结构体，包含 SIMD kernel 执行所需的全部 tiling 参数。
 * Host 侧填充此结构体并通过 <<<>>> launch 传递至 kernel（by value）。
 */

#ifndef SPMM_OP_TILING_DATA_H_
#define SPMM_OP_TILING_DATA_H_

#include <cstdint>

// 线程块最大线程数：128 线程。
// Host 侧 ComputeSpmmOpBlockSplits 使用此常量计算 block 切分。
constexpr uint32_t SPMM_OP_MAX_THREADS_PER_BLOCK = 128u;

// Workspace 对齐要求（字节）
constexpr size_t SPMM_OP_WS_ALIGNMENT = 64u;

// ALG2 预处理数据在 workspace 内的固定起始偏移（保留给 header）
constexpr size_t SPMM_OP_REORDER_OFFSET = 256u;

// n维 tile 大小：每次处理 2048 列（DMA 单次搬运 8KB/4KB，SIMD 按 128 元素分批计算）
constexpr int32_t SPMM_OP_N_TILE = 2048;

// CSR 批量加载最大元素数（单行 nnz 上限，超过则回退 GM 路径）
constexpr int32_t SPMM_OP_MAX_CSR_BATCH = 4096;
// rowOffsets 批量加载最大行数（m 上限，超过则回退 GM 路径）
constexpr int32_t SPMM_OP_MAX_ROW_OFF_BATCH = 4096;

// dtype 编码（kernel 内部分发）
constexpr int32_t SPMM_OP_DTYPE_FP32 = 0;
constexpr int32_t SPMM_OP_DTYPE_FP16 = 1;

// rowOffsetType 编码（kernel 内部分发）
constexpr int32_t SPMM_OP_IDX_RT_I32 = 0;  // rowOffsetsType = int32
constexpr int32_t SPMM_OP_IDX_RT_I64 = 1;  // rowOffsetsType = int64

// ALG 编码（kernel 内部分发）
constexpr int32_t SPMM_OP_ALG_TYPE_1 = 1;
constexpr int32_t SPMM_OP_ALG_TYPE_2 = 2;

// opB 编码（kernel 内部索引计算）
constexpr int32_t SPMM_OP_OPB_NON_TRANSPOSE = 0;
constexpr int32_t SPMM_OP_OPB_TRANSPOSE = 1;

// order_pair 编码（orderB * 2 + orderC，编码值与既有 spmm.h 的 SPMM_ORDER_* 一致）
// RR=0, RC=1, CR=2, CC=3
constexpr int32_t SPMM_OP_ORDER_RR = 0;
constexpr int32_t SPMM_OP_ORDER_RC = 1;
constexpr int32_t SPMM_OP_ORDER_CR = 2;
constexpr int32_t SPMM_OP_ORDER_CC = 3;

/// Tiling 数据结构，host 侧构造并通过 <<<>>> launch 传入 kernel（by value）。
///
/// 字段来源：
///   - 静态（descr，createDescr 绑定）：m, indexBase, algType, opB, dtype,
///     rowOffsetType, highPrecision（后两者为兼容性后追加，位于结构体末尾）
///   - 动态（matB/matC，execute 传入）：n, ldb, ldc, orderPair
///   - 动态（alpha/beta + pointerMode）：alpha/beta 或 alphaPtr/betaPtr
///   - 计算量（host tiling 阶段）：rowsPerBlock
///
/// pointerMode 区分 alpha/beta 来源：
///   HOST mode   (alphaPtr == 0ULL): kernel 从 tiling.alpha / beta 读值
///   DEVICE mode (alphaPtr != 0ULL): kernel 从 alphaPtr / betaPtr 指向的 device 内存读值
///
/// algType 区分行映射模式（reorder / bin_edge 地址通过 kernel __gm__ 参数传入，不放入 tiling）：
///   ALG1 (SPMM_OP_ALG_TYPE_1): 使用原始行号，binEdge/reorder 均为 nullptr
///   ALG2 (SPMM_OP_ALG_TYPE_2): 使用 reorder[row] 作为原始行号，binEdge 用于负载均衡行范围切分
struct SpmmOpTilingData {
    int32_t m;              ///< CSR 行数（固定）
    int32_t n;              ///< 输出列数（matC.cols，execute 时填入）
    int32_t indexBase;      ///< CSR indexBase（0 = ZERO 或 1 = ONE）
    int32_t algType;        ///< SPMM_OP_ALG_TYPE_1 或 _2
    int32_t opB;            ///< SPMM_OP_OPB_NON_TRANSPOSE 或 _TRANSPOSE
    int32_t orderPair;      ///< SPMM_OP_ORDER_RR/RC/CR/CC
    int32_t dtype;          ///< SPMM_OP_DTYPE_FP32 或 _FP16
    int32_t ldb;            ///< matB 的 leading dimension
    int32_t ldc;            ///< matC 的 leading dimension
    float alpha;            ///< α 标量（HOST mode：已解引用的值）
    float beta;             ///< β 标量（HOST mode：已解引用的值）
    /// 每 block 处理的行数（ALG1 均匀切分；ALG2 由 binEdge 覆盖，
    /// 用于 ComputeRowRangeAndThreads 行范围计算）
    uint32_t rowsPerBlock;
    uint64_t alphaPtr;      ///< DEVICE mode: α 的 device 地址（HOST mode: 0）
    uint64_t betaPtr;       ///< DEVICE mode: β 的 device 地址（HOST mode: 0）
    // 兼容性后追加字段（保持既有字段偏移不变，不影响 <<<>>> by-value 布局兼容性）
    int32_t rowOffsetType;  ///< SPMM_OP_IDX_RT_I32 或 _I64
    int32_t highPrecision;  ///< 0 或 1（仅 FP32 + ALG1_HIGH_PRECISION 时为 1）
    int32_t k;              ///< CSR 列数（归约维，用于 GlobalTensor sizing）
    int32_t nTile;          ///< SIMD n维 tile 大小，默认 2048
    // reorder / bin_edge 地址通过 kernel __gm__ 参数传入，不放入 tiling
};

#endif  // SPMM_OP_TILING_DATA_H_
