/**
 * ----------------------------------------------------------------------------------------------------------
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under
 * the terms and conditions of CANN Open Software License Agreement Version 2.0
 * (the "License"). Please refer to the License for details. You may not use
 * this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
 * AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
 * FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
 * for the full text of the License.
 * ----------------------------------------------------------------------------------------------------------
 */

/*!
 * \file sparse2dense_tiling_data.h
 * \brief SparseToDense TilingData 结构体定义（Host / Kernel 共用）。
 *
 * 定义 Sparse2DenseTilingData 结构体，包含 SIMT kernel 执行所需的全部 tiling 参数。
 * Host 侧填充此结构体并通过 <<<>>> launch 传递至 kernel（by value）。
 */

#ifndef SPARSE2DENSE_TILING_DATA_H_
#define SPARSE2DENSE_TILING_DATA_H_

#include <cstdint>

// SIMT 线程块最大线程数
constexpr uint32_t kSparse2DenseMaxThreadsPerBlock = 128u;

// Warp 大小（向上取整对齐用）
constexpr uint32_t kSparse2DenseWarpSize = 32u;

// 值类型编码（kernel 内部分发）
constexpr int32_t SPARSE2DENSE_VAL_F32  = 0;
constexpr int32_t SPARSE2DENSE_VAL_F16  = 1;
constexpr int32_t SPARSE2DENSE_VAL_BF16 = 2;
constexpr int32_t SPARSE2DENSE_VAL_I32  = 3;
constexpr int32_t SPARSE2DENSE_VAL_I8   = 4;

// 稀疏格式编码（kernel 内部分发）
constexpr int32_t SPARSE2DENSE_FMT_CSR  = 0;
constexpr int32_t SPARSE2DENSE_FMT_CSC  = 1;
constexpr int32_t SPARSE2DENSE_FMT_COO  = 2;

/// Tiling 数据结构，host 侧构造并通过 <<<>>> launch 传入 kernel。
struct Sparse2DenseTilingData {
    int32_t  m;            ///< 矩阵行数
    int32_t  n;            ///< 矩阵列数
    int32_t  indexBase;    ///< 索引基（0 或 1）
    int32_t  valueType;    ///< SPARSE2DENSE_VAL_* 编码
    int32_t  isColMajor;   ///< 输出稠密矩阵布局：1=列主序，0=行主序
    int32_t  ld;           ///< leading dimension（列主序时 >= m，行主序时 >= n）
    uint32_t perBlock;  ///< 每 Core 处理的单元数（CSR/CSC 为行/列数，COO 为 nnz 数）
    int32_t  format;       ///< SPARSE2DENSE_FMT_* 编码
    uint64_t nnz;          ///< 非零元总数（COO grid-stride 用）
};

#endif  // SPARSE2DENSE_TILING_DATA_H_
