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

#pragma once

#include <cstdint>

// TEMPLATE: SIMT Tiling 数据结构（host/kernel 共享）
// SIMT 特有字段：nthreads（每 block 线程数，host 侧计算）
// 标量参数（alpha/beta/format/trans 等）直接放入 tiling 传给 kernel，避免 kernel 再读 GM
// SIMT 常量：SIMT_MIN_THREAD_NUM=128, SIMT_MAX_THREAD_NUM=2048
struct {{Op}}TilingData {
    uint32_t nthreads;

    // TEMPLATE: 稀疏矩阵维度参数
    uint32_t totalRows;
    uint32_t totalCols;
    uint32_t totalNnz;

    // TEMPLATE: 数据类型标识（ACL_SPARSE_R_32F / ACL_SPARSE_R_16F 等）
    // uint32_t valType;

    // TEMPLATE: 稀疏格式标识（ACL_SPARSE_FORMAT_CSR / ACL_SPARSE_FORMAT_COO 等）
    // uint32_t format;

    // TEMPLATE: 稀疏算子可选参数
    // uint32_t trans;          // ACL_SPARSE_OP_N / ACL_SPARSE_OP_T / ACL_SPARSE_OP_C
    // float alpha;
    // float beta;
};
