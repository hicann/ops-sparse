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
 * \file scatter_tiling_data.h
 * \brief aclsparseScatter TilingData 结构体定义（Host / Kernel 共用）。
 *
 * 不使用 TilingKey：组合维度小（3 dtype × 2 idxType = 6），idxBase 为标量语义（0/1），
 * 用独立字段 + dispatcher 分发比 uint64_t 编码更清晰。
 */

#ifndef SCATTER_TILING_DATA_H_
#define SCATTER_TILING_DATA_H_

#include <cstdint>

// SIMT 线程块最大线程数（scatter 为纯 GM 读写轻量 kernel，
// 256 平衡并行度（掩盖 GM 延迟）与调度开销）。
constexpr uint32_t kScatterMaxThreadsPerBlock = 256u;

// 索引类型编码（kernel 内部分发用）
constexpr int32_t SCATTER_IDX_I32 = 0;  // idxType = int32_t
constexpr int32_t SCATTER_IDX_I64 = 1;  // idxType = int64_t

// 值类型编码（kernel 内部分发用）
constexpr int32_t SCATTER_VAL_FP32 = 0;
constexpr int32_t SCATTER_VAL_FP16 = 1;
constexpr int32_t SCATTER_VAL_BF16 = 2;

struct ScatterTilingData {
    uint64_t nnz;        ///< 非零元素数量（grid-stride 范围上界）
    int32_t idxBase;     ///< 索引基址偏移：0=ZERO_BASE, 1=ONE_BASE
    int32_t idxType;     ///< SCATTER_IDX_I32 或 SCATTER_IDX_I64
    int32_t valType;     ///< SCATTER_VAL_FP32 / FP16 / BF16
};

#endif  // SCATTER_TILING_DATA_H_
