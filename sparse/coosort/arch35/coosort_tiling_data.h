/**
 * ----------------------------------------------------------------------------------------------------------
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ----------------------------------------------------------------------------------------------------------
 */

/*!
 * \file coosort_tiling_data.h
 * \brief coosort TilingData 结构体定义（Host / Kernel 共用）。
 *
 * aclsparseXcoosort 单核直排与多核 run 生成均使用 AscendC Sort API
 *（RADIX_SORT，两趟 LSB radix）：
 *   - Pass1 次键 Sort → Gather 主键 → Pass2 主键 Sort，全程 int32 signed TQue + 向量化，
 *     原生支持负数无需符号位归一。两趟 stable Sort 合成双键字典序。
 *   - sharedTmpSize 由 host 侧 GetSortMaxMinTmpSize 取 max 计算，kernel 内 pipe.InitBuffer 从 UB 分配。
 */

#ifndef COOSORT_TILING_DATA_H_
#define COOSORT_TILING_DATA_H_

#include <cstdint>

/// bufferSizeExt 返回大小的对齐粒度。该值用于 workspace 容量计算，
/// 不表示调用方传入的 pBuffer 指针必须额外满足 128 字节对齐。
constexpr uint32_t kCoosortAlign = 128;

/// 单核直排路径阈值。超过该阈值后进入多核、多 run 路径。
///
/// 该值不是整个算子的 nnz 上限。多核路径会根据运行时 UB 容量计算安全 runSize；
/// 当 runCount 大于启动核数时，每个核循环排序多个 run，再在 GM 中做多轮稳定归并。
constexpr uint32_t kCoosortSingleCoreMaxNnz = 2048;

/// coosort TilingData（Host 计算，单核和多核 Kernel 共用）。
struct CoosortTilingData {
    uint32_t nnz;          ///< 非零元个数
    uint32_t sortByRow;    ///< 1=ByRow(row 主/col 次)；0=ByColumn(col 主/row 次)
    uint32_t sortTmpSize;  ///< Sort sharedTmpBuffer 字节数（host 侧 GetSortMaxMinTmpSize 取 max）
    uint32_t coreNum;      ///< Host 选择的多核路径 AIV 启动核数
    uint32_t tileSize;     ///< 一个 Phase 1 有序 run 的元素数
    uint32_t runCount;     ///< Phase 1 生成的 run 总数
    uint32_t mergeSize;    ///< 单个 merge-path 输出片段可包含的最大 COO 记录数
};

#endif  // COOSORT_TILING_DATA_H_
