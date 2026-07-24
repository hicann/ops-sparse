/**
 * ----------------------------------------------------------------------------------------------------------
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under
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
 * \file csrsort_tiling_data.h
 * \brief aclsparseXcsrsort TilingData 结构体定义（Host / Kernel 共用）。
 *
 * 每行 segment 内使用 AscendC Sort<int32_t>（RADIX_SORT，稳定升序）单键排序；
 * 超长行切多 run 后在 GM workspace 中做多轮稳定归并。
 * sortTmpBytes 由 host 侧 GetSortMaxMinTmpSize 取 max 计算，kernel 内
 * pipe.InitBuffer 从 UB 分配。
 */

#ifndef CSRSORT_TILING_DATA_H_
#define CSRSORT_TILING_DATA_H_

#include <cstdint>

constexpr uint32_t kCsrsortSimtWarpSize = 32U;
constexpr uint32_t kCsrsortSimtMaxThreads = 2048U;

/// csrsort TilingData（Host 计算，Kernel 消费）。
struct CsrsortTilingData {
  uint32_t m;         ///< 矩阵行数
  uint32_t nnz;       ///< 非零元总数
  uint32_t indexBase; ///< 0 或 1，来自 descrA.indexBase
  uint32_t runSize;   ///< 单 run 最大元素数（UB 容量反推）
  uint32_t
      coreNum; ///< Host 选择的 AIV 启动核数，Kernel 按累计 nnz 切完整行区间
  uint32_t sortTmpBytes; ///< Sort<int32_t> 临时缓冲区字节数
  uint32_t
      simtThreads; ///< SIMT VF 实际启动线程数，32 对齐且不超过 launch bound
};

#endif // CSRSORT_TILING_DATA_H_
