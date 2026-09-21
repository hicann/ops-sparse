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
 * \file cscsort_tiling_data.h
 * \brief aclsparseXcscsort TilingData 结构体定义（Host / Kernel 共用，arch22）。
 *
 * arch22 无 SIMT 且 AscendC Sort 仅支持 half/float 键：热路径用唯一 float
 * 合成键 + 硬件 Sort 降序全排构造稳定升序；键宽超预算的 run/列回退 kernel
 * 自实现标量两路稳定归并。长列先按 runSize 分块排序写入 GM workspace，
 * 再多趟归并（GM 写经 UB 中转，见 kernel 注释）。
 */

#ifndef CSCSORT_TILING_DATA_H_
#define CSCSORT_TILING_DATA_H_

#include <cstdint>

/// cscsort TilingData（Host 计算，Kernel 消费）。
struct CscsortTilingData {
    uint32_t m;  ///< host 透传的矩阵行数（kernel 不使用：行域上界由 kernel 内 ReduceMax 实测，占位 m/行号超 m 均按值正确排序）
    uint32_t n;             ///< 矩阵列数（压缩维度，kernel 二分定位列边界用）
    uint32_t nnz;           ///< 非零元总数
    uint32_t indexBase;     ///< 0 或 1，来自 descrA.indexBase
    uint32_t runSize;       ///< 单 run 最大元素数（UB 容量/键宽预算/全排上限反推）
    uint32_t coreNum;       ///< Host 选择的 AIV 启动核数，Kernel 按累计 nnz 切完整列区间
    uint32_t sortTmpBytes;  ///< 硬件 Sort 树 tmp 段字节数（8B/元素，按 Align32(runSize)）
};

#endif  // CSCSORT_TILING_DATA_H_
