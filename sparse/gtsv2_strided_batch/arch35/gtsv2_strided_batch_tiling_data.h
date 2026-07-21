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
 * \file gtsv2_strided_batch_tiling_data.h
 * \brief aclsparseSgtsv2StridedBatch TilingData 结构体定义（Host / Kernel 共用）。
 */

#ifndef GTSV2_STRIDED_BATCH_TILING_DATA_H_
#define GTSV2_STRIDED_BATCH_TILING_DATA_H_

#include <cstdint>

// 共享常量，CR 算法需保存 a/b/c/d 四路系数
#define GTSV2_NUM_SAVE_ARRAYS 4
// CR 最大层数上限：仅约束内层 UB CR 层数（<= 11），栈数组尺寸不变
#define GTSV2_MAX_CR_LEVELS 16
// 外层 GM CR 层数上限（L <= 30 -> K <= 19，留 1 余量）
#define GTSV2_MAX_OUTER_CR_LEVELS 20
// Host workspace 布局与 Kernel halo 窗口共用的保护元素数
//（GM 工作区数组尾部过读保护，兼作外层正向 staging / 反向 x 窗口 halo 长度）
#define GTSV2_GUARD_FLOATS 8

/// TilingData for gtsv2_strided_batch
/// 纯 UB 路径 m <= 2048（行为不变）；外层 GM 分块路径 2048 < m <= 2^30
struct Gtsv2StridedBatchTilingData {
    int32_t m;              // 每个三对角系统的维度（3 <= m <= 2^30）
    int32_t batchCount;     // 总批量数
    int32_t batchStride;    // batch 间元素步长（>= ceil(m_pad/8)*8，且为 8 的整数倍）
    int32_t batchPerCoreBase; // 每 Core 基础 batch 数 = batchCount / numBlocks
    int32_t remainder;       // 前 remainder 个核多处理 1 个 batch
    int32_t numLayers;      // 内层 CR 层数：纯 UB 路径 = L；外层路径 = 11（常量）
    int32_t saveBufSize;    // 内层单路 UB save 字节数（按内层 mPad 计算，公式不变）
    int32_t useGmWorkspace; // 0 = 纯 UB 路径，1 = 外层 GM 分块路径
    // Host 预计算，消除 Host/Kernel 双源独立计算
    int32_t mPad;           // 内层 mPad（外层路径 = 2048），Host 计算后传递
    int32_t alignedM;       // 内层 alignedM（外层路径 = 2048），DataCopy 32B 对齐所需
    // ---- 外层 GM 分块路径字段 ----
    int32_t outerLevels;            // 外层 GM CR 层数 K = max(0, L - 11)；纯 UB 路径 = 0
    int32_t outerTileElems;         // 外层 tile 输出元素数 P = 1024（8 的倍数）
    int64_t mPadOuter;              // 完整 mPad = 2^ceil(log2(m))（int64，防 2^31 附近溢出）
    int64_t regionBByteOffset;      // RegionB 相对 batch base 的字节偏移 = 4 * S_A * sizeof(float)
    int64_t saveRegionByteOffset;   // SaveRegion 相对 batch base 的字节偏移
    int64_t workspacePerBatchBytes; // 每 batch workspace 总字节（128B 对齐后）
};

#endif  // GTSV2_STRIDED_BATCH_TILING_DATA_H_
