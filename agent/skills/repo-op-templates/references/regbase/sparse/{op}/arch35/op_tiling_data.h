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

#pragma pack(push, 4)

// TEMPLATE: RegBase Tiling 结构体（host/kernel 共享）
// 与 SIMD 的区别：需要 UB buffer 大小信息（host 侧预计算）
struct {{Op}}TilingData {
    // TEMPLATE: 基础维度参数
    uint32_t totalRows;
    uint32_t totalCols;

    // TEMPLATE: 数据类型标识（用于 kernel 侧分支，如 ACL_SPARSE_R_32F / ACL_SPARSE_R_16F）
    // uint32_t valType;

    // TEMPLATE: 多核切分参数
    uint32_t rowsPerCore;
    uint32_t remainderRows;
    uint32_t usedCoreNum;

    // TEMPLATE: 稀疏算子特有参数
    // uint32_t maxRowNnz;

    // TEMPLATE: UB buffer 大小（RegBase 特有，host 侧按 256B 对齐预计算）
    // uint32_t bufColInd;
    // uint32_t bufValues;
    // uint32_t bufAcc;
};

#pragma pack(pop)
