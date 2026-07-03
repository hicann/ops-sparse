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

// TEMPLATE: Tiling 数据结构头文件（host/kernel 共享）
// - host 侧计算后以 const 引用传入 kernel_do，kernel 以 by value 接收（运行时 launch 参数自动拷贝）
// - 字段完全由算子的 Tiling 策略决定

#pragma once

#include <cstdint>

// 稀疏算子典型 Tiling 结构（以 SpMV CSR 按行切分为例）：
//   uint32_t totalRows;      — 矩阵总行数
//   uint32_t totalCols;      — 矩阵总列数
//   uint32_t rowsPerCore;    — 每核处理的行数（均分部分）
//   uint32_t remainderRows;  — 余数行（前 remainderRows 个核多处理 1 行）
//   uint32_t maxRowNnz;      — 单行最大非零元数（用于 UB 预算）
struct {{Op}}TilingData {
    // TEMPLATE: 按算子需求填写字段
};

// TEMPLATE: 根据 dtype 调整（float=8, half=16, double=4）
constexpr uint32_t ELEMENTS_PER_BLOCK = 32 / sizeof(/* dtype */float);
