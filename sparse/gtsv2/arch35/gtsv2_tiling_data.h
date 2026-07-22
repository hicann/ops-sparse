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
 * \file gtsv2_tiling_data.h
 * \brief aclsparseSgtsv2 TilingData 结构体定义（Host / Kernel 共用）。
 *
 * SIMT Thomas 算法（带部分选主元）：kernel 直接读写 GM，workspace 存
 * forward 中间 d'、du' 和 du2' 值（含主元交换产生的 fill-in）。
 */

#ifndef GTSV2_TILING_DATA_H_
#define GTSV2_TILING_DATA_H_

#include <cstdint>

struct Gtsv2TilingData {
    int32_t m;              // 系统规模（行数/列数）
    int32_t n;              // 右端项列数
    int32_t ldb;            // B 矩阵 leading dimension
    int32_t numBlocks;      // SIMT launch block 数
    int32_t reserved;       // 保留字段（对齐填充）
};

#endif // GTSV2_TILING_DATA_H_
