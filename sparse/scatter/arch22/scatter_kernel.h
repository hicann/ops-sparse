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
 * \file scatter_kernel.h
 * \brief scatter kernel_do 启动器声明（Host / Kernel 共用）。
 */

#ifndef SCATTER_KERNEL_H
#define SCATTER_KERNEL_H

#include <stdint.h>
#include "scatter.h"

// scatter kernel_do launcher declaration (Host / Kernel shared).
// Defined in scatter_kernel.cpp; linked with C++ linkage (no extern "C"),
// matching the definition. Host code includes this header instead of writing
// an extern forward declaration (rule G.EXP.05-CPP).
//
// Tiling 随 kernel 启动参数一起下发（const 引用传递），Host 侧不再单独分配
// device 内存存 tiling，对齐 arch35 scatter 与 spmv arch22 的实现模式。
void scatter_kernel_do(
    void *valDev, void *idxDev, void *yDev,
    const ScatterTilingData &tiling, uint32_t blockNum, void *stream);

#endif // SCATTER_KERNEL_H
