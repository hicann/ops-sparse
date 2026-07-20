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
#include "{{op}}_tiling_data.h"

// kernel_do 函数声明：host 侧调用此函数启动 kernel
// RegBase 与 SIMD 使用相同的调用约定（tiling const 引用，无 GM tiling）
void {{op}}_kernel_do(/* GM_ADDR 各数据指针（按算子需求调整）, */
                       uint32_t numBlocks, const {{Op}}TilingData& tiling, void* stream);
