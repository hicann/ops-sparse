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
 * \file gather_kernel.h
 * \brief gather kernel_do 签名声明（Host / Kernel 共用）。
 *
 * Gather:  X.values[i] = Y[X.indices[i] - idxBase]  for i = 0 .. nnz-1.
 */

#pragma once

#include <cstdint>
#include "acl/acl_base_rt.h"
#include "gather_tiling_data.h"

#ifndef GM_ADDR
#define GM_ADDR uint8_t *
#endif

extern "C" {

// Gather kernel launch 入口
void gather_kernel_do(
    GM_ADDR indices,
    GM_ADDR yValues,
    GM_ADDR xValues,
    const GatherTilingData &tiling,
    aclrtStream stream);
}  // extern "C"
