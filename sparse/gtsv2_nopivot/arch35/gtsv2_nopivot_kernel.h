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
 * \file gtsv2_nopivot_kernel.h
 * \brief aclsparseSgtsv2Nopivot kernel_do 声明（Host / Kernel 共用）。
 */

#ifndef GTSV2_NOPIVOT_KERNEL_H_
#define GTSV2_NOPIVOT_KERNEL_H_

#include <cstdint>
#include "gtsv2_nopivot_tiling_data.h"

#ifndef GM_ADDR
#define GM_ADDR uint8_t *
#endif

// SIMT block 宽度: 每个 block 内的线程数 (参考 gtsv_interleaved_batch)
constexpr uint32_t kGtsv2ThreadsPerBlock = 32U;

void gtsv2_nopivot_kernel_do(
    GM_ADDR dl, GM_ADDR d, GM_ADDR du, GM_ADDR B,
    GM_ADDR workspace,
    const Gtsv2NopivotTilingData &tiling,
    uint32_t numBlocks,
    void *stream);

#endif // GTSV2_NOPIVOT_KERNEL_H_
