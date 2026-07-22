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
 * \file gtsv2_kernel.h
 * \brief aclsparseSgtsv2 kernel_do 声明（Host / Kernel 共用）。
 */

#ifndef GTSV2_KERNEL_H_
#define GTSV2_KERNEL_H_

#include <cstdint>
#include "gtsv2_tiling_data.h"

#ifndef GM_ADDR
#define GM_ADDR uint8_t *
#endif

// SIMT block 宽度: 每个 block 内的线程数 (与 gtsv2_nopivot / gtsv_interleaved_batch 一致)
// 使用算子前缀 gtsv2Pivot 避免与 gtsv2_nopivot_kernel.h 中的 kGtsv2ThreadsPerBlock 重名。
constexpr uint32_t kGtsv2PivotThreadsPerBlock = 32U;

// Number of workspace arrays per RHS column: d' (modified main diagonal),
// du' (modified super-diagonal with fill-in), du2' (2nd super-diagonal fill-in).
constexpr int32_t kGtsv2WorkspaceArraysPerRhs = 3;

void gtsv2_kernel_do(
    GM_ADDR dl, GM_ADDR d, GM_ADDR du, GM_ADDR B,
    GM_ADDR workspace,
    const Gtsv2TilingData &tiling,
    uint32_t numBlocks,
    void *stream);

#endif // GTSV2_KERNEL_H_
