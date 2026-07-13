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
 * \file gtsv_interleaved_batch_kernel.h
 * \brief aclsparseSgtsvInterleavedBatch kernel_do 声明（Host / Kernel 共用）。
 */

#ifndef GTSV_INTERLEAVED_BATCH_KERNEL_H_
#define GTSV_INTERLEAVED_BATCH_KERNEL_H_

#include "gtsv_interleaved_batch_tiling_data.h"

#ifndef GM_ADDR
#define GM_ADDR uint8_t *
#endif

// Batch count per thread block (SIMT model: 1 thread handles 1 batch).
// Shared between host (used to compute numBlocks) and kernel (used to
// map blockIdx.x * kThreadsPerBlock + threadIdx.x to a batch index).
// Any change here MUST be propagated to both sides.
static constexpr uint32_t kGtsvBatchPerBlock = 64u;

void gtsv_interleaved_batch_kernel_do(
    GM_ADDR dl, GM_ADDR d, GM_ADDR du, GM_ADDR x,
    GM_ADDR workspace,
    const GtsvInterleavedBatchTilingData &tiling,
    uint32_t numBlocks,
    void *stream);

#endif // GTSV_INTERLEAVED_BATCH_KERNEL_H_
