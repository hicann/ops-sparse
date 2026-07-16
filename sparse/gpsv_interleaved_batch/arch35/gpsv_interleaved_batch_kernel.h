/**
 * ----------------------------------------------------------------------------------------------------------
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ----------------------------------------------------------------------------------------------------------
 */

/*!
 * \file gpsv_interleaved_batch_kernel.h
 * \brief aclsparseSgpsvInterleavedBatch kernel_do declaration (Host / Kernel shared).
 */

#ifndef GPSV_INTERLEAVED_BATCH_KERNEL_H_
#define GPSV_INTERLEAVED_BATCH_KERNEL_H_

#include <cstdint>
#include "gpsv_interleaved_batch_tiling_data.h"

#ifndef GM_ADDR
#define GM_ADDR uint8_t *
#endif

// One warp (32 threads) per block: independent batches are processed in
// parallel within the same warp via the grid-stride pattern. Raising above
// 32 is possible but requires verifying UB/register pressure per thread.
inline constexpr uint32_t kGpsvBatchPerBlock = 32u;

void gpsv_interleaved_batch_kernel_do(
    GM_ADDR ds,
    GM_ADDR dl,
    GM_ADDR d,
    GM_ADDR du,
    GM_ADDR dw,
    GM_ADDR x,
    GM_ADDR workspace,
    const GpsvInterleavedBatchTilingData &tiling,
    uint32_t numBlocks,
    void *stream);

#endif // GPSV_INTERLEAVED_BATCH_KERNEL_H_
