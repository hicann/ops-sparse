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
 * \file gtsv2_strided_batch_kernel.h
 * \brief aclsparseSgtsv2StridedBatch kernel_do 声明（Host / Kernel 共用）。
 *
 * 独立 kernel.h 声明 kernel_do，禁止 host.cpp 中 extern 前向声明。
 */

#ifndef GTSV2_STRIDED_BATCH_KERNEL_H_
#define GTSV2_STRIDED_BATCH_KERNEL_H_

#include <cstdint>
#include "gtsv2_strided_batch_tiling_data.h"

#ifndef GM_ADDR
#define GM_ADDR uint8_t *
#endif

extern "C" {

void gtsv2_strided_batch_kernel_do(
    GM_ADDR gmDl, GM_ADDR gmD, GM_ADDR gmDu, GM_ADDR gmX, GM_ADDR gmBuffer,
    const Gtsv2StridedBatchTilingData &tiling,
    uint32_t numBlocks,
    void *stream);

}  // extern "C"

#endif  // GTSV2_STRIDED_BATCH_KERNEL_H_
