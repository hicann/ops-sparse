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
 * \file sddmm_kernel.h
 * \brief sddmm kernel_launch 声明（Host / Kernel 共用）。
 */

#ifndef SDDMM_ARCH22_KERNEL_H_
#define SDDMM_ARCH22_KERNEL_H_

#include <cstdint>
#include "sddmm.h"
#include "cann_ops_sparse.h"

#ifndef GM_ADDR
#define GM_ADDR uint8_t *
#endif

extern "C" aclsparseStatus_t sddmm_kernel_launch(
    GM_ADDR matA, GM_ADDR matB,
    GM_ADDR csrRowOffsets, GM_ADDR csrColIndices, GM_ADDR csrValues,
    const SddmmTilingData &tiling,
    uint32_t numBlocks, void *stream);

#endif // SDDMM_ARCH22_KERNEL_H_
