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
 * @file csr2coo_kernel.h
 * @brief aclsparseXcsr2coo kernel_do function declaration (shared between Host and Kernel).
 *
 * Tiling is passed by const reference (no H2D copy); kernel receives it by value
 * via the launch parameter automatically.
 */

#ifndef CSR2COO_KERNEL_H_
#define CSR2COO_KERNEL_H_

#include "csr2coo_tiling_data.h"

#ifndef GM_ADDR
#define GM_ADDR uint8_t *
#endif

extern "C" void csr2coo_kernel_do(
    GM_ADDR gmRowPtr,
    GM_ADDR gmCooRowInd,
    const csr2coo::Csr2CooTilingData &tiling,
    uint32_t numBlocks,
    void *stream);

#endif // CSR2COO_KERNEL_H_
