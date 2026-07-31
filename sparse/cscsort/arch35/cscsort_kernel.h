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
 * \file cscsort_kernel.h
 * \brief aclsparseXcscsort kernel_do 声明（Host / Kernel 共用）。
 */

#ifndef CSCSORT_KERNEL_H_
#define CSCSORT_KERNEL_H_

#include "cscsort_tiling_data.h"

#ifndef GM_ADDR
#define GM_ADDR uint8_t *
#endif

void cscsort_kernel_do(
    GM_ADDR cscColPtr,
    GM_ADDR cscRowInd,
    GM_ADDR P,
    GM_ADDR workspace,
    const CscsortTilingData &tiling,
    uint32_t numBlocks,
    void *stream);

#endif  // CSCSORT_KERNEL_H_
