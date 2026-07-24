/**
 * ----------------------------------------------------------------------------------------------------------
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under
 * the terms and conditions of CANN Open Software License Agreement Version 2.0
 * (the "License"). Please refer to the License for details. You may not use
 * this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
 * AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
 * FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
 * for the full text of the License.
 * ----------------------------------------------------------------------------------------------------------
 */

/*!
 * \file csrsort_kernel.h
 * \brief aclsparseXcsrsort kernel_do 声明（Host / Kernel 共用）。
 */

#ifndef CSRSORT_KERNEL_H_
#define CSRSORT_KERNEL_H_

#include "csrsort_tiling_data.h"

void csrsort_kernel_do(uint8_t *csrRowPtr, uint8_t *csrColInd, uint8_t *P,
                       uint8_t *workspace, const CsrsortTilingData &tiling,
                       uint32_t numBlocks, void *stream);

#endif // CSRSORT_KERNEL_H_
