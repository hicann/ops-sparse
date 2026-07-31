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

#ifndef DENSETOSPARSE_KERNEL_H_
#define DENSETOSPARSE_KERNEL_H_

#include "densetosparse_tiling_data.h"

#ifndef GM_ADDR
#define GM_ADDR uint8_t *
#endif

void densetosparse_analysis_kernel_do(
    GM_ADDR dense, GM_ADDR offsets, GM_ADDR workspace, uint32_t numBlocks,
    const DenseToSparseTilingData &tiling, void *stream);
void densetosparse_convert_kernel_do(
    GM_ADDR dense, GM_ADDR workspace, GM_ADDR offsets, GM_ADDR indices,
    GM_ADDR rowIndices, GM_ADDR colIndices, GM_ADDR values, GM_ADDR ellColInd,
    uint32_t numBlocks, const DenseToSparseTilingData &tiling, void *stream);

#endif // DENSETOSPARSE_KERNEL_H_
