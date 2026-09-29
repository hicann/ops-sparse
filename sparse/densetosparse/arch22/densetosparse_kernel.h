/**
 * ----------------------------------------------------------------------------------------------------------
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software: you can redistribute it and/or modify it under the terms of conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You should not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ----------------------------------------------------------------------------------------------------------
 */

#ifndef DENSETOSPARSE_ARCH22_KERNEL_H_
#define DENSETOSPARSE_ARCH22_KERNEL_H_

#include "densetosparse_tiling_data.h"

// Analysis orchestration (header -> count -> multi-level scan -> add-base ->
// offsets[CSR/CSC] -> total). BELL performs no structure discovery. Defined in
// densetosparse_count_kernel.cpp.
void densetosparse_analysis_kernel_do(GM_ADDR dense, GM_ADDR offsets,
    GM_ADDR workspace, uint32_t numBlocks,
    const DenseToSparseTilingData &tiling, void *stream);

// Convert orchestration. BELL dispatches to its dedicated translation unit;
// CSR/CSC/COO trust the Analysis-published workspace prefix (same-buffer
// protocol enforced on the host side). Defined in densetosparse_kernel.cpp.
void densetosparse_convert_kernel_do(GM_ADDR dense, GM_ADDR workspace,
    GM_ADDR offsets, GM_ADDR indices, GM_ADDR rowIndices, GM_ADDR colIndices,
    GM_ADDR values, GM_ADDR ellColInd, uint32_t numBlocks,
    const DenseToSparseTilingData &tiling, void *stream);

// BELL-only entry, implemented in densetosparse_bell_kernel.cpp (separate
// translation unit so its transpose vector chains cannot degrade the
// CSR/CSC/COO code generation).
void densetosparse_bell_kernel_do(GM_ADDR dense, GM_ADDR ellColInd,
    GM_ADDR values, uint32_t numBlocks,
    const DenseToSparseTilingData &tiling, void *stream);

#endif // DENSETOSPARSE_ARCH22_KERNEL_H_
