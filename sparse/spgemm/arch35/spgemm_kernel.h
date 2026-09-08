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

#ifndef SPGEMM_KERNEL_H_
#define SPGEMM_KERNEL_H_

#include <cstdint>
#include "spgemm_tiling_data.h"

#ifndef GM_ADDR
#define GM_ADDR uint8_t *
#endif

extern "C" {

void spgemm_validate_kernel_do(
    GM_ADDR rowPtrA, GM_ADDR colIndA, GM_ADDR rowPtrB, GM_ADDR colIndB,
    GM_ADDR regularOffsets, GM_ADDR regularTotal, GM_ADDR regular,
    GM_ADDR error, const SpGemmValidateTilingData &tiling,
    uint32_t numBlocks, void *stream);

void spgemm_work_kernel_do(
    GM_ADDR rowPtrA, GM_ADDR colIndA, GM_ADDR rowPtrB,
    GM_ADDR productCounts, GM_ADDR regular, GM_ADDR error,
    const SpGemmWorkTilingData &tiling, uint32_t numBlocks, void *stream);

void spgemm_scan_i64_kernel_do(
    GM_ADDR counts, GM_ADDR offsets, GM_ADDR blockSums,
    GM_ADDR blockOffsets, GM_ADDR total, GM_ADDR regular, GM_ADDR error,
    const SpGemmScanTilingData &tiling, uint32_t numBlocks, void *stream);

void spgemm_compute_kernel_do(
    GM_ADDR rowPtrA, GM_ADDR colIndA, GM_ADDR valA,
    GM_ADDR rowPtrB, GM_ADDR colIndB, GM_ADDR valB,
    GM_ADDR oldRowPtrC, GM_ADDR oldColIndC, GM_ADDR oldValC,
    GM_ADDR productOffsets, GM_ADDR cursors,
    GM_ADDR candidateCols, GM_ADDR candidateVals, GM_ADDR uniqueCounts,
    GM_ADDR error, const SpGemmComputeTilingData &tiling,
    uint32_t numBlocks, void *stream);

void spgemm_scan_i32_kernel_do(
    GM_ADDR counts, GM_ADDR offsets, GM_ADDR blockSums,
    GM_ADDR blockOffsets, GM_ADDR total, GM_ADDR error,
    const SpGemmScanTilingData &tiling, uint32_t numBlocks, void *stream);

void spgemm_copy_kernel_do(
    GM_ADDR productOffsets, GM_ADDR uniqueCounts,
    GM_ADDR candidateCols, GM_ADDR candidateVals,
    GM_ADDR rowPtrC, GM_ADDR colIndC, GM_ADDR valC,
    const SpGemmCopyTilingData &tiling, uint32_t numBlocks, void *stream);

}  // extern "C"

#endif  // SPGEMM_KERNEL_H_
