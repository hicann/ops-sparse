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

#ifndef SPSV_TILING_DATA_H_
#define SPSV_TILING_DATA_H_

#include <cstdint>

// Maximum number of SIMT threads per block, shared by kernel and host code.
// Canonical definition; spsv.h re-exports into namespace spsv.
constexpr uint32_t kSimtMaxThreads = 2048u;

// TilingData is ~160 bytes, passed by value as kernel parameter.
// This is within the kernel parameter size limit for 351x.
struct SpsvTilingData {
    int64_t m;
    int64_t nnz;

    float alpha;
    int32_t fillMode;
    int32_t diagType;
    int32_t opA;
    int32_t format;
    int32_t indexType;     // 0=I32, 1=I64 for rowPtr (ptrType)
    int32_t colIndType;    // 0=I32, 1=I64 for colInd (idxType); may differ from indexType
    int32_t idxBase;       // 0=0-based, 1=1-based (Fortran indexing)

    int64_t levelPtrOffset;
    int64_t levelRowOffset;
    int64_t diagPtrOffset;
    int64_t validCountOffset;
    // numLevels: set to 0 by host in tiling params. At runtime, the analysis
    // kernel writes the computed numLevels into workspace (as SpsvTilingData
    // at workspace offset 0). The solve kernel reads numLevels from workspace,
    // NOT from this tiling field.
    int32_t numLevels;

    // permType: 0 = int32_t workspace arrays (perm, transPerm, diagPtr);
    //           1 = int64_t workspace arrays (used when nnz > INT32_MAX with I64 index).
    int32_t permType;

    int64_t csrRowPtrOffset;
    int64_t csrColIndOffset;
    int64_t csrValuesOffset;
    int64_t permOffset;

    int64_t transRowPtrOffset;
    int64_t transColIndOffset;
    int64_t transValuesOffset;
    int64_t transPermOffset;

    int32_t numSlices;
    int32_t sliceWidth;

    uint32_t nthreads;
    uint32_t numBlocks;

    // Device pointer to alpha value (float). 0 means "invalid / not set";
    // when non-zero, the kernel reads alpha from this device address instead
    // of the 'alpha' field above. The caller must ensure the pointer is at
    // least 4-byte aligned (float alignment requirement).
    uint64_t alphaDevicePtr;
};

#endif // SPSV_TILING_DATA_H_
