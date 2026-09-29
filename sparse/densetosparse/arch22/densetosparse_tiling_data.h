/**
 * ----------------------------------------------------------------------------------------------------------
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software: you can redistribute it and/or modify it under the terms of conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ----------------------------------------------------------------------------------------------------------
 */

#ifndef DENSETOSPARSE_ARCH22_TILING_DATA_H_
#define DENSETOSPARSE_ARCH22_TILING_DATA_H_

#include <cstdint>

#ifndef GM_ADDR
#define GM_ADDR uint8_t *
#endif

// Minor-dimension elements covered by one CSR/CSC count/convert unit
// (scheduling granularity, not a hardware limit).
constexpr uint64_t kD2sMinorChunk = 4096;
// Flat logical elements per COO unit.
constexpr uint64_t kD2sCooTile = 1024;
// uint64 entries per scan segment. 128 (instead of arch35's 1024) keeps all
// AIV cores busy on one level and shortens the serial per-segment add chain.
constexpr uint64_t kD2sScanBatch = 128;
// Host-side work quantum used to decide the launch core count.
constexpr uint64_t kD2sWorkPerCore = 4096;

// Workspace layout (uint64 level protocol, header shared with arch35):
//   status 4B @0 | nnz 8B @32 | level0 @64 | scan levels (each 32B aligned)
constexpr uint64_t kD2sStatusOffset = 0;
constexpr uint64_t kD2sNnzOffset = 32;
constexpr uint64_t kD2sLevel0Offset = 64;
constexpr uint64_t kD2sWorkspaceHeaderBytes = 64;
constexpr uint32_t kD2sWorkspaceAlignment = 32;

constexpr int32_t kD2sDeviceStatusSuccess = 0;
constexpr int32_t kD2sDeviceStatusInvalid = 1;

// format/order encodings = aclsparseFormat_t / aclsparseOrder_t values.
constexpr uint32_t kD2sFormatCoo = 0;
constexpr uint32_t kD2sFormatCsr = 1;
constexpr uint32_t kD2sFormatCsc = 2;
constexpr uint32_t kD2sFormatBell = 3;
constexpr uint32_t kD2sOrderRow = 0;
constexpr uint32_t kD2sOrderCol = 1;

// Judge-chain tile capacity in logical elements: real dtypes 4096;
// complex64 is modeled as 2 interleaved fp32 (T=float, GM_STRIDE=2), so its
// 2048 logical elements also expand to 4096 float operands.
constexpr uint32_t kD2sTileElems = 4096;
constexpr uint32_t kD2sTileElemsComplex = 2048;

struct DenseToSparseTilingData {
    // Geometry (u64 to survive INT32_MAX dims before validation clamps).
    uint64_t rows;
    uint64_t cols;
    uint64_t ld;
    uint64_t nnz;          // Analysis-published nnz (BELL/payload sized by it)
    uint64_t ellBlockSize; // BELL block size
    uint64_t ellCols;      // BELL ellCols
    uint64_t unitCount;    // total count/convert units (scan level0 depth)
    // Workspace byte offsets.
    uint64_t statusOffset;
    uint64_t nnzOffset;
    uint64_t level0Offset;
    // Enum encodings matching aclsparse public headers.
    uint32_t format;       // COO/CSR/CSC/BLOCKED_ELL
    uint32_t order;        // ROW/COL
    uint32_t base;         // index base 0/1
    uint32_t elementBytes; // 1/2/4/8 (complex64 = 8)
    uint32_t numBlocks;    // launched AIV core count
    uint32_t tileLen;      // logical elements per judge tile
};

#endif // DENSETOSPARSE_ARCH22_TILING_DATA_H_
