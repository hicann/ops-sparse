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

#ifndef CUBE_SPMM_H_
#define CUBE_SPMM_H_

#include <climits>
#include <cstdint>
#include <limits>

#ifndef __gm__
#define __gm__
#endif

#ifndef GM_ADDR
#define GM_ADDR uint8_t *
#endif

namespace cube_spmm {

constexpr int32_t kTileM = 16;
constexpr int32_t kTileK = 16;
constexpr int32_t kTileN = 16;
constexpr int32_t kN = 128;

// N-direction processing tile size. Local buffers for B/CO1 are allocated for
// min(N, kNChunkSize) columns to stay within hardware capacity.
constexpr int32_t kNChunkSize = 512;

// Maximum value that fits in the uint16_t fields used by AscendC DataCopy APIs.
constexpr int64_t kMaxUint16 = 65535LL;

// Validates that the matrix dimensions M, K, N are within the range supported by
// the current Cube SpMM implementation. M, K, N are stored as int32_t in
// TilingData and bounded by INT32_MAX; only the BCSR block-count product M * K
// is allowed up to INT64_MAX, which requires rwPtr to be int64_t.
// DataCopyParams.blockLen is uint16_t, which bounds N.
// Returns false if any limit is violated.
inline bool AreDimensionsSupported(int64_t M, int64_t K, int64_t N)
{
    constexpr int64_t kMaxInt32 = static_cast<int64_t>(INT32_MAX);
    constexpr int64_t kMaxInt64 = std::numeric_limits<int64_t>::max();
    if (M <= 0 || M > kMaxInt32 ||
        K <= 0 || K > kMaxInt32 ||
        N <= 0 || N > kMaxInt32) {
        return false;
    }
    // Keep M * K within int64_t so that BCSR block-index products never overflow.
    if (M > kMaxInt64 / K) {
        return false;
    }
    // DataCopyParams.blockLen is uint16_t (measured in 32B units). The largest
    // blockLen written by the kernel is for the C matrix: kTileM rows * N cols *
    // sizeof(float) / 32. This bounds the supported N.
    if (N > (kMaxUint16 * 32) / (static_cast<int64_t>(kTileM) * static_cast<int64_t>(sizeof(float)))) {
        return false;
    }
    return true;
}

// Tiling data shared between host and kernel. Passed to the kernel by value
// as a launch argument (travels with the <<<>>> kernel launch), so no H2D
// copy into workspace is needed.
typedef struct CubeSpmmTilingData {
    int32_t M;
    int32_t N;
    int32_t K;
    uint32_t usedCoreNum;
    uint32_t lastKLength;
    int32_t tileM;  // M-direction tile/block size, currently must be 16.
    int32_t tileN;  // N-direction tile size, currently must be 16.
    int32_t tailM;  // Number of valid rows in the last row window (1..tileM).
    int32_t bLd;    // Leading dimension of dense B in GM (>= N).
    int32_t cLd;    // Leading dimension of dense C in GM (>= N).
} CubeSpmmTilingData;

}  // namespace cube_spmm

// Cube SpMM requires no device workspace: tiling travels with the kernel
// launch as a by-value argument, and the kernel uses only local buffers.

#endif  // CUBE_SPMM_H_
