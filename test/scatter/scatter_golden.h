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

#ifndef TEST_SCATTER_SCATTER_GOLDEN_H_
#define TEST_SCATTER_SCATTER_GOLDEN_H_

#include <cstdint>
#include <string>
#include <vector>

namespace sparse_test {

// ============================================================================
// CPU golden reference for aclsparseScatter.
//
// Semantics (aligned with cusparseScatter):
//   for i in [0, nnz):
//       Y[indices[i] - idxBase] = values[i]   // byte-level copy
//
// This is a pure data-rearrangement operation (no floating-point arithmetic).
// Values are copied as raw bytes, preserving bit-exact representation for
// all dtypes (FP32, FP16, BF16) including ±0, ±inf, NaN, subnormals.
//
// NOTE: scatter 无对应的 MKL / Eigen 标准稀疏库接口（cusparseScatter 本身即
// cuSPARSE 专有接口），故 golden 采用自建逐位实现。实现逐位正确，已由 L0/L1
// 用例（含 ±0 / ±inf / NaN / subnormal 特殊值逐字节比对）验证。
//
// Template parameters:
//   IdxT  — index type (int32_t for I32, int64_t for I64)
//   valSize — sizeof(value type) in bytes (4 for FP32, 2 for FP16/BF16)
//
// Parameters:
//   dnSize       — dense vector Y length
//   nnz          — nonzero count
//   indices      — host indices array (already includes idxBase offset)
//   valuesBytes  — nnz * valSize bytes (raw value bit patterns)
//   yInitBytes   — dnSize * valSize bytes (Y initial values)
//   idxBase      — 0 (ZERO) or 1 (ONE); golden converts to 0-based internally
//
// Returns: Y bytes after scatter (dnSize * valSize bytes)
// ============================================================================
template <typename IdxT>
inline std::vector<uint8_t> ScatterGolden(
    int64_t dnSize,
    int64_t nnz,
    const IdxT* indices,
    const std::vector<uint8_t>& valuesBytes,
    const std::vector<uint8_t>& yInitBytes,
    int idxBase,
    size_t valSize)
{
    // Copy initial Y (so unscattered positions keep their original values)
    std::vector<uint8_t> yBytes = yInitBytes;

    // dnSize is the logical Y length; not directly used in golden since
    // yInitBytes already encodes the full Y array size. Kept in signature
    // for API documentation alignment with the operator.
    (void)dnSize;

    // nnz == 0: Y must remain unchanged
    if (nnz == 0) {
        return yBytes;
    }

    // Scatter: Y[indices[i] - idxBase] = values[i], byte-by-byte copy
    for (int64_t i = 0; i < nnz; i++) {
        int64_t pos = static_cast<int64_t>(indices[i]) - static_cast<int64_t>(idxBase);
        // pos range is guaranteed valid [0, dnSize) by test-side index generation;
        // out-of-bounds cases are handled by L2 exception tests (not golden path)
        const uint8_t* src = &valuesBytes[static_cast<size_t>(i) * valSize];
        uint8_t* dst = &yBytes[static_cast<size_t>(pos) * valSize];
        for (size_t b = 0; b < valSize; b++) {
            dst[b] = src[b];
        }
    }
    return yBytes;
}

// ============================================================================
// Convenience dispatch: compute golden Y bytes given dtype string.
// Accepts indices as raw bytes + idxType string, dispatches to correct
// ScatterGolden<int32_t> or ScatterGolden<int64_t>.
// ============================================================================
inline std::vector<uint8_t> ComputeScatterGolden(
    int64_t dnSize,
    int64_t nnz,
    const std::vector<uint8_t>& indicesBytes,
    const std::vector<uint8_t>& valuesBytes,
    const std::vector<uint8_t>& yInitBytes,
    const std::string& idxType,
    int idxBase,
    size_t valSize)
{
    if (idxType == "I64") {
        const int64_t* idxPtr = reinterpret_cast<const int64_t*>(indicesBytes.data());
        return ScatterGolden<int64_t>(dnSize, nnz, idxPtr, valuesBytes, yInitBytes, idxBase, valSize);
    }
    // default I32
    const int32_t* idxPtr = reinterpret_cast<const int32_t*>(indicesBytes.data());
    return ScatterGolden<int32_t>(dnSize, nnz, idxPtr, valuesBytes, yInitBytes, idxBase, valSize);
}

}  // namespace sparse_test

#endif  // TEST_SCATTER_SCATTER_GOLDEN_H_
