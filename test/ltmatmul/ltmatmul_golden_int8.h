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

#ifndef TEST_MATMUL_GOLDEN_INT8_H_
#define TEST_MATMUL_GOLDEN_INT8_H_

// =============================================================================
// INT8-specific golden functions for aclsparseLtMatmul.
//
// Split from matmul_golden.h to reduce file size [codecheck: oversized header].
// Included by matmul_golden.h; users who include matmul_golden.h automatically
// get these functions.
//
// Dependencies (defined in matmul_golden_types.h, available via include):
//   - MatmulPruneInputA (prune helper)
// =============================================================================

#include "ltmatmul_golden_types.h"  // shared types, conversions, prune helpers

namespace sparse_test {

// -----------------------------------------------------------------------------
// [codecheck-dup] MatmulInt32Accumulate extracted from the common (i,p,j)
// INT32 accumulation loop shared by MatmulAlphaBetaInt32 (scalar alpha/beta)
// and MatmulAlphaBetaInt32Vec (per-row vector alpha/beta).
// D_int32[i][j] = Σ A_pruned[i][p] * B[p][j]  (int32 accumulation)
// -----------------------------------------------------------------------------
inline void MatmulInt32Accumulate(
    const std::vector<int8_t>& A_pruned,
    const std::vector<int8_t>& Bf,
    std::vector<int32_t>& Df,
    int32_t m, int32_t k, int32_t n)
{
    const size_t mn = static_cast<size_t>(m) * static_cast<size_t>(n);
    Df.assign(mn, 0);
    for (int32_t i = 0; i < m; ++i) {
        const int8_t* aRow = A_pruned.data() + static_cast<int64_t>(i) * k;
        int32_t* dRow = Df.data() + static_cast<int64_t>(i) * n;
        for (int32_t p = 0; p < k; ++p) {
            int8_t a = aRow[p];
            if (a == 0) { continue; }  // skip pruned zeros
            const int8_t* bRow = Bf.data() + static_cast<int64_t>(p) * n;
            for (int32_t j = 0; j < n; ++j) {
                dRow[j] += static_cast<int32_t>(a) * static_cast<int32_t>(bRow[j]);
            }
        }
    }
}

// -----------------------------------------------------------------------------
// INT32 accumulation matmul for INT8 input.
// D_int32[i][j] = Σ A_pruned[i][p] * B[p][j]  (int32 accumulation)
// Then applies alpha/beta: D = round(α·float(acc) + β·float(C))
// When α=1.0, β=0.0: D = acc (exact integer, no rounding needed).
// -----------------------------------------------------------------------------
inline void MatmulAlphaBetaInt32(
    const std::vector<int8_t>& A_pruned,
    const std::vector<int8_t>& Bf,
    const std::vector<int8_t>& Cf,
    std::vector<int32_t>& Df,
    int32_t m, int32_t k, int32_t n,
    float alpha, float beta)
{
    MatmulInt32Accumulate(A_pruned, Bf, Df, m, k, n);
    // Apply alpha/beta scaling (only when non-trivial).
    if (alpha != 1.0f || beta != 0.0f) {
        for (int32_t i = 0; i < m; ++i) {
            const int8_t* cRow = Cf.data() + static_cast<int64_t>(i) * n;
            int32_t* dRow = Df.data() + static_cast<int64_t>(i) * n;
            for (int32_t j = 0; j < n; ++j) {
                float scaled = alpha * static_cast<float>(dRow[j])
                             + beta * static_cast<float>(cRow[j]);
                dRow[j] = static_cast<int32_t>(std::round(scaled));
            }
        }
    }
}

// -----------------------------------------------------------------------------
// Per-row vector scaling variant: alpha/beta are float[M] arrays.
// D[i,j] = alphaVec[i] * acc[i,j] + betaVec[i] * C[i,j]
// [codecheck-dup] Reuses MatmulInt32Accumulate for the common accumulation loop.
// -----------------------------------------------------------------------------
inline void MatmulAlphaBetaInt32Vec(
    const std::vector<int8_t>& A_pruned,
    const std::vector<int8_t>& Bf,
    const std::vector<int8_t>& Cf,
    std::vector<int32_t>& Df,
    int32_t m, int32_t k, int32_t n,
    const std::vector<float>& alphaVec,
    const std::vector<float>& betaVec)
{
    MatmulInt32Accumulate(A_pruned, Bf, Df, m, k, n);
    for (int32_t i = 0; i < m; ++i) {
        const int8_t* cRow = Cf.data() + static_cast<int64_t>(i) * n;
        int32_t* dRow = Df.data() + static_cast<int64_t>(i) * n;
        float aV = alphaVec[static_cast<size_t>(i)];
        float bV = betaVec.empty() ? 0.0f : betaVec[static_cast<size_t>(i)];
        if (aV != 1.0f || bV != 0.0f) {
            for (int32_t j = 0; j < n; ++j) {
                float scaled = aV * static_cast<float>(dRow[j])
                             + bV * static_cast<float>(cRow[j]);
                dRow[j] = static_cast<int32_t>(std::round(scaled));
            }
        }
    }
}

// -----------------------------------------------------------------------------
// Saturate INT32 → INT8 cast.
// -----------------------------------------------------------------------------
inline void SaturateCastToInt8(const std::vector<int32_t>& src, std::vector<int8_t>& dst)
{
    dst.resize(src.size());
    for (size_t i = 0; i < src.size(); ++i) {
        float v = static_cast<float>(src[i]);
        if (v > 127.0f) { v = 127.0f; }
        if (v < -128.0f) { v = -128.0f; }
        int32_t rounded = static_cast<int32_t>(std::round(v));
        if (rounded > 127) { rounded = 127; }
        if (rounded < -128) { rounded = -128; }
        dst[i] = static_cast<int8_t>(rounded);
    }
}

// -----------------------------------------------------------------------------
// INT8 prune: prune int8_t matrix (group=4, keep=2, compare absolute values).
// Promotes to FP32, prunes, converts back (zeros stay zeros, kept values exact).
//
// pruneAlg dispatch (mirrors FP32/FP16 MatmulPruneInputAWithAlg in
// ltmatmul_golden_types.h):
//   "TILE"  -> SpMMAPruneGolden<int8_t> TILE path (4x4 tile 2:2, row/col joint
//              constraint; edge tiles degenerate to STRIP).
//   "STRIP" -> existing 1D top-N per group (default, backward compatible).
// -----------------------------------------------------------------------------
inline void PruneInt8Matrix(const std::vector<int8_t>& A, std::vector<int8_t>& A_pruned,
                             int32_t m, int32_t k, bool isColOrder,
                             const std::string& pruneAlg = "STRIP")
{
    if (pruneAlg == "TILE") {
        // TILE prune: delegate to SpMMAPruneGolden<int8_t>'s TILE path.
        // SpMMAPruneGolden promotes int8->FP32, prunes in FP32 (4x4 tile 2:2),
        // and converts back via AlgSetAttrDtypeTrait<int8_t>::fromFp32
        // (round + saturate). Kept int8 values are exact (int8->FP32 is lossless
        // and round of an integer float is itself); zeros stay zero.
        // alongRow = !isColOrder for NON_TRANSPOSE, matching the STRIP path and
        // the FP32/FP16 MatmulPruneInputAWithAlg convention.
        SpMMAPruneGolden<int8_t>(A, A_pruned, m, k, 0, 1, false, !isColOrder, "TILE");
        return;
    }
    // STRIP path: promote to FP32, prune (1D top-N), convert back.
    std::vector<float> Af(static_cast<size_t>(m) * k);
    for (size_t i = 0; i < Af.size(); ++i) {
        Af[i] = static_cast<float>(A[i]);
    }
    std::vector<float> Af_pruned;
    MatmulPruneInputA<int8_t>(Af, Af_pruned, m, k, isColOrder);
    A_pruned.resize(Af.size());
    for (size_t i = 0; i < Af.size(); ++i) {
        A_pruned[i] = (Af_pruned[i] == 0.0f)
            ? static_cast<int8_t>(0)
            : static_cast<int8_t>(std::round(Af_pruned[i]));
    }
}

// -----------------------------------------------------------------------------
// Golden entry: sparse×dense INT8 (INT32 accumulation).
// output_dtype: "INT32" → D is int32_t; "INT8" → D is int8_t (saturated).
// pruneAlg: "STRIP" (default) or "TILE".
// -----------------------------------------------------------------------------
inline void LtMatmulGoldenRunInt8(
    const std::vector<int8_t>& A, const std::vector<int8_t>& B,
    const std::vector<int8_t>& C,
    std::vector<int32_t>& D_int32,
    int32_t m, int32_t k, int32_t n,
    float alpha, float beta,
    bool isColOrder, bool pruneB,
    const std::string& pruneAlg = "STRIP")
{
    // Step 1: prune the sparse matrix (TILE or STRIP, per pruneAlg).
    std::vector<int8_t> A_pruned;
    std::vector<int8_t> B_pruned;
    if (pruneB) {
        PruneInt8Matrix(B, B_pruned, k, n, isColOrder, pruneAlg);
    } else {
        PruneInt8Matrix(A, A_pruned, m, k, isColOrder, pruneAlg);
    }

    // Step 2: INT32 accumulation matmul.
    if (pruneB) {
        MatmulAlphaBetaInt32(A, B_pruned, C, D_int32, m, k, n, alpha, beta);
    } else {
        MatmulAlphaBetaInt32(A_pruned, B, C, D_int32, m, k, n, alpha, beta);
    }
}

inline void LtMatmulGoldenRunInt8WithNpuPrune(
    const std::vector<int8_t>& A, const std::vector<int8_t>& B,
    const std::vector<int8_t>& C,
    std::vector<int8_t>& D_int8,
    int32_t m, int32_t k, int32_t n,
    float alpha, float beta,
    const std::vector<int8_t>& npuPrunedA,
    bool isSparseA)
{
    std::vector<int32_t> D_int32;
    if (isSparseA && !npuPrunedA.empty()) {
        MatmulAlphaBetaInt32(npuPrunedA, B, C, D_int32, m, k, n, alpha, beta);
    } else if (!isSparseA) {
        std::vector<int8_t> B_pruned;
        PruneInt8Matrix(B, B_pruned, k, n, false);
        MatmulAlphaBetaInt32(A, B_pruned, C, D_int32, m, k, n, alpha, beta);
    } else {
        std::vector<int8_t> A_pruned;
        PruneInt8Matrix(A, A_pruned, m, k, false);
        MatmulAlphaBetaInt32(A_pruned, B, C, D_int32, m, k, n, alpha, beta);
    }
    SaturateCastToInt8(D_int32, D_int8);
}

inline void LtMatmulGoldenRunInt8VecWithNpuPrune(
    const std::vector<int8_t>& A, const std::vector<int8_t>& B,
    const std::vector<int8_t>& C,
    std::vector<int8_t>& D_int8,
    int32_t m, int32_t k, int32_t n,
    const std::vector<float>& alphaVec,
    const std::vector<float>& betaVec,
    const std::vector<int8_t>& npuPrunedA,
    bool isSparseA)
{
    std::vector<int32_t> D_int32;
    if (isSparseA && !npuPrunedA.empty()) {
        MatmulAlphaBetaInt32Vec(npuPrunedA, B, C, D_int32, m, k, n, alphaVec, betaVec);
    } else if (!isSparseA) {
        std::vector<int8_t> B_pruned;
        PruneInt8Matrix(B, B_pruned, k, n, false);
        MatmulAlphaBetaInt32Vec(A, B_pruned, C, D_int32, m, k, n, alphaVec, betaVec);
    } else {
        std::vector<int8_t> A_pruned;
        PruneInt8Matrix(A, A_pruned, m, k, false);
        MatmulAlphaBetaInt32Vec(A_pruned, B, C, D_int32, m, k, n, alphaVec, betaVec);
    }
    SaturateCastToInt8(D_int32, D_int8);
}

// -----------------------------------------------------------------------------
// Golden entry: dense×dense INT8 (INT32 accumulation, no prune).
// -----------------------------------------------------------------------------
inline void LtMatmulDenseGoldenRunInt8(
    const std::vector<int8_t>& A, const std::vector<int8_t>& B,
    const std::vector<int8_t>& C,
    std::vector<int32_t>& D_int32,
    int32_t m, int32_t k, int32_t n,
    float alpha, float beta)
{
    // No prune — direct INT32 matmul on dense A.
    MatmulAlphaBetaInt32(A, B, C, D_int32, m, k, n, alpha, beta);
}

inline void LtMatmulDenseGoldenRunInt8ToInt8(
    const std::vector<int8_t>& A, const std::vector<int8_t>& B,
    const std::vector<int8_t>& C,
    std::vector<int8_t>& D_int8,
    int32_t m, int32_t k, int32_t n,
    float alpha, float beta)
{
    std::vector<int32_t> D_int32;
    LtMatmulDenseGoldenRunInt8(A, B, C, D_int32, m, k, n, alpha, beta);
    SaturateCastToInt8(D_int32, D_int8);
}

// -----------------------------------------------------------------------------
// Vector scaling golden entries: alpha/beta are per-row float[M] vectors.
// -----------------------------------------------------------------------------

// sparse×dense INT8 with vector scaling (INT32 output)
inline void LtMatmulGoldenRunInt8Vec(
    const std::vector<int8_t>& A, const std::vector<int8_t>& B,
    const std::vector<int8_t>& C,
    std::vector<int32_t>& D_int32,
    int32_t m, int32_t k, int32_t n,
    const std::vector<float>& alphaVec, const std::vector<float>& betaVec,
    bool isColOrder, bool pruneB,
    const std::string& pruneAlg = "STRIP")
{
    std::vector<int8_t> A_pruned;
    std::vector<int8_t> B_pruned;
    if (pruneB) {
        PruneInt8Matrix(B, B_pruned, k, n, isColOrder, pruneAlg);
        MatmulAlphaBetaInt32Vec(A, B_pruned, C, D_int32, m, k, n, alphaVec, betaVec);
    } else {
        PruneInt8Matrix(A, A_pruned, m, k, isColOrder, pruneAlg);
        MatmulAlphaBetaInt32Vec(A_pruned, B, C, D_int32, m, k, n, alphaVec, betaVec);
    }
}

// dense×dense INT8 with vector scaling (INT32 output)
inline void LtMatmulDenseGoldenRunInt8Vec(
    const std::vector<int8_t>& A, const std::vector<int8_t>& B,
    const std::vector<int8_t>& C,
    std::vector<int32_t>& D_int32,
    int32_t m, int32_t k, int32_t n,
    const std::vector<float>& alphaVec, const std::vector<float>& betaVec)
{
    MatmulAlphaBetaInt32Vec(A, B, C, D_int32, m, k, n, alphaVec, betaVec);
}

// dense×dense INT8 with vector scaling (INT8 output)
inline void LtMatmulDenseGoldenRunInt8ToInt8Vec(
    const std::vector<int8_t>& A, const std::vector<int8_t>& B,
    const std::vector<int8_t>& C,
    std::vector<int8_t>& D_int8,
    int32_t m, int32_t k, int32_t n,
    const std::vector<float>& alphaVec, const std::vector<float>& betaVec)
{
    std::vector<int32_t> D_int32;
    LtMatmulDenseGoldenRunInt8Vec(A, B, C, D_int32, m, k, n, alphaVec, betaVec);
    SaturateCastToInt8(D_int32, D_int8);
}

}  // namespace sparse_test

#endif  // TEST_MATMUL_GOLDEN_INT8_H_
