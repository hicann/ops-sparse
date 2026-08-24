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

#ifndef TEST_MATMUL_GOLDEN_H_
#define TEST_MATMUL_GOLDEN_H_

// =============================================================================
// CPU Golden reference for aclsparseLtMatmul (v2: 4 dtype × 2 path).
//
// Paths:
//   sparse×dense: D = α·op(A_pruned)·op(B) + β·C  (with 2:4 prune front-end)
//   dense×dense:  D = α·op(A)·op(B) + β·C          (no prune)
//
// dtype dispatch:
//   FP32/FP16: reuse LtMatmulGoldenRun<T> from ltmatmul_golden_types.h
//   BF16:      new bf16_bits_t trait + LtMatmulGoldenRun<bf16_bits_t>
//   INT8:      LtMatmulGoldenRunInt8 (INT32 accumulation) [in matmul_golden_int8.h]
//
// This header includes:
//   - ltmatmul_golden_types.h: shared types, conversions, prune helpers, CPU golden steps
//   - matmul_golden_int8.h:  INT8-specific golden functions
// And provides FP32/FP16/BF16 golden entry points (with/without vector scaling).
// =============================================================================

#include "ltmatmul_golden_types.h"  // bf16_bits_t, conversions, prune helpers, CPU golden steps
#include "ltmatmul_golden_int8.h"   // INT8 golden entries

namespace sparse_test {

// -----------------------------------------------------------------------------
// Golden entry: sparse×dense for FP32/FP16/BF16 (FP32 accumulation).
// Reuses LtMatmulGoldenRun<T> from ltmatmul_golden_types.h, but with
// MatmulDtypeTrait for prune group parameters.
// For BF16, T=bf16_bits_t uses the BF16 trait.
//
// pruneAlg: "STRIP" (default) or "TILE".
// -----------------------------------------------------------------------------
template <typename T>
inline void LtMatmulGoldenRunWithPruneAlg(
    const std::vector<T>& A, const std::vector<T>& B, const std::vector<T>& C,
    std::vector<T>& D,
    int32_t m, int32_t k, int32_t n,
    float alpha, float beta,
    int32_t alg_config_id, int32_t split_k,
    bool isColOrder, bool pruneB,
    const std::string& pruneAlg = "STRIP")
{
    (void)alg_config_id;
    (void)split_k;

    // Step 1: promote inputs to FP32.
    std::vector<float> Af, Bf, Cf;
    PromoteInputsToFp32<T>(A, B, C, Af, Bf, Cf, m, k, n);

    // Step 2: structured prune on the sparse matrix (FP32), with pruneAlg.
    std::vector<float> Df;
    if (pruneB) {
        std::vector<float> B_pruned;
        MatmulPruneInputAWithAlg<T>(Bf, B_pruned, k, n, isColOrder, pruneAlg);
        MatmulAlphaBetaFp32(Af, B_pruned, Cf, Df, m, k, n, alpha, beta);
    } else {
        std::vector<float> A_pruned;
        MatmulPruneInputAWithAlg<T>(Af, A_pruned, m, k, isColOrder, pruneAlg);
        MatmulAlphaBetaFp32(A_pruned, Bf, Cf, Df, m, k, n, alpha, beta);
    }

    // Step 3: truncate back to T (BF16 uses RNE, FP16 uses RNE, FP32 identity).
    const size_t mn = static_cast<size_t>(m) * static_cast<size_t>(n);
    TruncateToOutput<T>(Df, D, mn);
}

// -----------------------------------------------------------------------------
// Golden entry: dense×dense for FP32/FP16/BF16 (skip prune).
// D = α·A·B + β·C, FP32 accumulation, no prune front-end.
// -----------------------------------------------------------------------------
template <typename T>
inline void LtMatmulDenseGoldenRun(
    const std::vector<T>& A, const std::vector<T>& B, const std::vector<T>& C,
    std::vector<T>& D,
    int32_t m, int32_t k, int32_t n,
    float alpha, float beta,
    int32_t alg_config_id, int32_t split_k)
{
    (void)alg_config_id;
    (void)split_k;

    // Step 1: promote inputs to FP32 (no prune).
    std::vector<float> Af, Bf, Cf;
    PromoteInputsToFp32<T>(A, B, C, Af, Bf, Cf, m, k, n);

    // Step 2: dense matmul + alpha/beta.
    std::vector<float> Df;
    MatmulAlphaBetaFp32(Af, Bf, Cf, Df, m, k, n, alpha, beta);

    // Step 3: truncate back to T.
    const size_t mn = static_cast<size_t>(m) * static_cast<size_t>(n);
    TruncateToOutput<T>(Df, D, mn);
}

// -----------------------------------------------------------------------------
// Vector scaling golden entries: alpha/beta are per-row float[M] vectors.
// D[i,j] = alphaVec[i] * acc[i,j] + betaVec[i] * C[i,j]
// -----------------------------------------------------------------------------

// sparse×dense FP32/FP16/BF16 with vector scaling
template <typename T>
inline void LtMatmulGoldenRunWithPruneAlgVec(
    const std::vector<T>& A, const std::vector<T>& B, const std::vector<T>& C,
    std::vector<T>& D,
    int32_t m, int32_t k, int32_t n,
    const std::vector<float>& alphaVec, const std::vector<float>& betaVec,
    bool isColOrder, bool pruneB,
    const std::string& pruneAlg = "STRIP")
{
    std::vector<float> Af, Bf, Cf;
    PromoteInputsToFp32<T>(A, B, C, Af, Bf, Cf, m, k, n);
    std::vector<float> Df;
    if (pruneB) {
        std::vector<float> B_pruned;
        MatmulPruneInputAWithAlg<T>(Bf, B_pruned, k, n, isColOrder, pruneAlg);
        MatmulAlphaBetaFp32Vec(Af, B_pruned, Cf, Df, m, k, n, alphaVec, betaVec);
    } else {
        std::vector<float> A_pruned;
        MatmulPruneInputAWithAlg<T>(Af, A_pruned, m, k, isColOrder, pruneAlg);
        MatmulAlphaBetaFp32Vec(A_pruned, Bf, Cf, Df, m, k, n, alphaVec, betaVec);
    }
    const size_t mn = static_cast<size_t>(m) * static_cast<size_t>(n);
    TruncateToOutput<T>(Df, D, mn);
}

// dense×dense FP32/FP16/BF16 with vector scaling
template <typename T>
inline void LtMatmulDenseGoldenRunVec(
    const std::vector<T>& A, const std::vector<T>& B, const std::vector<T>& C,
    std::vector<T>& D,
    int32_t m, int32_t k, int32_t n,
    const std::vector<float>& alphaVec, const std::vector<float>& betaVec)
{
    std::vector<float> Af, Bf, Cf;
    PromoteInputsToFp32<T>(A, B, C, Af, Bf, Cf, m, k, n);
    std::vector<float> Df;
    MatmulAlphaBetaFp32Vec(Af, Bf, Cf, Df, m, k, n, alphaVec, betaVec);
    const size_t mn = static_cast<size_t>(m) * static_cast<size_t>(n);
    TruncateToOutput<T>(Df, D, mn);
}

}  // namespace sparse_test

#endif  // TEST_MATMUL_GOLDEN_H_
