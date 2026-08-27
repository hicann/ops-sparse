/**
 * ----------------------------------------------------------------------------------------------------------
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms of the
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR
 * PURPOSE.
 * ----------------------------------------------------------------------------------------------------------
 */

#ifndef TEST_SPMM_OP_SPMM_OP_GOLDEN_H_
#define TEST_SPMM_OP_SPMM_OP_GOLDEN_H_

#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

#include "verify.h"  // applyMixedTolerance / Verifier (MIXED_TOLERANCE mode)

namespace sparse_test {

// ============================================================================
// SpMMOp sparse CSR structure (FP64 values).
// Used by both golden and NPU wrapper (via MakeSpmmSparsity) so that the
// sparsity pattern and values are bit-for-bit identical on both sides.
// ============================================================================
struct SpmmCsr {
    std::vector<int32_t> rowOffsets;  // size = m + 1, 0-based (or 1-based if indexBase==1)
    std::vector<int32_t> colIndices;  // size = nnz, ascending within each row (unless unsorted)
    std::vector<double> values;       // size = nnz, A nonzero values (FP64)
    int64_t m = 0;  // A rows
    int64_t k = 0;  // A cols (reduction dim)
    int64_t nnz = 0;
    int32_t indexBase = 0;  // 0 = ZERO, 1 = ONE (rowOffsets/colIndices offset by +1)
};

// CSR pattern post-processing helpers
inline void ShuffleCsrRowPairs(SpmmCsr& csr, int64_t m, uint32_t seed)
{
    if (static_cast<int64_t>(csr.rowOffsets.size()) < m + 1) {
        return;
    }
    for (int64_t i = 0; i < m; i++) {
        int32_t s = csr.rowOffsets[static_cast<size_t>(i)];
        int32_t e = csr.rowOffsets[static_cast<size_t>(i + 1)];
        if (s < 0 || e > static_cast<int32_t>(csr.colIndices.size())) {
            continue;
        }
        if (e - s <= 1) {
            continue;
        }
        // Deterministic per-row seed for reproducibility
        std::mt19937 rowRng(seed + static_cast<uint32_t>(i));
        for (int32_t j = e - 1; j > s; --j) {
            int32_t kIdx = s + (rowRng() % static_cast<uint32_t>(j - s + 1));
            std::swap(csr.colIndices[static_cast<size_t>(j)],
                      csr.colIndices[static_cast<size_t>(kIdx)]);
            std::swap(csr.values[static_cast<size_t>(j)],
                      csr.values[static_cast<size_t>(kIdx)]);
        }
    }
}

inline void ApplyIndexBaseOffset(SpmmCsr& csr, int32_t indexBase)
{
    if (indexBase == 1) {
        for (auto& ro : csr.rowOffsets) ro += 1;
        for (auto& ci : csr.colIndices) ci += 1;
    }
    csr.indexBase = indexBase;
}

// ============================================================================
// Deterministic sparsity pattern generator (per test plan §2.2.2).
// Position (i, j) is nonzero  <=>  (i * 7 + j * 13) % 100 < int(ratio * 100)
// values generated with std::mt19937 + uniform_real_distribution<double>
// over [value_lo, value_hi], seeded by `seed` for reproducibility.
// If `unsorted` is true, each row's (colInd, values) pairs are shuffled with
// a deterministic per-row seed (seed + row_index) so the pattern is no longer
// ascending within a row — but the golden result is unchanged (dot product is
// order-independent).
// If `indexBase` is 1, rowOffsets and colIndices are offset by +1 to simulate
// 1-based indexing (cuSPARSE ACL_SPARSE_INDEX_BASE_ONE semantics).
// ============================================================================
inline SpmmCsr MakeSpmmSparsity(int64_t m, int64_t k, double sparsity_ratio,
    double value_lo, double value_hi, uint32_t seed,
    bool unsorted = false, int32_t indexBase = 0)
{
    SpmmCsr csr;
    csr.m = m;
    csr.k = k;
    csr.rowOffsets.assign(static_cast<size_t>(m) + 1, 0);
    if (m <= 0 || k <= 0) {
        csr.indexBase = indexBase;
        if (indexBase == 1) {
            // Even for empty matrix, rowOffsets[0] should be 1-based
            for (auto& ro : csr.rowOffsets) ro += 1;
        }
        return csr;
    }

    int32_t threshold = static_cast<int32_t>(sparsity_ratio * 100);
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> dist(value_lo, value_hi);

    int64_t nnz = 0;
    for (int64_t i = 0; i < m; i++) {
        csr.rowOffsets[static_cast<size_t>(i)] = static_cast<int32_t>(nnz);
        for (int64_t j = 0; j < k; j++) {
            int32_t key = static_cast<int32_t>(((i % 100) * 7 + (j % 100) * 13) % 100);
            if (key < threshold) {
                csr.colIndices.push_back(static_cast<int32_t>(j));
                csr.values.push_back(dist(rng));
                nnz++;
            }
        }
    }
    if (nnz > static_cast<int64_t>(INT32_MAX)) {
        return csr;
    }
    csr.rowOffsets[static_cast<size_t>(m)] = static_cast<int32_t>(nnz);
    csr.nnz = nnz;

    // Shuffle (colInd, values) pairs within each row if unsorted
    if (unsorted) {
        ShuffleCsrRowPairs(csr, m, seed);
    }

    // Apply 1-based indexing offset if indexBase == 1
    ApplyIndexBaseOffset(csr, indexBase);
    return csr;
}

// ============================================================================
// SpmmGoldenDotProduct: extracted from SpmmGolden's inner for(p) loop to
// reduce nesting depth (ISSUE-R4-002). Handles bounds checks, B access
// (transposed or non-transposed), and dot accumulation.
// ============================================================================
inline double SpmmGoldenDotProduct(
    int32_t s, int32_t e, int32_t ib, int64_t k, int64_t n, int64_t j,
    bool bTransposed, const SpmmCsr& csrA, const std::vector<double>& B)
{
    double dot = 0.0;
    for (int32_t p = s; p < e; p++) {
        if (p < 0 || p >= static_cast<int32_t>(csrA.colIndices.size())) {
            continue;
        }
        int32_t t = csrA.colIndices[static_cast<size_t>(p)] - ib;
        if (t < 0 || t >= k) {
            continue;
        }
        double bVal;
        if (!bTransposed) {
            bVal = B[static_cast<size_t>(t) * static_cast<size_t>(n) +
                     static_cast<size_t>(j)];
        } else {
            bVal = B[static_cast<size_t>(j) * static_cast<size_t>(k) +
                     static_cast<size_t>(t)];
        }
        dot += csrA.values[static_cast<size_t>(p)] * bVal;
    }
    return dot;
}

/**
 * SpMMOp golden reference (FP64 triple-loop, per test plan §2.2.3).
 * C[i,j] = beta * C_init[i,j] + alpha * sum_{p=rowOff[i]}^{rowOff[i+1]-1}
 *          A.values[p] * B_access(colInd[p], j)
 * B_access depends on opB:
 *   opB = NON_TRANSPOSE: B is k*n row-major, B[t, j] = B[t*n + j]
 *   opB = TRANSPOSE:     B is n*k row-major, B[j, t] = B[j*k + t]
 * Returns std::vector<double> of length m*n (canonical row-major).
 */
inline std::vector<double> SpmmGolden(
    int64_t m, int64_t n, int64_t k,
    const SpmmCsr& csrA,
    const std::vector<double>& B,       // canonical row-major, k*n (NON_T) or n*k (T)
    const std::vector<double>& C_init,  // canonical row-major, m*n
    double alpha, double beta,
    aclsparseOperation_t opB = ACL_SPARSE_OP_NON_TRANSPOSE)
{
    std::vector<double> out(static_cast<size_t>(m) * static_cast<size_t>(n), 0.0);
    if (m <= 0 || n <= 0) {
        return out;
    }
    const bool bTransposed = (opB == ACL_SPARSE_OP_TRANSPOSE);
    // When indexBase == 1, rowOffsets/colIndices are offset by +1, so subtract
    // indexBase to recover the 0-based indices used for array access.
    const int32_t ib = csrA.indexBase;
    for (int64_t i = 0; i < m; i++) {
        int32_t s = csrA.rowOffsets[static_cast<size_t>(i)] - ib;
        int32_t e = csrA.rowOffsets[static_cast<size_t>(i + 1)] - ib;
        for (int64_t j = 0; j < n; j++) {
            double dot = SpmmGoldenDotProduct(s, e, ib, k, n, j,
                bTransposed, csrA, B);
            out[static_cast<size_t>(i) * static_cast<size_t>(n) +
                static_cast<size_t>(j)] =
                alpha * dot + beta * C_init[static_cast<size_t>(i) * static_cast<size_t>(n) +
                                           static_cast<size_t>(j)];
        }
    }
    return out;
}

// ============================================================================
// FP16 bit conversion helpers (same algorithm as sddmm_golden.h).
// Host stores FP16 as uint16_t IEEE-754 bit patterns.
// ============================================================================

inline uint16_t Fp32ToFp16Bits(float v)
{
    uint32_t bits;
    __builtin_memcpy(&bits, &v, sizeof(float));
    uint32_t sign = (bits >> 16) & 0x8000u;
    uint32_t rawExp = (bits >> 23) & 0xFFu;
    int32_t exp = static_cast<int32_t>(rawExp) - 127;
    uint32_t mant = bits & 0x007FFFFFu;

    if (rawExp == 0xFFu) {
        uint32_t nanBit = (mant != 0u) ? 0x0200u : 0u;
        return static_cast<uint16_t>(sign | 0x7C00u | nanBit);
    }
    if (exp >= 16) {
        return static_cast<uint16_t>(sign | 0x7C00u);
    }
    if (exp >= -14) {
        uint32_t mant10 = mant >> 13;
        uint32_t remainder = mant & 0x1FFFu;
        uint32_t result = (static_cast<uint32_t>(exp + 15) << 10) | mant10;
        if (remainder > 0x1000u || (remainder == 0x1000u && (mant10 & 1u))) {
            result += 1u;
        }
        return static_cast<uint16_t>(sign | result);
    }
    if (exp >= -25) {
        uint32_t significand = 0x00800000u | mant;
        uint32_t shift = static_cast<uint32_t>(-exp - 1);
        uint32_t frac = significand >> shift;
        uint32_t remainder = significand & ((1u << shift) - 1u);
        uint32_t half = 1u << (shift - 1u);
        if (remainder > half || (remainder == half && (frac & 1u))) {
            frac += 1u;
        }
        return static_cast<uint16_t>(sign | frac);
    }
    return static_cast<uint16_t>(sign);
}

inline float Fp16BitsToFp32(uint16_t h)
{
    uint32_t sign = static_cast<uint32_t>(h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1Fu;
    uint32_t mant = h & 0x03FFu;
    uint32_t f;
    if (exp == 0) {
        if (mant == 0) {
            f = sign;
        } else {
            uint32_t shift = __builtin_clz(mant) - 21;
            mant <<= shift;
            exp = 1;
            f = sign | ((exp + 127 - 15 - shift) << 23) | ((mant << 13) & 0x7FFFFFu);
        }
    } else if (exp == 31) {
        f = sign | 0x7F800000u | (mant << 13);
    } else {
        f = sign | ((exp + 127 - 15) << 23) | (mant << 13);
    }
    float v;
    __builtin_memcpy(&v, &f, sizeof(float));
    return v;
}

// ============================================================================
// Host data conversion helpers: FP64 -> dtype-sized host vectors for NPU input.
// ============================================================================

inline std::vector<float> DoublesToFp32(const std::vector<double>& v)
{
    std::vector<float> out(v.size());
    for (size_t i = 0; i < v.size(); i++) {
        out[i] = static_cast<float>(v[i]);
    }
    return out;
}

inline std::vector<uint16_t> DoublesToFp16(const std::vector<double>& v)
{
    std::vector<uint16_t> out(v.size());
    for (size_t i = 0; i < v.size(); i++) {
        out[i] = Fp32ToFp16Bits(static_cast<float>(v[i]));
    }
    return out;
}

}  // namespace sparse_test

#endif  // TEST_SPMM_OP_SPMM_OP_GOLDEN_H_
