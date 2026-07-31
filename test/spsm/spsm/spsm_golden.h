/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TEST_SPSM_SPSM_GOLDEN_H_
#define TEST_SPSM_SPSM_GOLDEN_H_

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <random>
#include <vector>

#include "fill.h"

#ifndef SPARSE_TEST_USE_EIGEN
#error "spsm_golden.h requires Eigen (enable EIGEN in ops_sparse_add_gtest_tests)."
#endif
#include <Eigen/Dense>

namespace sparse_test {

// ============================================================================
// Triangular CSR generators
//
// UNIT diag: CSR stores only off-diagonal uplo entries (diagonal implicit=1.0)
// NON_UNIT diag: CSR stores off-diagonal uplo entries + explicit diagonal
//                 (diagonal value in [diagLo, diagHi], avoids 0)
// ============================================================================

// UNIT diagonal: off-diagonal uplo part only (existing, kept for compatibility).
inline CsrMatrix makeTriangularCsr(int m, bool isLower, double density,
                                    double valueLo, double valueHi, uint32_t seed) {
    CsrMatrix out;
    out.rows = m;
    out.cols = m;
    out.rowOffsets.assign(static_cast<size_t>(m) + 1, 0);
    if (m <= 0) return out;

    density = std::clamp(density, 0.0, 1.0);
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> valDist(static_cast<float>(valueLo),
                                                   static_cast<float>(valueHi));
    std::uniform_real_distribution<float> zeroDist(0.0f, 1.0f);

    int64_t currentNnz = 0;
    std::vector<int32_t> rowCols;
    for (int i = 0; i < m; ++i) {
        out.rowOffsets[i] = static_cast<int32_t>(currentNnz);
        int colLo = isLower ? 0 : (i + 1);
        int colHi = isLower ? (i - 1) : (m - 1);
        int span = colHi - colLo + 1;
        if (span > 0 && density > 0.0) {
            for (int c = colLo; c <= colHi; ++c) {
                if (zeroDist(rng) < static_cast<float>(density)) {
                    rowCols.push_back(c);
                }
            }
            for (int32_t c : rowCols) {
                out.colIndices.push_back(c);
                float v = valDist(rng);
                if (v == 0.0f) v = 1.0f;
                out.values.push_back(v);
                ++currentNnz;
            }
            rowCols.clear();
        }
    }
    out.rowOffsets[m] = static_cast<int32_t>(currentNnz);
    out.nnz = currentNnz;
    return out;
}

inline void SortAndAppendEntries(std::vector<std::pair<int32_t, float>>& rowEntries,
                                 CsrMatrix& out, int64_t& currentNnz) {
    std::sort(rowEntries.begin(), rowEntries.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });
    for (const auto& [c, v] : rowEntries) {
        out.colIndices.push_back(c);
        out.values.push_back(v);
        ++currentNnz;
    }
}

inline void GenerateOffDiagEntries(std::mt19937& rng,
                                   std::uniform_real_distribution<float>& valDist,
                                   std::uniform_real_distribution<float>& zeroDist,
                                   double density, int colLo, int colHi,
                                   std::vector<std::pair<int32_t, float>>& rowEntries) {
    if (density <= 0.0 || colLo > colHi) return;
    for (int c = colLo; c <= colHi; ++c) {
        if (zeroDist(rng) < static_cast<float>(density)) {
            float v = valDist(rng);
            if (v == 0.0f) v = 1.0f;
            rowEntries.emplace_back(c, v);
        }
    }
}

struct SpsmRngState {
    std::mt19937 rng;
    std::uniform_real_distribution<float> valDist;
    std::uniform_real_distribution<float> diagDist;
    std::uniform_real_distribution<float> zeroDist;
};

inline SpsmRngState MakeSpsmRngState(uint32_t seed, double valueLo, double valueHi,
                                     double diagLo, double diagHi) {
    return SpsmRngState{
        std::mt19937(seed),
        std::uniform_real_distribution<float>(static_cast<float>(valueLo), static_cast<float>(valueHi)),
        std::uniform_real_distribution<float>(static_cast<float>(diagLo), static_cast<float>(diagHi)),
        std::uniform_real_distribution<float>(0.0f, 1.0f)
    };
}

inline void AppendDiagEntry(SpsmRngState& rs, int32_t i,
                            std::vector<std::pair<int32_t, float>>& rowEntries) {
    float dv = rs.diagDist(rs.rng);
    if (dv == 0.0f) dv = 1.0f;
    rowEntries.emplace_back(i, dv);
}

// NON_UNIT diagonal: off-diagonal uplo part + explicit diagonal entries.
// Diagonal values in [diagLo, diagHi] (avoids 0 for non-singular cases).
inline CsrMatrix makeTriangularCsrNonUnit(int m, bool isLower, double density,
                                          double valueLo, double valueHi,
                                          double diagLo, double diagHi, uint32_t seed) {
    CsrMatrix out;
    out.rows = m;
    out.cols = m;
    out.rowOffsets.assign(static_cast<size_t>(m) + 1, 0);
    if (m <= 0) return out;

    density = std::clamp(density, 0.0, 1.0);
    auto rs = MakeSpsmRngState(seed, valueLo, valueHi, diagLo, diagHi);

    int64_t currentNnz = 0;
    std::vector<std::pair<int32_t, float>> rowEntries;
    for (int i = 0; i < m; ++i) {
        out.rowOffsets[i] = static_cast<int32_t>(currentNnz);
        rowEntries.clear();
        int colLo = isLower ? 0 : (i + 1);
        int colHi = isLower ? (i - 1) : (m - 1);
        GenerateOffDiagEntries(rs.rng, rs.valDist, rs.zeroDist, density, colLo, colHi, rowEntries);
        // Explicit diagonal (kept sorted: diagonal sorts after lower-part, before upper-part)
        AppendDiagEntry(rs, i, rowEntries);
        SortAndAppendEntries(rowEntries, out, currentNnz);
    }
    out.rowOffsets[m] = static_cast<int32_t>(currentNnz);
    out.nnz = currentNnz;
    return out;
}

// Banded triangular CSR: off-diagonal uplo entries only within |i-j| <= bw.
// isUnitDiag=true -> diagonal implicit; false -> explicit diagonal [diagLo, diagHi].
inline CsrMatrix makeBandedTriangularCsr(int m, bool isLower, int bw,
                                         double valueLo, double valueHi,
                                         bool isUnitDiag,
                                         double diagLo, double diagHi, uint32_t seed) {
    CsrMatrix out;
    out.rows = m;
    out.cols = m;
    out.rowOffsets.assign(static_cast<size_t>(m) + 1, 0);
    if (m <= 0) return out;

    auto rs = MakeSpsmRngState(seed, valueLo, valueHi, diagLo, diagHi);

    int64_t currentNnz = 0;
    std::vector<std::pair<int32_t, float>> rowEntries;
    for (int i = 0; i < m; ++i) {
        out.rowOffsets[i] = static_cast<int32_t>(currentNnz);
        rowEntries.clear();
        int colLo, colHi;
        if (isLower) {
            colLo = std::max(0, i - bw);
            colHi = i - 1;
        } else {
            colLo = i + 1;
            colHi = std::min(m - 1, i + bw);
        }
        GenerateOffDiagEntries(rs.rng, rs.valDist, rs.zeroDist, 0.5, colLo, colHi, rowEntries);
        if (!isUnitDiag) {
            AppendDiagEntry(rs, i, rowEntries);
        }
        SortAndAppendEntries(rowEntries, out, currentNnz);
    }
    out.rowOffsets[m] = static_cast<int32_t>(currentNnz);
    out.nnz = currentNnz;
    return out;
}

// Block-diagonal triangular CSR: m must be divisible by blk.
// Within each blk×blk block, only uplo triangle is nonzero; outside blocks all zero.
inline CsrMatrix makeBlockDiagTriangularCsr(int m, bool isLower, int blk,
                                            double density, double valueLo, double valueHi,
                                            bool isUnitDiag,
                                            double diagLo, double diagHi, uint32_t seed) {
    CsrMatrix out;
    out.rows = m;
    out.cols = m;
    out.rowOffsets.assign(static_cast<size_t>(m) + 1, 0);
    if (m <= 0 || blk == 0) return out;

    density = std::clamp(density, 0.0, 1.0);
    auto rs = MakeSpsmRngState(seed, valueLo, valueHi, diagLo, diagHi);

    int64_t currentNnz = 0;
    std::vector<std::pair<int32_t, float>> rowEntries;
    int numBlocks = (m + blk - 1) / blk;
    for (int b = 0; b < numBlocks; ++b) {
        int blkStart = b * blk;
        int blkEnd = std::min(m, blkStart + blk);
        for (int i = blkStart; i < blkEnd; ++i) {
            out.rowOffsets[i] = static_cast<int32_t>(currentNnz);
            rowEntries.clear();
            int colLo, colHi;
            if (isLower) {
                colLo = blkStart;
                colHi = i - 1;
            } else {
                colLo = i + 1;
                colHi = blkEnd - 1;
            }
            GenerateOffDiagEntries(rs.rng, rs.valDist, rs.zeroDist, density, colLo, colHi, rowEntries);
            if (!isUnitDiag) {
                AppendDiagEntry(rs, i, rowEntries);
            }
            SortAndAppendEntries(rowEntries, out, currentNnz);
        }
    }
    out.rowOffsets[m] = static_cast<int32_t>(currentNnz);
    out.nnz = currentNnz;
    return out;
}

// Singular NON_UNIT triangular CSR: diagonal stored but at least one entry is 0.
// Generates a normal NON_UNIT triangular CSR, then zeroes out one diagonal element.
inline CsrMatrix makeSingularTriangularCsr(int m, bool isLower, double density,
                                           double valueLo, double valueHi,
                                           double diagLo, double diagHi, uint32_t seed) {
    CsrMatrix out = makeTriangularCsrNonUnit(m, isLower, density, valueLo, valueHi,
                                              diagLo, diagHi, seed);
    // Zero out the middle diagonal element to create singularity.
    if (m <= 0) return out;
    int targetRow = m / 2;
    int rowStart = out.rowOffsets[targetRow];
    int rowEnd = out.rowOffsets[targetRow + 1];
    bool zeroed = false;
    for (int p = rowStart; p < rowEnd; ++p) {
        if (out.colIndices[p] == targetRow) {
            out.values[p] = 0.0f;
            zeroed = true;
            break;
        }
    }
    // If diagonal not found (shouldn't happen for NON_UNIT), add a zero diagonal entry
    if (!zeroed && rowEnd > rowStart) {
        out.values[rowStart] = 0.0f;
    }
    return out;
}

// Named RNG seed salt for shuffleRowInternal (decouples the per-row shuffle
// RNG stream from the base matrix seed). constexpr at namespace scope has
// internal linkage, so no ODR risk across translation units.
constexpr uint32_t kShuffleSeedSalt = 0xDEADBEEFu;

// Shuffle colInd within each row (unsorted indices). Values are shuffled in
// tandem to preserve (colInd, value) correspondence. rowOffsets unchanged.
inline void shuffleRowInternal(CsrMatrix& csr, uint32_t seed) {
    if (csr.nnz <= 0) return;
    std::mt19937 rng(seed ^ kShuffleSeedSalt);
    for (int i = 0; i < static_cast<int>(csr.rows); ++i) {
        int start = csr.rowOffsets[i];
        int end = csr.rowOffsets[i + 1];
        int len = end - start;
        if (len <= 1) continue;
        // Fisher-Yates shuffle on [start, end)
        for (int k = len - 1; k > 0; --k) {
            int j = std::uniform_int_distribution<int>(0, k)(rng);
            std::swap(csr.colIndices[start + k], csr.colIndices[start + j]);
            std::swap(csr.values[start + k], csr.values[start + j]);
        }
    }
}

// Apply 1-based index base: add 1 to all colInd (for indexBase=ONE).
// The golden/wrapper must subtract 1 before use; this only affects the raw CSR
// passed to the NPU descriptor. Golden receives 0-based internally.
inline void applyIndexBaseOne(CsrMatrix& csr) {
    for (auto& c : csr.colIndices) c += 1;
}

// ============================================================================
// Golden result: solution X (flat vector, layout depends on order).
// COL order: column-major (ldb x n); ROW order: row-major (ldb x n).
// The golden computes in FP64 internally; output is cast to FP32 for unified
// verification against the NPU result.
// ============================================================================
struct SpsmGoldenResult {
    std::vector<float> X;  // size ldb * n, layout per order
};

inline Eigen::MatrixXd SpsmGoldenBuildMatrixA(const std::vector<int32_t>& csrRowPtr,
                                              const std::vector<int32_t>& csrColInd,
                                              const std::vector<float>& csrVals,
                                              int m, bool isUnitDiag) {
    Eigen::MatrixXd A = Eigen::MatrixXd::Zero(m, m);
    for (int i = 0; i < m; ++i) {
        int32_t rowStart = csrRowPtr[i];
        int32_t rowEnd = csrRowPtr[i + 1];
        for (int32_t p = rowStart; p < rowEnd; ++p) {
            int32_t j = csrColInd[p];
            if (j < 0 || j >= m) continue;
            A(i, j) = static_cast<double>(csrVals[p]);
        }
        if (isUnitDiag) {
            A(i, i) = 1.0;
        }
    }
    return A;
}

inline Eigen::MatrixXd SpsmGoldenBuildRhs(const std::vector<float>& B,
                                          int m, int n, int ldb,
                                          bool isRowOrder,
                                          double alphaD) {
    Eigen::MatrixXd rhs = Eigen::MatrixXd::Zero(m, n);
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < m; ++i) {
            int64_t off = isRowOrder
                ? static_cast<int64_t>(i) * ldb + j
                : static_cast<int64_t>(j) * ldb + i;
            rhs(i, j) = alphaD * static_cast<double>(B[off]);
        }
    }
    return rhs;
}

inline Eigen::MatrixXd SpsmGoldenTriangularSolve(const Eigen::MatrixXd& A,
                                                 const Eigen::MatrixXd& rhs,
                                                 bool isLower, bool isTranspose,
                                                 bool isUnitDiag) {
    Eigen::MatrixXd X(A.rows(), rhs.cols());
    auto solve = [&](auto triView) { X = triView.solve(rhs); };
    if (!isTranspose) {
        if (isLower) {
            if (isUnitDiag) solve(A.triangularView<Eigen::UnitLower>());
            else            solve(A.triangularView<Eigen::Lower>());
        } else {
            if (isUnitDiag) solve(A.triangularView<Eigen::UnitUpper>());
            else            solve(A.triangularView<Eigen::Upper>());
        }
    } else {
        if (isLower) {
            if (isUnitDiag) solve(A.transpose().triangularView<Eigen::UnitUpper>());
            else            solve(A.transpose().triangularView<Eigen::Upper>());
        } else {
            if (isUnitDiag) solve(A.transpose().triangularView<Eigen::UnitLower>());
            else            solve(A.transpose().triangularView<Eigen::Lower>());
        }
    }
    return X;
}

inline void SpsmGoldenWriteOutput(const Eigen::MatrixXd& X,
                                  int m, int n, int ldb,
                                  bool isRowOrder, SpsmGoldenResult& result) {
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < m; ++i) {
            int64_t off = isRowOrder
                ? static_cast<int64_t>(i) * ldb + j
                : static_cast<int64_t>(j) * ldb + i;
            result.X[off] = static_cast<float>(X(i, j));
        }
    }
}

// ============================================================================
// SpSM CPU golden reference (Eigen FP64).
//
// Solves op(A) * X = alpha * B where:
//   - A is m x m triangular (CSR)
//   - isUnitDiag=true  -> diagonal implicit 1.0, CSR stores only off-diag
//   - isUnitDiag=false -> diagonal explicit in CSR (NON_UNIT, v2 new)
//   - isRowOrder=true  -> B/X are row-major (element (i,j) at i*ldb+j)
//   - isRowOrder=false -> B/X are column-major (element (i,j) at j*ldb+i)
//   - op(A) = A      when isTranspose == false
//   - op(A) = A^T    when isTranspose == true
//
// 8 (opA × fillMode × diagType) combinations via Eigen triangularView:
//   (N, LOWER, UNIT)      -> UnitLower
//   (N, UPPER, UNIT)      -> UnitUpper
//   (N, LOWER, NON_UNIT)  -> Lower       (v2 new)
//   (N, UPPER, NON_UNIT)  -> Upper       (v2 new)
//   (T, LOWER, UNIT)      -> A^T UnitUpper
//   (T, UPPER, UNIT)      -> A^T UnitLower
//   (T, LOWER, NON_UNIT)  -> A^T Upper   (v2 new)
//   (T, UPPER, NON_UNIT)  -> A^T Lower   (v2 new)
//
// Inputs (host side, FP32):
//   csrRowPtr : size m+1, 0-based
//   csrColInd : size nnz, 0-based (caller must normalize ONE->ZERO before call)
//   csrVals   : size nnz
//   B         : size ldb * n, layout per isRowOrder
// ============================================================================
inline SpsmGoldenResult SpsmGolden(
    const std::vector<int32_t>& csrRowPtr,
    const std::vector<int32_t>& csrColInd,
    const std::vector<float>& csrVals,
    int m, int n, int ldb,
    const std::vector<float>& B,
    float alpha,
    bool isLower,
    bool isTranspose,
    bool isUnitDiag,    // v2 new: false -> NON_UNIT
    bool isRowOrder) {  // v2 new: true -> ROW order
    SpsmGoldenResult result;
    // Buffer size: COL order -> ldb * n (n columns, stride ldb);
    //              ROW order -> ldb * m (m rows, stride ldb).
    const int64_t outSize = isRowOrder
        ? static_cast<int64_t>(ldb) * m
        : static_cast<int64_t>(ldb) * n;
    result.X.assign(static_cast<size_t>(outSize), 0.0f);
    if (m <= 0 || n <= 0) return result;

    double alphaD = static_cast<double>(alpha);

    Eigen::MatrixXd A = SpsmGoldenBuildMatrixA(csrRowPtr, csrColInd, csrVals, m, isUnitDiag);
    Eigen::MatrixXd rhs = SpsmGoldenBuildRhs(B, m, n, ldb, isRowOrder, alphaD);
    Eigen::MatrixXd X = SpsmGoldenTriangularSolve(A, rhs, isLower, isTranspose, isUnitDiag);
    SpsmGoldenWriteOutput(X, m, n, ldb, isRowOrder, result);
    return result;
}

}  // namespace sparse_test

#endif  // TEST_SPSM_SPSM_GOLDEN_H_
