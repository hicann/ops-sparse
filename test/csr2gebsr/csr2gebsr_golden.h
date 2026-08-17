/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root directory of the software repository for the full text of the License.
 */

#ifndef TEST_CSR2GEBSR_CSR2GEBSR_GOLDEN_H_
#define TEST_CSR2GEBSR_CSR2GEBSR_GOLDEN_H_

#ifdef SPARSE_TEST_USE_EIGEN
#include <Eigen/Dense>
#include <Eigen/Sparse>
#endif

#include <algorithm>
#include <cstdint>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "dtype_utils.h"  // floatToHalf, halfToFloat, applyTypeRoundTrip
#include "fill.h"

namespace sparse_test {

/// 安全向上取整（防御除零）
static inline int SafeCeilDivInt(int value, int divisor) {
    return (divisor > 0) ? (value + divisor - 1) / divisor : 0;
}

/// 安全除法（防御除零）
static inline int SafeDivInt(int value, int divisor) {
    return (divisor > 0) ? value / divisor : 0;
}

// ============================================================================
// IEEE 754 half-precision conversion helpers
//
// floatToHalf / halfToFloat now live in dtype_utils.h.
// The wrappers below preserve backward-compatible names for callers that
// reference them as sparse_test::floatToHalf / sparse_test::halfToFloat.
// ============================================================================

// floatToHalf and halfToFloat are provided by dtype_utils.h.

// ============================================================================
// Helper: Apply type round-trip to CSR values for non-FP32 tests
//
// For FP16/BF16/INT32 tests, the NPU stores values in the target type,
// so the golden must also apply the same type conversion to ensure
// bit-exact comparison. The round-trip is: float -> T -> float.
//
// Delegates to dtype_utils.h::applyTypeRoundTrip.
// ============================================================================

inline void ApplyTypeRoundTrip(std::vector<float>& values, const std::string& dtype) {
    applyTypeRoundTrip(values, dtype);
}

// ============================================================================
// Golden result structure for csr2gebsr: CSR -> GEBSR format conversion
// ============================================================================

struct GebsrResult {
    std::vector<int32_t> bsrRowPtr;   // size: mb + 1
    std::vector<int32_t> bsrColInd;   // size: nnzb
    std::vector<float>   bsrVal;      // size: nnzb * rowBlockDim * colBlockDim
    int32_t              nnzb;        // total nonzero block count
};

// ============================================================================
// Internal: block entry used during golden construction
// ============================================================================

namespace detail {

struct BlockEntry {
    int blockCol;
    std::vector<float> blockValues;  // size: blockSize, zero-initialized
};

// ---------------------------------------------------------------------------
// Build per-block-row nonzero block lists from CSR input.
//
// For each block row, scans the corresponding CSR rows, groups nonzero
// elements into blocks of size rowBlockDim x colBlockDim, and records
// their column-block index and filled values. Also fills bsrRowPtr via
// exclusive prefix sum of per-row nonzero-block counts.
//
// Returns the per-block-row list of BlockEntry (sorted by blockCol).
// ---------------------------------------------------------------------------
// Build nonzero block list for a single block row from its CSR rows.
// Returns sorted vector of BlockEntry (sorted by blockCol).
// ---------------------------------------------------------------------------
inline std::vector<BlockEntry> BuildSingleBlockRow(
    const CsrMatrix& csr, int rowStart, int rowEnd, int n,
    int rowBlockDim, int colBlockDim,
    bool dirRow, int indexBaseA, int blockSize)
{
    std::map<int, std::vector<float>> blockMap;

    for (int i = rowStart; i < rowEnd; i++) {
        int localRow = i - rowStart;
        int32_t csrStart = csr.rowOffsets[i] - indexBaseA;
        int32_t csrEnd = csr.rowOffsets[i + 1] - indexBaseA;

        for (int32_t k = csrStart; k < csrEnd; k++) {
            int j = csr.colIndices[k] - indexBaseA;
            if (j < 0 || j >= n) continue;

            int blockCol = SafeDivInt(j, colBlockDim);
            int localCol = (colBlockDim > 0) ? (j % colBlockDim) : 0;

            auto it = blockMap.find(blockCol);
            if (it == blockMap.end()) {
                blockMap[blockCol] = std::vector<float>(blockSize, 0.0f);
                it = blockMap.find(blockCol);
            }

            int pos = dirRow ? (localRow * colBlockDim + localCol)
                             : (localCol * rowBlockDim + localRow);
            it->second[pos] = csr.values[k];
        }
    }

    std::vector<BlockEntry> entries;
    for (auto& [bc, vals] : blockMap) {
        BlockEntry entry;
        entry.blockCol = bc;
        entry.blockValues = std::move(vals);
        entries.push_back(std::move(entry));
    }
    return entries;
}

// ---------------------------------------------------------------------------
// Build per-block-row nonzero block lists from CSR input.
// ---------------------------------------------------------------------------
inline std::vector<std::vector<BlockEntry>> BuildBlockRows(
    const CsrMatrix& csr, int m, int n,
    int rowBlockDim, int colBlockDim,
    bool dirRow, int indexBaseA, int blockSize)
{
    int mb = (m > 0) ? SafeCeilDivInt(m, rowBlockDim) : 0;
    std::vector<std::vector<BlockEntry>> blockRows(mb);

    for (int bi = 0; bi < mb; bi++) {
        int rowStart = bi * rowBlockDim;
        int rowEnd = std::min(rowStart + rowBlockDim, m);
        blockRows[bi] = BuildSingleBlockRow(
            csr, rowStart, rowEnd, n, rowBlockDim, colBlockDim,
            dirRow, indexBaseA, blockSize);
    }

    return blockRows;
}

// ---------------------------------------------------------------------------
// Fill bsrRowPtrC via exclusive prefix sum of per-block-row counts.
// ---------------------------------------------------------------------------
inline void FillBsrRowPtr(
    const std::vector<std::vector<BlockEntry>>& blockRows,
    std::vector<int32_t>& bsrRowPtr, int indexBaseC)
{
    int mb = static_cast<int>(blockRows.size());
    bsrRowPtr.resize(mb + 1);
    bsrRowPtr[0] = indexBaseC;
    for (int bi = 0; bi < mb; bi++) {
        bsrRowPtr[bi + 1] = bsrRowPtr[bi] +
                            static_cast<int32_t>(blockRows[bi].size());
    }
}

// ---------------------------------------------------------------------------
// Fill bsrColIndC and bsrValC from the sorted per-block-row lists.
// ---------------------------------------------------------------------------
inline void FillOutputArrays(
    const std::vector<std::vector<BlockEntry>>& blockRows,
    int32_t nnzb, int blockSize, int indexBaseC,
    std::vector<int32_t>& bsrColInd, std::vector<float>& bsrVal)
{
    bsrColInd.resize(nnzb);
    bsrVal.resize(static_cast<int64_t>(nnzb) * blockSize, 0.0f);

    int idx = 0;
    for (const auto& rowBlocks : blockRows) {
        for (const auto& entry : rowBlocks) {
            bsrColInd[idx] = entry.blockCol + indexBaseC;
            int64_t valOffset = static_cast<int64_t>(idx) * blockSize;
            for (int k = 0; k < blockSize; k++) {
                bsrVal[valOffset + k] = entry.blockValues[k];
            }
            idx++;
        }
    }
}

}  // namespace detail

// ============================================================================
// CPU golden reference: CSR -> GEBSR format conversion
//
// This is a pure data-rearrangement operation (no floating-point computation).
// The algorithm:
//   1. Divide the m x n matrix into mb x nb blocks of size rowBlockDim x colBlockDim
//   2. For each block row, scan CSR rows to find nonzero blocks
//   3. Build bsrRowPtrC via exclusive prefix sum of nonzero-block counts
//   4. Fill bsrColIndC with column-block indices of nonzero blocks
//   5. Fill bsrValC by extracting values from CSR and arranging per dir
//
// indexBase handling:
//   - Input CSR indices are offset by indexBaseA (subtract to get 0-based)
//   - Output GEBSR indices are offset by indexBaseC (add to get target base)
//
// Zero-padding:
//   - If m is not divisible by rowBlockDim, pad with zeros at bottom
//   - If n is not divisible by colBlockDim, pad with zeros at right
// ============================================================================

inline GebsrResult Csr2GebsrGolden(
    const CsrMatrix& csr,
    int m, int n,
    int rowBlockDim, int colBlockDim,
    bool dirRow,          // true = ROW (row-major within block), false = COLUMN
    int indexBaseA,       // 0 or 1
    int indexBaseC)       // 0 or 1
{
    GebsrResult result;
    result.nnzb = 0;

    int mb = (m > 0) ? SafeCeilDivInt(m, rowBlockDim) : 0;
    int blockSize = rowBlockDim * colBlockDim;

    // Handle empty matrix
    if (m <= 0 || n <= 0 || csr.nnz <= 0) {
        result.bsrRowPtr.assign(mb + 1, indexBaseC);
        result.nnzb = 0;
        return result;
    }

    // Step 1: Build per-block-row nonzero block lists
    auto blockRows = detail::BuildBlockRows(
        csr, m, n, rowBlockDim, colBlockDim, dirRow, indexBaseA, blockSize);

    // Step 2: Fill bsrRowPtrC via prefix sum
    detail::FillBsrRowPtr(blockRows, result.bsrRowPtr, indexBaseC);
    result.nnzb = result.bsrRowPtr[mb] - indexBaseC;

    // Step 3: Fill bsrColIndC and bsrValC
    detail::FillOutputArrays(
        blockRows, result.nnzb, blockSize, indexBaseC,
        result.bsrColInd, result.bsrVal);

    return result;
}

// ============================================================================
// Eigen cross-validation: independently verify golden output structure + values
//
// Uses Eigen::SparseMatrix to build an independent block-sparse representation
// from the original CSR data, then compares bsrRowPtr/bsrColInd/bsrVal against
// the hand-written golden. Catches indexBase offset errors, dir layout errors,
// and block grouping bugs that a single-implementation golden cannot detect.
// ============================================================================
#ifdef SPARSE_TEST_USE_EIGEN

#include <stdexcept>

/// 从 CSR 构建 dense 矩阵（padded 到块对齐维度）
inline Eigen::MatrixXd EigenBuildDenseFromCsr(
    const CsrMatrix& csr, int m, int n, int indexBaseA,
    int mb, int nb, int rowBlockDim, int colBlockDim)
{
    Eigen::MatrixXd dense = Eigen::MatrixXd::Zero(mb * rowBlockDim, nb * colBlockDim);
    for (int i = 0; i < m; i++) {
        int32_t rowStart = csr.rowOffsets[i] - indexBaseA;
        int32_t rowEnd = csr.rowOffsets[i + 1] - indexBaseA;
        for (int32_t k = rowStart; k < rowEnd; k++) {
            int j = csr.colIndices[k] - indexBaseA;
            if (j >= 0 && j < n) {
                dense(i, j) = static_cast<double>(csr.values[k]);
            }
        }
    }
    return dense;
}

/// 从 dense 矩阵提取非零块结构 + 块值
inline void EigenExtractBlocks(
    const Eigen::MatrixXd& dense,
    int m, int n, int rowBlockDim, int colBlockDim, bool dirRow,
    const CsrMatrix& csr, int indexBaseA,
    int mb, int nb,
    std::map<std::pair<int,int>, std::vector<double>>& blockVals)
{
    int paddedM = mb * rowBlockDim;
    int paddedN = nb * colBlockDim;

    for (int i = 0; i < m; i++) {
        int32_t rowStart = csr.rowOffsets[i] - indexBaseA;
        int32_t rowEnd = csr.rowOffsets[i + 1] - indexBaseA;
        int bi = SafeDivInt(i, rowBlockDim);
        for (int32_t k = rowStart; k < rowEnd; k++) {
            int j = csr.colIndices[k] - indexBaseA;
            if (j >= 0 && j < n) {
                int bj = SafeDivInt(j, colBlockDim);
                if (blockVals.find({bi, bj}) == blockVals.end()) {
                    blockVals[{bi, bj}] = std::vector<double>(
                        static_cast<size_t>(rowBlockDim) * colBlockDim, 0.0);
                }
            }
        }
    }

    for (const auto& [coord, _] : blockVals) {
        for (int r = 0; r < rowBlockDim; r++) {
            for (int c = 0; c < colBlockDim; c++) {
                int gr = coord.first * rowBlockDim + r;
                int gc = coord.second * colBlockDim + c;
                double v = (gr < paddedM && gc < paddedN) ? dense(gr, gc) : 0.0;
                int pos = dirRow ? (r * colBlockDim + c) : (c * rowBlockDim + r);
                blockVals[coord][static_cast<size_t>(pos)] = v;
            }
        }
    }
}

/// 构建 Eigen dense 矩阵 + 块结构
inline void EigenBuildBlockStructure(
    const CsrMatrix& csr, int m, int n,
    int rowBlockDim, int colBlockDim,
    bool dirRow, int indexBaseA,
    int mb, int nb,
    Eigen::SparseMatrix<double, Eigen::RowMajor>& blockStruct,
    std::map<std::pair<int,int>, std::vector<double>>& blockVals)
{
    auto dense = EigenBuildDenseFromCsr(csr, m, n, indexBaseA, mb, nb, rowBlockDim, colBlockDim);
    EigenExtractBlocks(dense, m, n, rowBlockDim, colBlockDim, dirRow,
        csr, indexBaseA, mb, nb, blockVals);

    using Triplet = Eigen::Triplet<double>;
    std::vector<Triplet> blockTriplets;
    blockTriplets.reserve(blockVals.size());
    for (const auto& [coord, _] : blockVals) {
        blockTriplets.emplace_back(coord.first, coord.second, 1.0);
    }
    blockStruct.resize(mb, nb);
    blockStruct.setFromTriplets(blockTriplets.begin(), blockTriplets.end());
}

/// 校验 bsrRowPtr + bsrColInd 结构
inline void EigenVerifyStructure(
    const Eigen::SparseMatrix<double, Eigen::RowMajor>& blockStruct,
    int mb, int indexBaseC,
    const GebsrResult& golden)
{
    int eigenNnzb = static_cast<int>(blockStruct.nonZeros());
    if (eigenNnzb != golden.nnzb) {
        throw std::logic_error("Eigen cross-check: nnzb mismatch: eigen="
            + std::to_string(eigenNnzb) + " golden="
            + std::to_string(golden.nnzb));
    }
    for (int i = 0; i <= mb; i++) {
        int32_t expect = static_cast<int32_t>(blockStruct.outerIndexPtr()[i]) + indexBaseC;
        if (golden.bsrRowPtr[static_cast<size_t>(i)] != expect) {
            throw std::logic_error("Eigen cross-check: bsrRowPtr mismatch at row "
                + std::to_string(i));
        }
    }
    for (int p = 0; p < eigenNnzb; p++) {
        int expectCol = static_cast<int>(blockStruct.innerIndexPtr()[p]) + indexBaseC;
        if (golden.bsrColInd[static_cast<size_t>(p)] != expectCol) {
            throw std::logic_error("Eigen cross-check: bsrColInd mismatch at idx "
                + std::to_string(p));
        }
    }
}

/// 校验 bsrVal 值
inline void EigenVerifyValues(
    const Eigen::SparseMatrix<double, Eigen::RowMajor>& blockStruct,
    const std::map<std::pair<int,int>, std::vector<double>>& blockVals,
    int mb, int rowBlockDim, int colBlockDim,
    const GebsrResult& golden)
{
    int blockSize = rowBlockDim * colBlockDim;
    int idx = 0;
    for (int bi = 0; bi < mb; bi++) {
        int rowStart = blockStruct.outerIndexPtr()[bi];
        int rowEnd = blockStruct.outerIndexPtr()[bi + 1];
        for (int p = rowStart; p < rowEnd; p++) {
            int bj = static_cast<int>(blockStruct.innerIndexPtr()[p]);
            auto it = blockVals.find({bi, bj});
            if (it == blockVals.end()) {
                throw std::logic_error("Eigen cross-check: block not found");
            }
            for (int k = 0; k < blockSize; k++) {
                float goldenVal = golden.bsrVal[
                    static_cast<size_t>(idx) * blockSize + k];
                double eigenVal = it->second[static_cast<size_t>(k)];
                if (std::fabs(static_cast<double>(goldenVal) - eigenVal) > 1e-5) {
                    throw std::logic_error("Eigen cross-check: bsrVal mismatch at block "
                        + std::to_string(idx) + " offset " + std::to_string(k));
                }
            }
            idx++;
        }
    }
}

inline void Csr2GebsrEigenCrossCheck(
    const CsrMatrix& csr,
    int m, int n,
    int rowBlockDim, int colBlockDim,
    bool dirRow,
    int indexBaseA, int indexBaseC,
    const GebsrResult& golden)
{
    int mb = (m > 0) ? SafeCeilDivInt(m, rowBlockDim) : 0;
    int nb = (n > 0) ? SafeCeilDivInt(n, colBlockDim) : 0;

    Eigen::SparseMatrix<double, Eigen::RowMajor> blockStruct;
    std::map<std::pair<int,int>, std::vector<double>> blockVals;

    EigenBuildBlockStructure(csr, m, n, rowBlockDim, colBlockDim,
        dirRow, indexBaseA, mb, nb, blockStruct, blockVals);

    EigenVerifyStructure(blockStruct, mb, indexBaseC, golden);

    EigenVerifyValues(blockStruct, blockVals, mb, rowBlockDim, colBlockDim, golden);
}

#endif  // SPARSE_TEST_USE_EIGEN

}  // namespace sparse_test

#endif  // TEST_CSR2GEBSR_CSR2GEBSR_GOLDEN_H_
