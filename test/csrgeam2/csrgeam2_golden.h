/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TEST_CSRGEAM2_CSRGEAM2_GOLDEN_H_
#define TEST_CSRGEAM2_CSRGEAM2_GOLDEN_H_

#include <cstdint>
#include <vector>

#include "fill.h"

namespace sparse_test {

// ============================================================================
// Golden result structure for csrgeam2: C = alpha * A + beta * B
// ============================================================================

struct CsrGeam2GoldenResult {
    std::vector<int32_t> rowPtrC;   // size m+1, in indexBaseC
    std::vector<int32_t> colIndC;   // size nnzC, in indexBaseC
    std::vector<float>   valuesC;   // size nnzC
    int32_t              nnzC;      // total nonzero count
};

// ============================================================================
// Helper: Count nnzC per row using two-pointer merge of sorted column indices.
//
// NOTE: This two-pointer merge algorithm intentionally duplicates the sparsity
// pattern determination logic used in the kernel. This is by design: the golden
// reference must be implemented independently from the kernel to provide a
// trustworthy correctness baseline.
// ============================================================================

inline std::vector<int32_t> CountNnzPerRow(
    const CsrMatrix &csrA, const CsrMatrix &csrB,
    int m, int indexBaseA, int indexBaseB)
{
    std::vector<int32_t> nnzPerRow(m, 0);

    for (int i = 0; i < m; i++) {
        int32_t pA = csrA.rowOffsets[i] - indexBaseA;
        int32_t qA = csrA.rowOffsets[i + 1] - indexBaseA;
        int32_t pB = csrB.rowOffsets[i] - indexBaseB;
        int32_t qB = csrB.rowOffsets[i + 1] - indexBaseB;

        int32_t count = 0;
        while (pA < qA && pB < qB) {
            int32_t colA0 = csrA.colIndices[pA] - indexBaseA;
            int32_t colB0 = csrB.colIndices[pB] - indexBaseB;
            if (colA0 < colB0) {
                pA++;
            } else if (colA0 > colB0) {
                pB++;
            } else {
                pA++;
                pB++;
            }
            count++;
        }
        count += (qA - pA);
        count += (qB - pB);
        nnzPerRow[i] = count;
    }

    return nnzPerRow;
}

// ============================================================================
// Helper: Fill colIndC and valuesC using two-pointer merge (FP64 arithmetic).
//
// NOTE: Per cuSPARSE/csrgeam2 semantics, C's nonzero pattern is ALWAYS A∪B,
// even when alpha=0 or beta=0. Values in the union that are not present in
// A (or B) are treated as 0 in the computation.
//
// NOTE: This two-pointer merge algorithm intentionally duplicates the merge
// logic used in the kernel. This is by design: the golden reference must be
// implemented independently from the kernel to provide a trustworthy
// correctness baseline.
// ============================================================================

inline void FillMergeResult(
    const CsrMatrix &csrA, const CsrMatrix &csrB,
    float alpha, float beta,
    int indexBaseA, int indexBaseB, int indexBaseC,
    int m,
    const std::vector<int32_t> &rowPtrC,
    std::vector<int32_t> &colIndC,
    std::vector<float> &valuesC)
{
    for (int i = 0; i < m; i++) {
        int32_t pA = csrA.rowOffsets[i] - indexBaseA;
        int32_t qA = csrA.rowOffsets[i + 1] - indexBaseA;
        int32_t pB = csrB.rowOffsets[i] - indexBaseB;
        int32_t qB = csrB.rowOffsets[i + 1] - indexBaseB;
        int32_t pC = rowPtrC[i] - indexBaseC;

        while (pA < qA && pB < qB) {
            int32_t colA0 = csrA.colIndices[pA] - indexBaseA;
            int32_t colB0 = csrB.colIndices[pB] - indexBaseB;

            if (colA0 < colB0) {
                colIndC[pC] = colA0 + indexBaseC;
                valuesC[pC] = static_cast<float>(
                    static_cast<double>(alpha) * static_cast<double>(csrA.values[pA]));
                pA++;
                pC++;
            } else if (colA0 > colB0) {
                colIndC[pC] = colB0 + indexBaseC;
                valuesC[pC] = static_cast<float>(
                    static_cast<double>(beta) * static_cast<double>(csrB.values[pB]));
                pB++;
                pC++;
            } else {
                colIndC[pC] = colA0 + indexBaseC;
                valuesC[pC] = static_cast<float>(
                    static_cast<double>(alpha) * static_cast<double>(csrA.values[pA]) +
                    static_cast<double>(beta)  * static_cast<double>(csrB.values[pB]));
                pA++;
                pB++;
                pC++;
            }
        }

        // Remaining from A only
        while (pA < qA) {
            int32_t colA0 = csrA.colIndices[pA] - indexBaseA;
            colIndC[pC] = colA0 + indexBaseC;
            valuesC[pC] = static_cast<float>(
                static_cast<double>(alpha) * static_cast<double>(csrA.values[pA]));
            pA++;
            pC++;
        }

        // Remaining from B only
        while (pB < qB) {
            int32_t colB0 = csrB.colIndices[pB] - indexBaseB;
            colIndC[pC] = colB0 + indexBaseC;
            valuesC[pC] = static_cast<float>(
                static_cast<double>(beta) * static_cast<double>(csrB.values[pB]));
            pB++;
            pC++;
        }
    }
}

// ============================================================================
// Eigen-free golden reference implementation
//
// Uses double (FP64) arithmetic for the alpha*valA + beta*valB computation
// to avoid CPU-side precision loss, then casts back to float for output.
//
// This implementation does NOT require Eigen. It performs the row-by-row
// sorted merge manually, which is the exact same algorithm as the kernel
// (two-pointer merge of sorted column indices).
// ============================================================================

inline CsrGeam2GoldenResult CsrGeam2Golden(
    const CsrMatrix &csrA,
    const CsrMatrix &csrB,
    float alpha, float beta,
    int indexBaseA, int indexBaseB, int indexBaseC)
{
    int m = static_cast<int>(csrA.rows);

    CsrGeam2GoldenResult result;
    result.nnzC = 0;

    if (m <= 0 || static_cast<int>(csrA.cols) <= 0) {
        result.rowPtrC.assign(1, indexBaseC);
        return result;
    }

    result.rowPtrC.resize(m + 1);
    result.rowPtrC[0] = indexBaseC;

    // Pass 1: count nnzC per row (union of A and B column indices)
    auto nnzPerRow = CountNnzPerRow(csrA, csrB, m, indexBaseA, indexBaseB);

    // Compute rowPtrC (exclusive prefix sum with baseC offset)
    for (int i = 0; i < m; i++) {
        result.rowPtrC[i + 1] = result.rowPtrC[i] + nnzPerRow[i];
    }
    result.nnzC = result.rowPtrC[m] - indexBaseC;

    // Pass 2: compute colIndC and valuesC
    result.colIndC.resize(result.nnzC, 0);
    result.valuesC.resize(result.nnzC, 0.0f);

    FillMergeResult(csrA, csrB, alpha, beta,
                    indexBaseA, indexBaseB, indexBaseC, m,
                    result.rowPtrC, result.colIndC, result.valuesC);

    return result;
}

}  // namespace sparse_test

#endif  // TEST_CSRGEAM2_CSRGEAM2_GOLDEN_H_
