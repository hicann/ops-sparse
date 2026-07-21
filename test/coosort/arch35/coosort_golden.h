/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TEST_COOSORT_ARCH35_COOSORT_GOLDEN_H_
#define TEST_COOSORT_ARCH35_COOSORT_GOLDEN_H_

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <vector>

namespace sparse_test {

// ============================================================================
// Host-side semantic anchor for aclsparseXcoosort (kernel-independent).
//
// Sorting contract:
//   - Dual-key lexicographic stable sort.
//       ByRow    : primary = row, secondary = col
//       ByColumn : primary = col, secondary = row
//   - Stable: elements with identical (primary, secondary) keep input order.
//   - Signed integers compared directly (negative row/col supported).
//   - P semantics: caller presets P = 0:1:(nnz-1); after sort
//         sortedRows[i] = origRows[P[i]]
//         sortedCols[i] = origCols[P[i]]
//         sortedVal[i]  = origVal[P[i]]    (values are not part of this op,
//                                          P lets the caller gather them)
//   - cooRowsA / cooColsA are reordered in-place, keeping (row,col) pairs.
//
// This golden computes the expected permutation P and the expected reordered
// rows/cols from the ORIGINAL input. It does NOT mutate its inputs.
// ============================================================================

struct CoosortGoldenResult {
    std::vector<int32_t> rows;  // expected reordered rows
    std::vector<int32_t> cols;  // expected reordered cols
    std::vector<int32_t> P;     // expected permutation (orig index per sorted slot)
};

// Compute expected sort result given original rows/cols and sort direction.
// rows/cols are the ORIGINAL (pre-sort) input; they are read-only here.
inline CoosortGoldenResult coosortGolden(
    const std::vector<int32_t> &rows, const std::vector<int32_t> &cols, bool sortByRow)
{
    CoosortGoldenResult r;
    const int nnz = static_cast<int>(rows.size());
    r.rows.resize(nnz);
    r.cols.resize(nnz);
    r.P.resize(nnz);

    std::vector<int32_t> perm(nnz);
    std::iota(perm.begin(), perm.end(), 0);

    // Dual-key stable sort on the index array.
    //   ByRow   : primary=row, secondary=col
    //   ByColumn: primary=col, secondary=row
    std::stable_sort(perm.begin(), perm.end(), [&](int a, int b) {
        int32_t pa = sortByRow ? rows[a] : cols[a];
        int32_t pb = sortByRow ? rows[b] : cols[b];
        if (pa != pb)
            return pa < pb;
        int32_t sa = sortByRow ? cols[a] : rows[a];
        int32_t sb = sortByRow ? cols[b] : rows[b];
        return sa < sb;
    });

    for (int i = 0; i < nnz; i++) {
        r.P[i] = perm[i];
        r.rows[i] = rows[perm[i]];
        r.cols[i] = cols[perm[i]];
    }
    return r;
}

// Expected workspace size lower bound sanity check.
// The single-core path reserves nnz*12 bytes; the multi-core path returns a
// larger two-half ping-pong workspace. Therefore nnz*12 aligned to 128 remains
// a valid minimum reasonableness bound, not an exact expectation.
constexpr size_t kCoosortAlign = 128;

inline size_t coosortExpectedBufferSizeMin(int nnz)
{
    if (nnz <= 0)
        return 0;
    size_t raw = static_cast<size_t>(nnz) * 12u;
    return ((raw + kCoosortAlign - 1) / kCoosortAlign) * kCoosortAlign;
}

}  // namespace sparse_test

#endif  // TEST_COOSORT_ARCH35_COOSORT_GOLDEN_H_
