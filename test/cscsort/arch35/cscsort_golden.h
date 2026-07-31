/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root directory of the software repository for the full text of the License.
 */

#ifndef TEST_CSCSORT_ARCH35_CSCSORT_GOLDEN_H_
#define TEST_CSCSORT_ARCH35_CSCSORT_GOLDEN_H_

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <vector>

namespace sparse_test {

// ============================================================================
// Host-side semantic anchor for aclsparseXcscsort (kernel-independent).
//
// Column-wise dual of aclsparseXcsrsort. Sorting contract:
//   - Per-column stable sort of cscRowInd in ascending order.
//   - P is reordered using the same permutation.
//   - cscColPtr is not modified.
//   - indexBase: 0 or 1; begin = cscColPtr[col] - indexBase.
// ============================================================================

struct CscsortGoldenResult {
    std::vector<int32_t> rowInd;
    std::vector<int32_t> P;
};

inline CscsortGoldenResult cscsortGolden(const std::vector<int32_t> &colPtr,
                                         const std::vector<int32_t> &rowInd,
                                         const std::vector<int32_t> &P, int n, int indexBase)
{
    CscsortGoldenResult r;
    r.rowInd = rowInd;
    r.P = P;

    for (int col = 0; col < n; col++) {
        int begin = colPtr[col] - indexBase;
        int end = colPtr[col + 1] - indexBase;
        int len = end - begin;
        if (len <= 1) {
            continue;
        }
        std::vector<int32_t> perm(len);
        std::iota(perm.begin(), perm.end(), 0);
        std::stable_sort(perm.begin(), perm.end(), [&](int a, int b) {
            return r.rowInd[begin + a] < r.rowInd[begin + b];
        });
        std::vector<int32_t> sortedRow(len);
        std::vector<int32_t> sortedP(len);
        for (int i = 0; i < len; i++) {
            sortedRow[i] = r.rowInd[begin + perm[i]];
            sortedP[i] = r.P[begin + perm[i]];
        }
        for (int i = 0; i < len; i++) {
            r.rowInd[begin + i] = sortedRow[i];
            r.P[begin + i] = sortedP[i];
        }
    }
    return r;
}

}  // namespace sparse_test

#endif  // TEST_CSCSORT_ARCH35_CSCSORT_GOLDEN_H_
