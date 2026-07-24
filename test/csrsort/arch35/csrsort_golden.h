/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under
 * the terms and conditions of CANN Open Software License Agreement Version 2.0
 * (the "License"). Please refer to the License for details. You may not use
 * this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
 * AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
 * FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
 * for the full text of the License.
 */

#ifndef TEST_CSRSORT_ARCH35_CSRSORT_GOLDEN_H_
#define TEST_CSRSORT_ARCH35_CSRSORT_GOLDEN_H_

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <vector>

namespace sparse_test {

// ============================================================================
// Host-side semantic anchor for aclsparseXcsrsort (kernel-independent).
//
// Sorting contract:
//   - Per-row stable sort of csrColInd in ascending order.
//   - P is reordered using the same permutation.
//   - csrRowPtr is not modified.
//   - indexBase: 0 or 1; begin = csrRowPtr[row] - indexBase.
// ============================================================================

struct CsrsortGoldenResult {
  std::vector<int32_t> colInd;
  std::vector<int32_t> P;
};

inline CsrsortGoldenResult csrsortGolden(const std::vector<int32_t> &rowPtr,
                                         const std::vector<int32_t> &colInd,
                                         const std::vector<int32_t> &P, int m,
                                         int indexBase) {
  CsrsortGoldenResult r;
  r.colInd = colInd;
  r.P = P;

  for (int row = 0; row < m; row++) {
    int begin = rowPtr[row] - indexBase;
    int end = rowPtr[row + 1] - indexBase;
    int len = end - begin;
    if (len <= 1) {
      continue;
    }
    std::vector<int32_t> perm(len);
    std::iota(perm.begin(), perm.end(), 0);
    std::stable_sort(perm.begin(), perm.end(), [&](int a, int b) {
      return r.colInd[begin + a] < r.colInd[begin + b];
    });
    std::vector<int32_t> sortedCol(len);
    std::vector<int32_t> sortedP(len);
    for (int i = 0; i < len; i++) {
      sortedCol[i] = r.colInd[begin + perm[i]];
      sortedP[i] = r.P[begin + perm[i]];
    }
    for (int i = 0; i < len; i++) {
      r.colInd[begin + i] = sortedCol[i];
      r.P[begin + i] = sortedP[i];
    }
  }
  return r;
}

} // namespace sparse_test

#endif // TEST_CSRSORT_ARCH35_CSRSORT_GOLDEN_H_
