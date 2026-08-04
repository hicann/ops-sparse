/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TEST_CSR2COO_CSR2COO_GOLDEN_H_
#define TEST_CSR2COO_CSR2COO_GOLDEN_H_

#include <cstdint>
#include <vector>

namespace sparse_test {

// ============================================================================
// Golden reference for aclsparseXcsr2coo (Legacy API): CSR row-pointer -> COO row-index
//
// This is a purely index-expansion operation (no floating-point arithmetic).
// The comparison is bitwise exact (integer equality).
//
// Algorithm (format-agnostic — works for CSR rowPtr or CSC colPtr):
//   for i in [0, m):
//     rowStart = ptrArray[i]   - idxBase   (0-based offset)
//     rowEnd   = ptrArray[i+1] - idxBase
//     for j in [rowStart, rowEnd):
//       cooIndices[j] = i + idxBase            (idxBase-aware)
//
// idxBase support: input/output index arrays are offset by idxBase (0 or 1).
//
// Fixed int32_t — no templates needed (Legacy API INT32 only).
// ============================================================================

inline std::vector<int32_t> Csr2CooGolden(
    const std::vector<int32_t> &ptrArray,
    int64_t m,
    int64_t nnz,
    int idxBase)
{
    std::vector<int32_t> cooIndices(nnz, 0);

    if (static_cast<int64_t>(ptrArray.size()) < m + 1) {
        return cooIndices;
    }

    for (int64_t i = 0; i < m; i++) {
        int64_t rowStart = static_cast<int64_t>(ptrArray[static_cast<size_t>(i)]) - idxBase;
        int64_t rowEnd = static_cast<int64_t>(ptrArray[static_cast<size_t>(i + 1)]) - idxBase;
        if (rowStart < 0 || rowEnd > nnz || rowStart > rowEnd) {
            continue;
        }
        for (int64_t j = rowStart; j < rowEnd; j++) {
            cooIndices[static_cast<size_t>(j)] = static_cast<int32_t>(i) + static_cast<int32_t>(idxBase);
        }
    }

    return cooIndices;
}

}  // namespace sparse_test

#endif  // TEST_CSR2COO_CSR2COO_GOLDEN_H_
