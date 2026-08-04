/**
 * ----------------------------------------------------------------------------------------------------------
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ----------------------------------------------------------------------------------------------------------
 */

#ifndef TEST_COO2CSR_COO2CSR_GOLDEN_H_
#define TEST_COO2CSR_COO2CSR_GOLDEN_H_

#include <cstdint>
#include <vector>

// Note: This golden does NOT include "fill.h" because it operates purely on
// int32 index arrays. No sparse matrix structures (CsrMatrix/CooMatrix) are needed.

namespace sparse_test {

// ============================================================================
// CPU golden reference for aclsparseXcoo2csr: COO row indices -> CSR row pointers
//
// Algorithm: counting + prefix sum (identical to the operator semantics).
//
// NOTE: This golden does NOT use Eigen SparseMatrix<double> FP64 because the
// coo2csr operator is a pure integer index-array transformation (no floating-point
// arithmetic). Using int32_t directly avoids unnecessary type conversions and
// guarantees bit-exact results. This is a justified deviation from the standard
// Eigen FP64 golden pattern.
//
// Input:  cooRowInd[0..nnz-1]  — sorted COO row indices (in idxBase)
// Output: csrRowPtr[0..m]      — CSR row pointers (in idxBase)
//
// Parameter validation matches the NPU operator's parameter constraints:
//   - nnz >= 0, m >= 0
//   - idxBase must be 0 or 1
//   - When nnz == 0 or m == 0: csrRowPtr filled with idxBase
// ============================================================================

inline std::vector<int32_t> Coo2CsrGolden(
    const std::vector<int32_t>& cooRowInd,
    int nnz, int m, int idxBase)
{
    // m 或 nnz 为负时返回空（防御，正常路径上游已校验）
    if (m < 0 || nnz < 0) {
        return {};
    }

    // Initialize all elements to idxBase (correct for nnz=0 or m=0)
    std::vector<int32_t> csrRowPtr(static_cast<size_t>(m) + 1, idxBase);

    if (nnz == 0 || m == 0) return csrRowPtr;

    // Step 1: Count nonzeros per row
    std::vector<int32_t> rowCount(m, 0);
    for (int k = 0; k < nnz; k++) {
        // uint32 减法避免有符号溢出（与 kernel Coo2CsrRleCount 一致）
        int row = static_cast<int>(static_cast<unsigned>(cooRowInd[k]) - static_cast<unsigned>(idxBase));
        if (row >= 0 && row < m) {
            rowCount[row]++;
        }
    }

    // Step 2: Prefix sum to generate csrRowPtr (int64 runningSum 防溢出，与 kernel 一致)
    csrRowPtr[0] = idxBase;
    int64_t runningSum = 0;
    for (int i = 0; i < m; i++) {
        runningSum += rowCount[i];
        csrRowPtr[i + 1] = static_cast<int32_t>(runningSum + idxBase);
    }

    return csrRowPtr;
}

}  // namespace sparse_test

#endif  // TEST_COO2CSR_COO2CSR_GOLDEN_H_
