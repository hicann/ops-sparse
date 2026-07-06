/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TEST_NNZ_SNNZ_SNNZ_GOLDEN_H_
#define TEST_NNZ_SNNZ_SNNZ_GOLDEN_H_

#include <cstdint>
#include <vector>

/**
 * @brief Reference implementation for aclsparseSnnz.
 *
 * Counts non-zero elements in a dense column-major matrix.
 * This is a simple CPU-based reference implementation for verification.
 *
 * @param A         Dense matrix data (column-major, lda x n)
 * @param m         Number of rows
 * @param n         Number of columns
 * @param lda       Leading dimension (>= m for column-major)
 * @param dirRow    true = count per row, false = count per column
 * @param nnzPerUnit Output: nnz count per row (size m) or per column (size n)
 * @param nnzTotal  Output: total nnz count
 */
static void ReferenceNnz(const float *A, int m, int n, int lda,
                         bool dirRow,
                         std::vector<int32_t> &nnzPerUnit,
                         int32_t &nnzTotal) {
    if (dirRow) {
        nnzPerUnit.assign(m, 0);
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (A[static_cast<int64_t>(j) * lda + i] != 0.0f) {
                    nnzPerUnit[i]++;
                }
            }
        }
    } else {
        nnzPerUnit.assign(n, 0);
        for (int j = 0; j < n; j++) {
            for (int i = 0; i < m; i++) {
                if (A[static_cast<int64_t>(j) * lda + i] != 0.0f) {
                    nnzPerUnit[j]++;
                }
            }
        }
    }

    nnzTotal = 0;
    for (size_t i = 0; i < nnzPerUnit.size(); i++) {
        nnzTotal += nnzPerUnit[i];
    }
}

#endif  // TEST_NNZ_SNNZ_SNNZ_GOLDEN_H_
