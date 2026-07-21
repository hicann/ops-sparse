/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TEST_GTSV2_NOPIVOT_GTSV2_NOPIVOT_GOLDEN_H_
#define TEST_GTSV2_NOPIVOT_GTSV2_NOPIVOT_GOLDEN_H_

#include <cstdint>
#include <vector>

#include "fill.h"

namespace sparse_test {

/**
 * FP64 Thomas algorithm golden reference for gtsv2_nopivot (column-major layout).
 *
 * Solves A * X = B where A is a tridiagonal matrix defined by diagonals (dl, d, du):
 *   - dl: [m] sub-diagonal,  dl[0] is unused (0), dl[1..m-1] are the sub-diagonal entries
 *   - d:  [m] main diagonal
 *   - du: [m] super-diagonal, du[m-1] is unused (0), du[0..m-2] are the super-diagonal entries
 *
 * B is an m x n dense matrix in column-major layout (ldb x n).
 * Output X overwrites B in-place (also column-major, ldb x n).
 *
 * Each RHS column is solved independently using FP64 arithmetic.
 */
inline void Gtsv2NopivotGolden(
    const std::vector<float>& dl,
    const std::vector<float>& d,
    const std::vector<float>& du,
    std::vector<float>& B,  // in-place: input RHS, output solution
    int m, int n, int ldb)
{
    if (m <= 0 || n <= 0) return;

    // Pre-allocate scratch buffers to avoid repeated allocation for large n
    std::vector<double> c_prime(m, 0.0);
    std::vector<double> b_prime(m, 0.0);

    // Solve each RHS column independently using FP64 Thomas algorithm
    for (int j = 0; j < n; j++) {
        // Forward elimination: row 0
        {
            double d0  = static_cast<double>(d[0]);
            double du0 = static_cast<double>(du[0]);
            double b0  = static_cast<double>(B[0 + j * ldb]);
            c_prime[0] = du0 / d0;
            b_prime[0] = b0 / d0;
        }

        // Forward elimination: rows 1..m-1
        for (int i = 1; i < m; i++) {
            double dl_i = static_cast<double>(dl[i]);  // dl[0]=0, dl[1..m-1] are valid
            double d_i  = static_cast<double>(d[i]);
            double du_i = static_cast<double>(du[i]);
            double b_i  = static_cast<double>(B[i + j * ldb]);

            double denom = d_i - dl_i * c_prime[i - 1];
            if (i < m - 1) {
                c_prime[i] = du_i / denom;
            }
            b_prime[i] = (b_i - dl_i * b_prime[i - 1]) / denom;
        }

        // Back substitution
        B[(m - 1) + j * ldb] = static_cast<float>(b_prime[m - 1]);
        for (int i = m - 2; i >= 0; i--) {
            B[i + j * ldb] = static_cast<float>(b_prime[i] - c_prime[i] * static_cast<double>(B[(i + 1) + j * ldb]));
        }
    }
}

}  // namespace sparse_test

#endif  // TEST_GTSV2_NOPIVOT_GTSV2_NOPIVOT_GOLDEN_H_
