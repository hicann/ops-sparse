/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TEST_SPSV_GOLDEN_H_
#define TEST_SPSV_GOLDEN_H_

#include <Eigen/Eigen>
#include <Eigen/SparseCore>
#include <vector>
#include "../frame/fill.h"

namespace sparse_test {

template <typename T>
std::vector<double> ComputeRhs(const std::vector<T>& x, T alpha) {
    std::vector<double> rhs(x.size());
    for (size_t i = 0; i < x.size(); i++) {
        rhs[i] = static_cast<double>(alpha) * static_cast<double>(x[i]);
    }
    return rhs;
}

inline Eigen::SparseMatrix<double, Eigen::RowMajor> BuildEigenMatrix(const CsrMatrix& csr) {
    int m = static_cast<int>(csr.rows);
    Eigen::SparseMatrix<double, Eigen::RowMajor> A(m, m);
    std::vector<Eigen::Triplet<double>> triplets;
    triplets.reserve(static_cast<size_t>(csr.nnz));
    for (int i = 0; i < m; i++) {
        for (int j = static_cast<int>(csr.rowOffsets[i]); j < static_cast<int>(csr.rowOffsets[i + 1]); j++) {
            triplets.emplace_back(i, static_cast<int>(csr.colIndices[j]),
                                  static_cast<double>(csr.values[j]));
        }
    }
    A.setFromTriplets(triplets.begin(), triplets.end());
    return A;
}

template <bool FORWARD>
void SolveTriangularLoop(const CsrMatrix& csr, const std::vector<double>& rhs,
                         bool unitDiag, std::vector<double>& y) {
    int m = static_cast<int>(csr.rows);
    if constexpr (FORWARD) {
        for (int i = 0; i < m; i++) {
            double sum = rhs[i];
            double diagVal = 0.0;
            for (int k = csr.rowOffsets[i]; k < csr.rowOffsets[i + 1]; k++) {
                int col = csr.colIndices[k];
                if (col < i) {
                    sum -= static_cast<double>(csr.values[k]) * y[col];
                } else if (col == i) {
                    diagVal = static_cast<double>(csr.values[k]);
                }
            }
            y[i] = unitDiag ? sum : sum / diagVal;
        }
    } else {
        for (int i = m - 1; i >= 0; i--) {
            double sum = rhs[i];
            double diagVal = 0.0;
            for (int k = csr.rowOffsets[i]; k < csr.rowOffsets[i + 1]; k++) {
                int col = csr.colIndices[k];
                if (col > i) {
                    sum -= static_cast<double>(csr.values[k]) * y[col];
                } else if (col == i) {
                    diagVal = static_cast<double>(csr.values[k]);
                }
            }
            y[i] = unitDiag ? sum : sum / diagVal;
        }
    }
}

inline bool DetermineForward(bool lower, bool transpose) {
    int fm = lower ? 0 : 1;
    int op = transpose ? 1 : 0;
    if (fm == 0 && op == 0) return true;
    if (fm == 1 && op != 0) return true;
    return false;
}

template <typename T>
std::vector<T> SpSVGolden(
    const CsrMatrix& csr,
    const std::vector<T>& x,
    T alpha,
    bool lower,
    bool unitDiag,
    bool transpose)
{
    int m = static_cast<int>(csr.rows);
    if (m <= 0) return {};

    auto A = BuildEigenMatrix(csr);
    auto rhsVec = ComputeRhs(x, alpha);
    Eigen::VectorXd rhs(m);
    for (int i = 0; i < m; i++) rhs(i) = rhsVec[i];

    Eigen::VectorXd result(m);
    if (!transpose) {
        if (lower && !unitDiag) {
            result = A.triangularView<Eigen::Lower>().solve(rhs);
        } else if (lower && unitDiag) {
            result = A.triangularView<Eigen::Lower | Eigen::UnitDiag>().solve(rhs);
        } else if (!lower && !unitDiag) {
            result = A.triangularView<Eigen::Upper>().solve(rhs);
        } else {
            result = A.triangularView<Eigen::Upper | Eigen::UnitDiag>().solve(rhs);
        }
    } else {
        if (lower && !unitDiag) {
            result = A.triangularView<Eigen::Lower>().transpose().solve(rhs);
        } else if (lower && unitDiag) {
            result = A.triangularView<Eigen::Lower | Eigen::UnitDiag>().transpose().solve(rhs);
        } else if (!lower && !unitDiag) {
            result = A.triangularView<Eigen::Upper>().transpose().solve(rhs);
        } else {
            result = A.triangularView<Eigen::Upper | Eigen::UnitDiag>().transpose().solve(rhs);
        }
    }

    std::vector<T> output(m);
    for (int i = 0; i < m; i++) {
        output[i] = static_cast<T>(result(i));
    }
    return output;
}

template <typename T>
std::vector<T> SpSVGoldenManual(
    const CsrMatrix& csr,
    const std::vector<T>& x,
    T alpha,
    bool lower,
    bool unitDiag,
    bool transpose)
{
    int m = static_cast<int>(csr.rows);
    if (m <= 0) return {};

    auto rhs = ComputeRhs(x, alpha);
    std::vector<double> y(m, 0.0);

    bool forward = DetermineForward(lower, transpose);
    if (forward) {
        SolveTriangularLoop<true>(csr, rhs, unitDiag, y);
    } else {
        SolveTriangularLoop<false>(csr, rhs, unitDiag, y);
    }

    std::vector<T> output(m);
    for (int i = 0; i < m; i++) {
        output[i] = static_cast<T>(y[i]);
    }
    return output;
}

}

#endif
