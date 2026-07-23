/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * SpMVOp CPU golden reference (Eigen SparseMatrix<double> FP64).
 * Formula: Z = alpha * A * X + beta * Y
 */

#ifndef TEST_SPMV_OP_GOLDEN_H_
#define TEST_SPMV_OP_GOLDEN_H_

#include <Eigen/Eigen>
#include <Eigen/SparseCore>
#include <vector>
#include "../frame/fill.h"

namespace sparse_test {

// SpMVOp golden: Z = alpha * A * X + beta * Y
// A: CSR matrix (rows = p.m, cols = p.k)
// X: dense vector (length = p.k)
// Y: dense vector (length = p.m)
// Returns: Z dense vector (length = p.m)
//
// Uses Eigen SparseMatrix<double> FP64 to avoid FP32 accumulation errors.
inline std::vector<float> SpMVOpGolden(
    const CsrMatrix& csr,
    const std::vector<float>& x,
    const std::vector<float>& y,
    float alpha, float beta)
{
    using Scalar = double;
    const int rows = static_cast<int>(csr.rows);
    const int cols = static_cast<int>(csr.cols);
    const int nnz = static_cast<int>(csr.nnz);

    if (rows <= 0) {
        return {};
    }

    // k=0: A*X=0, Z = beta * Y
    if (cols <= 0) {
        std::vector<float> output(rows);
        for (int i = 0; i < rows; i++) {
            output[i] = beta * y[i];
        }
        return output;
    }

    // Build Eigen SparseMatrix (RowMajor = CSR)
    Eigen::SparseMatrix<Scalar, Eigen::RowMajor> A(rows, cols);
    std::vector<Eigen::Triplet<Scalar>> triplets;
    triplets.reserve(nnz);
    for (int i = 0; i < rows; i++) {
        for (int j = csr.rowOffsets[i]; j < csr.rowOffsets[i + 1]; j++) {
            triplets.emplace_back(i, csr.colIndices[j],
                                   static_cast<Scalar>(csr.values[j]));
        }
    }
    A.setFromTriplets(triplets.begin(), triplets.end());

    // Build dense vectors (FP64)
    Eigen::VectorXd xVec(cols), yVec(rows);
    for (int i = 0; i < cols; i++) xVec(i) = static_cast<Scalar>(x[i]);
    for (int i = 0; i < rows; i++) yVec(i) = static_cast<Scalar>(y[i]);

    // Compute Z = alpha * A * X + beta * Y
    Eigen::VectorXd z = static_cast<Scalar>(alpha) * (A * xVec)
                      + static_cast<Scalar>(beta)  * yVec;

    // Convert back to FP32
    std::vector<float> output(rows);
    for (int i = 0; i < rows; i++) {
        output[i] = static_cast<float>(z(i));
    }
    return output;
}

}  // namespace sparse_test

#endif
