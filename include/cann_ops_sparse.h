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

#ifndef CANN_OPS_SPARSE_H
#define CANN_OPS_SPARSE_H

#include <cstdint>

namespace cann_ops_sparse {

/**
 * @brief Sparse Matrix-Vector Multiplication (SpMV)
 *
 * Performs y = alpha * A * x + beta * y
 * where A is a sparse matrix in CSR format.
 *
 * @param m Number of rows in matrix A
 * @param n Number of columns in matrix A
 * @param nnz Number of non-zero elements in matrix A
 * @param alpha Scalar multiplier for A * x
 * @param csrVal Array of non-zero values (size nnz)
 * @param csrRowPtr Array of row pointers (size m+1)
 * @param csrColInd Array of column indices (size nnz)
 * @param x Vector x (size n)
 * @param beta Scalar multiplier for y
 * @param y Vector y (size m), input and output
 * @return Status code (0 for success)
 */
int spmv(int m, int n, int nnz,
         float alpha,
         const float* csrVal,
         const int* csrRowPtr,
         const int* csrColInd,
         const float* x,
         float beta,
         float* y);

/**
 * @brief Sparse Matrix-Matrix Multiplication (SpMM)
 *
 * Performs C = alpha * A * B + beta * C
 * where A is a sparse matrix in CSR format, B and C are dense matrices.
 *
 * @param m Number of rows in matrix A and C
 * @param n Number of columns in matrix B and C
 * @param k Number of columns in matrix A and rows in matrix B
 * @param nnz Number of non-zero elements in matrix A
 * @param alpha Scalar multiplier for A * B
 * @param csrVal Array of non-zero values (size nnz)
 * @param csrRowPtr Array of row pointers (size m+1)
 * @param csrColInd Array of column indices (size nnz)
 * @param B Dense matrix B (size k x n, column-major or row-major)
 * @param ldb Leading dimension of B
 * @param beta Scalar multiplier for C
 * @param C Dense matrix C (size m x n)
 * @param ldc Leading dimension of C
 * @return Status code (0 for success)
 */
int spmm(int m, int n, int k, int nnz,
         float alpha,
         const float* csrVal,
         const int* csrRowPtr,
         const int* csrColInd,
         const float* B, int ldb,
         float beta,
         float* C, int ldc);

} // namespace cann_ops_sparse

#endif // CANN_OPS_SPARSE_H
