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

#include <iostream>
#include <vector>
#include <cstdlib>
#include "cann_ops_sparse.h"

int main()
{
    std::cout << "========== SpMM Test ==========" << std::endl;

    // Simple test case: 3x3 sparse matrix A multiplied by 3x2 dense matrix B
    int m = 3;   // rows in A and C
    int n = 2;   // cols in B and C
    int k = 3;   // cols in A, rows in B
    int nnz = 5; // non-zeros in A

    // CSR format for A
    std::vector<float> csrVal = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    std::vector<int> csrRowPtr = {0, 2, 3, 5};
    std::vector<int> csrColInd = {0, 2, 1, 0, 2};

    // Dense matrix B (3x2, row-major)
    std::vector<float> B = {1.0f, 2.0f,
                            3.0f, 4.0f,
                            5.0f, 6.0f};

    // Output matrix C (3x2)
    std::vector<float> C(m * n, 0.0f);

    float alpha = 1.0f;
    float beta = 0.0f;

    int result = cann_ops_sparse::spmm(m, n, k, nnz, alpha,
                                       csrVal.data(), csrRowPtr.data(), csrColInd.data(),
                                       B.data(), k,
                                       beta,
                                       C.data(), n);

    if (result == 0) {
        std::cout << "SpMM test PASSED" << std::endl;
        return 0;
    } else {
        std::cout << "SpMM test FAILED with code: " << result << std::endl;
        return 1;
    }
}
