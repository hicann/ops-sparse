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
    std::cout << "========== SpMV Test ==========" << std::endl;

    // Simple test case: 3x3 sparse matrix
    // A = [1 0 2]
    //     [0 3 0]
    //     [4 0 5]
    // x = [1, 2, 3]
    // y = alpha * A * x + beta * y

    int m = 3;  // rows
    int n = 3;  // cols
    int nnz = 5; // non-zeros

    // CSR format
    std::vector<float> csrVal = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    std::vector<int> csrRowPtr = {0, 2, 3, 5};
    std::vector<int> csrColInd = {0, 2, 1, 0, 2};

    std::vector<float> x = {1.0f, 2.0f, 3.0f};
    std::vector<float> y = {0.0f, 0.0f, 0.0f};

    float alpha = 1.0f;
    float beta = 0.0f;

    int result = cann_ops_sparse::spmv(m, n, nnz, alpha,
                                       csrVal.data(), csrRowPtr.data(), csrColInd.data(),
                                       x.data(), beta, y.data());

    if (result == 0) {
        std::cout << "SpMV test PASSED" << std::endl;
        return 0;
    } else {
        std::cout << "SpMV test FAILED with code: " << result << std::endl;
        return 1;
    }
}
