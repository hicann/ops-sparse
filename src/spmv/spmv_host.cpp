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

#include "cann_ops_sparse.h"
#include <cstddef>

namespace cann_ops_sparse {

// Host-side implementation stub for SpMV
// In a full implementation, this would:
// 1. Prepare tiling and launch kernel
// 2. Handle memory management
// 3. Call the Ascend kernel

int spmv(int m, int n, int nnz,
         float alpha,
         const float* csrVal,
         const int* csrRowPtr,
         const int* csrColInd,
         const float* x,
         float beta,
         float* y)
{
    // Stub implementation - just return success
    // Real implementation would launch Ascend kernel here
    (void)m;
    (void)n;
    (void)nnz;
    (void)alpha;
    (void)csrVal;
    (void)csrRowPtr;
    (void)csrColInd;
    (void)x;
    (void)beta;
    (void)y;
    return 0;
}

} // namespace cann_ops_sparse
