/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once

#include <cstddef>
#include <cstdint>

#include "acl/acl.h"
#include "cann_ops_sparse.h"
#include "descriptor_manager.h"

/**
 * @brief NPU wrapper for aclsparseSnnz (Legacy API).
 *
 * Encapsulates the full device-side workflow:
 *   1. Host->Device copy of input dense matrix A
 *   2. Device memory allocation for nnzPerUnit and nnzTotal outputs
 *   3. MatDescr creation (GENERAL + ZERO-based)
 *   4. aclsparseSnnz invocation
 *   5. Stream synchronization
 *   6. Device->Host copy of results
 *   7. MatDescr destruction
 *
 * @param handle   aclsparse handle (must already have stream set)
 * @param dir      ACL_SPARSE_DIRECTION_ROW or ACL_SPARSE_DIRECTION_COLUMN
 * @param m        number of rows
 * @param n        number of columns
 * @param A        host pointer to dense column-major matrix (size lda*n)
 * @param lda      leading dimension of A
 * @param nnzPerUnit  host output buffer for per-row/column nnz counts (size m or n)
 * @param totalUnits  number of output units (m if ROW, n if COLUMN)
 * @param nnzTotal    host output pointer for total nnz count
 * @return aclsparseStatus_t
 */
inline aclsparseStatus_t aclsparseSnnz_npu(
    aclsparseHandle_t handle,
    aclsparseDirection_t dir,
    int m,
    int n,
    const float *A,
    int lda,
    int32_t *nnzPerUnit,
    int totalUnits,
    int32_t *nnzTotal,
    aclrtStream stream,
    bool useHostPointerMode = false)
{
    if (handle == nullptr) {
        return ACL_SPARSE_STATUS_NOT_INITIALIZED;
    }

    // Set pointer mode
    aclsparsePointerMode_t ptrMode = useHostPointerMode
        ? ACL_SPARSE_POINTER_MODE_HOST
        : ACL_SPARSE_POINTER_MODE_DEVICE;
    aclsparseSetPointerMode(handle, ptrMode);

    // 1. H2D: copy input matrix A to device
    auto dA = sparse_test::DeviceBuffer::copyFrom(A, static_cast<size_t>(lda) * n * sizeof(float));

    // 2. Allocate device output buffer for nnzPerUnit (always device)
    auto dNnzPerUnit = sparse_test::DeviceBuffer::alloc(static_cast<size_t>(totalUnits) * sizeof(int32_t));

    // 3. Create MatDescr (Legacy API)
    aclsparseMatDescr_t matDescr = nullptr;
    auto descrRet = aclsparseCreateMatDescr(&matDescr);
    if (descrRet != ACL_SPARSE_STATUS_SUCCESS) {
        return descrRet;
    }
    aclsparseSetMatType(matDescr, ACL_SPARSE_MATRIX_TYPE_GENERAL);
    aclsparseSetMatIndexBase(matDescr, ACL_SPARSE_INDEX_BASE_ZERO);

    // 4. Call aclsparseSnnz
    aclsparseStatus_t ret;
    if (useHostPointerMode) {
        // HOST mode: pass host pointer directly as nnzTotalDevHostPtr
        // 函数异步返回，D2H 挂在 stream 上，需要显式同步后读取 host 指针
        ret = aclsparseSnnz(handle, dir, m, n, matDescr,
                             static_cast<const float *>(dA.raw()), lda,
                             static_cast<int *>(dNnzPerUnit.get()),
                             nnzTotal);
        aclrtSynchronizeStream(stream);
    } else {
        // DEVICE mode: allocate device buffer for total
        auto dNnzTotal = sparse_test::DeviceBuffer::alloc(sizeof(int32_t));
        ret = aclsparseSnnz(handle, dir, m, n, matDescr,
                             static_cast<const float *>(dA.raw()), lda,
                             static_cast<int *>(dNnzPerUnit.get()),
                             static_cast<int *>(dNnzTotal.get()));

        // Synchronize and D2H for total
        aclrtSynchronizeStream(stream);
        if (nnzTotal != nullptr) {
            dNnzTotal.copyToHost(nnzTotal, sizeof(int32_t));
        }

        // D2H for nnzPerUnit
        if (nnzPerUnit != nullptr) {
            dNnzPerUnit.copyToHost(nnzPerUnit, static_cast<size_t>(totalUnits) * sizeof(int32_t));
        }

        aclsparseDestroyMatDescr(matDescr);
        return ret;
    }

    // 5. D2H: copy nnzPerUnit back to host (HOST mode path)
    if (nnzPerUnit != nullptr) {
        dNnzPerUnit.copyToHost(nnzPerUnit, static_cast<size_t>(totalUnits) * sizeof(int32_t));
    }

    // 7. Cleanup MatDescr
    aclsparseDestroyMatDescr(matDescr);

    return ret;
}
