/**
 * ----------------------------------------------------------------------------------------------------------
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ----------------------------------------------------------------------------------------------------------
 */

#ifndef TEST_MATMUL_EXCEPTION_HELPER_H_
#define TEST_MATMUL_EXCEPTION_HELPER_H_

// =============================================================================
// Exception test helpers for aclsparseLtMatmul.
//
// Extracted from matmul_npu_wrapper.h to reduce header file size.
//
// Contents:
//   - MatmulRawParams: raw matmul call params with injectable invalid values
//   - MatmulNpuRaw: direct aclsparseLtMatmul wrapper for exception tests
//   - MatmulExceptionCtx: RAII context for building a valid 8x8 plan
// =============================================================================

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "acl/acl.h"
#include "cann_ops_sparseLt.h"
#include "descriptor_manager.h"

#include "ltmatmul_npu_wrapper.h"

// -----------------------------------------------------------------------------
// Exception test: raw matmul call with injectable invalid parameters.
// -----------------------------------------------------------------------------
struct MatmulRawParams {
    aclsparseLtConstHandle_t handle = nullptr;
    aclsparseLtConstMatmulPlan_t plan = nullptr;
    const void* alpha = nullptr;
    const void* matA = nullptr;
    const void* matB = nullptr;
    const void* beta = nullptr;
    const void* matC = nullptr;
    void* matD = nullptr;
    void* workspace = nullptr;
    aclrtStream* streams = nullptr;
    int32_t numStreams = 1;
};

inline aclsparseStatus_t MatmulNpuRaw(const MatmulRawParams& p)
{
    aclsparseLtConstMatmulPlan_t plan = p.plan;
    return aclsparseLtMatmul(p.handle, &plan, p.alpha, p.matA, p.matB,
                             p.beta, p.matC, p.matD, p.workspace,
                             p.streams, p.numStreams);
}

// -----------------------------------------------------------------------------
// Helper: build a valid plan for exception tests (8x8 FP32 descriptors).
// -----------------------------------------------------------------------------
struct MatmulExceptionCtx {
    sparse_test::SparseLtHandleGuard handle;
    sparse_test::SparseLtMatDescGuard matA;
    sparse_test::SparseLtDnMatDescGuard matB;
    sparse_test::SparseLtDnMatDescGuard matC;
    sparse_test::SparseLtDnMatDescGuard matD;
    sparse_test::SparseLtMatmulDescGuard matmulDesc;
    sparse_test::SparseLtAlgSelectionGuard algSel;
    sparse_test::SparseLtPlanGuard plan;

    aclrtStream stream = nullptr;
    sparse_test::DeviceBuffer dA;
    sparse_test::DeviceBuffer dB;
    sparse_test::DeviceBuffer dC;
    sparse_test::DeviceBuffer dD;
    sparse_test::DeviceBuffer dWorkspace;
    size_t workspaceSize = 0;

    MatmulExceptionCtx(int32_t m = 8, int32_t k = 8, int32_t n = 8,
                       aclDataType dtype = ACL_FLOAT,
                       bool isDense = false)
        : handle(),
          matA(handle.get(), k, m, m, 16, dtype,
               ACL_SPARSE_ORDER_ROW,
                ACL_SPARSE_LT_SPARSITY_50_PERCENT),
          matB(handle.get(), k, n, n, 16, dtype, ACL_SPARSE_ORDER_ROW),
          matC(handle.get(), m, n, n, 16, dtype, ACL_SPARSE_ORDER_ROW),
          matD(handle.get(), m, n, n, 16, dtype, ACL_SPARSE_ORDER_ROW),
          matmulDesc(handle.get(), ACL_SPARSE_OP_NON_TRANSPOSE, ACL_SPARSE_OP_NON_TRANSPOSE,
                     matA.get(), matB.get(), matC.get(), matD.get(),
                     ACL_SPARSE_COMPUTE_32F),
          algSel(handle.get(), matmulDesc.get(), ACL_SPARSE_LT_MATMUL_ALG_DEFAULT),
          plan(handle.get(), matmulDesc.get(), algSel.get())
    {
        aclError streamRet = aclrtCreateStream(&stream);
        if (streamRet != ACL_SUCCESS) {
            throw std::runtime_error("MatmulExceptionCtx: aclrtCreateStream failed");
        }
        const size_t mk = static_cast<size_t>(m) * k;
        const size_t kn = static_cast<size_t>(k) * n;
        const size_t mn = static_cast<size_t>(m) * n;
        std::vector<float> hA(mk, 1.0f);
        std::vector<float> hB(kn, 1.0f);
        std::vector<float> hC(mn, 0.0f);
        dA = sparse_test::DeviceBuffer::copyFrom(hA.data(), mk * sizeof(float));
        dB = sparse_test::DeviceBuffer::copyFrom(hB.data(), kn * sizeof(float));
        dC = sparse_test::DeviceBuffer::copyFrom(hC.data(), mn * sizeof(float));
        dD = sparse_test::DeviceBuffer::alloc(mn * sizeof(float));
        aclsparseStatus_t wsRet = aclsparseLtMatmulGetWorkspace(handle.get(), plan.cptr(), &workspaceSize);
        if (wsRet != ACL_SPARSE_STATUS_SUCCESS) {
            throw std::runtime_error("MatmulExceptionCtx: GetWorkspaceSize failed");
        }
        if (workspaceSize > 0) {
            dWorkspace = sparse_test::DeviceBuffer::alloc(workspaceSize);
        }
    }

    ~MatmulExceptionCtx() {
        if (stream) { aclrtDestroyStream(stream); }
    }
};

#endif  // TEST_MATMUL_EXCEPTION_HELPER_H_
