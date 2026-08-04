/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CSR2COO_NPU_WRAPPER_H_
#define CSR2COO_NPU_WRAPPER_H_

#include <cstdint>
#include <iostream>
#include <vector>
#include <algorithm>

#include "acl/acl.h"
#include "cann_ops_sparse.h"
#include "descriptor_manager.h"

// ============================================================================
// NPU Result structure for csr2coo (Legacy API)
// ============================================================================

struct Csr2CooNpuResult {
    std::vector<int32_t> cooRowInd;      // output (nnz elements)
    aclsparseStatus_t    ret;            // API return code
};

// ============================================================================
// NPU wrapper: single flat call — no SpMatDescr, no GetBufferSize
//
// Legacy API signature:
//   aclsparseXcsr2coo(handle, csrRowPtr, nnz, m, cooRowInd, idxBase)
//
// Steps:
//   1. Copy host csrRowPtr to device (H2D)
//   2. Alloc device cooRowInd
//   3. Call aclsparseXcsr2coo
//   4. Sync stream
//   5. Copy device cooRowInd to host (D2H)
//
// All device memory managed via DeviceBuffer RAII.
// Handle managed via HandleManager RAII (passed in by caller).
// ============================================================================

inline Csr2CooNpuResult NpuCsr2coo(
    sparse_test::HandleManager &handle,
    aclrtStream stream,
    const int32_t *hostCsrRowPtr,
    int64_t nnz,
    int64_t m,
    aclsparseIndexBase_t idxBase)
{
    using namespace sparse_test;
    Csr2CooNpuResult result{};

    // Fix 6: nnz<0 防护。static_cast<size_t>(nnz) 对负值会产生巨大值导致错误 alloc，
    // 此处提前拦截（Host 校验已拦截，此为防御性二次校验）。
    if (nnz < 0) {
        result.ret = ACL_SPARSE_STATUS_INVALID_VALUE;
        return result;
    }

    // T5: m<0 防护（与 nnz<0 防护对称）。static_cast<size_t>(m+1) 对负值会产生巨大值。
    if (m < 0) {
        result.ret = ACL_SPARSE_STATUS_INVALID_VALUE;
        return result;
    }

    size_t rowPtrBytes = static_cast<size_t>(m + 1) * sizeof(int32_t);
    constexpr size_t kMinAllocBytes = 1;
    size_t idxBytes = std::max(static_cast<size_t>(nnz) * sizeof(int32_t), kMinAllocBytes);

    auto dRowPtr = DeviceBuffer::copyFrom(hostCsrRowPtr, rowPtrBytes);
    auto dCooRowInd = DeviceBuffer::alloc(idxBytes);

    result.ret = aclsparseXcsr2coo(
        handle.get(),
        static_cast<const int32_t *>(dRowPtr.get()),
        nnz,
        m,
        static_cast<int32_t *>(dCooRowInd.get()),
        idxBase);

    if (result.ret != ACL_SPARSE_STATUS_SUCCESS) {
        return result;
    }

    if (nnz == 0) {
        return result;
    }

    aclError syncRet = aclrtSynchronizeStream(stream);
    if (syncRet != ACL_SUCCESS) {
        std::cerr << "[csr2coo_npu_wrapper] aclrtSynchronizeStream failed, aclError=" << syncRet << std::endl;
        result.ret = ACL_SPARSE_STATUS_INTERNAL_ERROR;
        return result;
    }

    result.cooRowInd.resize(static_cast<size_t>(nnz));
    dCooRowInd.copyToHost(result.cooRowInd.data(),
                          static_cast<size_t>(nnz) * sizeof(int32_t));

    return result;
}

#endif  // CSR2COO_NPU_WRAPPER_H_
