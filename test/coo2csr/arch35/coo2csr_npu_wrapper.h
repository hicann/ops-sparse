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

#ifndef COO2CSR_NPU_WRAPPER_H_
#define COO2CSR_NPU_WRAPPER_H_

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "acl/acl.h"
#include "cann_ops_sparse.h"
#include "descriptor_manager.h"

static_assert(sizeof(int) == sizeof(int32_t), "int must be 32-bit");

// ============================================================================
// NPU result structure for coo2csr
// ============================================================================

struct Coo2CsrNpuResult {
    std::vector<int32_t> csrRowPtr;   // size m+1
    aclsparseStatus_t computeRet;     // API return status
};

// ============================================================================
// NPU wrapper: Full coo2csr workflow (Legacy API)
//
// Step 1: DeviceBuffer::copyFrom() — upload cooRowInd to device
// Step 2: DeviceBuffer::alloc()    — allocate csrRowPtr output space
// Step 3: aclsparseXcoo2csr()       — execute conversion
// Step 4: aclrtSynchronizeStream() — wait for completion
// Step 5: DeviceBuffer::copyToHost() — download csrRowPtr to host
//
// Device memory managed via DeviceBuffer RAII.
// Handle managed via HandleManager RAII.
// Every ACL call checks return status and logs on failure.
// ============================================================================

// ----------------------------------------------------------------------------
// Helper: Ensure handle workspace is sufficient for large m.
//
// Default handle workspace is 4 MiB; coo2csr needs m*4 + aivCoreNum*4 bytes.
// For m near 1048576 the default is insufficient — provide a user workspace.
// On success, dUserWorkspace holds the allocated buffer (if any).
// Returns ACL_SPARSE_STATUS_SUCCESS or the aclsparseSetWorkspace error code.
// ----------------------------------------------------------------------------
inline aclsparseStatus_t Coo2CsrEnsureWorkspace(
    sparse_test::HandleManager& handle,
    int m,
    sparse_test::DeviceBuffer& dUserWorkspace)
{
    constexpr size_t kDefaultWs = 4U * 1024U * 1024U;
    size_t neededWorkspace = static_cast<size_t>(m) * sizeof(int32_t) + 4096U;
    if (neededWorkspace <= kDefaultWs) {
        return ACL_SPARSE_STATUS_SUCCESS;
    }
    dUserWorkspace = sparse_test::DeviceBuffer::alloc(neededWorkspace);
    return aclsparseSetWorkspace(
        handle.get(), dUserWorkspace.get(), neededWorkspace);
}

inline Coo2CsrNpuResult Coo2CsrNpu(
    sparse_test::HandleManager& handle,
    aclrtStream stream,
    const int32_t* cooRowIndHost,
    int nnz,
    int m,
    aclsparseIndexBase_t idxBase)
{
    using namespace sparse_test;
    Coo2CsrNpuResult result{};

    // --- Step 1: Copy cooRowInd to device ---
    // Guard against zero-size allocation when nnz=0.
    // When nnz == 0, the API does not dereference cooRowInd (validated by the
    // Nnz0NullCooRowInd exception case returning SUCCESS). However, the host
    // pointer passed in may be nullptr, which would cause aclrtMemcpy to fail.
    // In that case we only allocate device memory (no H2D copy) so the wrapper
    // still hands a valid device pointer to the operator — matching the
    // operator's documented nnz=0 contract.
    size_t cooRowIndBytes = std::max(static_cast<size_t>(nnz) * sizeof(int32_t), size_t(1));
    DeviceBuffer dCooRowInd;
    if (nnz > 0 && cooRowIndHost != nullptr) {
        dCooRowInd = DeviceBuffer::copyFrom(cooRowIndHost, cooRowIndBytes);
    } else {
        dCooRowInd = DeviceBuffer::alloc(cooRowIndBytes);
    }

    // --- Step 2: Allocate csrRowPtr output buffer on device ---
    auto dCsrRowPtr = DeviceBuffer::alloc((static_cast<size_t>(m) + 1) * sizeof(int32_t));

    // --- Step 2.5: Ensure workspace sufficient for large m ---
    DeviceBuffer dUserWorkspace;
    aclsparseStatus_t wsRet = Coo2CsrEnsureWorkspace(handle, m, dUserWorkspace);
    if (wsRet != ACL_SPARSE_STATUS_SUCCESS) {
        std::cerr << "[NPU] aclsparseSetWorkspace failed: " << wsRet << std::endl;
        result.computeRet = wsRet;
        return result;
    }

    // --- Step 3: Call aclsparseXcoo2csr ---
    result.computeRet = aclsparseXcoo2csr(handle.get(), reinterpret_cast<const int*>(dCooRowInd.get()),
        nnz, m, reinterpret_cast<int*>(dCsrRowPtr.get()), idxBase);

    if (result.computeRet != ACL_SPARSE_STATUS_SUCCESS) {
        std::cerr << "[NPU] aclsparseXcoo2csr failed: " << result.computeRet << std::endl;
        auto syncRet = aclrtSynchronizeStream(stream);
        if (syncRet != ACL_SUCCESS) {
            std::cerr << "[NPU] aclrtSynchronizeStream failed during error cleanup: " << syncRet << std::endl;
        }
        return result;
    }

    // --- Step 4: Synchronize ---
    auto syncRet = aclrtSynchronizeStream(stream);
    if (syncRet != ACL_SUCCESS) {
        std::cerr << "[NPU] aclrtSynchronizeStream failed: " << syncRet << std::endl;
        result.computeRet = ACL_SPARSE_STATUS_EXECUTION_FAILED;
        return result;
    }

    // --- Step 5: Copy csrRowPtr back to host ---
    size_t csrRowPtrBytes = (static_cast<size_t>(m) + 1) * sizeof(int32_t);
    result.csrRowPtr.resize(static_cast<size_t>(m) + 1);
    try {
        dCsrRowPtr.copyToHost(result.csrRowPtr.data(), csrRowPtrBytes);
    } catch (const std::runtime_error& e) {
        std::cerr << "[NPU] copyToHost failed: " << e.what() << std::endl;
        result.computeRet = ACL_SPARSE_STATUS_EXECUTION_FAILED;
        return result;
    }

    return result;
}

#endif  // COO2CSR_NPU_WRAPPER_H_
