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

#ifndef CSR2CSC_EX2_NPU_WRAPPER_H_
#define CSR2CSC_EX2_NPU_WRAPPER_H_

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <vector>

#include "acl/acl.h"
#include "cann_ops_sparse.h"
#include "descriptor_manager.h"

using sparse_test::DeviceBuffer;

// ============================================================================
// Helper: Get byte size per element for a given ACL data type
// ============================================================================

inline size_t AclDataTypeSize(aclDataType dt) {
    switch (dt) {
        case ACL_INT8:    return 1;
        case ACL_FLOAT16: return 2;
        case ACL_BF16:    return 2;
        case ACL_FLOAT:   return 4;
        default:          return 4;
    }
}

// ============================================================================
// NPU Result structure for csr2csc_ex2
// ============================================================================

struct Csr2CscNpuResult {
    std::vector<int32_t> cscColPtr;   // size n+1
    std::vector<int32_t> cscRowInd;   // size nnz
    std::vector<uint8_t> cscVal;      // size nnz * valSize bytes (raw)
    int32_t              nnz;

    aclsparseStatus_t bufferSizeRet;
    aclsparseStatus_t computeRet;
};

// ============================================================================
// NPU wrapper: Full csr2csc_ex2 workflow
// Step 1: bufferSize  Step 2: csr2cscEx2  Step 3: D2H copy results
//
// Device memory managed via DeviceBuffer RAII.
// Handle managed via HandleManager RAII.
// ============================================================================

inline void CopyCsr2CscResultToHost(
    DeviceBuffer& dCscColPtr, DeviceBuffer& dCscRowInd, DeviceBuffer& dCscVal,
    int n, int nnz, size_t valSize, Csr2CscNpuResult& result)
{
    result.cscColPtr.resize(n + 1);
    dCscColPtr.copyToHost(result.cscColPtr.data(), (n + 1) * sizeof(int32_t));
    if (nnz > 0) {
        result.cscRowInd.resize(nnz);
        dCscRowInd.copyToHost(result.cscRowInd.data(), nnz * sizeof(int32_t));
        result.cscVal.resize(static_cast<size_t>(nnz) * valSize);
        dCscVal.copyToHost(result.cscVal.data(), static_cast<size_t>(nnz) * valSize);
    }
}

inline aclsparseStatus_t RunCsr2CscOnDevice(
    sparse_test::HandleManager& handle,
    int m, int n, int nnz,
    DeviceBuffer& dVal, DeviceBuffer& dRowPtr, DeviceBuffer& dColInd,
    DeviceBuffer& dCscVal, DeviceBuffer& dCscColPtr, DeviceBuffer& dCscRowInd,
    aclDataType valType, aclsparseAction_t copyValues, aclsparseIndexBase_t idxBase,
    Csr2CscNpuResult& result)
{
    size_t workspaceSize = 0;
    result.bufferSizeRet = aclsparseCsr2cscEx2_bufferSize(
        handle.get(), m, n, nnz,
        dVal.get(), reinterpret_cast<const int*>(dRowPtr.get()),
        reinterpret_cast<const int*>(dColInd.get()),
        dCscVal.get(), reinterpret_cast<int*>(dCscColPtr.get()),
        reinterpret_cast<int*>(dCscRowInd.get()),
        valType, copyValues, idxBase,
        ACL_SPARSE_CSR2CSC_ALG_DEFAULT,
        &workspaceSize);
    if (result.bufferSizeRet != ACL_SPARSE_STATUS_SUCCESS) {
        std::cerr << "[NPU] Csr2cscEx2_bufferSize failed: " << result.bufferSizeRet << std::endl;
        return result.bufferSizeRet;
    }
    if (workspaceSize == 0) workspaceSize = 1;
    auto dWorkspace = DeviceBuffer::alloc(workspaceSize);
    result.computeRet = aclsparseCsr2cscEx2(
        handle.get(), m, n, nnz,
        dVal.get(), reinterpret_cast<const int*>(dRowPtr.get()),
        reinterpret_cast<const int*>(dColInd.get()),
        dCscVal.get(), reinterpret_cast<int*>(dCscColPtr.get()),
        reinterpret_cast<int*>(dCscRowInd.get()),
        valType, copyValues, idxBase,
        ACL_SPARSE_CSR2CSC_ALG_DEFAULT,
        dWorkspace.get());
    if (result.computeRet != ACL_SPARSE_STATUS_SUCCESS) {
        std::cerr << "[NPU] Csr2cscEx2 failed: " << result.computeRet << std::endl;
        return result.computeRet;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

inline Csr2CscNpuResult Csr2CscNpu(
    sparse_test::HandleManager& handle,
    aclrtStream stream,
    int32_t m, int32_t n, int32_t nnz,
    const void* csrValHost,
    const int32_t* csrRowPtrHost,
    const int32_t* csrColIndHost,
    aclDataType valType,
    aclsparseAction_t copyValues,
    aclsparseIndexBase_t idxBase)
{
    using namespace sparse_test;
    Csr2CscNpuResult result{};
    result.nnz = nnz;
    size_t valSize = AclDataTypeSize(valType);
    auto dRowPtr = DeviceBuffer::copyFrom(csrRowPtrHost, (m + 1) * sizeof(int32_t));
    size_t colIndBytes = std::max(static_cast<size_t>(nnz) * sizeof(int32_t), size_t(1));
    size_t valBytes    = std::max(static_cast<size_t>(nnz) * valSize, size_t(1));
    auto dColInd = (nnz > 0)
        ? DeviceBuffer::copyFrom(csrColIndHost, static_cast<size_t>(nnz) * sizeof(int32_t))
        : DeviceBuffer::alloc(colIndBytes);
    auto dVal = (nnz > 0 && csrValHost != nullptr)
        ? DeviceBuffer::copyFrom(csrValHost, static_cast<size_t>(nnz) * valSize)
        : DeviceBuffer::alloc(valBytes);
    auto dCscColPtr = DeviceBuffer::alloc((n + 1) * sizeof(int32_t));
    auto dCscRowInd = DeviceBuffer::alloc(colIndBytes);
    auto dCscVal    = DeviceBuffer::alloc(valBytes);

    // SYMBOLIC 模式下用 magic 填充 cscVal，便于 test 检测算子是否误写 cscVal。
    // nnz == 0 时 cscVal 不会被读取（CopyCsr2CscResultToHost 跳过），无需 fill。
    if (copyValues == ACL_SPARSE_ACTION_SYMBOLIC && nnz > 0) {
        std::vector<uint8_t> magic(valBytes, 0xDE);
        aclrtMemcpy(dCscVal.get(), valBytes, magic.data(), valBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    }

    auto st = RunCsr2CscOnDevice(handle, m, n, nnz,
        dVal, dRowPtr, dColInd, dCscVal, dCscColPtr, dCscRowInd,
        valType, copyValues, idxBase, result);
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        return result;
    }

    if (aclrtSynchronizeStream(stream) != ACL_SUCCESS) {
        std::cerr << "[NPU] aclrtSynchronizeStream failed" << std::endl;
        result.computeRet = ACL_SPARSE_STATUS_EXECUTION_FAILED;
        return result;
    }
    CopyCsr2CscResultToHost(dCscColPtr, dCscRowInd, dCscVal, n, nnz, valSize, result);
    return result;
}

#endif  // CSR2CSC_EX2_NPU_WRAPPER_H_
