/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TEST_GTSV2_STRIDED_BATCH_ARCH35_GTSV2_STRIDED_BATCH_NPU_WRAPPER_H_
#define TEST_GTSV2_STRIDED_BATCH_ARCH35_GTSV2_STRIDED_BATCH_NPU_WRAPPER_H_

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <vector>

#include "acl/acl.h"
#include "cann_ops_sparse.h"
#include "descriptor_manager.h"

// ============================================================================
// NPU wrapper for aclsparseSgtsv2StridedBatch (Legacy API, FP32)
//
// Workflow:
//   1. Copy dl/d/du/x from host to device (DeviceBuffer RAII).
//   2. Call aclsparseSgtsv2StridedBatch_bufferSizeExt to query workspace size.
//      - m <= 2048 (pure UB path): bufferSize == 0.
//      - m >  2048 (GM workspace path): bufferSize > 0.
//      When bufferSize == 0, pBuffer may be nullptr (pure UB path does not use
//      it). We allocate a minimal 128-byte placeholder for consistency.
//   3. Allocate pBuffer (aclrtMalloc guarantees 128B alignment).
//   4. Call aclsparseSgtsv2StridedBatch to launch the kernel (async).
//   5. aclrtSynchronizeStream to wait for completion.
//   6. Copy x back from device to host.
//   7. RAII releases all device memory automatically on scope exit.
//
// Every ACL call's return value is captured and surfaced in the result struct
// so callers can assert on each stage independently.
// ============================================================================

struct Sgtsv2StridedBatchNpuResult {
    std::vector<float>    x;                                              // D2H solution (in-place overwrite)
    aclsparseStatus_t     bufferSizeExtRet = ACL_SPARSE_STATUS_INTERNAL_ERROR;
    aclsparseStatus_t     computeRet       = ACL_SPARSE_STATUS_INTERNAL_ERROR;
    size_t                bufferSize       = 0;                           // workspace bytes
};

// Sync the stream and copy device result back to host.
inline Sgtsv2StridedBatchNpuResult SyncAndCopyResult(
    aclrtStream stream, sparse_test::DeviceBuffer& dX,
    size_t totalElems, size_t totalBytes,
    Sgtsv2StridedBatchNpuResult result)
{
    auto aclRet = aclrtSynchronizeStream(stream);
    if (aclRet != ACL_SUCCESS) {
        std::cerr << "[NPU] aclrtSynchronizeStream failed: " << aclRet << std::endl;
        result.computeRet = ACL_SPARSE_STATUS_INTERNAL_ERROR;
        return result;
    }
    result.x.resize(totalElems);
    dX.copyToHost(result.x.data(), totalBytes);
    return result;
}

// Query workspace size via bufferSizeExt, storing results in `result`.
inline size_t QueryWorkspace(
    sparse_test::HandleManager& handle,
    const float* pDl, const float* pD, const float* pDu, const float* pX,
    int m, int batchCount, int batchStride,
    Sgtsv2StridedBatchNpuResult& result)
{
    size_t bufferSize = 0;
    result.bufferSizeExtRet = aclsparseSgtsv2StridedBatch_bufferSizeExt(
        handle.get(), m, pDl, pD, pDu, pX,
        batchCount, batchStride, &bufferSize);
    result.bufferSize = bufferSize;
    if (result.bufferSizeExtRet != ACL_SPARSE_STATUS_SUCCESS) {
        std::cerr << "[NPU] bufferSizeExt failed: " << result.bufferSizeExtRet << std::endl;
    }
    return bufferSize;
}

// Launch the gtsv2StridedBatch kernel. pBuffer is passed by caller (RAII in outer scope).
inline aclsparseStatus_t LaunchGtsv2StridedBatch(
    sparse_test::HandleManager& handle,
    const float* pDl, const float* pD, const float* pDu, float* pX,
    int m, int batchCount, int batchStride, void* pBuffer)
{
    return aclsparseSgtsv2StridedBatch(
        handle.get(), m, pDl, pD, pDu, pX,
        batchCount, batchStride, pBuffer);
}

inline Sgtsv2StridedBatchNpuResult Sgtsv2StridedBatchNpu(
    sparse_test::HandleManager& handle,
    aclrtStream stream,
    const std::vector<float>& dl,
    const std::vector<float>& d,
    const std::vector<float>& du,
    const std::vector<float>& x,
    int m,
    int batchCount,
    int batchStride)
{
    using namespace sparse_test;
    Sgtsv2StridedBatchNpuResult result{};

    if (batchCount <= 0 || m <= 0 || batchStride < m) {
        result.bufferSizeExtRet = ACL_SPARSE_STATUS_SUCCESS;
        result.computeRet       = ACL_SPARSE_STATUS_SUCCESS;
        result.x.assign(x.begin(), x.end());
        return result;
    }

    size_t totalElems = static_cast<size_t>(batchCount) * static_cast<size_t>(batchStride);
    size_t totalBytes = totalElems * sizeof(float);

    DeviceBuffer dDl = DeviceBuffer::copyFrom(dl.data(), totalBytes);
    DeviceBuffer dD  = DeviceBuffer::copyFrom(d.data(),  totalBytes);
    DeviceBuffer dDu = DeviceBuffer::copyFrom(du.data(), totalBytes);
    DeviceBuffer dX  = DeviceBuffer::copyFrom(x.data(),  totalBytes);
    handle.setStream(stream);

    const float* pDl = reinterpret_cast<const float*>(dDl.raw());
    const float* pD  = reinterpret_cast<const float*>(dD.raw());
    const float* pDu = reinterpret_cast<const float*>(dDu.raw());
    float*       pX  = reinterpret_cast<float*>(dX.get());

    size_t bufferSize = QueryWorkspace(
        handle, pDl, pD, pDu, pX, m, batchCount, batchStride, result);
    if (result.bufferSizeExtRet != ACL_SPARSE_STATUS_SUCCESS) {
        return result;
    }

    // Allocate pBuffer with lifetime spanning the async kernel launch and sync.
    size_t allocSize = (bufferSize == 0) ? 128 : bufferSize;
    DeviceBuffer dBuffer = DeviceBuffer::alloc(allocSize);

    result.computeRet = LaunchGtsv2StridedBatch(
        handle, pDl, pD, pDu, pX, m, batchCount, batchStride, dBuffer.get());
    if (result.computeRet != ACL_SPARSE_STATUS_SUCCESS) {
        std::cerr << "[NPU] gtsv2StridedBatch failed: " << result.computeRet << std::endl;
        return result;
    }

    return SyncAndCopyResult(stream, dX, totalElems, totalBytes, result);
}

#endif  // TEST_GTSV2_STRIDED_BATCH_ARCH35_GTSV2_STRIDED_BATCH_NPU_WRAPPER_H_
