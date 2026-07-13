/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TEST_GTSV_INTERLEAVED_BATCH_NPU_WRAPPER_H_
#define TEST_GTSV_INTERLEAVED_BATCH_NPU_WRAPPER_H_

#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "descriptor_manager.h"
#include "sparse_test.h"

namespace sparse_test {

/**
 * NPU wrapper for aclsparseSgtsvInterleavedBatch (Legacy API, no descriptors).
 *
 * Encapsulates:
 *  - H2D of dl/d/du/x (b input)
 *  - bufferSizeExt query
 *  - workspace allocation
 *  - aclsparseSgtsvInterleavedBatch call
 *  - stream sync
 *  - D2H of x (solution output)
 *
 * All device memory managed by DeviceBuffer RAII.
 */
inline std::vector<float> GtsvInterleavedBatchNpu(
    aclsparseHandle_t handle,
    aclrtStream stream,
    int algo,
    int m,
    const std::vector<float>& dl,
    const std::vector<float>& d,
    const std::vector<float>& du,
    const std::vector<float>& xIn,   // b (right-hand side)
    int batchCount)
{
    int total = m * batchCount;
    std::vector<float> xOut(total, 0.0f);
    if (m <= 0 || batchCount <= 0) return xOut;

    // 1. Host-to-Device copies (RAII manages device memory)
    auto dDl = DeviceBuffer::copyFrom(dl.data(),  total * sizeof(float));
    auto dD  = DeviceBuffer::copyFrom(d.data(),   total * sizeof(float));
    auto dDu = DeviceBuffer::copyFrom(du.data(),  total * sizeof(float));
    // Copy xIn (b) as initial right-hand side; will be overwritten with solution
    auto dX  = DeviceBuffer::copyFrom(xIn.data(), total * sizeof(float));

    // 2. Query workspace buffer size (pass device pointers to match API signature)
    size_t bufferSize = 0;
    auto ret = aclsparseSgtsvInterleavedBatch_bufferSizeExt(
        handle, algo, m,
        static_cast<const float*>(dDl.get()),
        static_cast<const float*>(dD.get()),
        static_cast<const float*>(dDu.get()),
        static_cast<const float*>(dX.get()),
        batchCount, &bufferSize);
    if (ret != ACL_SPARSE_STATUS_SUCCESS) {
        throw std::runtime_error(
            "aclsparseSgtsvInterleavedBatch_bufferSizeExt failed, status="
            + std::to_string(ret));
    }

    // 3. Allocate workspace (128-byte alignment by aclrtMalloc)
    DeviceBuffer dBuffer;
    if (bufferSize > 0) {
        dBuffer = DeviceBuffer::alloc(bufferSize);
    }

    // 4. Call the GTSV solver
    ret = aclsparseSgtsvInterleavedBatch(
        handle, algo, m,
        static_cast<float*>(dDl.get()),
        static_cast<float*>(dD.get()),
        static_cast<float*>(dDu.get()),
        static_cast<float*>(dX.get()),
        batchCount,
        dBuffer.raw());
    if (ret != ACL_SPARSE_STATUS_SUCCESS) {
        throw std::runtime_error(
            "aclsparseSgtsvInterleavedBatch failed, status="
            + std::to_string(ret));
    }

    // 5. Stream synchronization
    auto aclRet = aclrtSynchronizeStream(stream);
    if (aclRet != ACL_SUCCESS) {
        throw std::runtime_error(
            "aclrtSynchronizeStream failed, aclError="
            + std::to_string(aclRet));
    }

    // 6. Device-to-Host copy of the solution
    dX.copyToHost(xOut.data(), total * sizeof(float));

    return xOut;
}

/**
 * Variant that queries bufferSizeExt and returns workspace size (for testing).
 * Note: Passes dummy pointers for dl/d/du/x since the implementation ignores them.
 */
inline size_t GtsvInterleavedBatchBufferSizeExt(
    aclsparseHandle_t handle, int algo, int m, int batchCount)
{
    size_t bufferSize = 0;
    int total = m * batchCount;
    std::vector<float> dummy(total, 1.0f);
    auto dDummy = DeviceBuffer::copyFrom(dummy.data(), total * sizeof(float));

    auto ret = aclsparseSgtsvInterleavedBatch_bufferSizeExt(
        handle, algo, m,
        static_cast<const float*>(dDummy.get()),
        static_cast<const float*>(dDummy.get()),
        static_cast<const float*>(dDummy.get()),
        static_cast<const float*>(dDummy.get()),
        batchCount, &bufferSize);
    if (ret != ACL_SPARSE_STATUS_SUCCESS) {
        throw std::runtime_error(
            "aclsparseSgtsvInterleavedBatch_bufferSizeExt failed, status="
            + std::to_string(ret));
    }
    return bufferSize;
}

}  // namespace sparse_test

#endif
