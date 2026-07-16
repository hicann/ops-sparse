/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TEST_GPSV_INTERLEAVED_BATCH_NPU_WRAPPER_H_
#define TEST_GPSV_INTERLEAVED_BATCH_NPU_WRAPPER_H_

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

#include "descriptor_manager.h"
#include "sparse_test.h"

namespace sparse_test {

/**
 * NPU wrapper for aclsparseSgpsvInterleavedBatch (Legacy API, no descriptors).
 *
 * Encapsulates:
 *  - H2D of ds/dl/d/du/dw/x (b input)
 *  - bufferSizeExt query
 *  - workspace allocation
 *  - aclsparseSgpsvInterleavedBatch call
 *  - stream sync + D2H of x (solution output)
 *
 * All device memory managed by DeviceBuffer RAII.
 */
inline std::vector<float> GpsvInterleavedBatchNpu(
    aclsparseHandle_t handle, aclrtStream stream, int algo, int m,
    const std::vector<float>& ds, const std::vector<float>& dl,
    const std::vector<float>& d, const std::vector<float>& du,
    const std::vector<float>& dw, const std::vector<float>& xIn,
    int batchCount)
{
    int total = m * batchCount;
    std::vector<float> xOut(total, 0.0f);
    if (m <= 0 || batchCount <= 0) return xOut;

    // 1. Host-to-Device copies (RAII manages device memory)
    auto dDs = DeviceBuffer::copyFrom(ds.data(), total * sizeof(float));
    auto dDl = DeviceBuffer::copyFrom(dl.data(), total * sizeof(float));
    auto dD  = DeviceBuffer::copyFrom(d.data(),  total * sizeof(float));
    auto dDu = DeviceBuffer::copyFrom(du.data(), total * sizeof(float));
    auto dDw = DeviceBuffer::copyFrom(dw.data(), total * sizeof(float));
    auto dX  = DeviceBuffer::copyFrom(xIn.data(), total * sizeof(float));  // b in, overwritten with solution

    // 2. Query workspace buffer size
    size_t bufferSize = 0;
    auto ret = aclsparseSgpsvInterleavedBatch_bufferSizeExt(
        handle, algo, m,
        static_cast<const float*>(dDs.get()), static_cast<const float*>(dDl.get()),
        static_cast<const float*>(dD.get()),  static_cast<const float*>(dDu.get()),
        static_cast<const float*>(dDw.get()), static_cast<const float*>(dX.get()),
        batchCount, &bufferSize);
    if (ret != ACL_SPARSE_STATUS_SUCCESS) {
        throw std::runtime_error(
            "aclsparseSgpsvInterleavedBatch_bufferSizeExt failed, status="
            + std::to_string(ret));
    }

    // 3. Allocate workspace + call solver + sync + D2H
    DeviceBuffer dBuffer;
    if (bufferSize > 0) dBuffer = DeviceBuffer::alloc(bufferSize);
    ret = aclsparseSgpsvInterleavedBatch(
        handle, algo, m,
        static_cast<float*>(dDs.get()), static_cast<float*>(dDl.get()),
        static_cast<float*>(dD.get()),  static_cast<float*>(dDu.get()),
        static_cast<float*>(dDw.get()), static_cast<float*>(dX.get()),
        batchCount, dBuffer.raw());
    if (ret != ACL_SPARSE_STATUS_SUCCESS) {
        throw std::runtime_error(
            "aclsparseSgpsvInterleavedBatch failed, status=" + std::to_string(ret));
    }

    auto aclRet = aclrtSynchronizeStream(stream);
    if (aclRet != ACL_SUCCESS) {
        throw std::runtime_error("aclrtSynchronizeStream failed, aclError=" + std::to_string(aclRet));
    }
    dX.copyToHost(xOut.data(), total * sizeof(float));
    return xOut;
}

}  // namespace sparse_test

#endif
