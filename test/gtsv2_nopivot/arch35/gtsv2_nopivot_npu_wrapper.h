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
#include <iostream>
#include <vector>

#include "acl/acl.h"
#include "cann_ops_sparse.h"
#include "descriptor_manager.h"

// ============================================================================
// NPU Workflow Result
// ============================================================================

struct Gtsv2NopivotNpuResult {
    std::vector<float> X;    // solution matrix (column-major, ldb x n)
    aclsparseStatus_t bufferSizeRet;
    aclsparseStatus_t computeRet;
};

// ============================================================================
// Helper: allocate device buffer with 128-byte alignment for pBuffer
// ============================================================================
struct AlignedDeviceBuffer {
    sparse_test::DeviceBuffer backing;
    void* alignedPtr = nullptr;

    AlignedDeviceBuffer() = default;
    AlignedDeviceBuffer(const AlignedDeviceBuffer&) = delete;
    AlignedDeviceBuffer& operator=(const AlignedDeviceBuffer&) = delete;
    AlignedDeviceBuffer(AlignedDeviceBuffer&&) = default;
    AlignedDeviceBuffer& operator=(AlignedDeviceBuffer&&) = default;

    static AlignedDeviceBuffer allocAligned(size_t size) {
        AlignedDeviceBuffer b;
        if (size == 0) size = 16;
        b.backing = sparse_test::DeviceBuffer::alloc(size + 127);
        uintptr_t addr = reinterpret_cast<uintptr_t>(b.backing.get());
        b.alignedPtr = reinterpret_cast<void*>((addr + 127) & ~127ULL);
        return b;
    }

    void* get() { return alignedPtr; }
};

// ============================================================================
// NPU wrapper: Query workspace size, then solve
// Step 1: aclsparseSgtsv2Nopivot_bufferSizeExt
// Step 2: aclsparseSgtsv2Nopivot (in-place, B is both input RHS and output X)
// ============================================================================

inline Gtsv2NopivotNpuResult Gtsv2NopivotNpu(
    sparse_test::HandleManager& handle, aclrtStream stream,
    const std::vector<float>& dl_host,
    const std::vector<float>& d_host,
    const std::vector<float>& du_host,
    const std::vector<float>& B_host,
    int m, int n, int ldb)
{
    using namespace sparse_test;
    Gtsv2NopivotNpuResult result{};

    // Validate inputs
    if (m < 3 || n < 1) {
        result.bufferSizeRet = ACL_SPARSE_STATUS_SUCCESS;  // early return, no error
        return result;
    }

    handle.setStream(stream);

    // Copy diagonals to device
    auto dDl = DeviceBuffer::copyFrom(dl_host.data(), m * sizeof(float));
    auto dD  = DeviceBuffer::copyFrom(d_host.data(), m * sizeof(float));
    auto dDu = DeviceBuffer::copyFrom(du_host.data(), m * sizeof(float));

    // Copy B (RHS) to device — B is modified in-place to become solution X
    int64_t B_size = static_cast<int64_t>(ldb) * n;
    auto dB = DeviceBuffer::copyFrom(B_host.data(), B_size * sizeof(float));

    // Step 1: Query workspace size
    // bufferSizeExt takes const float* for B (no modification)
    size_t bufferSize = 0;
    result.bufferSizeRet = aclsparseSgtsv2Nopivot_bufferSizeExt(
        handle.get(), m, n,
        reinterpret_cast<const float*>(dDl.raw()),
        reinterpret_cast<const float*>(dD.raw()),
        reinterpret_cast<const float*>(dDu.raw()),
        reinterpret_cast<const float*>(dB.raw()),
        ldb, &bufferSize);

    if (result.bufferSizeRet != ACL_SPARSE_STATUS_SUCCESS) {
        std::cerr << "[NPU] aclsparseSgtsv2Nopivot_bufferSizeExt failed: " << result.bufferSizeRet << std::endl;
        return result;
    }

    // Allocate workspace buffer with 128-byte alignment
    AlignedDeviceBuffer workspace = AlignedDeviceBuffer::allocAligned(bufferSize);

    // Step 2: Solve (in-place, B is overwritten with solution X)
    result.computeRet = aclsparseSgtsv2Nopivot(
        handle.get(), m, n,
        reinterpret_cast<const float*>(dDl.raw()),
        reinterpret_cast<const float*>(dD.raw()),
        reinterpret_cast<const float*>(dDu.raw()),
        reinterpret_cast<float*>(dB.get()),
        ldb, workspace.get());

    if (result.computeRet != ACL_SPARSE_STATUS_SUCCESS) {
        std::cerr << "[NPU] aclsparseSgtsv2Nopivot failed: " << result.computeRet << std::endl;
        return result;
    }

    // Synchronize stream after kernel launch
    if (aclrtSynchronizeStream(stream) != ACL_SUCCESS) {
        std::cerr << "[NPU] aclrtSynchronizeStream failed" << std::endl;
        result.computeRet = ACL_SPARSE_STATUS_EXECUTION_FAILED;
        return result;
    }

    // Copy solution back to host
    result.X.resize(B_size);
    dB.copyToHost(result.X.data(), B_size * sizeof(float));

    return result;
}
