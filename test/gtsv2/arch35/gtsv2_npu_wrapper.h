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

struct Gtsv2NpuResult {
    std::vector<float> X;    // solution matrix (column-major, ldb x n)
    aclsparseStatus_t bufferSizeRet;
    aclsparseStatus_t computeRet;
};

// ============================================================================
// Helper: allocate device buffer with 128-byte alignment for pBuffer
//
// The framework's DeviceBuffer (descriptor_manager.h) does not support aligned
// allocation. This wrapper over-allocates and manually aligns to satisfy the
// 128-byte alignment requirement of aclsparseSgtsv2's pBuffer parameter.
// If the framework gains aligned allocation support in the future, this local
// wrapper should be removed in favor of the framework API.
// ============================================================================
constexpr size_t kAlignSize = 128;
constexpr size_t kAlignMask = kAlignSize - 1;
constexpr size_t kMinAllocSize = 16;  // minimum allocation to avoid zero-size malloc

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
        if (size == 0) size = kMinAllocSize;
        b.backing = sparse_test::DeviceBuffer::alloc(size + kAlignMask);
        uintptr_t addr = reinterpret_cast<uintptr_t>(b.backing.get());
        b.alignedPtr = reinterpret_cast<void*>((addr + kAlignMask) & ~static_cast<uintptr_t>(kAlignMask));
        return b;
    }

    void* get() { return alignedPtr; }
};

// ============================================================================
// Helper: Copy tridiagonal data and RHS to device, return device buffers
// ============================================================================
struct Gtsv2DeviceData {
    sparse_test::DeviceBuffer dDl;
    sparse_test::DeviceBuffer dD;
    sparse_test::DeviceBuffer dDu;
    sparse_test::DeviceBuffer dB;
    int64_t bSize = 0;
};

inline Gtsv2DeviceData PrepareDeviceData(
    const std::vector<float>& dl_host,
    const std::vector<float>& d_host,
    const std::vector<float>& du_host,
    const std::vector<float>& B_host,
    int m, int n, int ldb)
{
    using namespace sparse_test;
    Gtsv2DeviceData d;
    d.dDl = DeviceBuffer::copyFrom(dl_host.data(), m * sizeof(float));
    d.dD  = DeviceBuffer::copyFrom(d_host.data(), m * sizeof(float));
    d.dDu = DeviceBuffer::copyFrom(du_host.data(), m * sizeof(float));
    d.bSize = static_cast<int64_t>(ldb) * n;
    size_t bAllocSize = (d.bSize > 0) ? static_cast<size_t>(d.bSize) * sizeof(float) : 1;
    d.dB = (d.bSize > 0)
        ? DeviceBuffer::copyFrom(B_host.data(), bAllocSize)
        : DeviceBuffer::alloc(bAllocSize);
    return d;
}

// ============================================================================
// Helper: Solve on NPU and copy result back to host
// ============================================================================
inline void SolveAndCopyBack(
    sparse_test::HandleManager& handle, aclrtStream stream,
    Gtsv2DeviceData& d, size_t bufferSize,
    Gtsv2NpuResult& result, int m, int n, int ldb)
{
    AlignedDeviceBuffer workspace = AlignedDeviceBuffer::allocAligned(bufferSize);

    result.computeRet = aclsparseSgtsv2(
        handle.get(), m, n,
        reinterpret_cast<const float*>(d.dDl.raw()),
        reinterpret_cast<const float*>(d.dD.raw()),
        reinterpret_cast<const float*>(d.dDu.raw()),
        reinterpret_cast<float*>(d.dB.get()),
        ldb, workspace.get());

    if (result.computeRet != ACL_SPARSE_STATUS_SUCCESS) {
        std::cerr << "[NPU] aclsparseSgtsv2 failed: " << result.computeRet << std::endl;
        return;
    }

    if (aclrtSynchronizeStream(stream) != ACL_SUCCESS) {
        std::cerr << "[NPU] aclrtSynchronizeStream failed" << std::endl;
        result.computeRet = ACL_SPARSE_STATUS_EXECUTION_FAILED;
        return;
    }

    result.X.resize(d.bSize);
    d.dB.copyToHost(result.X.data(), d.bSize * sizeof(float));
}

// ============================================================================
// NPU wrapper: Query workspace size, then solve
// Step 1: aclsparseSgtsv2_bufferSizeExt
// Step 2: aclsparseSgtsv2 (in-place, B is both input RHS and output X)
// ============================================================================

inline Gtsv2NpuResult Gtsv2Npu(
    sparse_test::HandleManager& handle, aclrtStream stream,
    const std::vector<float>& dl_host,
    const std::vector<float>& d_host,
    const std::vector<float>& du_host,
    const std::vector<float>& B_host,
    int m, int n, int ldb)
{
    using namespace sparse_test;
    Gtsv2NpuResult result{};

    // Validate inputs: only skip for m < 3 (invalid system size).
    // n=0 is a valid API path (early return SUCCESS) — still call bufferSizeExt
    // to exercise the n=0 code path in the operator itself.
    if (m < 3) {
        result.bufferSizeRet = ACL_SPARSE_STATUS_SUCCESS;
        return result;
    }

    handle.setStream(stream);

    Gtsv2DeviceData d = PrepareDeviceData(dl_host, d_host, du_host, B_host, m, n, ldb);

    // Step 1: Query workspace size
    size_t bufferSize = 0;
    result.bufferSizeRet = aclsparseSgtsv2_bufferSizeExt(
        handle.get(), m, n,
        reinterpret_cast<const float*>(d.dDl.raw()),
        reinterpret_cast<const float*>(d.dD.raw()),
        reinterpret_cast<const float*>(d.dDu.raw()),
        reinterpret_cast<const float*>(d.dB.raw()),
        ldb, &bufferSize);

    if (result.bufferSizeRet != ACL_SPARSE_STATUS_SUCCESS) {
        std::cerr << "[NPU] aclsparseSgtsv2_bufferSizeExt failed: " << result.bufferSizeRet << std::endl;
        return result;
    }

    // n=0: API returns SUCCESS with bufferSize=0, no compute needed
    if (n == 0) {
        return result;
    }

    // Step 2: Solve and copy back
    SolveAndCopyBack(handle, stream, d, bufferSize, result, m, n, ldb);
    return result;
}
