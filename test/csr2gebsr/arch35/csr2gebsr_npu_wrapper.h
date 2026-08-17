/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root directory of the software repository for the full text of the License.
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

#include "acl/acl.h"
#include "cann_ops_sparse.h"
#include "descriptor_manager.h"
#include "dtype_utils.h"  // convertFloatToDtypeBytes, convertDtypeBytesToFloat

/// 安全向上取整（防御除零）
static inline int SafeCeilDivMb(int m, int rowBlockDim) {
    return (rowBlockDim > 0 && m > 0) ? (m + rowBlockDim - 1) / rowBlockDim : 0;
}

// ============================================================================
// RAII MatDescr wrapper for Legacy API
// ============================================================================

class MatDescrGuard {
public:
    MatDescrGuard() {
        auto s = aclsparseCreateMatDescr(&descr_);
        if (s != ACL_SPARSE_STATUS_SUCCESS) {
            throw std::runtime_error("aclsparseCreateMatDescr failed");
        }
        aclsparseSetMatType(descr_, ACL_SPARSE_MATRIX_TYPE_GENERAL);
        aclsparseSetMatIndexBase(descr_, ACL_SPARSE_INDEX_BASE_ZERO);
    }

    void setIndexBase(int base) {
        aclsparseSetMatIndexBase(descr_,
            (base == 1) ? ACL_SPARSE_INDEX_BASE_ONE : ACL_SPARSE_INDEX_BASE_ZERO);
    }

    aclsparseMatDescr_t get() { return descr_; }
    aclsparseMatDescr_t cget() const { return descr_; }

    ~MatDescrGuard() {
        if (descr_) aclsparseDestroyMatDescr(descr_);
    }

    MatDescrGuard(const MatDescrGuard&) = delete;
    MatDescrGuard& operator=(const MatDescrGuard&) = delete;
    MatDescrGuard(MatDescrGuard&& other) noexcept : descr_(other.descr_) {
        other.descr_ = nullptr;
    }

private:
    aclsparseMatDescr_t descr_ = nullptr;
};

// ============================================================================
// NPU Workflow Result
// ============================================================================

struct Csr2GebsrNpuResult {
    std::vector<int32_t> bsrRowPtr;   // size: mb + 1
    std::vector<int32_t> bsrColInd;   // size: nnzb
    std::vector<float>   bsrVal;      // size: nnzb * rowBlockDim * colBlockDim (as float)
    int32_t              nnzb;

    // Expose bufferSize value (not only the status) so the test can
    // assert it equals the expected formula (mb + 1 + mb*nb) * sizeof(int32_t).
    size_t               bufferSize = 0;

    aclsparseStatus_t bufferSizeRet;
    aclsparseStatus_t nnzRet;
    aclsparseStatus_t convertRet;
};

// ============================================================================
// Dtype-specific bufferSize dispatch
// ============================================================================

inline aclsparseStatus_t CallBufferSize(
    aclsparseHandle_t handle, aclsparseDirection_t dir,
    int m, int n,
    aclsparseMatDescr_t descrA,
    const void* csrValBytes, const int* csrRowPtrA, const int* csrColIndA,
    int rowBlockDim, int colBlockDim,
    size_t* pBufferSize,
    const std::string& dtype)
{
    if (dtype == "FP32") {
        return aclsparseScsr2gebsr_bufferSize(handle, dir, m, n, descrA,
            reinterpret_cast<const float*>(csrValBytes), csrRowPtrA, csrColIndA,
            rowBlockDim, colBlockDim, pBufferSize);
    } else if (dtype == "FP16") {
        return aclsparseHcsr2gebsr_bufferSize(handle, dir, m, n, descrA,
            csrValBytes, csrRowPtrA, csrColIndA,
            rowBlockDim, colBlockDim, pBufferSize);
    } else if (dtype == "BF16") {
        return aclsparseBhcsr2gebsr_bufferSize(handle, dir, m, n, descrA,
            csrValBytes, csrRowPtrA, csrColIndA,
            rowBlockDim, colBlockDim, pBufferSize);
    } else if (dtype == "INT32") {
        return aclsparseIcsr2gebsr_bufferSize(handle, dir, m, n, descrA,
            reinterpret_cast<const int*>(csrValBytes), csrRowPtrA, csrColIndA,
            rowBlockDim, colBlockDim, pBufferSize);
    }
    return ACL_SPARSE_STATUS_INVALID_VALUE;
}

// ============================================================================
// Dtype-specific convert dispatch
// ============================================================================

inline aclsparseStatus_t CallConvert(
    aclsparseHandle_t handle, aclsparseDirection_t dir,
    int m, int n,
    aclsparseMatDescr_t descrA,
    const void* csrValBytes, const int* csrRowPtrA, const int* csrColIndA,
    aclsparseMatDescr_t descrC,
    void* bsrValBytes, int* bsrRowPtrC, int* bsrColIndC,
    int rowBlockDim, int colBlockDim,
    void* pBuffer,
    const std::string& dtype)
{
    if (dtype == "FP32") {
        return aclsparseScsr2gebsr(handle, dir, m, n, descrA,
            reinterpret_cast<const float*>(csrValBytes), csrRowPtrA, csrColIndA,
            descrC,
            reinterpret_cast<float*>(bsrValBytes), bsrRowPtrC, bsrColIndC,
            rowBlockDim, colBlockDim, pBuffer);
    } else if (dtype == "FP16") {
        return aclsparseHcsr2gebsr(handle, dir, m, n, descrA,
            csrValBytes, csrRowPtrA, csrColIndA,
            descrC,
            bsrValBytes, bsrRowPtrC, bsrColIndC,
            rowBlockDim, colBlockDim, pBuffer);
    } else if (dtype == "BF16") {
        return aclsparseBhcsr2gebsr(handle, dir, m, n, descrA,
            csrValBytes, csrRowPtrA, csrColIndA,
            descrC,
            bsrValBytes, bsrRowPtrC, bsrColIndC,
            rowBlockDim, colBlockDim, pBuffer);
    } else if (dtype == "INT32") {
        return aclsparseIcsr2gebsr(handle, dir, m, n, descrA,
            reinterpret_cast<const int*>(csrValBytes), csrRowPtrA, csrColIndA,
            descrC,
            reinterpret_cast<int*>(bsrValBytes), bsrRowPtrC, bsrColIndC,
            rowBlockDim, colBlockDim, pBuffer);
    }
    return ACL_SPARSE_STATUS_INVALID_VALUE;
}

// ============================================================================
// Helper: Get value size in bytes for a given dtype
// ============================================================================

inline size_t GetValueSize(const std::string& dtype) {
    if (dtype == "FP32") return sizeof(float);
    if (dtype == "FP16") return sizeof(uint16_t);
    if (dtype == "BF16") return sizeof(uint16_t);
    if (dtype == "INT32") return sizeof(int32_t);
    return sizeof(float);
}

// ============================================================================
// Internal helpers for Csr2GebsrNpu (split into focused sub-steps)
// ============================================================================

namespace csr2gebsr_npu_detail {

// Device buffers holding CSR input data
struct CsrDeviceBuffers {
    sparse_test::DeviceBuffer dRowPtrA;
    sparse_test::DeviceBuffer dColIndA;
    sparse_test::DeviceBuffer dValA;
    std::vector<uint8_t> csrValBytes;  // dtype-converted value bytes (host-side)
};

// Copy CSR host data to device, converting values to the target dtype.
inline CsrDeviceBuffers PrepareCsrDeviceBuffers(
    const std::vector<int32_t>& csrRowPtrA_host,
    const std::vector<int32_t>& csrColIndA_host,
    const std::vector<float>& csrValues_host,
    int m, int nnz, const std::string& dtype)
{
    using namespace sparse_test;
    CsrDeviceBuffers buf;
    buf.csrValBytes = convertFloatToDtypeBytes(csrValues_host, dtype);

    buf.dRowPtrA = DeviceBuffer::copyFrom(
        csrRowPtrA_host.data(), (m + 1) * sizeof(int32_t));
    buf.dColIndA = (nnz > 0)
        ? DeviceBuffer::copyFrom(csrColIndA_host.data(), nnz * sizeof(int32_t))
        : DeviceBuffer();
    buf.dValA = (nnz > 0)
        ? DeviceBuffer::copyFrom(buf.csrValBytes.data(), buf.csrValBytes.size())
        : DeviceBuffer();
    return buf;
}

// Step 1: Query workspace buffer size and allocate workspace + bsrRowPtr.
struct BufferSizeResult {
    aclsparseStatus_t status;
    size_t bufferSize;
    sparse_test::DeviceBuffer dWorkspace;
    sparse_test::DeviceBuffer dBsrRowPtr;
};

inline BufferSizeResult RunBufferSizeStep(
    sparse_test::HandleManager& handle, aclsparseDirection_t dir,
    int m, int n, int nnz,
    aclsparseMatDescr_t descrA,
    const CsrDeviceBuffers& csrBuf,
    int rowBlockDim, int colBlockDim,
    const std::string& dtype)
{
    using namespace sparse_test;
    BufferSizeResult r{};
    r.bufferSize = 0;

    r.status = CallBufferSize(
        handle.get(), dir, m, n, descrA,
        (nnz > 0) ? csrBuf.dValA.raw() : nullptr,
        reinterpret_cast<const int*>(csrBuf.dRowPtrA.raw()),
        (nnz > 0) ? reinterpret_cast<const int*>(csrBuf.dColIndA.raw()) : nullptr,
        rowBlockDim, colBlockDim, &r.bufferSize, dtype);

    if (r.status == ACL_SPARSE_STATUS_SUCCESS) {
        if (r.bufferSize > 0) {
            r.dWorkspace = DeviceBuffer::alloc(r.bufferSize);
        }
        int mb = SafeCeilDivMb(m, rowBlockDim);
        r.dBsrRowPtr = DeviceBuffer::alloc((mb + 1) * sizeof(int32_t));
    }
    return r;
}

// Step 2: Compute bsrRowPtrC and nnzb via Nnz kernel.
// nnzTotalDevHostPtr is caller-provided. In HOST pointer mode it points
// to a host int (operator D2H-copies nnzb); in DEVICE mode it points to a
// device int (operator D2D-copies nnzb, exercising LaunchPrefixSumAndCopyback
// and FillEmptyBsrRowPtrC DEVICE branches). The caller reads back nnzb from
// the device buffer when useDevicePointerMode is true.
struct NnzResult {
    aclsparseStatus_t status;
};

inline NnzResult RunNnzStep(
    sparse_test::HandleManager& handle, aclsparseDirection_t dir,
    int m, int n, int nnz,
    aclsparseMatDescr_t descrA, aclsparseMatDescr_t descrC,
    const CsrDeviceBuffers& csrBuf,
    sparse_test::DeviceBuffer& dBsrRowPtr,
    sparse_test::DeviceBuffer& dWorkspace,
    size_t bufferSize,
    int rowBlockDim, int colBlockDim,
    int* nnzTotalDevHostPtr,
    aclrtStream stream)
{
    NnzResult r{};

    r.status = aclsparseXcsr2gebsrNnz(
        handle.get(), dir, m, n, descrA,
        reinterpret_cast<const int*>(csrBuf.dRowPtrA.raw()),
        (nnz > 0) ? reinterpret_cast<const int*>(csrBuf.dColIndA.raw()) : nullptr,
        descrC,
        reinterpret_cast<int*>(dBsrRowPtr.get()),
        rowBlockDim, colBlockDim,
        nnzTotalDevHostPtr,
        (bufferSize > 0) ? dWorkspace.get() : nullptr);

    if (r.status == ACL_SPARSE_STATUS_SUCCESS) {
        // NPU test wrapper: sync stream to ensure Nnz kernel completes before
        // reading nnzb. (Test wrapper sync for readback; operator host.cpp has
        // its own sync rules.)
        aclrtSynchronizeStream(stream);
    }
    return r;
}

// Step 3: Execute CSR -> GEBSR conversion.
inline aclsparseStatus_t RunConvertStep(
    sparse_test::HandleManager& handle, aclsparseDirection_t dir,
    int m, int n, int nnz,
    aclsparseMatDescr_t descrA, aclsparseMatDescr_t descrC,
    const CsrDeviceBuffers& csrBuf,
    sparse_test::DeviceBuffer& dBsrVal,
    sparse_test::DeviceBuffer& dBsrRowPtr,
    sparse_test::DeviceBuffer& dBsrColInd,
    sparse_test::DeviceBuffer& dWorkspace,
    size_t bufferSize,
    int rowBlockDim, int colBlockDim,
    const std::string& dtype)
{
    return CallConvert(
        handle.get(), dir, m, n, descrA,
        (nnz > 0) ? csrBuf.dValA.raw() : nullptr,
        reinterpret_cast<const int*>(csrBuf.dRowPtrA.raw()),
        (nnz > 0) ? reinterpret_cast<const int*>(csrBuf.dColIndA.raw()) : nullptr,
        descrC,
        dBsrVal.get(),
        reinterpret_cast<int*>(dBsrRowPtr.get()),
        reinterpret_cast<int*>(dBsrColInd.get()),
        rowBlockDim, colBlockDim,
        (bufferSize > 0) ? dWorkspace.get() : nullptr,
        dtype);
}

// Read back device buffers into the result struct.
inline void ReadbackResults(
    Csr2GebsrNpuResult& result,
    sparse_test::DeviceBuffer& dBsrRowPtr,
    sparse_test::DeviceBuffer& dBsrColInd,
    sparse_test::DeviceBuffer& dBsrVal,
    int mb, int blockSize, size_t valSize,
    const std::string& dtype)
{
    using namespace sparse_test;

    // Read back bsrRowPtrC (may have been updated by step 3)
    dBsrRowPtr.copyToHost(result.bsrRowPtr.data(), (mb + 1) * sizeof(int32_t));

    // Read back bsrColIndC
    result.bsrColInd.resize(result.nnzb);
    dBsrColInd.copyToHost(result.bsrColInd.data(), result.nnzb * sizeof(int32_t));

    // Read back bsrValC (convert from target dtype to float)
    size_t totalValBytes = static_cast<size_t>(result.nnzb) * blockSize * valSize;
    std::vector<uint8_t> valBytesHost(totalValBytes);
    dBsrVal.copyToHost(valBytesHost.data(), totalValBytes);

    size_t totalValCount = static_cast<size_t>(result.nnzb) * blockSize;
    result.bsrVal = convertDtypeBytesToFloat(valBytesHost.data(), totalValCount, dtype);
}

}  // namespace csr2gebsr_npu_detail

// ============================================================================
// NPU wrapper: Full csr2gebsr three-step workflow
// ============================================================================

inline void RunConvertAndReadback(
    sparse_test::HandleManager& handle, aclsparseDirection_t dir,
    int m, int n, int nnz, int rowBlockDim, int colBlockDim,
    const MatDescrGuard& descrA, const MatDescrGuard& descrC,
    const csr2gebsr_npu_detail::CsrDeviceBuffers& csrBuf,
    sparse_test::DeviceBuffer& dBsrRowPtr, sparse_test::DeviceBuffer& dWorkspace,
    size_t bufferSize, size_t valSize, int mb,
    const std::string& dtype, aclrtStream stream,
    Csr2GebsrNpuResult& result)
{
    using namespace sparse_test;
    int blockSize = rowBlockDim * colBlockDim;
    auto dBsrColInd = DeviceBuffer::alloc(result.nnzb * sizeof(int32_t));
    auto dBsrVal = DeviceBuffer::alloc(
        static_cast<size_t>(result.nnzb) * blockSize * valSize);

    result.convertRet = csr2gebsr_npu_detail::RunConvertStep(
        handle, dir, m, n, nnz, descrA.cget(), descrC.cget(),
        csrBuf, dBsrVal, dBsrRowPtr, dBsrColInd,
        dWorkspace, bufferSize, rowBlockDim, colBlockDim, dtype);
    if (result.convertRet != ACL_SPARSE_STATUS_SUCCESS) { return; }

    aclrtSynchronizeStream(stream);
    csr2gebsr_npu_detail::ReadbackResults(
        result, dBsrRowPtr, dBsrColInd, dBsrVal,
        mb, blockSize, valSize, dtype);
}

inline Csr2GebsrNpuResult Csr2GebsrNpu(
    sparse_test::HandleManager& handle,
    aclrtStream stream,
    const std::vector<int32_t>& csrRowPtrA_host,
    const std::vector<int32_t>& csrColIndA_host,
    const std::vector<float>& csrValues_host,
    int m, int n, int nnz,
    int rowBlockDim, int colBlockDim,
    aclsparseDirection_t dir,
    int indexBaseA, int indexBaseC,
    const std::string& dtype,
    bool useDevicePointerMode = false)
{
    using namespace sparse_test;
    Csr2GebsrNpuResult result{};
    result.nnzb = 0;

    int mb = SafeCeilDivMb(m, rowBlockDim);

    aclsparseStatus_t pmRet = aclsparseSetPointerMode(handle.get(),
        useDevicePointerMode ? ACL_SPARSE_POINTER_MODE_DEVICE : ACL_SPARSE_POINTER_MODE_HOST);
    if (pmRet != ACL_SPARSE_STATUS_SUCCESS) {
        result.bufferSizeRet = pmRet;
        return result;
    }

    auto csrBuf = csr2gebsr_npu_detail::PrepareCsrDeviceBuffers(
        csrRowPtrA_host, csrColIndA_host, csrValues_host, m, nnz, dtype);
    size_t valSize = GetValueSize(dtype);

    MatDescrGuard descrA_guard;
    descrA_guard.setIndexBase(indexBaseA);
    MatDescrGuard descrC_guard;
    descrC_guard.setIndexBase(indexBaseC);

    auto bufResult = csr2gebsr_npu_detail::RunBufferSizeStep(
        handle, dir, m, n, nnz, descrA_guard.cget(),
        csrBuf, rowBlockDim, colBlockDim, dtype);
    result.bufferSizeRet = bufResult.status;
    result.bufferSize = bufResult.bufferSize;
    if (result.bufferSizeRet != ACL_SPARSE_STATUS_SUCCESS) { return result; }

    DeviceBuffer dNnzb;
    int* nnzPtr = useDevicePointerMode
        ? (dNnzb = DeviceBuffer::alloc(sizeof(int32_t)), reinterpret_cast<int*>(dNnzb.get()))
        : &result.nnzb;

    auto nnzResult = csr2gebsr_npu_detail::RunNnzStep(
        handle, dir, m, n, nnz, descrA_guard.cget(), descrC_guard.cget(),
        csrBuf, bufResult.dBsrRowPtr, bufResult.dWorkspace,
        bufResult.bufferSize, rowBlockDim, colBlockDim, nnzPtr, stream);
    result.nnzRet = nnzResult.status;
    if (result.nnzRet != ACL_SPARSE_STATUS_SUCCESS) { return result; }

    if (useDevicePointerMode) { dNnzb.copyToHost(&result.nnzb, sizeof(int32_t)); }

    result.bsrRowPtr.resize(mb + 1);
    bufResult.dBsrRowPtr.copyToHost(result.bsrRowPtr.data(), (mb + 1) * sizeof(int32_t));

    if (result.nnzb <= 0) { result.convertRet = ACL_SPARSE_STATUS_SUCCESS; return result; }

    RunConvertAndReadback(handle, dir, m, n, nnz, rowBlockDim, colBlockDim,
        descrA_guard, descrC_guard, csrBuf, bufResult.dBsrRowPtr,
        bufResult.dWorkspace, bufResult.bufferSize, valSize, mb, dtype, stream, result);
    return result;
}
