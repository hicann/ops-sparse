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

struct CsrGeam2NpuResult {
    std::vector<int32_t> rowPtrC;   // size m+1
    std::vector<int32_t> colIndC;   // size nnzC
    std::vector<float>   valuesC;   // size nnzC
    int32_t              nnzC;      // total nonzero count

    // Individual API return codes for debugging
    aclsparseStatus_t bufferSizeRet;
    aclsparseStatus_t nnzRet;
    aclsparseStatus_t computeRet;
};

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
// Helper: Device-side CSR data buffers
// ============================================================================

struct DeviceCsrBuffers {
    sparse_test::DeviceBuffer rowPtr;
    sparse_test::DeviceBuffer colInd;
    sparse_test::DeviceBuffer val;
};

inline DeviceCsrBuffers CopyCsrToDevice(
    sparse_test::HandleManager &handle,
    const std::vector<int32_t> &rowPtr_host,
    const std::vector<int32_t> &colInd_host,
    const std::vector<float> &values_host,
    int m, int nnz)
{
    (void)handle;
    DeviceCsrBuffers bufs;
    bufs.rowPtr = sparse_test::DeviceBuffer::copyFrom(
        rowPtr_host.data(), (m + 1) * sizeof(int32_t));
    if (nnz > 0) {
        bufs.colInd = sparse_test::DeviceBuffer::copyFrom(
            colInd_host.data(), nnz * sizeof(int32_t));
        bufs.val = sparse_test::DeviceBuffer::copyFrom(
            values_host.data(), nnz * sizeof(float));
    }
    return bufs;
}

// ============================================================================
// Helper: Prepare scalar pointers (host or device)
// ============================================================================

struct ScalarPtrs {
    float alphaVal;                    // host scalar value
    float betaVal;                     // host scalar value
    const float *alphaPtr;
    const float *betaPtr;
    sparse_test::DeviceBuffer dAlpha;  // valid only in device mode
    sparse_test::DeviceBuffer dBeta;   // valid only in device mode
};

inline ScalarPtrs PrepareScalars(float alpha, float beta, bool useHostMode) {
    ScalarPtrs s{};
    s.alphaVal = alpha;
    s.betaVal = beta;
    if (useHostMode) {
        s.alphaPtr = &s.alphaVal;
        s.betaPtr = &s.betaVal;
    } else {
        s.dAlpha = sparse_test::DeviceBuffer::copyFrom(&alpha, sizeof(float));
        s.dBeta = sparse_test::DeviceBuffer::copyFrom(&beta, sizeof(float));
        s.alphaPtr = reinterpret_cast<const float*>(s.dAlpha.raw());
        s.betaPtr = reinterpret_cast<const float*>(s.dBeta.raw());
    }
    return s;
}

// ============================================================================
// Helper: Prepare nnzC pointer (host or device)
// ============================================================================

struct NnzCPtrHolder {
    int *ptr = nullptr;
    sparse_test::DeviceBuffer dNnzC;  // valid only in device mode
};

inline NnzCPtrHolder PrepareNnzCPtr(bool useHostMode, int *hostNnzC) {
    NnzCPtrHolder h;
    if (useHostMode) {
        h.ptr = hostNnzC;
    } else {
        h.dNnzC = sparse_test::DeviceBuffer::alloc(sizeof(int32_t));
        h.ptr = reinterpret_cast<int*>(h.dNnzC.get());
    }
    return h;
}

// ============================================================================
// Step 1: Query workspace size
// ============================================================================

inline aclsparseStatus_t QueryCsrGeam2BufferSize(
    sparse_test::HandleManager &handle, int m, int n,
    const float *alphaPtr, const float *betaPtr,
    aclsparseMatDescr_t descrA, int nnzA,
    const float *dValA, const int *dRowPtrA, const int *dColIndA,
    aclsparseMatDescr_t descrB, int nnzB,
    const float *dValB, const int *dRowPtrB, const int *dColIndB,
    aclsparseMatDescr_t descrC,
    size_t *workspaceSize)
{
    return aclsparseScsrgeam2_bufferSizeExt(
        handle.get(), m, n, alphaPtr,
        descrA, nnzA, dValA, dRowPtrA, dColIndA,
        betaPtr,
        descrB, nnzB, dValB, dRowPtrB, dColIndB,
        descrC, nullptr, nullptr, nullptr, workspaceSize);
}

// ============================================================================
// Step 2: Compute rowPtrC and nnzC
// ============================================================================

inline aclsparseStatus_t ComputeGeam2Nnz(
    sparse_test::HandleManager &handle, int m, int n,
    aclsparseMatDescr_t descrA, int nnzA,
    const int *dRowPtrA, const int *dColIndA,
    aclsparseMatDescr_t descrB, int nnzB,
    const int *dRowPtrB, const int *dColIndB,
    aclsparseMatDescr_t descrC,
    int *dRowPtrC, int *nnzCPtr,
    void *workspace)
{
    return aclsparseXcsrgeam2Nnz(
        handle.get(), m, n,
        descrA, nnzA, dRowPtrA, dColIndA,
        descrB, nnzB, dRowPtrB, dColIndB,
        descrC,
        dRowPtrC, nnzCPtr, workspace);
}

// ============================================================================
// Step 3: Compute colIndC and valuesC
// ============================================================================

inline aclsparseStatus_t ComputeGeam2Values(
    sparse_test::HandleManager &handle, int m, int n,
    const float *alphaPtr, const float *betaPtr,
    aclsparseMatDescr_t descrA, int nnzA,
    const float *dValA, const int *dRowPtrA, const int *dColIndA,
    aclsparseMatDescr_t descrB, int nnzB,
    const float *dValB, const int *dRowPtrB, const int *dColIndB,
    aclsparseMatDescr_t descrC,
    float *dValC, int *dRowPtrC, int *dColIndC,
    void *workspace)
{
    return aclsparseScsrgeam2(
        handle.get(), m, n, alphaPtr,
        descrA, nnzA, dValA, dRowPtrA, dColIndA,
        betaPtr,
        descrB, nnzB, dValB, dRowPtrB, dColIndB,
        descrC, dValC, dRowPtrC, dColIndC, workspace);
}

// ============================================================================
// Helper: Prepared inputs for csrgeam2 NPU operation
// ============================================================================

struct Geam2PreparedInputs {
    DeviceCsrBuffers bufsA;
    DeviceCsrBuffers bufsB;
    MatDescrGuard descrA;
    MatDescrGuard descrB;
    MatDescrGuard descrC;
};

// Populate prepared inputs: device buffers and MatDescr
inline void PrepareGeam2Inputs(
    sparse_test::HandleManager &handle,
    const std::vector<int32_t> &rowPtrA_host,
    const std::vector<int32_t> &colIndA_host,
    const std::vector<float> &valuesA_host, int nnzA,
    const std::vector<int32_t> &rowPtrB_host,
    const std::vector<int32_t> &colIndB_host,
    const std::vector<float> &valuesB_host, int nnzB,
    int m, float alpha, float beta,
    int indexBaseA, int indexBaseB, int indexBaseC,
    Geam2PreparedInputs &inputs)
{
    using namespace sparse_test;
    inputs.bufsA = CopyCsrToDevice(handle, rowPtrA_host, colIndA_host, valuesA_host, m, nnzA);
    inputs.bufsB = CopyCsrToDevice(handle, rowPtrB_host, colIndB_host, valuesB_host, m, nnzB);
    inputs.descrA.setIndexBase(indexBaseA);
    inputs.descrB.setIndexBase(indexBaseB);
    inputs.descrC.setIndexBase(indexBaseC);
}

// ============================================================================
// Helper: Query bufferSize and allocate workspace + rowPtrC
// ============================================================================

struct Geam2Workspace {
    sparse_test::DeviceBuffer dWorkspace;
    sparse_test::DeviceBuffer dRowPtrC;
    NnzCPtrHolder nnzCHolder;
};

inline aclsparseStatus_t AllocateGeam2Workspace(
    sparse_test::HandleManager &handle,
    int m, int n, int nnzA, int nnzB,
    const Geam2PreparedInputs &inputs,
    const ScalarPtrs &scalars,
    Geam2Workspace &ws, size_t &workspaceSize)
{
    using namespace sparse_test;
    const float *valA = reinterpret_cast<const float*>(inputs.bufsA.val.raw());
    const int *rowA = reinterpret_cast<const int*>(inputs.bufsA.rowPtr.raw());
    const int *colA = reinterpret_cast<const int*>(inputs.bufsA.colInd.raw());
    const float *valB = reinterpret_cast<const float*>(inputs.bufsB.val.raw());
    const int *rowB = reinterpret_cast<const int*>(inputs.bufsB.rowPtr.raw());
    const int *colB = reinterpret_cast<const int*>(inputs.bufsB.colInd.raw());
    aclsparseStatus_t ret = QueryCsrGeam2BufferSize(
        handle, m, n, scalars.alphaPtr, scalars.betaPtr,
        inputs.descrA.cget(), nnzA, valA, rowA, colA,
        inputs.descrB.cget(), nnzB, valB, rowB, colB,
        inputs.descrC.cget(), &workspaceSize);
    if (ret != ACL_SPARSE_STATUS_SUCCESS) {
        return ret;
    }
    // Defensive: host guarantees (m+1)*sizeof(int32_t) >= 4, so workspaceSize>0
    // in practice. Guard retained against future host-side changes (#29).
    if (workspaceSize == 0) {
        workspaceSize = 16;
    }
    ws.dWorkspace = DeviceBuffer::alloc(workspaceSize);
    ws.dRowPtrC = DeviceBuffer::alloc((m + 1) * sizeof(int32_t));
    return ACL_SPARSE_STATUS_SUCCESS;
}

// ============================================================================
// Helper: Run nnz step, sync, read back nnzC and rowPtrC
// ============================================================================

inline aclsparseStatus_t RunGeam2NnzStep(
    sparse_test::HandleManager &handle,
    aclrtStream stream,
    int m, int n, int nnzA, int nnzB,
    const Geam2PreparedInputs &inputs,
    Geam2Workspace &ws,
    bool useHostPointerMode,
    int &nnzCHost,
    int32_t &nnzCOut,
    std::vector<int32_t> &rowPtrCOut)
{
    using namespace sparse_test;
    int nnzC_host = 0;
    ws.nnzCHolder = PrepareNnzCPtr(useHostPointerMode, &nnzC_host);

    aclsparseStatus_t ret = ComputeGeam2Nnz(
        handle, m, n,
        inputs.descrA.cget(), nnzA,
        reinterpret_cast<const int*>(inputs.bufsA.rowPtr.raw()),
        reinterpret_cast<const int*>(inputs.bufsA.colInd.raw()),
        inputs.descrB.cget(), nnzB,
        reinterpret_cast<const int*>(inputs.bufsB.rowPtr.raw()),
        reinterpret_cast<const int*>(inputs.bufsB.colInd.raw()),
        inputs.descrC.cget(),
        reinterpret_cast<int*>(ws.dRowPtrC.get()),
        ws.nnzCHolder.ptr,
        ws.dWorkspace.get());

    if (ret != ACL_SPARSE_STATUS_SUCCESS) {
        std::cerr << "[NPU] Xcsrgeam2Nnz failed: " << ret << std::endl;
        return ret;
    }

    aclrtSynchronizeStream(stream);
    nnzCHost = nnzC_host;
    nnzCOut = useHostPointerMode ? nnzC_host : 0;
    if (!useHostPointerMode) {
        ws.nnzCHolder.dNnzC.copyToHost(&nnzCOut, sizeof(int32_t));
    }
    rowPtrCOut.resize(m + 1);
    ws.dRowPtrC.copyToHost(rowPtrCOut.data(), (m + 1) * sizeof(int32_t));
    return ACL_SPARSE_STATUS_SUCCESS;
}

// ============================================================================
// Helper: Compute values and gather all results to host
// ============================================================================

inline aclsparseStatus_t ComputeAndGatherGeam2Results(
    sparse_test::HandleManager &handle,
    aclrtStream stream,
    int m, int n, int nnzA, int nnzB, int32_t nnzC,
    const Geam2PreparedInputs &inputs,
    const ScalarPtrs &scalars,
    Geam2Workspace &ws,
    CsrGeam2NpuResult &result)
{
    using namespace sparse_test;
    auto dColIndC = DeviceBuffer::alloc(nnzC * sizeof(int32_t));
    auto dValC = DeviceBuffer::alloc(nnzC * sizeof(float));

    const float *valA = reinterpret_cast<const float*>(inputs.bufsA.val.raw());
    const int *rowA = reinterpret_cast<const int*>(inputs.bufsA.rowPtr.raw());
    const int *colA = reinterpret_cast<const int*>(inputs.bufsA.colInd.raw());
    const float *valB = reinterpret_cast<const float*>(inputs.bufsB.val.raw());
    const int *rowB = reinterpret_cast<const int*>(inputs.bufsB.rowPtr.raw());
    const int *colB = reinterpret_cast<const int*>(inputs.bufsB.colInd.raw());
    float *outValC = reinterpret_cast<float*>(dValC.get());
    int *outRowC = reinterpret_cast<int*>(ws.dRowPtrC.get());
    int *outColC = reinterpret_cast<int*>(dColIndC.get());

    result.computeRet = ComputeGeam2Values(
        handle, m, n, scalars.alphaPtr, scalars.betaPtr,
        inputs.descrA.cget(), nnzA, valA, rowA, colA,
        inputs.descrB.cget(), nnzB, valB, rowB, colB,
        inputs.descrC.cget(), outValC, outRowC, outColC, ws.dWorkspace.get());

    if (result.computeRet != ACL_SPARSE_STATUS_SUCCESS) {
        std::cerr << "[NPU] Scsrgeam2 failed: " << result.computeRet << std::endl;
        return result.computeRet;
    }

    aclrtSynchronizeStream(stream);
    result.rowPtrC.resize(m + 1);
    ws.dRowPtrC.copyToHost(result.rowPtrC.data(), (m + 1) * sizeof(int32_t));
    result.colIndC.resize(nnzC);
    dColIndC.copyToHost(result.colIndC.data(), nnzC * sizeof(int32_t));
    result.valuesC.resize(nnzC);
    dValC.copyToHost(result.valuesC.data(), nnzC * sizeof(float));
    result.nnzC = nnzC;
    return ACL_SPARSE_STATUS_SUCCESS;
}

// ============================================================================
// Helper: Check and finalize empty output (nnzC <= 0)
// ============================================================================

inline bool HandleEmptyGeam2Output(int32_t nnzC, int indexBaseC, CsrGeam2NpuResult &result) {
    if (nnzC <= 0) {
        result.nnzC = 0;
        result.rowPtrC.assign(1, indexBaseC);
        result.colIndC.clear();
        result.valuesC.clear();
        return true;
    }
    return false;
}

// ============================================================================
// NPU wrapper: Full csrgeam2 three-step workflow
// Step 1: bufferSizeExt  Step 2: Xcsrgeam2Nnz  Step 3: Scsrgeam2
// Device memory managed via DeviceBuffer RAII; MatDescr via MatDescrGuard RAII.
// ============================================================================

inline CsrGeam2NpuResult CsrGeam2Npu(
    sparse_test::HandleManager &handle, aclrtStream stream,
    const std::vector<int32_t> &rowPtrA_host,
    const std::vector<int32_t> &colIndA_host,
    const std::vector<float> &valuesA_host, int nnzA,
    const std::vector<int32_t> &rowPtrB_host,
    const std::vector<int32_t> &colIndB_host,
    const std::vector<float> &valuesB_host, int nnzB,
    int m, int n, float alpha, float beta,
    int indexBaseA, int indexBaseB, int indexBaseC,
    bool useHostPointerMode)
{
    using namespace sparse_test;
    CsrGeam2NpuResult result{};
    if (m <= 0 || n <= 0) {
        result.rowPtrC.assign(1, indexBaseC);
        return result;
    }
    aclsparseSetPointerMode(handle.get(), useHostPointerMode
        ? ACL_SPARSE_POINTER_MODE_HOST : ACL_SPARSE_POINTER_MODE_DEVICE);
    Geam2PreparedInputs inputs;
    PrepareGeam2Inputs(handle,
        rowPtrA_host, colIndA_host, valuesA_host, nnzA,
        rowPtrB_host, colIndB_host, valuesB_host, nnzB,
        m, alpha, beta,
        indexBaseA, indexBaseB, indexBaseC, inputs);
    auto scalars = PrepareScalars(alpha, beta, useHostPointerMode);
    Geam2Workspace ws;
    size_t workSize = 0;
    result.bufferSizeRet = AllocateGeam2Workspace(
        handle, m, n, nnzA, nnzB, inputs, scalars, ws, workSize);
    if (result.bufferSizeRet != ACL_SPARSE_STATUS_SUCCESS) {
        return result;
    }
    int nnzCHost = 0;
    int32_t nnzC = 0;
    result.nnzRet = RunGeam2NnzStep(
        handle, stream, m, n, nnzA, nnzB, inputs, ws,
        useHostPointerMode, nnzCHost, nnzC, result.rowPtrC);
    if (result.nnzRet != ACL_SPARSE_STATUS_SUCCESS) {
        return result;
    }
    if (HandleEmptyGeam2Output(nnzC, indexBaseC, result)) {
        result.nnzC = nnzC;
        return result;
    }
    ComputeAndGatherGeam2Results(
        handle, stream, m, n, nnzA, nnzB, nnzC, inputs, scalars, ws, result);
    return result;
}
