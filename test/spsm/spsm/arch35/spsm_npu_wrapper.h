/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TEST_SPSM_SPSM_ARCH35_SPSM_NPU_WRAPPER_H_
#define TEST_SPSM_SPSM_ARCH35_SPSM_NPU_WRAPPER_H_

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>

#include "acl/acl.h"
#include "cann_ops_sparse.h"
#include "descriptor_manager.h"
#include "spsm.h"

// ============================================================================
// v2 wrapper: SpSM three-stage Generic API with opB parameter.
//
// The CANN SpSM API (cann_ops_sparse.h §584-613) now includes opB in all three
// stages (BufferSize / Analysis / Solve), aligned with cuSPARSE cusparseSpSM.
// This wrapper transmits opB (default NON_TRANSPOSE; T -> NOT_SUPPORTED).
//
// v2 new features:
//   - diagType: UNIT / NON_UNIT (transmitted via aclsparseSpMatSetAttribute)
//   - order: ROW / COL (transmitted via DnMatManager create with order param)
//   - inplace: matB/matC share the same device buffer
//   - indexBase: ZERO / ONE (transmitted via aclsparseCreateCsr idxBase)
// ============================================================================

// ============================================================================
// RAII guard for aclsparseSpSMDescr_t (cross-stage opaque descriptor).
// ============================================================================
class SpsmDescrGuard {
public:
    SpsmDescrGuard() {
        auto s = aclsparseSpSMCreateDescr(&descr_);
        if (s != ACL_SPARSE_STATUS_SUCCESS) {
            throw std::runtime_error("aclsparseSpSMCreateDescr failed");
        }
    }
    ~SpsmDescrGuard() {
        if (descr_) aclsparseSpSMDestroyDescr(descr_);
    }
    aclsparseSpSMDescr_t get() { return descr_; }
    SpsmDescrGuard(const SpsmDescrGuard&) = delete;
    SpsmDescrGuard& operator=(const SpsmDescrGuard&) = delete;
private:
    aclsparseSpSMDescr_t descr_ = nullptr;
};

// ============================================================================
// NPU workflow result.
//   X              : solution (layout per order, ldb x n), FP32
//   bufferSizeRet  : return code of aclsparseSpSMBufferSize
//   analysisRet    : return code of aclsparseSpSMAnalysis
//   solveRet       : return code of aclsparseSpSM
//   timingMs       : per-stage wall-clock latency (ms) for performance capture.
//                    [0]=BufferSize [1]=Analysis [2]=Solve [3]=total(SpsmNpu)
// ============================================================================
struct SpsmNpuResult {
    std::vector<float> X;
    aclsparseStatus_t bufferSizeRet = ACL_SPARSE_STATUS_SUCCESS;
    aclsparseStatus_t analysisRet = ACL_SPARSE_STATUS_SUCCESS;
    aclsparseStatus_t solveRet = ACL_SPARSE_STATUS_SUCCESS;
    double timingMs[4] = {0.0, 0.0, 0.0, 0.0};
    // Performance metadata (read back from workspace tiling after Analysis).
    int32_t levelCount = 0;     // L: total level count (parallelism indicator)
    int32_t kChunkSize = 0;     // n-dim vectorization chunk size
    int32_t maxRowLen = 0;      // max row nonzero count
};

struct SpsmNpuContext {
    aclDataType dtype = ACL_FLOAT;
    aclDataType computeType = ACL_FLOAT;
    int64_t nnz64 = 0;
    size_t valElemBytes = 0;
    size_t bElemCount = 0;
    float alphaFp32 = 0.0f;
    const void* alphaPtr = nullptr;
    std::vector<uint8_t> hCsrValsBytes;
    std::vector<uint8_t> hBBytes;
    sparse_test::DeviceBuffer dRowPtr;
    sparse_test::DeviceBuffer dColInd;
    sparse_test::DeviceBuffer dCsrVals;
    sparse_test::DeviceBuffer dC;
    sparse_test::DeviceBuffer dB;
    void* matBValuesPtr = nullptr;
    sparse_test::SpMatManager matA;
    sparse_test::DnMatManager matB;
    sparse_test::DnMatManager matC;
    sparse_test::DeviceBuffer dBuffer;
};

inline bool SpsmNpuValidateInputs(int m, int n, int ldb, SpsmNpuResult& result) {
    (void)ldb;  // retained for API stability; result is empty for degenerate m/n
    if (m <= 0 || n <= 0) {
        // m/n<=0 (incl. negative n) -> no valid output. Use clear() instead of
        // assign(static_cast<size_t>(ldb) * n, ...) which overflows size_t when
        // n is negative and would attempt a multi-GB allocation.
        result.X.clear();
        return false;
    }
    return true;
}

inline void SpsmNpuPrepareHostBuffers(SpsmNpuContext& ctx,
    const std::vector<float>& csrVals, const std::vector<float>& B_host,
    int32_t nnz, int m, int n, int ldb, float alpha,
    aclsparseOrder_t order) {
    ctx.dtype = ACL_FLOAT;
    ctx.computeType = ACL_FLOAT;
    ctx.nnz64 = static_cast<int64_t>(nnz);
    ctx.valElemBytes = sizeof(float);
    // Buffer size: COL order -> ldb * n; ROW order -> ldb * m (v2 §3.10).
    ctx.bElemCount = (order == ACL_SPARSE_ORDER_ROW)
        ? static_cast<size_t>(ldb) * m
        : static_cast<size_t>(ldb) * n;
    ctx.hCsrValsBytes.insert(ctx.hCsrValsBytes.end(),
        reinterpret_cast<const uint8_t*>(csrVals.data()),
        reinterpret_cast<const uint8_t*>(csrVals.data()) + nnz * sizeof(float));
    ctx.hBBytes.insert(ctx.hBBytes.end(),
        reinterpret_cast<const uint8_t*>(B_host.data()),
        reinterpret_cast<const uint8_t*>(B_host.data()) + ctx.bElemCount * sizeof(float));
    ctx.alphaFp32 = alpha;
    ctx.alphaPtr = &ctx.alphaFp32;
}

inline void SpsmNpuPrepareDeviceBuffers(SpsmNpuContext& ctx,
    const std::vector<int32_t>& csrRowPtr,
    const std::vector<int32_t>& csrColInd,
    int32_t nnz, int m, bool inplace) {
    using namespace sparse_test;
    ctx.dRowPtr = DeviceBuffer::copyFrom(csrRowPtr.data(), (m + 1) * sizeof(int32_t));
    if (nnz > 0) {
        ctx.dColInd = DeviceBuffer::copyFrom(csrColInd.data(), nnz * sizeof(int32_t));
        ctx.dCsrVals = DeviceBuffer::copyFrom(ctx.hCsrValsBytes.data(), ctx.hCsrValsBytes.size());
    } else {
        ctx.dColInd = DeviceBuffer::alloc(sizeof(int32_t));
        ctx.dCsrVals = DeviceBuffer::alloc(ctx.valElemBytes);
    }
    // inplace=true: matB and matC share the same device buffer.
    //   Copy B_host into the shared buffer; Solve writes X back into it.
    // inplace=false: separate dB (input) and dC (output).
    if (inplace) {
        ctx.dC = DeviceBuffer::copyFrom(ctx.hBBytes.data(), ctx.hBBytes.size());
        ctx.matBValuesPtr = ctx.dC.get();
    } else {
        ctx.dC = DeviceBuffer::alloc(ctx.bElemCount * ctx.valElemBytes);
        ctx.dB = DeviceBuffer::copyFrom(ctx.hBBytes.data(), ctx.hBBytes.size());
        ctx.matBValuesPtr = ctx.dB.get();
    }
}

inline bool SpsmNpuSetupDescriptors(SpsmNpuContext& ctx,
    int m, int n, int ldb, aclsparseFillMode_t uplo,
    aclsparseDiagType_t diagType, aclsparseOrder_t order,
    aclsparseIndexBase_t idxBase, SpsmNpuResult& result) {
    using namespace sparse_test;
    ctx.matA = SpMatManager::createCsr(m, m, ctx.nnz64, ctx.dRowPtr.get(),
        ctx.dColInd.get(), ctx.dCsrVals.get(), ACL_SPARSE_INDEX_32I,
        ACL_SPARSE_INDEX_32I, idxBase, ctx.dtype);
    auto setRet = aclsparseSpMatSetAttribute(ctx.matA.get(), ACL_SPARSE_SPMAT_FILL_MODE, &uplo, sizeof(uplo));
    if (setRet != ACL_SPARSE_STATUS_SUCCESS) {
        std::cerr << "[NPU] aclsparseSpMatSetAttribute(FILL_MODE) failed: " << setRet << std::endl;
        result.bufferSizeRet = setRet;
        return false;
    }
    setRet = aclsparseSpMatSetAttribute(ctx.matA.get(), ACL_SPARSE_SPMAT_DIAG_TYPE, &diagType, sizeof(diagType));
    if (setRet != ACL_SPARSE_STATUS_SUCCESS) {
        std::cerr << "[NPU] aclsparseSpMatSetAttribute(DIAG_TYPE) failed: " << setRet << std::endl;
        result.bufferSizeRet = setRet;
        return false;
    }
    ctx.matB = DnMatManager::createConst(m, n, ldb, ctx.matBValuesPtr, ctx.dtype, order);
    ctx.matC = DnMatManager::create(m, n, ldb, ctx.dC.get(), ctx.dtype, order);
    return true;
}

inline bool SpsmNpuRunAnalysis(SpsmNpuContext& ctx,
    aclsparseSpSMDescr_t spsmDescr, sparse_test::HandleManager& handle,
    aclsparseOperation_t opA, aclsparseOperation_t opB,
    SpsmNpuResult& result) {
    using Clock = std::chrono::high_resolution_clock;
    size_t bufferSize = 0;
    auto t0 = Clock::now();
    result.bufferSizeRet = aclsparseSpSMBufferSize(handle.get(), opA, opB,
        ctx.alphaPtr, ctx.matA.cget(), ctx.matB.cget(), ctx.matC.get(),
        ctx.computeType, ACL_SPARSE_SPSM_ALG_DEFAULT, spsmDescr, &bufferSize);
    result.timingMs[0] = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
    if (result.bufferSizeRet != ACL_SPARSE_STATUS_SUCCESS) {
        std::cerr << "[NPU] aclsparseSpSMBufferSize failed: " << result.bufferSizeRet << std::endl;
        return false;
    }
    if (bufferSize == 0) bufferSize = 16;
    ctx.dBuffer = sparse_test::DeviceBuffer::alloc(bufferSize);
    t0 = Clock::now();
    result.analysisRet = aclsparseSpSMAnalysis(handle.get(), opA, opB,
        ctx.alphaPtr, ctx.matA.cget(), ctx.matB.cget(), ctx.matC.get(),
        ctx.computeType, ACL_SPARSE_SPSM_ALG_DEFAULT, spsmDescr, ctx.dBuffer.get());
    result.timingMs[1] = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
    if (result.analysisRet != ACL_SPARSE_STATUS_SUCCESS) {
        std::cerr << "[NPU] aclsparseSpSMAnalysis failed: " << result.analysisRet << std::endl;
        return false;
    }
    // Read back performance metadata from the descriptor's cached tiling.
    // 违规 1 消除: tiling 不再落盘 workspace GM (host by-value 缓存到 spsmDescr->cachedTiling)。
    // 旧的 workspace aclrtMemcpy 回读会读到未初始化内存, 改为直接从描述符缓存读取。
    //
    // 经公共 getter GetSpsmCachedTiling 读取 (spsm.h 提供), 不再经 offsetof 直读
    // opaque 描述符内部布局, 避免描述符布局变更后静默读到错误字段。
    // 不影响精度验证（仅用于 perf 日志输出）。
    const SpsmTilingData& cachedTiling = GetSpsmCachedTiling(spsmDescr);
    result.levelCount = cachedTiling.L;
    result.kChunkSize = cachedTiling.kChunkSize;
    result.maxRowLen = cachedTiling.maxRowLen;
    return true;
}

inline bool SpsmNpuRunSolve(SpsmNpuContext& ctx,
    aclsparseSpSMDescr_t spsmDescr, sparse_test::HandleManager& handle,
    aclrtStream stream, aclsparseOperation_t opA, aclsparseOperation_t opB,
    SpsmNpuResult& result) {
    using Clock = std::chrono::high_resolution_clock;
    auto t0 = Clock::now();
    result.solveRet = aclsparseSpSM(handle.get(), opA, opB, ctx.alphaPtr,
        ctx.matA.cget(), ctx.matB.cget(), ctx.matC.get(), ctx.computeType,
        ACL_SPARSE_SPSM_ALG_DEFAULT, spsmDescr);
    result.timingMs[2] = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
    if (result.solveRet != ACL_SPARSE_STATUS_SUCCESS) {
        std::cerr << "[NPU] aclsparseSpSM failed: " << result.solveRet << std::endl;
        return false;
    }
    if (aclrtSynchronizeStream(stream) != ACL_SUCCESS) {
        std::cerr << "[NPU] aclrtSynchronizeStream failed" << std::endl;
        result.solveRet = ACL_SPARSE_STATUS_EXECUTION_FAILED;
        return false;
    }
    return true;
}

inline void SpsmNpuCollectResults(SpsmNpuContext& ctx, SpsmNpuResult& result) {
    std::vector<uint8_t> hCBytes(ctx.bElemCount * ctx.valElemBytes);
    ctx.dC.copyToHost(hCBytes.data(), hCBytes.size());
    result.X.resize(ctx.bElemCount);
    std::copy_n(reinterpret_cast<const float*>(hCBytes.data()), ctx.bElemCount, result.X.data());
}

// ============================================================================
// NPU wrapper: SpSM three-stage Generic API (v2, with opB).
//
// @param handle     HandleManager (RAII)
// @param stream     aclrt stream
// @param csrRowPtr  host CSR row pointers (size m+1)
// @param csrColInd  host CSR column indices (size nnz)
// @param csrVals    host CSR values (size nnz, FP32)
// @param nnz        nonzero count
// @param B_host     host dense RHS (layout per order, ldb x n, FP32)
// @param m          A is m x m, B/X are m x n
// @param n          number of RHS columns
// @param ldb        leading dimension of B / X
// @param alpha      scalar (FP32)
// @param opA        ACL_SPARSE_OP_NON_TRANSPOSE / ACL_SPARSE_OP_TRANSPOSE
// @param opB        ACL_SPARSE_OP_NON_TRANSPOSE (T -> NOT_SUPPORTED)  [v2 new]
// @param uplo       ACL_SPARSE_FILL_MODE_LOWER / ACL_SPARSE_FILL_MODE_UPPER
// @param diagType   ACL_SPARSE_DIAG_TYPE_UNIT / ACL_SPARSE_DIAG_TYPE_NON_UNIT [v2 new]
// @param order      ACL_SPARSE_ORDER_COL / ACL_SPARSE_ORDER_ROW [v2 new]
// @param inplace    true -> matB/matC share device buffer [v2 new]
// @param idxBase    ACL_SPARSE_INDEX_BASE_ZERO / ACL_SPARSE_INDEX_BASE_ONE [v2 new]
// ============================================================================
inline SpsmNpuResult SpsmNpu(
    sparse_test::HandleManager& handle, aclrtStream stream,
    const std::vector<int32_t>& csrRowPtr,
    const std::vector<int32_t>& csrColInd,
    const std::vector<float>& csrVals,
    int32_t nnz,
    const std::vector<float>& B_host,
    int m, int n, int ldb,
    float alpha,
    aclsparseOperation_t opA,
    aclsparseOperation_t opB,                    // v2 new
    aclsparseFillMode_t uplo,
    aclsparseDiagType_t diagType,                // v2 new
    aclsparseOrder_t order,                       // v2 new
    bool inplace,                                 // v2 new
    aclsparseIndexBase_t idxBase) {               // v2 new
    using namespace sparse_test;
    SpsmNpuResult result{};
    if (!SpsmNpuValidateInputs(m, n, ldb, result)) return result;

    handle.setStream(stream);

    SpsmNpuContext ctx;
    SpsmNpuPrepareHostBuffers(ctx, csrVals, B_host, nnz, m, n, ldb, alpha, order);
    SpsmNpuPrepareDeviceBuffers(ctx, csrRowPtr, csrColInd, nnz, m, inplace);
    if (!SpsmNpuSetupDescriptors(ctx, m, n, ldb, uplo, diagType, order, idxBase, result)) {
        return result;
    }

    SpsmDescrGuard spsmDescr;
    using Clock = std::chrono::high_resolution_clock;
    auto tTotal = Clock::now();
    if (!SpsmNpuRunAnalysis(ctx, spsmDescr.get(), handle, opA, opB, result)) return result;
    if (!SpsmNpuRunSolve(ctx, spsmDescr.get(), handle, stream, opA, opB, result)) return result;
    result.timingMs[3] = std::chrono::duration<double, std::milli>(Clock::now() - tTotal).count();

    SpsmNpuCollectResults(ctx, result);
    return result;
}

#endif  // TEST_SPSM_SPSM_ARCH35_SPSM_NPU_WRAPPER_H_
