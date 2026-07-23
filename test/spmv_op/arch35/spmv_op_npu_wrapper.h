/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * SpMVOp NPU wrapper (uses RAII managers from test/frame/descriptor_manager.h).
 * Calls: SpMVOp_bufferSize / createDescr / createPlan / execute
 */

#ifndef TEST_SPMV_OP_NPU_WRAPPER_H_
#define TEST_SPMV_OP_NPU_WRAPPER_H_

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "acl/acl.h"
#include "cann_ops_sparse.h"
#include "descriptor_manager.h"
#include "fill.h"
#include "sparse_test.h"

namespace sparse_test {

// Shuffle column indices within each CSR row (for unsorted index tests).
inline CsrMatrix ShuffleCsrIndices(CsrMatrix csr, uint32_t seed) {
    std::mt19937 rng(seed + 999);
    const int rows = static_cast<int>(csr.rows);
    for (int i = 0; i < rows; i++) {
        int lo = csr.rowOffsets[i];
        int hi = csr.rowOffsets[i + 1];
        if (hi - lo > 1) {
            std::vector<std::pair<int32_t, float>> pairs;
            pairs.reserve(hi - lo);
            for (int j = lo; j < hi; j++) {
                pairs.emplace_back(csr.colIndices[j], csr.values[j]);
            }
            std::shuffle(pairs.begin(), pairs.end(), rng);
            for (int j = lo; j < hi; j++) {
                csr.colIndices[j] = pairs[j - lo].first;
                csr.values[j]     = pairs[j - lo].second;
            }
        }
    }
    return csr;
}

// Map algorithm string to enum
inline aclsparseSpMVOpAlg_t ParseSpMVOpAlg(const std::string& s) {
    if (s == "ALG2")  return ACL_SPARSE_SPMVOP_ALG2;
    if (s == "ALG1")  return ACL_SPARSE_SPMVOP_ALG1;
    return ACL_SPARSE_SPMVOP_ALG_DEFAULT;
}

// Map row-offset type string to ACL enum
inline aclsparseIndexType_t ParseRowIndexType(const std::string& s) {
    if (s == "i64") return ACL_SPARSE_INDEX_64I;
    return ACL_SPARSE_INDEX_32I;
}

// SpMVOp NPU wrapper: Z = alpha * A * X + beta * Y
//
// Parameters:
//   spHandle    - aclsparse handle (RAII)
//   stream      - ACL stream
//   csr         - input CSR matrix A (host, FP32 values, int32 colIndices)
//   x, y        - host vectors
//   alpha, beta - scalars
//   ptrMode     - "HOST" or "DEVICE"
//   rowIdxType  - ACL_SPARSE_INDEX_32I or ACL_SPARSE_INDEX_64I
//   algEnum     - algorithm (DEFAULT/ALG1/ALG2)
//   caseId      - for error messages
//
// Returns: Z as float vector (size = m); empty vector if m == 0.
//
// Throws std::runtime_error on any ACL API failure.
//  RAII guard for SpMVOp description
struct SpMVOpDescrGuard {
    aclsparseSpMVOpDescr_t d{nullptr};
    explicit SpMVOpDescrGuard(aclsparseSpMVOpDescr_t p) : d(p) {}
    ~SpMVOpDescrGuard() { if (d) aclsparseSpMVOp_destroyDescr(d); }
    SpMVOpDescrGuard(const SpMVOpDescrGuard&) = delete;
    SpMVOpDescrGuard& operator=(const SpMVOpDescrGuard&) = delete;
};

struct SpMVOpPlanGuard {
    aclsparseSpMVOpPlan_t p{nullptr};
    explicit SpMVOpPlanGuard(aclsparseSpMVOpPlan_t ptr) : p(ptr) {}
    ~SpMVOpPlanGuard() { if (p) aclsparseSpMVOp_destroyPlan(p); }
    SpMVOpPlanGuard(const SpMVOpPlanGuard&) = delete;
    SpMVOpPlanGuard& operator=(const SpMVOpPlanGuard&) = delete;
};

// bufferSize + createDescr: prepare internal SpMVOp descriptor and workspace
inline aclsparseSpMVOpDescr_t CreateSpMVOpDescrFromMat(
    HandleManager& spHandle, aclsparseOperation_t opA,
    SpMatManager& matA, DnVecManager& vecX, DnVecManager& vecY, DnVecManager& vecZ,
    aclsparseSpMVOpAlg_t algEnum, DeviceBuffer& workBuffer, const std::string& caseId)
{
    size_t bufferSz = 0;
    auto st = aclsparseSpMVOp_bufferSize(
        spHandle.get(), opA, matA.get(), vecX.cget(), vecY.cget(), vecZ.get(),
        ACL_FLOAT, algEnum, &bufferSz);
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        throw std::runtime_error("[" + caseId + "] bufferSize failed, status=" +
                                  std::to_string(static_cast<int>(st)));
    }
    workBuffer = (bufferSz > 0) ? DeviceBuffer::alloc(bufferSz)
                                : DeviceBuffer::alloc(1);
    aclsparseSpMVOpDescr_t opDescr = nullptr;
    st = aclsparseSpMVOp_createDescr(
        spHandle.get(), &opDescr, opA, matA.get(),
        vecX.cget(), vecY.cget(), vecZ.get(),
        ACL_FLOAT, algEnum, workBuffer.get());
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        throw std::runtime_error("[" + caseId + "] createDescr failed, status=" +
                                  std::to_string(static_cast<int>(st)));
    }
    return opDescr;
}

// createPlan + execute + sync + copyback
inline std::vector<float> PlanExecuteCopyback(
    HandleManager& spHandle, aclrtStream stream, int64_t m,
    aclsparseSpMVOpDescr_t opDescr,
    DnVecManager& vecX, DnVecManager& vecY, DnVecManager& vecZ,
    DeviceBuffer& dY, DeviceBuffer& dZ, int aliasZY,
    const void* alphaPtr, const void* betaPtr, const std::string& caseId)
{
    aclsparseSpMVOpPlan_t plan = nullptr;
    auto st = aclsparseSpMVOp_createPlan(spHandle.get(), opDescr, &plan, nullptr, 0);
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        throw std::runtime_error("[" + caseId + "] createPlan failed, status=" +
                                  std::to_string(static_cast<int>(st)));
    }
    SpMVOpPlanGuard planGuard(plan);
    st = aclsparseSpMVOp(
        spHandle.get(), plan, alphaPtr, betaPtr,
        vecX.cget(), vecY.cget(), vecZ.get());
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        throw std::runtime_error("[" + caseId + "] execute failed, status=" +
                                  std::to_string(static_cast<int>(st)));
    }
    if (aclrtSynchronizeStream(stream) != ACL_SUCCESS) {
        throw std::runtime_error("[" + caseId + "] SynchronizeStream failed");
    }
    std::vector<float> zHost(m);
    if (aliasZY) {
        dY.copyToHost(zHost.data(), m * sizeof(float));
    } else {
        dZ.copyToHost(zHost.data(), m * sizeof(float));
    }
    return zHost;
}

// Helper: prepare device buffers for SpMVOp
struct SpMVOpBuffers {
    DeviceBuffer dRowOffsets, dColIndices, dValues, dX, dY, dZ, dAlpha, dBeta;
    std::vector<int64_t> rowOffsetsI64;
};

inline SpMVOpBuffers PrepareSpMVOpBuffers(
    HandleManager& spHandle,
    const CsrMatrix& csr,
    const std::vector<float>& x,
    const std::vector<float>& y,
    float alpha, float beta,
    const std::string& ptrMode,
    aclsparseIndexType_t rowIdxType,
    const std::string& caseId)
{
    const int64_t m = csr.rows, k = csr.cols, nnz = csr.nnz;
    SpMVOpBuffers bufs;

    if (nnz > 0) {
        bufs.dColIndices = DeviceBuffer::copyFrom(csr.colIndices.data(), nnz * sizeof(int32_t));
        bufs.dValues = DeviceBuffer::copyFrom(csr.values.data(), nnz * sizeof(float));
    }
    if (k > 0) {
        bufs.dX = DeviceBuffer::copyFrom(x.data(), k * sizeof(float));
    }
    bufs.dY = DeviceBuffer::copyFrom(y.data(), m * sizeof(float));
    bufs.dZ = DeviceBuffer::alloc(m * sizeof(float));

    // Copy row offsets to device, converting to i64 if needed
    const void* srcRoPtr = csr.rowOffsets.data();
    size_t roSize = (m + 1) * sizeof(int32_t);
    if (rowIdxType == ACL_SPARSE_INDEX_64I) {
        bufs.rowOffsetsI64.resize(m + 1);
        for (int64_t i = 0; i <= m; ++i) {
            bufs.rowOffsetsI64[i] = csr.rowOffsets[i];
        }
        srcRoPtr = bufs.rowOffsetsI64.data();
        roSize = (m + 1) * sizeof(int64_t);
    }
    bufs.dRowOffsets = DeviceBuffer::copyFrom(srcRoPtr, roSize);

    // Allocate device buffers for alpha/beta (used when ptrMode == "DEVICE")
    bufs.dAlpha = DeviceBuffer::copyFrom(&alpha, sizeof(float));
    bufs.dBeta = DeviceBuffer::copyFrom(&beta, sizeof(float));

    // Set pointer mode
    auto mode = (ptrMode == "DEVICE") ? ACL_SPARSE_POINTER_MODE_DEVICE : ACL_SPARSE_POINTER_MODE_HOST;
    auto st = aclsparseSetPointerMode(spHandle.get(), mode);
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        throw std::runtime_error("[" + caseId + "] aclsparseSetPointerMode failed, status=" +
                                  std::to_string(static_cast<int>(st)));
    }

    return bufs;
}

// SpMVOp NPU wrapper: Z = alpha * A * X + beta * Y
inline std::vector<float> SpMVOpNpuWrapper(
    HandleManager& spHandle,
    aclrtStream stream,
    const CsrMatrix& csr,
    const std::vector<float>& x,
    const std::vector<float>& y,
    float alpha, float beta,
    const std::string& ptrMode,
    aclsparseIndexType_t rowIdxType,
    aclsparseSpMVOpAlg_t algEnum,
    int aliasZY,
    const std::string& caseId)
{
    const int64_t m = csr.rows, k = csr.cols, nnz = csr.nnz;

    // Early exit for empty matrices
    if (m <= 0) {
        return std::vector<float>(m, 0.0f);
    }

    // Prepare device buffers: CSR data, vectors, scalars, pointer mode
    auto bufs = PrepareSpMVOpBuffers(spHandle, csr, x, y, alpha, beta,
                                      ptrMode, rowIdxType, caseId);

    // Determine alpha/beta pointers based on ptrMode
    const void* alphaPtr = (ptrMode == "DEVICE") ? bufs.dAlpha.get() : &alpha;
    const void* betaPtr = (ptrMode == "DEVICE") ? bufs.dBeta.get() : &beta;

    // Create matrix & vector descriptors
    void* zPtr = aliasZY ? bufs.dY.get() : bufs.dZ.get();
    SpMatManager matA = SpMatManager::createConstCsr(
        m, k, nnz, bufs.dRowOffsets.get(), bufs.dColIndices.get(), bufs.dValues.get(),
        rowIdxType, ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_BASE_ZERO, ACL_FLOAT);
    DnVecManager vecX = DnVecManager::createConst(k, bufs.dX.get(), ACL_FLOAT);
    DnVecManager vecY = DnVecManager::createConst(m, bufs.dY.get(), ACL_FLOAT);
    DnVecManager vecZ = DnVecManager::create(m, zPtr, ACL_FLOAT);

    // bufferSize + createDescr + plan + execute + copyback
    DeviceBuffer workBuffer;
    auto opDescr = CreateSpMVOpDescrFromMat(
        spHandle, ACL_SPARSE_OP_NON_TRANSPOSE, matA, vecX, vecY, vecZ,
        algEnum, workBuffer, caseId);
    SpMVOpDescrGuard descrGuard(opDescr);

    return PlanExecuteCopyback(
        spHandle, stream, m, opDescr, vecX, vecY, vecZ, bufs.dY, bufs.dZ, aliasZY,
        alphaPtr, betaPtr, caseId);
}

}  // namespace sparse_test

#endif
