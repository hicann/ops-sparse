/**
 * ----------------------------------------------------------------------------------------------------------
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms of the
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR
 * PURPOSE.
 * ----------------------------------------------------------------------------------------------------------
 */

#ifndef TEST_SPMM_OP_SPMM_OP_ARCH35_SPMM_OP_NPU_WRAPPER_H_
#define TEST_SPMM_OP_SPMM_OP_ARCH35_SPMM_OP_NPU_WRAPPER_H_

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "acl/acl.h"
#include "cann_ops_sparse.h"
#include "descriptor_manager.h"
#include "spmm_op_golden.h"  // SpmmCsr + MakeSpmmSparsity + FP16 bit helpers

namespace sparse_test {

// ============================================================================
// RAII guards for SpMMOp descr/plan (per test plan §2.4.1).
// ============================================================================

struct SpmmOpDescrGuard {
    aclsparseSpMMOpDescr_t d{nullptr};
    explicit SpmmOpDescrGuard(aclsparseSpMMOpDescr_t p) : d(p) {}
    ~SpmmOpDescrGuard()
    {
        if (d) {
            aclsparseSpMMOp_destroyDescr(d);
        }
    }
    void release() { d = nullptr; }  // Skip cleanup (use after NPU crash)
    SpmmOpDescrGuard(const SpmmOpDescrGuard&) = delete;
    SpmmOpDescrGuard& operator=(const SpmmOpDescrGuard&) = delete;
};

struct SpmmOpPlanGuard {
    aclsparseSpMMOpPlan_t p{nullptr};
    explicit SpmmOpPlanGuard(aclsparseSpMMOpPlan_t p) : p(p) {}
    ~SpmmOpPlanGuard()
    {
        if (p) {
            aclsparseSpMMOp_destroyPlan(p);
        }
    }
    void release() { p = nullptr; }  // Skip cleanup (use after NPU crash)
    SpmmOpPlanGuard(const SpmmOpPlanGuard&) = delete;
    SpmmOpPlanGuard& operator=(const SpmmOpPlanGuard&) = delete;
};

// ============================================================================
// NPU workflow result for SpMMOp.
// valuesOut holds C output (m*n dense) converted to FP64.
// ============================================================================

struct SpmmOpNpuResult {
    std::vector<double> valuesOut;  // C output, size = m*n, FP64
    size_t bufferSize = 0;

    aclsparseStatus_t bufferSizeRet = ACL_SPARSE_STATUS_SUCCESS;
    aclsparseStatus_t createDescrRet = ACL_SPARSE_STATUS_SUCCESS;
    aclsparseStatus_t createPlanRet = ACL_SPARSE_STATUS_SUCCESS;
    aclsparseStatus_t executeRet = ACL_SPARSE_STATUS_SUCCESS;
};

// ============================================================================
// String -> ACL enum helpers.
// ============================================================================

inline aclDataType ParseDtype(const std::string& s)
{
    if (s == "ACL_FLOAT16") {
        return ACL_FLOAT16;
    }
    if (s == "ACL_FLOAT") {
        return ACL_FLOAT;
    }
    return ACL_FLOAT;
}

inline aclsparseOperation_t ParseOperation(const std::string& s)
{
    if (s == "ACL_SPARSE_OP_TRANSPOSE") {
        return ACL_SPARSE_OP_TRANSPOSE;
    }
    if (s == "ACL_SPARSE_OP_CONJUGATE_TRANSPOSE") {
        return ACL_SPARSE_OP_CONJUGATE_TRANSPOSE;
    }
    return ACL_SPARSE_OP_NON_TRANSPOSE;
}

inline aclsparseOrder_t ParseOrder(const std::string& s)
{
    if (s == "ACL_SPARSE_ORDER_COL") {
        return ACL_SPARSE_ORDER_COL;
    }
    return ACL_SPARSE_ORDER_ROW;
}

inline aclsparseSpMMOpAlg_t ParseSpmmOpAlg(const std::string& s)
{
    if (s == "ACL_SPARSE_SPMMOP_ALG1") {
        return ACL_SPARSE_SPMMOP_ALG1;
    }
    if (s == "ACL_SPARSE_SPMMOP_ALG2") {
        return ACL_SPARSE_SPMMOP_ALG2;
    }
    if (s == "ACL_SPARSE_SPMMOP_ALG1_HIGH_PRECISION") {
        return ACL_SPARSE_SPMMOP_ALG1_HIGH_PRECISION;
    }
    return ACL_SPARSE_SPMMOP_ALG_DEFAULT;
}

inline aclsparseIndexBase_t ParseIndexBase(const std::string& s)
{
    if (s == "ACL_SPARSE_INDEX_BASE_ONE") {
        return ACL_SPARSE_INDEX_BASE_ONE;
    }
    return ACL_SPARSE_INDEX_BASE_ZERO;
}

inline aclsparseIndexType_t ParseIndexType(const std::string& s)
{
    if (s == "ACL_SPARSE_INDEX_64I") {
        return ACL_SPARSE_INDEX_64I;
    }
    return ACL_SPARSE_INDEX_32I;
}

// ============================================================================
// Row-major -> col-major repack helper (per test plan §2.4.3).
// ============================================================================
template <typename T>
static std::vector<T> RepackRowToCol(const std::vector<T>& rowBuf,
    int64_t rows, int64_t cols)
{
    if (rows <= 0 || cols <= 0) {
        return rowBuf;
    }
    std::vector<T> colBuf(rowBuf.size());
    for (int64_t i = 0; i < rows; ++i) {
        for (int64_t j = 0; j < cols; ++j) {
            colBuf[static_cast<size_t>(j) * static_cast<size_t>(rows) +
                   static_cast<size_t>(i)] =
                rowBuf[static_cast<size_t>(i) * static_cast<size_t>(cols) +
                       static_cast<size_t>(j)];
        }
    }
    return colBuf;
}

// Reverse: col-major -> row-major (inverse of RepackRowToCol).
template <typename T>
static std::vector<T> RepackColToRow(const std::vector<T>& colBuf,
    int64_t rows, int64_t cols)
{
    if (rows <= 0 || cols <= 0) {
        return colBuf;
    }
    std::vector<T> rowBuf(colBuf.size());
    for (int64_t i = 0; i < rows; ++i) {
        for (int64_t j = 0; j < cols; ++j) {
            rowBuf[static_cast<size_t>(i) * static_cast<size_t>(cols) +
                   static_cast<size_t>(j)] =
                colBuf[static_cast<size_t>(j) * static_cast<size_t>(rows) +
                       static_cast<size_t>(i)];
        }
    }
    return rowBuf;
}

// ============================================================================
// Helper 1: PrepareCsrDeviceBuffers
// Creates device buffers for CSR data (rowOffsets, colIndices, values).
// Handles I64 rowOffset conversion and m<=0 guard for roSize.
// rowOffsets always present (m+1 ints). colInd/values empty when nnz == 0.
// I64 rowOffsets conversion (ref: spmv_op_npu_wrapper.h:185-196 pattern):
// when rowOffsetType == ACL_SPARSE_INDEX_64I, convert int32 rowOffsets to
// int64 vector before copying to device.
// Also creates matA (CSR const descriptor) from the device buffers.
// rowOffsetType: 32I or 64I (I64 conversion done above)
// colIdxType: always 32I (colIndices vector is int32_t)
// indexBase: ZERO or ONE (passed through from test param)
// ============================================================================

template <typename T>
static void PrepareCsrDeviceBuffers(
    int64_t m, int64_t k, int64_t nnz, aclDataType dtype,
    aclsparseIndexType_t rowOffsetType, aclsparseIndexBase_t indexBase,
    const std::vector<int32_t>& rowOffsets, const std::vector<int32_t>& colIndices,
    const std::vector<T>& hAValues,
    DeviceBuffer& dRowOff, DeviceBuffer& dColInd, DeviceBuffer& dVals,
    SpMatManager& matA)
{
    std::vector<int64_t> rowOffsetsI64;
    const void* roSrcPtr = rowOffsets.data();
    size_t roSize = static_cast<size_t>(m + 1) * sizeof(int32_t);
    if (rowOffsetType == ACL_SPARSE_INDEX_64I) {
        rowOffsetsI64.resize(static_cast<size_t>(m + 1));
        for (int64_t i = 0; i <= m; ++i) {
            rowOffsetsI64[static_cast<size_t>(i)] =
                static_cast<int64_t>(rowOffsets[static_cast<size_t>(i)]);
        }
        roSrcPtr = rowOffsetsI64.data();
        roSize = static_cast<size_t>(m + 1) * sizeof(int64_t);
    }
    // Guard against m=0: still need 1 entry (rowOffsets[0])
    if (m <= 0) {
        roSize = (rowOffsetType == ACL_SPARSE_INDEX_64I) ? sizeof(int64_t) : sizeof(int32_t);
    }
    if (roSrcPtr == nullptr || roSize == 0) {
        dRowOff = DeviceBuffer::alloc(roSize > 0 ? roSize : sizeof(int32_t));
    } else {
        dRowOff = DeviceBuffer::copyFrom(roSrcPtr, roSize);
    }
    if (nnz > 0) {
        dColInd = DeviceBuffer::copyFrom(
            colIndices.data(), static_cast<size_t>(nnz) * sizeof(int32_t));
        dVals = DeviceBuffer::copyFrom(
            hAValues.data(), static_cast<size_t>(nnz) * sizeof(T));
    }

    // --- Create matA (CSR, const) --------------------------------------------
    matA = SpMatManager::createConstCsr(
        m, k, nnz, dRowOff.get(), dColInd.get(), dVals.get(),
        rowOffsetType, ACL_SPARSE_INDEX_32I,
        indexBase, dtype);
}

// ============================================================================
// Helper 2: PrepareDnMatDeviceBuffers
// Computes B/C descriptor shapes based on opB, repacks host buffers for COL
// order, creates device buffers, and creates DnMat descriptors (matB const,
// matC non-const).
// When a dimension is 0 (empty matrix, e.g. n=0), the computed ld may be 0,
// which aclsparseCreateDnMat rejects (ld <= 0). Use a non-zero fallback
// since ld is irrelevant for empty matrices (no elements to access).
// Golden always computes from canonical row-major. NPU side order=COL
// requires host buffer packed in column-major.
// Element count is 0 when dims are 0; allocate at least 1 byte to avoid
// null device pointers that some ACL APIs reject.
// ============================================================================

template <typename T>
static void PrepareDnMatDeviceBuffers(
    int64_t m, int64_t n, int64_t k,
    aclsparseOperation_t opB, aclsparseOrder_t orderB, aclsparseOrder_t orderC,
    aclDataType dtype,
    const std::vector<T>& hB, const std::vector<T>& hCInit,
    DnMatManager& matB, DnMatManager& matC,
    DeviceBuffer& dB, DeviceBuffer& dC)
{
    // --- Determine B descriptor shape based on opB ---------------------------
    const bool bTransposed = (opB == ACL_SPARSE_OP_TRANSPOSE);
    const int64_t bRows = bTransposed ? n : k;
    const int64_t bCols = bTransposed ? k : n;
    const int64_t bLdRaw = (orderB == ACL_SPARSE_ORDER_ROW) ? bCols : bRows;
    const int64_t bLd = (bLdRaw > 0) ? bLdRaw : 1;
    const int64_t cLdRaw = (orderC == ACL_SPARSE_ORDER_ROW) ? n : m;
    const int64_t cLd = (cLdRaw > 0) ? cLdRaw : 1;

    // --- Repack B/C for COL order --------------------------------------------
    std::vector<T> hBPacked = hB;
    if (orderB == ACL_SPARSE_ORDER_COL && bRows > 0 && bCols > 0) {
        hBPacked = RepackRowToCol(hB, bRows, bCols);
    }
    std::vector<T> hCInitPacked = hCInit;
    if (orderC == ACL_SPARSE_ORDER_COL && m > 0 && n > 0) {
        hCInitPacked = RepackRowToCol(hCInit, m, n);
    }

    // --- Device buffers for B and C ------------------------------------------
    size_t bElemCount = static_cast<size_t>(bRows) * static_cast<size_t>(bCols);
    size_t cElemCount = static_cast<size_t>(m) * static_cast<size_t>(n);
    dB = (bElemCount > 0)
        ? DeviceBuffer::copyFrom(hBPacked.data(), bElemCount * sizeof(T))
        : DeviceBuffer::alloc(sizeof(T));
    dC = (cElemCount > 0)
        ? DeviceBuffer::copyFrom(hCInitPacked.data(), cElemCount * sizeof(T))
        : DeviceBuffer::alloc(sizeof(T));

    // --- Create matB (dense, const) and matC (dense, non-const) --------------
    matB = DnMatManager::createConst(
        bRows, bCols, bLd, dB.raw(), dtype, orderB);
    matC = DnMatManager::create(
        m, n, cLd, dC.raw(), dtype, orderC);
}

// ============================================================================
// Helper 3: PrepareAlphaBeta
// Sets up alpha/beta pointers based on pointerMode (HOST or DEVICE).
// For DEVICE mode, copies alpha/beta to device buffers.
// For HOST mode, points alphaPtr/betaPtr to the caller-owned host floats.
// alphaHost/betaHost are caller-owned so that HOST-mode pointers remain valid
// after this helper returns (they are used later in aclsparseSpMMOp execute).
// Returns ACL_SPARSE_STATUS_SUCCESS on success, or the error from
// aclsparseSetPointerMode on failure.
// ============================================================================

static aclsparseStatus_t PrepareAlphaBeta(
    HandleManager& handle, aclsparsePointerMode_t pointerMode,
    float alpha, float beta,
    float& alphaHost, float& betaHost,
    const void*& alphaPtr, const void*& betaPtr,
    DeviceBuffer& dAlpha, DeviceBuffer& dBeta)
{
    alphaHost = alpha;
    betaHost = beta;

    if (pointerMode == ACL_SPARSE_POINTER_MODE_DEVICE) {
        dAlpha = DeviceBuffer::copyFrom(&alphaHost, sizeof(float));
        dBeta = DeviceBuffer::copyFrom(&betaHost, sizeof(float));
        alphaPtr = dAlpha.get();
        betaPtr = dBeta.get();
    } else {
        alphaPtr = &alphaHost;
        betaPtr = &betaHost;
    }
    auto st = aclsparseSetPointerMode(handle.get(), pointerMode);
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        std::cerr << "[SpMMOp NPU] aclsparseSetPointerMode failed: " << st << std::endl;
        return st;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

// ============================================================================
// Helper 4a: QueryBufferSizeAndAlloc
// Stage 1 of the SpMMOp lifecycle: queries bufferSize and allocates the
// workspace buffer (ALG2 requires it). Sets result.bufferSizeRet.
// ============================================================================

static aclsparseStatus_t QueryBufferSizeAndAlloc(
    SpmmOpNpuResult& result, HandleManager& handle,
    aclsparseOperation_t opB, SpMatManager& matA, DnMatManager& matB, DnMatManager& matC,
    aclDataType computeType, aclsparseSpMMOpAlg_t alg, DeviceBuffer& dBuffer)
{
    // --- Stage 1: bufferSize -------------------------------------------------
    result.bufferSizeRet = aclsparseSpMMOp_bufferSize(
        handle.get(), ACL_SPARSE_OP_NON_TRANSPOSE, opB,
        matA.cget(), matB.cget(), matC.get(),
        computeType, alg, &result.bufferSize);
    if (result.bufferSizeRet != ACL_SPARSE_STATUS_SUCCESS) {
        std::cerr << "[SpMMOp NPU] bufferSize failed: "
                  << result.bufferSizeRet << std::endl;
        return result.bufferSizeRet;
    }

    // --- Allocate workspace (ALG2 requires it) -------------------------------
    if (result.bufferSize > 0) {
        dBuffer = DeviceBuffer::alloc(result.bufferSize);
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

// ============================================================================
// Helper 4b: CreateDescrAndPlan
// Stages 2+3 of the SpMMOp lifecycle: createDescr + createPlan.
// Sets result.createDescrRet / createPlanRet.
// Returns raw descr/plan via out-params so the caller can wrap them in RAII
// guards (SpmmOpDescrGuard / SpmmOpPlanGuard are non-copyable, non-movable).
// ============================================================================

static aclsparseStatus_t CreateDescrAndPlan(
    SpmmOpNpuResult& result, HandleManager& handle,
    aclsparseOperation_t opB, SpMatManager& matA, DnMatManager& matB, DnMatManager& matC,
    aclDataType computeType, aclsparseSpMMOpAlg_t alg, DeviceBuffer& dBuffer,
    aclsparseSpMMOpDescr_t& descr, aclsparseSpMMOpPlan_t& plan)
{
    // --- Stage 2: createDescr ------------------------------------------------
    result.createDescrRet = aclsparseSpMMOp_createDescr(
        handle.get(), &descr,
        ACL_SPARSE_OP_NON_TRANSPOSE, opB,
        matA.cget(), matB.cget(), matC.get(),
        computeType, alg, dBuffer.get());
    if (result.createDescrRet != ACL_SPARSE_STATUS_SUCCESS) {
        std::cerr << "[SpMMOp NPU] createDescr failed: "
                  << result.createDescrRet << std::endl;
        return result.createDescrRet;
    }

    // --- Stage 3: createPlan -------------------------------------------------
    result.createPlanRet = aclsparseSpMMOp_createPlan(
        handle.get(), descr, &plan, nullptr, 0);
    if (result.createPlanRet != ACL_SPARSE_STATUS_SUCCESS) {
        std::cerr << "[SpMMOp NPU] createPlan failed: "
                  << result.createPlanRet << std::endl;
        return result.createPlanRet;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

// ============================================================================
// Helper 4: RunSpmmOpLifecycle
// Executes the 7-function SpMMOp lifecycle:
//   bufferSize -> createDescr -> createPlan -> [setGlobalUserData]
//   -> execute -> sync -> (destroyPlan + destroyDescr via RAII)
// Sets result.bufferSizeRet / createDescrRet / createPlanRet / executeRet
// fields. Allocates dBuffer internally based on result.bufferSize.
// Uses RAII guards (SpmmOpDescrGuard / SpmmOpPlanGuard) for automatic cleanup.
// On sync failure (kernel crash): releases guards to skip destroy calls
// (NPU is in bad state, calling destroy would segfault), sets
// result.executeRet = ACL_SPARSE_STATUS_EXECUTION_FAILED.
// Returns the failing status on any stage failure, or
// ACL_SPARSE_STATUS_SUCCESS on success.
// ============================================================================

static aclsparseStatus_t RunSpmmOpLifecycle(
    SpmmOpNpuResult& result,
    HandleManager& handle, aclrtStream stream,
    aclsparseSpMMOpAlg_t alg, aclDataType computeType,
    aclsparseOperation_t opB,
    SpMatManager& matA, DnMatManager& matB, DnMatManager& matC,
    const void* alphaPtr, const void* betaPtr,
    DeviceBuffer& dBuffer, bool callSetGlobalUserData)
{
    auto st = QueryBufferSizeAndAlloc(result, handle, opB, matA, matB, matC,
        computeType, alg, dBuffer);
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        return st;
    }

    aclsparseSpMMOpDescr_t descr = nullptr;
    aclsparseSpMMOpPlan_t plan = nullptr;
    st = CreateDescrAndPlan(result, handle, opB, matA, matB, matC,
        computeType, alg, dBuffer, descr, plan);
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        return st;
    }
    SpmmOpDescrGuard descrGuard(descr);
    SpmmOpPlanGuard planGuard(plan);

    // --- Stage 4: setGlobalUserData (optional, no-op) ------------------------
    if (callSetGlobalUserData) {
        aclsparseSpMMOp_setGlobalUserData(handle.get(), plan, nullptr, nullptr, 0);
    }

    // --- Stage 5: execute ----------------------------------------------------
    result.executeRet = aclsparseSpMMOp(
        handle.get(), plan, alphaPtr, betaPtr, matB.cget(), matC.get());
    if (result.executeRet != ACL_SPARSE_STATUS_SUCCESS) {
        std::cerr << "[SpMMOp NPU] execute failed: "
                  << result.executeRet << std::endl;
        return result.executeRet;
    }

    // --- Wait for kernel completion ------------------------------------------
    auto aclRet = aclrtSynchronizeStream(stream);
    if (aclRet != ACL_SUCCESS) {
        std::cerr << "[SpMMOp NPU] aclrtSynchronizeStream failed: " << aclRet
                  << " (kernel crash likely, skipping cleanup to avoid segfault)"
                  << std::endl;
        result.executeRet = ACL_SPARSE_STATUS_EXECUTION_FAILED;
        // Release RAII guards to skip destroy calls — NPU is in bad state,
        // calling destroyDescr/destroyPlan would segfault.
        planGuard.release();
        descrGuard.release();
        return ACL_SPARSE_STATUS_EXECUTION_FAILED;
    }

    // Stages 6+7: destroyPlan + destroyDescr handled by RAII guards
    return ACL_SPARSE_STATUS_SUCCESS;
}

// ============================================================================
// Helper 5: ReadBackCOutput
// Reads C output from device, reverses col-major repack if needed, and
// converts to FP64 for golden comparison.
// ============================================================================

template <typename T>
static void ReadBackCOutput(
    DeviceBuffer& dC, int64_t m, int64_t n,
    aclsparseOrder_t orderC, size_t cElemCount,
    std::vector<double>& valuesOut)
{
    if (cElemCount > 0) {
        std::vector<T> hCOut(cElemCount);
        dC.copyToHost(hCOut.data(), cElemCount * sizeof(T));

        // Reverse repack if C is col-major
        if (orderC == ACL_SPARSE_ORDER_COL && m > 0 && n > 0) {
            hCOut = RepackColToRow(hCOut, m, n);
        }

        valuesOut.resize(cElemCount);
        for (size_t i = 0; i < cElemCount; i++) {
            if constexpr (std::is_same_v<T, float>) {
                valuesOut[i] = static_cast<double>(hCOut[i]);
            } else {
                valuesOut[i] = static_cast<double>(Fp16BitsToFp32(hCOut[i]));
            }
        }
    }
}

// ============================================================================
// Helper 6: ExecuteSpmmOpAndReadback
// Runs the alpha/beta setup, SpMMOp lifecycle, and C output readback on
// already-prepared matA/matB/matC descriptors. Extracted from SpmmOpNpu to
// keep the orchestrator under NBNC limits.
// Returns a SpmmOpNpuResult populated with stage return codes and (on
// success) the read-back C output values.
// ============================================================================
template <typename T>
static SpmmOpNpuResult ExecuteSpmmOpAndReadback(
    HandleManager& handle, aclrtStream stream,
    int64_t m, int64_t n,
    aclsparseSpMMOpAlg_t alg, aclDataType computeType,
    aclsparseOperation_t opB, aclsparseOrder_t orderC,
    SpMatManager& matA, DnMatManager& matB, DnMatManager& matC,
    DeviceBuffer& dC,
    aclsparsePointerMode_t pointerMode,
    float alpha, float beta,
    bool callSetGlobalUserData)
{
    SpmmOpNpuResult result;
    float alphaHost;
    float betaHost;
    DeviceBuffer dAlpha;
    DeviceBuffer dBeta;
    const void* alphaPtr = nullptr;
    const void* betaPtr = nullptr;
    auto alphaBetaStatus = PrepareAlphaBeta(
        handle, pointerMode, alpha, beta,
        alphaHost, betaHost, alphaPtr, betaPtr, dAlpha, dBeta);
    if (alphaBetaStatus != ACL_SPARSE_STATUS_SUCCESS) {
        result.executeRet = alphaBetaStatus;
        return result;
    }
    DeviceBuffer dBuffer;
    auto lifecycleStatus = RunSpmmOpLifecycle(
        result, handle, stream, alg, computeType, opB,
        matA, matB, matC, alphaPtr, betaPtr,
        dBuffer, callSetGlobalUserData);
    if (lifecycleStatus != ACL_SPARSE_STATUS_SUCCESS) {
        return result;
    }
    size_t cElemCount = static_cast<size_t>(m) * static_cast<size_t>(n);
    ReadBackCOutput<T>(dC, m, n, orderC, cElemCount, result.valuesOut);
    return result;
}

// ============================================================================
// SpMMOp NPU 7-function lifecycle (RAII-managed, per test plan §2.4).
// Template parameter T:
//   float    -> FP32 path (dtype = ACL_FLOAT)
//   uint16_t -> FP16 path (dtype = ACL_FLOAT16, host stores IEEE-754 bit patterns)
// Lifecycle: bufferSize -> createDescr -> createPlan -> [setGlobalUserData]
//            -> execute -> destroyPlan -> destroyDescr
// matA (CSR) is bound at createDescr stage (const).
// matB/matC (DnMat) are passed at execute stage.
// alpha/beta through pointer_mode (HOST/DEVICE).
// Orchestrator: delegates to PrepareCsrDeviceBuffers + PrepareDnMatDeviceBuffers
// for device buffer/descriptor setup, then ExecuteSpmmOpAndReadback for the
// remaining lifecycle stages (alpha/beta, execute, readback).
// ============================================================================

template <typename T>
inline SpmmOpNpuResult SpmmOpNpu(
    HandleManager& handle, aclrtStream stream,
    int64_t m, int64_t n, int64_t k,
    aclsparseOperation_t opB,
    aclsparseOrder_t orderB, aclsparseOrder_t orderC,
    float alpha, float beta,
    aclDataType dtype, aclDataType computeType,
    aclsparseSpMMOpAlg_t alg,
    aclsparsePointerMode_t pointerMode,
    const std::vector<int32_t>& rowOffsets,
    const std::vector<int32_t>& colIndices,
    const std::vector<T>& hAValues,   // CSR values (nnz)
    const std::vector<T>& hB,         // dense B (canonical row-major)
    const std::vector<T>& hCInit,     // dense C initial (canonical row-major)
    int64_t nnz,
    bool callSetGlobalUserData = false,
    aclsparseIndexType_t rowOffsetType = ACL_SPARSE_INDEX_32I,
    aclsparseIndexBase_t indexBase = ACL_SPARSE_INDEX_BASE_ZERO)
{
    SpmmOpNpuResult result;
    handle.setStream(stream);

    // --- Helper 1: Device buffers for CSR data + matA creation ---------------
    DeviceBuffer dRowOff;
    DeviceBuffer dColInd;
    DeviceBuffer dVals;
    SpMatManager matA;
    PrepareCsrDeviceBuffers<T>(
        m, k, nnz, dtype, rowOffsetType, indexBase,
        rowOffsets, colIndices, hAValues,
        dRowOff, dColInd, dVals, matA);

    // --- Helper 2: Device buffers + descriptors for B and C ------------------
    DnMatManager matB;
    DnMatManager matC;
    DeviceBuffer dB;
    DeviceBuffer dC;
    PrepareDnMatDeviceBuffers<T>(
        m, n, k, opB, orderB, orderC, dtype, hB, hCInit,
        matB, matC, dB, dC);

    // --- Helpers 3-5: alpha/beta, lifecycle, readback ------------------------
    return ExecuteSpmmOpAndReadback<T>(
        handle, stream, m, n, alg, computeType, opB, orderC,
        matA, matB, matC, dC, pointerMode, alpha, beta,
        callSetGlobalUserData);
}

}  // namespace sparse_test

#endif  // TEST_SPMM_OP_SPMM_OP_ARCH35_SPMM_OP_NPU_WRAPPER_H_
