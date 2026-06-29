/**
 * ----------------------------------------------------------------------------------------------------------
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ----------------------------------------------------------------------------------------------------------
 */

#include <iostream>
#include <stdint.h>
#include <stdlib.h>
#include <new>

#include "cann_ops_sparse.h"
#include "cann_ops_sparse_common.h"
#include "aclsparse_descr_internal.h"
#include "spmm.h"
#include "spmm_csr_mat.h"

#ifndef __CCE_AICORE__
#include "tiling/platform/platform_ascendc.h"
#endif

extern "C" void spmm_kernel_launch(
    const void *csrRowOffsets, const void *csrColInd,
    const void *csrValues, const void *matB, void *matC,
    void *workspaceGM, void *tilingGM,
    int32_t dataType, uint32_t blockDim, void *stream);

namespace {

// Outer launch blockDim = AIV core count from CANN platform config
// (e.g. Ascend950PR_9589: vector_core_cnt=64). Falls back to 24 (A2 default)
// if PlatformAscendC is unavailable.
constexpr uint32_t kSpmmBlockDimFallback = 24u;

uint32_t GetSpmmBlockDim()
{
    static uint32_t cached = 0u;
    if (cached != 0u) {
        return cached;
    }
#ifndef __CCE_AICORE__
    auto *plat = platform_ascendc::PlatformAscendCManager::GetInstance();
    if (plat != nullptr) {
        const uint32_t aiv = plat->GetCoreNumAiv();
        if (aiv > 0u) {
            cached = aiv;
            return cached;
        }
    }
#endif
    cached = kSpmmBlockDimFallback;
    return cached;
}

float ScalarReadF32(const void *p) {
    if (p == nullptr) {
        return 0.0f;
    }
    return *static_cast<const float *>(p);
}

int32_t ScalarReadI32(const void *p) {
    if (p == nullptr) {
        return 0;
    }
    return *static_cast<const int32_t *>(p);
}

void SetTilingAlphaBeta(SpmmTilingData *td,
    const void *alpha, const void *beta, aclDataType computeType)
{
    if (computeType == ACL_INT32) {
        td->alpha_host = static_cast<float>(ScalarReadI32(alpha));
        td->beta_host = static_cast<float>(ScalarReadI32(beta));
    } else {
        td->alpha_host = ScalarReadF32(alpha);
        td->beta_host = ScalarReadF32(beta);
    }
}

int32_t ComputeSpmmHighPrecision(aclsparseSpMMAlg_t alg, int32_t dataType)
{
    return (alg == ACL_SPARSE_SPMM_CSR_FP32_HIGH_PRECISION_ALG &&
            dataType == SPMM_DTYPE_FP32) ?
               1 :
               0;
}

struct SpmmSupportedDtypeCombo {
    aclDataType matA;
    aclDataType matB;
    aclDataType matC;
    aclDataType computeType;
};

static bool IsSupportedSpmmDtypeCombo(const aclsparseSpMatDescr *matA,
                                      const aclsparseDnMatDescr *matB,
                                      const aclsparseDnMatDescr *matC,
                                      aclDataType computeType)
{
    static const SpmmSupportedDtypeCombo kCombos[] = {
        {ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT},
        {ACL_FLOAT16, ACL_FLOAT16, ACL_FLOAT16, ACL_FLOAT},
        {ACL_INT8, ACL_INT8, ACL_INT32, ACL_INT32},
    };
    for (const auto &combo : kCombos) {
        if (matA->valueType == combo.matA && matB->valueType == combo.matB &&
            matC->valueType == combo.matC && computeType == combo.computeType) {
            return true;
        }
    }
    return false;
}

static bool SpmmDimensionsMatch(const aclsparseSpMatDescr *matA,
                                const aclsparseDnMatDescr *matB,
                                const aclsparseDnMatDescr *matC)
{
    if (matA->cols != static_cast<uint64_t>(matB->rows)) {
        return false;
    }
    return matA->rows == static_cast<uint64_t>(matC->rows) &&
           matB->cols == matC->cols;
}

static bool IsSupportedSpmmAlg(aclsparseSpMMAlg_t alg)
{
    return alg == ACL_SPARSE_SPMM_ALG_DEFAULT ||
           alg == ACL_SPARSE_SPMM_CSR_ALG1 ||
           alg == ACL_SPARSE_SPMM_CSR_FP32_HIGH_PRECISION_ALG;
}

static aclsparseStatus_t ValidateSpmmOperations(aclsparseOperation_t opA,
                                                aclsparseOperation_t opB)
{
    if (opA != ACL_SPARSE_OP_NON_TRANSPOSE) {
        return ACL_SPARSE_STATUS_NOT_SUPPORTED; // V1 仅 N
    }
    if (opB != ACL_SPARSE_OP_NON_TRANSPOSE && opB != ACL_SPARSE_OP_TRANSPOSE) {
        return ACL_SPARSE_STATUS_NOT_SUPPORTED;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

aclsparseStatus_t ValidateSpmmInputs(const aclsparseSpMatDescr *matA,
                                   const aclsparseDnMatDescr *matB,
                                   const aclsparseDnMatDescr *matC,
                                   aclsparseOperation_t opA,
                                   aclsparseOperation_t opB,
                                   aclDataType computeType,
                                   aclsparseSpMMAlg_t alg)
{
    if (matA == nullptr || matB == nullptr || matC == nullptr) {
        return ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR;
    }
    aclsparseStatus_t st = ValidateSpmmOperations(opA, opB);
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        return st;
    }
    if (matA->format != ACL_SPARSE_FORMAT_CSR) {
        return ACL_SPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED;
    }
    if (!IsSupportedSpmmDtypeCombo(matA, matB, matC, computeType)) {
        return ACL_SPARSE_STATUS_NOT_SUPPORTED;
    }
    if (!SpmmDimensionsMatch(matA, matB, matC)) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (!IsSupportedSpmmAlg(alg)) {
        return ACL_SPARSE_STATUS_NOT_SUPPORTED;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

// Returns workspace start byte offsets for the TilingData / reorder / bin-edge
// regions, computed once and shared by GetBufferSize / Preprocess / Spmm.
struct WsOffsets {
    int64_t tilingOff;
    int64_t reorderOff;
    int64_t binEdgeOff;
    int64_t totalBytes;
};

WsOffsets ComputeWsOffsets(int64_t m, int32_t blockDim) {
    WsOffsets o;
    o.tilingOff  = SPMM_WS_HEADER_BYTES;
    o.reorderOff = spmm_align_up(o.tilingOff + static_cast<int64_t>(sizeof(SpmmTilingData)), SPMM_WS_ALIGN);
    o.binEdgeOff = spmm_align_up(o.reorderOff + static_cast<int64_t>(sizeof(int32_t)) * m, SPMM_WS_ALIGN);
    o.totalBytes = spmm_align_up(o.binEdgeOff + static_cast<int64_t>(sizeof(int32_t)) * (blockDim + 1), SPMM_WS_ALIGN);
    return o;
}

} // namespace

aclsparseStatus_t aclsparseSpMMGetBufferSize(
    aclsparseHandle_t /*handle*/, aclsparseOperation_t opA, aclsparseOperation_t opB,
    const void * /*alpha*/, aclsparseConstSpMatDescr_t matA, aclsparseConstDnMatDescr_t matB,
    const void * /*beta*/, aclsparseDnMatDescr_t matC, aclDataType computeType,
    aclsparseSpMMAlg_t alg, size_t *size)
{
    if (size == nullptr) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    aclsparseSpMatDescr *matAInner = (aclsparseSpMatDescr *)matA;
    aclsparseDnMatDescr *matBInner = (aclsparseDnMatDescr *)matB;
    aclsparseDnMatDescr *matCInner = (aclsparseDnMatDescr *)matC;
    aclsparseStatus_t st = ValidateSpmmInputs(matAInner, matBInner, matCInner,
                                            opA, opB, computeType, alg);
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        return st;
    }
    WsOffsets off = ComputeWsOffsets(static_cast<int64_t>(matAInner->rows),
                                     static_cast<int32_t>(GetSpmmBlockDim()));
    *size = static_cast<size_t>(off.totalBytes);
    return ACL_SPARSE_STATUS_SUCCESS;
}

// 计算 SpMM 的 reorder/bin/tiling 并写入 device workspace buffer。
// 该函数只读 matA，不修改其内部状态（buffer 是否 active 由调用方决定）。
static aclsparseStatus_t SpmmBuildTilingToBuffer(
    aclsparseSpMatDescr *matAInner, aclsparseDnMatDescr *matBInner,
    aclsparseDnMatDescr *matCInner, aclsparseOperation_t opB,
    const void *alpha, const void *beta, aclDataType computeType,
    aclsparseSpMMAlg_t alg, void *buffer)
{
    const int64_t m = static_cast<int64_t>(matAInner->rows);
    WsOffsets off = ComputeWsOffsets(m, static_cast<int32_t>(GetSpmmBlockDim()));

    // 1) Compute reorder table + bin edges, copy into device workspace.
    SpmmCsrMat csr(matAInner, static_cast<int32_t>(GetSpmmBlockDim()));
    uint32_t pre = csr.DoPreProcess(static_cast<uint8_t *>(buffer),
                                    off.reorderOff,
                                    off.binEdgeOff,
                                    /*identity=*/false);
    if (pre != ACL_SPARSE_STATUS_SUCCESS) {
        return static_cast<aclsparseStatus_t>(pre);
    }

    // 2) Build TilingData on host, write to device workspace.
    SpmmTilingData td{};
    td.m   = static_cast<int32_t>(matAInner->rows);
    td.n   = static_cast<int32_t>(matCInner->cols);
    td.ldb = static_cast<int32_t>(matBInner->ld);
    td.ldc = static_cast<int32_t>(matCInner->ld);
    td.reorder_offset  = static_cast<int32_t>(off.reorderOff);
    td.bin_edge_offset = static_cast<int32_t>(off.binEdgeOff);
    td.high_precision  = ComputeSpmmHighPrecision(alg, SpmmDataTypeFromAcl(matAInner->valueType));
    td.opB             = (opB == ACL_SPARSE_OP_TRANSPOSE) ? 1 : 0;
    td.order_pair      = static_cast<int32_t>(matBInner->order) * 2 +
                         static_cast<int32_t>(matCInner->order);
    SetTilingAlphaBeta(&td, alpha, beta, computeType);

    aclError aclRet = aclrtMemcpy(static_cast<uint8_t *>(buffer) + off.tilingOff,
                                  sizeof(SpmmTilingData),
                                  &td, sizeof(SpmmTilingData),
                                  ACL_MEMCPY_HOST_TO_DEVICE);
    if (aclRet != ACL_ERROR_NONE) {
        return ACL_SPARSE_STATUS_EXECUTION_FAILED;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

static aclsparseStatus_t SpmmRefreshActiveTiling(
    void *buffer, const WsOffsets &off,
    aclsparseSpMatDescr *matA, aclsparseDnMatDescr *matB, aclsparseDnMatDescr *matC,
    aclsparseOperation_t opB, const void *alpha, const void *beta,
    aclDataType computeType, aclsparseSpMMAlg_t alg)
{
    SpmmTilingData td;
    aclError ret = aclrtMemcpy(&td, sizeof(SpmmTilingData),
                               static_cast<uint8_t *>(buffer) + off.tilingOff,
                               sizeof(SpmmTilingData),
                               ACL_MEMCPY_DEVICE_TO_HOST);
    if (ret != ACL_ERROR_NONE) {
        return ACL_SPARSE_STATUS_EXECUTION_FAILED;
    }
    td.opB = (opB == ACL_SPARSE_OP_TRANSPOSE) ? 1 : 0;
    td.order_pair = static_cast<int32_t>(matB->order) * 2 +
                    static_cast<int32_t>(matC->order);
    td.high_precision = ComputeSpmmHighPrecision(alg, SpmmDataTypeFromAcl(matA->valueType));
    SetTilingAlphaBeta(&td, alpha, beta, computeType);
    ret = aclrtMemcpy(static_cast<uint8_t *>(buffer) + off.tilingOff,
                      sizeof(SpmmTilingData),
                      &td, sizeof(SpmmTilingData),
                      ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != ACL_ERROR_NONE) {
        return ACL_SPARSE_STATUS_EXECUTION_FAILED;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

static aclsparseStatus_t SpmmEnsureTilingReady(
    aclsparseSpMatDescr *matA, aclsparseDnMatDescr *matB, aclsparseDnMatDescr *matC,
    aclsparseOperation_t opB, const void *alpha, const void *beta,
    aclDataType computeType, aclsparseSpMMAlg_t alg, void *buffer, const WsOffsets &off)
{
    if (matA->activeBuffer == buffer) {
        return SpmmRefreshActiveTiling(buffer, off, matA, matB, matC,
                                       opB, alpha, beta, computeType, alg);
    }
    return SpmmBuildTilingToBuffer(matA, matB, matC, opB,
                                   alpha, beta, computeType, alg, buffer);
}

static aclsparseStatus_t SpmmRunKernel(
    aclsparseSpMatDescr *matA, aclsparseDnMatDescr *matB, aclsparseDnMatDescr *matC,
    void *buffer, const WsOffsets &off, aclrtStream stream)
{
    spmm_kernel_launch(matA->ptrs, matA->idxs, matA->values,
                       matB->values, matC->values, buffer,
                       static_cast<uint8_t *>(buffer) + off.tilingOff,
                       SpmmDataTypeFromAcl(matA->valueType),
                       GetSpmmBlockDim(), stream);
    return ACL_SPARSE_STATUS_SUCCESS;
}

aclsparseStatus_t aclsparseSpMMPreprocess(
    aclsparseHandle_t /*handle*/, aclsparseOperation_t opA, aclsparseOperation_t opB,
    const void *alpha, aclsparseConstSpMatDescr_t matA, aclsparseConstDnMatDescr_t matB,
    const void *beta, aclsparseDnMatDescr_t matC, aclDataType computeType,
    aclsparseSpMMAlg_t alg, void *buffer)
{
    aclsparseSpMatDescr *matAInner = (aclsparseSpMatDescr *)matA;
    aclsparseDnMatDescr *matBInner = (aclsparseDnMatDescr *)matB;
    aclsparseDnMatDescr *matCInner = (aclsparseDnMatDescr *)matC;
    aclsparseStatus_t st = ValidateSpmmInputs(matAInner, matBInner, matCInner,
                                            opA, opB, computeType, alg);
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        return st;
    }
    if (buffer == nullptr) {
        return ACL_SPARSE_STATUS_INSUFFICIENT_RESOURCES;
    }

    st = SpmmBuildTilingToBuffer(matAInner, matBInner, matCInner, opB,
                                 alpha, beta, computeType, alg, buffer);
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        return st;
    }

    // 让这块 buffer 成为 matA 的 active buffer（Preprocess 写 matA 内部状态）。
    matAInner->activeBuffer = buffer;
    return ACL_SPARSE_STATUS_SUCCESS;
}

aclsparseStatus_t aclsparseSpMM(
    aclsparseHandle_t handle, aclsparseOperation_t opA, aclsparseOperation_t opB,
    const void *alpha, aclsparseConstSpMatDescr_t matA, aclsparseConstDnMatDescr_t matB,
    const void *beta, aclsparseDnMatDescr_t matC, aclDataType computeType,
    aclsparseSpMMAlg_t alg, void *buffer)
{
    aclsparseSpMatDescr *matAInner = (aclsparseSpMatDescr *)matA;
    aclsparseDnMatDescr *matBInner = (aclsparseDnMatDescr *)matB;
    aclsparseDnMatDescr *matCInner = (aclsparseDnMatDescr *)matC;
    aclsparseStatus_t st = ValidateSpmmInputs(matAInner, matBInner, matCInner,
                                              opA, opB, computeType, alg);
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        return st;
    }
    if (buffer == nullptr) {
        return ACL_SPARSE_STATUS_INSUFFICIENT_RESOURCES;
    }
    if (handle == nullptr) {
        return ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR;
    }
    aclrtStream stream = nullptr;
    aclsparseGetStream(handle, &stream);
    WsOffsets off = ComputeWsOffsets(static_cast<int64_t>(matAInner->rows),
                                     static_cast<int32_t>(GetSpmmBlockDim()));
    st = SpmmEnsureTilingReady(matAInner, matBInner, matCInner, opB,
                               alpha, beta, computeType, alg, buffer, off);
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        return st;
    }
    return SpmmRunKernel(matAInner, matBInner, matCInner, buffer, off, stream);
}
