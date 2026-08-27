/**
 * ----------------------------------------------------------------------------------------------------------
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ----------------------------------------------------------------------------------------------------------
 */

/*
 * SpGEMM Host-side implementation — SpMM-style 3-stage + 7-interface API.
 *
 * Aligned with ops-sparse aclsparseSpMM convention:
 *   1. GetBufferSize  — validate + query workspace size
 *   2. Preprocess     — symbolic phase: C structure + reorder/binEdge → workspace
 *   3. SpGEMM         — numeric phase: fill C values (reuses structure via activeBuffer)
 *
 * Structure reuse: matC->activeBuffer == buffer => symbolic phase already done,
 * SpGEMM skips to numeric kernel directly (same as SpMM's SpmmEnsureTilingReady).
 *
 * Buffer 策略：
 *   3-stage API uses a single buffer (compatibility path, aligned with SpMM).
 *   7-interface API separates buffer1 (symbolic) and buffer2 (numeric):
 *     - WorkEstimation uses buffer1 for symbolic phase
 *     - Compute uses buffer1 for probe (if needed) + buffer2 for numeric
 *     - Copy uses buffer2 for result staging
 *   Both buffers have the same size (workspace layout is identical), but are
 *   independent allocations — allowing symbolic results in buffer1 to persist
 *   while numeric phase writes to buffer2.
 */

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>

#include "acl/acl.h"
#include "cann_ops_sparse.h"
#include "spgemm.h"
#include "aclsparse_host_utils.h"
#include "aclsparse_descr_internal.h"
#include "aclsparse_handle_internal.h"
#include "spgemm_csr_mat.h"

#ifndef __CCE_AICORE__
#include "tiling/platform/platform_ascendc.h"
#endif

namespace {

/* ops-sparse 风格：复用 aclsparse_host_utils.h 的 GetAivCoreCount()，fallback 64 (950PR). */
constexpr uint32_t kSpgemmBlockDimFallback = 64u;

uint32_t GetSpgemmBlockDim()
{
    uint32_t aiv = GetAivCoreCount();
    return (aiv > 0u) ? aiv : kSpgemmBlockDimFallback;
}

/* 使用 aclsparseGetStream/aclsparseGetPointerMode API 获取流和指针模式。
 * ToMatInnerConst/ToMatInner 用于获取 matA/matB/matC 内部结构。 */
inline struct aclsparseSpMatDescr *ToMatInnerConst(aclsparseConstSpMatDescr_t desc) {
    return const_cast<struct aclsparseSpMatDescr *>(
        reinterpret_cast<const struct aclsparseSpMatDescr *>(desc));
}
inline struct aclsparseSpMatDescr *ToMatInner(aclsparseSpMatDescr_t desc) {
    return reinterpret_cast<struct aclsparseSpMatDescr *>(desc);
}

int32_t GetComputeDtypeSize(aclDataType computeType) {
    switch (computeType) {
        case ACL_FLOAT:   return 4;
        case ACL_FLOAT16: return 2;
        case ACL_BF16:    return 2;
        default:          return 4;
    }
}

float ScalarReadF32(const void *p) {
    if (p == nullptr) return 0.0f;
    return *static_cast<const float *>(p);
}

/* Read beta scalar with pointer-mode awareness: HOST → direct deref,
 * DEVICE → D2H copy (avoids host segfault on device pointer). */
static aclsparseStatus_t SpgemmReadBetaScalar(
    aclsparseHandle_t handle, const void *beta, float &out) {
    aclsparsePointerMode_t pointerMode = ACL_SPARSE_POINTER_MODE_HOST;
    aclsparseGetPointerMode(handle, &pointerMode);
    if (pointerMode == ACL_SPARSE_POINTER_MODE_HOST) {
        out = ScalarReadF32(beta);
    } else {
        aclrtStream stream = nullptr;
        aclsparseGetStream(handle, &stream);
        aclError pmRet = aclrtMemcpy(&out, sizeof(float), beta,
                                     sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
        if (pmRet != ACL_ERROR_NONE) return ACL_SPARSE_STATUS_EXECUTION_FAILED;
        aclrtSynchronizeStream(stream);
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

/* Zero matC values when C_in is not valid; preserve when cInDataValid && β≠0. */
static aclsparseStatus_t SpgemmZeroOrPreserveCIn(
    aclsparseHandle_t handle, aclsparseSpMatDescr *matCInner,
    aclDataType computeType, float betaVal,
    int32_t cInDataValid, int64_t nnzC) {
    if (betaVal != 0.0f && cInDataValid == 1 && matCInner->values != nullptr) {
        if (static_cast<int64_t>(matCInner->nnz) != nnzC) {
            OP_LOGE("spgemm", "β≠0 + C_in: matC->nnz=%lu != nnzC=%ld",
                    (unsigned long)matCInner->nnz, nnzC);
            return ACL_SPARSE_STATUS_INVALID_VALUE;
        }
        return ACL_SPARSE_STATUS_SUCCESS;
    }
    if (matCInner->values != nullptr && matCInner->nnz > 0) {
        int32_t dtypeSize = GetComputeDtypeSize(computeType);
        int64_t bytes = static_cast<int64_t>(matCInner->nnz) * dtypeSize;
        aclrtStream stream = nullptr;
        aclsparseGetStream(handle, &stream);
        aclError zeroRet = aclrtMemsetAsync(matCInner->values,
                                            static_cast<size_t>(bytes), 0,
                                            static_cast<size_t>(bytes), stream);
        if (zeroRet != ACL_ERROR_NONE) return ACL_SPARSE_STATUS_EXECUTION_FAILED;
        aclrtSynchronizeStream(stream);
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

static bool IsSupportedDtype(aclDataType dt) {
    return dt == ACL_FLOAT || dt == ACL_FLOAT16 || dt == ACL_BF16;
}

static bool IsSupportedSpgemmAlg(aclsparseSpGEMMAlg_t alg)
{
    return alg == ACL_SPARSE_SPGEMM_ALG_DEFAULT ||
           alg == ACL_SPARSE_SPGEMM_ALG1;
}

/* ---- 输入校验子函数 ---- */

static aclsparseStatus_t ValidateSpMatPointers(
    const aclsparseSpMatDescr *matA, const aclsparseSpMatDescr *matB,
    const aclsparseSpMatDescr *matC)
{
    if (matA == nullptr || matB == nullptr || matC == nullptr) {
        OP_LOGE("spgemm", "matA/matB/matC is nullptr");
        return ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

static aclsparseStatus_t ValidateSpMatFormat(
    const aclsparseSpMatDescr *matA, const aclsparseSpMatDescr *matB,
    const aclsparseSpMatDescr *matC, aclsparseOperation_t opA,
    aclsparseOperation_t opB)
{
    if (opA != ACL_SPARSE_OP_NON_TRANSPOSE) {
        OP_LOGE("spgemm", "opA must be NON_TRANSPOSE");
        return ACL_SPARSE_STATUS_NOT_SUPPORTED;
    }
    if (opB != ACL_SPARSE_OP_NON_TRANSPOSE) {
        OP_LOGE("spgemm", "opB must be NON_TRANSPOSE");
        return ACL_SPARSE_STATUS_NOT_SUPPORTED;
    }
    if (matA->format != ACL_SPARSE_FORMAT_CSR ||
        matB->format != ACL_SPARSE_FORMAT_CSR ||
        matC->format != ACL_SPARSE_FORMAT_CSR) {
        OP_LOGE("spgemm", "only CSR format is supported");
        return ACL_SPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

static aclsparseStatus_t ValidateSpMatDtype(
    const aclsparseSpMatDescr *matA, const aclsparseSpMatDescr *matB,
    const aclsparseSpMatDescr *matC, aclDataType computeType,
    aclsparseSpGEMMAlg_t alg)
{
    aclsparseStatus_t idxSt = AclsparseValidateSupportedCsrIndexTypes(matA->ptrType, matA->IdxType);
    if (idxSt != ACL_SPARSE_STATUS_SUCCESS) return idxSt;
    idxSt = AclsparseValidateSupportedCsrIndexTypes(matB->ptrType, matB->IdxType);
    if (idxSt != ACL_SPARSE_STATUS_SUCCESS) return idxSt;
    idxSt = AclsparseValidateSupportedCsrIndexTypes(matC->ptrType, matC->IdxType);
    if (idxSt != ACL_SPARSE_STATUS_SUCCESS) return idxSt;

    if (matA->baseType != ACL_SPARSE_INDEX_BASE_ZERO ||
        matB->baseType != ACL_SPARSE_INDEX_BASE_ZERO ||
        matC->baseType != ACL_SPARSE_INDEX_BASE_ZERO) {
        OP_LOGE("spgemm", "only zero-based index is supported");
        return ACL_SPARSE_STATUS_NOT_SUPPORTED;
    }
    if (!IsSupportedDtype(matA->valueType) || !IsSupportedDtype(matB->valueType) ||
        !IsSupportedDtype(matC->valueType)) {
        OP_LOGE("spgemm", "unsupported value dtype");
        return ACL_SPARSE_STATUS_NOT_SUPPORTED;
    }
    if (matA->valueType != matB->valueType || matA->valueType != matC->valueType ||
        matA->valueType != computeType) {
        OP_LOGE("spgemm", "A/B/C/computeType must be the same dtype");
        return ACL_SPARSE_STATUS_NOT_SUPPORTED;
    }
    if (!IsSupportedSpgemmAlg(alg)) {
        OP_LOGE("spgemm", "unsupported alg");
        return ACL_SPARSE_STATUS_NOT_SUPPORTED;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

static aclsparseStatus_t ValidateSpMatDimensions(
    const aclsparseSpMatDescr *matA, const aclsparseSpMatDescr *matB,
    const aclsparseSpMatDescr *matC)
{
    if (matA->cols != matB->rows) {
        OP_LOGE("spgemm", "dimension mismatch: A.cols=%lu, B.rows=%lu",
                (unsigned long)matA->cols, (unsigned long)matB->rows);
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (matA->rows != matC->rows || matB->cols != matC->cols) {
        OP_LOGE("spgemm", "dimension mismatch: A.rows=%lu C.rows=%lu, B.cols=%lu C.cols=%lu",
                (unsigned long)matA->rows, (unsigned long)matC->rows,
                (unsigned long)matB->cols, (unsigned long)matC->cols);
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (matA->rows > static_cast<uint64_t>(INT32_MAX) ||
        matA->cols > static_cast<uint64_t>(INT32_MAX) ||
        matB->cols > static_cast<uint64_t>(INT32_MAX)) {
        OP_LOGE("spgemm", "matrix dimensions exceed INT32_MAX");
        return ACL_SPARSE_STATUS_NOT_SUPPORTED;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

static aclsparseStatus_t ValidateSpgemmInputs(
    const aclsparseSpMatDescr *matA,
    const aclsparseSpMatDescr *matB,
    const aclsparseSpMatDescr *matC,
    aclsparseOperation_t opA,
    aclsparseOperation_t opB,
    aclDataType computeType,
    aclsparseSpGEMMAlg_t alg)
{
    aclsparseStatus_t st = ValidateSpMatPointers(matA, matB, matC);
    if (st != ACL_SPARSE_STATUS_SUCCESS) return st;
    st = ValidateSpMatFormat(matA, matB, matC, opA, opB);
    if (st != ACL_SPARSE_STATUS_SUCCESS) return st;
    st = ValidateSpMatDtype(matA, matB, matC, computeType, alg);
    if (st != ACL_SPARSE_STATUS_SUCCESS) return st;
    st = ValidateSpMatDimensions(matA, matB, matC);
    if (st != ACL_SPARSE_STATUS_SUCCESS) return st;
    /* β≠0 时 cuSPARSE 要求 C_in 结构与结果一致。
     * 由于结果结构在符号阶段前未知，这里只校验 matC 的预分配容量 (nnz) 足够大。
     * β≠0 且 matC->nnz==0 时，C_in 为空 -> β·C_in=0，等价 beta=0，允许通过。
     * β≠0 且 matC->nnz>0 时，要求用户已按正确结构填充 C_in；符号阶段后会做
     *   nnzC == matC->nnz 的精确校验 (见 SpGEMM numeric phase). */
    return ACL_SPARSE_STATUS_SUCCESS;
}

/* ---- 公共启动上下文 ---- */
struct SpgemmLaunchCtx {
    int32_t m, n, k;
    uint32_t blockDim;
    int32_t computeDtypeSize;
    SpgemmWsOffsets off;
    aclrtStream stream;
    aclsparseSpMatDescr *matAInner;
    aclsparseSpMatDescr *matBInner;
    aclsparseSpMatDescr *matCInner;
};

/* SpgemmPrepareLaunch: 校验输入 + 提取维度 + 计算 blockDim + workspace 偏移。
 * GetBufferSize、Preprocess、SpGEMM、WorkEstimation、Compute 共用。 */
static aclsparseStatus_t SpgemmPrepareLaunch(
    aclsparseHandle_t handle,
    aclsparseConstSpMatDescr_t matA, aclsparseConstSpMatDescr_t matB,
    aclsparseSpMatDescr_t matC,
    aclsparseOperation_t opA, aclsparseOperation_t opB,
    aclDataType computeType, aclsparseSpGEMMAlg_t alg,
    SpgemmLaunchCtx &ctx)
{
    ctx.matAInner = ToMatInnerConst(matA);
    ctx.matBInner = ToMatInnerConst(matB);
    ctx.matCInner = ToMatInner(matC);
    aclsparseStatus_t st = ValidateSpgemmInputs(ctx.matAInner, ctx.matBInner, ctx.matCInner,
                                                 opA, opB, computeType, alg);
    if (st != ACL_SPARSE_STATUS_SUCCESS) return st;

    /* 通过 aclsparseGetStream API 获取流，避免直接解引用内部结构 */
    ctx.stream = nullptr;
    if (handle != nullptr) {
        aclsparseGetStream(handle, &ctx.stream);
    }
    ctx.m = static_cast<int32_t>(ctx.matAInner->rows);
    ctx.n = static_cast<int32_t>(ctx.matBInner->cols);
    ctx.k = static_cast<int32_t>(ctx.matAInner->cols);
    ctx.blockDim = GetSpgemmBlockDim();
    if (ctx.blockDim > static_cast<uint32_t>(ctx.m) && ctx.m > 0) {
        ctx.blockDim = static_cast<uint32_t>(ctx.m);
    }
    if (ctx.blockDim < 1u) ctx.blockDim = 1u;

    ctx.computeDtypeSize = GetComputeDtypeSize(computeType);
    ctx.off = ComputeSpgemmWsOffsets(ctx.m, ctx.n, ctx.k,
                                      static_cast<int64_t>(ctx.matAInner->nnz),
                                      static_cast<int32_t>(ctx.blockDim),
                                      ctx.computeDtypeSize);
    return ACL_SPARSE_STATUS_SUCCESS;
}

/* ---- Preprocess 各阶段子函数 ---- */

/* SpgemmPreprocessHostData: 拷贝 B rowPtr + 校验排序 + D2H 拷贝 + 行权重 + bin-pack。 */
static aclsparseStatus_t SpgemmPreprocessHostData(
    aclsparseSpMatDescr *matAInner, aclsparseSpMatDescr *matBInner,
    int32_t m, int32_t k, uint32_t blockDim,
    void *buffer, const SpgemmWsOffsets &off, aclrtStream stream)
{
    /* Copy B rowPtr to workspace */
    aclsparseStatus_t bRet = SpgemmCopyBRowPtrToWorkspace(matBInner, buffer, off, stream);
    if (bRet != ACL_SPARSE_STATUS_SUCCESS) return bRet;

    /* 校验 A/B 列索引是否有序 */
    aclsparseStatus_t sortSt = SpgemmValidateCsrSorted(matAInner, "matA");
    if (sortSt != ACL_SPARSE_STATUS_SUCCESS) return sortSt;
    sortSt = SpgemmValidateCsrSorted(matBInner, "matB");
    if (sortSt != ACL_SPARSE_STATUS_SUCCESS) return sortSt;

    /* D2H copy A rowPtr + A colInd + B rowPtr for host-side weight computation */
    std::vector<int32_t> aRowPtrHost(m + 1);
    aclError aclRet = aclrtMemcpy(aRowPtrHost.data(), (m + 1) * sizeof(int32_t),
                           matAInner->ptrs, (m + 1) * sizeof(int32_t),
                           ACL_MEMCPY_DEVICE_TO_HOST);
    if (aclRet != ACL_ERROR_NONE) {
        OP_LOGE("spgemm", "D2H copy A rowPtr failed, ret=%d", aclRet);
        return ACL_SPARSE_STATUS_EXECUTION_FAILED;
    }

    std::vector<int32_t> aColIndHost(matAInner->nnz);
    aclRet = aclrtMemcpy(aColIndHost.data(), matAInner->nnz * sizeof(int32_t),
                          matAInner->idxs, matAInner->nnz * sizeof(int32_t),
                          ACL_MEMCPY_DEVICE_TO_HOST);
    if (aclRet != ACL_ERROR_NONE) {
        OP_LOGE("spgemm", "D2H copy A colInd failed, ret=%d", aclRet);
        return ACL_SPARSE_STATUS_EXECUTION_FAILED;
    }

    std::vector<int32_t> bRowPtrHost(k + 1);
    aclRet = aclrtMemcpy(bRowPtrHost.data(), (k + 1) * sizeof(int32_t),
                          matBInner->ptrs, (k + 1) * sizeof(int32_t),
                          ACL_MEMCPY_DEVICE_TO_HOST);
    if (aclRet != ACL_ERROR_NONE) {
        OP_LOGE("spgemm", "D2H copy B rowPtr failed, ret=%d", aclRet);
        return ACL_SPARSE_STATUS_EXECUTION_FAILED;
    }

    /* Compute row weights + greedy bin-packing */
    std::vector<int32_t> rowWeights(m, 0);
    SpgemmComputeRowWeights(aRowPtrHost, aColIndHost, bRowPtrHost, rowWeights);
    SpgemmRowBinPackResult binPack;
    SpgemmGreedyRowBinPack(rowWeights, static_cast<int32_t>(blockDim), binPack);

    bRet = SpgemmWriteReorderToWorkspace(binPack, buffer, off, stream);
    if (bRet != ACL_SPARSE_STATUS_SUCCESS) return bRet;

    return ACL_SPARSE_STATUS_SUCCESS;
}

/* SpgemmPreprocessRunKernels: 符号阶段 kernel + prefixsum + nnzC 回读 + 容量校验。 */
static aclsparseStatus_t SpgemmPreprocessRunKernels(
    aclsparseSpMatDescr *matAInner, aclsparseSpMatDescr *matBInner,
    aclsparseSpMatDescr *matCInner,
    int32_t m, int32_t n, uint32_t blockDim,
    void *buffer, const SpgemmWsOffsets &off, aclrtStream stream)
{
    /* Zero nnzPerRow before symbolic kernel writes to it */
    int64_t nnzPerRowBytes = m * sizeof(int32_t);
    aclError zeroRet = aclrtMemsetAsync(
        static_cast<uint8_t *>(buffer) + off.nnzPerRowOff,
        nnzPerRowBytes, 0, nnzPerRowBytes, stream);
    if (zeroRet != ACL_ERROR_NONE) {
        OP_LOGE("spgemm", "aclrtMemsetAsync for nnzPerRow failed, ret=%d", zeroRet);
        return ACL_SPARSE_STATUS_EXECUTION_FAILED;
    }

    /* Sync: ensure H2D copies + D2D visible to kernel */
    aclError aclRet = aclrtSynchronizeStream(stream);
    if (aclRet != ACL_ERROR_NONE) {
        OP_LOGE("spgemm", "Sync before symbolic kernel failed, ret=%d", aclRet);
        return ACL_SPARSE_STATUS_EXECUTION_FAILED;
    }

    /* --- Symbolic kernel: count nnzPerRow --- */
    SpgemmSymbolicTilingData symTiling{};
    symTiling.m = m;
    symTiling.n = n;
    symTiling.blockDim = static_cast<int32_t>(blockDim);
    symTiling.baseA = 0;
    symTiling.baseB = 0;
    symTiling.baseC = 0;
    symTiling.reorderOffset = static_cast<int32_t>(off.reorderOff);
    symTiling.binEdgeOffset = static_cast<int32_t>(off.binEdgeOff);
    symTiling.symBitmapOffset = off.gmAccumOff;

    GM_ADDR symTilingGM = reinterpret_cast<GM_ADDR>(
        static_cast<uint8_t *>(buffer) + off.tilingOff);
    aclrtMemcpy(reinterpret_cast<void *>(symTilingGM), sizeof(symTiling),
                &symTiling, sizeof(symTiling), ACL_MEMCPY_HOST_TO_DEVICE);

    spgemm_symbolic_kernel_do(
        reinterpret_cast<GM_ADDR>(matAInner->ptrs),
        reinterpret_cast<GM_ADDR>(matAInner->idxs),
        reinterpret_cast<GM_ADDR>(static_cast<uint8_t *>(buffer) + off.bRowPtrOff),
        reinterpret_cast<GM_ADDR>(matBInner->idxs),
        reinterpret_cast<GM_ADDR>(static_cast<uint8_t *>(buffer) + off.nnzPerRowOff),
        reinterpret_cast<GM_ADDR>(buffer),
        symTilingGM, blockDim, stream);

    /* Sync: symbolic kernel must finish before prefixsum reads nnzPerRow */
    aclRet = aclrtSynchronizeStream(stream);
    if (aclRet != ACL_ERROR_NONE) {
        OP_LOGE("spgemm", "Sync after symbolic kernel failed, ret=%d", aclRet);
        return ACL_SPARSE_STATUS_EXECUTION_FAILED;
    }

    /* --- Prefix sum to build rowPtrC --- */
    SpgemmPrefixSumTilingData psTiling{};
    psTiling.m = m;
    psTiling.baseC = 0;

    GM_ADDR psTilingGM = reinterpret_cast<GM_ADDR>(
        static_cast<uint8_t *>(buffer) + off.tilingOff);
    aclrtMemcpy(reinterpret_cast<void *>(psTilingGM), sizeof(psTiling),
                &psTiling, sizeof(psTiling), ACL_MEMCPY_HOST_TO_DEVICE);

    int32_t *nnzCDev = reinterpret_cast<int32_t *>(
        static_cast<uint8_t *>(buffer) + off.rowPtrCOff) + (m + 1);

    spgemm_prefixsum_kernel_do(
        reinterpret_cast<GM_ADDR>(static_cast<uint8_t *>(buffer) + off.nnzPerRowOff),
        reinterpret_cast<GM_ADDR>(matCInner->ptrs),
        reinterpret_cast<GM_ADDR>(nnzCDev),
        psTilingGM, stream);

    /* Sync: prefixsum must complete before D2H read of nnzC */
    aclRet = aclrtSynchronizeStream(stream);
    if (aclRet != ACL_ERROR_NONE) {
        OP_LOGE("spgemm", "Sync after prefixsum kernel failed, ret=%d", aclRet);
        return ACL_SPARSE_STATUS_EXECUTION_FAILED;
    }

    /* D2H read nnzC */
    int32_t nnzCHost = 0;
    aclRet = aclrtMemcpy(&nnzCHost, sizeof(int32_t), nnzCDev, sizeof(int32_t),
                         ACL_MEMCPY_DEVICE_TO_HOST);
    if (aclRet != ACL_ERROR_NONE) {
        OP_LOGE("spgemm", "D2H copy nnzC failed, ret=%d", aclRet);
        return ACL_SPARSE_STATUS_EXECUTION_FAILED;
    }

    /* Capacity check */
    int64_t maxNnzC = static_cast<int64_t>(m) * static_cast<int64_t>(n);
    if (maxNnzC > 500000000) maxNnzC = 500000000;
    if (maxNnzC < 1024) maxNnzC = 1024;

    if (static_cast<int64_t>(nnzCHost) > maxNnzC) {
        OP_LOGE("spgemm", "nnzC=%d exceeds workspace capacity maxNnzC=%ld", nnzCHost, maxNnzC);
        return ACL_SPARSE_STATUS_INSUFFICIENT_RESOURCES;
    }

    matCInner->nnz = static_cast<uint64_t>(nnzCHost);
    return ACL_SPARSE_STATUS_SUCCESS;
}

} // namespace

extern "C" {

/* ===== Stage 1: GetBufferSize — validate + query workspace size ===== */

aclsparseStatus_t aclsparseSpGEMMGetBufferSize(
    aclsparseHandle_t handle,
    aclsparseOperation_t opA, aclsparseOperation_t opB,
    const void * /*alpha*/,
    aclsparseConstSpMatDescr_t matA,
    aclsparseConstSpMatDescr_t matB,
    const void * /*beta*/,
    aclsparseSpMatDescr_t matC,
    aclDataType computeType,
    aclsparseSpGEMMAlg_t alg,
    size_t *size)
{
    if (size == nullptr) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    SpgemmLaunchCtx ctx;
    aclsparseStatus_t st = SpgemmPrepareLaunch(handle, matA, matB, matC,
                                                 opA, opB, computeType, alg, ctx);
    if (st != ACL_SPARSE_STATUS_SUCCESS) return st;
    *size = static_cast<size_t>(ctx.off.totalBytes);
    return ACL_SPARSE_STATUS_SUCCESS;
}

/* ===== Stage 2: Preprocess — symbolic phase ===== */
/* Computes C structure (rowPtrC, nnzC) + reorder/binEdge, writes to workspace.
 * Sets matC->activeBuffer = buffer for structure reuse detection. */

aclsparseStatus_t aclsparseSpGEMMPreprocess(
    aclsparseHandle_t handle,
    aclsparseOperation_t opA, aclsparseOperation_t opB,
    const void * /*alpha*/,
    aclsparseConstSpMatDescr_t matA,
    aclsparseConstSpMatDescr_t matB,
    const void * /*beta*/,
    aclsparseSpMatDescr_t matC,
    aclDataType computeType,
    aclsparseSpGEMMAlg_t alg,
    void *buffer)
{
    if (handle == nullptr) return ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR;
    if (buffer == nullptr) return ACL_SPARSE_STATUS_INSUFFICIENT_RESOURCES;

    SpgemmLaunchCtx ctx;
    aclsparseStatus_t st = SpgemmPrepareLaunch(handle, matA, matB, matC,
                                                 opA, opB, computeType, alg, ctx);
    if (st != ACL_SPARSE_STATUS_SUCCESS) return st;

    aclsparseSpMatDescr *matAInner = ctx.matAInner;
    aclsparseSpMatDescr *matBInner = ctx.matBInner;
    aclsparseSpMatDescr *matCInner = ctx.matCInner;
    int32_t m = ctx.m;
    int32_t n = ctx.n;
    uint32_t blockDim = ctx.blockDim;
    aclrtStream stream = ctx.stream;
    SpgemmWsOffsets off = ctx.off;

    /* Zero workspace buffer */
    aclError aclRet = aclrtMemsetAsync(buffer, static_cast<size_t>(off.totalBytes), 0,
                                       static_cast<size_t>(off.totalBytes), stream);
    if (aclRet != ACL_ERROR_NONE) {
        OP_LOGE("spgemm", "aclrtMemsetAsync failed, ret=%d", aclRet);
        return ACL_SPARSE_STATUS_EXECUTION_FAILED;
    }

    /* Empty matrix fast path */
    if (matAInner->nnz == 0 || matBInner->nnz == 0) {
        SpgemmFillTilingData fillTiling{};
        fillTiling.count = m + 1;
        fillTiling.value = 0;
        GM_ADDR fillTilingGM = reinterpret_cast<GM_ADDR>(
            static_cast<uint8_t *>(buffer) + off.tilingOff);
        aclrtMemcpy(reinterpret_cast<void *>(fillTilingGM), sizeof(fillTiling),
                    &fillTiling, sizeof(fillTiling), ACL_MEMCPY_HOST_TO_DEVICE);
        spgemm_fill_kernel_do(
            reinterpret_cast<GM_ADDR>(matCInner->ptrs), fillTilingGM, stream);
        matCInner->nnz = 0;
        matCInner->activeBuffer = buffer;
        return ACL_SPARSE_STATUS_SUCCESS;
    }

    /* Stage 1: host-side data preparation (copy B + validate + D2H + weights + bin-pack) */
    st = SpgemmPreprocessHostData(matAInner, matBInner, m, ctx.k, blockDim, buffer, off, stream);
    if (st != ACL_SPARSE_STATUS_SUCCESS) return st;

    /* Stage 2: symbolic + prefixsum kernels + nnzC readback */
    st = SpgemmPreprocessRunKernels(matAInner, matBInner, matCInner,
                                     m, n, blockDim, buffer, off, stream);
    if (st != ACL_SPARSE_STATUS_SUCCESS) return st;

    /* Mark structure as computed in this buffer (SpMM-style activeBuffer) */
    matCInner->activeBuffer = buffer;
    return ACL_SPARSE_STATUS_SUCCESS;
}

/* ===== Stage 3: SpGEMM — numeric phase ===== */
/* Fills C values based on structure from Preprocess.
 * If matC->activeBuffer == buffer, structure is reused (skip symbolic).
 * Otherwise, auto-runs Preprocess first (full symbolic+numeric). */

aclsparseStatus_t aclsparseSpGEMM(
    aclsparseHandle_t handle,
    aclsparseOperation_t opA, aclsparseOperation_t opB,
    const void *alpha,
    aclsparseConstSpMatDescr_t matA,
    aclsparseConstSpMatDescr_t matB,
    const void *beta,
    aclsparseSpMatDescr_t matC,
    aclDataType computeType,
    aclsparseSpGEMMAlg_t alg,
    void *buffer)
{
    if (handle == nullptr) return ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR;
    if (buffer == nullptr) return ACL_SPARSE_STATUS_INSUFFICIENT_RESOURCES;

    SpgemmLaunchCtx ctx;
    aclsparseStatus_t st = SpgemmPrepareLaunch(handle, matA, matB, matC,
                                                 opA, opB, computeType, alg, ctx);
    if (st != ACL_SPARSE_STATUS_SUCCESS) return st;

    aclsparseSpMatDescr *matAInner = ctx.matAInner;
    aclsparseSpMatDescr *matBInner = ctx.matBInner;
    aclsparseSpMatDescr *matCInner = ctx.matCInner;
    int32_t m = ctx.m;
    int32_t n = ctx.n;
    uint32_t blockDim = ctx.blockDim;
    aclrtStream stream = ctx.stream;
    SpgemmWsOffsets off = ctx.off;
    int32_t computeDtypeSize = ctx.computeDtypeSize;
    int32_t dataType = SpgemmDataTypeFromAcl(computeType);

    /* Structure reuse: if Preprocess was called with this buffer, skip symbolic.
     * Otherwise, run Preprocess inline (auto-preprocess for single-call convenience). */
    if (matCInner->activeBuffer != buffer) {
        st = aclsparseSpGEMMPreprocess(handle, opA, opB, alpha, matA, matB, beta, matC,
                                       computeType, alg, buffer);
        if (st != ACL_SPARSE_STATUS_SUCCESS) return st;
    }

    /* Empty matrix: numeric phase is no-op (rowPtrC already zero-filled by Preprocess) */
    if (matAInner->nnz == 0 || matBInner->nnz == 0 || matCInner->nnz == 0) {
        return ACL_SPARSE_STATUS_SUCCESS;
    }

    /* β≠0 + cInDataValid==1 路径：用户已填充 C_in 数据，保留不清零。
     * cInDataValid==0 或 β=0：清零 valuesC 避免 kernel 读取未初始化数据。
     * 由 aclsparseSpGEMMCompute 传入 matCInner->cInDataValid；3 阶段直调时默认 0。 */
    if (matCInner->values != nullptr && matCInner->nnz > 0 &&
        matCInner->cInDataValid == 0) {
        int64_t valuesBytes = static_cast<int64_t>(matCInner->nnz) * computeDtypeSize;
        aclError zeroRet = aclrtMemsetAsync(matCInner->values,
                                            static_cast<size_t>(valuesBytes),
                                            0,
                                            static_cast<size_t>(valuesBytes),
                                            stream);
        if (zeroRet != ACL_ERROR_NONE) {
            OP_LOGE("spgemm", "aclrtMemsetAsync for valuesC failed, ret=%d", zeroRet);
            return ACL_SPARSE_STATUS_EXECUTION_FAILED;
        }
        /* Sync: memset must complete before numeric kernel reads valuesC */
        aclError syncRet = aclrtSynchronizeStream(stream);
        if (syncRet != ACL_ERROR_NONE) {
            OP_LOGE("spgemm", "Sync after valuesC memset failed, ret=%d", syncRet);
            return ACL_SPARSE_STATUS_EXECUTION_FAILED;
        }
    }

    /* --- Numeric kernel: fill colIdxC + valuesC --- */
    SpgemmNumericTilingData numTiling{};
    numTiling.m = m;
    numTiling.n = n;
    numTiling.blockDim = static_cast<int32_t>(blockDim);
    numTiling.baseA = 0;
    numTiling.baseB = 0;
    numTiling.baseC = 0;
    numTiling.reorderOffset = static_cast<int32_t>(off.reorderOff);
    numTiling.binEdgeOffset = static_cast<int32_t>(off.binEdgeOff);
    numTiling.rowPtrCOffset = static_cast<int32_t>(off.rowPtrCOff);
    numTiling.nnzPerRowOffset = static_cast<int32_t>(off.nnzPerRowOff);
    numTiling.accumOffset = off.gmAccumOff;       /* legacy field, same as numAccumOffset */
    numTiling.numAccumOffset = off.gmAccumOff;     /* n>64 时 GM-backed 每 block 累加器 */
    numTiling.symBitmapOffset = off.gmAccumOff;

    /* 对齐 SpMM 风格，用 aclsparseGetPointerMode API 获取指针模式 */
    aclsparsePointerMode_t pointerMode = ACL_SPARSE_POINTER_MODE_HOST;
    aclsparseGetPointerMode(handle, &pointerMode);
    if (pointerMode == ACL_SPARSE_POINTER_MODE_HOST) {
        numTiling.alphaHost = ScalarReadF32(alpha);
        numTiling.betaHost  = ScalarReadF32(beta);
        numTiling.alphaPtr = 0;
        numTiling.betaPtr  = 0;
    } else {
        numTiling.alphaPtr = reinterpret_cast<uint64_t>(alpha);
        numTiling.betaPtr  = reinterpret_cast<uint64_t>(beta);
        numTiling.alphaHost = 0.0f;
        numTiling.betaHost  = 0.0f;
    }

    /* Fix: tiling 通过 GM 传递（对齐 SpMM 风格） */
    GM_ADDR numTilingGM = reinterpret_cast<GM_ADDR>(
        static_cast<uint8_t *>(buffer) + off.tilingOff);
    aclrtMemcpy(reinterpret_cast<void *>(numTilingGM), sizeof(numTiling),
                &numTiling, sizeof(numTiling), ACL_MEMCPY_HOST_TO_DEVICE);

    spgemm_numeric_kernel_do(
        reinterpret_cast<GM_ADDR>(matAInner->ptrs),
        reinterpret_cast<GM_ADDR>(matAInner->idxs),
        reinterpret_cast<GM_ADDR>(matAInner->values),
        reinterpret_cast<GM_ADDR>(static_cast<uint8_t *>(buffer) + off.bRowPtrOff),
        reinterpret_cast<GM_ADDR>(matBInner->idxs),
        reinterpret_cast<GM_ADDR>(matBInner->values),
        reinterpret_cast<GM_ADDR>(matCInner->ptrs),
        reinterpret_cast<GM_ADDR>(matCInner->idxs),
        reinterpret_cast<GM_ADDR>(matCInner->values),
        reinterpret_cast<GM_ADDR>(static_cast<uint8_t *>(buffer) + off.nnzPerRowOff),
        reinterpret_cast<GM_ADDR>(buffer),
        numTilingGM, dataType, blockDim, stream);

    return ACL_SPARSE_STATUS_SUCCESS;
}

/* 7 接口完整 API
 *
 * 生命周期：CreateDescr → WorkEstimation → [EstimateMemory] → Compute
 *           → GetNumProducts → Copy → DestroyDescr
 *
 * Compute 实现"调两次"模式：
 *   第一次：探测 maxNnzC 上界（仅符号阶段，无数值）
 *   第二次：执行符号 + 数值，容量已验证
 *
 * Buffer 分离：
 *   buffer1: WorkEstimation + Compute（符号阶段 workspace）
 *   buffer2: Compute + Copy（数值阶段 workspace + 结果暂存）
 *   buffer3: EstimateMemory（仅 ALG2/3；ALG_DEFAULT 返回 0）
 */

/* ---- 1. CreateDescr / 2. DestroyDescr ---- */

aclsparseStatus_t aclsparseSpGEMMCreateDescr(aclsparseSpGEMMDescr_t *descr)
{
    if (descr == nullptr) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    SpgemmDescr *inner = new(std::nothrow) SpgemmDescr{};
    if (inner == nullptr) {
        return ACL_SPARSE_STATUS_ALLOC_FAILED;
    }
    inner->numProds = 0;
    inner->nnzC = 0;
    inner->maxNnzC = 0;
    inner->phase = SPGEMM_PHASE_CREATED;
    inner->cInDataValid = 0;
    inner->buffer1Size = 0;
    inner->buffer2Size = 0;
    inner->buffer3Size = 0;
    *descr = reinterpret_cast<aclsparseSpGEMMDescr_t>(inner);
    return ACL_SPARSE_STATUS_SUCCESS;
}

aclsparseStatus_t aclsparseSpGEMMDestroyDescr(aclsparseSpGEMMDescr_t descr)
{
    if (descr == nullptr) {
        return ACL_SPARSE_STATUS_SUCCESS;
    }
    SpgemmDescr *inner = reinterpret_cast<SpgemmDescr *>(descr);
    delete inner;
    return ACL_SPARSE_STATUS_SUCCESS;
}

/* ---- 3. WorkEstimation ---- */
/* Validates inputs, estimates numProds, computes buffer1 size.
 * If buffer1 is NULL → size query only (no device work).
 * If buffer1 is non-NULL → runs symbolic kernel to get precise numProds + nnzC. */

aclsparseStatus_t aclsparseSpGEMMWorkEstimation(
    aclsparseHandle_t handle,
    aclsparseSpGEMMDescr_t descr,
    size_t *buffer1Size,
    aclsparseOperation_t opA, aclsparseOperation_t opB,
    const void * /*alpha*/,
    aclsparseConstSpMatDescr_t matA,
    aclsparseConstSpMatDescr_t matB,
    const void * /*beta*/,
    aclsparseSpMatDescr_t matC,
    aclDataType computeType,
    aclsparseSpGEMMAlg_t alg,
    void *buffer1)
{
    if (buffer1Size == nullptr) return ACL_SPARSE_STATUS_INVALID_VALUE;
    if (descr == nullptr) return ACL_SPARSE_STATUS_INVALID_VALUE;

    SpgemmLaunchCtx ctx;
    aclsparseStatus_t st = SpgemmPrepareLaunch(handle, matA, matB, matC,
                                                 opA, opB, computeType, alg, ctx);
    if (st != ACL_SPARSE_STATUS_SUCCESS) return st;

    aclsparseSpMatDescr *matAInner = ctx.matAInner;
    aclsparseSpMatDescr *matBInner = ctx.matBInner;
    aclsparseSpMatDescr *matCInner = ctx.matCInner;
    SpgemmDescr *desc = reinterpret_cast<SpgemmDescr *>(descr);
    int32_t m = ctx.m;
    int32_t n = ctx.n;
    int32_t k = ctx.k;
    SpgemmWsOffsets off = ctx.off;
    desc->buffer1Size = off.totalBytes;
    *buffer1Size = static_cast<size_t>(off.totalBytes);

    /* Estimate numProds via host-side computation (D2H A rowPtr + A colInd + B rowPtr) */
    std::vector<int32_t> aRowPtrHost(m + 1);
    aclError aclRet = aclrtMemcpy(aRowPtrHost.data(), (m + 1) * sizeof(int32_t),
                                   matAInner->ptrs, (m + 1) * sizeof(int32_t),
                                   ACL_MEMCPY_DEVICE_TO_HOST);
    if (aclRet != ACL_ERROR_NONE) return ACL_SPARSE_STATUS_EXECUTION_FAILED;

    std::vector<int32_t> aColIndHost(matAInner->nnz);
    aclRet = aclrtMemcpy(aColIndHost.data(), matAInner->nnz * sizeof(int32_t),
                          matAInner->idxs, matAInner->nnz * sizeof(int32_t),
                          ACL_MEMCPY_DEVICE_TO_HOST);
    if (aclRet != ACL_ERROR_NONE) return ACL_SPARSE_STATUS_EXECUTION_FAILED;

    std::vector<int32_t> bRowPtrHost(k + 1);
    aclRet = aclrtMemcpy(bRowPtrHost.data(), (k + 1) * sizeof(int32_t),
                          matBInner->ptrs, (k + 1) * sizeof(int32_t),
                          ACL_MEMCPY_DEVICE_TO_HOST);
    if (aclRet != ACL_ERROR_NONE) return ACL_SPARSE_STATUS_EXECUTION_FAILED;

    desc->numProds = SpgemmComputeNumProds(aRowPtrHost, aColIndHost, bRowPtrHost);

    /* If buffer1 is provided, run symbolic phase to get precise nnzC */
    if (buffer1 != nullptr && matAInner->nnz > 0 && matBInner->nnz > 0) {
        /* Run Preprocess with buffer1 to compute symbolic structure */
        st = aclsparseSpGEMMPreprocess(handle, opA, opB, nullptr, matA, matB,
                                        nullptr, matC, computeType, alg, buffer1);
        if (st != ACL_SPARSE_STATUS_SUCCESS) return st;
        desc->nnzC = static_cast<int64_t>(matCInner->nnz);
        desc->maxNnzC = static_cast<int64_t>(m) * static_cast<int64_t>(n);
        if (desc->maxNnzC > 500000000) desc->maxNnzC = 500000000;
        if (desc->maxNnzC < 1024) desc->maxNnzC = 1024;
        desc->phase = SPGEMM_PHASE_SYMBOLIC_DONE;
    } else {
        desc->phase = SPGEMM_PHASE_WORK_ESTIMATE;
    }

    /* buffer2 = same size as buffer1 (numeric phase reuses workspace layout) */
    desc->buffer2Size = off.totalBytes;

    return ACL_SPARSE_STATUS_SUCCESS;
}

/* ---- 4. EstimateMemory ---- */
/* ALG_DEFAULT → buffer3Size = 0 (no extra memory needed).
 * ALG2/ALG3 → would compute additional buffer3, but currently not implemented. */

aclsparseStatus_t aclsparseSpGEMMEstimateMemory(
    aclsparseHandle_t /*handle*/,
    aclsparseSpGEMMDescr_t descr,
    size_t *buffer3Size,
    aclsparseSpMatDescr_t /*matC*/,
    aclDataType /*computeType*/,
    aclsparseSpGEMMAlg_t alg,
    void * /*buffer3*/)
{
    if (buffer3Size == nullptr) return ACL_SPARSE_STATUS_INVALID_VALUE;
    if (descr == nullptr) return ACL_SPARSE_STATUS_INVALID_VALUE;

    SpgemmDescr *desc = reinterpret_cast<SpgemmDescr *>(descr);

    /* ALG_DEFAULT / ALG1: no buffer3 needed */
    if (alg == ACL_SPARSE_SPGEMM_ALG_DEFAULT || alg == ACL_SPARSE_SPGEMM_ALG1) {
        *buffer3Size = 0;
        desc->buffer3Size = 0;
        return ACL_SPARSE_STATUS_SUCCESS;
    }

    /* ALG2/ALG3: reserved — not yet implemented */
    *buffer3Size = 0;
    desc->buffer3Size = 0;
    return ACL_SPARSE_STATUS_NOT_SUPPORTED;
}

/* ---- 5. Compute ---- */
/* 符号 + 数值阶段。"调两次"模式：先探测内存上界、再执行。
 *
 * 第一次（探测）：仅跑符号 kernel → 得到 nnzC → 校验容量。
 *   若 nnzC > maxNnzC → 返回 INSUFFICIENT_RESOURCES（用户需重新分配）。
 * 第二次（执行）：跑符号（若未完成）+ 数值 → 填充 C 值。
 *
 * descr->phase 状态机跟踪符号阶段是否已由 WorkEstimation 完成
 * （若已完成，Compute 跳过探测）。 */

aclsparseStatus_t aclsparseSpGEMMCompute(
    aclsparseHandle_t handle,
    aclsparseSpGEMMDescr_t descr,
    aclsparseOperation_t opA, aclsparseOperation_t opB,
    const void *alpha,
    aclsparseConstSpMatDescr_t matA,
    aclsparseConstSpMatDescr_t matB,
    const void *beta,
    aclsparseSpMatDescr_t matC,
    aclDataType computeType,
    aclsparseSpGEMMAlg_t alg,
    void *buffer1,
    void *buffer2)
{
    if (descr == nullptr) return ACL_SPARSE_STATUS_INVALID_VALUE;
    if (buffer1 == nullptr || buffer2 == nullptr) return ACL_SPARSE_STATUS_INSUFFICIENT_RESOURCES;

    SpgemmDescr *desc = reinterpret_cast<SpgemmDescr *>(descr);
    SpgemmLaunchCtx ctx;
    aclsparseStatus_t st = SpgemmPrepareLaunch(handle, matA, matB, matC,
                                                 opA, opB, computeType, alg, ctx);
    if (st != ACL_SPARSE_STATUS_SUCCESS) return st;

    aclsparseSpMatDescr *matCInner = ctx.matCInner;

    /* Phase 1: Probe — ensure symbolic structure is computed and capacity verified.
     * If WorkEstimation already did symbolic (phase == SYMBOLIC_DONE), skip. */
    if (desc->phase < SPGEMM_PHASE_SYMBOLIC_DONE) {
        st = aclsparseSpGEMMPreprocess(handle, opA, opB, alpha, matA, matB,
                                        beta, matC, computeType, alg, buffer1);
        if (st != ACL_SPARSE_STATUS_SUCCESS) return st;
        desc->nnzC = static_cast<int64_t>(matCInner->nnz);
        desc->phase = SPGEMM_PHASE_SYMBOLIC_DONE;
    }

    /* Capacity check (host-compute "double-call" probe) */
    if (desc->nnzC > desc->maxNnzC) {
        OP_LOGE("spgemm", "Compute probe: nnzC=%ld exceeds maxNnzC=%ld",
                desc->nnzC, desc->maxNnzC);
        return ACL_SPARSE_STATUS_INSUFFICIENT_RESOURCES;
    }

    /* Phase 2: prepare valuesC — zero or preserve C_in based on β/cInDataValid */
    float betaVal = 0.0f;
    st = SpgemmReadBetaScalar(handle, beta, betaVal);
    if (st != ACL_SPARSE_STATUS_SUCCESS) return st;
    st = SpgemmZeroOrPreserveCIn(handle, matCInner, computeType,
                                  betaVal, desc->cInDataValid, desc->nnzC);
    if (st != ACL_SPARSE_STATUS_SUCCESS) return st;

    /* Run numeric phase using buffer2.
     * Propagate cInDataValid to matC so the 3-stage aclsparseSpGEMM
     * does not memset valuesC when user has filled valid C_in data. */
    matCInner->cInDataValid = desc->cInDataValid;
    st = aclsparseSpGEMM(handle, opA, opB, alpha, matA, matB, beta, matC,
                         computeType, alg, buffer2);
    if (st != ACL_SPARSE_STATUS_SUCCESS) return st;

    desc->phase = SPGEMM_PHASE_COMPUTED;
    return ACL_SPARSE_STATUS_SUCCESS;
}

/* ---- 5b. SetCInValid — 标记 matC->values 包含有效 C_in ---- */
/* 当 β≠0 且 matC 带内容时，按 cuSPARSE 语义叠加 β·C_in。
 * 用户在 WorkEstimation（符号完成）后、填充 matC->values 为 C_in 数据后调用。 */

aclsparseStatus_t aclsparseSpGEMMSetCInValid(
    aclsparseSpGEMMDescr_t descr,
    int32_t cInDataValid)
{
    if (descr == nullptr) return ACL_SPARSE_STATUS_INVALID_VALUE;
    SpgemmDescr *desc = reinterpret_cast<SpgemmDescr *>(descr);
    if (desc->phase < SPGEMM_PHASE_SYMBOLIC_DONE) {
        OP_LOGE("spgemm", "SetCInValid called before symbolic phase (phase=%d), "
                "call WorkEstimation first", desc->phase);
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    desc->cInDataValid = cInDataValid ? 1 : 0;
    return ACL_SPARSE_STATUS_SUCCESS;
}

/* ---- 6. GetNumProducts ---- */

aclsparseStatus_t aclsparseSpGEMMGetNumProducts(
    aclsparseSpGEMMDescr_t descr,
    int64_t *numProducts)
{
    if (numProducts == nullptr) return ACL_SPARSE_STATUS_INVALID_VALUE;
    if (descr == nullptr) return ACL_SPARSE_STATUS_INVALID_VALUE;
    SpgemmDescr *desc = reinterpret_cast<SpgemmDescr *>(descr);
    *numProducts = desc->numProds;
    return ACL_SPARSE_STATUS_SUCCESS;
}

/* ---- 7. Copy ---- */
/* Writes result from workspace buffer2 to matC CSR arrays.
 * In the current implementation, numeric kernel writes directly to matC,
 * so Copy is effectively a no-op (structure already in matC from Preprocess).
 * This interface exists for API completeness and future workspace-staging
 * architectures where Compute writes to an intermediate buffer. */

aclsparseStatus_t aclsparseSpGEMMCopy(
    aclsparseHandle_t /*handle*/,
    aclsparseSpGEMMDescr_t descr,
    aclsparseOperation_t /*opA*/, aclsparseOperation_t /*opB*/,
    const void * /*alpha*/,
    aclsparseConstSpMatDescr_t /*matA*/,
    aclsparseConstSpMatDescr_t /*matB*/,
    const void * /*beta*/,
    aclsparseSpMatDescr_t matC,
    aclDataType /*computeType*/,
    aclsparseSpGEMMAlg_t /*alg*/,
    void * /*buffer2*/)
{
    if (descr == nullptr) return ACL_SPARSE_STATUS_INVALID_VALUE;
    SpgemmDescr *desc = reinterpret_cast<SpgemmDescr *>(descr);

    if (desc->phase < SPGEMM_PHASE_COMPUTED) {
        OP_LOGE("spgemm", "Copy called before Compute (phase=%d)", desc->phase);
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }

    /* In current implementation, numeric kernel writes directly to matC.
     * Copy is a no-op — matC already contains the final result.
     * Future: if Compute stages to workspace, Copy would transfer to matC. */
    aclsparseSpMatDescr *matCInner = ToMatInner(matC);
    matCInner->activeBuffer = nullptr;  /* clear reuse marker */
    desc->phase = SPGEMM_PHASE_COPIED;

    return ACL_SPARSE_STATUS_SUCCESS;
}

} /* extern "C" */
