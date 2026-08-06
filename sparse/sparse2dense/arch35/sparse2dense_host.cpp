/**
 * ----------------------------------------------------------------------------------------------------------
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under
 * the terms and conditions of CANN Open Software License Agreement Version 2.0
 * (the "License"). Please refer to the License for details. You may not use
 * this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
 * AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
 * FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
 * for the full text of the License.
 * ----------------------------------------------------------------------------------------------------------
 */

/*!
 * \file sparse2dense_host.cpp
 * \brief SparseToDense Host 侧实现：2 个 Generic API 入口。
 *
 * API 清单：
 *   F1. aclsparseSparseToDense_bufferSize — 查询 workspace 大小
 *   F2. aclsparseSparseToDense            — 执行 Sparse→Dense 转换
 *
 * 支持 CSR / CSC / COO 三种稀疏格式，值类型 FP32/FP16/BF16/INT32/INT8。
 * 对标 cuSPARSE cusparseSparseToDense / cusparseSparseToDense_bufferSize。
 * 仅支持 arch35（DAV-3510）。
 */

#include <algorithm>
#include <cstdint>

#include "log/log.h"
#include "cann_ops_sparse.h"
#include "aclsparse_handle_internal.h"
#include "aclsparse_descr_internal.h"
#include "aclsparse_host_utils.h"
#include "sparse2dense.h"
#include "sparse2dense_tiling_data.h"

// Host 侧不引入 kernel_operator.h，此处提供 GM_ADDR 的 host 编译回退定义。
// NPU 侧由 toolkit (kernel_utils_macros.h) 自动定义为 __gm__ uint8_t*。
#ifndef GM_ADDR
#define GM_ADDR uint8_t *
#endif
#include "sparse2dense_kernel.h"

static constexpr const char *kTag = "aclsparseSparseToDense";

namespace {

// ===========================================================================
// 公共参数校验
// ===========================================================================

static aclsparseStatus_t ValidateSparseDescriptor(const aclsparseSpMatDescr *matInner)
{
    if (matInner->format != ACL_SPARSE_FORMAT_CSR &&
        matInner->format != ACL_SPARSE_FORMAT_CSC &&
        matInner->format != ACL_SPARSE_FORMAT_COO) {
        OP_LOGE(kTag, "unsupported format %d (CSR/CSC/COO required)",
                static_cast<int>(matInner->format));
        return ACL_SPARSE_STATUS_NOT_SUPPORTED;
    }
    aclsparseStatus_t idxSt =
        AclsparseValidateSupportedCsrIndexTypes(matInner->ptrType, matInner->IdxType);
    if (idxSt != ACL_SPARSE_STATUS_SUCCESS) {
        OP_LOGE(kTag, "unsupported index type ptr=%d idx=%d (only ACL_SPARSE_INDEX_32I)",
                matInner->ptrType, matInner->IdxType);
        return idxSt;
    }
    if (matInner->baseType != ACL_SPARSE_INDEX_BASE_ZERO &&
        matInner->baseType != ACL_SPARSE_INDEX_BASE_ONE) {
        OP_LOGE(kTag, "unsupported indexBase %d", static_cast<int>(matInner->baseType));
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    aclDataType valType = matInner->valueType;
    if (valType != ACL_FLOAT && valType != ACL_FLOAT16 &&
        valType != ACL_BF16 && valType != ACL_INT32 &&
        valType != ACL_INT8) {
        OP_LOGE(kTag, "unsupported valueType %d", static_cast<int>(valType));
        return ACL_SPARSE_STATUS_NOT_SUPPORTED;
    }
    if (matInner->rows > static_cast<uint64_t>(INT32_MAX) ||
        matInner->cols > static_cast<uint64_t>(INT32_MAX)) {
        OP_LOGE(kTag, "m or n exceeds INT32_MAX, not supported");
        return ACL_SPARSE_STATUS_NOT_SUPPORTED;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

static aclsparseStatus_t ValidateDenseDescriptor(const aclsparseDnMatDescr *dnInner,
                                                  aclDataType valType,
                                                  uint64_t sparseRows, uint64_t sparseCols)
{
    if (dnInner->valueType != valType) {
        OP_LOGE(kTag, "matB valueType %d != matA valueType %d",
                static_cast<int>(dnInner->valueType), static_cast<int>(valType));
        return ACL_SPARSE_STATUS_NOT_SUPPORTED;
    }
    if (dnInner->rows != static_cast<int64_t>(sparseRows) ||
        dnInner->cols != static_cast<int64_t>(sparseCols)) {
        OP_LOGE(kTag, "dimension mismatch: matA(%llu x %llu) vs matB(%ld x %ld)",
                static_cast<unsigned long long>(sparseRows),
                static_cast<unsigned long long>(sparseCols),
                dnInner->rows, dnInner->cols);
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (dnInner->ld < 0) {
        OP_LOGE(kTag, "invalid ld=%ld", dnInner->ld);
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (dnInner->ld > static_cast<int64_t>(INT32_MAX)) {
        OP_LOGE(kTag, "ld=%ld exceeds INT32_MAX", dnInner->ld);
        return ACL_SPARSE_STATUS_NOT_SUPPORTED;
    }
    bool isColMajor = (dnInner->order == ACL_SPARSE_ORDER_COL);
    int64_t minLd = isColMajor ? dnInner->rows : dnInner->cols;
    if (dnInner->ld < minLd) {
        OP_LOGE(kTag, "ld=%ld < %s minimum %ld",
                dnInner->ld, isColMajor ? "rows" : "cols", minLd);
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (dnInner->values == nullptr) {
        OP_LOGE(kTag, "matB.values is nullptr");
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

static aclsparseStatus_t ValidateSparseToDenseParams(
    aclsparseHandle_t handle, aclsparseConstSpMatDescr_t matA,
    aclsparseDnMatDescr_t matB)
{
    if (handle == nullptr) {
        OP_LOGE(kTag, "handle is nullptr");
        return ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR;
    }
    if (matA == nullptr || matB == nullptr) {
        OP_LOGE(kTag, "matA or matB is nullptr");
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    auto *matInner = Sparse2DenseToConstMatInner(matA);
    aclsparseStatus_t st = ValidateSparseDescriptor(matInner);
    if (st != ACL_SPARSE_STATUS_SUCCESS) return st;
    auto *dnInner = Sparse2DenseToDnMatInner(matB);
    return ValidateDenseDescriptor(dnInner, matInner->valueType,
                                   matInner->rows, matInner->cols);
}

// ===========================================================================
// block 切分
// CSR: 按行数 m 切分；CSC: 按列数 n 切分；COO: 按 nnz 切分
// ===========================================================================
static aclsparseStatus_t ComputeBlockSplits(
    int32_t dim, uint32_t &useBlocks, uint32_t &perBlock)
{
    if (dim <= 0) {
        useBlocks = 0;
        perBlock = 0;
        return ACL_SPARSE_STATUS_SUCCESS;
    }
    uint32_t aivCoreNum = GetAivCoreCount();
    CHECK_RET(aivCoreNum > 0,
              OP_LOGE(kTag, "GetAivCoreCount returned 0");
              return ACL_SPARSE_STATUS_INTERNAL_ERROR);
    useBlocks = std::min(aivCoreNum,
        CeilDiv<uint32_t>(static_cast<uint32_t>(dim), kSparse2DenseMaxThreadsPerBlock));
    if (useBlocks == 0) {
        useBlocks = 1;
    }
    perBlock = CeilDiv<uint32_t>(static_cast<uint32_t>(dim), useBlocks);
    if (perBlock == 0) {
        perBlock = 1;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

// 计算稠密矩阵实际存储字节数（用于 memset 清零）。
// 行主序：m 行 × ld stride；列主序：n 列 × ld stride。
static inline size_t DenseStorageBytes(int32_t m, int32_t n, int32_t ld,
                                       bool isColMajor, size_t elemSize)
{
    int64_t elems = isColMajor
        ? static_cast<int64_t>(n) * ld
        : static_cast<int64_t>(m) * ld;
    return static_cast<size_t>(elems) * elemSize;
}

}  // namespace

// ===========================================================================
// Public APIs
// ===========================================================================
extern "C" {

// F1: bufferSize
aclsparseStatus_t aclsparseSparseToDense_bufferSize(
    aclsparseHandle_t handle,
    aclsparseConstSpMatDescr_t matA,
    aclsparseDnMatDescr_t matB,
    aclsparseSparseToDenseAlg_t alg,
    size_t *bufferSize)
{
    aclsparseStatus_t st = ValidateSparseToDenseParams(handle, matA, matB);
    if (st != ACL_SPARSE_STATUS_SUCCESS) { return st; }

    if (bufferSize == nullptr) {
        OP_LOGE(kTag, "bufferSize is nullptr");
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }

    if (alg != ACL_SPARSE_SPARSETODENSE_ALG_DEFAULT) {
        OP_LOGE(kTag, "unsupported alg %d", static_cast<int>(alg));
        return ACL_SPARSE_STATUS_NOT_SUPPORTED;
    }

    *bufferSize = 0;
    return ACL_SPARSE_STATUS_SUCCESS;
}

// F2: execute
static aclsparseStatus_t ZeroDenseOutput(const aclsparseDnMatDescr *dnInner,
                                          size_t dnMatBytes)
{
    aclError ret = aclrtMemset(dnInner->values, dnMatBytes, 0, dnMatBytes);
    if (ret != ACL_SUCCESS) {
        OP_LOGE(kTag, "aclrtMemset failed, ret=%d", ret);
        return ACL_SPARSE_STATUS_INTERNAL_ERROR;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

static aclsparseStatus_t ValidateAndZeroOutput(const aclsparseSpMatDescr *matInner,
                                                const aclsparseDnMatDescr *dnInner,
                                                int32_t m, int32_t n,
                                                size_t elemSize, size_t dnMatBytes)
{
    if (m == 0 || n == 0) {
        if (dnMatBytes > 0) {
            aclsparseStatus_t zst = ZeroDenseOutput(dnInner, dnMatBytes);
            if (zst != ACL_SPARSE_STATUS_SUCCESS) return zst;
        }
        OP_LOGD(kTag, "empty matrix, output zeroed, skip kernel launch");
        return ACL_SPARSE_STATUS_SUCCESS;
    }
    if (matInner->nnz == 0) {
        aclsparseStatus_t zst = ZeroDenseOutput(dnInner, dnMatBytes);
        if (zst != ACL_SPARSE_STATUS_SUCCESS) return zst;
        OP_LOGD(kTag, "nnz=0, output zeroed, skip kernel launch");
        return ACL_SPARSE_STATUS_SUCCESS;
    }
    if (matInner->ptrs == nullptr) {
        OP_LOGE(kTag, "sparse ptrs is null");
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (matInner->idxs == nullptr || matInner->values == nullptr) {
        OP_LOGE(kTag, "idxs or values is null (nnz=%llu > 0)",
                static_cast<unsigned long long>(matInner->nnz));
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    return ZeroDenseOutput(dnInner, dnMatBytes);
}

static int32_t ComputeSplitDim(const aclsparseSpMatDescr *matInner,
                                int32_t m, int32_t n,
                                aclsparseStatus_t &status)
{
    status = ACL_SPARSE_STATUS_SUCCESS;
    if (matInner->format == ACL_SPARSE_FORMAT_CSC) return n;
    if (matInner->format == ACL_SPARSE_FORMAT_COO) {
        if (matInner->nnz > static_cast<uint64_t>(INT32_MAX)) {
            OP_LOGE(kTag, "nnz exceeds INT32_MAX for COO split, not supported");
            status = ACL_SPARSE_STATUS_NOT_SUPPORTED;
            return 0;
        }
        return static_cast<int32_t>(matInner->nnz);
    }
    return m;
}

static aclsparseStatus_t LaunchSparseToDenseKernel(
    const aclsparseSpMatDescr *matInner, const aclsparseDnMatDescr *dnInner,
    int32_t m, int32_t n, bool isColMajor, aclrtStream stream)
{
    aclsparseStatus_t dimStatus = ACL_SPARSE_STATUS_SUCCESS;
    int32_t splitDim = ComputeSplitDim(matInner, m, n, dimStatus);
    if (dimStatus != ACL_SPARSE_STATUS_SUCCESS) { return dimStatus; }

    uint32_t useBlocks = 0;
    uint32_t perBlock = 0;
    aclsparseStatus_t st = ComputeBlockSplits(splitDim, useBlocks, perBlock);
    if (st != ACL_SPARSE_STATUS_SUCCESS) { return st; }

    Sparse2DenseTilingData tiling{};
    tiling.m = m;
    tiling.n = n;
    tiling.indexBase = static_cast<int32_t>(matInner->baseType);
    tiling.valueType = Sparse2DenseValTypeFromAcl(matInner->valueType);
    tiling.isColMajor = isColMajor ? 1 : 0;
    tiling.ld = static_cast<int32_t>(dnInner->ld);
    tiling.perBlock = perBlock;
    tiling.format = Sparse2DenseFormatFromAcl(matInner->format);
    tiling.nnz = matInner->nnz;

    sparse2dense_kernel_do(
        reinterpret_cast<GM_ADDR>(matInner->ptrs),
        reinterpret_cast<GM_ADDR>(matInner->idxs),
        reinterpret_cast<GM_ADDR>(matInner->values),
        reinterpret_cast<GM_ADDR>(dnInner->values),
        tiling, useBlocks, stream);

    OP_LOGI(kTag, "kernel launched: m=%d, n=%d, fmt=%d, numBlocks=%u, valType=%d, order=%s",
            m, n, tiling.format, useBlocks, tiling.valueType,
            tiling.isColMajor ? "col-major" : "row-major");
    return ACL_SPARSE_STATUS_SUCCESS;
}

aclsparseStatus_t aclsparseSparseToDense(
    aclsparseHandle_t handle,
    aclsparseConstSpMatDescr_t matA,
    aclsparseDnMatDescr_t matB,
    aclsparseSparseToDenseAlg_t alg,
    void *buffer)
{
    aclsparseStatus_t st = ValidateSparseToDenseParams(handle, matA, matB);
    if (st != ACL_SPARSE_STATUS_SUCCESS) { return st; }
    if (alg != ACL_SPARSE_SPARSETODENSE_ALG_DEFAULT) {
        OP_LOGE(kTag, "unsupported alg %d", static_cast<int>(alg));
        return ACL_SPARSE_STATUS_NOT_SUPPORTED;
    }
    (void)buffer;

    auto *h = Sparse2DenseToInternalHandle(handle);
    auto *matInner = Sparse2DenseToConstMatInner(matA);
    auto *dnInner = Sparse2DenseToDnMatInner(matB);

    aclrtStream stream = h->stream;
    if (stream == nullptr) {
        OP_LOGE(kTag, "stream is nullptr, please call aclsparseSetStream first");
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }

    int32_t m = static_cast<int32_t>(matInner->rows);
    int32_t n = static_cast<int32_t>(matInner->cols);
    size_t elemSize = AclDataTypeSize(matInner->valueType);
    const bool isColMajor = (dnInner->order == ACL_SPARSE_ORDER_COL);
    size_t dnMatBytes = DenseStorageBytes(m, n, static_cast<int32_t>(dnInner->ld),
                                          isColMajor, elemSize);

    aclsparseStatus_t zst = ValidateAndZeroOutput(matInner, dnInner, m, n,
                                                   elemSize, dnMatBytes);
    if (zst != ACL_SPARSE_STATUS_SUCCESS) { return zst; }
    if (m == 0 || n == 0 || matInner->nnz == 0) {
        return ACL_SPARSE_STATUS_SUCCESS;
    }

    return LaunchSparseToDenseKernel(matInner, dnInner, m, n, isColMajor, stream);
}

}  // extern "C"
