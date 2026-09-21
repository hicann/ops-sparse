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

#include <climits>
#include <cstdint>
#include <algorithm>

#include "acl/acl.h"
#include "aclsparse_host_utils.h"
#include "aclsparse_descr_internal.h"
#include "aclsparse_handle_internal.h"
#include "cann_ops_sparse.h"
#include "spmv.h"

extern "C" void spmv_arch35_kernel_launch(
    const void* rowOffsets, const void* colInd, const void* values, const void* xVec, void* yVec, void* workspace,
    const SpmvWorkspaceLayout& layout, const SpmvTilingData& tiling, uint32_t blockDim, void* stream);

extern "C" void spmv_arch35_preprocess_launch(
    const void* rowOffsets, const void* colInd, void* workspace, const SpmvWorkspaceLayout& layout, int32_t rows,
    int32_t cols, int32_t nnz, int64_t rowOffsetsStride, int64_t colIndStride, void* stream);

namespace {

inline aclsparseSpMatDescr* ToMatInner(aclsparseConstSpMatDescr_t descr)
{
    return const_cast<aclsparseSpMatDescr*>(reinterpret_cast<const aclsparseSpMatDescr*>(descr));
}

inline aclsparseDnVecDescr* ToVecInner(aclsparseConstDnVecDescr_t descr)
{
    return const_cast<aclsparseDnVecDescr*>(reinterpret_cast<const aclsparseDnVecDescr*>(descr));
}

bool IsSupportedAlg(aclsparseSpMVAlg_t alg)
{
    return alg == ACL_SPARSE_SPMV_ALG_DEFAULT || alg == ACL_SPARSE_SPMV_CSR_ALG1 || alg == ACL_SPARSE_SPMV_CSR_ALG2;
}

int32_t GetSpmvDataType(aclDataType inputType, aclDataType outputType, aclDataType computeType)
{
    struct TypeMapping {
        aclDataType input;
        aclDataType output;
        aclDataType compute;
        int32_t kernelType;
    };
    constexpr TypeMapping mappings[] = {
        {ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, SPMV_DTYPE_FP32_FP32},
        {ACL_INT8, ACL_INT32, ACL_INT32, SPMV_DTYPE_INT8_INT32},
        {ACL_INT8, ACL_FLOAT, ACL_FLOAT, SPMV_DTYPE_INT8_FP32},
        {ACL_FLOAT16, ACL_FLOAT, ACL_FLOAT, SPMV_DTYPE_FP16_FP32},
        {ACL_FLOAT16, ACL_FLOAT16, ACL_FLOAT, SPMV_DTYPE_FP16_FP16},
        {ACL_BF16, ACL_FLOAT, ACL_FLOAT, SPMV_DTYPE_BF16_FP32},
        {ACL_BF16, ACL_BF16, ACL_FLOAT, SPMV_DTYPE_BF16_BF16},
    };
    for (const auto& mapping : mappings) {
        if (mapping.input == inputType && mapping.output == outputType && mapping.compute == computeType) {
            return mapping.kernelType;
        }
    }
    return -1;
}

struct SpmvValidatedInputs {
    aclsparseContext* handle{nullptr};
    aclsparseSpMatDescr* mat{nullptr};
    aclsparseDnVecDescr* x{nullptr};
    aclsparseDnVecDescr* y{nullptr};
    int32_t dataType{-1};
    int32_t rows{0};
    int32_t cols{0};
    int32_t nnz{0};
};

aclsparseStatus_t ValidateSpmvLayout(SpmvValidatedInputs& out)
{
    if (out.mat->format != ACL_SPARSE_FORMAT_CSR) {
        OP_LOGE("aclsparseSpMV", "unsupported matrix format=%d; expected CSR", static_cast<int>(out.mat->format));
        return ACL_SPARSE_STATUS_NOT_SUPPORTED;
    }
    if (out.mat->baseType != ACL_SPARSE_INDEX_BASE_ZERO) {
        OP_LOGE("aclsparseSpMV", "unsupported index base=%d; expected zero", static_cast<int>(out.mat->baseType));
        return ACL_SPARSE_STATUS_NOT_SUPPORTED;
    }
    if (out.mat->ptrType != ACL_SPARSE_INDEX_32I || out.mat->IdxType != ACL_SPARSE_INDEX_32I) {
        OP_LOGE("aclsparseSpMV", "unsupported index types: rowOffsets=%d, colInd=%d; expected INT32",
                static_cast<int>(out.mat->ptrType), static_cast<int>(out.mat->IdxType));
        return ACL_SPARSE_STATUS_NOT_SUPPORTED;
    }
    if (out.mat->ptrStride <= 0 || out.mat->idxStride <= 0 || out.mat->valueStride <= 0 || out.x->stride <= 0 ||
        out.y->stride <= 0) {
        OP_LOGE("aclsparseSpMV", "strides must be positive: rowOffsets=%lld, colInd=%lld, values=%lld, x=%lld, y=%lld",
                static_cast<long long>(out.mat->ptrStride), static_cast<long long>(out.mat->idxStride),
                static_cast<long long>(out.mat->valueStride), static_cast<long long>(out.x->stride),
                static_cast<long long>(out.y->stride));
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (out.mat->rows > static_cast<uint64_t>(INT32_MAX) || out.mat->cols > static_cast<uint64_t>(INT32_MAX) ||
        out.mat->nnz > static_cast<uint64_t>(INT32_MAX)) {
        OP_LOGE("aclsparseSpMV", "rows/cols/nnz exceed INT32_MAX: rows=%llu, cols=%llu, nnz=%llu",
                static_cast<unsigned long long>(out.mat->rows), static_cast<unsigned long long>(out.mat->cols),
                static_cast<unsigned long long>(out.mat->nnz));
        return ACL_SPARSE_STATUS_NOT_SUPPORTED;
    }
    if (out.mat->nnz > 0 && (out.mat->rows == 0 || out.mat->cols == 0)) {
        OP_LOGE("aclsparseSpMV", "nnz must be zero when rows or cols is zero");
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }

    out.rows = static_cast<int32_t>(out.mat->rows);
    out.cols = static_cast<int32_t>(out.mat->cols);
    out.nnz = static_cast<int32_t>(out.mat->nnz);
    return ACL_SPARSE_STATUS_SUCCESS;
}

aclsparseStatus_t ValidateSpmvTypesAndShape(SpmvValidatedInputs& out, aclsparseOperation_t opA, aclDataType computeType)
{
    if (out.x->valueType != out.mat->valueType) {
        OP_LOGE("aclsparseSpMV", "x dtype=%d does not match matrix dtype=%d",
                static_cast<int>(out.x->valueType), static_cast<int>(out.mat->valueType));
        return ACL_SPARSE_STATUS_NOT_SUPPORTED;
    }
    out.dataType = GetSpmvDataType(out.mat->valueType, out.y->valueType, computeType);
    if (out.dataType < 0) {
        OP_LOGE("aclsparseSpMV", "unsupported dtype combination: matrix=%d, y=%d, compute=%d",
                static_cast<int>(out.mat->valueType), static_cast<int>(out.y->valueType), static_cast<int>(computeType));
        return ACL_SPARSE_STATUS_NOT_SUPPORTED;
    }

    const uint64_t requiredX = opA == ACL_SPARSE_OP_TRANSPOSE ? out.mat->rows : out.mat->cols;
    const uint64_t requiredY = opA == ACL_SPARSE_OP_TRANSPOSE ? out.mat->cols : out.mat->rows;
    if (out.x->nums < requiredX || out.y->nums < requiredY) {
        OP_LOGE("aclsparseSpMV", "vector too short: x=%llu (need %llu), y=%llu (need %llu)",
                static_cast<unsigned long long>(out.x->nums), static_cast<unsigned long long>(requiredX),
                static_cast<unsigned long long>(out.y->nums), static_cast<unsigned long long>(requiredY));
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

// Only stages that read device storage require bound pointers. A workspace
// query depends on descriptor metadata, not allocated matrix/vector storage.
aclsparseStatus_t ValidateSpmvPointers(const SpmvValidatedInputs& inputs, bool execute)
{
    // Execute calls this only after handling an empty output.
    if (execute && inputs.y->values == nullptr) {
        OP_LOGE("aclsparseSpMV", "y values are unbound for nonempty output");
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (inputs.nnz > 0 && (inputs.mat->ptrs == nullptr || inputs.mat->idxs == nullptr)) {
        OP_LOGE("aclsparseSpMV", "CSR indices unbound for nnz>0: rowOffsets_null=%d, colInd_null=%d",
                inputs.mat->ptrs == nullptr, inputs.mat->idxs == nullptr);
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (execute && inputs.nnz > 0 && (inputs.mat->values == nullptr || inputs.x->values == nullptr)) {
        OP_LOGE("aclsparseSpMV", "compute inputs unbound for nnz>0: values_null=%d, x_null=%d",
                inputs.mat->values == nullptr, inputs.x->values == nullptr);
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

aclsparseStatus_t ValidateSpmvInputs(
    aclsparseHandle_t handle, aclsparseOperation_t opA, const void* alpha, aclsparseConstSpMatDescr_t matA,
    aclsparseConstDnVecDescr_t vecX, const void* beta, aclsparseDnVecDescr_t vecY, aclDataType computeType,
    aclsparseSpMVAlg_t alg, SpmvValidatedInputs& out)
{
    if (handle == nullptr) {
        OP_LOGE("aclsparseSpMV", "handle is nullptr");
        return ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR;
    }
    if (matA == nullptr || vecX == nullptr || vecY == nullptr || alpha == nullptr || beta == nullptr) {
        OP_LOGE("aclsparseSpMV", "null argument: matA_null=%d, vecX_null=%d, vecY_null=%d, alpha_null=%d, beta_null=%d",
                matA == nullptr, vecX == nullptr, vecY == nullptr, alpha == nullptr, beta == nullptr);
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (opA != ACL_SPARSE_OP_NON_TRANSPOSE && opA != ACL_SPARSE_OP_TRANSPOSE) {
        OP_LOGE("aclsparseSpMV", "unsupported operation=%d", static_cast<int>(opA));
        return ACL_SPARSE_STATUS_NOT_SUPPORTED;
    }
    if (!IsSupportedAlg(alg)) {
        OP_LOGE("aclsparseSpMV", "unsupported algorithm=%d", static_cast<int>(alg));
        return ACL_SPARSE_STATUS_NOT_SUPPORTED;
    }
    out.handle = reinterpret_cast<aclsparseContext*>(handle);
    out.mat = ToMatInner(matA);
    out.x = ToVecInner(vecX);
    out.y = ToVecInner(vecY);
    const aclsparseStatus_t status = ValidateSpmvLayout(out);
    if (status != ACL_SPARSE_STATUS_SUCCESS) {
        return status;
    }
    return ValidateSpmvTypesAndShape(out, opA, computeType);
}

SpmvWorkspaceLayout GetWorkspaceLayout(const SpmvValidatedInputs& inputs)
{
    SpmvWorkspaceLayout layout{};
    layout.Compute(inputs.cols, inputs.nnz);
    return layout;
}

SpmvTilingData BuildTiling(
    const SpmvValidatedInputs& inputs, aclsparseOperation_t opA, const void* alpha, const void* beta,
    aclDataType computeType, uint32_t blockDim)
{
    SpmvTilingData tiling{};
    tiling.rows = inputs.rows;
    tiling.cols = inputs.cols;
    tiling.nnz = inputs.nnz;
    tiling.dataType = inputs.dataType;
    tiling.transpose = opA == ACL_SPARSE_OP_TRANSPOSE ? 1 : 0;
    tiling.computeType = static_cast<int32_t>(computeType);
    tiling.rowOffsetsStride = inputs.mat->ptrStride;
    tiling.colIndStride = inputs.mat->idxStride;
    tiling.valuesStride = inputs.mat->valueStride;
    tiling.xStride = inputs.x->stride;
    tiling.yStride = inputs.y->stride;
    const uint32_t outputSize = static_cast<uint32_t>(tiling.transpose != 0 ? inputs.cols : inputs.rows);
    tiling.workPerBlock = 1u;
    if (outputSize > 0u && blockDim > 0u) {
        tiling.workPerBlock = (outputSize - 1u) / blockDim + 1u;
    }

    if (inputs.handle->pointerMode == ACL_SPARSE_POINTER_MODE_DEVICE) {
        tiling.alphaDevicePtr = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(alpha));
        tiling.betaDevicePtr = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(beta));
    } else if (computeType == ACL_INT32) {
        tiling.alphaInt = *static_cast<const int32_t*>(alpha);
        tiling.betaInt = *static_cast<const int32_t*>(beta);
    } else {
        tiling.alphaFloat = *static_cast<const float*>(alpha);
        tiling.betaFloat = *static_cast<const float*>(beta);
    }
    return tiling;
}

aclsparseStatus_t PrepareTranspose(SpmvValidatedInputs& inputs, void* externalBuffer)
{
    if (inputs.nnz == 0) {
        inputs.mat->activeSpmvBuffer = externalBuffer;
        return ACL_SPARSE_STATUS_SUCCESS;
    }
    if (externalBuffer == nullptr) {
        OP_LOGE("aclsparseSpMVPreprocess", "externalBuffer is nullptr for transpose with nnz>0");
        return ACL_SPARSE_STATUS_INSUFFICIENT_RESOURCES;
    }
    if (inputs.handle->stream == nullptr) {
        OP_LOGE("aclsparseSpMVPreprocess", "handle stream is nullptr");
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    const aclsparseStatus_t status = ValidateSpmvPointers(inputs, false);
    if (status != ACL_SPARSE_STATUS_SUCCESS) {
        return status;
    }

    const SpmvWorkspaceLayout layout = GetWorkspaceLayout(inputs);
    spmv_arch35_preprocess_launch(
        inputs.mat->ptrs, inputs.mat->idxs, externalBuffer, layout, inputs.rows, inputs.cols, inputs.nnz,
        inputs.mat->ptrStride, inputs.mat->idxStride, inputs.handle->stream);
    inputs.mat->activeSpmvBuffer = externalBuffer;
    return ACL_SPARSE_STATUS_SUCCESS;
}

} // namespace

extern "C" {

aclsparseStatus_t aclsparseSpMVGetBufferSize(
    aclsparseHandle_t handle, aclsparseOperation_t opA, const void* alpha, aclsparseConstSpMatDescr_t matA,
    aclsparseConstDnVecDescr_t vecX, const void* beta, aclsparseDnVecDescr_t vecY, aclDataType computeType,
    aclsparseSpMVAlg_t alg, size_t* bufferSize)
{
    if (bufferSize == nullptr) {
        OP_LOGE("aclsparseSpMVGetBufferSize", "bufferSize output pointer is nullptr");
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    SpmvValidatedInputs inputs{};
    const aclsparseStatus_t status =
        ValidateSpmvInputs(handle, opA, alpha, matA, vecX, beta, vecY, computeType, alg, inputs);
    if (status != ACL_SPARSE_STATUS_SUCCESS) {
        return status;
    }
    if (opA == ACL_SPARSE_OP_TRANSPOSE && inputs.nnz > 0) {
        *bufferSize = GetWorkspaceLayout(inputs).totalBytes;
    } else {
        *bufferSize = 0u;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

aclsparseStatus_t aclsparseSpMVPreprocess(
    aclsparseHandle_t handle, aclsparseOperation_t opA, const void* alpha, aclsparseConstSpMatDescr_t matA,
    aclsparseConstDnVecDescr_t vecX, const void* beta, aclsparseDnVecDescr_t vecY, aclDataType computeType,
    aclsparseSpMVAlg_t alg, void* externalBuffer)
{
    SpmvValidatedInputs inputs{};
    const aclsparseStatus_t status =
        ValidateSpmvInputs(handle, opA, alpha, matA, vecX, beta, vecY, computeType, alg, inputs);
    if (status != ACL_SPARSE_STATUS_SUCCESS) {
        return status;
    }
    if (opA == ACL_SPARSE_OP_NON_TRANSPOSE) {
        return ACL_SPARSE_STATUS_SUCCESS;
    }
    if (inputs.nnz > 0 && GetAivCoreCount() == 0u) {
        OP_LOGE("aclsparseSpMVPreprocess", "GetAivCoreCount returned 0");
        return ACL_SPARSE_STATUS_INTERNAL_ERROR;
    }
    return PrepareTranspose(inputs, externalBuffer);
}

aclsparseStatus_t aclsparseSpMV(
    aclsparseHandle_t handle, aclsparseOperation_t opA, const void* alpha, aclsparseConstSpMatDescr_t matA,
    aclsparseConstDnVecDescr_t vecX, const void* beta, aclsparseDnVecDescr_t vecY, aclDataType computeType,
    aclsparseSpMVAlg_t alg, void* externalBuffer)
{
    SpmvValidatedInputs inputs{};
    aclsparseStatus_t status = ValidateSpmvInputs(handle, opA, alpha, matA, vecX, beta, vecY, computeType, alg, inputs);
    if (status != ACL_SPARSE_STATUS_SUCCESS) {
        return status;
    }
    if (inputs.handle->stream == nullptr) {
        OP_LOGE("aclsparseSpMV", "handle stream is nullptr");
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    const uint32_t outputSize = static_cast<uint32_t>(opA == ACL_SPARSE_OP_TRANSPOSE ? inputs.cols : inputs.rows);
    if (outputSize == 0u) {
        return ACL_SPARSE_STATUS_SUCCESS;
    }
    status = ValidateSpmvPointers(inputs, true);
    if (status != ACL_SPARSE_STATUS_SUCCESS) {
        return status;
    }
    const uint32_t aivCoreNum = GetAivCoreCount();
    if (aivCoreNum == 0u) {
        OP_LOGE("aclsparseSpMV", "GetAivCoreCount returned 0");
        return ACL_SPARSE_STATUS_INTERNAL_ERROR;
    }
    const uint32_t blockDim = std::min(outputSize, aivCoreNum);

    if (opA == ACL_SPARSE_OP_TRANSPOSE && inputs.nnz > 0) {
        if (externalBuffer == nullptr) {
            OP_LOGE("aclsparseSpMV", "externalBuffer is nullptr for transpose with nnz>0");
            return ACL_SPARSE_STATUS_INSUFFICIENT_RESOURCES;
        }
        if (inputs.mat->activeSpmvBuffer != externalBuffer) {
            status = PrepareTranspose(inputs, externalBuffer);
            if (status != ACL_SPARSE_STATUS_SUCCESS) {
                return status;
            }
        }
    }

    const SpmvWorkspaceLayout layout = GetWorkspaceLayout(inputs);
    const SpmvTilingData tiling = BuildTiling(inputs, opA, alpha, beta, computeType, blockDim);
    spmv_arch35_kernel_launch(
        inputs.mat->ptrs, inputs.mat->idxs, inputs.mat->values, inputs.x->values, inputs.y->values, externalBuffer,
        layout, tiling, blockDim, inputs.handle->stream);
    return ACL_SPARSE_STATUS_SUCCESS;
}

} // extern "C"
