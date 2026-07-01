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

/*!
 * \file aclsparse_descr.cpp
 * \brief 稀疏/稠密向量、稠密矩阵与 CSR 稀疏矩阵描述符的公共构造/销毁接口。
 *
 * 这些 API 与具体算子架构无关，需在 arch22(SpMV) / arch35(SpMM) 等所有 SOC 构建中可用。
 */

#include "cann_ops_sparse.h"
#include "aclsparse_descr_internal.h"

#include <new>

extern "C" {

aclsparseStatus_t aclsparseCreateDnVec(aclsparseDnVecDescr_t *dnVecDescr, int64_t size, void *values,
                                       aclDataType valueType)
{
    if (dnVecDescr == nullptr) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    auto *inner = new (std::nothrow) aclsparseDnVecDescr();
    if (inner == nullptr) {
        return ACL_SPARSE_STATUS_ALLOC_FAILED;
    }

    inner->nums = static_cast<uint64_t>(size);
    inner->values = values;
    inner->valueType = valueType;
    *dnVecDescr = inner;
    return ACL_SPARSE_STATUS_SUCCESS;
}

aclsparseStatus_t aclsparseCreateConstDnVec(aclsparseConstDnVecDescr_t *dnVecDescr, int64_t size,
    const void *values, aclDataType valueType)
{
    aclsparseDnVecDescr_t tmp = nullptr;
    aclsparseStatus_t st = aclsparseCreateDnVec(&tmp, size, const_cast<void *>(values), valueType);
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        return st;
    }
    *dnVecDescr = tmp;
    return ACL_SPARSE_STATUS_SUCCESS;
}

aclsparseStatus_t aclsparseDestroyDnVec(aclsparseConstDnVecDescr_t dnVecDescr)
{
    if (dnVecDescr == nullptr) {
        return ACL_SPARSE_STATUS_SUCCESS;
    }
    delete const_cast<aclsparseDnVecDescr *>(dnVecDescr);
    return ACL_SPARSE_STATUS_SUCCESS;
}

aclsparseStatus_t aclsparseCreateCsr(aclsparseSpMatDescr_t *spMatDescr, int64_t rows, int64_t cols, int64_t nnz,
    void *csrRowOffsets, void *csrColInd, void *csrValues, aclsparseIndexType_t csrRowOffsetsType,
    aclsparseIndexType_t csrColIndType, aclsparseIndexBase_t idxBase, aclDataType valueType)
{
    if (spMatDescr == nullptr) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    aclsparseStatus_t idxSt =
        AclsparseValidateSupportedCsrIndexTypes(csrRowOffsetsType, csrColIndType);
    if (idxSt != ACL_SPARSE_STATUS_SUCCESS) {
        return idxSt;
    }
    auto *inner = new (std::nothrow) aclsparseSpMatDescr();
    if (inner == nullptr) {
        return ACL_SPARSE_STATUS_ALLOC_FAILED;
    }

    inner->format = ACL_SPARSE_FORMAT_CSR;
    inner->rows = static_cast<uint64_t>(rows);
    inner->cols = static_cast<uint64_t>(cols);
    inner->nnz = static_cast<uint64_t>(nnz);
    inner->ptrs = csrRowOffsets;
    inner->idxs = csrColInd;
    inner->values = csrValues;
    inner->baseType = idxBase;
    inner->ptrType = csrRowOffsetsType;
    inner->IdxType = csrColIndType;
    inner->valueType = valueType;
    *spMatDescr = inner;
    return ACL_SPARSE_STATUS_SUCCESS;
}

aclsparseStatus_t aclsparseCreateConstCsr(aclsparseConstSpMatDescr_t *spMatDescr, int64_t rows, int64_t cols,
    int64_t nnz, const void *csrRowOffsets, const void *csrColInd, const void *csrValues,
    aclsparseIndexType_t csrRowOffsetsType, aclsparseIndexType_t csrColIndType,
    aclsparseIndexBase_t idxBase, aclDataType valueType)
{
    aclsparseSpMatDescr_t tmp = nullptr;
    aclsparseStatus_t st = aclsparseCreateCsr(&tmp, rows, cols, nnz,
        const_cast<void *>(csrRowOffsets), const_cast<void *>(csrColInd),
        const_cast<void *>(csrValues), csrRowOffsetsType, csrColIndType, idxBase, valueType);
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        return st;
    }
    *spMatDescr = tmp;
    return ACL_SPARSE_STATUS_SUCCESS;
}

aclsparseStatus_t aclsparseDestroySpMat(aclsparseConstSpMatDescr_t spMatDescr)
{
    if (spMatDescr == nullptr) {
        return ACL_SPARSE_STATUS_SUCCESS;
    }
    delete const_cast<aclsparseSpMatDescr *>(spMatDescr);
    return ACL_SPARSE_STATUS_SUCCESS;
}

aclsparseStatus_t aclsparseCreateDnMat(aclsparseDnMatDescr_t *dnMatDescr,
    int64_t rows, int64_t cols, int64_t ld, void *values,
    aclDataType valueType, aclsparseOrder_t order)
{
    if (dnMatDescr == nullptr) {
        return ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR;
    }
    if (rows <= 0 || cols <= 0 || ld <= 0) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (values == nullptr) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (order != ACL_SPARSE_ORDER_ROW && order != ACL_SPARSE_ORDER_COL) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    // ld 约束：行主序需 >= cols；列主序需 >= rows。
    if (order == ACL_SPARSE_ORDER_ROW && ld < cols) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (order == ACL_SPARSE_ORDER_COL && ld < rows) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    auto *inner = new (std::nothrow) aclsparseDnMatDescr();
    if (inner == nullptr) {
        return ACL_SPARSE_STATUS_ALLOC_FAILED;
    }
    inner->rows = rows;
    inner->cols = cols;
    inner->ld = ld;
    inner->order = order;
    inner->values = values;
    inner->valueType = valueType;
    *dnMatDescr = inner;
    return ACL_SPARSE_STATUS_SUCCESS;
}

aclsparseStatus_t aclsparseDestroyDnMat(aclsparseConstDnMatDescr_t dnMatDescr)
{
    if (dnMatDescr == nullptr) {
        return ACL_SPARSE_STATUS_SUCCESS;
    }
    delete const_cast<aclsparseDnMatDescr *>(dnMatDescr);
    return ACL_SPARSE_STATUS_SUCCESS;
}

aclsparseStatus_t aclsparseCreateConstDnMat(aclsparseConstDnMatDescr_t *dnMatDescr,
    int64_t rows, int64_t cols, int64_t ld, const void *values,
    aclDataType valueType, aclsparseOrder_t order)
{
    if (dnMatDescr == nullptr) {
        return ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR;
    }
    aclsparseDnMatDescr_t tmp = nullptr;
    aclsparseStatus_t st = aclsparseCreateDnMat(&tmp, rows, cols, ld,
        const_cast<void *>(values), valueType, order);
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        return st;
    }
    *dnMatDescr = tmp;
    return ACL_SPARSE_STATUS_SUCCESS;
}

} // extern "C"
