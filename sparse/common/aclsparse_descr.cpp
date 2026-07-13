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
    if (size < 0) {
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
    if (dnVecDescr == nullptr) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
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

aclsparseStatus_t aclsparseCreateSpVec(aclsparseSpVecDescr_t *spVecDescr, int64_t size,
    int64_t nnz, void *indices, void *values, aclsparseIndexType_t idxType,
    aclsparseIndexBase_t idxBase, aclDataType valueType)
{
    if (spVecDescr == nullptr) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (size < 0 || nnz < 0 || nnz > size) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    auto *inner = new (std::nothrow) aclsparseSpVecDescr();
    if (inner == nullptr) {
        return ACL_SPARSE_STATUS_ALLOC_FAILED;
    }

    inner->size = static_cast<uint64_t>(size);
    inner->nnz = static_cast<uint64_t>(nnz);
    inner->indices = indices;
    inner->values = values;
    inner->idxType = idxType;
    inner->idxBase = idxBase;
    inner->valueType = valueType;
    *spVecDescr = inner;
    return ACL_SPARSE_STATUS_SUCCESS;
}

aclsparseStatus_t aclsparseCreateConstSpVec(aclsparseConstSpVecDescr_t *spVecDescr, int64_t size,
    int64_t nnz, const void *indices, const void *values, aclsparseIndexType_t idxType,
    aclsparseIndexBase_t idxBase, aclDataType valueType)
{
    if (spVecDescr == nullptr) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    aclsparseSpVecDescr_t tmp = nullptr;
    aclsparseStatus_t st = aclsparseCreateSpVec(&tmp, size, nnz,
        const_cast<void *>(indices), const_cast<void *>(values), idxType, idxBase, valueType);
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        return st;
    }
    *spVecDescr = tmp;
    return ACL_SPARSE_STATUS_SUCCESS;
}

aclsparseStatus_t aclsparseDestroySpVec(aclsparseConstSpVecDescr_t spVecDescr)
{
    if (spVecDescr == nullptr) {
        return ACL_SPARSE_STATUS_SUCCESS;
    }
    delete const_cast<aclsparseSpVecDescr *>(spVecDescr);
    return ACL_SPARSE_STATUS_SUCCESS;
}

aclsparseStatus_t aclsparseSpVecGet(aclsparseSpVecDescr_t spVecDescr, int64_t *size,
    int64_t *nnz, void **indices, void **values, aclsparseIndexType_t *idxType,
    aclsparseIndexBase_t *idxBase, aclDataType *valueType)
{
    if (spVecDescr == nullptr) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (size != nullptr) {
        *size = static_cast<int64_t>(spVecDescr->size);
    }
    if (nnz != nullptr) {
        *nnz = static_cast<int64_t>(spVecDescr->nnz);
    }
    if (indices != nullptr) {
        *indices = spVecDescr->indices;
    }
    if (values != nullptr) {
        *values = spVecDescr->values;
    }
    if (idxType != nullptr) {
        *idxType = spVecDescr->idxType;
    }
    if (idxBase != nullptr) {
        *idxBase = spVecDescr->idxBase;
    }
    if (valueType != nullptr) {
        *valueType = spVecDescr->valueType;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

aclsparseStatus_t aclsparseConstSpVecGet(aclsparseConstSpVecDescr_t spVecDescr, int64_t *size,
    int64_t *nnz, const void **indices, const void **values, aclsparseIndexType_t *idxType,
    aclsparseIndexBase_t *idxBase, aclDataType *valueType)
{
    if (spVecDescr == nullptr) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (size != nullptr) {
        *size = static_cast<int64_t>(spVecDescr->size);
    }
    if (nnz != nullptr) {
        *nnz = static_cast<int64_t>(spVecDescr->nnz);
    }
    if (indices != nullptr) {
        *indices = spVecDescr->indices;
    }
    if (values != nullptr) {
        *values = spVecDescr->values;
    }
    if (idxType != nullptr) {
        *idxType = spVecDescr->idxType;
    }
    if (idxBase != nullptr) {
        *idxBase = spVecDescr->idxBase;
    }
    if (valueType != nullptr) {
        *valueType = spVecDescr->valueType;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

aclsparseStatus_t aclsparseSpVecGetIndexBase(aclsparseConstSpVecDescr_t spVecDescr,
    aclsparseIndexBase_t *idxBase)
{
    if (spVecDescr == nullptr) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (idxBase != nullptr) {
        *idxBase = spVecDescr->idxBase;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

aclsparseStatus_t aclsparseSpVecGetValues(aclsparseSpVecDescr_t spVecDescr, void **values)
{
    if (spVecDescr == nullptr) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (values != nullptr) {
        *values = spVecDescr->values;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

aclsparseStatus_t aclsparseConstSpVecGetValues(aclsparseConstSpVecDescr_t spVecDescr,
    const void **values)
{
    if (spVecDescr == nullptr) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (values != nullptr) {
        *values = spVecDescr->values;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

aclsparseStatus_t aclsparseSpVecSetValues(aclsparseSpVecDescr_t spVecDescr, void *values)
{
    if (spVecDescr == nullptr) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    spVecDescr->values = values;
    return ACL_SPARSE_STATUS_SUCCESS;
}

aclsparseStatus_t aclsparseDnVecGet(aclsparseDnVecDescr_t dnVecDescr, int64_t *size,
                                    void **values, aclDataType *valueType)
{
    if (dnVecDescr == nullptr) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (size != nullptr) {
        *size = static_cast<int64_t>(dnVecDescr->nums);
    }
    if (values != nullptr) {
        *values = dnVecDescr->values;
    }
    if (valueType != nullptr) {
        *valueType = dnVecDescr->valueType;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

aclsparseStatus_t aclsparseConstDnVecGet(aclsparseConstDnVecDescr_t dnVecDescr, int64_t *size,
                                         const void **values, aclDataType *valueType)
{
    if (dnVecDescr == nullptr) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (size != nullptr) {
        *size = static_cast<int64_t>(dnVecDescr->nums);
    }
    if (values != nullptr) {
        *values = dnVecDescr->values;
    }
    if (valueType != nullptr) {
        *valueType = dnVecDescr->valueType;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

aclsparseStatus_t aclsparseDnVecGetValues(aclsparseDnVecDescr_t dnVecDescr, void **values)
{
    if (dnVecDescr == nullptr) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (values != nullptr) {
        *values = dnVecDescr->values;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

aclsparseStatus_t aclsparseConstDnVecGetValues(aclsparseConstDnVecDescr_t dnVecDescr,
                                               const void **values)
{
    if (dnVecDescr == nullptr) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (values != nullptr) {
        *values = dnVecDescr->values;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

aclsparseStatus_t aclsparseDnVecSetValues(aclsparseDnVecDescr_t dnVecDescr, void *values)
{
    if (dnVecDescr == nullptr) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    dnVecDescr->values = values;
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
    if (spMatDescr == nullptr) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
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

aclsparseStatus_t aclsparseSpMatGetFormat(aclsparseConstSpMatDescr_t spMatDescr,
    aclsparseFormat_t *format)
{
    if (spMatDescr == nullptr) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (format != nullptr) {
        *format = spMatDescr->format;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

aclsparseStatus_t aclsparseSpMatGetValues(aclsparseSpMatDescr_t spMatDescr, void **values)
{
    if (spMatDescr == nullptr) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (values != nullptr) {
        *values = spMatDescr->values;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

aclsparseStatus_t aclsparseConstSpMatGetValues(aclsparseConstSpMatDescr_t spMatDescr,
    const void **values)
{
    if (spMatDescr == nullptr) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (values != nullptr) {
        *values = spMatDescr->values;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

aclsparseStatus_t aclsparseSpMatSetValues(aclsparseSpMatDescr_t spMatDescr, void *values)
{
    if (spMatDescr == nullptr) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    spMatDescr->values = values;
    return ACL_SPARSE_STATUS_SUCCESS;
}

aclsparseStatus_t aclsparseSpMatGetSize(aclsparseConstSpMatDescr_t spMatDescr,
    int64_t *rows, int64_t *cols, int64_t *nnz)
{
    if (spMatDescr == nullptr) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (rows != nullptr) {
        *rows = static_cast<int64_t>(spMatDescr->rows);
    }
    if (cols != nullptr) {
        *cols = static_cast<int64_t>(spMatDescr->cols);
    }
    if (nnz != nullptr) {
        *nnz = static_cast<int64_t>(spMatDescr->nnz);
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

aclsparseStatus_t aclsparseCreateDnMat(aclsparseDnMatDescr_t *dnMatDescr,
    int64_t rows, int64_t cols, int64_t ld, void *values,
    aclDataType valueType, aclsparseOrder_t order)
{
    if (dnMatDescr == nullptr) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
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

aclsparseStatus_t aclsparseDnMatGet(aclsparseDnMatDescr_t dnMatDescr, int64_t *rows, int64_t *cols,
                                    int64_t *ld, void **values, aclDataType *valueType,
                                    aclsparseOrder_t *order)
{
    if (dnMatDescr == nullptr) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (rows != nullptr) {
        *rows = dnMatDescr->rows;
    }
    if (cols != nullptr) {
        *cols = dnMatDescr->cols;
    }
    if (ld != nullptr) {
        *ld = dnMatDescr->ld;
    }
    if (values != nullptr) {
        *values = dnMatDescr->values;
    }
    if (valueType != nullptr) {
        *valueType = dnMatDescr->valueType;
    }
    if (order != nullptr) {
        *order = dnMatDescr->order;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

aclsparseStatus_t aclsparseConstDnMatGet(aclsparseConstDnMatDescr_t dnMatDescr, int64_t *rows,
                                         int64_t *cols, int64_t *ld, const void **values,
                                         aclDataType *valueType, aclsparseOrder_t *order)
{
    if (dnMatDescr == nullptr) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (rows != nullptr) {
        *rows = dnMatDescr->rows;
    }
    if (cols != nullptr) {
        *cols = dnMatDescr->cols;
    }
    if (ld != nullptr) {
        *ld = dnMatDescr->ld;
    }
    if (values != nullptr) {
        *values = dnMatDescr->values;
    }
    if (valueType != nullptr) {
        *valueType = dnMatDescr->valueType;
    }
    if (order != nullptr) {
        *order = dnMatDescr->order;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

aclsparseStatus_t aclsparseDnMatGetValues(aclsparseDnMatDescr_t dnMatDescr, void **values)
{
    if (dnMatDescr == nullptr) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (values != nullptr) {
        *values = dnMatDescr->values;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

aclsparseStatus_t aclsparseConstDnMatGetValues(aclsparseConstDnMatDescr_t dnMatDescr,
                                               const void **values)
{
    if (dnMatDescr == nullptr) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (values != nullptr) {
        *values = dnMatDescr->values;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

aclsparseStatus_t aclsparseDnMatSetValues(aclsparseDnMatDescr_t dnMatDescr, void *values)
{
    if (dnMatDescr == nullptr) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    dnMatDescr->values = values;
    return ACL_SPARSE_STATUS_SUCCESS;
}

aclsparseStatus_t aclsparseCreateConstDnMat(aclsparseConstDnMatDescr_t *dnMatDescr,
    int64_t rows, int64_t cols, int64_t ld, const void *values,
    aclDataType valueType, aclsparseOrder_t order)
{
    if (dnMatDescr == nullptr) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
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
