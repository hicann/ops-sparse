/**
 * ----------------------------------------------------------------------------------------------------------
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ----------------------------------------------------------------------------------------------------------
 */

/*!
 * \file coo_get_host.cpp
 * \brief aclsparseCooGet / aclsparseConstCooGet Host 侧实现（arch35）。
 *
 * 对标 cuSPARSE cusparseCreateCoo / cusparseCooGet。CooGet 为 SparseAccessor：
 * 从 COO 稀疏矩阵描述符中读出 rows / cols / nnz / cooRowInd / cooColInd /
 * cooValues / cooIdxType / idxBase / valueType 全部字段，语义与 cuSPARSE
 * 完全一致，确保 cuSPARSE 用户可无缝迁移。
 *
 * 描述符内部结构复用 aclsparseSpMatDescr（见 aclsparse_descr_internal.h）：
 *   - ptrs  <- cooRowInd
 *   - idxs  <- cooColInd
 *   - values<- cooValues
 *   - ptrType == IdxType == cooIdxType（COO 行/列索引同类型）
 *   - baseType <- idxBase, valueType <- valueType, format <- ACL_SPARSE_FORMAT_COO
 */

#include "log/log.h"
#include "cann_ops_sparse.h"
#include "aclsparse_descr_internal.h"

template <typename PtrOut>
static inline aclsparseStatus_t CooGetImpl(const char *fn, const aclsparseSpMatDescr *d,
    int64_t *rows, int64_t *cols, int64_t *nnz,
    PtrOut cooRowInd, PtrOut cooColInd, PtrOut cooValues,
    aclsparseIndexType_t *cooIdxType, aclsparseIndexBase_t *idxBase, aclDataType *valueType)
{
    if (d == nullptr) {
        OP_LOGE(fn, "spMatDescr is nullptr");
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (d->format != ACL_SPARSE_FORMAT_COO) {
        OP_LOGE(fn, "descriptor format %d is not COO", static_cast<int>(d->format));
        return ACL_SPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED;
    }
    if (rows != nullptr) {
        *rows = static_cast<int64_t>(d->rows);
    }
    if (cols != nullptr) {
        *cols = static_cast<int64_t>(d->cols);
    }
    if (nnz != nullptr) {
        *nnz = static_cast<int64_t>(d->nnz);
    }
    if (cooRowInd != nullptr) {
        *cooRowInd = d->ptrs;
    }
    if (cooColInd != nullptr) {
        *cooColInd = d->idxs;
    }
    if (cooValues != nullptr) {
        *cooValues = d->values;
    }
    if (cooIdxType != nullptr) {
        *cooIdxType = d->ptrType;
    }
    if (idxBase != nullptr) {
        *idxBase = d->baseType;
    }
    if (valueType != nullptr) {
        *valueType = d->valueType;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

extern "C" {

aclsparseStatus_t aclsparseCooGet(aclsparseSpMatDescr_t spMatDescr, int64_t *rows, int64_t *cols, int64_t *nnz,
    void **cooRowInd, void **cooColInd, void **cooValues, aclsparseIndexType_t *cooIdxType,
    aclsparseIndexBase_t *idxBase, aclDataType *valueType)
{
    return CooGetImpl("aclsparseCooGet", spMatDescr, rows, cols, nnz, cooRowInd, cooColInd, cooValues,
        cooIdxType, idxBase, valueType);
}

aclsparseStatus_t aclsparseConstCooGet(aclsparseConstSpMatDescr_t spMatDescr, int64_t *rows, int64_t *cols,
    int64_t *nnz, const void **cooRowInd, const void **cooColInd, const void **cooValues,
    aclsparseIndexType_t *cooIdxType, aclsparseIndexBase_t *idxBase, aclDataType *valueType)
{
    return CooGetImpl("aclsparseConstCooGet", spMatDescr, rows, cols, nnz, cooRowInd, cooColInd, cooValues,
        cooIdxType, idxBase, valueType);
}

} // extern "C"
