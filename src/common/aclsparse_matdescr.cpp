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
 * \file aclsparse_matdescr.cpp
 * \brief Legacy MatDescr 生命周期与属性管理实现。
 */

#include "cann_ops_sparse.h"
#include "aclsparse_descr_internal.h"

#include <new>

extern "C" {

aclsparseStatus_t aclsparseCreateMatDescr(aclsparseMatDescr_t *descr)
{
    if (descr == nullptr) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }

    auto *d = new (std::nothrow) aclsparseMatDescr();
    if (d == nullptr) {
        return ACL_SPARSE_STATUS_ALLOC_FAILED;
    }

    *descr = d;
    return ACL_SPARSE_STATUS_SUCCESS;
}

aclsparseStatus_t aclsparseDestroyMatDescr(aclsparseMatDescr_t descr)
{
    if (descr == nullptr) {
        return ACL_SPARSE_STATUS_SUCCESS;
    }

    delete descr;
    return ACL_SPARSE_STATUS_SUCCESS;
}

void aclsparseSetMatType(aclsparseMatDescr_t descr, aclsparseMatrixType_t type)
{
    if (descr != nullptr) {
        descr->type = type;
    }
}

void aclsparseSetMatIndexBase(aclsparseMatDescr_t descr, aclsparseIndexBase_t base)
{
    if (descr != nullptr) {
        descr->indexBase = base;
    }
}

void aclsparseSetMatDiagType(aclsparseMatDescr_t descr, aclsparseDiagType_t diagType)
{
    if (descr != nullptr) {
        descr->diagType = diagType;
    }
}

void aclsparseSetMatFillMode(aclsparseMatDescr_t descr, aclsparseFillMode_t fillMode)
{
    if (descr != nullptr) {
        descr->fillMode = fillMode;
    }
}

aclsparseMatrixType_t aclsparseGetMatType(aclsparseMatDescr_t descr)
{
    if (descr == nullptr) {
        return ACL_SPARSE_MATRIX_TYPE_GENERAL;
    }
    return descr->type;
}

aclsparseIndexBase_t aclsparseGetMatIndexBase(aclsparseMatDescr_t descr)
{
    if (descr == nullptr) {
        return ACL_SPARSE_INDEX_BASE_ZERO;
    }
    return descr->indexBase;
}

aclsparseDiagType_t aclsparseGetMatDiagType(aclsparseMatDescr_t descr)
{
    if (descr == nullptr) {
        return ACL_SPARSE_DIAG_TYPE_NON_UNIT;
    }
    return descr->diagType;
}

aclsparseFillMode_t aclsparseGetMatFillMode(aclsparseMatDescr_t descr)
{
    if (descr == nullptr) {
        return ACL_SPARSE_FILL_MODE_LOWER;
    }
    return descr->fillMode;
}

} // extern "C"
