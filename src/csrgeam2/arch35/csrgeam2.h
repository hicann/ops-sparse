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
 * \file csrgeam2.h
 * \brief csrgeam2 Host 侧辅助函数：MatDescr 解析、handle 转换。
 */

#ifndef CSRGEAM2_H_
#define CSRGEAM2_H_

#include <cstdint>

#include "acl/acl.h"
#include "cann_ops_sparse.h"
#include "aclsparse_handle_internal.h"
#include "aclsparse_descr_internal.h"

/// 将 opaque handle 转为内部结构体指针
inline struct aclsparseContext *Csrgeam2ToInternalHandle(aclsparseHandle_t handle)
{
    return reinterpret_cast<struct aclsparseContext *>(handle);
}

/// 校验 MatDescr：type 必须为 GENERAL，indexBase 必须为 ZERO 或 ONE
inline aclsparseStatus_t Csrgeam2ValidateMatDescr(const char *apiTag,
                                                 const char *paramName,
                                                 const aclsparseMatDescr_t descr)
{
    if (descr == nullptr) {
        OP_LOGE(apiTag, "%s is nullptr", paramName);
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }

    aclsparseMatrixType_t matType = aclsparseGetMatType(descr);
    if (matType != ACL_SPARSE_MATRIX_TYPE_GENERAL) {
        OP_LOGE(apiTag, "%s: unsupported matrix type %d",
                paramName, static_cast<int>(matType));
        return ACL_SPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED;
    }

    aclsparseIndexBase_t indexBase = aclsparseGetMatIndexBase(descr);
    if (indexBase != ACL_SPARSE_INDEX_BASE_ZERO &&
        indexBase != ACL_SPARSE_INDEX_BASE_ONE) {
        OP_LOGE(apiTag, "%s: invalid indexBase %d",
                paramName, static_cast<int>(indexBase));
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }

    return ACL_SPARSE_STATUS_SUCCESS;
}

/// 从 MatDescr 提取 indexBase 的整数值 (0 或 1)
/// Precondition: descr must have been validated by Csrgeam2ValidateMatDescr
/// (legal indexBase). For illegal base values, this function silently maps to 0 (#24).
inline int32_t Csrgeam2GetBase(const aclsparseMatDescr_t descr)
{
    aclsparseIndexBase_t base = aclsparseGetMatIndexBase(descr);
    return (base == ACL_SPARSE_INDEX_BASE_ONE) ? 1 : 0;
}

#endif  // CSRGEAM2_H_
