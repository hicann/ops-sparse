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
 * \file aclsparse_auxiliary.cpp
 * \brief ops-sparse handle 生命周期、stream、版本信息等辅助函数实现。
 */

#include "cann_ops_sparse.h"
#include "cann_ops_sparse_common.h"
#include "aclsparse_handle_internal.h"

#include <new>

namespace {

/**
 * @brief 将对外 void* 句柄安全转换为内部 aclsparseContext*。
 */
inline aclsparseContext *ToInternal(aclsparseHandle_t handle)
{
    return reinterpret_cast<aclsparseContext *>(handle);
}

} // namespace

extern "C" {

aclsparseStatus_t aclsparseCreate(aclsparseHandle_t *handle)
{
    if (handle == nullptr) {
        return ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR;
    }

    // 防止重复创建覆盖已有指针导致的内存泄漏。
    if (*handle != nullptr) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }

    auto *h = new (std::nothrow) aclsparseContext();
    if (h == nullptr) {
        return ACL_SPARSE_STATUS_ALLOC_FAILED;
    }

    h->stream = nullptr;

    *handle = reinterpret_cast<aclsparseHandle_t>(h);
    return ACL_SPARSE_STATUS_SUCCESS;
}

aclsparseStatus_t aclsparseDestroy(aclsparseHandle_t handle)
{
    if (handle == nullptr) {
        return ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR;
    }

    auto *h = ToInternal(handle);

    // stream 由用户托管，此处仅清理句柄自身字段。
    h->stream = nullptr;

    delete h;
    return ACL_SPARSE_STATUS_SUCCESS;
}

aclsparseStatus_t aclsparseSetStream(aclsparseHandle_t handle, aclrtStream stream)
{
    if (handle == nullptr) {
        return ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR;
    }

    auto *h = ToInternal(handle);
    h->stream = stream;
    return ACL_SPARSE_STATUS_SUCCESS;
}

aclsparseStatus_t aclsparseGetStream(aclsparseHandle_t handle, aclrtStream *stream)
{
    if (handle == nullptr) {
        return ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR;
    }
    if (stream == nullptr) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }

    auto *h = ToInternal(handle);
    *stream = h->stream;
    return ACL_SPARSE_STATUS_SUCCESS;
}

aclsparseStatus_t aclsparseGetVersion(aclsparseHandle_t handle, int *version)
{
    if (version == nullptr) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }

    *version = ACLSPARSE_VERSION;
    return ACL_SPARSE_STATUS_SUCCESS;
}

} // extern "C"
