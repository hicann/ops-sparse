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
 * \file aclsparselt_auxiliary.cpp
 * \brief ops-sparseLt handle 生命周期管理函数实现。
 */

#include "cann_ops_sparseLt.h"
#include "aclsparselt_handle_internal.h"

#include <new>

namespace {

/**
 * @brief 将对外句柄安全转换为内部 aclsparseLtContext*。
 */
inline aclsparseLtContext* ToInternal(aclsparseLtHandle_t handle)
{
    return reinterpret_cast<aclsparseLtContext*>(handle);
}

} // namespace

extern "C" {

aclsparseStatus_t aclsparseLtInit(aclsparseLtHandle_t* handle)
{
    if (handle == nullptr) {
        return ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR;
    }

    if (*handle != nullptr) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }

    auto* ctx = new (std::nothrow) aclsparseLtContext();
    if (ctx == nullptr) {
        return ACL_SPARSE_STATUS_ALLOC_FAILED;
    }

    int32_t deviceId = 0;
    aclError aclRet = aclrtGetDevice(&deviceId);
    if (aclRet != ACL_SUCCESS) {
        delete ctx;
        return ACL_SPARSE_STATUS_NOT_INITIALIZED;
    }
    ctx->device_id = deviceId;
    ctx->stream = nullptr;
    ctx->initialized = true;

    *handle = reinterpret_cast<aclsparseLtHandle_t>(ctx);
    return ACL_SPARSE_STATUS_SUCCESS;
}

aclsparseStatus_t aclsparseLtDestroy(const aclsparseLtHandle_t* handle)
{
    if (handle == nullptr) {
        return ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR;
    }

    if (*handle == nullptr) {
        return ACL_SPARSE_STATUS_SUCCESS;
    }

    auto* ctx = const_cast<aclsparseLtContext*>(*handle);

    if (ctx->stream != nullptr) {
        aclError aclRet = aclrtSynchronizeStream(ctx->stream);
        if (aclRet != ACL_SUCCESS) {
            return ACL_SPARSE_STATUS_EXECUTION_FAILED;
        }
    }

    ctx->stream = nullptr;
    ctx->initialized = false;

    delete ctx;
    *const_cast<aclsparseLtHandle_t*>(handle) = nullptr;
    return ACL_SPARSE_STATUS_SUCCESS;
}

} // extern "C"
