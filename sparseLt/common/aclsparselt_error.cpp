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
 * \file aclsparselt_error.cpp
 * \brief ops-sparseLt 错误码转字符串函数实现。
 */

#include "cann_ops_sparseLt.h"

extern "C" {

const char* aclsparseLtGetErrorName(aclsparseStatus_t status)
{
    switch (status) {
        case ACL_SPARSE_STATUS_SUCCESS:
            return "ACL_SPARSE_STATUS_SUCCESS";
        case ACL_SPARSE_STATUS_NOT_INITIALIZED:
            return "ACL_SPARSE_STATUS_NOT_INITIALIZED";
        case ACL_SPARSE_STATUS_ALLOC_FAILED:
            return "ACL_SPARSE_STATUS_ALLOC_FAILED";
        case ACL_SPARSE_STATUS_INVALID_VALUE:
            return "ACL_SPARSE_STATUS_INVALID_VALUE";
        case ACL_SPARSE_STATUS_ARCH_MISMATCH:
            return "ACL_SPARSE_STATUS_ARCH_MISMATCH";
        case ACL_SPARSE_STATUS_EXECUTION_FAILED:
            return "ACL_SPARSE_STATUS_EXECUTION_FAILED";
        case ACL_SPARSE_STATUS_INTERNAL_ERROR:
            return "ACL_SPARSE_STATUS_INTERNAL_ERROR";
        case ACL_SPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
            return "ACL_SPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED";
        case ACL_SPARSE_STATUS_NOT_SUPPORTED:
            return "ACL_SPARSE_STATUS_NOT_SUPPORTED";
        case ACL_SPARSE_STATUS_INSUFFICIENT_RESOURCES:
            return "ACL_SPARSE_STATUS_INSUFFICIENT_RESOURCES";
        case ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR:
            return "ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR";
        default:
            return "unrecognized error code";
    }
}

const char* aclsparseLtGetErrorString(aclsparseStatus_t status)
{
    switch (status) {
        case ACL_SPARSE_STATUS_SUCCESS:
            return "The operation completed successfully";
        case ACL_SPARSE_STATUS_NOT_INITIALIZED:
            return "The aclsparseLt library was not initialized. This is usually caused by "
                   "the lack of a prior call, an error in the Ascend Runtime API called by "
                   "the routine, or an error in the hardware setup";
        case ACL_SPARSE_STATUS_ALLOC_FAILED:
            return "Resource allocation failed inside the library. This is usually caused by "
                   "a device memory allocation (aclrtMalloc) or by a host memory allocation failure";
        case ACL_SPARSE_STATUS_INVALID_VALUE:
            return "An unsupported value or parameter was passed to the function";
        case ACL_SPARSE_STATUS_ARCH_MISMATCH:
            return "The function requires a feature absent from the device architecture";
        case ACL_SPARSE_STATUS_EXECUTION_FAILED:
            return "The NPU program failed to execute. This is often caused by a launch "
                   "failure of the kernel on the NPU";
        case ACL_SPARSE_STATUS_INTERNAL_ERROR:
            return "An internal operation failed";
        case ACL_SPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
            return "The matrix type is not supported by this function";
        case ACL_SPARSE_STATUS_NOT_SUPPORTED:
            return "The operation or data type combination is currently not supported by the function";
        case ACL_SPARSE_STATUS_INSUFFICIENT_RESOURCES:
            return "The resources for the computation, such as NPU global or shared memory, "
                   "are not sufficient to complete the operation";
        case ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR:
            return "The handle is a null pointer";
        default:
            return "unrecognized error code";
    }
}

} // extern "C"
