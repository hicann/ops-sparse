/**
 * ----------------------------------------------------------------------------------------------------------
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ----------------------------------------------------------------------------------------------------------
 */

#ifndef CANN_OPS_SPARSELT_H_
#define CANN_OPS_SPARSELT_H_

#include "cann_ops_sparse.h"

// aclsparseLtHandle_t: opaque handle for the aclsparseLt library context.
struct aclsparseLtContext;
typedef struct aclsparseLtContext* aclsparseLtHandle_t;

#ifdef __cplusplus
extern "C" {
#endif

/* ========== Library Management Functions ========== */

/**
 * @brief 初始化 aclsparseLt 库句柄。
 *
 * 在主机端分配轻量级硬件资源，创建 aclsparseLt 库上下文。
 * 必须在调用任何其他 aclsparseLt 函数之前调用。
 * 库上下文绑定到当前 NPU 设备，多设备使用需为每个设备创建独立 handle。
 *
 * @param handle OUT, HOST, aclsparseLt 库句柄输出参数。调用前 *handle 必须为 nullptr。
 * @return ACL_SPARSE_STATUS_SUCCESS 成功
 *         ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR handle 指针为空
 *         ACL_SPARSE_STATUS_INVALID_VALUE *handle 非空（防止覆盖已有句柄导致泄漏）
 *         ACL_SPARSE_STATUS_ALLOC_FAILED 内存分配失败
 */
aclsparseStatus_t aclsparseLtInit(aclsparseLtHandle_t* handle);

/**
 * @brief 释放 aclsparseLt 库句柄占用的全部资源。
 *
 * 与特定 handle 关联的最后一次调用。调用后该 handle 不可再使用。
 *
 * @param handle IN, HOST, 指向要销毁的 aclsparseLt 库句柄的 const 指针。
 *               若 handle 指针本身为 nullptr 则返回 HANDLE_IS_NULLPTR；
 *               若 handle 指向的句柄值为 nullptr 则视为空操作，返回 SUCCESS。
 * @return ACL_SPARSE_STATUS_SUCCESS 成功
 *         ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR handle 指针为空
 */
aclsparseStatus_t aclsparseLtDestroy(const aclsparseLtHandle_t* handle);

/**
 * @brief 返回状态码对应的枚举名字符串。
 *
 * @param status IN, HOST, 要转换的状态码。
 * @return const char* 指向枚举名字符串的指针（如 "ACL_SPARSE_STATUS_SUCCESS"）。
 *         未识别的状态码返回 "unrecognized error code"。
 */
const char* aclsparseLtGetErrorName(aclsparseStatus_t status);

/**
 * @brief 返回状态码对应的描述性字符串。
 *
 * @param status IN, HOST, 要转换的状态码。
 * @return const char* 指向描述性字符串的指针。
 *         未识别的状态码返回 "unrecognized error code"。
 */
const char* aclsparseLtGetErrorString(aclsparseStatus_t status);

#ifdef __cplusplus
}
#endif

#endif
