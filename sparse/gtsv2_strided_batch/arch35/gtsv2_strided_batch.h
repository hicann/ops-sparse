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
 * \file gtsv2_strided_batch.h
 * \brief aclsparseSgtsv2StridedBatch Host 侧辅助函数（handle 转换）。
 *
 * gtsv2 系列为 Legacy API（精度前缀 S、扁平参数、无 MatDescr），
 * 本头文件仅提供 handle 内部结构体转换函数。
 */

#ifndef GTSV2_STRIDED_BATCH_H_
#define GTSV2_STRIDED_BATCH_H_

#include "cann_ops_sparse.h"
#include "aclsparse_handle_internal.h"

/// 将 opaque handle 转为内部 aclsparseContext 指针
inline struct aclsparseContext *Gtsv2ToInternalHandle(aclsparseHandle_t handle)
{
    return reinterpret_cast<struct aclsparseContext *>(handle);
}

#endif  // GTSV2_STRIDED_BATCH_H_
