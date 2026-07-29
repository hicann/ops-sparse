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
 * \file scatter.h
 * \brief aclsparseScatter Host 侧辅助函数：描述符转换。
 *
 * vecX 为 aclsparseConstSpVecDescr_t（const），vecY 为 aclsparseDnVecDescr_t（非 const，输出）。
 * const 描述符的 const_cast 统一在 ScatterToSpVecInner 中完成，禁止在业务代码中直接 cast。
 */

#ifndef SCATTER_H_
#define SCATTER_H_

#include "cann_ops_sparse.h"
#include "aclsparse_handle_internal.h"
#include "aclsparse_descr_internal.h"

/// 将 opaque handle 转为内部结构体指针
inline struct aclsparseContext *ScatterToInternalHandle(aclsparseHandle_t handle)
{
    return reinterpret_cast<struct aclsparseContext *>(handle);
}

/// 将 const SpVec 描述符转为内部结构体指针（解除 const，仅读用途）
inline struct aclsparseSpVecDescr *ScatterToSpVecInner(aclsparseConstSpVecDescr_t desc)
{
    return const_cast<struct aclsparseSpVecDescr *>(
        reinterpret_cast<const struct aclsparseSpVecDescr *>(desc));
}

/// 将 DnVec 描述符转为内部结构体指针（vecY 非 const，输出）
inline struct aclsparseDnVecDescr *ScatterToDnVecInner(aclsparseDnVecDescr_t desc)
{
    return reinterpret_cast<struct aclsparseDnVecDescr *>(desc);
}

#endif  // SCATTER_H_
