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
 * \file aclsparselt_matmul_plan_internal.h
 * \brief ops-sparseLt matmul plan 内部结构定义（不对外暴露）。
 */

#ifndef ACLSPARSELT_MATMUL_PLAN_INTERNAL_H
#define ACLSPARSELT_MATMUL_PLAN_INTERNAL_H

#include "cann_ops_sparseLt.h"

/**
 * @brief matmul plan 内部结构体。
 *
 * 绑定 matmul 描述符与算法选择描述符，作为后续 matmul 执行的规划对象。
 * matmulDescr 与 algSelection 均为非所有权引用（Destroy 时不会销毁它们）。
 */
struct aclsparseLtMatmulPlan {
    aclsparseLtMatmulDescriptor_t matmulDescr = nullptr;       ///< matmul 描述符引用（非所有权）
    aclsparseLtMatmulAlgSelection_t algSelection = nullptr;    ///< 算法选择描述符引用（非所有权）
};

/** @brief 对外句柄安全转换为内部结构体指针。 */
inline aclsparseLtMatmulPlan* ToInternal(aclsparseLtMatmulPlan_t desc)
{
    return reinterpret_cast<aclsparseLtMatmulPlan*>(desc);
}

#endif // ACLSPARSELT_MATMUL_PLAN_INTERNAL_H
