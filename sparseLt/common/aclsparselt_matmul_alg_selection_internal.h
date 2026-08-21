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
 * \file aclsparselt_matmul_alg_selection_internal.h
 * \brief ops-sparseLt matmul 算法选择描述符内部结构定义（不对外暴露）。
 */

#ifndef ACLSPARSELT_MATMUL_ALG_SELECTION_INTERNAL_H
#define ACLSPARSELT_MATMUL_ALG_SELECTION_INTERNAL_H

#include "cann_ops_sparseLt.h"

/**
 * @brief matmul 算法选择描述符内部结构体。
 *
 * 持有算法模式及其关联的 matmul 描述符引用（非所有权）。
 */
struct aclsparseLtMatmulAlgSelection {
    aclsparseLtMatmulDescriptor_t matmulDescr = nullptr;            ///< matmul 描述符引用（非所有权）
    aclsparseLtMatmulAlg_t alg = ACL_SPARSE_LT_MATMUL_ALG_DEFAULT;  ///< 算法模式
};

/** @brief 判断 matmul 描述符引用是否有效（指针非空且句柄已初始化）。 */
inline bool IsMatmulDescrValid(const aclsparseLtMatmulDescriptor_t* matmulDescr)
{
    return matmulDescr != nullptr && *matmulDescr != nullptr;
}

/** @brief 对外句柄安全转换为内部结构体指针。 */
inline aclsparseLtMatmulAlgSelection* ToInternal(aclsparseLtMatmulAlgSelection_t desc)
{
    return reinterpret_cast<aclsparseLtMatmulAlgSelection*>(desc);
}

#endif // ACLSPARSELT_MATMUL_ALG_SELECTION_INTERNAL_H
