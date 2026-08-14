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
 * \file aclsparselt_matmul_descriptor_internal.h
 * \brief ops-sparseLt matmul 描述符内部结构定义（不对外暴露）。
 */

#ifndef ACLSPARSELT_MATMUL_DESCRIPTOR_INTERNAL_H
#define ACLSPARSELT_MATMUL_DESCRIPTOR_INTERNAL_H

#include "cann_ops_sparseLt.h"
#include "aclsparselt_mat_descriptor_internal.h"

/**
 * @brief matmul 描述符内部结构体。
 *
 * 持有 matmul 运算的完整元信息：操作类型、四个矩阵描述符引用、计算精度。
 * matA/matB/matC/matD 为非所有权引用（Destroy 时不会销毁它们）。
 */
struct aclsparseLtMatmulDescriptor {
    aclsparseOperation_t opA = ACL_SPARSE_OP_NON_TRANSPOSE; ///< 作用于 A 的操作
    aclsparseOperation_t opB = ACL_SPARSE_OP_NON_TRANSPOSE; ///< 作用于 B 的操作
    aclsparseLtMatDescriptor_t matA = nullptr;  ///< 矩阵 A 描述符引用（非所有权）
    aclsparseLtMatDescriptor_t matB = nullptr;  ///< 矩阵 B 描述符引用（非所有权）
    aclsparseLtMatDescriptor_t matC = nullptr;  ///< 矩阵 C 描述符引用（非所有权）
    aclsparseLtMatDescriptor_t matD = nullptr;  ///< 矩阵 D 描述符引用（非所有权）
    aclsparseComputeType_t computeType = ACL_SPARSE_COMPUTE_16F; ///< 计算精度
};

/** @brief 对外句柄安全转换为内部结构体指针。 */
inline aclsparseLtMatmulDescriptor* ToInternal(aclsparseLtMatmulDescriptor_t desc)
{
    return reinterpret_cast<aclsparseLtMatmulDescriptor*>(desc);
}

#endif // ACLSPARSELT_MATMUL_DESCRIPTOR_INTERNAL_H
