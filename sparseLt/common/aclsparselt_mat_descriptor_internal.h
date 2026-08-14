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
 * \file aclsparselt_mat_descriptor_internal.h
 * \brief ops-sparseLt 矩阵描述符内部结构定义（不对外暴露）。
 */

#ifndef ACLSPARSELT_MAT_DESCRIPTOR_INTERNAL_H
#define ACLSPARSELT_MAT_DESCRIPTOR_INTERNAL_H

#include <cstdint>
#include "acl/acl.h"
#include "cann_ops_sparseLt.h"

/**
 * @brief 矩阵描述符内部结构体（dense / structured 共用）。
 *
 * 通过 isStructured 字段区分稠密与结构化稀疏矩阵。
 * 对外完全隐藏，仅在 sparseLt/common/ 实现文件中可见。
 */
struct aclsparseLtMatDescriptor {
    int64_t rows = 0;                              ///< 行数
    int64_t cols = 0;                              ///< 列数
    int64_t ld = 0;                                ///< leading dimension
    uint32_t alignment = 0;                        ///< 内存对齐字节数
    aclDataType valueType = ACL_DT_UNDEFINED;      ///< 矩阵数据存储类型
    aclsparseOrder_t order = ACL_SPARSE_ORDER_ROW; ///< 内存布局
    aclsparseLtSparsity_t sparsity = ACL_SPARSE_LT_SPARSITY_50_PERCENT; ///< 稀疏模式（仅 structured 有效）
    bool isStructured = false;                     ///< false=dense, true=structured
    // —— batch 预留字段（本期不可通过 API 配置，SetAttribute 未实现）——
    int32_t numBatches = 1;                        ///< batch 数量，默认 1（非批量）
    int64_t batchStride = 0;                       ///< batch 步长，默认 0
};

/** @brief 对外句柄安全转换为内部结构体指针。 */
inline aclsparseLtMatDescriptor* ToInternal(aclsparseLtMatDescriptor_t desc)
{
    return reinterpret_cast<aclsparseLtMatDescriptor*>(desc);
}

inline const aclsparseLtMatDescriptor* ToInternalConst(const aclsparseLtMatDescriptor_t* desc)
{
    return reinterpret_cast<const aclsparseLtMatDescriptor*>(*desc);
}

#endif // ACLSPARSELT_MAT_DESCRIPTOR_INTERNAL_H
