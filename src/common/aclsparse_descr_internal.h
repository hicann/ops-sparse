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
 * \file aclsparse_descr_internal.h
 * \brief ops-sparse 描述符内部结构体（稀疏矩阵 / 稠密向量 / 稠密矩阵）。
 *
 * 这些结构体仅在库实现内部可见，对外只前向声明为不透明的 struct，
 * 对外句柄类型 aclsparseSpMatDescr_t / aclsparseDnVecDescr_t / aclsparseDnMatDescr_t
 * 即为指向这些 struct 的指针（含 const 变体）。
 * 与 aclsparseContext 保持一致：用默认成员初始化 + new/delete 管理生命周期，
 * 避免把内部布局暴露到对外头文件、也避免 malloc 后字段未初始化的隐患。
 */

#ifndef ACLSPARSE_DESCR_INTERNAL_H
#define ACLSPARSE_DESCR_INTERNAL_H

#include <cstdint>
#include <acl/acl.h>
#include "cann_ops_sparse.h"

// 稀疏矩阵描述符内部结构（CSR / CSC 共用）。
struct aclsparseSpMatDescr {
    // 由 *Preprocess 写入：记录当前已预处理(active)的 workspace buffer。
    // SpMM/SpMV 据此决定走快路径(复用)还是就地重算（active buffer 机制）。
    const void *activeBuffer = nullptr;
    aclsparseFormat_t format{};
    uint64_t rows = 0;
    uint64_t cols = 0;
    uint64_t nnz = 0;
    void *ptrs = nullptr;
    void *idxs = nullptr;
    void *values = nullptr;
    aclsparseIndexBase_t baseType{};
    aclsparseIndexType_t ptrType{};
    aclsparseIndexType_t IdxType{};
    aclDataType valueType{};
};

// 校验 CSR 稀疏矩阵索引类型：当前仅支持 32 位且 ptr/col 类型一致。
inline aclsparseStatus_t AclsparseValidateSupportedCsrIndexTypes(aclsparseIndexType_t ptrType,
                                                                 aclsparseIndexType_t idxType)
{
    if (ptrType != idxType) {
        return ACL_SPARSE_STATUS_NOT_SUPPORTED;
    }
    if (ptrType == ACL_SPARSE_INDEX_32I) {
        return ACL_SPARSE_STATUS_SUCCESS;
    }
    return ACL_SPARSE_STATUS_NOT_SUPPORTED;
}

// 稠密向量描述符内部结构（SpMV 的 x / y）。
struct aclsparseDnVecDescr {
    uint64_t nums = 0;
    void *values = nullptr;
    aclDataType valueType{};
};

// 稠密矩阵描述符内部结构（SpMM 的 B / C）。
struct aclsparseDnMatDescr {
    int64_t rows = 0;
    int64_t cols = 0;
    int64_t ld = 0;
    aclsparseOrder_t order{};
    void *values = nullptr;
    aclDataType valueType{};
};

#endif // ACLSPARSE_DESCR_INTERNAL_H
