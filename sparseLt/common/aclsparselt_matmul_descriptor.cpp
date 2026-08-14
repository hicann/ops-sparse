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
 * \file aclsparselt_matmul_descriptor.cpp
 * \brief ops-sparseLt matmul 描述符初始化与销毁函数实现。
 */

#include "cann_ops_sparseLt.h"
#include "aclsparselt_matmul_descriptor_internal.h"

#include <new>

namespace {

/// C/D 矩阵 rows/cols 维度上限
constexpr int64_t kMaxDim = 2097120;

/**
 * @brief 判断 op 是否为合法的 aclsparseOperation_t 枚举值。
 *
 * descriptor init 阶段仅支持 NON_TRANSPOSE / TRANSPOSE，
 * 不支持 CONJUGATE_TRANSPOSE。
 */
inline bool IsValidOperation(aclsparseOperation_t op)
{
    return op == ACL_SPARSE_OP_NON_TRANSPOSE ||
           op == ACL_SPARSE_OP_TRANSPOSE;
}

/**
 * @brief 判断 computeType 是否为合法的 aclsparseComputeType_t 枚举值。
 */
inline bool IsValidComputeType(aclsparseComputeType_t t)
{
    return t == ACL_SPARSE_COMPUTE_16F ||
           t == ACL_SPARSE_COMPUTE_32F ||
           t == ACL_SPARSE_COMPUTE_32I;
}

/**
 * @brief 判断 valueType 是否属于 INT8/FP8/FP4 低精度档（需施加操作布局组合约束）。
 *
 * 1 字节高精度类型 HIFLOAT8 不在此档，与 FP32/FP16/BF16 一样不受组合约束。
 */
inline bool IsLowPrecisionType(aclDataType t)
{
    return t == ACL_INT8 ||
           t == ACL_FLOAT8_E4M3FN ||
           t == ACL_FLOAT8_E5M2 ||
           t == ACL_FLOAT8_E8M0 ||
           t == ACL_FLOAT4_E2M1 ||
           t == ACL_FLOAT4_E1M2;
}

/**
 * @brief 判断矩阵描述符引用是否有效（指针非空且句柄已初始化）。
 */
inline bool IsMatDescrValid(const aclsparseLtMatDescriptor_t* mat)
{
    return mat != nullptr && *mat != nullptr;
}

/**
 * @brief 判断矩阵描述符的 rows/cols 是否在维度上限内（<= 2097120）。
 */
inline bool IsDimWithinLimit(const aclsparseLtMatDescriptor* desc)
{
    return desc->rows <= kMaxDim && desc->cols <= kMaxDim;
}

/**
 * @brief 校验 INT8/FP8/FP4 类型下的操作与布局组合约束。
 *
 * 缩写：N = ACL_SPARSE_OP_NON_TRANSPOSE, T = ACL_SPARSE_OP_TRANSPOSE。
 * | orderA | orderB | 要求 opA | 要求 opB |
 * | COL    | COL    | T        | N        |
 * | ROW    | ROW    | N        | T        |
 * | ROW    | COL    | N        | N        |
 * | COL    | ROW    | T        | T        |
 * CONJUGATE_TRANSPOSE 已在 IsValidOperation 中被拒绝，不会进入此函数。
 */
inline bool IsValidOpLayoutCombination(aclsparseOrder_t orderA, aclsparseOrder_t orderB,
                                       aclsparseOperation_t opA, aclsparseOperation_t opB)
{
    if (orderA == ACL_SPARSE_ORDER_COL && orderB == ACL_SPARSE_ORDER_COL) {
        return opA == ACL_SPARSE_OP_TRANSPOSE && opB == ACL_SPARSE_OP_NON_TRANSPOSE;
    }
    if (orderA == ACL_SPARSE_ORDER_ROW && orderB == ACL_SPARSE_ORDER_ROW) {
        return opA == ACL_SPARSE_OP_NON_TRANSPOSE && opB == ACL_SPARSE_OP_TRANSPOSE;
    }
    if (orderA == ACL_SPARSE_ORDER_ROW && orderB == ACL_SPARSE_ORDER_COL) {
        return opA == ACL_SPARSE_OP_NON_TRANSPOSE && opB == ACL_SPARSE_OP_NON_TRANSPOSE;
    }
    if (orderA == ACL_SPARSE_ORDER_COL && orderB == ACL_SPARSE_ORDER_ROW) {
        return opA == ACL_SPARSE_OP_TRANSPOSE && opB == ACL_SPARSE_OP_TRANSPOSE;
    }
    return false;
}

/**
 * @brief 校验 matmul 描述符的参数与约束。
 *
 * 校验顺序：mat nullptr → opA/opB 枚举 → computeType 枚举 → 结构化稀疏位置 →
 *           C/D 一致性 → 操作布局组合 → 维度上下限。
 * 所有校验在内存分配之前完成。
 */
aclsparseStatus_t ValidateMatmulDescriptorParams(
    aclsparseOperation_t opA, aclsparseOperation_t opB,
    const aclsparseLtMatDescriptor_t* matA,
    const aclsparseLtMatDescriptor_t* matB,
    const aclsparseLtMatDescriptor_t* matC,
    const aclsparseLtMatDescriptor_t* matD,
    aclsparseComputeType_t computeType)
{
    if (!IsMatDescrValid(matA) || !IsMatDescrValid(matB) ||
        !IsMatDescrValid(matC) || !IsMatDescrValid(matD)) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (!IsValidOperation(opA) || !IsValidOperation(opB)) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (!IsValidComputeType(computeType)) {
        return ACL_SPARSE_STATUS_NOT_SUPPORTED;
    }

    const aclsparseLtMatDescriptor* descA = ToInternalConst(matA);
    const aclsparseLtMatDescriptor* descB = ToInternalConst(matB);
    const aclsparseLtMatDescriptor* descC = ToInternalConst(matC);
    const aclsparseLtMatDescriptor* descD = ToInternalConst(matD);

    // 结构化稀疏位置约束：matA 与 matB 有且仅有一个为 structured
    if (descA->isStructured == descB->isStructured) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    // matC/matD 须为 dense（非 structured）
    if (descC->isStructured || descD->isStructured) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    // C/D 一致性约束：相同 ld 和 order
    if (descC->ld != descD->ld || descC->order != descD->order) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    // 操作与布局组合约束（仅 INT8/FP8/FP4 类型，以 matA.valueType 为准）
    if (IsLowPrecisionType(descA->valueType)) {
        if (!IsValidOpLayoutCombination(descA->order, descB->order, opA, opB)) {
            return ACL_SPARSE_STATUS_INVALID_VALUE;
        }
    }
    // 维度上下限约束：matC/matD 的 rows、cols 均 <= 2097120
    if (!IsDimWithinLimit(descC) || !IsDimWithinLimit(descD)) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

/**
 * @brief 填充 matmul 描述符字段。
 */
inline void FillMatmulDescriptor(aclsparseLtMatmulDescriptor* desc,
                                 aclsparseOperation_t opA, aclsparseOperation_t opB,
                                 const aclsparseLtMatDescriptor_t* matA,
                                 const aclsparseLtMatDescriptor_t* matB,
                                 const aclsparseLtMatDescriptor_t* matC,
                                 const aclsparseLtMatDescriptor_t* matD,
                                 aclsparseComputeType_t computeType)
{
    desc->opA = opA;
    desc->opB = opB;
    desc->matA = *matA;
    desc->matB = *matB;
    desc->matC = *matC;
    desc->matD = *matD;
    desc->computeType = computeType;
}

} // namespace

extern "C" {

aclsparseStatus_t aclsparseLtMatmulDescriptorInit(
    const aclsparseLtHandle_t*             handle,
    aclsparseLtMatmulDescriptor_t*         matmulDescr,
    aclsparseOperation_t                   opA,
    aclsparseOperation_t                   opB,
    const aclsparseLtMatDescriptor_t*      matA,
    const aclsparseLtMatDescriptor_t*      matB,
    const aclsparseLtMatDescriptor_t*      matC,
    const aclsparseLtMatDescriptor_t*      matD,
    aclsparseComputeType_t                 computeType)
{
    if (handle == nullptr) {
        return ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR;
    }
    if (matmulDescr == nullptr || *matmulDescr != nullptr) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }

    aclsparseStatus_t st = ValidateMatmulDescriptorParams(opA, opB, matA, matB, matC, matD, computeType);
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        return st;
    }

    auto* desc = new (std::nothrow) aclsparseLtMatmulDescriptor();
    if (desc == nullptr) {
        return ACL_SPARSE_STATUS_ALLOC_FAILED;
    }

    FillMatmulDescriptor(desc, opA, opB, matA, matB, matC, matD, computeType);
    *matmulDescr = desc;
    return ACL_SPARSE_STATUS_SUCCESS;
}

aclsparseStatus_t aclsparseLtMatmulDescriptorDestroy(aclsparseLtMatmulDescriptor_t* matmulDescr)
{
    if (matmulDescr == nullptr) {
        return ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR;
    }
    if (*matmulDescr == nullptr) {
        return ACL_SPARSE_STATUS_SUCCESS;
    }

    auto* desc = *matmulDescr;
    // matA/matB/matC/matD 为非所有权引用，Destroy 不销毁它们
    delete desc;
    *matmulDescr = nullptr;
    return ACL_SPARSE_STATUS_SUCCESS;
}

} // extern "C"
