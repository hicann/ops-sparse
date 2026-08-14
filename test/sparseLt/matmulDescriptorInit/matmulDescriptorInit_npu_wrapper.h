/**
 * ----------------------------------------------------------------------------------------------------------
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under
 * the terms and conditions of CANN Open Software License Agreement Version 2.0
 * (the "License"). Please refer to the License for details. You may not use
 * this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
 * AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
 * FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
 * for the full text of the License.
 * ----------------------------------------------------------------------------------------------------------
 */

/**
 * @file matmulDescriptorInit_npu_wrapper.h
 * @brief Host 侧调用封装：RAII Guard（Handle/MatDescr/MatmulDescr）+ 字段验证
 * helper。
 *
 * 整合 RAII Guard 设计与字段断言策略。
 * 字段验证通过 internal.h 白盒访问，字段名依据内部结构体定义。
 *
 * 本接口为 Host 侧描述符初始化，无 NPU kernel，仅需
 * aclInit/SetDevice/CreateStream 环境 （复用 test/frame/sparse_test.h 的
 * AclEnvScope）。
 */

#ifndef TEST_SPARSELT_MATMULDESCRIPTORINIT_NPU_WRAPPER_H_
#define TEST_SPARSELT_MATMULDESCRIPTORINIT_NPU_WRAPPER_H_

#include "cann_ops_sparseLt.h"

#include "acl/acl.h"
#include "matmulDescriptorInit_param.h" // MatParam（VerifyMatFields/InitMatFromParam 依赖）

#include <gtest/gtest.h> // ADD_FAILURE / EXPECT_EQ（VerifyMatFields/VerifyMatmulFields 使用）

#include <cstdint>

// 白盒访问内部结构体字段
#include "aclsparselt_mat_descriptor_internal.h"
#include "aclsparselt_matmul_descriptor_internal.h"

namespace sparse_test
{

// ============================================================================
// RAII Guard
// 构造失败（Init 返回非 SUCCESS）时
// descr_=nullptr，析构为空操作，避免双重释放。
// ============================================================================

class SparseLtHandleGuard
{
  public:
    SparseLtHandleGuard() { status_ = aclsparseLtInit(&handle_); }
    ~SparseLtHandleGuard()
    {
        if (handle_ != nullptr)
        {
            aclsparseLtDestroy(&handle_);
        }
    }
    SparseLtHandleGuard(const SparseLtHandleGuard&) = delete;
    SparseLtHandleGuard& operator=(const SparseLtHandleGuard&) = delete;

    aclsparseLtHandle_t get() const { return handle_; }
    // 返回指向 handle 的指针，用于传给签名要求 const aclsparseLtHandle_t* 的 Init
    // 接口
    const aclsparseLtHandle_t* ptr() const { return &handle_; }
    aclsparseStatus_t status() const { return status_; }
    void release() { handle_ = nullptr; }

  private:
    aclsparseLtHandle_t handle_ = nullptr;
    aclsparseStatus_t status_ = ACL_SPARSE_STATUS_SUCCESS;
};

class MatDescrGuard
{
  public:
    MatDescrGuard() = default;
    ~MatDescrGuard()
    {
        if (descr_ != nullptr)
        {
            aclsparseLtMatDescriptorDestroy(&descr_);
        }
    }
    MatDescrGuard(const MatDescrGuard&) = delete;
    MatDescrGuard& operator=(const MatDescrGuard&) = delete;

    // Dense Init
    aclsparseStatus_t initDense(const aclsparseLtHandle_t* handle, int64_t rows, int64_t cols, int64_t ld,
                                uint32_t alignment, aclDataType valueType, aclsparseOrder_t order)
    {
        return aclsparseLtDenseDescriptorInit(handle, &descr_, rows, cols, ld, alignment, valueType, order);
    }

    // Structured Init
    aclsparseStatus_t initStructured(const aclsparseLtHandle_t* handle, int64_t rows, int64_t cols, int64_t ld,
                                     uint32_t alignment, aclDataType valueType, aclsparseOrder_t order,
                                     aclsparseLtSparsity_t sparsity)
    {
        return aclsparseLtStructuredDescriptorInit(handle, &descr_, rows, cols, ld, alignment, valueType, order,
                                                   sparsity);
    }

    aclsparseLtMatDescriptor_t get() const { return descr_; }
    aclsparseLtMatDescriptor_t* ptr() { return &descr_; }
    void release() { descr_ = nullptr; }

  private:
    aclsparseLtMatDescriptor_t descr_ = nullptr;
};

class MatmulDescrGuard
{
  public:
    MatmulDescrGuard() = default;
    ~MatmulDescrGuard()
    {
        if (descr_ != nullptr)
        {
            aclsparseLtMatmulDescriptorDestroy(&descr_);
        }
    }
    MatmulDescrGuard(const MatmulDescrGuard&) = delete;
    MatmulDescrGuard& operator=(const MatmulDescrGuard&) = delete;

    aclsparseStatus_t init(const aclsparseLtHandle_t* handle, aclsparseOperation_t opA, aclsparseOperation_t opB,
                           const aclsparseLtMatDescriptor_t* matA, const aclsparseLtMatDescriptor_t* matB,
                           const aclsparseLtMatDescriptor_t* matC, const aclsparseLtMatDescriptor_t* matD,
                           aclsparseComputeType_t computeType)
    {
        return aclsparseLtMatmulDescriptorInit(handle, &descr_, opA, opB, matA, matB, matC, matD, computeType);
    }

    aclsparseLtMatmulDescriptor_t get() const { return descr_; }
    aclsparseLtMatmulDescriptor_t* ptr() { return &descr_; }
    void release() { descr_ = nullptr; }

  private:
    aclsparseLtMatmulDescriptor_t descr_ = nullptr;
};

// ============================================================================
// 字段验证 helper：字段断言策略与字段定义
// 仅对 expect_status=SUCCESS 且 verify_fields=1 的用例调用。
// ============================================================================

// 验证 MatDescriptor 字段
// 返回 false 表示 internal 为空（已通过 ADD_FAILURE 记录），调用方应中止后续校验。
inline bool VerifyMatFields(aclsparseLtMatDescriptor_t descr, const MatParam& expected)
{
    auto* internal = ToInternal(descr);
    if (internal == nullptr) { ADD_FAILURE() << "VerifyMatFields: internal descriptor is null"; return false; }
    EXPECT_EQ(internal->rows, expected.rows);
    EXPECT_EQ(internal->cols, expected.cols);
    EXPECT_EQ(internal->ld, expected.ld);
    EXPECT_EQ(internal->alignment, expected.align);
    EXPECT_EQ(internal->valueType, expected.dtype);
    EXPECT_EQ(internal->order, expected.order);
    EXPECT_EQ(internal->isStructured, expected.isStructured);
    if (expected.isStructured)
    {
        EXPECT_EQ(internal->sparsity, expected.sparsity);
    }
    // batch 预留字段默认值
    EXPECT_EQ(internal->numBatches, 1);
    EXPECT_EQ(internal->batchStride, 0);
    return true;
}

// 验证 MatmulDescriptor 字段
// matA~matD 字段断言指针地址与前置构造的描述符一致
// 返回 false 表示 internal 为空（已通过 ADD_FAILURE 记录），调用方应中止后续校验。
inline bool VerifyMatmulFields(aclsparseLtMatmulDescriptor_t descr, aclsparseOperation_t opA, aclsparseOperation_t opB,
                                aclsparseLtMatDescriptor_t matA, aclsparseLtMatDescriptor_t matB,
                                aclsparseLtMatDescriptor_t matC, aclsparseLtMatDescriptor_t matD,
                                aclsparseComputeType_t computeType)
{
    auto* internal = ToInternal(descr);
    if (internal == nullptr) { ADD_FAILURE() << "VerifyMatmulFields: internal descriptor is null"; return false; }
    EXPECT_EQ(internal->opA, opA);
    EXPECT_EQ(internal->opB, opB);
    EXPECT_EQ(internal->matA, matA);
    EXPECT_EQ(internal->matB, matB);
    EXPECT_EQ(internal->matC, matC);
    EXPECT_EQ(internal->matD, matD);
    EXPECT_EQ(internal->computeType, computeType);
    return true;
}

// ============================================================================
// 构造辅助：依据 MatParam 调用对应 Init（dense 或 structured）
// 返回 Init 状态码；若 MatParam.valid=false 则不构造，返回 SUCCESS 且 descr
// 保持 nullptr。
// ============================================================================

inline aclsparseStatus_t InitMatFromParam(const aclsparseLtHandle_t* handle, MatDescrGuard& guard, const MatParam& p)
{
    if (!p.valid)
    {
        return ACL_SPARSE_STATUS_SUCCESS; // 该 mat 不参与构造
    }
    if (p.isStructured)
    {
        return guard.initStructured(handle, p.rows, p.cols, p.ld, p.align, p.dtype, p.order, p.sparsity);
    }
    return guard.initDense(handle, p.rows, p.cols, p.ld, p.align, p.dtype, p.order);
}

} // namespace sparse_test

#endif // TEST_SPARSELT_MATMULDESCRIPTORINIT_NPU_WRAPPER_H_
