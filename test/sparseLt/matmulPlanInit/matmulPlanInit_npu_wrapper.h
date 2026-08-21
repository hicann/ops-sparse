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
 * @file matmulPlanInit_npu_wrapper.h
 * @brief Host 侧调用封装：RAII Guard（Handle/MatDescr/MatmulDescr/AlgSelection/Plan）+ 字段验证。
 *
 * 隔离设计：本文件完全自包含，所有 RAII Guard 在此独立定义。
 */

#ifndef TEST_SPARSELT_MATMULPLANINIT_NPU_WRAPPER_H_
#define TEST_SPARSELT_MATMULPLANINIT_NPU_WRAPPER_H_

#include "cann_ops_sparseLt.h"

#include "acl/acl.h"
#include "matmulPlanInit_param.h"

#include <gtest/gtest.h>

#include <cstdint>

// 白盒访问内部结构体字段
#include "aclsparselt_mat_descriptor_internal.h"
#include "aclsparselt_matmul_descriptor_internal.h"
#include "aclsparselt_matmul_alg_selection_internal.h"
#include "aclsparselt_matmul_plan_internal.h"

namespace sparse_test
{

// ============================================================================
// RAII Guard
// ============================================================================

class PlanHandleGuard
{
  public:
    PlanHandleGuard() { status_ = aclsparseLtInit(&handle_); }
    ~PlanHandleGuard()
    {
        if (handle_ != nullptr)
        {
            aclsparseLtDestroy(&handle_);
        }
    }
    PlanHandleGuard(const PlanHandleGuard&) = delete;
    PlanHandleGuard& operator=(const PlanHandleGuard&) = delete;

    aclsparseLtHandle_t get() const { return handle_; }
    const aclsparseLtHandle_t* ptr() const { return &handle_; }
    aclsparseStatus_t status() const { return status_; }
    void release() { handle_ = nullptr; }

  private:
    aclsparseLtHandle_t handle_ = nullptr;
    aclsparseStatus_t status_ = ACL_SPARSE_STATUS_SUCCESS;
};

class PlanMatDescrGuard
{
  public:
    PlanMatDescrGuard() = default;
    ~PlanMatDescrGuard()
    {
        if (descr_ != nullptr)
        {
            aclsparseLtMatDescriptorDestroy(&descr_);
        }
    }
    PlanMatDescrGuard(const PlanMatDescrGuard&) = delete;
    PlanMatDescrGuard& operator=(const PlanMatDescrGuard&) = delete;

    aclsparseStatus_t initDense(const aclsparseLtHandle_t* handle, int64_t rows, int64_t cols, int64_t ld,
                                uint32_t alignment, aclDataType valueType, aclsparseOrder_t order)
    {
        return aclsparseLtDenseDescriptorInit(handle, &descr_, rows, cols, ld, alignment, valueType, order);
    }

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

class PlanMatmulDescrGuard
{
  public:
    PlanMatmulDescrGuard() = default;
    ~PlanMatmulDescrGuard()
    {
        if (descr_ != nullptr)
        {
            aclsparseLtMatmulDescriptorDestroy(&descr_);
        }
    }
    PlanMatmulDescrGuard(const PlanMatmulDescrGuard&) = delete;
    PlanMatmulDescrGuard& operator=(const PlanMatmulDescrGuard&) = delete;

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

class AlgSelectionGuard
{
  public:
    AlgSelectionGuard() = default;
    ~AlgSelectionGuard()
    {
        if (sel_ != nullptr)
        {
            aclsparseLtMatmulAlgSelectionDestroy(&sel_);
        }
    }
    AlgSelectionGuard(const AlgSelectionGuard&) = delete;
    AlgSelectionGuard& operator=(const AlgSelectionGuard&) = delete;

    aclsparseStatus_t init(const aclsparseLtHandle_t* handle,
                           const aclsparseLtMatmulDescriptor_t* matmulDescr,
                           aclsparseLtMatmulAlg_t alg)
    {
        return aclsparseLtMatmulAlgSelectionInit(handle, &sel_, matmulDescr, alg);
    }

    aclsparseLtMatmulAlgSelection_t get() const { return sel_; }
    aclsparseLtMatmulAlgSelection_t* ptr() { return &sel_; }
    void release() { sel_ = nullptr; }

  private:
    aclsparseLtMatmulAlgSelection_t sel_ = nullptr;
};

class PlanGuard
{
  public:
    PlanGuard() = default;
    ~PlanGuard()
    {
        if (plan_ != nullptr)
        {
            aclsparseLtMatmulPlanDestroy(&plan_);
        }
    }
    PlanGuard(const PlanGuard&) = delete;
    PlanGuard& operator=(const PlanGuard&) = delete;

    aclsparseStatus_t init(const aclsparseLtHandle_t* handle,
                           const aclsparseLtMatmulDescriptor_t* matmulDescr,
                           const aclsparseLtMatmulAlgSelection_t* algSelection)
    {
        return aclsparseLtMatmulPlanInit(handle, &plan_, matmulDescr, algSelection);
    }

    aclsparseLtMatmulPlan_t get() const { return plan_; }
    aclsparseLtMatmulPlan_t* ptr() { return &plan_; }
    void release() { plan_ = nullptr; }

  private:
    aclsparseLtMatmulPlan_t plan_ = nullptr;
};

// ============================================================================
// 字段验证 helper（白盒访问 internal.h）
// ============================================================================

inline bool VerifyAlgSelectionFields(aclsparseLtMatmulAlgSelection_t sel,
                                     aclsparseLtMatmulDescriptor_t matmulDescr,
                                     aclsparseLtMatmulAlg_t alg)
{
    auto* internal = ToInternal(sel);
    if (internal == nullptr) { ADD_FAILURE() << "VerifyAlgSelectionFields: internal is null"; return false; }
    EXPECT_EQ(internal->matmulDescr, matmulDescr);
    EXPECT_EQ(internal->alg, alg);
    return !::testing::Test::HasFailure();
}

inline bool VerifyPlanFields(aclsparseLtMatmulPlan_t plan,
                             aclsparseLtMatmulDescriptor_t matmulDescr,
                             aclsparseLtMatmulAlgSelection_t algSelection)
{
    auto* internal = ToInternal(plan);
    if (internal == nullptr) { ADD_FAILURE() << "VerifyPlanFields: internal is null"; return false; }
    EXPECT_EQ(internal->matmulDescr, matmulDescr);
    EXPECT_EQ(internal->algSelection, algSelection);
    return !::testing::Test::HasFailure();
}

// ============================================================================
// 构造辅助：依据 PlanMatParam 调用对应 Init
// ============================================================================

inline aclsparseStatus_t InitMatFromPlanParam(const aclsparseLtHandle_t* handle, PlanMatDescrGuard& guard,
                                              const PlanMatParam& p)
{
    if (!p.valid)
    {
        return ACL_SPARSE_STATUS_SUCCESS;
    }
    if (p.isStructured)
    {
        return guard.initStructured(handle, p.rows, p.cols, p.ld, p.align, p.dtype, p.order, p.sparsity);
    }
    return guard.initDense(handle, p.rows, p.cols, p.ld, p.align, p.dtype, p.order);
}

} // namespace sparse_test

#endif // TEST_SPARSELT_MATMULPLANINIT_NPU_WRAPPER_H_
