/**
 * ----------------------------------------------------------------------------------------------------------
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ----------------------------------------------------------------------------------------------------------
 */

#ifndef TEST_LTMATMUL_TEST_UTILS_H_
#define TEST_LTMATMUL_TEST_UTILS_H_

// =============================================================================
// Foundational test utilities for the ltmatmul test binary.
//
// Provides:
//   - NpuDtypeTrait / MatmulComputeTrait
//   - RAII guards for aclsparseLt descriptors / selections / plans / events
//   - SparseLtMatmulChainParams / SparseLtMatmulChainCtx
//   - Shared helpers: SetAlgAttributes, PreparePhysicalA/B,
//     TransposeBPruned (TransposeToRowMajor is in shared transpose_utils.h)
//   - AlgSetAttributeNpu: A-sparse chain (setAttr -> getAttr -> plan -> prune -> matmul)
//     used by alignment-32 end-to-end exception tests
//
// Consumed by ltmatmul_npu_wrapper.h (which holds the RunMatmulNpu chain).
// =============================================================================

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

#include "acl/acl.h"
#include "cann_ops_sparseLt.h"
#include "descriptor_manager.h"
#include "aclsparselt_internal.h"  // fill_tiling_dims, compute_ws_layout, ToInternal
#include "transpose_utils.h"  // shared TransposeToRowMajor

#include "../ltmatmul_golden.h"    // for bf16_bits_t, golden types

namespace sparse_test {

// -----------------------------------------------------------------------------
// NpuDtypeTrait: maps C++ storage types to ACL dtype + element size.
// -----------------------------------------------------------------------------
template <typename T>
struct NpuDtypeTrait;

template <>
struct NpuDtypeTrait<float> {
    static constexpr aclDataType kAclDtype = ACL_FLOAT;
    static constexpr size_t kEltSize = sizeof(float);
};

template <>
struct NpuDtypeTrait<uint16_t> {
    static constexpr aclDataType kAclDtype = ACL_FLOAT16;
    static constexpr size_t kEltSize = sizeof(uint16_t);
};

template <>
struct NpuDtypeTrait<bf16_bits_t> {
    static constexpr aclDataType kAclDtype = ACL_BF16;
    static constexpr size_t kEltSize = sizeof(uint16_t);
};

template <>
struct NpuDtypeTrait<int8_t> {
    static constexpr aclDataType kAclDtype = ACL_INT8;
    static constexpr size_t kEltSize = sizeof(int8_t);
};

template <>
struct NpuDtypeTrait<int32_t> {
    static constexpr aclDataType kAclDtype = ACL_INT32;
    static constexpr size_t kEltSize = sizeof(int32_t);
};

// Compute type trait: FP32/FP16/BF16 -> 32F, INT8 -> 32I.
template <typename T>
struct MatmulComputeTrait {
    static constexpr aclsparseComputeType_t kComputeType = ACL_SPARSE_COMPUTE_32F;
};
template <>
struct MatmulComputeTrait<int8_t> {
    static constexpr aclsparseComputeType_t kComputeType = ACL_SPARSE_COMPUTE_32I;
};

// -----------------------------------------------------------------------------
// BiasDtypeTrait: bias dtype follows C dtype, except INT8 uses FP32
// (cuSPARSELt规定 INT8 路径 bias = FP32).
//   FP32(float)        → bias = FP32
//   FP16(uint16_t)     → bias = FP16
//   BF16(bf16_bits_t)  → bias = BF16
//   INT8(int8_t)       → bias = FP32 (exception per cuSPARSELt)
// -----------------------------------------------------------------------------
template <typename T>
struct BiasDtypeTrait {
    static constexpr aclDataType kBiasAclDtype = NpuDtypeTrait<T>::kAclDtype;
    static constexpr size_t kBiasEltSize = NpuDtypeTrait<T>::kEltSize;
};
template <>
struct BiasDtypeTrait<int8_t> {
    static constexpr aclDataType kBiasAclDtype = ACL_FLOAT;
    static constexpr size_t kBiasEltSize = sizeof(float);
};

// -----------------------------------------------------------------------------
// RAII Guards for aclsparseLt descriptors / selections / plans / events.
// -----------------------------------------------------------------------------

class SparseLtHandleGuard {
public:
    SparseLtHandleGuard() {
        auto s = aclsparseLtInit(&handle_);
        if (s != ACL_SPARSE_STATUS_SUCCESS) {
            throw std::runtime_error("aclsparseLtInit failed");
        }
    }
    ~SparseLtHandleGuard() { if (handle_) aclsparseLtDestroy(&handle_); }
    SparseLtHandleGuard(const SparseLtHandleGuard&) = delete;
    SparseLtHandleGuard& operator=(const SparseLtHandleGuard&) = delete;
    aclsparseLtConstHandle_t get() const { return &handle_; }
private:
    aclsparseLtHandle_t handle_ = nullptr;
};

class SparseLtMatDescGuard {
public:
    SparseLtMatDescGuard(aclsparseLtConstHandle_t handle,
                         int64_t rows, int64_t cols, int64_t ld,
                         uint32_t alignment, aclDataType valueType,
                         aclsparseOrder_t order,
                         aclsparseLtSparsity_t sparsity) {
        auto s = aclsparseLtStructuredDescriptorInit(handle, &descr_, rows, cols, ld,
                                                     alignment, valueType, order, sparsity);
        if (s != ACL_SPARSE_STATUS_SUCCESS) {
            throw std::runtime_error("aclsparseLtStructuredDescriptorInit failed");
        }
    }
    ~SparseLtMatDescGuard() { aclsparseLtMatDescriptorDestroy(&descr_); }
    SparseLtMatDescGuard(const SparseLtMatDescGuard&) = delete;
    SparseLtMatDescGuard& operator=(const SparseLtMatDescGuard&) = delete;
    aclsparseLtMatDescriptor_t& get() { return descr_; }
    aclsparseLtMatDescriptor_t get() const { return descr_; }
    aclsparseLtConstMatDescriptor_t* cptr() { return const_cast<aclsparseLtConstMatDescriptor_t*>(&descr_); }
private:
    aclsparseLtMatDescriptor_t descr_ = nullptr;
};

class SparseLtDnMatDescGuard {
public:
    SparseLtDnMatDescGuard(aclsparseLtConstHandle_t handle,
                           int64_t rows, int64_t cols, int64_t ld,
                           uint32_t alignment, aclDataType valueType,
                           aclsparseOrder_t order) {
        auto s = aclsparseLtDenseDescriptorInit(handle, &descr_, rows, cols, ld,
                                                alignment, valueType, order);
        if (s != ACL_SPARSE_STATUS_SUCCESS) {
            throw std::runtime_error("aclsparseLtDenseDescriptorInit failed");
        }
    }
    ~SparseLtDnMatDescGuard() { aclsparseLtMatDescriptorDestroy(&descr_); }
    SparseLtDnMatDescGuard(const SparseLtDnMatDescGuard&) = delete;
    SparseLtDnMatDescGuard& operator=(const SparseLtDnMatDescGuard&) = delete;
    aclsparseLtMatDescriptor_t& get() { return descr_; }
    aclsparseLtMatDescriptor_t get() const { return descr_; }
    aclsparseLtConstMatDescriptor_t* cptr() { return const_cast<aclsparseLtConstMatDescriptor_t*>(&descr_); }
private:
    aclsparseLtMatDescriptor_t descr_ = nullptr;
};

class SparseLtMatmulDescGuard {
public:
    SparseLtMatmulDescGuard(aclsparseLtConstHandle_t handle,
                            aclsparseOperation_t opA, aclsparseOperation_t opB,
                            aclsparseLtMatDescriptor_t A,
                            aclsparseLtMatDescriptor_t B,
                            aclsparseLtMatDescriptor_t C,
                            aclsparseLtMatDescriptor_t D,
                            aclsparseComputeType_t computeType) {
        auto s = aclsparseLtMatmulDescriptorInit(handle, &descr_, opA, opB, &A, &B, &C, &D, computeType);
        if (s != ACL_SPARSE_STATUS_SUCCESS) {
            throw std::runtime_error("aclsparseLtMatmulDescriptorInit failed");
        }
    }
    ~SparseLtMatmulDescGuard() { aclsparseLtMatmulDescriptorDestroy(&descr_); }
    SparseLtMatmulDescGuard(const SparseLtMatmulDescGuard&) = delete;
    SparseLtMatmulDescGuard& operator=(const SparseLtMatmulDescGuard&) = delete;
    aclsparseLtMatmulDescriptor_t& get() { return descr_; }
    aclsparseLtMatmulDescriptor_t get() const { return descr_; }
    aclsparseLtMatmulDescriptor_t* ptr() { return &descr_; }
    aclsparseLtConstMatmulDescriptor_t* cptr() { return const_cast<aclsparseLtConstMatmulDescriptor_t*>(&descr_); }
private:
    aclsparseLtMatmulDescriptor_t descr_ = nullptr;
};

class SparseLtAlgSelectionGuard {
public:
    SparseLtAlgSelectionGuard(aclsparseLtConstHandle_t handle,
                              aclsparseLtMatmulDescriptor_t matmulDesc,
                              aclsparseLtMatmulAlg_t alg = ACL_SPARSE_LT_MATMUL_ALG_DEFAULT) {
        auto s = aclsparseLtMatmulAlgSelectionInit(handle, &sel_, &matmulDesc, alg);
        if (s != ACL_SPARSE_STATUS_SUCCESS) {
            throw std::runtime_error("aclsparseLtMatmulAlgSelectionInit failed");
        }
    }
    ~SparseLtAlgSelectionGuard() { aclsparseLtMatmulAlgSelectionDestroy(&sel_); }
    SparseLtAlgSelectionGuard(const SparseLtAlgSelectionGuard&) = delete;
    SparseLtAlgSelectionGuard& operator=(const SparseLtAlgSelectionGuard&) = delete;
    aclsparseLtMatmulAlgSelection_t& get() { return sel_; }
    aclsparseLtMatmulAlgSelection_t get() const { return sel_; }
    aclsparseLtMatmulAlgSelection_t* ptr() { return &sel_; }
    aclsparseLtConstMatmulAlgSelection_t* cptr() { return const_cast<aclsparseLtConstMatmulAlgSelection_t*>(&sel_); }
private:
    aclsparseLtMatmulAlgSelection_t sel_ = nullptr;
};

class SparseLtPlanGuard {
public:
    SparseLtPlanGuard(aclsparseLtConstHandle_t handle,
                      aclsparseLtMatmulDescriptor_t matmulDesc,
                      aclsparseLtMatmulAlgSelection_t algSel) {
        auto s = aclsparseLtMatmulPlanInit(handle, &plan_, &matmulDesc, &algSel);
        if (s != ACL_SPARSE_STATUS_SUCCESS) {
            throw std::runtime_error("aclsparseLtMatmulPlanInit failed");
        }
    }
    ~SparseLtPlanGuard() { aclsparseLtMatmulPlanDestroy(&plan_); }
    SparseLtPlanGuard(const SparseLtPlanGuard&) = delete;
    SparseLtPlanGuard& operator=(const SparseLtPlanGuard&) = delete;
    aclsparseLtMatmulPlan_t& get() { return plan_; }
    aclsparseLtMatmulPlan_t get() const { return plan_; }
    aclsparseLtConstMatmulPlan_t* cptr() { return const_cast<aclsparseLtConstMatmulPlan_t*>(&plan_); }
private:
    aclsparseLtMatmulPlan_t plan_ = nullptr;
};

class EventGuard {
public:
    EventGuard() {
        if (aclrtCreateEvent(&ev_) != ACL_SUCCESS) {
            throw std::runtime_error("aclrtCreateEvent failed");
        }
    }
    ~EventGuard() { if (ev_) aclrtDestroyEvent(ev_); }
    EventGuard(const EventGuard&) = delete;
    EventGuard& operator=(const EventGuard&) = delete;
    void record(aclrtStream stream) { aclrtRecordEvent(ev_, stream); }
    static float elapsedMs(const EventGuard& start, const EventGuard& stop) {
        float ms = 0.0f;
        aclrtEventElapsedTime(&ms, start.ev_, stop.ev_);
        return ms;
    }
private:
    aclrtEvent ev_ = nullptr;
};

// -----------------------------------------------------------------------------
// SparseLtMatmulChainParams / SparseLtMatmulChainCtx (from alg_set_attribute).
// Updated chain params with opB support.
// -----------------------------------------------------------------------------
struct SparseLtMatmulChainParams {
    int64_t m;           // logical m
    int64_t k;           // logical k
    int64_t n;           // logical n
    int64_t aLd;         // matA leading dimension (physical)
    uint32_t alignment;
    aclDataType dtype;
    aclsparseComputeType_t computeType;
    aclsparseOrder_t order;
    aclsparseOperation_t opA;
    aclsparseOperation_t opB;
    void* dAPruned;
    void* dB;
    void* dC;
    void* dD;
};

// Updated chain ctx: computes physical matA/matB dimensions
// from logical m/k/n and opA/opB.
struct SparseLtMatmulChainCtx {
    SparseLtHandleGuard handle;
    SparseLtMatDescGuard matA;
    SparseLtDnMatDescGuard matB;
    SparseLtDnMatDescGuard matC;
    SparseLtDnMatDescGuard matD;
    SparseLtMatmulDescGuard matmulDesc;
    SparseLtAlgSelectionGuard algSel;

    explicit SparseLtMatmulChainCtx(const SparseLtMatmulChainParams& p)
        : handle(),
          matA(handle.get(),
               (p.opA == ACL_SPARSE_OP_TRANSPOSE) ? p.k : p.m,
               (p.opA == ACL_SPARSE_OP_TRANSPOSE) ? p.m : p.k,
               p.aLd, p.alignment, p.dtype,
               p.order, ACL_SPARSE_LT_SPARSITY_50_PERCENT),
          matB(handle.get(),
               (p.opB == ACL_SPARSE_OP_TRANSPOSE) ? p.n : p.k,
               (p.opB == ACL_SPARSE_OP_TRANSPOSE) ? p.k : p.n,
               (p.opB == ACL_SPARSE_OP_TRANSPOSE) ? p.k : p.n,
               p.alignment, p.dtype, ACL_SPARSE_ORDER_ROW),
          matC(handle.get(), p.m, p.n, p.n, p.alignment, p.dtype,
               ACL_SPARSE_ORDER_ROW),
          matD(handle.get(), p.m, p.n, p.n, p.alignment, p.dtype,
               ACL_SPARSE_ORDER_ROW),
          matmulDesc(handle.get(), p.opA, p.opB,
                     matA.get(), matB.get(), matC.get(), matD.get(), p.computeType),
          algSel(handle.get(), matmulDesc.get(), ACL_SPARSE_LT_MATMUL_ALG_DEFAULT) {}
};

inline bool SetAlgAttributes(aclsparseLtConstHandle_t handle,
                              aclsparseLtMatmulAlgSelection_t algSel,
                              int32_t algConfigId, int32_t splitK,
                              aclsparseStatus_t& cfgRet, aclsparseStatus_t& splitRet)
{
    cfgRet = aclsparseLtMatmulAlgSetAttribute(
        handle, &algSel, ACLSPARSELT_MATMUL_ALG_CONFIG_ID,
        &algConfigId, sizeof(int32_t));
    if (cfgRet != ACL_SPARSE_STATUS_SUCCESS) {
        std::cerr << "[NPU] AlgSetAttribute(ALG_CONFIG_ID) failed: " << cfgRet << std::endl;
        return false;
    }
    splitRet = aclsparseLtMatmulAlgSetAttribute(
        handle, &algSel, ACLSPARSELT_MATMUL_SPLIT_K,
        &splitK, sizeof(int32_t));
    if (splitRet != ACL_SPARSE_STATUS_SUCCESS) {
        std::cerr << "[NPU] AlgSetAttribute(SPLIT_K) failed: " << splitRet << std::endl;
        return false;
    }
    return true;
}

template <typename T>
inline std::vector<T> PreparePhysicalA(const std::vector<T>& hA,
    int32_t m, int32_t k, bool transA, aclsparseOrder_t order, int64_t& aLd,
    int32_t num_batches = 1, int64_t batch_stride = 0)
{
    if (!transA) {
        // For NON_TRANSPOSE, the prune kernel always reads the
        // matrix as row-major (m, k) with stride ld, regardless of order.
        // The order field only selects the pruning direction (alongRow vs
        // alongCol), not the data layout. Setting aLd=k (the actual row-major
        // stride) ensures correct data access when m != k. Previously aLd=m
        // was used for COL order, which only works for square matrices (m==k).
        aLd = static_cast<int64_t>(k);
        return hA;
    }
    aLd = (order == ACL_SPARSE_ORDER_COL)
        ? static_cast<int64_t>(k) : static_cast<int64_t>(m);
    if (order == ACL_SPARSE_ORDER_ROW) {
        if (num_batches <= 1) {
            return TransposeToRowMajor<T>(hA, m, k, aLd);
        }
        // Batch: transpose each batch's (m,k) block separately, preserving
        // the batch-stride layout (padding between batches stays as zeros).
        const size_t matSize = static_cast<size_t>(m) * k;
        const size_t totalElems = static_cast<size_t>(num_batches) * static_cast<size_t>(batch_stride);
        std::vector<T> result(totalElems, T{0});
        for (int32_t b = 0; b < num_batches; ++b) {
            size_t off = static_cast<size_t>(b) * static_cast<size_t>(batch_stride);
            std::vector<T> batchData(hA.begin() + off, hA.begin() + off + matSize);
            auto transposed = TransposeToRowMajor<T>(batchData, m, k, aLd);
            std::copy(transposed.begin(), transposed.end(), result.begin() + off);
        }
        return result;
    }
    return hA;
}

template <typename T>
inline std::vector<T> PreparePhysicalB(const std::vector<T>& hB,
    int32_t k, int32_t n, bool transB,
    int32_t num_batches = 1, int64_t batch_stride = 0)
{
    if (!transB) { return hB; }
    if (num_batches <= 1) {
        return TransposeToRowMajor<T>(hB, k, n, static_cast<int64_t>(k));
    }
    // Batch: transpose each batch's (k,n) block separately.
    const size_t matSize = static_cast<size_t>(k) * n;
    const size_t totalElems = static_cast<size_t>(num_batches) * static_cast<size_t>(batch_stride);
    std::vector<T> result(totalElems, T{0});
    for (int32_t b = 0; b < num_batches; ++b) {
        size_t off = static_cast<size_t>(b) * static_cast<size_t>(batch_stride);
        std::vector<T> batchData(hB.begin() + off, hB.begin() + off + matSize);
        auto transposed = TransposeToRowMajor<T>(batchData, k, n, static_cast<int64_t>(k));
        std::copy(transposed.begin(), transposed.end(), result.begin() + off);
    }
    return result;
}

}  // namespace sparse_test


// AlgSetAttributeNpu chain moved to ltmatmul_alg_npu.h
// to reduce header file size. Include it here for backward compat.
#include "ltmatmul_alg_npu.h"

#endif  // TEST_LTMATMUL_TEST_UTILS_H_
