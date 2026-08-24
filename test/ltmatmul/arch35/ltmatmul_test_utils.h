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
//   - Shared helpers: SetAlgAttributes, TransposeToRowMajor, PreparePhysicalA/B,
//     TransposeBPruned
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

// [TRANSPOSE] Updated chain ctx: computes physical matA/matB dimensions
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
inline std::vector<T> TransposeToRowMajor(const std::vector<T>& src,
    int32_t origRows, int32_t origCols, int64_t dstLd)
{
    std::vector<T> dst(static_cast<size_t>(origCols) * static_cast<size_t>(dstLd));
    for (int32_t j = 0; j < origCols; ++j) {
        for (int32_t i = 0; i < origRows; ++i) {
            dst[static_cast<size_t>(j) * dstLd + i] = src[static_cast<size_t>(i) * origCols + j];
        }
    }
    return dst;
}

template <typename T>
inline std::vector<T> PreparePhysicalA(const std::vector<T>& hA,
    int32_t m, int32_t k, bool transA, aclsparseOrder_t order, int64_t& aLd)
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
        return TransposeToRowMajor<T>(hA, m, k, aLd);
    }
    return hA;
}

template <typename T>
inline std::vector<T> PreparePhysicalB(const std::vector<T>& hB,
    int32_t k, int32_t n, bool transB)
{
    if (!transB) { return hB; }
    return TransposeToRowMajor<T>(hB, k, n, static_cast<int64_t>(k));
}

}  // namespace sparse_test

// =============================================================================
// Global namespace: NPU result structs + chain functions.
// =============================================================================

// -----------------------------------------------------------------------------
// AlgSetAttribute NPU result + chain functions (from alg_set_attribute).
// -----------------------------------------------------------------------------
struct AlgSetAttributeNpuResult {
    aclsparseStatus_t algSetCfgRet = ACL_SPARSE_STATUS_SUCCESS;
    aclsparseStatus_t algSetSplitRet = ACL_SPARSE_STATUS_SUCCESS;
    // getAttr roundtrip verification (setAttr -> getAttr readback).
    aclsparseStatus_t algGetCfgRet = ACL_SPARSE_STATUS_SUCCESS;
    aclsparseStatus_t algGetSplitRet = ACL_SPARSE_STATUS_SUCCESS;
    int32_t gotAlgConfigId = -1;
    int32_t gotSplitK = -1;
    aclsparseStatus_t pruneRet = ACL_SPARSE_STATUS_SUCCESS;
    aclsparseStatus_t matmulRet = ACL_SPARSE_STATUS_SUCCESS;
    size_t workspaceSize = 0;
    double npuMs = 0.0;
};

// -----------------------------------------------------------------------------
// AlgSetAttributeNpu: A-sparse chain used by alignment-32 end-to-end tests.
// Runs the full chain: setAttr → getAttr roundtrip → planInit → prune → matmul.
// (B-sparse variant removed — matmul path covers B-sparse via RunMatmulNpu.)
// -----------------------------------------------------------------------------
inline bool SetupAndRunAlgMatmul(
    aclsparseLtConstHandle_t handle,
    aclsparseLtMatmulDescriptor_t matmulDesc,
    aclsparseLtMatmulAlgSelection_t algSel,
    void* dA, void* dAPruned, void* dB, void* dC, void* dD,
    aclrtStream stream, float alpha, float beta,
    AlgSetAttributeNpuResult& result)
{
    using sparse_test::SparseLtPlanGuard;
    using sparse_test::EventGuard;
    using sparse_test::DeviceBuffer;

    SparseLtPlanGuard plan(handle, matmulDesc, algSel);

    auto wsRet = aclsparseLtMatmulGetWorkspaceSize(handle, plan.cptr(), &result.workspaceSize);
    if (wsRet != ACL_SPARSE_STATUS_SUCCESS) {
        std::cerr << "[NPU] MatmulGetWorkspaceSize failed: " << wsRet << std::endl;
        result.matmulRet = wsRet;
        return false;
    }
    DeviceBuffer dWorkspace = (result.workspaceSize > 0)
        ? DeviceBuffer::alloc(result.workspaceSize) : DeviceBuffer{};
    void* wsPtr = (result.workspaceSize > 0) ? dWorkspace.get() : nullptr;

    EventGuard evStart, evStop;
    evStart.record(stream);

    result.pruneRet = aclsparseLtSpMMAPrune(
        handle, const_cast<aclsparseLtConstMatmulDescriptor_t*>(&matmulDesc), dA, dAPruned,
        ACLSPARSELT_PRUNE_SPMMA_STRIP, stream);
    if (result.pruneRet != ACL_SPARSE_STATUS_SUCCESS) {
        std::cerr << "[NPU] SpMMAPrune failed: " << result.pruneRet << std::endl;
        return false;
    }

    result.matmulRet = aclsparseLtMatmul(
        handle, plan.cptr(),
        static_cast<const void*>(&alpha),
        dAPruned, dB,
        static_cast<const void*>(&beta), dC, dD,
        wsPtr, &stream, 1);

    evStop.record(stream);
    auto syncRet = aclrtSynchronizeStream(stream);
    if (result.matmulRet != ACL_SPARSE_STATUS_SUCCESS) {
        std::cerr << "[NPU] Matmul failed: " << result.matmulRet << std::endl;
        return false;
    }
    if (syncRet != ACL_SUCCESS) {
        std::cerr << "[NPU] aclrtSynchronizeStream failed: " << syncRet << std::endl;
        result.matmulRet = ACL_SPARSE_STATUS_EXECUTION_FAILED;
        return false;
    }

    result.npuMs = static_cast<double>(EventGuard::elapsedMs(evStart, evStop));
    return true;
}

// [TRANSPOSE] Updated AlgSetAttributeNpu with transA and transB support.
// transA=true, ROW: A data is transposed from (m,k) to (k,m) row-major.
// transA=true, COL: A data stays as (m,k) row-major (= (k,m) col-major with ld=k).
// transB=true: B data is transposed from (k,n) to (n,k) row-major.

template <typename T>
struct AlgNpuCommonCtx {
    static constexpr aclDataType kDtype = sparse_test::NpuDtypeTrait<T>::kAclDtype;
    static constexpr size_t kElt = sparse_test::NpuDtypeTrait<T>::kEltSize;
    static constexpr aclsparseComputeType_t kComputeType = ACL_SPARSE_COMPUTE_32F;
    bool transA;
    bool transB;
    size_t mk;
    size_t kn;
    size_t mn;
    int64_t aLd;
    int64_t physRowsA;
    int64_t physColsA;
    size_t aPhysSize;
    sparse_test::DeviceBuffer dA;
    sparse_test::DeviceBuffer dB;
    sparse_test::DeviceBuffer dC;
    sparse_test::DeviceBuffer dD;
};

template <typename T>
inline AlgNpuCommonCtx<T> InitAlgNpuCommon(
    int32_t m, int32_t k, int32_t n,
    const std::vector<T>& hA, const std::vector<T>& hB, const std::vector<T>& hC,
    aclsparseOrder_t order, aclsparseOperation_t opA, aclsparseOperation_t opB)
{
    using namespace sparse_test;
    AlgNpuCommonCtx<T> ctx;
    ctx.transA = (opA == ACL_SPARSE_OP_TRANSPOSE);
    ctx.transB = (opB == ACL_SPARSE_OP_TRANSPOSE);
    ctx.mk = static_cast<size_t>(m) * static_cast<size_t>(k);
    ctx.kn = static_cast<size_t>(k) * static_cast<size_t>(n);
    ctx.mn = static_cast<size_t>(m) * static_cast<size_t>(n);
    std::vector<T> hA_phys = PreparePhysicalA<T>(hA, m, k, ctx.transA, order, ctx.aLd);
    std::vector<T> hB_phys = PreparePhysicalB<T>(hB, k, n, ctx.transB);
    ctx.physRowsA = ctx.transA ? static_cast<int64_t>(k) : static_cast<int64_t>(m);
    ctx.physColsA = ctx.transA ? static_cast<int64_t>(m) : static_cast<int64_t>(k);
    ctx.aPhysSize = (order == ACL_SPARSE_ORDER_COL)
        ? static_cast<size_t>(ctx.physColsA) * static_cast<size_t>(ctx.aLd)
        : static_cast<size_t>(ctx.physRowsA) * static_cast<size_t>(ctx.aLd);
    ctx.dA = DeviceBuffer::copyFrom(hA_phys.data(), ctx.aPhysSize * ctx.kElt);
    ctx.dB = DeviceBuffer::copyFrom(hB_phys.data(),
        (ctx.transB ? static_cast<size_t>(n) * k : ctx.kn) * ctx.kElt);
    ctx.dC = DeviceBuffer::copyFrom(hC.data(), ctx.mn * ctx.kElt);
    ctx.dD = DeviceBuffer::alloc(ctx.mn * ctx.kElt);
    return ctx;
}

struct AlgNpuRunParams {
    aclrtStream stream;
    int32_t m;
    int32_t k;
    int32_t n;
    float alpha;
    float beta;
    int32_t alg_config_id;
    int32_t split_k;
    uint32_t alignment;
    aclsparseOrder_t order;
    aclsparseOperation_t opA;
    aclsparseOperation_t opB;
};

template <typename T>
inline AlgSetAttributeNpuResult RunAlgNpuASparseChain(
    AlgNpuCommonCtx<T>& ctx, const AlgNpuRunParams& p, std::vector<T>& hD)
{
    using namespace sparse_test;
    AlgSetAttributeNpuResult result{};
    DeviceBuffer dAPruned = DeviceBuffer::alloc(ctx.mk * ctx.kElt);

    SparseLtMatmulChainParams chainParams{
        static_cast<int64_t>(p.m), static_cast<int64_t>(p.k), static_cast<int64_t>(p.n),
        ctx.aLd, p.alignment, ctx.kDtype, ctx.kComputeType, p.order, p.opA, p.opB,
        dAPruned.get(), ctx.dB.get(), ctx.dC.get(), ctx.dD.get()
    };
    SparseLtMatmulChainCtx chain(chainParams);

    if (!SetAlgAttributes(chain.handle.get(), chain.algSel.get(),
                          p.alg_config_id, p.split_k,
                          result.algSetCfgRet, result.algSetSplitRet)) {
        return result;
    }

    // getAttr roundtrip verification: read back the values set by SetAlgAttributes.
    result.algGetCfgRet = aclsparseLtMatmulAlgGetAttribute(
        chain.handle.get(), chain.algSel.cptr(),
        ACLSPARSELT_MATMUL_ALG_CONFIG_ID, &result.gotAlgConfigId, sizeof(int32_t));
    result.algGetSplitRet = aclsparseLtMatmulAlgGetAttribute(
        chain.handle.get(), chain.algSel.cptr(),
        ACLSPARSELT_MATMUL_SPLIT_K, &result.gotSplitK, sizeof(int32_t));

    if (!SetupAndRunAlgMatmul(chain.handle.get(), chain.matmulDesc.get(),
                               chain.algSel.get(),
                               ctx.dA.get(), dAPruned.get(), ctx.dB.get(), ctx.dC.get(), ctx.dD.get(),
                               p.stream, p.alpha, p.beta, result)) {
        return result;
    }

    hD.resize(ctx.mn);
    ctx.dD.copyToHost(hD.data(), ctx.mn * ctx.kElt);
    return result;
}

// -----------------------------------------------------------------------------
// TransposeBPruned: transpose the pruned B matrix (shared by matmul B-sparse path).
// -----------------------------------------------------------------------------

template <typename T>
inline sparse_test::DeviceBuffer TransposeBPruned(
    aclrtStream stream, sparse_test::DeviceBuffer& dBPruned,
    int32_t k, int32_t n, size_t kn, size_t kElt)
{
    using namespace sparse_test;
    aclrtSynchronizeStream(stream);
    std::vector<T> hBPruned(kn);
    dBPruned.copyToHost(hBPruned.data(), kn * kElt);
    std::vector<T> hBPrunedT(kn);
    for (int32_t i = 0; i < k; ++i) {
        for (int32_t j = 0; j < n; ++j) {
            hBPrunedT[static_cast<size_t>(j) * k + i] =
                hBPruned[static_cast<size_t>(i) * n + j];
        }
    }
    return DeviceBuffer::copyFrom(hBPrunedT.data(), kn * kElt);
}

// -----------------------------------------------------------------------------
// AlgSetAttributeNpu: A-sparse chain used by alignment-32 end-to-end tests.
// Runs the full chain: setAttr → getAttr roundtrip → planInit → prune → matmul.
// (B-sparse variant removed — matmul path covers B-sparse via RunMatmulNpu.)
// -----------------------------------------------------------------------------
template <typename T>
inline AlgSetAttributeNpuResult AlgSetAttributeNpu(
    aclrtStream stream,
    const std::vector<T>& hA, const std::vector<T>& hB, const std::vector<T>& hC,
    std::vector<T>& hD,
    int32_t m, int32_t k, int32_t n,
    float alpha, float beta,
    int32_t alg_config_id, int32_t split_k,
    uint32_t alignment = 16,
    aclsparseOrder_t order = ACL_SPARSE_ORDER_ROW,
    aclsparseOperation_t opA = ACL_SPARSE_OP_NON_TRANSPOSE,
    aclsparseOperation_t opB = ACL_SPARSE_OP_NON_TRANSPOSE)
{
    AlgNpuRunParams p{stream, m, k, n, alpha, beta, alg_config_id, split_k,
                      alignment, order, opA, opB};
    auto ctx = InitAlgNpuCommon<T>(m, k, n, hA, hB, hC, order, opA, opB);
    return RunAlgNpuASparseChain<T>(ctx, p, hD);
}

#endif  // TEST_LTMATMUL_TEST_UTILS_H_
