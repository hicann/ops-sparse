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

#ifndef TEST_PRUNE_NPU_WRAPPER_H_
#define TEST_PRUNE_NPU_WRAPPER_H_

// =============================================================================
// Prune NPU wrapper (decoupled from alg_set_attribute / matmul test headers).
//
// Reuses RAII guards from matmulDescriptorInit_npu_wrapper.h (SparseLtHandleGuard,
// MatDescrGuard, MatmulDescrGuard). Provides NpuDtypeTrait, EventGuard, and
// PreparePhysicalA helper (extracted from the source repo's
// alg_set_attribute_npu_wrapper.h) so that prune tests do not pull in any
// alg_set_attribute or matmul code.
//
// Removed from the original:
//   - SparseLtAlgSelectionGuard / SparseLtPlanGuard (prune does not need
//     algSelection or plan)
//   - SetAlgAttributes (prune does not call AlgSetAttribute)
//   - algSetCfgRet / algSetSplitRet fields in PruneNpuResult
// =============================================================================

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include "acl/acl.h"
#include "cann_ops_sparseLt.h"
#include "descriptor_manager.h"
#include "matmulDescriptorInit_npu_wrapper.h"
#include "prune_test_util.h"

namespace sparse_test {

// -----------------------------------------------------------------------------
// const-pointer accessors for the RAII guards reused from
// matmulDescriptorInit_npu_wrapper.h.
//
// The guard's ptr() returns a pointer to the underlying descr_ member
// (aclsparseLtMatDescriptor_t* = struct aclsparseLtMatDescriptor**). Several
// aclsparseLt APIs (e.g. aclsparseLtSpMMAPrune) require the const variant
// (aclsparseLtConstMatmulDescriptor_t* = const struct**). const_cast adds the
// missing const-qualification on the pointed-to struct, which is a valid,
// well-defined conversion.
// -----------------------------------------------------------------------------
inline aclsparseLtConstMatDescriptor_t* matDescCptr(MatDescrGuard& g) {
    return const_cast<aclsparseLtConstMatDescriptor_t*>(g.ptr());
}
inline aclsparseLtConstMatmulDescriptor_t* matmulDescCptr(MatmulDescrGuard& g) {
    return const_cast<aclsparseLtConstMatmulDescriptor_t*>(g.ptr());
}

// -----------------------------------------------------------------------------
// NpuDtypeTrait: map storage type T to aclDataType + element size.
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
    static constexpr size_t kEltSize = sizeof(bf16_bits_t);
};

template <>
struct NpuDtypeTrait<int8_t> {
    static constexpr aclDataType kAclDtype = ACL_INT8;
    static constexpr size_t kEltSize = sizeof(int8_t);
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
// Physical matrix preparation (transpose logic for TRANSPOSE+ROW order).
// -----------------------------------------------------------------------------

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
        // stride) ensures correct data access when m != k.
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

}  // namespace sparse_test

// -----------------------------------------------------------------------------
// Prune NPU execution result (no algSet fields — prune is independent of
// algorithm selection).
// -----------------------------------------------------------------------------
struct PruneNpuResult {
    aclsparseStatus_t pruneRet = ACL_SPARSE_STATUS_SUCCESS;
    size_t workspaceSize = 0;
    double npuMs = 0.0;
};

// Common setup for PruneNpu / PruneNpuBSparse: dtype traits + result holder +
// suppression of unused CSV-signature params (alg_config_id, split_k).
template <typename T>
struct PruneNpuSetup {
    static constexpr aclDataType kDtype = sparse_test::NpuDtypeTrait<T>::kAclDtype;
    static constexpr size_t kElt = sparse_test::NpuDtypeTrait<T>::kEltSize;
    static constexpr aclsparseComputeType_t kComputeType =
        std::is_same_v<T, int8_t> ? ACL_SPARSE_COMPUTE_32I : ACL_SPARSE_COMPUTE_32F;
    PruneNpuResult result;
    explicit PruneNpuSetup(int32_t alg_config_id, int32_t split_k) : result{} {
        (void)alg_config_id;
        (void)split_k;
    }
};

inline bool RunPruneExecution(
    aclsparseLtConstHandle_t handle,
    aclsparseLtConstMatmulDescriptor_t matmulDesc,
    void* dA, void* dAPruned,
    aclsparseLtPruneAlg_t pruneAlg,
    aclrtStream stream,
    PruneNpuResult& result)
{
    using sparse_test::EventGuard;

    result.workspaceSize = 0;

    EventGuard evStart, evStop;
    evStart.record(stream);

    result.pruneRet = aclsparseLtSpMMAPrune(
        handle, &matmulDesc, dA, dAPruned,
        pruneAlg, stream);

    evStop.record(stream);
    auto syncRet = aclrtSynchronizeStream(stream);
    if (result.pruneRet != ACL_SPARSE_STATUS_SUCCESS) {
        std::cerr << "[NPU] SpMMAPrune failed: " << result.pruneRet << std::endl;
        return false;
    }
    if (syncRet != ACL_SUCCESS) {
        std::cerr << "[NPU] aclrtSynchronizeStream failed: " << syncRet << std::endl;
        result.pruneRet = ACL_SPARSE_STATUS_EXECUTION_FAILED;
        return false;
    }

    result.npuMs = static_cast<double>(EventGuard::elapsedMs(evStart, evStop));
    return true;
}

// [TRANSPOSE] PruneNpu with transA support.
// When transA=true:
//   ROW order: physical A is (k, m) row-major. We transpose hA from (m,k) to (k,m).
//   COL order: physical A is (k, m) col-major = (m,k) row-major with stride ld=k. No transpose.
// matA descriptor is created with rows=k, cols=m (physical dimensions).
// pruneAlg: ACLSPARSELT_PRUNE_SPMMA_STRIP or ACLSPARSELT_PRUNE_SPMMA_TILE.
// alg_config_id / split_k are accepted for CSV signature compatibility but
// are NOT used by prune (prune is independent of algorithm selection).
template <typename T>
inline PruneNpuResult PruneNpu(
    aclrtStream stream,
    const std::vector<T>& hA, std::vector<T>& hAPruned,
    int32_t m, int32_t k,
    int32_t alg_config_id, int32_t split_k,
    aclsparseOperation_t opA, aclsparseOrder_t order,
    aclsparseLtPruneAlg_t pruneAlg = ACLSPARSELT_PRUNE_SPMMA_STRIP)
{
    using namespace sparse_test;
    PruneNpuSetup<T> ctx(alg_config_id, split_k);
    auto& result = ctx.result;

    constexpr int32_t kPruneN = 128;
    const bool transA = (opA == ACL_SPARSE_OP_TRANSPOSE);

    int64_t aLd;
    std::vector<T> hA_phys = PreparePhysicalA<T>(hA, m, k, transA, order, aLd);

    const int64_t physRowsA = transA ? static_cast<int64_t>(k) : static_cast<int64_t>(m);
    const int64_t physColsA = transA ? static_cast<int64_t>(m) : static_cast<int64_t>(k);
    const size_t mk = static_cast<size_t>(m) * static_cast<size_t>(k);
    // For NON_TRANSPOSE, data is always row-major (m, k) with stride
    // aLd=k regardless of order (see PreparePhysicalA). Use the row-major size
    // formula (physRowsA * aLd) for NON_TRANSPOSE. The COL formula
    // (physColsA * aLd) is only valid for TRANSPOSE+COL where aLd=k and the
    // physical layout is column-major (k, m) with ld=k.
    const size_t aPhysSize = (transA && order == ACL_SPARSE_ORDER_COL)
        ? static_cast<size_t>(physColsA) * static_cast<size_t>(aLd)
        : static_cast<size_t>(physRowsA) * static_cast<size_t>(aLd);

    DeviceBuffer dA = DeviceBuffer::copyFrom(hA_phys.data(), aPhysSize * ctx.kElt);
    DeviceBuffer dAPruned = DeviceBuffer::alloc(mk * ctx.kElt);

    SparseLtHandleGuard handle;
    MatDescrGuard matA;
    matA.initStructured(handle.ptr(), physRowsA, physColsA, aLd, 16, ctx.kDtype,
                        order, ACL_SPARSE_LT_SPARSITY_50_PERCENT);
    const int64_t logicalK = transA ? physRowsA : physColsA;
    // INT8 (low-precision) has op/layout combination constraints in
    // MatmulDescriptorInit: ROW+ROW requires N+T, but the wrapper uses opB=N.
    // Setting matB order=COL satisfies ROW+COL -> N+N (and COL+COL -> T+N),
    // which is valid for all INT8 A-sparse op/order combinations.
    // matB is a dense placeholder; its order does not affect prune results.
    const aclsparseOrder_t matBOrder = std::is_same_v<T, int8_t>
        ? ACL_SPARSE_ORDER_COL : ACL_SPARSE_ORDER_ROW;
    MatDescrGuard matB;
    matB.initDense(handle.ptr(), logicalK, kPruneN, kPruneN, 16, ctx.kDtype,
                   matBOrder);
    MatDescrGuard matC;
    matC.initDense(handle.ptr(), m, kPruneN, kPruneN, 16, ctx.kDtype,
                   ACL_SPARSE_ORDER_ROW);
    MatDescrGuard matD;
    matD.initDense(handle.ptr(), m, kPruneN, kPruneN, 16, ctx.kDtype,
                   ACL_SPARSE_ORDER_ROW);
    MatmulDescrGuard md;
    md.init(handle.ptr(), opA, ACL_SPARSE_OP_NON_TRANSPOSE,
            matA.ptr(), matB.ptr(), matC.ptr(), matD.ptr(), ctx.kComputeType);

    if (!RunPruneExecution(handle.ptr(), md.get(),
                            dA.get(), dAPruned.get(), pruneAlg, stream, result)) {
        return result;
    }

    hAPruned.resize(mk);
    dAPruned.copyToHost(hAPruned.data(), mk * ctx.kElt);
    return result;
}

// [B-SPARSE] PruneNpu for B-sparse: B is the structured (2:4) matrix.
// Params mB/kB denote B's (rows, cols) = (logical k, logical n).
// matA is a dense placeholder so MatmulDescriptorInit can derive m/k/n;
// its shape is chosen so md->k == mB (B rows). matmulDescr implicitly
// detects matB as structured (sparsity=50_PERCENT) and prunes B.
// alg_config_id / split_k are accepted for CSV signature compatibility but
// are NOT used by prune.
template <typename T>
inline PruneNpuResult PruneNpuBSparse(
    aclrtStream stream,
    const std::vector<T>& hB, std::vector<T>& hBPruned,
    int32_t mB, int32_t kB,
    int32_t alg_config_id, int32_t split_k,
    aclsparseOperation_t opB, aclsparseOrder_t order,
    aclsparseLtPruneAlg_t pruneAlg = ACLSPARSELT_PRUNE_SPMMA_STRIP)
{
    using namespace sparse_test;
    PruneNpuSetup<T> ctx(alg_config_id, split_k);
    auto& result = ctx.result;

    const bool transB = (opB == ACL_SPARSE_OP_TRANSPOSE);

    // matA dense placeholder: rows=mB, cols=mB (square, so md->k = matA->cols = mB = B rows).
    const int64_t aRows = static_cast<int64_t>(mB);
    const int64_t aCols = static_cast<int64_t>(mB);
    const int64_t aLd = aCols;

    // B physical layout: same logic as PreparePhysicalA for A-sparse
    // (transB + order determine whether to physically transpose hB).
    int64_t bLd = 0;
    std::vector<T> hB_phys = PreparePhysicalA<T>(hB, mB, kB, transB, order, bLd);
    const int64_t bPhysRows = transB ? static_cast<int64_t>(kB) : static_cast<int64_t>(mB);
    const int64_t bPhysCols = transB ? static_cast<int64_t>(mB) : static_cast<int64_t>(kB);
    const size_t mk = static_cast<size_t>(mB) * static_cast<size_t>(kB);
    // Same row-major size fix as PruneNpu (see PreparePhysicalA).
    const size_t bPhysSize = (transB && order == ACL_SPARSE_ORDER_COL)
        ? static_cast<size_t>(bPhysCols) * static_cast<size_t>(bLd)
        : static_cast<size_t>(bPhysRows) * static_cast<size_t>(bLd);

    DeviceBuffer dB = DeviceBuffer::copyFrom(hB_phys.data(), bPhysSize * ctx.kElt);
    DeviceBuffer dBPruned = DeviceBuffer::alloc(mk * ctx.kElt);

    SparseLtHandleGuard handle;
    // matA: dense placeholder (sparsity=NONE via DenseDescriptorInit).
    MatDescrGuard matA;
    matA.initDense(handle.ptr(), aRows, aCols, aLd, 16, ctx.kDtype,
                   ACL_SPARSE_ORDER_ROW);
    // matB: structured (sparsity=50_PERCENT).
    MatDescrGuard matB;
    matB.initStructured(handle.ptr(), bPhysRows, bPhysCols, bLd, 16, ctx.kDtype,
                        order, ACL_SPARSE_LT_SPARSITY_50_PERCENT);
    // matC/matD: dense placeholder (mB, kB) — shapes satisfy md->m=md->k=mB, md->n=kB.
    MatDescrGuard matC;
    matC.initDense(handle.ptr(), mB, kB, kB, 16, ctx.kDtype,
                   ACL_SPARSE_ORDER_ROW);
    MatDescrGuard matD;
    matD.initDense(handle.ptr(), mB, kB, kB, 16, ctx.kDtype,
                   ACL_SPARSE_ORDER_ROW);
    MatmulDescrGuard md;
    md.init(handle.ptr(), ACL_SPARSE_OP_NON_TRANSPOSE, opB,
            matA.ptr(), matB.ptr(), matC.ptr(), matD.ptr(), ctx.kComputeType);

    if (!RunPruneExecution(handle.ptr(), md.get(),
                            dB.get(), dBPruned.get(), pruneAlg, stream, result)) {
        return result;
    }

    hBPruned.resize(mk);
    dBPruned.copyToHost(hBPruned.data(), mk * ctx.kElt);
    return result;
}

#endif  // TEST_PRUNE_NPU_WRAPPER_H_
