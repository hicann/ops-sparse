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

#ifndef TEST_MATMUL_NPU_WRAPPER_H_
#define TEST_MATMUL_NPU_WRAPPER_H_

#include "ltmatmul_test_utils.h"  // NpuDtypeTrait, RAII guards, AlgSetAttributeNpu chain

// =============================================================================
// Matmul NPU wrapper for the ltmatmul test binary.
//
// Provides:
//   - RunMatmulNpu: full matmul chain (4 dtype x 2 path, A-sparse / B-sparse / dense)
//     with built-in getAttr roundtrip verification
//
// Supports:
//   - 4 input dtypes: FP32(float), FP16(uint16_t), BF16(bf16_bits_t), INT8(int8_t)
//   - INT8 dual output: INT8(int8_t) or INT32(int32_t)
//   - 2 paths: sparse*dense (with prune) / dense*dense (no prune)
//   - A-sparse / B-sparse
//   - Transpose (transA, transB)
//   - split_k + split_k_mode (ONE_KERNEL / TWO_KERNELS)
//   - prune_alg (STRIP / TILE)
//   - AlgSetAttribute / AlgGetAttribute roundtrip
//
// Foundational types (traits, RAII guards, AlgSetAttributeNpu chain) live in
// ltmatmul_test_utils.h, included above.
// =============================================================================

// =============================================================================
// Matmul NPU result + chain functions (original ltmatmul_npu_wrapper.h).
// =============================================================================

// -----------------------------------------------------------------------------
// Matmul NPU result.
// -----------------------------------------------------------------------------
struct MatmulNpuResult {
    aclsparseStatus_t algSetCfgRet = ACL_SPARSE_STATUS_SUCCESS;
    aclsparseStatus_t algSetSplitRet = ACL_SPARSE_STATUS_SUCCESS;
    aclsparseStatus_t algSetSplitKModeRet = ACL_SPARSE_STATUS_SUCCESS;
    // getAttr roundtrip verification (setAttr -> getAttr readback).
    aclsparseStatus_t algGetCfgRet = ACL_SPARSE_STATUS_SUCCESS;
    aclsparseStatus_t algGetSplitRet = ACL_SPARSE_STATUS_SUCCESS;
    aclsparseStatus_t algGetSplitKModeRet = ACL_SPARSE_STATUS_SUCCESS;
    int32_t gotAlgConfigId = -1;
    int32_t gotSplitK = -1;
    int32_t gotSplitKMode = -1;
    aclsparseStatus_t descSetRet = ACL_SPARSE_STATUS_SUCCESS;
    aclsparseStatus_t pruneRet = ACL_SPARSE_STATUS_SUCCESS;
    aclsparseStatus_t wsRet = ACL_SPARSE_STATUS_SUCCESS;
    aclsparseStatus_t matmulRet = ACL_SPARSE_STATUS_SUCCESS;
    aclError syncRet = ACL_SUCCESS;
    aclError memcpyRet = ACL_SUCCESS;
    size_t workspaceSize = 0;
    double npuMs = 0.0;
    std::vector<int8_t> npuPrunedA;
};

// -----------------------------------------------------------------------------
// Set alg attributes including split_k_mode (v2 extension).
// -----------------------------------------------------------------------------
inline bool SetMatmulAlgAttributes(aclsparseLtConstHandle_t handle,
                                    aclsparseLtMatmulAlgSelection_t algSel,
                                    int32_t algConfigId, int32_t splitK,
                                    int32_t splitKMode,
                                    MatmulNpuResult& result)
{
    result.algSetCfgRet = aclsparseLtMatmulAlgSetAttribute(
        handle, &algSel, ACLSPARSELT_MATMUL_ALG_CONFIG_ID,
        &algConfigId, sizeof(int32_t));
    if (result.algSetCfgRet != ACL_SPARSE_STATUS_SUCCESS) { return false; }

    result.algSetSplitRet = aclsparseLtMatmulAlgSetAttribute(
        handle, &algSel, ACLSPARSELT_MATMUL_SPLIT_K,
        &splitK, sizeof(int32_t));
    if (result.algSetSplitRet != ACL_SPARSE_STATUS_SUCCESS) { return false; }

    result.algSetSplitKModeRet = aclsparseLtMatmulAlgSetAttribute(
        handle, &algSel, ACLSPARSELT_MATMUL_SPLIT_K_MODE,
        &splitKMode, sizeof(int32_t));
    if (result.algSetSplitKModeRet != ACL_SPARSE_STATUS_SUCCESS) { return false; }

    return true;
}

// -----------------------------------------------------------------------------
// Vector scaling: set attributes on matmul descriptor + copy vectors to device.
// Extracted from RunMatmulChain / RunMatmulBSparseChainWithTrans
// [codecheck: duplicate code].
// -----------------------------------------------------------------------------
struct VecScalingBuffers {
    sparse_test::DeviceBuffer dAlphaVec;
    sparse_test::DeviceBuffer dBetaVec;
};

inline VecScalingBuffers SetVecScalingAttrs(aclsparseLtConstHandle_t handle,
    sparse_test::SparseLtMatmulDescGuard& matmulDesc, int32_t m,
    int32_t alpha_vector_scaling, int32_t beta_vector_scaling,
    const std::vector<float>* alphaVec, const std::vector<float>* betaVec,
    MatmulNpuResult& result)
{
    VecScalingBuffers bufs;
    if (alpha_vector_scaling && alphaVec != nullptr) {
        int enable = 1;
        result.descSetRet = aclsparseLtMatmulDescSetAttribute(handle, &matmulDesc.get(),
            ACLSPARSELT_MATMUL_ALPHA_VECTOR_SCALING, &enable, sizeof(int));
        if (result.descSetRet != ACL_SPARSE_STATUS_SUCCESS) { return bufs; }
        bufs.dAlphaVec = sparse_test::DeviceBuffer::copyFrom(alphaVec->data(),
            static_cast<size_t>(m) * sizeof(float));
    }
    if (beta_vector_scaling && betaVec != nullptr) {
        int enable = 1;
        result.descSetRet = aclsparseLtMatmulDescSetAttribute(handle, &matmulDesc.get(),
            ACLSPARSELT_MATMUL_BETA_VECTOR_SCALING, &enable, sizeof(int));
        if (result.descSetRet != ACL_SPARSE_STATUS_SUCCESS) { return bufs; }
        bufs.dBetaVec = sparse_test::DeviceBuffer::copyFrom(betaVec->data(),
            static_cast<size_t>(m) * sizeof(float));
    }
    return bufs;
}

// -----------------------------------------------------------------------------
// Execute matmul + sync + record timing + copy D back to host.
// Extracted from RunMatmulChain / RunMatmulBSparseChainWithTrans
// [codecheck: duplicate code].
// -----------------------------------------------------------------------------
template <typename OutT>
inline void ExecMatmulAndCopyResult(MatmulNpuResult& result,
    aclsparseLtConstHandle_t handle, aclsparseLtConstMatmulPlan_t* plan,
    void* matAPtr, void* matBPtr, void* matC, void* matD, void* wsPtr,
    VecScalingBuffers& vecBufs, float alpha, float beta,
    int32_t alpha_vector_scaling, int32_t beta_vector_scaling,
    sparse_test::DeviceBuffer& dD, std::vector<OutT>& hD, size_t mn, size_t kOutElt)
{
    using namespace sparse_test;
    aclrtStream stream = 0;  // default stream
    float alphaF = alpha;
    float betaF = beta;
    const void* alphaPtr = (alpha_vector_scaling && vecBufs.dAlphaVec.get())
        ? vecBufs.dAlphaVec.get() : static_cast<const void*>(&alphaF);
    const void* betaPtr = (beta_vector_scaling && vecBufs.dBetaVec.get())
        ? vecBufs.dBetaVec.get() : static_cast<const void*>(&betaF);

    EventGuard evStart, evStop;
    evStart.record(stream);
    result.matmulRet = aclsparseLtMatmul(
        handle, plan, alphaPtr, matAPtr, matBPtr, betaPtr,
        matC, matD, wsPtr, &stream, 1);
    evStop.record(stream);
    auto syncRet = aclrtSynchronizeStream(stream);
    if (result.matmulRet != ACL_SPARSE_STATUS_SUCCESS) { return; }
    if (syncRet != ACL_SUCCESS) {
        result.matmulRet = ACL_SPARSE_STATUS_EXECUTION_FAILED;
        return;
    }
    result.npuMs = static_cast<double>(EventGuard::elapsedMs(evStart, evStop));
    hD.resize(mn);
    dD.copyToHost(hD.data(), mn * kOutElt);
}

// -----------------------------------------------------------------------------
// MatmulChainCtx: aggregated context for RunMatmulChain (descriptors + buffers).
// Uses unique_ptr for RAII guards since they have no default constructor.
// Extracted from RunMatmulChain to reduce NBNC.
// -----------------------------------------------------------------------------
struct MatmulChainCtx {
    sparse_test::SparseLtHandleGuard handle;
    std::unique_ptr<sparse_test::SparseLtMatDescGuard> matAStructured;
    std::unique_ptr<sparse_test::SparseLtDnMatDescGuard> matADense;
    std::unique_ptr<sparse_test::SparseLtMatDescGuard> matBStructured;
    std::unique_ptr<sparse_test::SparseLtDnMatDescGuard> matBDense;
    std::unique_ptr<sparse_test::SparseLtDnMatDescGuard> matC;
    std::unique_ptr<sparse_test::SparseLtDnMatDescGuard> matD;
    std::unique_ptr<sparse_test::SparseLtMatmulDescGuard> matmulDesc;
    sparse_test::DeviceBuffer dA, dB, dC, dD;
    size_t mk = 0;
};

// -----------------------------------------------------------------------------
// MatmulBSparseCtx: aggregated context for RunMatmulBSparseChainWithTrans.
// -----------------------------------------------------------------------------
struct MatmulBSparseCtx {
    sparse_test::SparseLtHandleGuard handle;
    std::unique_ptr<sparse_test::SparseLtDnMatDescGuard> matA;
    std::unique_ptr<sparse_test::SparseLtMatDescGuard> matB;
    std::unique_ptr<sparse_test::SparseLtDnMatDescGuard> matC;
    std::unique_ptr<sparse_test::SparseLtDnMatDescGuard> matD;
    std::unique_ptr<sparse_test::SparseLtMatmulDescGuard> matmulDesc;
    sparse_test::DeviceBuffer dA, dB, dC, dD;
    size_t kn = 0;
};

// -----------------------------------------------------------------------------
// PrepareMatmulChainBuffers: compute physical layout + allocate device buffers.
// Extracted from PrepareMatmulChainContext to reduce NBNC.
// -----------------------------------------------------------------------------
template <typename InT, typename OutT>
inline void PrepareMatmulChainBuffers(
    int32_t m, int32_t k, int32_t n,
    const std::vector<InT>& hA, const std::vector<InT>& hB, const std::vector<InT>& hC,
    aclsparseOrder_t order, aclsparseOperation_t opA, aclsparseOperation_t opB,
    MatmulChainCtx& ctx,
    int64_t& physRowsA, int64_t& physColsA, int64_t& aLd,
    int64_t& bPhysRows, int64_t& bPhysCols, int64_t& bLd)
{
    using namespace sparse_test;
    constexpr size_t kInElt = NpuDtypeTrait<InT>::kEltSize;
    constexpr size_t kOutElt = NpuDtypeTrait<OutT>::kEltSize;
    const bool transA = (opA == ACL_SPARSE_OP_TRANSPOSE);
    const bool transB = (opB == ACL_SPARSE_OP_TRANSPOSE);
    ctx.mk = static_cast<size_t>(m) * k;
    const size_t kn = static_cast<size_t>(k) * n;
    const size_t mn = static_cast<size_t>(m) * n;
    std::vector<InT> hA_phys = PreparePhysicalA<InT>(hA, m, k, transA, order, aLd);
    std::vector<InT> hB_phys = PreparePhysicalB<InT>(hB, k, n, transB);
    physRowsA = transA ? static_cast<int64_t>(k) : static_cast<int64_t>(m);
    physColsA = transA ? static_cast<int64_t>(m) : static_cast<int64_t>(k);
    const size_t aPhysSize = (order == ACL_SPARSE_ORDER_COL)
        ? static_cast<size_t>(physColsA) * static_cast<size_t>(aLd)
        : static_cast<size_t>(physRowsA) * static_cast<size_t>(aLd);
    const size_t bPhysSize = transB ? static_cast<size_t>(n) * k : kn;
    ctx.dA = DeviceBuffer::copyFrom(hA_phys.data(), aPhysSize * kInElt);
    ctx.dB = DeviceBuffer::copyFrom(hB_phys.data(), bPhysSize * kInElt);
    ctx.dC = DeviceBuffer::copyFrom(hC.data(), mn * kInElt);
    ctx.dD = DeviceBuffer::alloc(mn * kOutElt);
    bPhysRows = transB ? static_cast<int64_t>(n) : static_cast<int64_t>(k);
    bPhysCols = transB ? static_cast<int64_t>(k) : static_cast<int64_t>(n);
    bLd = transB ? static_cast<int64_t>(k) : static_cast<int64_t>(n);
}

// -----------------------------------------------------------------------------
// CreateMatmulABDescriptors: create matA/matB descriptors based on path
// (dense*dense / A-sparse / B-sparse). Extracted from PrepareMatmulChainContext.
// -----------------------------------------------------------------------------
template <typename InT>
inline void CreateMatmulABDescriptors(
    MatmulChainCtx& ctx, bool isDensePath, bool isSparseA,
    aclsparseOrder_t order, int64_t physRowsA, int64_t physColsA, int64_t aLd,
    int64_t bPhysRows, int64_t bPhysCols, int64_t bLd,
    aclsparseLtMatDescriptor_t& matADesc, aclsparseLtMatDescriptor_t& matBDesc)
{
    using namespace sparse_test;
    constexpr aclDataType kInDtype = NpuDtypeTrait<InT>::kAclDtype;
    if (isDensePath) {
        ctx.matADense = std::make_unique<SparseLtDnMatDescGuard>(
            ctx.handle.get(), physRowsA, physColsA, aLd, 16, kInDtype, order);
        matADesc = ctx.matADense->get();
        ctx.matBDense = std::make_unique<SparseLtDnMatDescGuard>(
            ctx.handle.get(), bPhysRows, bPhysCols, bLd, 16, kInDtype, ACL_SPARSE_ORDER_ROW);
        matBDesc = ctx.matBDense->get();
    } else if (isSparseA) {
        ctx.matAStructured = std::make_unique<SparseLtMatDescGuard>(
            ctx.handle.get(), physRowsA, physColsA, aLd, 16, kInDtype, order,
            ACL_SPARSE_LT_SPARSITY_50_PERCENT);
        matADesc = ctx.matAStructured->get();
        ctx.matBDense = std::make_unique<SparseLtDnMatDescGuard>(
            ctx.handle.get(), bPhysRows, bPhysCols, bLd, 16, kInDtype, ACL_SPARSE_ORDER_ROW);
        matBDesc = ctx.matBDense->get();
    } else {
        ctx.matADense = std::make_unique<SparseLtDnMatDescGuard>(
            ctx.handle.get(), physRowsA, physColsA, aLd, 16, kInDtype, order);
        matADesc = ctx.matADense->get();
        ctx.matBStructured = std::make_unique<SparseLtMatDescGuard>(
            ctx.handle.get(), bPhysRows, bPhysCols, bLd, 16, kInDtype,
            ACL_SPARSE_ORDER_ROW, ACL_SPARSE_LT_SPARSITY_50_PERCENT);
        matBDesc = ctx.matBStructured->get();
    }
}

// -----------------------------------------------------------------------------
// PrepareMatmulChainContext: compute physical layout + allocate device buffers
// + create descriptors + matmulDesc. Extracted from RunMatmulChain to reduce NBNC.
// -----------------------------------------------------------------------------
template <typename InT, typename OutT>
inline void PrepareMatmulChainContext(
    int32_t m, int32_t k, int32_t n,
    const std::vector<InT>& hA, const std::vector<InT>& hB, const std::vector<InT>& hC,
    aclsparseOrder_t order, aclsparseOperation_t opA, aclsparseOperation_t opB,
    bool isSparseA, bool isDensePath, MatmulChainCtx& ctx)
{
    using namespace sparse_test;
    constexpr aclDataType kInDtype = NpuDtypeTrait<InT>::kAclDtype;
    constexpr aclDataType kOutDtype = NpuDtypeTrait<OutT>::kAclDtype;
    constexpr aclsparseComputeType_t kComputeType = MatmulComputeTrait<InT>::kComputeType;
    int64_t physRowsA = 0, physColsA = 0, aLd = 0, bPhysRows = 0, bPhysCols = 0, bLd = 0;
    PrepareMatmulChainBuffers<InT, OutT>(m, k, n, hA, hB, hC, order, opA, opB,
        ctx, physRowsA, physColsA, aLd, bPhysRows, bPhysCols, bLd);
    aclsparseLtMatDescriptor_t matADesc = nullptr, matBDesc = nullptr;
    CreateMatmulABDescriptors<InT>(ctx, isDensePath, isSparseA, order,
        physRowsA, physColsA, aLd, bPhysRows, bPhysCols, bLd, matADesc, matBDesc);
    ctx.matC = std::make_unique<SparseLtDnMatDescGuard>(ctx.handle.get(), m, n, n, 16, kInDtype, ACL_SPARSE_ORDER_ROW);
    ctx.matD = std::make_unique<SparseLtDnMatDescGuard>(ctx.handle.get(), m, n, n, 16, kOutDtype, ACL_SPARSE_ORDER_ROW);
    ctx.matmulDesc = std::make_unique<SparseLtMatmulDescGuard>(ctx.handle.get(), opA, opB,
        matADesc, matBDesc, ctx.matC->get(), ctx.matD->get(), kComputeType);
}

// -----------------------------------------------------------------------------
// PreparePlanAndWorkspace: create algSel + plan + workspace.
// Extracted from RunMatmulChain/RunMatmulBSparseChainWithTrans to reduce NBNC.
// -----------------------------------------------------------------------------
inline bool PreparePlanAndWorkspace(MatmulNpuResult& result,
    sparse_test::SparseLtHandleGuard& handle,
    sparse_test::SparseLtMatmulDescGuard& matmulDesc,
    int32_t alg_config_id, int32_t split_k, int32_t split_k_mode,
    std::unique_ptr<sparse_test::SparseLtAlgSelectionGuard>& algSel,
    std::unique_ptr<sparse_test::SparseLtPlanGuard>& plan,
    sparse_test::DeviceBuffer& dWorkspace, void*& wsPtr)
{
    using namespace sparse_test;
    algSel = std::make_unique<SparseLtAlgSelectionGuard>(handle.get(), matmulDesc.get(), ACL_SPARSE_LT_MATMUL_ALG_DEFAULT);
    if (!SetMatmulAlgAttributes(handle.get(), algSel->get(),
                                alg_config_id, split_k, split_k_mode, result)) {
        return false;
    }
    // getAttr roundtrip verification: read back the values set by SetMatmulAlgAttributes.
    result.algGetCfgRet = aclsparseLtMatmulAlgGetAttribute(
        handle.get(), algSel->cptr(),
        ACLSPARSELT_MATMUL_ALG_CONFIG_ID, &result.gotAlgConfigId, sizeof(int32_t));
    result.algGetSplitRet = aclsparseLtMatmulAlgGetAttribute(
        handle.get(), algSel->cptr(),
        ACLSPARSELT_MATMUL_SPLIT_K, &result.gotSplitK, sizeof(int32_t));
    result.algGetSplitKModeRet = aclsparseLtMatmulAlgGetAttribute(
        handle.get(), algSel->cptr(),
        ACLSPARSELT_MATMUL_SPLIT_K_MODE, &result.gotSplitKMode, sizeof(int32_t));
    plan = std::make_unique<SparseLtPlanGuard>(handle.get(), matmulDesc.get(), algSel->get());
    result.wsRet = aclsparseLtMatmulGetWorkspaceSize(handle.get(), plan->cptr(), &result.workspaceSize);
    if (result.wsRet != ACL_SPARSE_STATUS_SUCCESS) { return false; }
    dWorkspace = (result.workspaceSize > 0)
        ? DeviceBuffer::alloc(result.workspaceSize) : DeviceBuffer{};
    wsPtr = (result.workspaceSize > 0) ? dWorkspace.get() : nullptr;
    return true;
}

// -----------------------------------------------------------------------------
// RunPruneForASparse: run prune on A-sparse path, update matAPtr + npuPrunedA.
// Extracted from RunMatmulChain to reduce NBNC.
// -----------------------------------------------------------------------------
inline bool RunPruneForASparse(MatmulNpuResult& result,
    aclsparseLtConstHandle_t handle, aclsparseLtConstMatmulDescriptor_t matmulDesc,
    sparse_test::DeviceBuffer& dA, sparse_test::DeviceBuffer& dPruned,
    size_t mk, size_t kInElt, aclsparseLtPruneAlg_t pruneAlg, void*& matAPtr)
{
    aclrtStream stream = 0;
    dPruned = sparse_test::DeviceBuffer::alloc(mk * kInElt);
    result.pruneRet = aclsparseLtSpMMAPrune(
        handle, &matmulDesc, dA.get(), dPruned.get(), pruneAlg, stream);
    if (result.pruneRet != ACL_SPARSE_STATUS_SUCCESS) { return false; }
    result.syncRet = aclrtSynchronizeStream(stream);
    if (result.syncRet != ACL_SUCCESS) {
        result.pruneRet = ACL_SPARSE_STATUS_EXECUTION_FAILED;
        return false;
    }
    result.npuPrunedA.resize(mk * kInElt);
    result.memcpyRet = aclrtMemcpy(result.npuPrunedA.data(), mk * kInElt, dPruned.get(), mk * kInElt,
                ACL_MEMCPY_DEVICE_TO_HOST);
    if (result.memcpyRet != ACL_SUCCESS) {
        result.pruneRet = ACL_SPARSE_STATUS_EXECUTION_FAILED;
        return false;
    }
    matAPtr = dPruned.get();
    return true;
}

// -----------------------------------------------------------------------------
// [codecheck-dup] MatmulRunArgs: common parameters for matmul chain functions.
// Extracted to eliminate duplicate parameter lists between RunMatmulChain,
// RunMatmulBSparseChainWithTrans, and RunMatmulNpu.
// -----------------------------------------------------------------------------
template <typename InT, typename OutT>
struct MatmulRunArgs {
    int32_t m, k, n;
    const std::vector<InT>& hA;
    const std::vector<InT>& hB;
    const std::vector<InT>& hC;
    std::vector<OutT>& hD;
    float alpha, beta;
    int32_t alg_config_id, split_k, split_k_mode;
    aclsparseOrder_t order;
    aclsparseOperation_t opA, opB;
    aclsparseLtPruneAlg_t pruneAlg;
    int32_t alpha_vector_scaling;
    int32_t beta_vector_scaling;
    const std::vector<float>* alphaVec;
    const std::vector<float>* betaVec;
};

// -----------------------------------------------------------------------------
// Run matmul chain for A-sparse or dense*dense path.
// (B-sparse path is handled separately by RunMatmulBSparseChainWithTrans
//  because it needs proper transB buffer lifetime management.)
// -----------------------------------------------------------------------------
template <typename InT, typename OutT>
inline MatmulNpuResult RunMatmulChain(
    const MatmulRunArgs<InT, OutT>& args, bool isSparseA, bool isDensePath)
{
    using namespace sparse_test;
    MatmulNpuResult result{};
    constexpr size_t kInElt = NpuDtypeTrait<InT>::kEltSize, kOutElt = NpuDtypeTrait<OutT>::kEltSize;
    const size_t mn = static_cast<size_t>(args.m) * args.n;
    MatmulChainCtx ctx;
    PrepareMatmulChainContext<InT, OutT>(args.m, args.k, args.n,
                                         args.hA, args.hB, args.hC,
                                         args.order, args.opA, args.opB,
                                         isSparseA, isDensePath, ctx);
    auto vecBufs = SetVecScalingAttrs(ctx.handle.get(), *ctx.matmulDesc, args.m,
        args.alpha_vector_scaling, args.beta_vector_scaling, args.alphaVec, args.betaVec, result);
    if (result.descSetRet != ACL_SPARSE_STATUS_SUCCESS) { return result; }
    std::unique_ptr<SparseLtAlgSelectionGuard> algSel;
    std::unique_ptr<SparseLtPlanGuard> plan;
    DeviceBuffer dWorkspace, dPruned;
    void* wsPtr = nullptr;
    if (!PreparePlanAndWorkspace(result, ctx.handle, *ctx.matmulDesc, args.alg_config_id,
                                 args.split_k, args.split_k_mode, algSel, plan, dWorkspace, wsPtr)) { return result; }
    void* matAPtr = ctx.dA.get();
    if (!isDensePath && isSparseA) {
        if (!RunPruneForASparse(result, ctx.handle.get(), ctx.matmulDesc->get(),
                ctx.dA, dPruned, ctx.mk, kInElt, args.pruneAlg, matAPtr)) { return result; }
    }
    ExecMatmulAndCopyResult<OutT>(result, ctx.handle.get(), plan->cptr(),
        matAPtr, ctx.dB.get(), ctx.dC.get(), ctx.dD.get(), wsPtr, vecBufs, args.alpha, args.beta,
        args.alpha_vector_scaling, args.beta_vector_scaling, ctx.dD, args.hD, mn, kOutElt);
    return result;
}

// -----------------------------------------------------------------------------
// PrepareBSparseChainContext: compute physical layout + allocate device buffers
// + create B-sparse descriptors. Extracted from RunMatmulBSparseChainWithTrans.
// -----------------------------------------------------------------------------
template <typename InT, typename OutT>
inline void PrepareBSparseChainContext(
    int32_t m, int32_t k, int32_t n,
    const std::vector<InT>& hA, const std::vector<InT>& hB, const std::vector<InT>& hC,
    aclsparseOrder_t order, aclsparseOperation_t opA, aclsparseOperation_t opB,
    MatmulBSparseCtx& ctx)
{
    using namespace sparse_test;
    constexpr aclDataType kInDtype = NpuDtypeTrait<InT>::kAclDtype;
    constexpr aclDataType kOutDtype = NpuDtypeTrait<OutT>::kAclDtype;
    constexpr aclsparseComputeType_t kComputeType = MatmulComputeTrait<InT>::kComputeType;
    constexpr size_t kInElt = NpuDtypeTrait<InT>::kEltSize;
    constexpr size_t kOutElt = NpuDtypeTrait<OutT>::kEltSize;
    const bool transA = (opA == ACL_SPARSE_OP_TRANSPOSE);
    const bool transB = (opB == ACL_SPARSE_OP_TRANSPOSE);
    const size_t mn = static_cast<size_t>(m) * n;
    ctx.kn = static_cast<size_t>(k) * n;
    int64_t aLd = 0, bLdOut = 0;
    std::vector<InT> hA_phys = PreparePhysicalA<InT>(hA, m, k, transA, ACL_SPARSE_ORDER_ROW, aLd);
    std::vector<InT> hB_phys = PreparePhysicalA<InT>(hB, k, n, transB, order, bLdOut);
    const int64_t physRowsA = transA ? static_cast<int64_t>(k) : static_cast<int64_t>(m);
    const int64_t physColsA = transA ? static_cast<int64_t>(m) : static_cast<int64_t>(k);
    const size_t aPhysSize = static_cast<size_t>(physRowsA) * static_cast<size_t>(aLd);
    const size_t bPhysSize = static_cast<size_t>(ctx.kn);
    ctx.dA = DeviceBuffer::copyFrom(hA_phys.data(), aPhysSize * kInElt);
    ctx.dB = DeviceBuffer::copyFrom(hB_phys.data(), bPhysSize * kInElt);
    ctx.dC = DeviceBuffer::copyFrom(hC.data(), mn * kInElt);
    ctx.dD = DeviceBuffer::alloc(mn * kOutElt);
    const int64_t bPhysRows = transB ? static_cast<int64_t>(n) : static_cast<int64_t>(k);
    const int64_t bPhysCols = transB ? static_cast<int64_t>(k) : static_cast<int64_t>(n);
    ctx.matA = std::make_unique<SparseLtDnMatDescGuard>(ctx.handle.get(), physRowsA, physColsA, aLd, 16, kInDtype, ACL_SPARSE_ORDER_ROW);
    ctx.matB = std::make_unique<SparseLtMatDescGuard>(ctx.handle.get(), bPhysRows, bPhysCols, bLdOut, 16, kInDtype,
                                order, ACL_SPARSE_LT_SPARSITY_50_PERCENT);
    ctx.matC = std::make_unique<SparseLtDnMatDescGuard>(ctx.handle.get(), m, n, n, 16, kInDtype, ACL_SPARSE_ORDER_ROW);
    ctx.matD = std::make_unique<SparseLtDnMatDescGuard>(ctx.handle.get(), m, n, n, 16, kOutDtype, ACL_SPARSE_ORDER_ROW);
    ctx.matmulDesc = std::make_unique<SparseLtMatmulDescGuard>(ctx.handle.get(), opA, opB,
        ctx.matA->get(), ctx.matB->get(), ctx.matC->get(), ctx.matD->get(), kComputeType);
}

// -----------------------------------------------------------------------------
// RunBSparsePruneAndTranspose: prune B + optional transB transpose.
// Extracted from RunMatmulBSparseChainWithTrans to reduce NBNC.
// -----------------------------------------------------------------------------
template <typename InT>
inline void RunBSparsePruneAndTranspose(MatmulNpuResult& result,
    aclsparseLtConstHandle_t handle, aclsparseLtConstMatmulDescriptor_t matmulDesc,
    sparse_test::DeviceBuffer& dB, sparse_test::DeviceBuffer& dBPruned,
    sparse_test::DeviceBuffer& dBPrunedTransposed,
    bool transB, int32_t k, int32_t n, size_t kn, size_t kInElt,
    aclsparseLtPruneAlg_t pruneAlg, void*& matBPtr)
{
    using namespace sparse_test;
    aclrtStream stream = 0;
    dBPruned = DeviceBuffer::alloc(kn * kInElt);
    result.pruneRet = aclsparseLtSpMMAPrune(
        handle, &matmulDesc, dB.get(), dBPruned.get(), pruneAlg, stream);
    if (result.pruneRet != ACL_SPARSE_STATUS_SUCCESS) { return; }
    matBPtr = dBPruned.get();
    if (transB) {
        result.syncRet = aclrtSynchronizeStream(stream);
        if (result.syncRet != ACL_SUCCESS) {
            result.pruneRet = ACL_SPARSE_STATUS_EXECUTION_FAILED;
            return;
        }
        dBPrunedTransposed = TransposeBPruned<InT>(stream, dBPruned, k, n, kn, kInElt);
        matBPtr = dBPrunedTransposed.get();
    }
}

// -----------------------------------------------------------------------------
// B-sparse chain with transB support (proper buffer lifetime management).
// -----------------------------------------------------------------------------
template <typename InT, typename OutT>
inline MatmulNpuResult RunMatmulBSparseChainWithTrans(
    const MatmulRunArgs<InT, OutT>& args)
{
    using namespace sparse_test;
    MatmulNpuResult result{};
    constexpr size_t kInElt = NpuDtypeTrait<InT>::kEltSize, kOutElt = NpuDtypeTrait<OutT>::kEltSize;
    const size_t mn = static_cast<size_t>(args.m) * args.n;
    MatmulBSparseCtx ctx;
    PrepareBSparseChainContext<InT, OutT>(args.m, args.k, args.n,
                                          args.hA, args.hB, args.hC,
                                          args.order, args.opA, args.opB, ctx);
    auto vecBufs = SetVecScalingAttrs(ctx.handle.get(), *ctx.matmulDesc, args.m,
        args.alpha_vector_scaling, args.beta_vector_scaling, args.alphaVec, args.betaVec, result);
    if (result.descSetRet != ACL_SPARSE_STATUS_SUCCESS) { return result; }
    std::unique_ptr<SparseLtAlgSelectionGuard> algSel;
    std::unique_ptr<SparseLtPlanGuard> plan;
    DeviceBuffer dWorkspace, dBPruned, dBPrunedTransposed;
    void* wsPtr = nullptr;
    if (!PreparePlanAndWorkspace(result, ctx.handle, *ctx.matmulDesc, args.alg_config_id,
                                 args.split_k, args.split_k_mode, algSel, plan, dWorkspace, wsPtr)) { return result; }
    const bool transB = (args.opB == ACL_SPARSE_OP_TRANSPOSE);
    void* matBPtr = nullptr;
    RunBSparsePruneAndTranspose<InT>(result, ctx.handle.get(), ctx.matmulDesc->get(),
        ctx.dB, dBPruned, dBPrunedTransposed, transB, args.k, args.n, ctx.kn, kInElt, args.pruneAlg, matBPtr);
    if (result.pruneRet != ACL_SPARSE_STATUS_SUCCESS) { return result; }
    ExecMatmulAndCopyResult<OutT>(result, ctx.handle.get(), plan->cptr(),
        ctx.dA.get(), matBPtr, ctx.dC.get(), ctx.dD.get(), wsPtr, vecBufs, args.alpha, args.beta,
        args.alpha_vector_scaling, args.beta_vector_scaling, ctx.dD, args.hD, mn, kOutElt);
    return result;
}

// -----------------------------------------------------------------------------
// Dispatch: select chain based on path (sparse A-sparse / sparse B-sparse / dense).
// -----------------------------------------------------------------------------
template <typename InT, typename OutT>
inline MatmulNpuResult RunMatmulNpu(
    int32_t m, int32_t k, int32_t n,
    const std::vector<InT>& hA, const std::vector<InT>& hB, const std::vector<InT>& hC,
    std::vector<OutT>& hD,
    float alpha, float beta,
    int32_t alg_config_id, int32_t split_k, int32_t split_k_mode,
    bool isSparseA, bool isDensePath,
    aclsparseOrder_t order,
    aclsparseOperation_t opA, aclsparseOperation_t opB,
    aclsparseLtPruneAlg_t pruneAlg = ACLSPARSELT_PRUNE_SPMMA_STRIP,
    int32_t alpha_vector_scaling = 0,
    int32_t beta_vector_scaling = 0,
    const std::vector<float>* alphaVec = nullptr,
    const std::vector<float>* betaVec = nullptr)
{
    MatmulRunArgs<InT, OutT> args{
        m, k, n, hA, hB, hC, hD,
        alpha, beta, alg_config_id, split_k, split_k_mode,
        order, opA, opB, pruneAlg,
        alpha_vector_scaling, beta_vector_scaling, alphaVec, betaVec
    };
    if (isDensePath) {
        // Dense*dense: no prune, A=dense, B=dense.
        return RunMatmulChain<InT, OutT>(args, true, true);
    } else if (isSparseA) {
        // A-sparse: prune A, matmul(A_pruned, B).
        return RunMatmulChain<InT, OutT>(args, true, false);
    } else {
        // B-sparse: prune B, matmul(A, B_pruned). Use dedicated handler for transB support.
        return RunMatmulBSparseChainWithTrans<InT, OutT>(args);
    }
}

#endif  // TEST_MATMUL_NPU_WRAPPER_H_

