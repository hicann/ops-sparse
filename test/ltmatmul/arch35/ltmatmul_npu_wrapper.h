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

#include <cfloat>

#include "ltmatmul_test_utils.h"  // NpuDtypeTrait, RAII guards, AlgSetAttributeNpu chain

// =============================================================================
// LT_TEST_NPU_EPILOGUE_ENABLED: compile-time gate for bias/activation
// SetAttribute calls on the NPU side.
//
// The operator side (feat/bias-activation-epilogue) has extended
// aclsparseLtMatmulDescAttribute_t with BIAS_POINTER(2), BIAS_STRIDE(3),
// ACTIVATION_RELU(4), ACTIVATION_RELU_UPPERBOUND(5), ACTIVATION_RELU_THRESHOLD(6),
// ACTIVATION_GELU(7), ACTIVATION_GELU_SCALING(8). So epilogue (stage 1:
// bias + activation) is enabled by default.
//
// LT_TEST_NPU_BATCH_ENABLED: gate for stage-2 batch attributes (NUM_BATCHES,
// BATCH_STRIDE on MatDesc). The operator side has implemented batch support
// (aclsparseLtMatDescSetAttribute with ACLSPARSELT_MAT_NUM_BATCHES /
//  ACLSPARSELT_MAT_BATCH_STRIDE); batch cases now run the full NPU path.
// =============================================================================
#ifndef LT_TEST_NPU_EPILOGUE_ENABLED
#define LT_TEST_NPU_EPILOGUE_ENABLED 1
#endif
#ifndef LT_TEST_NPU_BATCH_ENABLED
#define LT_TEST_NPU_BATCH_ENABLED 1
#endif

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
    // Epilogue SetAttribute return codes (always SUCCESS when epilogue disabled
    // or LT_TEST_NPU_EPILOGUE_ENABLED==0).
    aclsparseStatus_t biasSetRet = ACL_SPARSE_STATUS_SUCCESS;
    aclsparseStatus_t actSetRet = ACL_SPARSE_STATUS_SUCCESS;
    aclsparseStatus_t batchSetRet = ACL_SPARSE_STATUS_SUCCESS;
    bool epilogueApplied = false;  // true iff NPU actually applied bias/activation
};

#include "ltmatmul_npu_epilogue.h"  // EpilogueBuffers, SetEpilogueAttrs

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
// to eliminate duplicate code.
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
// SetBatchAttrsOnMatDesc: set NUM_BATCHES + BATCH_STRIDE on a matrix descriptor.
//
// Must be called BEFORE aclsparseLtMatmulDescriptorInit (which performs the
// A/B/C/D numBatches consistency check). All four matrices must share the same
// numBatches; batch_stride may differ per-matrix but the test uses a single
// stride from the CSV for all four.
//
// When LT_TEST_NPU_BATCH_ENABLED==0 or num_batches<=1, this is a no-op.
// -----------------------------------------------------------------------------
inline aclsparseStatus_t SetBatchAttrsOnMatDesc(aclsparseLtConstHandle_t handle,
    aclsparseLtMatDescriptor_t* matDesc, int32_t num_batches, int64_t batch_stride)
{
#if LT_TEST_NPU_BATCH_ENABLED
    if (num_batches <= 1) { return ACL_SPARSE_STATUS_SUCCESS; }
    aclsparseStatus_t st = aclsparseLtMatDescSetAttribute(handle, matDesc,
        ACLSPARSELT_MAT_NUM_BATCHES, &num_batches, sizeof(int32_t));
    if (st != ACL_SPARSE_STATUS_SUCCESS) { return st; }
    if (batch_stride > 0) {
        st = aclsparseLtMatDescSetAttribute(handle, matDesc,
            ACLSPARSELT_MAT_BATCH_STRIDE, &batch_stride, sizeof(int64_t));
    }
    return st;
#else
    (void)handle; (void)matDesc; (void)num_batches; (void)batch_stride;
    return ACL_SPARSE_STATUS_SUCCESS;
#endif
}

// Convenience: set batch attrs on all four matrix descriptors + record result.
inline void SetBatchAttrsOnAllMatDescs(aclsparseLtConstHandle_t handle,
    aclsparseLtMatDescriptor_t* matA, aclsparseLtMatDescriptor_t* matB,
    aclsparseLtMatDescriptor_t* matC, aclsparseLtMatDescriptor_t* matD,
    int32_t num_batches, int64_t batch_stride, MatmulNpuResult& result)
{
#if LT_TEST_NPU_BATCH_ENABLED
    if (num_batches <= 1) {
        result.batchSetRet = ACL_SPARSE_STATUS_SUCCESS;
        return;
    }
    aclsparseStatus_t st = ACL_SPARSE_STATUS_SUCCESS;
    st = SetBatchAttrsOnMatDesc(handle, matA, num_batches, batch_stride);
    if (st != ACL_SPARSE_STATUS_SUCCESS) { result.batchSetRet = st; return; }
    st = SetBatchAttrsOnMatDesc(handle, matB, num_batches, batch_stride);
    if (st != ACL_SPARSE_STATUS_SUCCESS) { result.batchSetRet = st; return; }
    st = SetBatchAttrsOnMatDesc(handle, matC, num_batches, batch_stride);
    if (st != ACL_SPARSE_STATUS_SUCCESS) { result.batchSetRet = st; return; }
    st = SetBatchAttrsOnMatDesc(handle, matD, num_batches, batch_stride);
    result.batchSetRet = st;
#else
    (void)handle; (void)matA; (void)matB; (void)matC; (void)matD;
    (void)num_batches; (void)batch_stride;
    result.batchSetRet = ACL_SPARSE_STATUS_SUCCESS;
#endif
}

// -----------------------------------------------------------------------------
// BatchTotalElems: compute total element count for batch-strided buffers.
// When num_batches>1 and batch_stride>0, each matrix buffer is
// num_batches * batch_stride elements (with padding between batches).
// Otherwise, single-batch element count.
// -----------------------------------------------------------------------------
inline size_t BatchTotalElems(size_t singleBatchElems, int32_t num_batches, int64_t batch_stride)
{
    if (num_batches > 1 && batch_stride > 0) {
        return static_cast<size_t>(num_batches) * static_cast<size_t>(batch_stride);
    }
    return singleBatchElems;
}

// -----------------------------------------------------------------------------
// Execute matmul + sync + record timing + copy D back to host.
// Extracted from RunMatmulChain / RunMatmulBSparseChainWithTrans
// to eliminate duplicate code.
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

#include "ltmatmul_npu_chain.h"  // chain runners (ctx, prepare, prune, RunMatmulChain/BSparse)

// -----------------------------------------------------------------------------
// Dispatch: select chain based on path (sparse A-sparse / sparse B-sparse / dense).
// -----------------------------------------------------------------------------
template <typename InT, typename OutT>
inline MatmulNpuResult RunMatmulNpu(
    int32_t m, int32_t k, int32_t n,
    const std::vector<InT>& hA, const std::vector<InT>& hB, const std::vector<InT>& hC,
    std::vector<OutT>& hD,
    float alpha, float beta,
    int32_t algConfigId, int32_t splitK, int32_t splitKMode,
    bool isSparseA, bool isDensePath,
    aclsparseOrder_t order,
    aclsparseOperation_t opA, aclsparseOperation_t opB,
    aclsparseLtPruneAlg_t pruneAlg = ACLSPARSELT_PRUNE_SPMMA_STRIP,
    int32_t alpha_vector_scaling = 0,
    int32_t beta_vector_scaling = 0,
    const std::vector<float>* alphaVec = nullptr,
    const std::vector<float>* betaVec = nullptr,
    // Epilogue (backward-compatible defaults: all-off).
    int32_t biasEnabled = 0,
    int64_t biasStride = 0,
    int32_t activationType = 0,
    float reluUpperBound = FLT_MAX,
    float reluThreshold = 0.0f,
    float geluScaling = 1.0f,
    int32_t numBatches = 1,
    int64_t batchStride = 0,
    const std::vector<float>* biasVec = nullptr)
{
    MatmulRunArgs<InT, OutT> args{
        m, k, n, hA, hB, hC, hD,
        alpha, beta, algConfigId, splitK, splitKMode,
        order, opA, opB, pruneAlg,
        alpha_vector_scaling, beta_vector_scaling, alphaVec, betaVec
    };
    args.biasEnabled = biasEnabled;
    args.biasStride = biasStride;
    args.activationType = activationType;
    args.reluUpperBound = reluUpperBound;
    args.reluThreshold = reluThreshold;
    args.geluScaling = geluScaling;
    args.numBatches = numBatches;
    args.batchStride = batchStride;
    args.biasVec = biasVec;
    // bias dtype follows C dtype (FP16→FP16, BF16→BF16, INT8→FP32).
    args.biasEltSize = sparse_test::BiasDtypeTrait<InT>::kBiasEltSize;
    args.biasDtype = sparse_test::BiasDtypeTrait<InT>::kBiasAclDtype;
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

