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

/**
 * @file ltmatmul_test.cpp
 * @brief GTest + CSV-driven test for aclsparseLtMatmul (v2: 4 dtype × 2 path).
 *
 * Simplified test binary: the matmul path runs the full chain
 * (init → descriptor → setAttr → planInit(internal getAttr) → prune → matmul)
 * and explicitly asserts getAttr roundtrip, so standalone alg_set/alg_get
 * TEST_P suites have been removed.
 *
 * Test suites:
 *   - MatmulTest (TEST_P, CSV-driven): functional precision, 4 dtype × 2 path.
 *   - MatmulExceptionTest (TEST_F): matmul parameter validation.
 *   - AlgSetAttributeExceptionTest (TEST_F): setAttr parameter validation +
 *     alignment-32 end-to-end (no standalone alg_set CSV).
 */

#include "test_common.h"

#include "ltmatmul_golden.h"
#include "ltmatmul_npu_wrapper.h"
#include "ltmatmul_exception_helper.h"  // MatmulExceptionCtx, MatmulRawParams, MatmulNpuRaw
#include "ltmatmul_param.h"

#include <algorithm>
#include <chrono>
#include <random>
#include <type_traits>

using namespace sparse_test;

namespace {

// ============================================================================
// Test fixture
// ============================================================================
class MatmulEnvMixin {
protected:
    inline static std::unique_ptr<AclEnvScope> env_;
    aclrtStream stream_ = nullptr;
    static void InitEnv() {
        env_ = std::make_unique<AclEnvScope>();
    }
    static void FiniEnv() { env_.reset(); }
    void SetUpStream() { stream_ = env_->stream(); }
};

class MatmulTestFixture : public testing::Test, public MatmulEnvMixin {
public:
    static void SetUpTestSuite() { InitEnv(); }
    static void TearDownTestSuite() { FiniEnv(); }
protected:
    void SetUp() override { SetUpStream(); }
};

template <typename P>
class MatmulParamFixture : public testing::TestWithParam<P>, public MatmulEnvMixin {
public:
    static void SetUpTestSuite() { InitEnv(); }
    static void TearDownTestSuite() { FiniEnv(); }
protected:
    void SetUp() override { SetUpStream(); }
};

// ============================================================================
// Helpers
// ============================================================================
inline uint32_t SeedForCase(int32_t caseId) { return 2000u + static_cast<uint32_t>(caseId); }

inline aclsparseLtPruneAlg_t ToNpuPruneAlg(const MatmulParam& p) {
    return p.isTilePrune() ? ACLSPARSELT_PRUNE_SPMMA_TILE : ACLSPARSELT_PRUNE_SPMMA_STRIP;
}

inline aclsparseOrder_t ToNpuOrder(const MatmulParam& p) {
    return p.isColOrder() ? ACL_SPARSE_ORDER_COL : ACL_SPARSE_ORDER_ROW;
}

inline aclsparseOperation_t ToNpuOp(bool trans) {
    return trans ? ACL_SPARSE_OP_TRANSPOSE : ACL_SPARSE_OP_NON_TRANSPOSE;
}

// Golden prune direction: goldenIsColOrder = (trans == !isColOrder)
inline bool GoldenIsColOrder(bool trans, bool isColOrder) {
    return (trans == !isColOrder);
}

// ----------------------------------------------------------------------------
// AlphaBetaVecs: generated per-row alpha/beta vectors (shared by 3 Run* funcs).
// ----------------------------------------------------------------------------
struct AlphaBetaVecs {
    std::vector<float> alphaVec;
    std::vector<float> betaVec;
    const std::vector<float>* alphaPtr = nullptr;
    const std::vector<float>* betaPtr = nullptr;
};

inline AlphaBetaVecs GenAlphaBetaVectors(const MatmulParam& p, uint32_t seed) {
    AlphaBetaVecs v;
    if (p.alpha_vector_scaling) {
        v.alphaVec = GenScalingVector(p.m, seed + 10);
    } else {
        v.alphaVec.assign(static_cast<size_t>(p.m), p.alpha);
    }
    if (p.beta_vector_scaling) {
        v.betaVec = GenScalingVector(p.m, seed + 20);
    } else {
        v.betaVec.assign(static_cast<size_t>(p.m), p.beta);
    }
    v.alphaPtr = p.alpha_vector_scaling ? &v.alphaVec : nullptr;
    v.betaPtr = p.beta_vector_scaling ? &v.betaVec : nullptr;
    return v;
}

// ----------------------------------------------------------------------------
// RunAndVerifyNpu: run NPU matmul + ASSERT return codes (shared by 3 Run* funcs).
// Returns void; npu result is written to the output parameter. ASSERT_ macros
// are valid in void-returning functions [codecheck: reduce cyclomatic complexity].
// ----------------------------------------------------------------------------
template <typename InT, typename OutT>
inline void RunAndVerifyNpu(const MatmulParam& p,
    const std::vector<InT>& hA, const std::vector<InT>& hB, const std::vector<InT>& hC,
    std::vector<OutT>& hD, const AlphaBetaVecs& vecs, MatmulNpuResult& npu)
{
    npu = RunMatmulNpu<InT, OutT>(p.m, p.k, p.n, hA, hB, hC, hD,
        p.alpha, p.beta, p.alg_config_id, p.split_k, p.split_k_mode,
        p.isSparseA(), p.isDensePath(),
        ToNpuOrder(p), ToNpuOp(p.isTransA()), ToNpuOp(p.isTransB()),
        ToNpuPruneAlg(p),
        p.alpha_vector_scaling, p.beta_vector_scaling,
        vecs.alphaPtr, vecs.betaPtr);
    ASSERT_EQ(npu.algSetCfgRet, ACL_SPARSE_STATUS_SUCCESS) << p.caseId() << " algConfigId set failed";
    ASSERT_EQ(npu.algSetSplitRet, ACL_SPARSE_STATUS_SUCCESS) << p.caseId() << " splitK set failed";
    ASSERT_EQ(npu.algSetSplitKModeRet, ACL_SPARSE_STATUS_SUCCESS) << p.caseId() << " splitKMode set failed";
    // getAttr roundtrip verification: getAttr must succeed and read back the set values.
    ASSERT_EQ(npu.algGetCfgRet, ACL_SPARSE_STATUS_SUCCESS) << p.caseId() << " algConfigId get failed";
    ASSERT_EQ(npu.algGetSplitRet, ACL_SPARSE_STATUS_SUCCESS) << p.caseId() << " splitK get failed";
    ASSERT_EQ(npu.algGetSplitKModeRet, ACL_SPARSE_STATUS_SUCCESS) << p.caseId() << " splitKMode get failed";
    EXPECT_EQ(npu.gotAlgConfigId, p.alg_config_id) << p.caseId() << " getAttr config id mismatch";
    EXPECT_EQ(npu.gotSplitK, p.split_k) << p.caseId() << " getAttr split_k mismatch";
    EXPECT_EQ(npu.gotSplitKMode, p.split_k_mode) << p.caseId() << " getAttr split_k_mode mismatch";
    ASSERT_EQ(npu.descSetRet, ACL_SPARSE_STATUS_SUCCESS) << p.caseId() << " descSetAttribute failed";
    if (!p.isDensePath()) {
        ASSERT_EQ(npu.pruneRet, ACL_SPARSE_STATUS_SUCCESS) << p.caseId() << " prune failed";
    }
    ASSERT_EQ(npu.wsRet, ACL_SPARSE_STATUS_SUCCESS) << p.caseId() << " workspace failed";
    ASSERT_EQ(npu.matmulRet, ACL_SPARSE_STATUS_SUCCESS) << p.caseId() << " matmul failed";
}

// ----------------------------------------------------------------------------
// FillMatmulRawParams: fill the common MatmulRawParams fields for exception
// tests. Individual tests override specific fields after calling this.
// ----------------------------------------------------------------------------
inline void FillMatmulRawParams(MatmulRawParams& p,
    aclsparseLtConstHandle_t handle, aclsparseLtConstMatmulPlan_t plan,
    const void* alpha, const void* matA, const void* matB,
    const void* beta, const void* matC, void* matD,
    void* workspace, aclrtStream* streams, int32_t numStreams)
{
    p.handle = handle;
    p.plan = plan;
    p.alpha = alpha;
    p.matA = matA;
    p.matB = matB;
    p.beta = beta;
    p.matC = matC;
    p.matD = matD;
    p.workspace = workspace;
    p.streams = streams;
    p.numStreams = numStreams;
}

// Verify with mixed tolerance for float-convertible types.
inline void VerifyMixedTol(const std::vector<float>& goldenF,
                            const std::vector<float>& npuF,
                            aclDataType dtype, double atol, double rtol,
                            const std::string& caseId)
{
    ASSERT_EQ(goldenF.size(), npuF.size());
    VerifyConfig cfg;
    applyMixedTolerance(cfg, dtype, goldenF.data(), goldenF.size());
    cfg.SetMixedTol(atol, rtol);
    EXPECT_TRUE(Verifier::verifyVector(npuF, goldenF, cfg, caseId));
}

// Verify INT32 output with exact comparison.
inline void VerifyExactInt32(const std::vector<int32_t>& golden,
                              const std::vector<int32_t>& npu,
                              const std::string& caseId)
{
    ASSERT_EQ(golden.size(), npu.size()) << caseId << ": size mismatch";
    int32_t mismatches = 0;
    for (size_t i = 0; i < golden.size(); ++i) {
        if (npu[i] != golden[i]) {
            if (mismatches < 5) {
                std::cout << "[" << caseId << "] mismatch at " << i
                          << ": golden=" << golden[i] << " npu=" << npu[i] << "\n";
            }
            ++mismatches;
        }
    }
    EXPECT_EQ(mismatches, 0) << caseId << ": " << mismatches << "/" << golden.size() << " mismatches";
}

// ----------------------------------------------------------------------------
// ComputeFloatGoldenWithNpuPrune: golden computation using NPU's pruned matrix.
// Extracted from RunFloatCase to reduce NBNC.
// ----------------------------------------------------------------------------
template <typename T>
inline void ComputeFloatGoldenWithNpuPrune(const MatmulParam& p, const MatmulNpuResult& npu,
    const std::vector<T>& hA, const std::vector<T>& hB, const std::vector<T>& hC,
    const AlphaBetaVecs& vecs, std::vector<T>& hGoldenD)
{
    std::vector<T> npuPrunedT;
    if (p.isSparseA()) {
        npuPrunedT.resize(static_cast<size_t>(p.m) * p.k);
    } else {
        npuPrunedT.resize(static_cast<size_t>(p.k) * p.n);
    }
    const size_t byteCount = npuPrunedT.size() * sizeof(T);
    std::copy(npu.npuPrunedA.begin(), npu.npuPrunedA.begin() + byteCount,
              reinterpret_cast<int8_t*>(npuPrunedT.data()));
    std::vector<float> Af, Bf, Cf;
    if (p.isSparseA()) {
        PromoteInputsToFp32<T>(npuPrunedT, hB, hC, Af, Bf, Cf, p.m, p.k, p.n);
    } else {
        PromoteInputsToFp32<T>(hA, npuPrunedT, hC, Af, Bf, Cf, p.m, p.k, p.n);
    }
    std::vector<float> Df;
    if (p.alpha_vector_scaling || p.beta_vector_scaling) {
        MatmulAlphaBetaFp32Vec(Af, Bf, Cf, Df, p.m, p.k, p.n, vecs.alphaVec, vecs.betaVec);
    } else {
        MatmulAlphaBetaFp32(Af, Bf, Cf, Df, p.m, p.k, p.n, p.alpha, p.beta);
    }
    const size_t mn = static_cast<size_t>(p.m) * p.n;
    TruncateToOutput<T>(Df, hGoldenD, mn);
}

// ----------------------------------------------------------------------------
// ComputeFloatGoldenStandard: golden computation without NPU pruned matrix.
// Extracted from RunFloatCase to reduce NBNC.
// ----------------------------------------------------------------------------
template <typename T>
inline void ComputeFloatGoldenStandard(const MatmulParam& p,
    const std::vector<T>& hA, const std::vector<T>& hB, const std::vector<T>& hC,
    const AlphaBetaVecs& vecs, bool goldenColOrder, bool bGoldenColOrder,
    std::vector<T>& hGoldenD)
{
    if (p.alpha_vector_scaling || p.beta_vector_scaling) {
        if (p.isDensePath()) {
            LtMatmulDenseGoldenRunVec<T>(hA, hB, hC, hGoldenD, p.m, p.k, p.n,
                                          vecs.alphaVec, vecs.betaVec);
        } else if (p.isSparseA()) {
            LtMatmulGoldenRunWithPruneAlgVec<T>(hA, hB, hC, hGoldenD, p.m, p.k, p.n,
                                                 vecs.alphaVec, vecs.betaVec,
                                                 goldenColOrder, false, p.prune_alg);
        } else {
            LtMatmulGoldenRunWithPruneAlgVec<T>(hA, hB, hC, hGoldenD, p.m, p.k, p.n,
                                                 vecs.alphaVec, vecs.betaVec,
                                                 bGoldenColOrder, true, p.prune_alg);
        }
    } else {
        if (p.isDensePath()) {
            LtMatmulDenseGoldenRun<T>(hA, hB, hC, hGoldenD, p.m, p.k, p.n,
                                       p.alpha, p.beta, p.alg_config_id, p.split_k);
        } else if (p.isSparseA()) {
            LtMatmulGoldenRunWithPruneAlg<T>(hA, hB, hC, hGoldenD, p.m, p.k, p.n,
                                               p.alpha, p.beta, p.alg_config_id, p.split_k,
                                               goldenColOrder, false, p.prune_alg);
        } else {
            LtMatmulGoldenRunWithPruneAlg<T>(hA, hB, hC, hGoldenD, p.m, p.k, p.n,
                                               p.alpha, p.beta, p.alg_config_id, p.split_k,
                                               bGoldenColOrder, true, p.prune_alg);
        }
    }
}

// ----------------------------------------------------------------------------
// VerifyFloatOutput: verify FP32/FP16/BF16 output with mixed tolerance.
// Extracted from RunFloatCase to reduce NBNC.
// ----------------------------------------------------------------------------
template <typename T>
inline void VerifyFloatOutput(const std::vector<T>& hGoldenD, const std::vector<T>& hNpuD,
    const std::string& caseId)
{
    std::vector<float> goldenF = MatmulToFloat(hGoldenD);
    std::vector<float> npuF = MatmulToFloat(hNpuD);
    aclDataType aclDtype;
    double atol, rtol;
    if constexpr (std::is_same_v<T, float>) {
        aclDtype = ACL_FLOAT; atol = 1e-4; rtol = 1e-4;
    } else if constexpr (std::is_same_v<T, uint16_t>) {
        aclDtype = ACL_FLOAT16; atol = 1e-3; rtol = 1e-3;
    } else {
        aclDtype = ACL_BF16; atol = 1e-2; rtol = 1e-2;
    }
    VerifyMixedTol(goldenF, npuF, aclDtype, atol, rtol, caseId);
}

// ============================================================================
// FP32/FP16/BF16 dispatch: data gen + golden + NPU + verify
// ============================================================================
template <typename T>
void RunFloatCase(const MatmulParam& p, aclrtStream /*stream*/)
{
    const uint32_t seed = SeedForCase(p.case_id);
    auto hA = MatmulGenTestMatrix<T>(p.m, p.k, p.range_low, p.range_high, seed);
    auto hB = MatmulGenTestMatrix<T>(p.k, p.n, p.range_low, p.range_high, seed + 1);
    auto hC = MatmulGenTestMatrix<T>(p.m, p.n, p.range_low, p.range_high, seed + 2);

    auto vecs = GenAlphaBetaVectors(p, seed);

    std::vector<T> hNpuD;
    MatmulNpuResult npu;
    RunAndVerifyNpu<T, T>(p, hA, hB, hC, hNpuD, vecs, npu);

    std::vector<T> hGoldenD;
    const bool goldenColOrder = GoldenIsColOrder(p.isTransA(), p.isColOrder());
    const bool bGoldenColOrder = GoldenIsColOrder(p.isTransB(), p.isColOrder());
    const bool hasNpuPrune = !npu.npuPrunedA.empty();
    if (hasNpuPrune && (p.isSparseA() || (!p.isDensePath() && !p.isSparseA()))) {
        ComputeFloatGoldenWithNpuPrune<T>(p, npu, hA, hB, hC, vecs, hGoldenD);
    } else {
        ComputeFloatGoldenStandard<T>(p, hA, hB, hC, vecs, goldenColOrder, bGoldenColOrder, hGoldenD);
    }
    VerifyFloatOutput<T>(hGoldenD, hNpuD, p.caseId());
}

// ----------------------------------------------------------------------------
// [codecheck-cc] Sub-functions extracted from ComputeInt8Int32Golden to reduce
// cyclomatic complexity (21 → ≤ 8 for each sub-function).
// ----------------------------------------------------------------------------

// NPU-pruned A-sparse path: use NPU pruned A directly.
inline void ComputeInt8GoldenWithNpuPruneA(const MatmulParam& p, const MatmulNpuResult& npu,
    const std::vector<int8_t>& hB, const std::vector<int8_t>& hC,
    const AlphaBetaVecs& vecs, std::vector<int32_t>& hGoldenD)
{
    std::vector<int32_t> D_int32;
    if (p.alpha_vector_scaling || p.beta_vector_scaling) {
        MatmulAlphaBetaInt32Vec(npu.npuPrunedA, hB, hC, D_int32, p.m, p.k, p.n, vecs.alphaVec, vecs.betaVec);
    } else {
        MatmulAlphaBetaInt32(npu.npuPrunedA, hB, hC, D_int32, p.m, p.k, p.n, p.alpha, p.beta);
    }
    hGoldenD = D_int32;
}

// NPU-pruned B-sparse path: use NPU pruned B directly.
inline void ComputeInt8GoldenWithNpuPruneB(const MatmulParam& p, const MatmulNpuResult& npu,
    const std::vector<int8_t>& hA, const std::vector<int8_t>& hB, const std::vector<int8_t>& hC,
    const AlphaBetaVecs& vecs, std::vector<int32_t>& hGoldenD)
{
    std::vector<int8_t> B_pruned(npu.npuPrunedA.begin(), npu.npuPrunedA.end());
    std::vector<int32_t> D_int32;
    if (p.alpha_vector_scaling || p.beta_vector_scaling) {
        MatmulAlphaBetaInt32Vec(hA, B_pruned, hC, D_int32, p.m, p.k, p.n, vecs.alphaVec, vecs.betaVec);
    } else {
        MatmulAlphaBetaInt32(hA, B_pruned, hC, D_int32, p.m, p.k, p.n, p.alpha, p.beta);
    }
    hGoldenD = D_int32;
}

// Standard golden path (no NPU pruned data): dispatch by vec-scaling + path.
inline void ComputeInt8GoldenStandard(const MatmulParam& p,
    const std::vector<int8_t>& hA, const std::vector<int8_t>& hB, const std::vector<int8_t>& hC,
    const AlphaBetaVecs& vecs, bool goldenColOrder, bool bGoldenColOrder,
    std::vector<int32_t>& hGoldenD)
{
    if (p.alpha_vector_scaling || p.beta_vector_scaling) {
        if (p.isDensePath()) {
            LtMatmulDenseGoldenRunInt8Vec(hA, hB, hC, hGoldenD, p.m, p.k, p.n,
                                           vecs.alphaVec, vecs.betaVec);
        } else if (p.isSparseA()) {
            LtMatmulGoldenRunInt8Vec(hA, hB, hC, hGoldenD, p.m, p.k, p.n,
                                        vecs.alphaVec, vecs.betaVec, goldenColOrder, false, p.prune_alg);
        } else {
            LtMatmulGoldenRunInt8Vec(hA, hB, hC, hGoldenD, p.m, p.k, p.n,
                                        vecs.alphaVec, vecs.betaVec, bGoldenColOrder, true, p.prune_alg);
        }
    } else {
        if (p.isDensePath()) {
            LtMatmulDenseGoldenRunInt8(hA, hB, hC, hGoldenD, p.m, p.k, p.n, p.alpha, p.beta);
        } else if (p.isSparseA()) {
            LtMatmulGoldenRunInt8(hA, hB, hC, hGoldenD, p.m, p.k, p.n,
                                    p.alpha, p.beta, goldenColOrder, false, p.prune_alg);
        } else {
            LtMatmulGoldenRunInt8(hA, hB, hC, hGoldenD, p.m, p.k, p.n,
                                    p.alpha, p.beta, bGoldenColOrder, true, p.prune_alg);
        }
    }
}

// ----------------------------------------------------------------------------
// ComputeInt8Int32Golden: golden computation for INT8->INT32 path.
// Dispatches to sub-functions based on NPU prune availability + path.
// ----------------------------------------------------------------------------
inline void ComputeInt8Int32Golden(const MatmulParam& p, const MatmulNpuResult& npu,
    const std::vector<int8_t>& hA, const std::vector<int8_t>& hB, const std::vector<int8_t>& hC,
    const AlphaBetaVecs& vecs, bool goldenColOrder, bool bGoldenColOrder,
    std::vector<int32_t>& hGoldenD)
{
    if (p.isSparseA() && !npu.npuPrunedA.empty()) {
        ComputeInt8GoldenWithNpuPruneA(p, npu, hB, hC, vecs, hGoldenD);
    } else if (!p.isSparseA() && !p.isDensePath() && !npu.npuPrunedA.empty()) {
        ComputeInt8GoldenWithNpuPruneB(p, npu, hA, hB, hC, vecs, hGoldenD);
    } else {
        ComputeInt8GoldenStandard(p, hA, hB, hC, vecs, goldenColOrder, bGoldenColOrder, hGoldenD);
    }
}

// ----------------------------------------------------------------------------
// VerifyInt8Int32Output: verify INT8->INT32 output (exact or mixed tolerance).
// Extracted from RunInt8Int32Case to reduce NBNC.
// ----------------------------------------------------------------------------
inline void VerifyInt8Int32Output(const MatmulParam& p,
    const std::vector<int32_t>& hGoldenD, const std::vector<int32_t>& hNpuD)
{
    if (p.alpha == 1.0f && p.beta == 0.0f) {
        VerifyExactInt32(hGoldenD, hNpuD, p.caseId());
    } else {
        std::vector<float> goldenF = MatmulToFloat(hGoldenD);
        std::vector<float> npuF = MatmulToFloat(hNpuD);
        VerifyConfig cfg;
        applyMixedTolerance(cfg, ACL_FLOAT, goldenF.data(), goldenF.size());
        cfg.SetMixedTol(1.0, 1.0);
        cfg.SetMixedMaxAbsErrLim(65.0);
        EXPECT_TRUE(Verifier::verifyVector(npuF, goldenF, cfg, p.caseId()));
    }
}

// ============================================================================
// INT8 INT32 output dispatch
// ============================================================================
void RunInt8Int32Case(const MatmulParam& p, aclrtStream /*stream*/)
{
    const uint32_t seed = SeedForCase(p.case_id);
    auto hA = MatmulGenTestMatrix<int8_t>(p.m, p.k, p.range_low, p.range_high, seed);
    auto hB = MatmulGenTestMatrix<int8_t>(p.k, p.n, p.range_low, p.range_high, seed + 1);
    auto hC = MatmulGenTestMatrix<int8_t>(p.m, p.n, p.range_low, p.range_high, seed + 2);

    auto vecs = GenAlphaBetaVectors(p, seed);

    std::vector<int32_t> hNpuD;
    MatmulNpuResult npu;
    RunAndVerifyNpu<int8_t, int32_t>(p, hA, hB, hC, hNpuD, vecs, npu);

    std::vector<int32_t> hGoldenD;
    const bool goldenColOrder = GoldenIsColOrder(p.isTransA(), p.isColOrder());
    const bool bGoldenColOrder = GoldenIsColOrder(p.isTransB(), p.isColOrder());
    ComputeInt8Int32Golden(p, npu, hA, hB, hC, vecs, goldenColOrder, bGoldenColOrder, hGoldenD);
    VerifyInt8Int32Output(p, hGoldenD, hNpuD);
}

// ============================================================================
// INT8 INT8 output dispatch
// ============================================================================
void RunInt8Int8Case(const MatmulParam& p, aclrtStream /*stream*/)
{
    const uint32_t seed = SeedForCase(p.case_id);
    auto hA = MatmulGenTestMatrix<int8_t>(p.m, p.k, p.range_low, p.range_high, seed);
    auto hB = MatmulGenTestMatrix<int8_t>(p.k, p.n, p.range_low, p.range_high, seed + 1);
    auto hC = MatmulGenTestMatrix<int8_t>(p.m, p.n, p.range_low, p.range_high, seed + 2);

    auto vecs = GenAlphaBetaVectors(p, seed);

    std::vector<int8_t> hNpuD;
    MatmulNpuResult npu;
    RunAndVerifyNpu<int8_t, int8_t>(p, hA, hB, hC, hNpuD, vecs, npu);

    std::vector<int8_t> hGoldenD;
    if (p.alpha_vector_scaling || p.beta_vector_scaling) {
        if (p.isDensePath()) {
            LtMatmulDenseGoldenRunInt8ToInt8Vec(hA, hB, hC, hGoldenD, p.m, p.k, p.n,
                                                 vecs.alphaVec, vecs.betaVec);
        } else {
            LtMatmulGoldenRunInt8VecWithNpuPrune(hA, hB, hC, hGoldenD, p.m, p.k, p.n,
                                                  vecs.alphaVec, vecs.betaVec, npu.npuPrunedA,
                                                  p.isSparseA());
        }
    } else {
        if (p.isDensePath()) {
            LtMatmulDenseGoldenRunInt8ToInt8(hA, hB, hC, hGoldenD, p.m, p.k, p.n, p.alpha, p.beta);
        } else {
            LtMatmulGoldenRunInt8WithNpuPrune(hA, hB, hC, hGoldenD, p.m, p.k, p.n,
                                               p.alpha, p.beta, npu.npuPrunedA,
                                               p.isSparseA());
        }
    }

    std::vector<float> goldenF = MatmulToFloat(hGoldenD);
    std::vector<float> npuF = MatmulToFloat(hNpuD);
    VerifyMixedTol(goldenF, npuF, ACL_FLOAT, 1.0, 1.0, p.caseId());
}

// ============================================================================
// Dispatch by dtype × output_dtype
// ============================================================================
void RunOneCase(const MatmulParam& p, aclrtStream stream)
{
    std::cout << "\n==== " << p.caseId() << " ===="
              << " m=" << p.m << " k=" << p.k << " n=" << p.n
              << " dtype=" << p.dtype << " out=" << p.output_dtype
              << " path=" << p.matrix_type
              << " a=" << p.alpha << " b=" << p.beta
              << " tA=" << (p.isTransA() ? "T" : "N")
              << " tB=" << (p.isTransB() ? "T" : "N")
              << " splitk=" << p.split_k << " mode=" << p.split_k_mode
              << " prune=" << p.prune_alg
              << " side=" << (p.isSparseA() ? "A" : "B")
              << " alphaVec=" << p.alpha_vector_scaling
              << " betaVec=" << p.beta_vector_scaling
              << " range=[" << p.range_low << "," << p.range_high << "]\n";

    if (p.isFp32()) {
        RunFloatCase<float>(p, stream);
    } else if (p.isFp16()) {
        RunFloatCase<uint16_t>(p, stream);
    } else if (p.isBf16()) {
        RunFloatCase<bf16_bits_t>(p, stream);
    } else if (p.isInt8() && p.isInt32Output()) {
        RunInt8Int32Case(p, stream);
    } else if (p.isInt8() && p.isInt8Output()) {
        RunInt8Int8Case(p, stream);
    } else {
        FAIL() << "Unknown dtype/output_dtype combination for " << p.caseId();
    }
}

}  // namespace

// ============================================================================
// CSV parameterized test
// ============================================================================
class MatmulTest : public MatmulParamFixture<MatmulParam> {
};

TEST_P(MatmulTest, FunctionalPrecision) {
    RunOneCase(GetParam(), stream_);
}

INSTANTIATE_TEST_SUITE_P(
    MatmulCases,
    MatmulTest,
    testing::ValuesIn(GetCasesFromCsv<MatmulParam>("ltmatmul_test.csv")),
    [](const testing::TestParamInfo<MatmulParam>& info) {
        return info.param.caseId();
    }
);

// ============================================================================
// L2 Exception tests (TEST_F)
// ============================================================================

class MatmulExceptionTest : public MatmulTestFixture {
};

// M-L2-01: null plan
TEST_F(MatmulExceptionTest, L2_01_NullPlan) {
    MatmulExceptionCtx ctx;
    MatmulRawParams p;
    float alpha = 1.0f, beta = 0.0f;
    aclrtStream stream = ctx.stream;
    FillMatmulRawParams(p, ctx.handle.get(), nullptr, &alpha,
        ctx.dA.get(), ctx.dB.get(), &beta, ctx.dC.get(), ctx.dD.get(),
        ctx.dWorkspace.get(), &stream, 1);
    auto ret = MatmulNpuRaw(p);
    EXPECT_EQ(ret, ACL_SPARSE_STATUS_INVALID_VALUE);
}

// M-L2-02: destroyed plan
// PlanDestroy now sets *plan = nullptr → delete → null
// (unified with all other Destroy functions). After destroy, plan is null, so
// the null check in validate_matmul_params fires first and returns
// INVALID_VALUE (a null plan is an invalid value — correct C API semantics).
TEST_F(MatmulExceptionTest, L2_02_DestroyedPlan) {
    // Build a plan, destroy it, then call matmul.
    SparseLtHandleGuard handle;
    SparseLtMatDescGuard matA(handle.get(), 8, 8, 8, 16, ACL_FLOAT,
                              ACL_SPARSE_ORDER_ROW, ACL_SPARSE_LT_SPARSITY_50_PERCENT);
    SparseLtDnMatDescGuard matB(handle.get(), 8, 8, 8, 16, ACL_FLOAT, ACL_SPARSE_ORDER_ROW);
    SparseLtDnMatDescGuard matC(handle.get(), 8, 8, 8, 16, ACL_FLOAT, ACL_SPARSE_ORDER_ROW);
    SparseLtDnMatDescGuard matD(handle.get(), 8, 8, 8, 16, ACL_FLOAT, ACL_SPARSE_ORDER_ROW);
    SparseLtMatmulDescGuard md(handle.get(), ACL_SPARSE_OP_NON_TRANSPOSE,
        ACL_SPARSE_OP_NON_TRANSPOSE, matA.get(), matB.get(), matC.get(), matD.get(),
        ACL_SPARSE_COMPUTE_32F);
    SparseLtAlgSelectionGuard algSel(handle.get(), md.get(), ACL_SPARSE_LT_MATMUL_ALG_DEFAULT);

    aclsparseLtMatmulPlan_t plan = nullptr;
    ASSERT_EQ(aclsparseLtMatmulPlanInit(handle.get(), &plan, md.ptr(), algSel.ptr()),
              ACL_SPARSE_STATUS_SUCCESS);
    ASSERT_EQ(aclsparseLtMatmulPlanDestroy(&plan), ACL_SPARSE_STATUS_SUCCESS);

    MatmulRawParams p;
    float alpha = 1.0f, beta = 0.0f;
    aclrtStream stream = stream_;
    FillMatmulRawParams(p, handle.get(), plan, &alpha,
        matA.get(), matB.get(), &beta, matC.get(), matD.get(),
        nullptr, &stream, 1);
    auto ret = MatmulNpuRaw(p);
    EXPECT_EQ(ret, ACL_SPARSE_STATUS_INVALID_VALUE);
}

// M-L2-03: null matB
TEST_F(MatmulExceptionTest, L2_03_NullMatB) {
    MatmulExceptionCtx ctx;
    MatmulRawParams p;
    float alpha = 1.0f, beta = 0.0f;
    aclrtStream stream = ctx.stream;
    FillMatmulRawParams(p, ctx.handle.get(), ctx.plan.get(), &alpha,
        ctx.dA.get(), nullptr, &beta, ctx.dC.get(), ctx.dD.get(),
        ctx.dWorkspace.get(), &stream, 1);
    auto ret = MatmulNpuRaw(p);
    EXPECT_EQ(ret, ACL_SPARSE_STATUS_INVALID_VALUE);
}

// M-L2-04: null matC
TEST_F(MatmulExceptionTest, L2_04_NullMatC) {
    MatmulExceptionCtx ctx;
    MatmulRawParams p;
    float alpha = 1.0f, beta = 0.0f;
    aclrtStream stream = ctx.stream;
    FillMatmulRawParams(p, ctx.handle.get(), ctx.plan.get(), &alpha,
        ctx.dA.get(), ctx.dB.get(), &beta, nullptr, ctx.dD.get(),
        ctx.dWorkspace.get(), &stream, 1);
    auto ret = MatmulNpuRaw(p);
    EXPECT_EQ(ret, ACL_SPARSE_STATUS_INVALID_VALUE);
}

// M-L2-05: null matD
TEST_F(MatmulExceptionTest, L2_05_NullMatD) {
    MatmulExceptionCtx ctx;
    MatmulRawParams p;
    float alpha = 1.0f, beta = 0.0f;
    aclrtStream stream = ctx.stream;
    FillMatmulRawParams(p, ctx.handle.get(), ctx.plan.get(), &alpha,
        ctx.dA.get(), ctx.dB.get(), &beta, ctx.dC.get(), nullptr,
        ctx.dWorkspace.get(), &stream, 1);
    auto ret = MatmulNpuRaw(p);
    EXPECT_EQ(ret, ACL_SPARSE_STATUS_INVALID_VALUE);
}

// M-L2-06: null workspace (workspaceSize > 0)
TEST_F(MatmulExceptionTest, L2_06_NullWorkspace) {
    MatmulExceptionCtx ctx;
    if (ctx.workspaceSize == 0) { GTEST_SKIP() << "workspace size is 0, skipping"; }
    MatmulRawParams p;
    float alpha = 1.0f, beta = 0.0f;
    aclrtStream stream = ctx.stream;
    FillMatmulRawParams(p, ctx.handle.get(), ctx.plan.get(), &alpha,
        ctx.dA.get(), ctx.dB.get(), &beta, ctx.dC.get(), ctx.dD.get(),
        nullptr, &stream, 1);
    auto ret = MatmulNpuRaw(p);
    EXPECT_EQ(ret, ACL_SPARSE_STATUS_INSUFFICIENT_RESOURCES);
}

// M-L2-07: null streams
TEST_F(MatmulExceptionTest, L2_07_NullStreams) {
    MatmulExceptionCtx ctx;
    MatmulRawParams p;
    float alpha = 1.0f, beta = 0.0f;
    FillMatmulRawParams(p, ctx.handle.get(), ctx.plan.get(), &alpha,
        ctx.dA.get(), ctx.dB.get(), &beta, ctx.dC.get(), ctx.dD.get(),
        ctx.dWorkspace.get(), nullptr, 1);
    auto ret = MatmulNpuRaw(p);
    EXPECT_EQ(ret, ACL_SPARSE_STATUS_INVALID_VALUE);
}

// M-L2-08: numStreams=0
TEST_F(MatmulExceptionTest, L2_08_NumStreamsZero) {
    MatmulExceptionCtx ctx;
    MatmulRawParams p;
    float alpha = 1.0f, beta = 0.0f;
    aclrtStream stream = ctx.stream;
    FillMatmulRawParams(p, ctx.handle.get(), ctx.plan.get(), &alpha,
        ctx.dA.get(), ctx.dB.get(), &beta, ctx.dC.get(), ctx.dD.get(),
        ctx.dWorkspace.get(), &stream, 0);
    auto ret = MatmulNpuRaw(p);
    EXPECT_EQ(ret, ACL_SPARSE_STATUS_INVALID_VALUE);
}

// M-L2-09: numStreams=-1
TEST_F(MatmulExceptionTest, L2_09_NumStreamsNegative) {
    MatmulExceptionCtx ctx;
    MatmulRawParams p;
    float alpha = 1.0f, beta = 0.0f;
    aclrtStream stream = ctx.stream;
    FillMatmulRawParams(p, ctx.handle.get(), ctx.plan.get(), &alpha,
        ctx.dA.get(), ctx.dB.get(), &beta, ctx.dC.get(), ctx.dD.get(),
        ctx.dWorkspace.get(), &stream, -1);
    auto ret = MatmulNpuRaw(p);
    EXPECT_EQ(ret, ACL_SPARSE_STATUS_INVALID_VALUE);
}

// M-L2-10: matB non-16-byte aligned
TEST_F(MatmulExceptionTest, L2_10_MatBNonAligned) {
    MatmulExceptionCtx ctx;
    // Allocate a buffer and offset by 1 byte to break 16-byte alignment.
    auto dBad = sparse_test::DeviceBuffer::alloc(8 * 8 * sizeof(float) + 1);
    void* badPtr = static_cast<uint8_t*>(dBad.get()) + 1;
    MatmulRawParams p;
    float alpha = 1.0f, beta = 0.0f;
    aclrtStream stream = ctx.stream;
    FillMatmulRawParams(p, ctx.handle.get(), ctx.plan.get(), &alpha,
        ctx.dA.get(), badPtr, &beta, ctx.dC.get(), ctx.dD.get(),
        ctx.dWorkspace.get(), &stream, 1);
    auto ret = MatmulNpuRaw(p);
    EXPECT_NE(ret, ACL_SPARSE_STATUS_SUCCESS);
}

// M-L2-21: alpha=null (fallback to 1.0)
TEST_F(MatmulExceptionTest, L2_21_AlphaNullFallback) {
    // Build a valid chain with alpha=null, expect SUCCESS.
    int32_t m = 128, k = 128, n = 128;
    std::vector<float> hA(m * k, 1.0f), hB(k * n, 1.0f), hC(m * n, 0.0f), hD;
    // Use the existing alg_set_attribute wrapper for this case.
    auto npu = AlgSetAttributeNpu<float>(stream_, hA, hB, hC, hD,
                                          m, k, n, 1.0f, 0.0f, 0, 1);
    // This test just verifies the chain works; alpha=null is tested via raw API.
    // For the raw API test:
    MatmulExceptionCtx ctx;
    MatmulRawParams p;
    float beta = 0.0f;
    aclrtStream stream = ctx.stream;
    FillMatmulRawParams(p, ctx.handle.get(), ctx.plan.get(), nullptr,
        ctx.dA.get(), ctx.dB.get(), &beta, ctx.dC.get(), ctx.dD.get(),
        ctx.dWorkspace.get(), &stream, 1);
    auto ret = MatmulNpuRaw(p);
    // alpha=null should fallback to 1.0 and succeed (or at least not fail due to alpha).
    EXPECT_NE(ret, ACL_SPARSE_STATUS_INVALID_VALUE);
}

// M-L2-22: beta=null (fallback to 0.0)
TEST_F(MatmulExceptionTest, L2_22_BetaNullFallback) {
    MatmulExceptionCtx ctx;
    MatmulRawParams p;
    float alpha = 1.0f;
    aclrtStream stream = ctx.stream;
    FillMatmulRawParams(p, ctx.handle.get(), ctx.plan.get(), &alpha,
        ctx.dA.get(), ctx.dB.get(), nullptr, ctx.dC.get(), ctx.dD.get(),
        ctx.dWorkspace.get(), &stream, 1);
    auto ret = MatmulNpuRaw(p);
    EXPECT_NE(ret, ACL_SPARSE_STATUS_INVALID_VALUE);
}

// M-L2-23: matA=null A-sparse → workspace fallback (should not return INVALID_VALUE)
// Note: the workspace[aPrunedOffset] region is uninitialized (no SpMMAPrune called),
// so the kernel result is undefined. This test only verifies the host-side API
// behavior: A-sparse path with matA=null and non-null workspace should fall back
// to workspace[aPrunedOffset], not return INVALID_VALUE.
TEST_F(MatmulExceptionTest, L2_23_MatANullASparseWorkspaceFallback) {
    MatmulExceptionCtx ctx;  // A-sparse by default (SPARSITY_50_PERCENT)
    MatmulRawParams p;
    float alpha = 1.0f, beta = 0.0f;
    aclrtStream stream = ctx.stream;
    FillMatmulRawParams(p, ctx.handle.get(), ctx.plan.get(), &alpha,
        nullptr, ctx.dB.get(), &beta, ctx.dC.get(), ctx.dD.get(),
        ctx.dWorkspace.get(), &stream, 1);  // non-null workspace
    auto ret = MatmulNpuRaw(p);
    // A-sparse with workspace should NOT return INVALID_VALUE for null matA.
    // (It may return SUCCESS or EXECUTION_FAILED depending on uninitialized data,
    //  but the host validation should pass.)
    EXPECT_NE(ret, ACL_SPARSE_STATUS_INVALID_VALUE);
}

// M-L2-37: DescriptorInit structured+structured → INVALID_VALUE
TEST_F(MatmulExceptionTest, L2_37_StructuredPlusStructured) {
    SparseLtHandleGuard handle;
    SparseLtMatDescGuard matA(handle.get(), 8, 8, 8, 16, ACL_FLOAT,
                              ACL_SPARSE_ORDER_ROW, ACL_SPARSE_LT_SPARSITY_50_PERCENT);
    SparseLtMatDescGuard matB(handle.get(), 8, 8, 8, 16, ACL_FLOAT,
                              ACL_SPARSE_ORDER_ROW, ACL_SPARSE_LT_SPARSITY_50_PERCENT);
    SparseLtDnMatDescGuard matC(handle.get(), 8, 8, 8, 16, ACL_FLOAT, ACL_SPARSE_ORDER_ROW);
    SparseLtDnMatDescGuard matD(handle.get(), 8, 8, 8, 16, ACL_FLOAT, ACL_SPARSE_ORDER_ROW);
    aclsparseLtMatmulDescriptor_t md = nullptr;
    auto ret = aclsparseLtMatmulDescriptorInit(
        handle.get(), &md, ACL_SPARSE_OP_NON_TRANSPOSE, ACL_SPARSE_OP_NON_TRANSPOSE,
        &matA.get(), &matB.get(), &matC.get(), &matD.get(), ACL_SPARSE_COMPUTE_32F);
    EXPECT_EQ(ret, ACL_SPARSE_STATUS_INVALID_VALUE);
    if (md) { aclsparseLtMatmulDescriptorDestroy(&md); }
}

// M-L2-40: unsupported dtype (ACL_FP8 or other) → NOT_SUPPORTED
TEST_F(MatmulExceptionTest, L2_40_UnsupportedDtype) {
    SparseLtHandleGuard handle;
    aclsparseLtMatDescriptor_t matA = nullptr;
    // Use a fake/unsupported dtype value. ACL_FP8_E4M3FN = 28 (if defined) or use a large value.
    aclDataType fakeDtype = static_cast<aclDataType>(99);
    auto ret = aclsparseLtStructuredDescriptorInit(
        handle.get(), &matA, 8, 8, 8, 16, fakeDtype,
        ACL_SPARSE_ORDER_ROW, ACL_SPARSE_LT_SPARSITY_50_PERCENT);
    EXPECT_EQ(ret, ACL_SPARSE_STATUS_NOT_SUPPORTED);
    if (matA) { aclsparseLtMatDescriptorDestroy(&matA); }
}

// M-L2-41: computeType mismatch (32F + INT8) → INVALID_VALUE
TEST_F(MatmulExceptionTest, L2_41_ComputeTypeMismatch) {
    SparseLtHandleGuard handle;
    // INT8 matrix with 32F compute type → mismatch.
    SparseLtMatDescGuard matA(handle.get(), 32, 32, 32, 16, ACL_INT8,
                              ACL_SPARSE_ORDER_ROW, ACL_SPARSE_LT_SPARSITY_50_PERCENT);
    SparseLtDnMatDescGuard matB(handle.get(), 32, 32, 32, 16, ACL_INT8, ACL_SPARSE_ORDER_ROW);
    SparseLtDnMatDescGuard matC(handle.get(), 32, 32, 32, 16, ACL_INT8, ACL_SPARSE_ORDER_ROW);
    SparseLtDnMatDescGuard matD(handle.get(), 32, 32, 32, 16, ACL_INT8, ACL_SPARSE_ORDER_ROW);
    aclsparseLtMatmulDescriptor_t md = nullptr;
    auto ret = aclsparseLtMatmulDescriptorInit(
        handle.get(), &md, ACL_SPARSE_OP_NON_TRANSPOSE, ACL_SPARSE_OP_NON_TRANSPOSE,
        &matA.get(), &matB.get(), &matC.get(), &matD.get(), ACL_SPARSE_COMPUTE_32F);
    EXPECT_NE(ret, ACL_SPARSE_STATUS_SUCCESS);
    if (md) { aclsparseLtMatmulDescriptorDestroy(&md); }
}

// ============================================================================
// Whitebox supplement (3.3): branches not covered by 3.2 CSV/TEST_F cases.
// ============================================================================

// WB-L2-42: null handle → HANDLE_IS_NULLPTR
// Covers validate_matmul_params branch: `handle == nullptr || *handle == nullptr`.
// (3.2 L2_01 covered null plan, but null handle was not directly exercised.)
TEST_F(MatmulExceptionTest, WB_L2_42_NullHandle) {
    MatmulExceptionCtx ctx;
    MatmulRawParams p;
    float alpha = 1.0f, beta = 0.0f;
    aclrtStream stream = ctx.stream;
    FillMatmulRawParams(p, nullptr, ctx.plan.get(), &alpha,
        ctx.dA.get(), ctx.dB.get(), &beta, ctx.dC.get(), ctx.dD.get(),
        ctx.dWorkspace.get(), &stream, 1);
    auto ret = MatmulNpuRaw(p);
    EXPECT_EQ(ret, ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR);
}

// WB-L2-43: streams[0]==null → SUCCESS (nullptr is valid default stream)
// streams[0] may be nullptr — the ACL runtime treats 0/nullptr as
// the default stream, which is valid for all aclrtMemcpy / kernel-launch APIs.
// The previous test expected INVALID_VALUE, but the implementation correctly
// accepts nullptr as the default stream and proceeds with kernel launch.
TEST_F(MatmulExceptionTest, WB_L2_43_Streams0Null) {
    MatmulExceptionCtx ctx;
    MatmulRawParams p;
    float alpha = 1.0f, beta = 0.0f;
    aclrtStream nullStream = nullptr;  // streams[0] == null (default stream)
    FillMatmulRawParams(p, ctx.handle.get(), ctx.plan.get(), &alpha,
        ctx.dA.get(), ctx.dB.get(), &beta, ctx.dC.get(), ctx.dD.get(),
        ctx.dWorkspace.get(), &nullStream, 1);  // numStreams > 0, streams[0] is null (valid default)
    auto ret = MatmulNpuRaw(p);
    EXPECT_EQ(ret, ACL_SPARSE_STATUS_SUCCESS);
}

// WB-L2-44: matA=null B-sparse → INVALID_VALUE
// Covers prepare_matmul_pointers branch: `!md->isSparseA` (B-sparse) when matA
// is null. 3.2 L2_23 covered A-sparse + matA=null (workspace fallback),
// L2_36 (dense×dense) not supported: upstream MatmulDescriptorInit requires
// exactly one structured descriptor (both-dense returns INVALID_VALUE).
// The B-sparse + matA=null branch is covered here by WB_L2_44.
TEST_F(MatmulExceptionTest, WB_L2_44_MatANullBSparse) {
    // Build a B-sparse plan: matA dense, matB structured (2:4 sparse).
    SparseLtHandleGuard handle;
    SparseLtDnMatDescGuard matA(handle.get(), 8, 8, 8, 16, ACL_FLOAT, ACL_SPARSE_ORDER_ROW);
    SparseLtMatDescGuard matB(handle.get(), 8, 8, 8, 16, ACL_FLOAT,
                              ACL_SPARSE_ORDER_ROW, ACL_SPARSE_LT_SPARSITY_50_PERCENT);
    SparseLtDnMatDescGuard matC(handle.get(), 8, 8, 8, 16, ACL_FLOAT, ACL_SPARSE_ORDER_ROW);
    SparseLtDnMatDescGuard matD(handle.get(), 8, 8, 8, 16, ACL_FLOAT, ACL_SPARSE_ORDER_ROW);
    SparseLtMatmulDescGuard md(handle.get(), ACL_SPARSE_OP_NON_TRANSPOSE,
        ACL_SPARSE_OP_NON_TRANSPOSE, matA.get(), matB.get(), matC.get(), matD.get(),
        ACL_SPARSE_COMPUTE_32F);
    SparseLtAlgSelectionGuard algSel(handle.get(), md.get(), ACL_SPARSE_LT_MATMUL_ALG_DEFAULT);
    SparseLtPlanGuard plan(handle.get(), md.get(), algSel.get());

    size_t wsSize = 0;
    ASSERT_EQ(aclsparseLtMatmulGetWorkspace(handle.get(), plan.cptr(), &wsSize),
              ACL_SPARSE_STATUS_SUCCESS);
    auto dWorkspace = sparse_test::DeviceBuffer::alloc(wsSize);
    auto dA = sparse_test::DeviceBuffer::alloc(8 * 8 * sizeof(float));
    auto dB = sparse_test::DeviceBuffer::alloc(8 * 8 * sizeof(float));
    auto dC = sparse_test::DeviceBuffer::alloc(8 * 8 * sizeof(float));
    auto dD = sparse_test::DeviceBuffer::alloc(8 * 8 * sizeof(float));

    MatmulRawParams p;
    float alpha = 1.0f, beta = 0.0f;
    aclrtStream stream = 0;
    FillMatmulRawParams(p, handle.get(), plan.get(), &alpha,
        nullptr, dB.get(), &beta, dC.get(), dD.get(),
        dWorkspace.get(), &stream, 1);  // B-sparse requires explicit dense A → INVALID_VALUE
    auto ret = MatmulNpuRaw(p);
    EXPECT_EQ(ret, ACL_SPARSE_STATUS_INVALID_VALUE);
}

// ============================================================================
// alg_set_attribute exception-path tests
// (standalone alg_set CSV TEST_P removed — matmul path covers setAttr roundtrip)
// ============================================================================

namespace {

class AlgAclEnvScopeMixin {
protected:
    inline static std::unique_ptr<AclEnvScope> env_;
    aclrtStream stream_ = nullptr;
    static void InitEnv() { env_ = std::make_unique<AclEnvScope>(); }
    static void FiniEnv() { env_.reset(); }
    void SetUpStream() { stream_ = env_->stream(); }
};

class AlgTestFixture : public testing::Test, public AlgAclEnvScopeMixin {
public:
    static void SetUpTestSuite() { InitEnv(); }
    static void TearDownTestSuite() { FiniEnv(); }
protected:
    void SetUp() override { SetUpStream(); }
};

inline void VerifyAlgMixedTolerance(const std::vector<float>& goldenF,
                                     const std::vector<float>& npuF,
                                     bool isFp16, const std::string& caseId)
{
    ASSERT_EQ(goldenF.size(), npuF.size());
    const aclDataType dtype = isFp16 ? ACL_FLOAT16 : ACL_FLOAT;
    VerifyConfig cfg;
    applyMixedTolerance(cfg, dtype, goldenF.data(), goldenF.size());
    EXPECT_TRUE(Verifier::verifyVector(npuF, goldenF, cfg, caseId));
}

}  // namespace

// =============================================================================
// alg_set_attribute exception-path tests
// =============================================================================

struct AlgTestCtx {
    SparseLtHandleGuard handle;
    SparseLtMatDescGuard matA;
    SparseLtDnMatDescGuard matB;
    SparseLtDnMatDescGuard matC;
    SparseLtDnMatDescGuard matD;
    SparseLtMatmulDescGuard md;
    SparseLtAlgSelectionGuard alg;

    AlgTestCtx()
        : handle(),
          matA(handle.get(), 8, 8, 8, 16, ACL_FLOAT,
               ACL_SPARSE_ORDER_ROW, ACL_SPARSE_LT_SPARSITY_50_PERCENT),
          matB(handle.get(), 8, 8, 8, 16, ACL_FLOAT,
               ACL_SPARSE_ORDER_ROW),
          matC(handle.get(), 8, 8, 8, 16, ACL_FLOAT,
               ACL_SPARSE_ORDER_ROW),
          matD(handle.get(), 8, 8, 8, 16, ACL_FLOAT,
               ACL_SPARSE_ORDER_ROW),
          md(handle.get(), ACL_SPARSE_OP_NON_TRANSPOSE, ACL_SPARSE_OP_NON_TRANSPOSE,
             matA.get(), matB.get(), matC.get(), matD.get(), ACL_SPARSE_COMPUTE_32F),
          alg(handle.get(), md.get(), ACL_SPARSE_LT_MATMUL_ALG_DEFAULT) {}
};

class AlgSetAttributeExceptionTest : public AlgTestFixture {
};

TEST_F(AlgSetAttributeExceptionTest, NullHandle) {
    int32_t val = 0;
    auto ret = aclsparseLtMatmulAlgSetAttribute(
        nullptr, nullptr, ACLSPARSELT_MATMUL_ALG_CONFIG_ID, &val, sizeof(int32_t));
    EXPECT_NE(ret, ACL_SPARSE_STATUS_SUCCESS);
}

TEST_F(AlgSetAttributeExceptionTest, InvalidAlgConfigId) {
    AlgTestCtx ctx;
    int32_t badId = 2;
    auto ret = aclsparseLtMatmulAlgSetAttribute(
        ctx.handle.get(), &ctx.alg.get(), ACLSPARSELT_MATMUL_ALG_CONFIG_ID, &badId, sizeof(int32_t));
    EXPECT_NE(ret, ACL_SPARSE_STATUS_SUCCESS);
}

TEST_F(AlgSetAttributeExceptionTest, InvalidSplitK) {
    AlgTestCtx ctx;
    int32_t badSplitK = 0;
    auto ret = aclsparseLtMatmulAlgSetAttribute(
        ctx.handle.get(), &ctx.alg.get(), ACLSPARSELT_MATMUL_SPLIT_K, &badSplitK, sizeof(int32_t));
    EXPECT_NE(ret, ACL_SPARSE_STATUS_SUCCESS);
}

TEST_F(AlgSetAttributeExceptionTest, InvalidSplitKAboveK) {
    AlgTestCtx ctx;
    int32_t badSplitK = 9;
    auto ret = aclsparseLtMatmulAlgSetAttribute(
        ctx.handle.get(), &ctx.alg.get(), ACLSPARSELT_MATMUL_SPLIT_K, &badSplitK, sizeof(int32_t));
    EXPECT_NE(ret, ACL_SPARSE_STATUS_SUCCESS);
}

TEST_F(AlgSetAttributeExceptionTest, InvalidSearchIterationsZero) {
    AlgTestCtx ctx;
    int32_t badIter = 0;
    auto ret = aclsparseLtMatmulAlgSetAttribute(
        ctx.handle.get(), &ctx.alg.get(), ACLSPARSELT_MATMUL_SEARCH_ITERATIONS, &badIter, sizeof(int32_t));
    EXPECT_NE(ret, ACL_SPARSE_STATUS_SUCCESS);
}

TEST_F(AlgSetAttributeExceptionTest, InvalidSearchIterationsNegative) {
    AlgTestCtx ctx;
    int32_t badIter = -1;
    auto ret = aclsparseLtMatmulAlgSetAttribute(
        ctx.handle.get(), &ctx.alg.get(), ACLSPARSELT_MATMUL_SEARCH_ITERATIONS, &badIter, sizeof(int32_t));
    EXPECT_NE(ret, ACL_SPARSE_STATUS_SUCCESS);
}

TEST_F(AlgSetAttributeExceptionTest, DataSizeMismatch) {
    AlgTestCtx ctx;
    int32_t val = 0;
    auto ret = aclsparseLtMatmulAlgSetAttribute(
        ctx.handle.get(), &ctx.alg.get(), ACLSPARSELT_MATMUL_ALG_CONFIG_ID, &val, sizeof(int16_t));
    EXPECT_NE(ret, ACL_SPARSE_STATUS_SUCCESS);
}

TEST_F(AlgSetAttributeExceptionTest, NullAlgSelection) {
    SparseLtHandleGuard handle;
    int32_t val = 0;
    auto ret = aclsparseLtMatmulAlgSetAttribute(
        handle.get(), nullptr, ACLSPARSELT_MATMUL_ALG_CONFIG_ID, &val, sizeof(int32_t));
    EXPECT_NE(ret, ACL_SPARSE_STATUS_SUCCESS);
}

TEST_F(AlgSetAttributeExceptionTest, ConfigMaxIdReadOnly) {
    AlgTestCtx ctx;
    int32_t val = 0;
    auto ret = aclsparseLtMatmulAlgSetAttribute(
        ctx.handle.get(), &ctx.alg.get(), ACLSPARSELT_MATMUL_ALG_CONFIG_MAX_ID, &val, sizeof(int32_t));
    EXPECT_EQ(ret, ACL_SPARSE_STATUS_NOT_SUPPORTED);
}

TEST_F(AlgSetAttributeExceptionTest, InvalidAlignmentZero) {
    SparseLtHandleGuard handle;
    aclsparseLtMatDescriptor_t matA = nullptr;
    auto ret = aclsparseLtStructuredDescriptorInit(
        handle.get(), &matA, 8, 8, 8, 0, ACL_FLOAT,
        ACL_SPARSE_ORDER_ROW, ACL_SPARSE_LT_SPARSITY_50_PERCENT);
    EXPECT_NE(ret, ACL_SPARSE_STATUS_SUCCESS);
    if (matA) { aclsparseLtMatDescriptorDestroy(&matA); }
}

TEST_F(AlgSetAttributeExceptionTest, InvalidAlignmentNotMultipleOf16) {
    SparseLtHandleGuard handle;
    aclsparseLtMatDescriptor_t matA = nullptr;
    auto ret = aclsparseLtStructuredDescriptorInit(
        handle.get(), &matA, 8, 8, 8, 8, ACL_FLOAT,
        ACL_SPARSE_ORDER_ROW, ACL_SPARSE_LT_SPARSITY_50_PERCENT);
    EXPECT_NE(ret, ACL_SPARSE_STATUS_SUCCESS);
    if (matA) { aclsparseLtMatDescriptorDestroy(&matA); }
}

TEST_F(AlgSetAttributeExceptionTest, InvalidAlignmentNotMultipleOf16b) {
    SparseLtHandleGuard handle;
    aclsparseLtMatDescriptor_t matA = nullptr;
    auto ret = aclsparseLtStructuredDescriptorInit(
        handle.get(), &matA, 8, 8, 8, 17, ACL_FLOAT,
        ACL_SPARSE_ORDER_ROW, ACL_SPARSE_LT_SPARSITY_50_PERCENT);
    EXPECT_NE(ret, ACL_SPARSE_STATUS_SUCCESS);
    if (matA) { aclsparseLtMatDescriptorDestroy(&matA); }
}

// v2 extended StructuredDescriptorInit to accept INT8 (and BF16),
// so ACL_INT8 no longer returns NOT_SUPPORTED. This case now uses ACL_DOUBLE,
// which is genuinely outside the supported set (FP32/FP16/BF16/INT8).
TEST_F(AlgSetAttributeExceptionTest, UnsupportedValueType) {
    SparseLtHandleGuard handle;
    aclsparseLtMatDescriptor_t matA = nullptr;
    auto ret = aclsparseLtStructuredDescriptorInit(
        handle.get(), &matA, 8, 8, 8, 16, ACL_DOUBLE,
        ACL_SPARSE_ORDER_ROW, ACL_SPARSE_LT_SPARSITY_50_PERCENT);
    EXPECT_EQ(ret, ACL_SPARSE_STATUS_NOT_SUPPORTED);
    if (matA) { aclsparseLtMatDescriptorDestroy(&matA); }
}

TEST_F(AlgSetAttributeExceptionTest, ConjugateTransposeNotSupported) {
    SparseLtHandleGuard handle;
    SparseLtMatDescGuard matA(handle.get(), 8, 8, 8, 16, ACL_FLOAT,
                              ACL_SPARSE_ORDER_ROW, ACL_SPARSE_LT_SPARSITY_50_PERCENT);
    SparseLtDnMatDescGuard matB(handle.get(), 8, 8, 8, 16, ACL_FLOAT,
                                ACL_SPARSE_ORDER_ROW);
    SparseLtDnMatDescGuard matC(handle.get(), 8, 8, 8, 16, ACL_FLOAT,
                                ACL_SPARSE_ORDER_ROW);
    SparseLtDnMatDescGuard matD(handle.get(), 8, 8, 8, 16, ACL_FLOAT,
                                ACL_SPARSE_ORDER_ROW);
    aclsparseLtMatmulDescriptor_t md = nullptr;
    auto ret = aclsparseLtMatmulDescriptorInit(
        handle.get(), &md, ACL_SPARSE_OP_CONJUGATE_TRANSPOSE, ACL_SPARSE_OP_NON_TRANSPOSE,
        &matA.get(), &matB.get(), &matC.get(), &matD.get(), ACL_SPARSE_COMPUTE_32F);
    EXPECT_EQ(ret, ACL_SPARSE_STATUS_INVALID_VALUE);
    if (md) { aclsparseLtMatmulDescriptorDestroy(&md); }
}

struct AlignmentE2EResult {
    std::vector<float> hGoldenD;
    std::vector<float> hD;
    AlgSetAttributeNpuResult npu;
    int32_t expectedAlgConfigId = 0;
    int32_t expectedSplitK = 1;
};

inline AlignmentE2EResult RunAlignmentE2E(aclrtStream stream,
                                           int32_t m, int32_t k, int32_t n,
                                           float alpha, float beta,
                                           int32_t algConfigId, int32_t splitK,
                                           uint32_t seedBase)
{
    auto Af = GenFp32Matrix(m, k, -1.0f, 1.0f, seedBase);
    auto Bf = GenFp32Matrix(k, n, -1.0f, 1.0f, seedBase + 1);
    auto Cf = GenFp32Matrix(m, n, -1.0f, 1.0f, seedBase + 2);

    AlignmentE2EResult r;
    r.expectedAlgConfigId = algConfigId;
    r.expectedSplitK = splitK;
    LtMatmulAlgSetAttributeGolden<float>(Af, Bf, Cf, r.hGoldenD,
                                         m, k, n, alpha, beta,
                                         algConfigId, splitK, false);
    r.npu = AlgSetAttributeNpu<float>(stream, Af, Bf, Cf, r.hD,
                                        m, k, n, alpha, beta,
                                        algConfigId, splitK, 32);
    return r;
}

inline void AssertAndVerifyAlignmentE2E(const AlignmentE2EResult& r, const char* caseName)
{
    ASSERT_EQ(r.npu.algSetCfgRet, ACL_SPARSE_STATUS_SUCCESS);
    ASSERT_EQ(r.npu.algSetSplitRet, ACL_SPARSE_STATUS_SUCCESS);
    // getAttr roundtrip verification.
    ASSERT_EQ(r.npu.algGetCfgRet, ACL_SPARSE_STATUS_SUCCESS);
    ASSERT_EQ(r.npu.algGetSplitRet, ACL_SPARSE_STATUS_SUCCESS);
    EXPECT_EQ(r.npu.gotAlgConfigId, r.expectedAlgConfigId);
    EXPECT_EQ(r.npu.gotSplitK, r.expectedSplitK);
    ASSERT_EQ(r.npu.pruneRet, ACL_SPARSE_STATUS_SUCCESS);
    ASSERT_EQ(r.npu.matmulRet, ACL_SPARSE_STATUS_SUCCESS);
    std::vector<float> npuF = AlgSetAttrToFloat(r.hD);
    std::vector<float> goldenF = AlgSetAttrToFloat(r.hGoldenD);
    VerifyAlgMixedTolerance(goldenF, npuF, false, caseName);
}

TEST_F(AlgSetAttributeExceptionTest, Alignment32EndToEnd) {
    auto r = RunAlignmentE2E(stream_, 128, 128, 128, 1.0f, 0.0f, 0, 1, 7000u);
    AssertAndVerifyAlignmentE2E(r, "Alignment32EndToEnd");
}

TEST_F(AlgSetAttributeExceptionTest, Alignment32Alg1EndToEnd) {
    auto r = RunAlignmentE2E(stream_, 128, 128, 128, 1.0f, 1.0f, 1, 1, 7100u);
    AssertAndVerifyAlignmentE2E(r, "Alignment32Alg1EndToEnd");
}

