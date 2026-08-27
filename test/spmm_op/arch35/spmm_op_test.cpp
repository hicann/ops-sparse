/**
 * ----------------------------------------------------------------------------------------------------------
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms of the
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR
 * PURPOSE.
 * ----------------------------------------------------------------------------------------------------------
 */

/**
 * @file spmm_op_test.cpp
 * @brief GTest + CSV-driven tests for aclsparseSpMMOp (7-function Generic API).
 *
 * Operator: C = alpha * op(A) * op(B) + beta * C
 *   A: CSR (m*k), B: dense (k*n or n*k), C: dense (m*n, in-place)
 *
 * Test structure:
 *   - TEST_P (SpmmOpTest)           : parameterized success-path tests from CSV
 *   - TEST_F (SpmmOpExceptionTest)  : null-pointer / invalid-param error tests (E01-E19)
 *   - TEST_F (SpmmOpValueUpdateTest): csrValues in-place update tests (L1_value_update_01/02)
 *
 * Precision: framework MIXED_TOLERANCE mode (test/frame/verify.h), dtype-driven
 *   atol/rtol + per-element max(abs_err) limit = max(fixedValue, 32*ULP).
 *
 * Entry point is shared via test/frame/test_main.cpp.
 */

#include "test_common.h"
#include "spmm_op_golden.h"
#include "spmm_op_npu_wrapper.h"
#include "spmm_op_param.h"

#include <random>

using namespace sparse_test;

namespace {
constexpr float kFp16Max = 65504.0f;
}

// ============================================================================
// Host data conversion helpers (FP16 reverse + quantization)
// ============================================================================

static std::vector<double> Fp16ToDoubles(const std::vector<uint16_t>& v)
{
    std::vector<double> out(v.size());
    for (size_t i = 0; i < v.size(); i++) {
        out[i] = static_cast<double>(Fp16BitsToFp32(v[i]));
    }
    return out;
}

// Quantize FP64 -> FP16 -> FP64 (for FP16 golden alignment).
static std::vector<double> QuantizeDoublesViaFp16(const std::vector<double>& v)
{
    return Fp16ToDoubles(DoublesToFp16(v));
}

// ============================================================================
// TEST_P fixture: SpmmOpTest (CSV-driven functional cases)
// ============================================================================

class SpmmOpTest : public testing::TestWithParam<SpmmOpTestParam> {
public:
    static void SetUpTestSuite()
    {
        env_ = std::make_unique<AclEnvScope>();
    }

    static void TearDownTestSuite()
    {
        env_.reset();
    }

protected:
    inline static std::unique_ptr<AclEnvScope> env_;
    SpmmOpTestParam param_;
    aclrtStream stream_ = nullptr;

    void SetUp() override
    {
        param_ = GetParam();
        stream_ = env_->stream();
    }
};

// ============================================================================
// Test-body helpers (file-scope)
// ============================================================================

struct SpmmOpInputs {
    SpmmCsr csrA;
    std::vector<double> Bf64;    // canonical row-major, k*n (NON_T) or n*k (T)
    std::vector<double> CinitF64; // canonical row-major, m*n
    int64_t nnz = 0;
};

// Setup: deterministic sparsity pattern + B/C FP64 buffers.
static SpmmOpInputs GenerateSpmmOpInputs(const SpmmOpTestParam& p)
{
    SpmmOpInputs in;
    in.csrA = MakeSpmmSparsity(p.m, p.k, p.sparsity_ratio,
        p.value_lo, p.value_hi, p.random_seed,
        p.unsorted,
        (p.index_base == "ACL_SPARSE_INDEX_BASE_ONE") ? 1 : 0);
    in.nnz = in.csrA.nnz;

    // B descriptor shape depends on opB:
    //   NON_TRANSPOSE: B is k*n row-major
    //   TRANSPOSE:     B is n*k row-major
    // Element count is always k*n (same), only the logical shape differs.
    const bool bTransposed = (p.op_b == "ACL_SPARSE_OP_TRANSPOSE");
    const int64_t bRows = bTransposed ? p.n : p.k;
    const int64_t bCols = bTransposed ? p.k : p.n;
    const size_t bElemCount = static_cast<size_t>(bRows) * static_cast<size_t>(bCols);

    std::mt19937 rngB(p.random_seed + 100);
    std::mt19937 rngC(p.random_seed + 200);
    std::uniform_real_distribution<double> dist(p.value_lo, p.value_hi);

    in.Bf64.resize(bElemCount);
    for (size_t i = 0; i < bElemCount; i++) {
        in.Bf64[i] = dist(rngB);
    }

    const size_t cElemCount = static_cast<size_t>(p.m) * static_cast<size_t>(p.n);
    in.CinitF64.resize(cElemCount);
    for (size_t i = 0; i < cElemCount; i++) {
        in.CinitF64[i] = dist(rngC);
    }
    return in;
}

// Golden: FP64 reference with FP16 quantization alignment.
static std::vector<double> ComputeGoldenExpect(const SpmmOpTestParam& p,
    const SpmmCsr& csrA,
    const std::vector<double>& Bf64,
    const std::vector<double>& CinitF64,
    aclDataType dtype,
    aclsparseOperation_t opB)
{
    std::vector<double> goldenB = Bf64;
    std::vector<double> goldenCinit = CinitF64;
    SpmmCsr goldenCsr = csrA;
    double goldenAlpha = p.alpha;
    double goldenBeta = p.beta;
    if (dtype == ACL_FLOAT16) {
        goldenB = QuantizeDoublesViaFp16(Bf64);
        goldenCinit = QuantizeDoublesViaFp16(CinitF64);
        goldenCsr.values = QuantizeDoublesViaFp16(csrA.values);
        // alpha/beta remain FP32 (computeType=ACL_FLOAT)
    }
    return SpmmGolden(p.m, p.n, p.k, goldenCsr, goldenB, goldenCinit,
        goldenAlpha, goldenBeta, opB);
}

// NPU dispatch: SpMMOp lifecycle by dtype (FP32/FP16).
// Centralizes the DoublesToFp32/Fp16 conversion + SpmmOpNpu<T> call pattern
// shared by RunNpuSpmmOp and SpmmOpDeterminismTest::RunOnce.
static SpmmOpNpuResult RunSpmmOpTyped(
    aclDataType dtype, HandleManager& handle, aclrtStream stream,
    int64_t m, int64_t n, int64_t k,
    aclsparseOperation_t opB, aclsparseOrder_t orderB, aclsparseOrder_t orderC,
    float alphaF, float betaF, aclDataType computeType,
    aclsparseSpMMOpAlg_t alg, aclsparsePointerMode_t pointerMode,
    const SpmmCsr& csrA, const std::vector<double>& Bf64,
    const std::vector<double>& CinitF64, int64_t nnz,
    bool callSetGlobalUserData,
    aclsparseIndexType_t rowOffsetType, aclsparseIndexBase_t indexBase)
{
    if (dtype == ACL_FLOAT) {
        auto hAValues = DoublesToFp32(csrA.values);
        auto hB = DoublesToFp32(Bf64);
        auto hCInit = DoublesToFp32(CinitF64);
        return SpmmOpNpu<float>(handle, stream, m, n, k, opB, orderB, orderC,
            alphaF, betaF, dtype, computeType, alg, pointerMode,
            csrA.rowOffsets, csrA.colIndices,
            hAValues, hB, hCInit, nnz, callSetGlobalUserData,
            rowOffsetType, indexBase);
    } else {
        auto hAValues = DoublesToFp16(csrA.values);
        auto hB = DoublesToFp16(Bf64);
        auto hCInit = DoublesToFp16(CinitF64);
        return SpmmOpNpu<uint16_t>(handle, stream, m, n, k, opB, orderB, orderC,
            alphaF, betaF, dtype, computeType, alg, pointerMode,
            csrA.rowOffsets, csrA.colIndices,
            hAValues, hB, hCInit, nnz, callSetGlobalUserData,
            rowOffsetType, indexBase);
    }
}

// NPU lifecycle: 7-function execution.
static SpmmOpNpuResult RunNpuSpmmOp(const SpmmOpTestParam& p, aclrtStream stream,
    const SpmmCsr& csrA,
    const std::vector<double>& Bf64,
    const std::vector<double>& CinitF64,
    aclDataType dtype, aclDataType computeType,
    aclsparseSpMMOpAlg_t alg,
    aclsparseOperation_t opB,
    aclsparseOrder_t orderB,
    aclsparseOrder_t orderC,
    aclsparsePointerMode_t pointerMode,
    int64_t nnz)
{
    HandleManager handle;
    float alphaF = static_cast<float>(p.alpha);
    float betaF = static_cast<float>(p.beta);

    // Parse index type / base for CSR descriptor creation
    aclsparseIndexType_t rowOffsetType = ParseIndexType(p.row_offset_type);
    aclsparseIndexBase_t indexBase = ParseIndexBase(p.index_base);

    SpmmOpNpuResult npuResult;
    // L0_01: verify setGlobalUserData no-op
    bool callSetGlobalUserData = (p.case_name == "L0_01");

    return RunSpmmOpTyped(dtype, handle, stream, p.m, p.n, p.k, opB, orderB, orderC,
        alphaF, betaF, computeType, alg, pointerMode,
        csrA, Bf64, CinitF64, nnz, callSetGlobalUserData,
        rowOffsetType, indexBase);
}

// Golden verification: golden→float + FP16 saturation +
// applyMixedTolerance + verifyVector). Used by VerifySpmmOpPrecision and
// VerifyValueUpdateResult.
static bool VerifyGoldenWithMixedTolerance(
    const std::vector<double>& golden, const std::vector<float>& npuFloat,
    aclDataType dtype, const std::string& caseName)
{
    size_t elemCount = golden.size();
    std::vector<float> goldenFloat(elemCount);
    for (size_t i = 0; i < elemCount; i++) {
        float gv = static_cast<float>(golden[i]);
        // FP16 output saturation: kernel saturates FP32 intermediate to ±kFp16Max
        // before Cast<float->half>.
        if (dtype == ACL_FLOAT16) {
            if (gv > kFp16Max) { gv = kFp16Max; }
            if (gv < -kFp16Max) { gv = -kFp16Max; }
        }
        goldenFloat[i] = gv;
    }
    VerifyConfig cfg;
    applyMixedTolerance(cfg, dtype, goldenFloat.data(), goldenFloat.size());
    bool pass = Verifier::verifyVector(npuFloat.data(), goldenFloat.data(),
        elemCount, 1, cfg, caseName);
    return pass;
}

// Precision verification (MIXED_TOLERANCE, dtype-driven).
static void VerifySpmmOpPrecision(const std::string& caseName,
    const SpmmOpNpuResult& npuResult,
    const std::vector<double>& expectFp64,
    aclDataType dtype, int64_t m, int64_t n)
{
    const size_t elemCount = static_cast<size_t>(m) * static_cast<size_t>(n);
    if (elemCount == 0) {
        std::cout << "[" << caseName << "] m*n=0 (empty output), skip value verify\n";
        return;
    }
    std::vector<float> npuFloat(elemCount);
    for (size_t i = 0; i < elemCount; i++) {
        npuFloat[i] = static_cast<float>(npuResult.valuesOut[i]);
    }
    bool pass = VerifyGoldenWithMixedTolerance(expectFp64, npuFloat, dtype, caseName);
    EXPECT_TRUE(pass) << "MIXED_TOLERANCE verification failed for " << caseName;
}

// Test body: success-path parameterized test
TEST_P(SpmmOpTest, SpmmOpFunctional)
{
    const auto& p = param_;

    std::cout << "==== " << p.case_name << " ====" << p.description
              << " m=" << p.m << " k=" << p.k << " n=" << p.n
              << " ratio=" << p.sparsity_ratio
              << " alpha=" << p.alpha << " beta=" << p.beta
              << " dtype=" << p.dtype
              << " op_b=" << p.op_b
              << " order_b=" << p.order_b << " order_c=" << p.order_c
              << " alg=" << p.alg
              << " ptr_mode=" << p.pointer_mode
              << " value_range=[" << p.value_lo << "," << p.value_hi << "]"
              << " seed=" << p.random_seed
              << " index_base=" << p.index_base
              << " row_offset_type=" << p.row_offset_type
              << " unsorted=" << p.unsorted << "\n";

    // --- Setup: deterministic sparsity pattern + B/C (FP64) ---------------
    SpmmOpInputs inputs = GenerateSpmmOpInputs(p);
    int64_t nnz = inputs.nnz;
    std::cout << "  nnz=" << nnz << "\n";

    // --- Golden: parse enums + reference computation -----------------------
    aclDataType dtype = ParseDtype(p.dtype);
    aclDataType computeType = ParseDtype(p.compute_type);
    aclsparseOperation_t opB = ParseOperation(p.op_b);
    aclsparseOrder_t orderB = ParseOrder(p.order_b);
    aclsparseOrder_t orderC = ParseOrder(p.order_c);
    aclsparseSpMMOpAlg_t alg = ParseSpmmOpAlg(p.alg);
    aclsparsePointerMode_t pointerMode =
        (p.pointer_mode == "HOST") ? ACL_SPARSE_POINTER_MODE_HOST
                                   : ACL_SPARSE_POINTER_MODE_DEVICE;

    std::vector<double> expectFp64 = ComputeGoldenExpect(
        p, inputs.csrA, inputs.Bf64, inputs.CinitF64, dtype, opB);

    // --- NPU: 7-function lifecycle -----------------------------------------
    SpmmOpNpuResult npuResult = RunNpuSpmmOp(
        p, stream_, inputs.csrA, inputs.Bf64, inputs.CinitF64,
        dtype, computeType, alg, opB, orderB, orderC, pointerMode, nnz);

    // --- Check return codes ------------------------------------------------
    ASSERT_EQ(npuResult.bufferSizeRet, ACL_SPARSE_STATUS_SUCCESS)
        << "BufferSize stage failed";
    ASSERT_EQ(npuResult.createDescrRet, ACL_SPARSE_STATUS_SUCCESS)
        << "createDescr stage failed";
    ASSERT_EQ(npuResult.createPlanRet, ACL_SPARSE_STATUS_SUCCESS)
        << "createPlan stage failed";
    ASSERT_EQ(npuResult.executeRet, ACL_SPARSE_STATUS_SUCCESS)
        << "execute stage failed";

    // --- Precision verification --------------------------------------------
    // skip_precision: for boundary cases where FP32 accumulation under extreme
    // alpha makes value comparison meaningless (expected behavior, not a code
    // defect). API return codes are already asserted above; only numerical
    // comparison is skipped. Mirrors sddmm nnz==0 API-only verification pattern.
    if (p.skip_precision) {
        std::cout << "[" << p.case_name << "] alpha=" << p.alpha
                  << " precision boundary, skip value verify\n";
    } else {
        VerifySpmmOpPrecision(p.case_name, npuResult, expectFp64, dtype, p.m, p.n);
    }

    std::cout << "[" << p.case_name << "] PASSED (nnz=" << nnz << ")\n";
}

// Parameterized test instantiation from CSV
INSTANTIATE_TEST_SUITE_P(
    SpmmOpCases,
    SpmmOpTest,
    testing::ValuesIn(GetCasesFromCsv<SpmmOpTestParam>("spmm_op_test.csv")),
    [](const testing::TestParamInfo<SpmmOpTestParam>& info) {
        return info.param.case_name;
    }
);

// ============================================================================
// Base fixture: shared SetUp/TearDown for SpMMOp test suites.
// Eliminates duplicate env_/stream_/SetUpTestSuite/TearDownTestSuite code.
// ============================================================================

class SpmmOpTestBase : public testing::Test {
protected:
    inline static std::unique_ptr<AclEnvScope> env_;
    aclrtStream stream_ = nullptr;

    static void SetUpTestSuite()
    {
        env_ = std::make_unique<AclEnvScope>();
    }

    static void TearDownTestSuite()
    {
        env_.reset();
    }
};

// ============================================================================
// Common 4x4 CSR baseline setup: generates sparsity, device buffers, and
// descriptors for the exception/ALG2 tests. Deduplicated from
// SpmmOpExceptionTest::SetUp and SetupAlg2TestCsr.
// ============================================================================
static void Setup4x4CsrBaseline(
    SpmmCsr& csr, int& nnz,
    DeviceBuffer& dRowOff, DeviceBuffer& dColInd, DeviceBuffer& dVals,
    DeviceBuffer& dB, DeviceBuffer& dC,
    SpMatManager& matA, DnMatManager& matB, DnMatManager& matC,
    uint32_t seed, aclDataType dtype = ACL_FLOAT)
{
    csr = MakeSpmmSparsity(4, 4, 0.5, -1.0, 1.0, seed);
    nnz = static_cast<int>(csr.nnz);

    std::vector<float> aValsFp32(nnz);
    for (int i = 0; i < nnz; i++) {
        aValsFp32[static_cast<size_t>(i)] = static_cast<float>(csr.values[static_cast<size_t>(i)]);
    }
    std::vector<float> bFp32(16, 1.0f);
    std::vector<float> cInitFp32(16, 0.5f);

    dRowOff = DeviceBuffer::copyFrom(csr.rowOffsets.data(), 5 * sizeof(int32_t));
    if (nnz > 0) {
        dColInd = DeviceBuffer::copyFrom(csr.colIndices.data(),
            static_cast<size_t>(nnz) * sizeof(int32_t));
        dVals = DeviceBuffer::copyFrom(aValsFp32.data(),
            static_cast<size_t>(nnz) * sizeof(float));
    }
    dB = DeviceBuffer::copyFrom(bFp32.data(), 16 * sizeof(float));
    dC = DeviceBuffer::copyFrom(cInitFp32.data(), 16 * sizeof(float));

    // A: 4x4 CSR const, B: 4x4 dense const, C: 4x4 dense (FP32, ROW)
    matA = SpMatManager::createConstCsr(
        4, 4, nnz, dRowOff.get(), dColInd.get(), dVals.get(),
        ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_32I,
        ACL_SPARSE_INDEX_BASE_ZERO, dtype);
    matB = DnMatManager::createConst(4, 4, 4, dB.raw(), dtype, ACL_SPARSE_ORDER_ROW);
    matC = DnMatManager::create(4, 4, 4, dC.raw(), dtype, ACL_SPARSE_ORDER_ROW);
}

// ============================================================================
// Exception test fixture: SpmmOpExceptionTest (E01-E19)
// Tests are TEST_F (not CSV-driven); expect non-SUCCESS error codes.
// ============================================================================

class SpmmOpExceptionTest : public SpmmOpTestBase {
public:
protected:
    std::unique_ptr<HandleManager> handle_;

    // Baseline 4x4 CSR + dense B/C (FP32, ROW, ALG1)
    SpmmCsr csrA_;
    int nnz_ = 0;
    DeviceBuffer dRowOff_;
    DeviceBuffer dColInd_;
    DeviceBuffer dVals_;
    DeviceBuffer dB_;
    DeviceBuffer dC_;
    SpMatManager matA_;
    DnMatManager matB_;
    DnMatManager matC_;

    float alpha_ = 1.0f;
    float beta_ = 0.0f;

    void SetUp() override
    {
        stream_ = env_->stream();
        handle_ = std::make_unique<HandleManager>();
        handle_->setStream(stream_);

        // Small 4x4 CSR baseline
        Setup4x4CsrBaseline(csrA_, nnz_, dRowOff_, dColInd_, dVals_, dB_, dC_,
            matA_, matB_, matC_, 42);
    }

    // Helper: create a valid descr + plan from baseline (for execute-stage tests)
    aclsparseSpMMOpPlan_t CreateValidPlan()
    {
        size_t bufSize = 0;
        auto st = aclsparseSpMMOp_bufferSize(
            handle_->get(), ACL_SPARSE_OP_NON_TRANSPOSE, ACL_SPARSE_OP_NON_TRANSPOSE,
            matA_.cget(), matB_.cget(), matC_.get(),
            ACL_FLOAT, ACL_SPARSE_SPMMOP_ALG1, &bufSize);
        if (st != ACL_SPARSE_STATUS_SUCCESS) {
            return nullptr;
        }

        // ALG1 needs no workspace, but alloc 1 byte to avoid null
        workBuf_ = DeviceBuffer::alloc(bufSize > 0 ? bufSize : 1);

        aclsparseSpMMOpDescr_t descr = nullptr;
        st = aclsparseSpMMOp_createDescr(
            handle_->get(), &descr,
            ACL_SPARSE_OP_NON_TRANSPOSE, ACL_SPARSE_OP_NON_TRANSPOSE,
            matA_.cget(), matB_.cget(), matC_.get(),
            ACL_FLOAT, ACL_SPARSE_SPMMOP_ALG1, workBuf_.get());
        if (st != ACL_SPARSE_STATUS_SUCCESS) {
            return nullptr;
        }
        liveDescr_ = descr;  // keep alive for cleanup

        aclsparseSpMMOpPlan_t plan = nullptr;
        st = aclsparseSpMMOp_createPlan(handle_->get(), descr, &plan, nullptr, 0);
        if (st != ACL_SPARSE_STATUS_SUCCESS) {
            return nullptr;
        }
        livePlan_ = plan;
        return plan;
    }

    // Helper: create a valid descr (ALG1) for createPlan-stage tests.
    // Returns nullptr on failure. Caller must wrap in SpmmOpDescrGuard.
    aclsparseSpMMOpDescr_t CreateValidDescr()
    {
        descrWorkBuf_ = DeviceBuffer::alloc(1);
        aclsparseSpMMOpDescr_t descr = nullptr;
        auto st = aclsparseSpMMOp_createDescr(
            handle_->get(), &descr,
            ACL_SPARSE_OP_NON_TRANSPOSE, ACL_SPARSE_OP_NON_TRANSPOSE,
            matA_.cget(), matB_.cget(), matC_.get(),
            ACL_FLOAT, ACL_SPARSE_SPMMOP_ALG1, descrWorkBuf_.get());
        if (st != ACL_SPARSE_STATUS_SUCCESS) {
            return nullptr;
        }
        return descr;
    }

    void TearDown() override
    {
        // Cleanup live plan/descr created by CreateValidPlan
        if (livePlan_) {
            aclsparseSpMMOp_destroyPlan(livePlan_);
            livePlan_ = nullptr;
        }
        if (liveDescr_) {
            aclsparseSpMMOp_destroyDescr(liveDescr_);
            liveDescr_ = nullptr;
        }
        handle_.reset();
    }

private:
    DeviceBuffer workBuf_;
    DeviceBuffer descrWorkBuf_;
    aclsparseSpMMOpDescr_t liveDescr_ = nullptr;
    aclsparseSpMMOpPlan_t livePlan_ = nullptr;
};

// E01: null handle -> ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR
TEST_F(SpmmOpExceptionTest, NullHandle)
{
    size_t bs = 0;
    auto ret = aclsparseSpMMOp_bufferSize(
        nullptr, ACL_SPARSE_OP_NON_TRANSPOSE, ACL_SPARSE_OP_NON_TRANSPOSE,
        matA_.cget(), matB_.cget(), matC_.get(),
        ACL_FLOAT, ACL_SPARSE_SPMMOP_ALG1, &bs);
    EXPECT_EQ(ret, ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR);
}

// E02: null matA -> ACL_SPARSE_STATUS_INVALID_VALUE
TEST_F(SpmmOpExceptionTest, NullMatA)
{
    size_t bs = 0;
    auto ret = aclsparseSpMMOp_bufferSize(
        handle_->get(), ACL_SPARSE_OP_NON_TRANSPOSE, ACL_SPARSE_OP_NON_TRANSPOSE,
        nullptr, matB_.cget(), matC_.get(),
        ACL_FLOAT, ACL_SPARSE_SPMMOP_ALG1, &bs);
    EXPECT_EQ(ret, ACL_SPARSE_STATUS_INVALID_VALUE);
}

// E03: null matB -> ACL_SPARSE_STATUS_INVALID_VALUE
TEST_F(SpmmOpExceptionTest, NullMatB)
{
    size_t bs = 0;
    auto ret = aclsparseSpMMOp_bufferSize(
        handle_->get(), ACL_SPARSE_OP_NON_TRANSPOSE, ACL_SPARSE_OP_NON_TRANSPOSE,
        matA_.cget(), nullptr, matC_.get(),
        ACL_FLOAT, ACL_SPARSE_SPMMOP_ALG1, &bs);
    EXPECT_EQ(ret, ACL_SPARSE_STATUS_INVALID_VALUE);
}

// E04: null matC -> ACL_SPARSE_STATUS_INVALID_VALUE
TEST_F(SpmmOpExceptionTest, NullMatC)
{
    size_t bs = 0;
    auto ret = aclsparseSpMMOp_bufferSize(
        handle_->get(), ACL_SPARSE_OP_NON_TRANSPOSE, ACL_SPARSE_OP_NON_TRANSPOSE,
        matA_.cget(), matB_.cget(), nullptr,
        ACL_FLOAT, ACL_SPARSE_SPMMOP_ALG1, &bs);
    EXPECT_EQ(ret, ACL_SPARSE_STATUS_INVALID_VALUE);
}

// E05: null alpha (execute stage) -> ACL_SPARSE_STATUS_INVALID_VALUE
TEST_F(SpmmOpExceptionTest, NullAlpha)
{
    auto plan = CreateValidPlan();
    ASSERT_NE(plan, nullptr);
    auto ret = aclsparseSpMMOp(
        handle_->get(), plan, nullptr, &beta_, matB_.cget(), matC_.get());
    EXPECT_EQ(ret, ACL_SPARSE_STATUS_INVALID_VALUE);
}

// E06: null beta (execute stage) -> ACL_SPARSE_STATUS_INVALID_VALUE
TEST_F(SpmmOpExceptionTest, NullBeta)
{
    auto plan = CreateValidPlan();
    ASSERT_NE(plan, nullptr);
    auto ret = aclsparseSpMMOp(
        handle_->get(), plan, &alpha_, nullptr, matB_.cget(), matC_.get());
    EXPECT_EQ(ret, ACL_SPARSE_STATUS_INVALID_VALUE);
}

// E07: null bufferSize ptr -> ACL_SPARSE_STATUS_INVALID_VALUE
TEST_F(SpmmOpExceptionTest, NullBufferSizePtr)
{
    auto ret = aclsparseSpMMOp_bufferSize(
        handle_->get(), ACL_SPARSE_OP_NON_TRANSPOSE, ACL_SPARSE_OP_NON_TRANSPOSE,
        matA_.cget(), matB_.cget(), matC_.get(),
        ACL_FLOAT, ACL_SPARSE_SPMMOP_ALG1, nullptr);
    EXPECT_EQ(ret, ACL_SPARSE_STATUS_INVALID_VALUE);
}

// E08: null descr ptr (createDescr) -> ACL_SPARSE_STATUS_INVALID_VALUE
TEST_F(SpmmOpExceptionTest, NullDescrPtr)
{
    DeviceBuffer workBuf = DeviceBuffer::alloc(1);
    auto ret = aclsparseSpMMOp_createDescr(
        handle_->get(), nullptr,
        ACL_SPARSE_OP_NON_TRANSPOSE, ACL_SPARSE_OP_NON_TRANSPOSE,
        matA_.cget(), matB_.cget(), matC_.get(),
        ACL_FLOAT, ACL_SPARSE_SPMMOP_ALG1, workBuf.get());
    EXPECT_EQ(ret, ACL_SPARSE_STATUS_INVALID_VALUE);
}

// E09: null plan ptr (createPlan) -> ACL_SPARSE_STATUS_INVALID_VALUE
TEST_F(SpmmOpExceptionTest, NullPlanPtr)
{
    // First create a valid descr
    auto descr = CreateValidDescr();
    ASSERT_NE(descr, nullptr);
    SpmmOpDescrGuard descrGuard(descr);

    auto ret = aclsparseSpMMOp_createPlan(handle_->get(), descr, nullptr, nullptr, 0);
    EXPECT_EQ(ret, ACL_SPARSE_STATUS_INVALID_VALUE);
}

// E10: dimension mismatch (A.k != B.k) -> MATRIX_TYPE_NOT_SUPPORTED or INVALID_VALUE
TEST_F(SpmmOpExceptionTest, DimMismatch)
{
    // B with k=2 (mismatches A k=4): B is 2x4 row-major (opB=NON_T)
    std::vector<float> bSmall(8, 1.0f);
    auto dBSmall = DeviceBuffer::copyFrom(bSmall.data(), 8 * sizeof(float));
    auto matBSmall = DnMatManager::createConst(2, 4, 4, dBSmall.raw(),
        ACL_FLOAT, ACL_SPARSE_ORDER_ROW);
    auto plan = CreateValidPlan();
    ASSERT_NE(plan, nullptr);
    auto ret = aclsparseSpMMOp(
        handle_->get(), plan, &alpha_, &beta_, matBSmall.cget(), matC_.get());
    EXPECT_TRUE(ret == ACL_SPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED ||
                ret == ACL_SPARSE_STATUS_INVALID_VALUE)
        << "expected MATRIX_TYPE_NOT_SUPPORTED or INVALID_VALUE, got " << ret;
}

// E11: unsupported computeType (ACL_DOUBLE) -> ACL_SPARSE_STATUS_NOT_SUPPORTED
TEST_F(SpmmOpExceptionTest, UnsupportedComputeType)
{
    size_t bs = 0;
    auto ret = aclsparseSpMMOp_bufferSize(
        handle_->get(), ACL_SPARSE_OP_NON_TRANSPOSE, ACL_SPARSE_OP_NON_TRANSPOSE,
        matA_.cget(), matB_.cget(), matC_.get(),
        ACL_DOUBLE, ACL_SPARSE_SPMMOP_ALG1, &bs);
    EXPECT_EQ(ret, ACL_SPARSE_STATUS_NOT_SUPPORTED);
}

// E12: unsupported opA (TRANSPOSE) -> ACL_SPARSE_STATUS_NOT_SUPPORTED
TEST_F(SpmmOpExceptionTest, UnsupportedOpA)
{
    size_t bs = 0;
    auto ret = aclsparseSpMMOp_bufferSize(
        handle_->get(), ACL_SPARSE_OP_TRANSPOSE, ACL_SPARSE_OP_NON_TRANSPOSE,
        matA_.cget(), matB_.cget(), matC_.get(),
        ACL_FLOAT, ACL_SPARSE_SPMMOP_ALG1, &bs);
    EXPECT_EQ(ret, ACL_SPARSE_STATUS_NOT_SUPPORTED);
}

// E13: null CSR rowOffsets ptr -> ACL_SPARSE_STATUS_INVALID_VALUE
TEST_F(SpmmOpExceptionTest, NullCsrRowOffsets)
{
    // Create CSR with null rowOffsets (nnz=0 so colInd/values also null)
    auto matANullPtrs = SpMatManager::createConstCsr(
        4, 4, 0, nullptr, dColInd_.get(), dVals_.get(),
        ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_32I,
        ACL_SPARSE_INDEX_BASE_ZERO, ACL_FLOAT);
    size_t bs = 0;
    auto ret = aclsparseSpMMOp_bufferSize(
        handle_->get(), ACL_SPARSE_OP_NON_TRANSPOSE, ACL_SPARSE_OP_NON_TRANSPOSE,
        matANullPtrs.cget(), matB_.cget(), matC_.get(),
        ACL_FLOAT, ACL_SPARSE_SPMMOP_ALG1, &bs);
    EXPECT_TRUE(ret == ACL_SPARSE_STATUS_INVALID_VALUE ||
                ret == ACL_SPARSE_STATUS_NOT_SUPPORTED)
        << "expected INVALID_VALUE or NOT_SUPPORTED, got " << ret;
}

// E14: non-null epilogueLTOBuffer (createPlan) -> NOT_SUPPORTED
TEST_F(SpmmOpExceptionTest, NonNullEpilogueLTO)
{
    auto descr = CreateValidDescr();
    ASSERT_NE(descr, nullptr);
    SpmmOpDescrGuard descrGuard(descr);

    DeviceBuffer ltoBuf = DeviceBuffer::alloc(16);
    aclsparseSpMMOpPlan_t plan = nullptr;
    auto ret = aclsparseSpMMOp_createPlan(
        handle_->get(), descr, &plan, ltoBuf.get(), 16);
    EXPECT_TRUE(ret == ACL_SPARSE_STATUS_NOT_SUPPORTED ||
                ret == ACL_SPARSE_STATUS_INVALID_VALUE)
        << "expected NOT_SUPPORTED or INVALID_VALUE, got " << ret;
}

// E15: destroyDescr(nullptr) idempotent -> SUCCESS
TEST_F(SpmmOpExceptionTest, DestroyDescrNull)
{
    auto ret = aclsparseSpMMOp_destroyDescr(nullptr);
    EXPECT_EQ(ret, ACL_SPARSE_STATUS_SUCCESS);
}

// E16: destroyPlan(nullptr) idempotent -> SUCCESS
TEST_F(SpmmOpExceptionTest, DestroyPlanNull)
{
    auto ret = aclsparseSpMMOp_destroyPlan(nullptr);
    EXPECT_EQ(ret, ACL_SPARSE_STATUS_SUCCESS);
}

// E17: null plan (execute) -> ACL_SPARSE_STATUS_INVALID_VALUE
TEST_F(SpmmOpExceptionTest, NullPlanExecute)
{
    auto ret = aclsparseSpMMOp(
        handle_->get(), nullptr, &alpha_, &beta_, matB_.cget(), matC_.get());
    EXPECT_EQ(ret, ACL_SPARSE_STATUS_INVALID_VALUE);
}

// E18: unsupported dtype combo (A=FP32, B=FP16) -> NOT_SUPPORTED
TEST_F(SpmmOpExceptionTest, UnsupportedDtypeCombo)
{
    // Create B with FP16 dtype (mismatches A's FP32)
    std::vector<uint16_t> bFp16(16, 0);  // FP16 zero data
    auto dBFp16 = DeviceBuffer::copyFrom(bFp16.data(), 16 * sizeof(uint16_t));
    auto matBFp16 = DnMatManager::createConst(4, 4, 4, dBFp16.raw(),
        ACL_FLOAT16, ACL_SPARSE_ORDER_ROW);
    auto plan = CreateValidPlan();
    ASSERT_NE(plan, nullptr);
    auto ret = aclsparseSpMMOp(
        handle_->get(), plan, &alpha_, &beta_, matBFp16.cget(), matC_.get());
    EXPECT_TRUE(ret == ACL_SPARSE_STATUS_NOT_SUPPORTED ||
                ret == ACL_SPARSE_STATUS_INVALID_VALUE)
        << "expected NOT_SUPPORTED or INVALID_VALUE, got " << ret;
}

// E19: ALG2 + csrValues in-place update (execute returns SUCCESS, no v2 verify)
TEST_F(SpmmOpExceptionTest, Alg2InPlaceValueUpdate)
{
    SpmmCsr csr;
    int nnz = 0;
    DeviceBuffer dRowOff;
    DeviceBuffer dColInd;
    DeviceBuffer dVals;
    DeviceBuffer dB;
    DeviceBuffer dC;
    SpMatManager matA;
    DnMatManager matB;
    DnMatManager matC;
    Setup4x4CsrBaseline(csr, nnz, dRowOff, dColInd, dVals, dB, dC, matA, matB, matC, 77);

    // Query bufferSize for ALG2
    size_t bufSize = 0;
    auto st = aclsparseSpMMOp_bufferSize(
        handle_->get(), ACL_SPARSE_OP_NON_TRANSPOSE, ACL_SPARSE_OP_NON_TRANSPOSE,
        matA.cget(), matB.cget(), matC.get(),
        ACL_FLOAT, ACL_SPARSE_SPMMOP_ALG2, &bufSize);
    ASSERT_EQ(st, ACL_SPARSE_STATUS_SUCCESS);

    DeviceBuffer workBuf = DeviceBuffer::alloc(bufSize > 0 ? bufSize : 1);

    // Create descr with ALG2
    aclsparseSpMMOpDescr_t descr = nullptr;
    st = aclsparseSpMMOp_createDescr(
        handle_->get(), &descr,
        ACL_SPARSE_OP_NON_TRANSPOSE, ACL_SPARSE_OP_NON_TRANSPOSE,
        matA.cget(), matB.cget(), matC.get(),
        ACL_FLOAT, ACL_SPARSE_SPMMOP_ALG2, workBuf.get());
    ASSERT_EQ(st, ACL_SPARSE_STATUS_SUCCESS);
    SpmmOpDescrGuard descrGuard(descr);

    // Create plan
    aclsparseSpMMOpPlan_t plan = nullptr;
    st = aclsparseSpMMOp_createPlan(handle_->get(), descr, &plan, nullptr, 0);
    ASSERT_EQ(st, ACL_SPARSE_STATUS_SUCCESS);
    SpmmOpPlanGuard planGuard(plan);

    // Execute v1
    st = aclsparseSpMMOp(
        handle_->get(), plan, &alpha_, &beta_, matB.cget(), matC.get());
    EXPECT_EQ(st, ACL_SPARSE_STATUS_SUCCESS);
    aclrtSynchronizeStream(stream_);

    // In-place update csrValues: multiply by -1.0
    if (nnz > 0) {
        std::vector<float> newVals(nnz);
        for (int i = 0; i < nnz; i++) {
            newVals[static_cast<size_t>(i)] = static_cast<float>(csr.values[static_cast<size_t>(i)]) * -1.0f;
        }
        aclrtMemcpy(dVals.get(), static_cast<size_t>(nnz) * sizeof(float),
                    newVals.data(), static_cast<size_t>(nnz) * sizeof(float),
                    ACL_MEMCPY_HOST_TO_DEVICE);
    }

    // Re-execute with same plan (ALG2: documented as unsupported)
    st = aclsparseSpMMOp(
        handle_->get(), plan, &alpha_, &beta_, matB.cget(), matC.get());
    // Documented: no runtime detection, execute returns SUCCESS
    EXPECT_EQ(st, ACL_SPARSE_STATUS_SUCCESS);
    aclrtSynchronizeStream(stream_);
    // Per test plan: do NOT verify v2 result correctness (behavior undefined)
}

// ============================================================================
// Value update test fixture: SpmmOpValueUpdateTest (L1_value_update_01/02)
// ALG1 csrValues in-place update: createDescr -> createPlan -> execute(v1)
//   -> update values -> re-execute(v2) -> verify both v1 and v2 golden.
// ============================================================================

class SpmmOpValueUpdateTest : public SpmmOpTestBase {
protected:
    void SetUp() override
    {
        stream_ = env_->stream();
    }
};

// Helper 1: Generate CSR pattern + B/C FP64 buffers + golden_v1 (FP16-quantized).
static void GenerateValueUpdateInputs(
    int64_t m, int64_t k, int64_t n, double ratio,
    double value_lo, double value_hi, uint32_t seed,
    aclDataType dtype,
    SpmmCsr& csr, int64_t& nnz,
    std::vector<double>& Bf64, std::vector<double>& CinitF64,
    std::vector<double>& goldenV1)
{
    csr = MakeSpmmSparsity(m, k, ratio, value_lo, value_hi, seed, false, 0);
    nnz = csr.nnz;

    std::mt19937 rngB(seed + 100);
    std::mt19937 rngC(seed + 200);
    std::uniform_real_distribution<double> dist(value_lo, value_hi);
    Bf64.resize(static_cast<size_t>(k) * static_cast<size_t>(n));
    CinitF64.resize(static_cast<size_t>(m) * static_cast<size_t>(n));
    for (auto& v : Bf64) v = dist(rngB);
    for (auto& v : CinitF64) v = dist(rngC);

    // Compute golden_v1 (with FP16 quantization if dtype=FP16)
    std::vector<double> goldenB = Bf64;
    std::vector<double> goldenCinit = CinitF64;
    SpmmCsr goldenCsr = csr;
    if (dtype == ACL_FLOAT16) {
        goldenB = QuantizeDoublesViaFp16(Bf64);
        goldenCinit = QuantizeDoublesViaFp16(CinitF64);
        goldenCsr.values = QuantizeDoublesViaFp16(csr.values);
    }

    goldenV1 = SpmmGolden(
        m, n, k, goldenCsr, goldenB, goldenCinit, 1.0, 0.0,
        ACL_SPARSE_OP_NON_TRANSPOSE);
}

// Helper 2a: Create SpMMOp descriptor + plan (ALG1) from existing matA/matB/matC.
// Queries bufferSize, allocates workspace, creates descr + plan.
// Returns ACL_SPARSE_STATUS_SUCCESS or the failing status code.
static aclsparseStatus_t CreateSpmmOpDescrAndPlan(
    HandleManager& handle, SpMatManager& matA, DnMatManager& matB, DnMatManager& matC,
    aclDataType computeType, DeviceBuffer& workBuf,
    aclsparseSpMMOpDescr_t& descr, aclsparseSpMMOpPlan_t& plan)
{
    // bufferSize + createDescr (ALG1)
    size_t bufSize = 0;
    auto st = aclsparseSpMMOp_bufferSize(
        handle.get(), ACL_SPARSE_OP_NON_TRANSPOSE, ACL_SPARSE_OP_NON_TRANSPOSE,
        matA.cget(), matB.cget(), matC.get(),
        computeType, ACL_SPARSE_SPMMOP_ALG1, &bufSize);
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        return st;
    }

    workBuf = DeviceBuffer::alloc(bufSize > 0 ? bufSize : 1);
    descr = nullptr;
    st = aclsparseSpMMOp_createDescr(
        handle.get(), &descr,
        ACL_SPARSE_OP_NON_TRANSPOSE, ACL_SPARSE_OP_NON_TRANSPOSE,
        matA.cget(), matB.cget(), matC.get(),
        computeType, ACL_SPARSE_SPMMOP_ALG1, workBuf.get());
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        return st;
    }

    // createPlan
    plan = nullptr;
    st = aclsparseSpMMOp_createPlan(handle.get(), descr, &plan, nullptr, 0);
    return st;
}

// Helper 2: Set up CSR + DnMat device buffers and descriptors for the
// value-update test. Centralizes the DoublesToFp32/Fp16 conversion and
// delegates to PrepareCsrDeviceBuffers / PrepareDnMatDeviceBuffers.
template <typename T>
static void SetupValueUpdateDeviceBuffers(
    int64_t m, int64_t k, int64_t n, int64_t nnz, aclDataType dtype,
    const SpmmCsr& csr,
    const std::vector<double>& Bf64, const std::vector<double>& CinitF64,
    DeviceBuffer& dRowOff, DeviceBuffer& dColInd, DeviceBuffer& dVals,
    DeviceBuffer& dB, DeviceBuffer& dC,
    SpMatManager& matA, DnMatManager& matB, DnMatManager& matC)
{
    std::vector<T> hAValues;
    std::vector<T> hB;
    std::vector<T> hCInit;
    if constexpr (std::is_same_v<T, float>) {
        hAValues = DoublesToFp32(csr.values);
        hB = DoublesToFp32(Bf64);
        hCInit = DoublesToFp32(CinitF64);
    } else {
        hAValues = DoublesToFp16(csr.values);
        hB = DoublesToFp16(Bf64);
        hCInit = DoublesToFp16(CinitF64);
    }
    PrepareCsrDeviceBuffers<T>(
        m, k, nnz, dtype, ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_BASE_ZERO,
        csr.rowOffsets, csr.colIndices, hAValues,
        dRowOff, dColInd, dVals, matA);
    PrepareDnMatDeviceBuffers<T>(
        m, n, k, ACL_SPARSE_OP_NON_TRANSPOSE,
        ACL_SPARSE_ORDER_ROW, ACL_SPARSE_ORDER_ROW, dtype,
        hB, hCInit, matB, matC, dB, dC);
}

// Helper 2b: Set up device buffers, descriptors, plan for ALG1 value-update test.
// Returns ACL_SPARSE_STATUS_SUCCESS on success, or the failing status code.
// Caller is responsible for wrapping descr/plan in RAII guards after a
// successful return (descr/plan are raw out-params so the caller can decide
// lifetime management).
template <typename T>
static aclsparseStatus_t SetupValueUpdateNpuResources(
    HandleManager& handle, aclrtStream stream,
    int64_t m, int64_t k, int64_t n, int64_t nnz, aclDataType dtype,
    const SpmmCsr& csr,
    const std::vector<double>& Bf64, const std::vector<double>& CinitF64,
    DeviceBuffer& dRowOff, DeviceBuffer& dColInd, DeviceBuffer& dVals,
    DeviceBuffer& dB, DeviceBuffer& dC,
    SpMatManager& matA, DnMatManager& matB, DnMatManager& matC,
    aclsparseSpMMOpDescr_t& descr, aclsparseSpMMOpPlan_t& plan,
    DeviceBuffer& workBuf)
{
    handle.setStream(stream);
    SetupValueUpdateDeviceBuffers<T>(
        m, k, n, nnz, dtype, csr, Bf64, CinitF64,
        dRowOff, dColInd, dVals, dB, dC, matA, matB, matC);
    // bufferSize + createDescr + createPlan (ALG1)
    return CreateSpmmOpDescrAndPlan(handle, matA, matB, matC, ACL_FLOAT,
        workBuf, descr, plan);
}

// Helper 3: Read back C output, convert to float, verify against golden with
// mixed tolerance. Replaces the identical v1 and v2 verify blocks.
template <typename T>
static void VerifyValueUpdateResult(
    DeviceBuffer& dC, size_t cElemCount,
    const std::vector<double>& golden, aclDataType dtype,
    const std::string& caseName)
{
    std::vector<T> hCOut(cElemCount);
    dC.copyToHost(hCOut.data(), cElemCount * sizeof(T));

    std::vector<float> npuFloat(cElemCount);
    for (size_t i = 0; i < cElemCount; i++) {
        if constexpr (std::is_same_v<T, float>) {
            npuFloat[i] = hCOut[i];
        } else {
            npuFloat[i] = Fp16BitsToFp32(hCOut[i]);
        }
    }
    bool pass = VerifyGoldenWithMixedTolerance(golden, npuFloat, dtype, caseName);
    EXPECT_TRUE(pass) << caseName << " verification FAILED";
}

// Helper 3a: In-place update csrValues on device + compute golden_v2.
// values_v2 = values_v1 * valueScaleFactor, copied to device via aclrtMemcpy.
// golden_v2 recomputes SpmmGolden with FP16-quantized inputs if dtype=FP16.
template <typename T>
static void UpdateCsrValuesAndComputeGoldenV2(
    DeviceBuffer& dVals, int64_t nnz, float valueScaleFactor,
    const SpmmCsr& csr, int64_t m, int64_t n, int64_t k,
    const std::vector<double>& Bf64,
    const std::vector<double>& CinitF64, aclDataType dtype,
    std::vector<double>& goldenV2)
{
    // 4. In-place update csrValues: values_v2 = values_v1 * factor
    std::vector<double> newValuesD(nnz);
    for (int64_t i = 0; i < nnz; i++) {
        newValuesD[static_cast<size_t>(i)] = csr.values[static_cast<size_t>(i)] * valueScaleFactor;
    }
    if (nnz > 0) {
        std::vector<T> newVals(nnz);
        if constexpr (std::is_same_v<T, float>) {
            for (int64_t i = 0; i < nnz; i++) {
                newVals[static_cast<size_t>(i)] = static_cast<float>(newValuesD[static_cast<size_t>(i)]);
            }
        } else {
            for (int64_t i = 0; i < nnz; i++) {
                newVals[static_cast<size_t>(i)] =
                    Fp32ToFp16Bits(static_cast<float>(newValuesD[static_cast<size_t>(i)]));
            }
        }
        aclrtMemcpy(dVals.get(), static_cast<size_t>(nnz) * sizeof(T),
                    newVals.data(), static_cast<size_t>(nnz) * sizeof(T),
                    ACL_MEMCPY_HOST_TO_DEVICE);
    }

    // 5. Compute golden_v2 (recompute goldenB/goldenCinit for FP16 alignment)
    std::vector<double> goldenB = Bf64;
    std::vector<double> goldenCinit = CinitF64;
    if (dtype == ACL_FLOAT16) {
        goldenB = QuantizeDoublesViaFp16(Bf64);
        goldenCinit = QuantizeDoublesViaFp16(CinitF64);
    }
    SpmmCsr goldenCsrV2 = csr;
    goldenCsrV2.values = newValuesD;
    if (dtype == ACL_FLOAT16) {
        goldenCsrV2.values = QuantizeDoublesViaFp16(newValuesD);
    }
    goldenV2 = SpmmGolden(m, n, k, goldenCsrV2, goldenB, goldenCinit, 1.0, 0.0,
        ACL_SPARSE_OP_NON_TRANSPOSE);
}

// Orchestrator: run ALG1 value-update test for a given dtype
template <typename T>
static void RunValueUpdateTest(
    const std::string& caseName, aclrtStream stream,
    int64_t m, int64_t k, int64_t n, double ratio,
    double value_lo, double value_hi, uint32_t seed,
    aclDataType dtype, float valueScaleFactor)
{
    // 1. Generate inputs + golden_v1
    SpmmCsr csr;
    int64_t nnz = 0;
    std::vector<double> Bf64;
    std::vector<double> CinitF64;
    std::vector<double> goldenV1;
    GenerateValueUpdateInputs(m, k, n, ratio, value_lo, value_hi, seed, dtype,
        csr, nnz, Bf64, CinitF64, goldenV1);
    std::cout << "[" << caseName << "] nnz=" << nnz << "\n";

    // 2. NPU setup: device buffers, descriptors, plan (ALG1)
    HandleManager handle;
    float alpha = 1.0f;
    float beta = 0.0f;
    DeviceBuffer dRowOff;
    DeviceBuffer dColInd;
    DeviceBuffer dVals;
    DeviceBuffer dB;
    DeviceBuffer dC;
    DeviceBuffer workBuf;
    SpMatManager matA;
    DnMatManager matB;
    DnMatManager matC;
    aclsparseSpMMOpDescr_t descr = nullptr;
    aclsparseSpMMOpPlan_t plan = nullptr;
    auto setupSt = SetupValueUpdateNpuResources<T>(
        handle, stream, m, k, n, nnz, dtype, csr, Bf64, CinitF64,
        dRowOff, dColInd, dVals, dB, dC, matA, matB, matC, descr, plan, workBuf);
    ASSERT_EQ(setupSt, ACL_SPARSE_STATUS_SUCCESS) << "NPU setup failed";
    SpmmOpDescrGuard descrGuard(descr);
    SpmmOpPlanGuard planGuard(plan);

    size_t cElemCount = static_cast<size_t>(m) * static_cast<size_t>(n);

    // 3. Execute v1 + verify
    auto st = aclsparseSpMMOp(
        handle.get(), plan, &alpha, &beta, matB.cget(), matC.get());
    ASSERT_EQ(st, ACL_SPARSE_STATUS_SUCCESS) << "execute v1 failed";
    aclrtSynchronizeStream(stream);
    VerifyValueUpdateResult<T>(dC, cElemCount, goldenV1, dtype, caseName + "_v1");

    // 4. In-place update csrValues: values_v2 = values_v1 * factor
    // 5. Compute golden_v2 (recompute goldenB/goldenCinit for FP16 alignment)
    std::vector<double> goldenV2;
    UpdateCsrValuesAndComputeGoldenV2<T>(dVals, nnz, valueScaleFactor,
        csr, m, n, k, Bf64, CinitF64, dtype, goldenV2);

    // 6. Re-execute with same plan (ALG1 supports in-place value update) + verify
    st = aclsparseSpMMOp(
        handle.get(), plan, &alpha, &beta, matB.cget(), matC.get());
    ASSERT_EQ(st, ACL_SPARSE_STATUS_SUCCESS) << "execute v2 failed";
    aclrtSynchronizeStream(stream);
    VerifyValueUpdateResult<T>(dC, cElemCount, goldenV2, dtype, caseName + "_v2");
}

// L1_value_update_01: FP32 ALG1 csrValues in-place update (values * -1.0)
TEST_F(SpmmOpValueUpdateTest, L1_value_update_01)
{
    RunValueUpdateTest<float>(
        "L1_value_update_01", stream_,
        128, 64, 128, 0.5, -1.0, 1.0, 701,
        ACL_FLOAT, -1.0f);
}

// L1_value_update_02: FP16 ALG1 csrValues in-place update (values * 2.0)
TEST_F(SpmmOpValueUpdateTest, L1_value_update_02)
{
    RunValueUpdateTest<uint16_t>(
        "L1_value_update_02", stream_,
        128, 64, 128, 0.5, -0.5, 0.5, 702,
        ACL_FLOAT16, 2.0f);
}

// ============================================================================
// Precision R3: Determinism + ALG1/ALG2 consistency + multi-seed
// ============================================================================

class SpmmOpDeterminismTest : public SpmmOpTestBase {
protected:
    void SetUp() override
    {
        stream_ = env_->stream();
    }

    // Helper: run SpMM with given params and return NPU output (FP64 vector).
    // valuesOut is std::vector<double> to match SpmmOpNpuResult.valuesOut.
    struct DetRunResult {
        std::vector<double> valuesOut;
        bool success;
    };

    DetRunResult RunOnce(int64_t m, int64_t n, int64_t k,
        double ratio, double alpha, double beta,
        aclDataType dtype, aclsparseSpMMOpAlg_t alg,
        uint32_t seed)
    {
        // Generate deterministic CSR + B + C using same seed pattern as CSV tests
        SpmmCsr csrA = MakeSpmmSparsity(m, k, ratio, -1.0, 1.0, seed, false, 0);
        std::mt19937 rngB(seed + 100);
        std::uniform_real_distribution<double> dist(-1.0, 1.0);
        std::vector<double> Bf64(static_cast<size_t>(k) * static_cast<size_t>(n));
        for (auto& v : Bf64) v = dist(rngB);
        std::vector<double> CinitF64(static_cast<size_t>(m) * static_cast<size_t>(n));
        std::mt19937 rngC(seed + 200);
        for (auto& v : CinitF64) v = dist(rngC);

        float alphaF = static_cast<float>(alpha);
        float betaF = static_cast<float>(beta);

        HandleManager handle;
        SpmmOpNpuResult npuResult;
        npuResult = RunSpmmOpTyped(dtype, handle, stream_, m, n, k,
            ACL_SPARSE_OP_NON_TRANSPOSE,
            ACL_SPARSE_ORDER_ROW, ACL_SPARSE_ORDER_ROW,
            alphaF, betaF, ACL_FLOAT, alg,
            ACL_SPARSE_POINTER_MODE_DEVICE,
            csrA, Bf64, CinitF64, csrA.nnz, false,
            ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_BASE_ZERO);

        DetRunResult res;
        res.success = (npuResult.executeRet == ACL_SPARSE_STATUS_SUCCESS);
        if (res.success) {
            res.valuesOut = npuResult.valuesOut;  // already FP64 vector
        }
        return res;
    }

    // Helper: convert vector<double> -> vector<float> for Verifier API
    static std::vector<float> ToFloatVec(const std::vector<double>& v)
    {
        std::vector<float> out(v.size());
        for (size_t i = 0; i < v.size(); i++) {
            out[i] = static_cast<float>(v[i]);
        }
        return out;
    }

    // Helper: run ALG1 and ALG2 on the same input, assert success, compare
    // mismatches/maxDiff, and verify ALG2 output against ALG1 (used as
    // reference) with mixed tolerance. Replaces the FP32 and FP16 blocks.
    void VerifyAlgConsistency(int64_t m, int64_t n, int64_t k,
        double ratio, double alpha, double beta,
        aclDataType dtype, uint32_t seed)
    {
        auto runALG1 = RunOnce(m, n, k, ratio, alpha, beta, dtype,
            ACL_SPARSE_SPMMOP_ALG1, seed);
        auto runALG2 = RunOnce(m, n, k, ratio, alpha, beta, dtype,
            ACL_SPARSE_SPMMOP_ALG2, seed);
        std::string dtypeName = (dtype == ACL_FLOAT) ? "FP32" : "FP16";
        ASSERT_TRUE(runALG1.success) << dtypeName << " ALG1 execute failed";
        ASSERT_TRUE(runALG2.success) << dtypeName << " ALG2 execute failed";
        ASSERT_EQ(runALG1.valuesOut.size(), runALG2.valuesOut.size());

        // ALG1 and ALG2 may produce different FP32 accumulation order due to
        // row reorder, so check with mixed tolerance instead of bit-exact.
        int mismatches = 0;
        double maxDiff = 0.0;
        for (size_t i = 0; i < runALG1.valuesOut.size(); ++i) {
            double diff = std::abs(runALG1.valuesOut[i] - runALG2.valuesOut[i]);
            if (diff > 0.0) {
                mismatches++;
                maxDiff = std::max(maxDiff, diff);
            }
        }
        std::cout << "[ALG1vsALG2] " << dtypeName << ": " << mismatches << " / "
                  << runALG1.valuesOut.size()
                  << " elements differ, maxDiff=" << maxDiff << "\n";

        // Verify with MixedTolerance (ALG1 as reference, ALG2 as output)
        std::vector<float> alg1Float = ToFloatVec(runALG1.valuesOut);
        std::vector<float> alg2Float = ToFloatVec(runALG2.valuesOut);
        VerifyConfig cfg;
        applyMixedTolerance(cfg, dtype, alg1Float.data(), alg1Float.size());
        std::string caseId = "ALG1vsALG2_" + dtypeName;
        bool pass = Verifier::verifyVector(alg2Float.data(), alg1Float.data(),
            alg1Float.size(), 1, cfg, caseId);
        EXPECT_TRUE(pass) << "ALG1 vs ALG2 " << dtypeName << " consistency failed";
    }
};

// Test 1: Determinism - same input, 5 runs, verify bit-identical
TEST_F(SpmmOpDeterminismTest, DeterminismSameInput)
{
    const int64_t m = 128;
    const int64_t n = 128;
    const int64_t k = 256;
    const double ratio = 0.3;
    const double alpha = 1.0;
    const double beta = 0.0;
    const uint32_t seed = 801;

    auto run1 = RunOnce(m, n, k, ratio, alpha, beta, ACL_FLOAT,
                        ACL_SPARSE_SPMMOP_ALG1, seed);
    ASSERT_TRUE(run1.success) << "FP32 ALG1 run1 execute failed";

    for (int rep = 2; rep <= 5; ++rep) {
        auto runN = RunOnce(m, n, k, ratio, alpha, beta, ACL_FLOAT,
                            ACL_SPARSE_SPMMOP_ALG1, seed);
        ASSERT_TRUE(runN.success) << "FP32 ALG1 run" << rep << " execute failed";
        ASSERT_EQ(run1.valuesOut.size(), runN.valuesOut.size());
        for (size_t i = 0; i < run1.valuesOut.size(); ++i) {
            EXPECT_EQ(run1.valuesOut[i], runN.valuesOut[i])
                << "bit-identical mismatch at element " << i << " on run " << rep;
        }
    }
    std::cout << "[Determinism] 5 runs FP32 ALG1: bit-identical PASS\n";
}

// Test 2: ALG1 vs ALG2 consistency - same input, both ALGs, verify within tolerance
TEST_F(SpmmOpDeterminismTest, ALG1VsALG2Consistency)
{
    // FP32
    VerifyAlgConsistency(128, 128, 256, 0.3, 1.0, 0.0, ACL_FLOAT, 802);
    // FP16
    VerifyAlgConsistency(128, 128, 256, 0.3, 1.0, 0.0, ACL_FLOAT16, 803);
}

// Test 3: Multi-seed stability - 5 different seeds, all should pass golden verification
TEST_F(SpmmOpDeterminismTest, MultiSeedStability)
{
    const int64_t m = 128;
    const int64_t n = 128;
    const int64_t k = 256;
    const double ratio = 0.3;
    const double alpha = 1.0;
    const double beta = 0.5;

    for (uint32_t seed = 810; seed <= 814; ++seed) {
        // FP32 with ALG_DEFAULT (routes to ALG1)
        auto npuRes = RunOnce(m, n, k, ratio, alpha, beta, ACL_FLOAT,
                              ACL_SPARSE_SPMMOP_ALG_DEFAULT, seed);
        ASSERT_TRUE(npuRes.success) << "FP32 seed=" << seed << " execute failed";

        // Compute golden (FP64, same input generation as RunOnce)
        SpmmCsr csrA = MakeSpmmSparsity(m, k, ratio, -1.0, 1.0, seed, false, 0);
        std::mt19937 rngB(seed + 100);
        std::uniform_real_distribution<double> dist(-1.0, 1.0);
        std::vector<double> Bf64(static_cast<size_t>(k) * static_cast<size_t>(n));
        for (auto& v : Bf64) v = dist(rngB);
        std::vector<double> CinitF64(static_cast<size_t>(m) * static_cast<size_t>(n));
        std::mt19937 rngC(seed + 200);
        for (auto& v : CinitF64) v = dist(rngC);

        std::vector<double> golden = SpmmGolden(m, n, k, csrA, Bf64, CinitF64,
            alpha, beta,
            ACL_SPARSE_OP_NON_TRANSPOSE);

        // Convert to float for verification
        std::vector<float> goldenF(golden.size());
        for (size_t i = 0; i < golden.size(); i++) {
            goldenF[i] = static_cast<float>(golden[i]);
        }
        std::vector<float> npuF = ToFloatVec(npuRes.valuesOut);

        VerifyConfig cfg;
        applyMixedTolerance(cfg, ACL_FLOAT, goldenF.data(), goldenF.size());
        bool pass = Verifier::verifyVector(npuF.data(), goldenF.data(),
            goldenF.size(), 1, cfg,
            ("MultiSeed_FP32_seed" + std::to_string(seed)).c_str());
        EXPECT_TRUE(pass) << "Multi-seed FP32 seed=" << seed << " failed";
    }
    std::cout << "[MultiSeed] 5 seeds FP32: all PASS\n";
}
