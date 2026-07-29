/**
 * ----------------------------------------------------------------------------------------------------------
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * ----------------------------------------------------------------------------------------------------------
 */

/**
 * @file sddmm_test.cpp
 * @brief GTest + CSV-driven tests for aclsparseSDDMM (Generic API).
 *
 * Operator: C_out = (alpha * X * op(Y) + beta * C) ∘ spy(C)
 *   X: m×k dense (row-major), Y: k×n or n×k dense (row-major, per opY), C: m×n CSR (values in place)
 *
 * Test structure:
 *   - TEST_P (SddmmTest)           : parameterized success-path tests from CSV
 *   - TEST_F (SddmmExceptionTest)  : null-pointer / invalid-param error tests (E1-E9)
 *
 * Precision: framework MIXED_TOLERANCE mode (test/frame/verify.h), dtype-driven
 *   atol/rtol + per-element max(abs_err) limit = max(fixedValue, 32*ULP).
 *
 * Entry point is shared via test/frame/test_main.cpp.
 */

#include "test_common.h"
#include "sddmm_golden.h"
#include "sddmm_npu_wrapper.h"
#include "sddmm_param.h"

#include <random>

using namespace sparse_test;

// ============================================================================
// Host data conversion helpers (FP64 -> dtype-sized host vectors for NPU input)
// DoublesToFp32 / DoublesToFp16 are provided by sddmm_golden.h (shared with
// sddmm_perf.cpp). Only FP16->FP64 reverse helpers remain test-local.
// ============================================================================

// Reverse: FP16 bits -> FP64. Mirrors the precision the NPU actually sees.
static std::vector<double> Fp16ToDoubles(const std::vector<uint16_t>& v) {
    std::vector<double> out(v.size());
    for (size_t i = 0; i < v.size(); i++) {
        out[i] = static_cast<double>(Fp16BitsToFp32(v[i]));
    }
    return out;
}

// Quantize FP64 -> FP16 -> FP64.
// For FP16 cases the NPU input is quantized to FP16 (see DoublesToFp16 in the
// NPU stage). Align the golden reference to the same quantized inputs so the
// comparison reflects the kernel's own computation precision rather than the
// FP16 input quantization error (~2^-10 per element, amplified by k-length dot
// product). Applied to X, Y, and the C initial values (beta * C term).
static std::vector<double> QuantizeDoublesViaFp16(const std::vector<double>& v) {
    return Fp16ToDoubles(DoublesToFp16(v));
}

// ============================================================================
// Row-major -> col-major repack helper.
//
// The golden reference always computes from a canonical row-major buffer
// (X[i*cols + t], Y indexed per opY). For col-major NPU descriptors the host
// input buffer must be packed in column-major layout so that the kernel's
// strided index formula (X[t*ldx + i] with ldx=rows) reads the correct logical
// element. This helper permutes a row-major buffer of shape (rows × cols) into
// the equivalent col-major buffer (same element count, different stride).
//
//   row-major:  buf[i * cols + j]   (ld = cols)
//   col-major:  buf[j * rows + i]   (ld = rows)
//
// The element count (rows*cols) is unchanged; only the in-memory ordering
// differs. Quantization (FP32/FP16 cast) is per-element and order-independent,
// so repacking after quantization is equivalent to repacking before.
// ============================================================================
template <typename T>
static std::vector<T> RepackRowToCol(const std::vector<T>& rowBuf,
                                      int64_t rows, int64_t cols) {
    if (rows <= 0 || cols <= 0) {
        return rowBuf;
    }
    std::vector<T> colBuf(rowBuf.size());
    for (int64_t i = 0; i < rows; ++i) {
        for (int64_t j = 0; j < cols; ++j) {
            colBuf[static_cast<size_t>(j) * static_cast<size_t>(rows) +
                   static_cast<size_t>(i)] =
                rowBuf[static_cast<size_t>(i) * static_cast<size_t>(cols) +
                       static_cast<size_t>(j)];
        }
    }
    return colBuf;
}

// ============================================================================
// GTest parameterized fixture: SddmmTest (CSV-driven functional cases)
// ============================================================================

class SddmmTest : public testing::TestWithParam<SddmmTestParam> {
public:
    static void SetUpTestSuite() {
        env_ = std::make_unique<AclEnvScope>();
    }

    static void TearDownTestSuite() {
        env_.reset();
    }

protected:
    inline static std::unique_ptr<AclEnvScope> env_;
    SddmmTestParam param_;
    aclrtStream stream_ = nullptr;

    void SetUp() override {
        param_ = GetParam();
        stream_ = env_->stream();
    }
};

// ============================================================================
// Test-body helpers (file-scope). Each helper covers one logical step of the
// TEST_P body so the body stays below the R7 NBNC limit.
//
// Note: ASSERT_* must remain in the TEST_P body — gtest ASSERT macros return
// from the current function, so they cannot abort the test if placed in a
// helper. EXPECT_* (non-aborting) is safe inside helpers.
// ============================================================================

// Aggregate of generated inputs shared by golden and NPU paths.
struct SddmmInputs {
    SddmmCsr csrC;
    std::vector<double> Xf64;
    std::vector<double> Yf64;
    int64_t nnz = 0;
};

// Step 1+2: deterministic sparsity pattern + X (m×k) / Y (n×k) FP64 buffers.
static SddmmInputs GenerateSddmmInputs(const SddmmTestParam& p) {
    SddmmInputs in;
    in.csrC = MakeSddmmSparsity(p.m, p.n, p.sparsity_ratio,
                                p.value_lo, p.value_hi, p.random_seed);
    in.nnz = in.csrC.nnz;

    std::mt19937 rngX(p.random_seed + 1);
    std::mt19937 rngY(p.random_seed + 2);
    std::uniform_real_distribution<double> dist(p.value_lo, p.value_hi);

    in.Xf64.resize(static_cast<size_t>(p.m) * static_cast<size_t>(p.k));
    in.Yf64.resize(static_cast<size_t>(p.n) * static_cast<size_t>(p.k));
    for (size_t i = 0; i < in.Xf64.size(); i++) {
        in.Xf64[i] = dist(rngX);
    }
    for (size_t i = 0; i < in.Yf64.size(); i++) {
        in.Yf64[i] = dist(rngY);
    }
    return in;
}

// Step 3: FP64 golden reference.
//
// For FP16 cases the NPU input is quantized to FP16 (see DoublesToFp16 in the
// NPU stage). Align the golden reference to the same quantized inputs so the
// comparison reflects kernel computation precision rather than input
// quantization error. The C initial values (beta * C term) are quantized
// too. FP32 cases keep the original FP64 inputs, so the FP32 golden path is
// unaffected.
static std::vector<double> ComputeGoldenExpect(const SddmmTestParam& p,
                                                const SddmmCsr& csrC,
                                                const std::vector<double>& Xf64,
                                                const std::vector<double>& Yf64,
                                                aclDataType dtype,
                                                aclsparseOperation_t opX,
                                                aclsparseOperation_t opY) {
    std::vector<double> goldenX = Xf64;
    std::vector<double> goldenY = Yf64;
    SddmmCsr goldenCsr = csrC;
    double goldenAlpha = p.alpha;
    double goldenBeta = p.beta;
    if (dtype == ACL_FLOAT16) {
        goldenX = QuantizeDoublesViaFp16(Xf64);
        goldenY = QuantizeDoublesViaFp16(Yf64);
        goldenCsr.values = QuantizeDoublesViaFp16(csrC.values);
        // alpha/beta are passed as half when computeType=ACL_FLOAT16 (ACL
        // convention). Quantize through FP16 round-trip so the golden uses the
        // same precision the NPU actually receives.
        goldenAlpha = static_cast<double>(Fp16BitsToFp32(Fp32ToFp16Bits(static_cast<float>(p.alpha))));
        goldenBeta = static_cast<double>(Fp16BitsToFp32(Fp32ToFp16Bits(static_cast<float>(p.beta))));
    }
    return SddmmGolden(p.m, p.n, p.k, goldenX, goldenY, goldenCsr,
                       goldenAlpha, goldenBeta, opX, opY);
}

// Step 5: NPU three-stage workflow.
//
// The golden reference always computes from the canonical row-major Xf64/Yf64
// buffers — order is irrelevant to the mathematical result. The NPU input,
// however, must be packed in the layout declared by orderX/orderY so the
// kernel's contiguous/strided index formula reads the correct logical element.
// Repack hX/hY from row-major to col-major when the corresponding order is COL
// (the element count is unchanged, only the stride differs). hCInit (CSR
// values) is layout-independent (no repack).
static SddmmNpuResult RunNpuSddmm(const SddmmTestParam& p, aclrtStream stream,
                                   const SddmmCsr& csrC,
                                   const std::vector<double>& Xf64,
                                   const std::vector<double>& Yf64,
                                   aclDataType dtype, aclDataType computeType,
                                   aclsparseSDDMMAlg_t alg,
                                   aclsparseOperation_t opX, aclsparseOperation_t opY,
                                   aclsparseOrder_t orderX, aclsparseOrder_t orderY,
                                   int64_t nnz) {
    HandleManager handle;
    float alphaF = static_cast<float>(p.alpha);
    float betaF = static_cast<float>(p.beta);
    const bool xTransposed = (opX == ACL_SPARSE_OP_TRANSPOSE);
    const int64_t xRows = xTransposed ? p.k : p.m;
    const int64_t xCols = xTransposed ? p.m : p.k;
    const bool yTransposed = (opY == ACL_SPARSE_OP_TRANSPOSE);
    const int64_t yRows = yTransposed ? p.n : p.k;
    const int64_t yCols = yTransposed ? p.k : p.n;

    SddmmNpuResult npuResult;
    if (dtype == ACL_FLOAT) {
        auto hCInit = DoublesToFp32(csrC.values);
        auto hX = DoublesToFp32(Xf64);
        auto hY = DoublesToFp32(Yf64);
        if (orderX == ACL_SPARSE_ORDER_COL) {
            hX = RepackRowToCol(hX, xRows, xCols);
        }
        if (orderY == ACL_SPARSE_ORDER_COL) {
            hY = RepackRowToCol(hY, yRows, yCols);
        }
        npuResult = SddmmNpu<float>(handle, stream, p.m, p.n, p.k,
                                     opX, opY, alphaF, betaF,
                                     dtype, computeType, alg, orderX, orderY,
                                     csrC.rowOffsets, csrC.colIndices,
                                     hCInit, hX, hY, nnz);
    } else {
        auto hCInit = DoublesToFp16(csrC.values);
        auto hX = DoublesToFp16(Xf64);
        auto hY = DoublesToFp16(Yf64);
        if (orderX == ACL_SPARSE_ORDER_COL) {
            hX = RepackRowToCol(hX, xRows, xCols);
        }
        if (orderY == ACL_SPARSE_ORDER_COL) {
            hY = RepackRowToCol(hY, yRows, yCols);
        }
        npuResult = SddmmNpu<uint16_t>(handle, stream, p.m, p.n, p.k,
                                        opX, opY, alphaF, betaF,
                                        dtype, computeType, alg, orderX, orderY,
                                        csrC.rowOffsets, csrC.colIndices,
                                        hCInit, hX, hY, nnz);
    }
    return npuResult;
}

// Step 7: precision verification (MIXED_TOLERANCE, dtype-driven).
// Caller must guarantee npuResult.valuesOut.size() == nnz before invoking.
// FP16 kernel saturates intermediate FP32 result to [-65504, 65504] before
// Cast<float->half> (sddmm_kernel.cpp L293-301). Mirror this saturation in the
// golden so whitebox saturation-path cases (WB_08) align with NPU output. FP32
// path is unaffected (no saturation).
static void VerifySddmmPrecision(const std::string& caseName,
                                  const SddmmNpuResult& npuResult,
                                  const std::vector<double>& expectFp64,
                                  aclDataType dtype, int64_t nnz) {
    if (nnz <= 0) {
        std::cout << "[" << caseName << "] nnz=0 (empty tensor), skip value verify\n";
        return;
    }
    std::vector<float> goldenFloat(static_cast<size_t>(nnz));
    std::vector<float> npuFloat(static_cast<size_t>(nnz));
    for (int64_t i = 0; i < nnz; i++) {
        float gv = static_cast<float>(expectFp64[static_cast<size_t>(i)]);
        if (dtype == ACL_FLOAT16) {
            if (gv > 65504.0f) { gv = 65504.0f; }
            if (gv < -65504.0f) { gv = -65504.0f; }
        }
        goldenFloat[static_cast<size_t>(i)] = gv;
        npuFloat[static_cast<size_t>(i)] =
            static_cast<float>(npuResult.valuesOut[static_cast<size_t>(i)]);
    }
    // Framework MIXED_TOLERANCE mode: dtype-driven atol/rtol + per-element
    // max(abs_err) limit = max(fixedValue, 32*ULP), requiredMatchedRatio=0.99.
    VerifyConfig cfg;
    applyMixedTolerance(cfg, dtype, goldenFloat.data(), goldenFloat.size());
    bool pass = Verifier::verifyVector(npuFloat.data(), goldenFloat.data(),
                                       static_cast<size_t>(nnz), 1, cfg, caseName);
    EXPECT_TRUE(pass) << "MIXED_TOLERANCE verification failed for " << caseName;
}

// Test body: success-path parameterized test
TEST_P(SddmmTest, SddmmFunctional) {
    const auto& p = param_;

    std::cout << "==== " << p.case_name << " ====" << p.description
              << " m=" << p.m << " k=" << p.k << " n=" << p.n
              << " ratio=" << p.sparsity_ratio
              << " alpha=" << p.alpha << " beta=" << p.beta
              << " dtype=" << p.dtype
              << " order_x=" << p.order_x << " order_y=" << p.order_y
              << " value_range=[" << p.value_lo << "," << p.value_hi << "]"
              << " seed=" << p.random_seed << "\n";

    // --- Step 1+2: deterministic sparsity pattern + X/Y (FP64) ---------------
    SddmmInputs inputs = GenerateSddmmInputs(p);
    int64_t nnz = inputs.nnz;
    std::cout << "  nnz=" << nnz << "\n";

    // --- Step 3: parse dtype / op / order enums + golden reference -----------
    aclDataType dtype = ParseDtype(p.dtype);
    aclsparseOperation_t opX = ParseOperation(p.op_x);
    aclsparseOperation_t opY = ParseOperation(p.op_y);
    aclsparseOrder_t orderX = ParseOrder(p.order_x);
    aclsparseOrder_t orderY = ParseOrder(p.order_y);
    std::vector<double> expectFp64 = ComputeGoldenExpect(p, inputs.csrC, inputs.Xf64,
                                                          inputs.Yf64, dtype, opX, opY);

    // --- Step 4: parse remaining ACL enums -----------------------------------
    aclDataType computeType = ParseDtype(p.compute_type);
    aclsparseSDDMMAlg_t alg = ParseSddmmAlg(p.alg);

    // --- Step 5: NPU three-stage workflow ------------------------------------
    SddmmNpuResult npuResult = RunNpuSddmm(p, stream_, inputs.csrC, inputs.Xf64,
                                            inputs.Yf64, dtype, computeType, alg,
                                            opX, opY, orderX, orderY, nnz);

    // --- Step 6: check return codes ------------------------------------------
    ASSERT_EQ(npuResult.bufferSizeRet, ACL_SPARSE_STATUS_SUCCESS)
        << "BufferSize stage failed";
    ASSERT_EQ(npuResult.preprocessRet, ACL_SPARSE_STATUS_SUCCESS)
        << "Preprocess stage failed";
    ASSERT_EQ(npuResult.executeRet, ACL_SPARSE_STATUS_SUCCESS)
        << "Execute stage failed";

    // --- Step 7: precision verification (MIXED_TOLERANCE, dtype-driven) ------
    if (nnz > 0) {
        ASSERT_EQ(npuResult.valuesOut.size(), static_cast<size_t>(nnz));
    }
    VerifySddmmPrecision(p.case_name, npuResult, expectFp64, dtype, nnz);

    std::cout << "[" << p.case_name << "] PASSED (nnz=" << nnz << ")\n";
}

// Parameterized test instantiation from CSV (loaded via csv_loader.h)
INSTANTIATE_TEST_SUITE_P(
    SddmmCases,
    SddmmTest,
    testing::ValuesIn(GetCasesFromCsv<SddmmTestParam>("sddmm_test.csv")),
    [](const testing::TestParamInfo<SddmmTestParam>& info) {
        return info.param.case_name;
    }
);

// ============================================================================
// Exception test fixture: SddmmExceptionTest (E1-E9, parameter validation)
// Tests are TEST_F (not CSV-driven); expect non-SUCCESS error codes.
// ============================================================================

class SddmmExceptionTest : public testing::Test {
public:
    static void SetUpTestSuite() {
        env_ = std::make_unique<AclEnvScope>();
    }

    static void TearDownTestSuite() {
        env_.reset();
    }

protected:
    inline static std::unique_ptr<AclEnvScope> env_;

    aclrtStream stream_ = nullptr;
    std::unique_ptr<HandleManager> handle_;
    SddmmCsr csrC_;
    int nnz_ = 0;

    // RAII device buffers and descriptors (auto-released on fixture destruction)
    DeviceBuffer dRowOff_;
    DeviceBuffer dColInd_;
    DeviceBuffer dVals_;
    DeviceBuffer dX_;
    DeviceBuffer dY_;
    DeviceBuffer dBuffer_;
    DnMatManager matX_;
    DnMatManager matY_;
    SpMatManager matC_;

    float alpha_ = 1.0f;
    float beta_ = 0.0f;
    size_t bufSize_ = 0;

    void SetUp() override {
        stream_ = env_->stream();
        handle_ = std::make_unique<HandleManager>();
        handle_->setStream(stream_);

        // Small 4×4 CSR for valid-parameter baseline
        csrC_ = MakeSddmmSparsity(4, 4, 0.5, -1.0, 1.0, 42);
        nnz_ = static_cast<int>(csrC_.nnz);

        std::vector<float> xFp32(16, 1.0f);
        std::vector<float> yFp32(16, 1.0f);
        std::vector<float> cInitFp32(nnz_);
        for (int i = 0; i < nnz_; i++) {
            cInitFp32[static_cast<size_t>(i)] = static_cast<float>(csrC_.values[static_cast<size_t>(i)]);
        }

        dRowOff_ = DeviceBuffer::copyFrom(csrC_.rowOffsets.data(), 5 * sizeof(int32_t));
        if (nnz_ > 0) {
            dColInd_ = DeviceBuffer::copyFrom(csrC_.colIndices.data(),
                                              static_cast<size_t>(nnz_) * sizeof(int32_t));
            dVals_ = DeviceBuffer::copyFrom(cInitFp32.data(),
                                            static_cast<size_t>(nnz_) * sizeof(float));
        }
        dX_ = DeviceBuffer::copyFrom(xFp32.data(), 16 * sizeof(float));
        dY_ = DeviceBuffer::copyFrom(yFp32.data(), 16 * sizeof(float));

        // X/Y: 4×4 row-major const dense; C: 4×4 CSR (FP32)
        matX_ = DnMatManager::createConst(4, 4, 4, dX_.raw(), ACL_FLOAT, ACL_SPARSE_ORDER_ROW);
        matY_ = DnMatManager::createConst(4, 4, 4, dY_.raw(), ACL_FLOAT, ACL_SPARSE_ORDER_ROW);
        matC_ = SpMatManager::createCsr(4, 4, nnz_, dRowOff_.get(), dColInd_.get(),
                                        dVals_.get(), ACL_SPARSE_INDEX_32I,
                                        ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_BASE_ZERO,
                                        ACL_FLOAT);

        // Query buffer size for the valid baseline (may fail if operator not
        // yet implemented; defensive fallback keeps E7 testable).
        bufSize_ = 0;
        auto ret = aclsparseSDDMMBufferSize(
            handle_->get(), ACL_SPARSE_OP_NON_TRANSPOSE, ACL_SPARSE_OP_NON_TRANSPOSE,
            &alpha_, matX_.cget(), matY_.cget(), &beta_, matC_.get(),
            ACL_FLOAT, ACL_SPARSE_SDDMM_ALG_DEFAULT, &bufSize_);
        if (ret != ACL_SPARSE_STATUS_SUCCESS || bufSize_ == 0) {
            bufSize_ = 16;
        }
        dBuffer_ = DeviceBuffer::alloc(bufSize_);
    }
};

// E1: null handle -> ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR
TEST_F(SddmmExceptionTest, NullHandle) {
    size_t bs = 0;
    auto ret = aclsparseSDDMMBufferSize(
        nullptr, ACL_SPARSE_OP_NON_TRANSPOSE, ACL_SPARSE_OP_NON_TRANSPOSE,
        &alpha_, matX_.cget(), matY_.cget(), &beta_, matC_.get(),
        ACL_FLOAT, ACL_SPARSE_SDDMM_ALG_DEFAULT, &bs);
    EXPECT_EQ(ret, ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR);
}

// E2: null matX -> ACL_SPARSE_STATUS_INVALID_VALUE
TEST_F(SddmmExceptionTest, NullMatX) {
    size_t bs = 0;
    auto ret = aclsparseSDDMMBufferSize(
        handle_->get(), ACL_SPARSE_OP_NON_TRANSPOSE, ACL_SPARSE_OP_NON_TRANSPOSE,
        &alpha_, nullptr, matY_.cget(), &beta_, matC_.get(),
        ACL_FLOAT, ACL_SPARSE_SDDMM_ALG_DEFAULT, &bs);
    EXPECT_EQ(ret, ACL_SPARSE_STATUS_INVALID_VALUE);
}

// E3: null matY -> ACL_SPARSE_STATUS_INVALID_VALUE
TEST_F(SddmmExceptionTest, NullMatY) {
    size_t bs = 0;
    auto ret = aclsparseSDDMMBufferSize(
        handle_->get(), ACL_SPARSE_OP_NON_TRANSPOSE, ACL_SPARSE_OP_NON_TRANSPOSE,
        &alpha_, matX_.cget(), nullptr, &beta_, matC_.get(),
        ACL_FLOAT, ACL_SPARSE_SDDMM_ALG_DEFAULT, &bs);
    EXPECT_EQ(ret, ACL_SPARSE_STATUS_INVALID_VALUE);
}

// E4: null matC -> ACL_SPARSE_STATUS_INVALID_VALUE
TEST_F(SddmmExceptionTest, NullMatC) {
    size_t bs = 0;
    auto ret = aclsparseSDDMMBufferSize(
        handle_->get(), ACL_SPARSE_OP_NON_TRANSPOSE, ACL_SPARSE_OP_NON_TRANSPOSE,
        &alpha_, matX_.cget(), matY_.cget(), &beta_, nullptr,
        ACL_FLOAT, ACL_SPARSE_SDDMM_ALG_DEFAULT, &bs);
    EXPECT_EQ(ret, ACL_SPARSE_STATUS_INVALID_VALUE);
}

// E5: null alpha -> ACL_SPARSE_STATUS_INVALID_VALUE
TEST_F(SddmmExceptionTest, NullAlpha) {
    size_t bs = 0;
    auto ret = aclsparseSDDMMBufferSize(
        handle_->get(), ACL_SPARSE_OP_NON_TRANSPOSE, ACL_SPARSE_OP_NON_TRANSPOSE,
        nullptr, matX_.cget(), matY_.cget(), &beta_, matC_.get(),
        ACL_FLOAT, ACL_SPARSE_SDDMM_ALG_DEFAULT, &bs);
    EXPECT_EQ(ret, ACL_SPARSE_STATUS_INVALID_VALUE);
}

// E6: null beta -> ACL_SPARSE_STATUS_INVALID_VALUE
TEST_F(SddmmExceptionTest, NullBeta) {
    size_t bs = 0;
    auto ret = aclsparseSDDMMBufferSize(
        handle_->get(), ACL_SPARSE_OP_NON_TRANSPOSE, ACL_SPARSE_OP_NON_TRANSPOSE,
        &alpha_, matX_.cget(), matY_.cget(), nullptr, matC_.get(),
        ACL_FLOAT, ACL_SPARSE_SDDMM_ALG_DEFAULT, &bs);
    EXPECT_EQ(ret, ACL_SPARSE_STATUS_INVALID_VALUE);
}

// E7: null externalBuffer (bufferSize > 0) -> ACL_SPARSE_STATUS_INVALID_VALUE
//     Tested at Execute stage (BufferSize only queries size).
TEST_F(SddmmExceptionTest, NullExternalBuffer) {
    auto ret = aclsparseSDDMM(
        handle_->get(), ACL_SPARSE_OP_NON_TRANSPOSE, ACL_SPARSE_OP_NON_TRANSPOSE,
        &alpha_, matX_.cget(), matY_.cget(), &beta_, matC_.get(),
        ACL_FLOAT, ACL_SPARSE_SDDMM_ALG_DEFAULT, nullptr);
    EXPECT_EQ(ret, ACL_SPARSE_STATUS_INVALID_VALUE);
}

// E8: dimension mismatch (X.k != Y.k) -> MATRIX_TYPE_NOT_SUPPORTED or INVALID_VALUE
//     Exact code is implementation-defined (test plan §4.3 / Q4).
TEST_F(SddmmExceptionTest, DimMismatch) {
    // Y with k=2 (mismatches X k=4): n=4, k=2 row-major
    std::vector<float> ySmall(8, 1.0f);
    auto dYSmall = DeviceBuffer::copyFrom(ySmall.data(), 8 * sizeof(float));
    auto matYSmall = DnMatManager::createConst(4, 2, 2, dYSmall.raw(),
                                               ACL_FLOAT, ACL_SPARSE_ORDER_ROW);
    size_t bs = 0;
    auto ret = aclsparseSDDMMBufferSize(
        handle_->get(), ACL_SPARSE_OP_NON_TRANSPOSE, ACL_SPARSE_OP_NON_TRANSPOSE,
        &alpha_, matX_.cget(), matYSmall.cget(), &beta_, matC_.get(),
        ACL_FLOAT, ACL_SPARSE_SDDMM_ALG_DEFAULT, &bs);
    EXPECT_TRUE(ret == ACL_SPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED ||
                ret == ACL_SPARSE_STATUS_INVALID_VALUE)
        << "expected MATRIX_TYPE_NOT_SUPPORTED or INVALID_VALUE, got " << ret;
}

// E9: unsupported computeType (ACL_DOUBLE) -> ACL_SPARSE_STATUS_NOT_SUPPORTED
TEST_F(SddmmExceptionTest, UnsupportedComputeType) {
    size_t bs = 0;
    auto ret = aclsparseSDDMMBufferSize(
        handle_->get(), ACL_SPARSE_OP_NON_TRANSPOSE, ACL_SPARSE_OP_NON_TRANSPOSE,
        &alpha_, matX_.cget(), matY_.cget(), &beta_, matC_.get(),
        ACL_DOUBLE, ACL_SPARSE_SDDMM_ALG_DEFAULT, &bs);
    EXPECT_EQ(ret, ACL_SPARSE_STATUS_NOT_SUPPORTED);
}

// E10: null CSR ptrs (rowOff) -> ACL_SPARSE_STATUS_INVALID_VALUE
//      Kernel reads ptrs[m] for nnz unconditionally, so ptrs must never be null.
TEST_F(SddmmExceptionTest, NullCsrPtrs) {
    auto matCNullPtrs = SpMatManager::createCsr(4, 4, 0, nullptr, dColInd_.get(),
                                                 dVals_.get(), ACL_SPARSE_INDEX_32I,
                                                 ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_BASE_ZERO,
                                                 ACL_FLOAT);
    size_t bs = 0;
    auto ret = aclsparseSDDMMBufferSize(
        handle_->get(), ACL_SPARSE_OP_NON_TRANSPOSE, ACL_SPARSE_OP_NON_TRANSPOSE,
        &alpha_, matX_.cget(), matY_.cget(), &beta_, matCNullPtrs.get(),
        ACL_FLOAT, ACL_SPARSE_SDDMM_ALG_DEFAULT, &bs);
    EXPECT_EQ(ret, ACL_SPARSE_STATUS_INVALID_VALUE);
}
