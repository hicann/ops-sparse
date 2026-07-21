/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * @file gtsv2_nopivot_test.cpp
 * @brief GTest + CSV-driven test cases for aclsparseSgtsv2Nopivot (Legacy API).
 *
 * Tests the tridiagonal solver without pivoting: A * X = B
 * where A is a tridiagonal matrix defined by diagonals (dl, d, du).
 * B is an m x n dense matrix in column-major layout.
 * Solution X overwrites B in-place.
 *
 * Test structure:
 *   - TEST_P (Gtsv2NopivotTest)         : parameterized success-path tests from CSV
 *   - TEST_F (Gtsv2NopivotExceptionTest): null-pointer / invalid-param error tests
 *
 * Test parameters are loaded from gtsv2_nopivot_test.csv. Entry point is shared
 * via test/frame/test_main.cpp.
 */

#include "test_common.h"

#include "../gtsv2_nopivot_golden.h"
#include "../gtsv2_nopivot_param.h"
#include "gtsv2_nopivot_npu_wrapper.h"

using namespace sparse_test;

// ============================================================================
// Diagonal matrix generation (参照 gtsv_interleaved_batch 模式)
// ============================================================================
static TridiagMatrix GenerateTridiagMatrix(const Gtsv2NopivotParam& p) {
    if (p.is_diag_dominant) {
        return makeDiagDominantTridiag(p.m, 1, p.seed);
    }
    double loRange = p.value_hi - p.value_lo;
    if (loRange < 1e-10) {
        float diagVal = static_cast<float>(p.value_lo);
        return makeConstantDiagTridiag(p.m, 1, diagVal);
    }
    return makeRandomTridiag(p.m, 1, p.value_lo, p.value_hi, p.seed);
}

// ============================================================================
// Helper: Print test case info
// ============================================================================
static void PrintCaseInfo(const Gtsv2NopivotParam& p) {
    std::cout << "==== " << p.case_name
              << " ==== m=" << p.m << " n=" << p.n << " ldb=" << p.ldb
              << " diag_dom=" << p.is_diag_dominant << " seed=" << p.seed << "\n";
}

// ============================================================================
// GTest parameterized fixture: Gtsv2NopivotTest
// ============================================================================
class Gtsv2NopivotTest : public testing::TestWithParam<Gtsv2NopivotParam> {
public:
    static void SetUpTestSuite() {
        env_ = std::make_unique<AclEnvScope>();
        spHandle_ = std::make_unique<HandleManager>();
        spHandle_->setStream(env_->stream());
    }

    static void TearDownTestSuite() {
        spHandle_.reset();
        env_.reset();
    }

protected:
    inline static std::unique_ptr<AclEnvScope> env_;
    inline static std::unique_ptr<HandleManager> spHandle_;

    Gtsv2NopivotParam param_;

    void SetUp() override {
        param_ = GetParam();
    }
};

// ============================================================================
// Test body: Success-path parameterized test
// ============================================================================
TEST_P(Gtsv2NopivotTest, Gtsv2NopivotSuccess) {
    const auto& p = param_;
    PrintCaseInfo(p);

    ASSERT_EQ(p.expect_result, "SUCCESS");

    int m = p.m;
    int n = p.n;
    int ldb = p.ldb;

    // 1. Generate tridiagonal matrix
    auto tri = GenerateTridiagMatrix(p);

    // 2. Generate RHS matrix B (column-major, ldb x n)
    //    Use RHS values in range -5.0 to 10.0
    auto B_host = makeFullColMajor(m, n, ldb, p.seed + 100, -5.0, 10.0);

    // 3. Golden reference: FP64 Thomas algorithm (in-place on B copy)
    auto B_golden = B_host;
    Gtsv2NopivotGolden(tri.dl, tri.d, tri.du, B_golden, m, n, ldb);

    // 4. NPU solve (in-place)
    auto npuResult = Gtsv2NopivotNpu(*spHandle_, env_->stream(),
                                      tri.dl, tri.d, tri.du,
                                      B_host, m, n, ldb);

    ASSERT_EQ(npuResult.bufferSizeRet, ACL_SPARSE_STATUS_SUCCESS);
    ASSERT_EQ(npuResult.computeRet, ACL_SPARSE_STATUS_SUCCESS);

    // 5. Verify: compare NPU solution against golden
    VerifyConfig cfg;
    cfg.SetMode(PrecisionMode::MERE_MARE)
       .SetMERE(p.mere_threshold)
       .SetMARE(p.mare_multiplier);

    // Only compare the m x n valid region (within ldb stride)
    // Extract the valid portion from column-major layout
    std::vector<float> X_npu, X_golden;
    X_npu.reserve(m * n);
    X_golden.reserve(m * n);
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < m; i++) {
            X_npu.push_back(npuResult.X[i + j * ldb]);
            X_golden.push_back(B_golden[i + j * ldb]);
        }
    }

    bool pass = Verifier::verifyVector(X_npu, X_golden, cfg, p.case_name);
    EXPECT_TRUE(pass);

    std::cout << "[" << p.case_name << "] PASSED\n";
}

// ============================================================================
// Parameterized test instantiation from CSV
// ============================================================================
INSTANTIATE_TEST_SUITE_P(
    Gtsv2NopivotCases,
    Gtsv2NopivotTest,
    testing::ValuesIn(GetCasesFromCsv<Gtsv2NopivotParam>("gtsv2_nopivot_test.csv")),
    [](const testing::TestParamInfo<Gtsv2NopivotParam>& info) {
        return info.param.case_name;
    }
);

// ============================================================================
// Exception test fixture: Gtsv2NopivotExceptionTest
// ============================================================================
class Gtsv2NopivotExceptionTest : public testing::Test {
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
    std::vector<float> dl_, d_, du_, B_;
    int m_ = 4;
    int n_ = 1;
    int ldb_ = 4;

    void SetUp() override {
        stream_ = env_->stream();
        handle_ = std::make_unique<HandleManager>();
        handle_->setStream(stream_);

        // Default valid diagonals: diagonal-dominant 4x4
        auto tri = makeDiagDominantTridiag(m_, 1, 100);
        dl_ = tri.dl;
        d_  = tri.d;
        du_ = tri.du;

        // Default valid RHS
        B_ = makeFullColMajor(m_, n_, ldb_, 200, -5.0, 10.0);
    }

    void TearDown() override {
        handle_.reset();
    }

    // Helper: call compute API with specified parameters (some may be nullptr/invalid)
    aclsparseStatus_t CallCompute(aclsparseHandle_t handle,
                                  int m, int n,
                                  const float* dl,
                                  const float* d,
                                  const float* du,
                                  float* B,
                                  int ldb,
                                  void* pBuffer) {
        return aclsparseSgtsv2Nopivot(
            handle, m, n, dl, d, du, B, ldb, pBuffer);
    }

    // Helper: call bufferSizeExt API with specified parameters
    aclsparseStatus_t CallBufferSize(aclsparseHandle_t handle,
                                     int m, int n,
                                     const float* dl,
                                     const float* d,
                                     const float* du,
                                     const float* B,
                                     int ldb) {
        size_t bufSize = 0;
        return aclsparseSgtsv2Nopivot_bufferSizeExt(
            handle, m, n, dl, d, du, B, ldb, &bufSize);
    }
};

// ============================================================================
// Exception Tests: Null handle / invalid params via bufferSizeExt
// bufferSizeExt is used for validation since it is a pure host-side function
// and does not require device-side resources for parameter validation.
// ============================================================================

// E1: NULL handle -> ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR
TEST_F(Gtsv2NopivotExceptionTest, NullHandle) {
    EXPECT_EQ(CallBufferSize(nullptr, m_, n_,
                              dl_.data(), d_.data(), du_.data(), B_.data(), ldb_),
              ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR);
}

// E2: m = 2 (< 3) -> ACL_SPARSE_STATUS_INVALID_VALUE
TEST_F(Gtsv2NopivotExceptionTest, MTooSmall_2) {
    EXPECT_EQ(CallBufferSize(handle_->get(), 2, n_,
                              dl_.data(), d_.data(), du_.data(), B_.data(), ldb_),
              ACL_SPARSE_STATUS_INVALID_VALUE);
}

// E3: m = 1 (< 3) -> ACL_SPARSE_STATUS_INVALID_VALUE
TEST_F(Gtsv2NopivotExceptionTest, MTooSmall_1) {
    EXPECT_EQ(CallBufferSize(handle_->get(), 1, n_,
                              dl_.data(), d_.data(), du_.data(), B_.data(), ldb_),
              ACL_SPARSE_STATUS_INVALID_VALUE);
}

// E4: m = 0 (< 3) -> ACL_SPARSE_STATUS_INVALID_VALUE
TEST_F(Gtsv2NopivotExceptionTest, MZero) {
    EXPECT_EQ(CallBufferSize(handle_->get(), 0, n_,
                              dl_.data(), d_.data(), du_.data(), B_.data(), ldb_),
              ACL_SPARSE_STATUS_INVALID_VALUE);
}

// E5: m = -1 (< 3) -> ACL_SPARSE_STATUS_INVALID_VALUE
TEST_F(Gtsv2NopivotExceptionTest, MNegative) {
    EXPECT_EQ(CallBufferSize(handle_->get(), -1, n_,
                              dl_.data(), d_.data(), du_.data(), B_.data(), ldb_),
              ACL_SPARSE_STATUS_INVALID_VALUE);
}

// E6: n = 0 (< 1) -> ACL_SPARSE_STATUS_INVALID_VALUE
TEST_F(Gtsv2NopivotExceptionTest, NZero) {
    EXPECT_EQ(CallBufferSize(handle_->get(), m_, 0,
                              dl_.data(), d_.data(), du_.data(), B_.data(), ldb_),
              ACL_SPARSE_STATUS_INVALID_VALUE);
}

// E7: n = -1 (< 1) -> ACL_SPARSE_STATUS_INVALID_VALUE
TEST_F(Gtsv2NopivotExceptionTest, NNegative) {
    EXPECT_EQ(CallBufferSize(handle_->get(), m_, -1,
                              dl_.data(), d_.data(), du_.data(), B_.data(), ldb_),
              ACL_SPARSE_STATUS_INVALID_VALUE);
}

// E8: dl = nullptr -> ACL_SPARSE_STATUS_INVALID_VALUE
TEST_F(Gtsv2NopivotExceptionTest, NullDl) {
    EXPECT_EQ(CallBufferSize(handle_->get(), m_, n_,
                              nullptr, d_.data(), du_.data(), B_.data(), ldb_),
              ACL_SPARSE_STATUS_INVALID_VALUE);
}

// E9: d = nullptr -> ACL_SPARSE_STATUS_INVALID_VALUE
TEST_F(Gtsv2NopivotExceptionTest, NullD) {
    EXPECT_EQ(CallBufferSize(handle_->get(), m_, n_,
                              dl_.data(), nullptr, du_.data(), B_.data(), ldb_),
              ACL_SPARSE_STATUS_INVALID_VALUE);
}

// E10: du = nullptr -> ACL_SPARSE_STATUS_INVALID_VALUE
TEST_F(Gtsv2NopivotExceptionTest, NullDu) {
    EXPECT_EQ(CallBufferSize(handle_->get(), m_, n_,
                              dl_.data(), d_.data(), nullptr, B_.data(), ldb_),
              ACL_SPARSE_STATUS_INVALID_VALUE);
}

// E11: B = nullptr -> ACL_SPARSE_STATUS_INVALID_VALUE
TEST_F(Gtsv2NopivotExceptionTest, NullB) {
    EXPECT_EQ(CallBufferSize(handle_->get(), m_, n_,
                              dl_.data(), d_.data(), du_.data(), nullptr, ldb_),
              ACL_SPARSE_STATUS_INVALID_VALUE);
}

// E12: ldb = 0 (< max(1, m)) -> ACL_SPARSE_STATUS_INVALID_VALUE
TEST_F(Gtsv2NopivotExceptionTest, LdbZero) {
    EXPECT_EQ(CallBufferSize(handle_->get(), m_, n_,
                              dl_.data(), d_.data(), du_.data(), B_.data(), 0),
              ACL_SPARSE_STATUS_INVALID_VALUE);
}

// E13: ldb = m-1 (< max(1, m)) -> ACL_SPARSE_STATUS_INVALID_VALUE
TEST_F(Gtsv2NopivotExceptionTest, LdbTooSmall) {
    EXPECT_EQ(CallBufferSize(handle_->get(), m_, m_ - 1,
                              dl_.data(), d_.data(), du_.data(), B_.data(), m_ - 1),
              ACL_SPARSE_STATUS_INVALID_VALUE);
}

// E14: pBuffer = nullptr (m > 1) -> ACL_SPARSE_STATUS_INVALID_VALUE
// This test requires the compute API (bufferSizeExt doesn't take pBuffer)
TEST_F(Gtsv2NopivotExceptionTest, NullPBuffer) {
    // Obtain valid workspace size first
    size_t bufSize = 0;
    auto ret = aclsparseSgtsv2Nopivot_bufferSizeExt(
        handle_->get(), m_, n_,
        dl_.data(), d_.data(), du_.data(), B_.data(), ldb_, &bufSize);
    ASSERT_EQ(ret, ACL_SPARSE_STATUS_SUCCESS);

    // Copy B to device for compute API call
    auto dB = DeviceBuffer::copyFrom(B_.data(), m_ * n_ * sizeof(float));
    auto dDl = DeviceBuffer::copyFrom(dl_.data(), m_ * sizeof(float));
    auto dD  = DeviceBuffer::copyFrom(d_.data(), m_ * sizeof(float));
    auto dDu = DeviceBuffer::copyFrom(du_.data(), m_ * sizeof(float));

    // Call compute with null pBuffer -> ACL_SPARSE_STATUS_INVALID_VALUE
    EXPECT_EQ(aclsparseSgtsv2Nopivot(
        handle_->get(), m_, n_,
        reinterpret_cast<const float*>(dDl.raw()),
        reinterpret_cast<const float*>(dD.raw()),
        reinterpret_cast<const float*>(dDu.raw()),
        reinterpret_cast<float*>(dB.get()),
        m_, nullptr),  // pBuffer = nullptr
        ACL_SPARSE_STATUS_INVALID_VALUE);
}

// E15: pBuffer not 128-byte aligned -> ACL_SPARSE_STATUS_INVALID_VALUE
TEST_F(Gtsv2NopivotExceptionTest, PBufferUnaligned) {
    // Obtain valid workspace size
    size_t bufSize = 0;
    auto ret = aclsparseSgtsv2Nopivot_bufferSizeExt(
        handle_->get(), m_, n_,
        dl_.data(), d_.data(), du_.data(), B_.data(), ldb_, &bufSize);
    ASSERT_EQ(ret, ACL_SPARSE_STATUS_SUCCESS);

    // Create an aligned buffer and offset by 4 bytes to break alignment
    if (bufSize == 0) bufSize = 256;
    auto alignedBuf = DeviceBuffer::alloc(bufSize + 128);

    // Copy B to device
    auto dB = DeviceBuffer::copyFrom(B_.data(), m_ * n_ * sizeof(float));
    auto dDl = DeviceBuffer::copyFrom(dl_.data(), m_ * sizeof(float));
    auto dD  = DeviceBuffer::copyFrom(d_.data(), m_ * sizeof(float));
    auto dDu = DeviceBuffer::copyFrom(du_.data(), m_ * sizeof(float));

    // Use pointer at offset 4 to break 128-byte alignment
    void* unaligned = reinterpret_cast<void*>(
        reinterpret_cast<uintptr_t>(alignedBuf.get()) + 4);

    EXPECT_EQ(aclsparseSgtsv2Nopivot(
        handle_->get(), m_, n_,
        reinterpret_cast<const float*>(dDl.raw()),
        reinterpret_cast<const float*>(dD.raw()),
        reinterpret_cast<const float*>(dDu.raw()),
        reinterpret_cast<float*>(dB.get()),
        m_, unaligned),  // unaligned pBuffer
        ACL_SPARSE_STATUS_INVALID_VALUE);
}
