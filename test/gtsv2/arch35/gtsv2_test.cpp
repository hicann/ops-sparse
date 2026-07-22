/**
 * ----------------------------------------------------------------------------------------------------------
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ----------------------------------------------------------------------------------------------------------
 */

/**
 * @file gtsv2_test.cpp
 * @brief GTest + CSV-driven test cases for aclsparseSgtsv2 (Legacy API).
 *
 * Tests the tridiagonal solver with partial pivoting: A * X = B
 * where A is a tridiagonal matrix defined by diagonals (dl, d, du).
 * B is an m x n dense matrix in column-major layout.
 * Solution X overwrites B in-place.
 *
 * Test structure:
 *   - TEST_P (Gtsv2Test)         : parameterized success/n=0/singular tests from CSV
 *   - TEST_F (Gtsv2ExceptionTest): null-pointer / invalid-param error tests
 *
 * Key differences from gtsv2_nopivot:
 *   - Partial pivoting Thomas algorithm (LAPACK dgtsv style)
 *   - n=0 returns SUCCESS (early return, not INVALID_VALUE)
 *   - Singular matrices produce Inf/NaN (verified via !std::isfinite)
 *
 * Test parameters are loaded from gtsv2_test.csv. Entry point is shared
 * via test/frame/test_main.cpp.
 */

#include "test_common.h"

#include <cmath>

#include "../gtsv2_golden.h"
#include "../gtsv2_param.h"
#include "gtsv2_npu_wrapper.h"

using namespace sparse_test;

// ============================================================================
// Seed resolution: use param seed if provided, else derive from caseId
// ============================================================================
static uint32_t DeriveSeed(const std::string& caseId) {
    uint32_t h = 0;
    for (char c : caseId) {
        h = h * 31u + static_cast<uint32_t>(c);
    }
    return (h % 100000u) + 100u;
}

static uint32_t GetSeed(const Gtsv2Param& p) {
    return (p.seed != 0) ? p.seed : DeriveSeed(p.caseId());
}

// ============================================================================
// Singular tridiagonal matrix construction (det = 0, produces Inf/NaN)
//
// m=3: d=[1,2,1], dl=[0,1,1], du=[1,1,0]  — det = 1*(2-1) - 1*(1) = 0
// m=4: d=[1,1,1,0], dl=[0,1,1,1], du=[1,1,1,0]
//        det = (1*1-1*1)*(1*0-1*1) - 1*1*1*0 = 0*(-1) - 0 = 0
//
// For isZeroDiag=true: all-zero diagonal (trivially singular)
// ============================================================================
static TridiagMatrix MakeSingularTridiag(int m, bool isZeroDiag) {
    if (isZeroDiag) {
        // All-zero diagonal — trivially singular (first pivot is 0)
        return makeConstantDiagTridiag(m, 1, 0.0f);
    }

    // Hand-written singular matrices (det = 0, verified mathematically)
    TridiagMatrix out;
    out.m = m;
    out.batchCount = 1;
    out.dl.assign(m, 0.0f);
    out.d.assign(m, 1.0f);
    out.du.assign(m, 0.0f);

    if (m == 3) {
        // det = 1*(2*1 - 1*1) - 1*(1*1 - 0) = 1*1 - 1*1 = 0
        out.d  = {1.0f, 2.0f, 1.0f};
        out.dl = {0.0f, 1.0f, 1.0f};
        out.du = {1.0f, 1.0f, 0.0f};
    } else if (m >= 4) {
        // det = (d0*d1 - du0*dl1)*(d2*d3 - du2*dl3) - d0*du1*dl2*d3
        //     = (1*1 - 1*1)*(1*0 - 1*1) - 1*1*1*0 = 0*(-1) - 0 = 0
        out.d  = {1.0f, 1.0f, 1.0f, 0.0f};
        out.dl = {0.0f, 1.0f, 1.0f, 1.0f};
        out.du = {1.0f, 1.0f, 1.0f, 0.0f};
        // Remaining rows (if m > 4): identity-like, non-singular extension
        for (int i = 4; i < m; i++) {
            out.d[i]  = 1.0f;
            out.dl[i] = 0.0f;
            out.du[i] = 0.0f;
        }
    }
    return out;
}

// ============================================================================
// White-box: crafted tridiagonal matrices for deterministic pivot-path coverage
//
// These generators (MakePivotAllSwap / MakePivotFirstOnly / MakePivotLastOnly /
// MakePivotExtremeDl / MakePivotExtremeD) produce matrices that deterministically
// trigger specific partial-pivoting branch paths in the gtsv2 kernel. They are
// gtsv2-specific test fixtures, not general-purpose matrix generators — fill.h
// provides generic tridiag generators (makeDiagDominantTridiag, makeRandomTridiag,
// etc.) but does not support deterministic pivot-path control. These local
// fixtures are kept here because extending fill.h with pivot-path-specific
// generators would couple the framework to a single algorithm's internals.
//
// Each generator produces a matrix that deterministically triggers a specific
// branch in the kernel's partial-pivoting Thomas algorithm:
//   - PIVOT_ALL_SWAP:   swap at every forward step (covers fill-in chain)
//   - PIVOT_FIRST_ONLY: swap only at step i=1, then no-swap chain
//   - PIVOT_LAST_ONLY:  no-swap chain, then swap only at last step i=m-1
//   - PIVOT_EXTREME_DL: |dl[i]| = 1e6 >> |d[i]| = 1 (extreme swap ratio)
//   - PIVOT_EXTREME_D:  |d[i]| = 1e6 >> |dl[i]| = 1 (extreme no-swap ratio)
// ============================================================================

// PIVOT_ALL_SWAP: d[i]=1, dl[i]=2, du[i]=1 (except dl[0]=0, du[m-1]=0)
// At every step i: |pivot_d| (small, from modified d') < |dl[i]|=2 -> swap.
// Non-singular (det != 0), well-conditioned for small m (m <= 16).
static TridiagMatrix MakePivotAllSwap(int m) {
    TridiagMatrix out;
    out.m = m;
    out.batchCount = 1;
    out.dl.assign(m, 0.0f);
    out.d.assign(m, 1.0f);
    out.du.assign(m, 0.0f);
    for (int i = 1; i < m; i++)  out.dl[i] = 2.0f;
    for (int i = 0; i < m - 1; i++) out.du[i] = 1.0f;
    return out;
}

// PIVOT_FIRST_ONLY: d[0]=1, dl[1]=100, du[0]=100 -> swap at step 1
// After swap: d'[1] = 100 - 0.01*1 ≈ 99.9 (large), so steps 2..m-1: |d'| >= 9 >= |dl|=1 -> no swap.
static TridiagMatrix MakePivotFirstOnly(int m) {
    TridiagMatrix out;
    out.m = m;
    out.batchCount = 1;
    out.dl.assign(m, 0.0f);
    out.d.assign(m, 10.0f);
    out.du.assign(m, 0.0f);
    out.d[0] = 1.0f;       // small first diagonal -> triggers swap at step 1
    out.dl[1] = 100.0f;    // large sub-diagonal -> |dl[1]| > |d[0]|
    out.du[0] = 100.0f;    // large super-diagonal -> d'[1] stays large after swap
    for (int i = 2; i < m; i++)     out.dl[i] = 1.0f;   // small -> no swap
    for (int i = 1; i < m - 1; i++) out.du[i] = 1.0f;
    return out;
}

// PIVOT_LAST_ONLY: d[0..m-2]=10, dl[1..m-2]=1 -> no swap (|d'| ≈ 9-10 >= 1)
// d[m-1]=1, dl[m-1]=100 -> |d'[m-2]| ≈ 9 < |dl[m-1]|=100 -> swap at last step.
static TridiagMatrix MakePivotLastOnly(int m) {
    TridiagMatrix out;
    out.m = m;
    out.batchCount = 1;
    out.dl.assign(m, 0.0f);
    out.d.assign(m, 10.0f);
    out.du.assign(m, 0.0f);
    out.d[m - 1] = 1.0f;       // small last diagonal
    out.dl[m - 1] = 100.0f;    // large last sub-diagonal -> swap at last step
    for (int i = 1; i < m - 1; i++) out.dl[i] = 1.0f;   // small -> no swap
    for (int i = 0; i < m - 1; i++) out.du[i] = 1.0f;
    return out;
}

// PIVOT_EXTREME_DL: d[i]=1, dl[i]=1e6, du[i]=1 -> extreme swap ratio
// mult = pivot_d/dl_i = 1/1e6 = 1e-6 (very small), fill-in = -1e-6 * du
static TridiagMatrix MakePivotExtremeDl(int m) {
    TridiagMatrix out;
    out.m = m;
    out.batchCount = 1;
    out.dl.assign(m, 0.0f);
    out.d.assign(m, 1.0f);
    out.du.assign(m, 0.0f);
    for (int i = 1; i < m; i++)     out.dl[i] = 1.0e6f;
    for (int i = 0; i < m - 1; i++) out.du[i] = 1.0f;
    return out;
}

// PIVOT_EXTREME_D: d[i]=1e6, dl[i]=1, du[i]=1 -> extreme no-swap ratio
// |d[i]| = 1e6 >> |dl[i]| = 1, so no swap ever triggers.
static TridiagMatrix MakePivotExtremeD(int m) {
    TridiagMatrix out;
    out.m = m;
    out.batchCount = 1;
    out.dl.assign(m, 0.0f);
    out.d.assign(m, 1.0e6f);
    out.du.assign(m, 0.0f);
    for (int i = 1; i < m; i++)     out.dl[i] = 1.0f;
    for (int i = 0; i < m - 1; i++) out.du[i] = 1.0f;
    return out;
}

// ============================================================================
// Tridiagonal matrix generation based on param fields
// ============================================================================
static TridiagMatrix GenerateTridiagMatrix(const Gtsv2Param& p) {
    uint32_t seed = GetSeed(p);

    if (p.matrixType == Gtsv2MatrixType::DIAG_DOMINANT) {
        return makeDiagDominantTridiag(p.m, 1, seed);
    }
    if (p.matrixType == Gtsv2MatrixType::IDENTITY) {
        return makeIdentityTridiag(p.m, 1);
    }
    if (p.matrixType == Gtsv2MatrixType::CONSTANT) {
        float val = static_cast<float>(p.value_lo);
        return makeConstantDiagTridiag(p.m, 1, val);
    }
    if (p.matrixType == Gtsv2MatrixType::SINGULAR) {
        bool isZeroDiag = (p.value_lo == 0.0 && p.value_hi == 0.0);
        return MakeSingularTridiag(p.m, isZeroDiag);
    }
    // White-box: deterministic pivot-path matrices
    if (p.matrixType == Gtsv2MatrixType::PIVOT_ALL_SWAP) {
        return MakePivotAllSwap(p.m);
    }
    if (p.matrixType == Gtsv2MatrixType::PIVOT_FIRST_ONLY) {
        return MakePivotFirstOnly(p.m);
    }
    if (p.matrixType == Gtsv2MatrixType::PIVOT_LAST_ONLY) {
        return MakePivotLastOnly(p.m);
    }
    if (p.matrixType == Gtsv2MatrixType::PIVOT_EXTREME_DL) {
        return MakePivotExtremeDl(p.m);
    }
    if (p.matrixType == Gtsv2MatrixType::PIVOT_EXTREME_D) {
        return MakePivotExtremeD(p.m);
    }
    // Default: random (triggers partial pivoting)
    return makeRandomTridiag(p.m, 1, p.value_lo, p.value_hi, seed);
}

// ============================================================================
// Helper: Print test case info
// ============================================================================
static void PrintCaseInfo(const Gtsv2Param& p) {
    std::cout << "==== " << p.caseId()
              << " ==== m=" << p.m << " n=" << p.n << " ldb=" << p.ldb
              << " matrixType=" << static_cast<int>(p.matrixType)
              << " value_lo=" << p.value_lo << " value_hi=" << p.value_hi
              << " rtol=" << p.rtol << " atol=" << p.atol
              << " mareMultiplier=" << p.mareMultiplier
              << " desc=" << p.description << "\n";
}

// ============================================================================
// Helper: Check if NPU output contains Inf/NaN (for singular matrix tests)
// ============================================================================
static bool HasInfOrNan(const std::vector<float>& X, int m, int n, int ldb) {
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < m; i++) {
            if (!std::isfinite(X[static_cast<int64_t>(i) + static_cast<int64_t>(j) * ldb])) {
                return true;
            }
        }
    }
    return false;
}

// ============================================================================
// GTest parameterized fixture: Gtsv2Test
// ============================================================================
class Gtsv2Test : public testing::TestWithParam<Gtsv2Param> {
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

    Gtsv2Param param_;

    void SetUp() override {
        param_ = GetParam();
    }
};

// ============================================================================
// Helper: Verify singular matrix produces Inf/NaN in both golden and NPU
// Verifies not just existence but position consistency: each element's
// isfinite status must match between golden and NPU output.
// ============================================================================
static void VerifySingularResult(const Gtsv2Param& p,
                                 const std::vector<float>& B_host,
                                 const TridiagMatrix& tri,
                                 const Gtsv2NpuResult& npuResult,
                                 int m, int n, int ldb) {
    auto B_golden = B_host;
    Gtsv2Golden(tri.dl, tri.d, tri.du, B_golden, m, n, ldb);
    bool goldenHasInfOrNan = HasInfOrNan(B_golden, m, n, ldb);
    bool npuHasInfOrNan = HasInfOrNan(npuResult.X, m, n, ldb);
    EXPECT_TRUE(npuHasInfOrNan)
        << "[" << p.caseId() << "] Singular matrix should produce Inf/NaN in NPU output";
    EXPECT_TRUE(goldenHasInfOrNan)
        << "[" << p.caseId() << "] Golden should also produce Inf/NaN for singular matrix";

    // Verify Inf/NaN position consistency: each element's isfinite status
    // must match between golden and NPU, catching wrong-position Inf/NaN bugs.
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < m; i++) {
            int64_t idx = static_cast<int64_t>(i) + static_cast<int64_t>(j) * ldb;
            bool goldenFinite = std::isfinite(B_golden[idx]);
            bool npuFinite = std::isfinite(npuResult.X[idx]);
            EXPECT_EQ(goldenFinite, npuFinite)
                << "[" << p.caseId() << "] Inf/NaN position mismatch at (" << i << "," << j
                << "): golden " << (goldenFinite ? "finite" : "Inf/NaN")
                << " vs NPU " << (npuFinite ? "finite" : "Inf/NaN");
        }
    }

    std::cout << "[" << p.caseId() << "] PASSED (singular Inf/NaN)\n";
}

// ============================================================================
// Helper: Extract valid m×n region from column-major layout and verify precision
// ============================================================================
static void VerifyPrecisionResult(const Gtsv2Param& p,
                                  const std::vector<float>& B_host,
                                  const TridiagMatrix& tri,
                                  const Gtsv2NpuResult& npuResult,
                                  int m, int n, int ldb) {
    auto B_golden = B_host;
    Gtsv2Golden(tri.dl, tri.d, tri.du, B_golden, m, n, ldb);

    VerifyConfig cfg;
    cfg.SetMode(PrecisionMode::MERE_MARE)
       .SetAbsTol(p.atol)
       .SetMERE(p.rtol)
       .SetMARE(p.mareMultiplier);

    std::vector<float> X_npu;
    std::vector<float> X_golden;
    X_npu.reserve(static_cast<size_t>(m) * n);
    X_golden.reserve(static_cast<size_t>(m) * n);
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < m; i++) {
            X_npu.push_back(npuResult.X[static_cast<int64_t>(i) + static_cast<int64_t>(j) * ldb]);
            X_golden.push_back(B_golden[static_cast<int64_t>(i) + static_cast<int64_t>(j) * ldb]);
        }
    }

    bool pass = Verifier::verifyVector(X_npu, X_golden, cfg, p.caseId());
    EXPECT_TRUE(pass);
    std::cout << "[" << p.caseId() << "] PASSED\n";
}

// ============================================================================
// Test body: Parameterized success / n=0 / singular test
// ============================================================================
TEST_P(Gtsv2Test, Gtsv2Success) {
    const auto& p = param_;
    PrintCaseInfo(p);

    int m = p.m;
    int n = p.n;
    int ldb = p.ldb;

    auto tri = GenerateTridiagMatrix(p);

    uint32_t rhsSeed = GetSeed(p) + 100;
    auto B_host = makeFullColMajor(m, n, ldb, rhsSeed, -5.0, 10.0);

    auto npuResult = Gtsv2Npu(*spHandle_, env_->stream(),
                                tri.dl, tri.d, tri.du,
                                B_host, m, n, ldb);

    EXPECT_EQ(npuResult.bufferSizeRet, ACL_SPARSE_STATUS_SUCCESS)
        << "[" << p.caseId() << "] bufferSizeExt should return SUCCESS";

    if (n == 0) {
        SUCCEED() << "[" << p.caseId() << "] n=0 early return SUCCESS";
        std::cout << "[" << p.caseId() << "] PASSED (n=0 SUCCESS)\n";
        return;
    }

    ASSERT_EQ(npuResult.computeRet, ACL_SPARSE_STATUS_SUCCESS)
        << "[" << p.caseId() << "] compute should return SUCCESS";

    if (p.matrixType == Gtsv2MatrixType::SINGULAR) {
        VerifySingularResult(p, B_host, tri, npuResult, m, n, ldb);
        return;
    }

    VerifyPrecisionResult(p, B_host, tri, npuResult, m, n, ldb);
}

// ============================================================================
// Parameterized test instantiation from CSV
// ============================================================================
INSTANTIATE_TEST_SUITE_P(
    Gtsv2Cases,
    Gtsv2Test,
    testing::ValuesIn(GetCasesFromCsv<Gtsv2Param>("gtsv2_test.csv")),
    [](const testing::TestParamInfo<Gtsv2Param>& info) {
        return info.param.caseId();
    }
);

// ============================================================================
// Exception test fixture: Gtsv2ExceptionTest
// ============================================================================
class Gtsv2ExceptionTest : public testing::Test {
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
        return aclsparseSgtsv2(
            handle, m, n, dl, d, du, B, ldb, pBuffer);
    }

    // Helper: call bufferSizeExt API with specified parameters
    aclsparseStatus_t CallBufferSize(aclsparseHandle_t handle,
                                     int m, int n,
                                     const float* dl,
                                     const float* d,
                                     const float* du,
                                     const float* B,
                                     int ldb,
                                     size_t* bufferSizeInBytes = nullptr) {
        size_t bufSize = 0;
        size_t* pBufSize = (bufferSizeInBytes != nullptr) ? bufferSizeInBytes : &bufSize;
        return aclsparseSgtsv2_bufferSizeExt(
            handle, m, n, dl, d, du, B, ldb, pBufSize);
    }
};

// ============================================================================
// Exception Tests (E1 - E16)
//
// Parameter validation tests. Most use bufferSizeExt (pure host-side
// validation, no device resources needed). pBuffer and B-NULL(compute)
// tests require the compute API.
// ============================================================================

// E1: NULL handle -> ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR
TEST_F(Gtsv2ExceptionTest, E1_NullHandle) {
    EXPECT_EQ(CallBufferSize(nullptr, m_, n_,
                              dl_.data(), d_.data(), du_.data(), B_.data(), ldb_),
              ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR);
}

// E2: m = 2 (< 3) -> ACL_SPARSE_STATUS_INVALID_VALUE
TEST_F(Gtsv2ExceptionTest, E2_MTooSmall_2) {
    EXPECT_EQ(CallBufferSize(handle_->get(), 2, n_,
                              dl_.data(), d_.data(), du_.data(), B_.data(), ldb_),
              ACL_SPARSE_STATUS_INVALID_VALUE);
}

// E3: m = 1 (< 3) -> ACL_SPARSE_STATUS_INVALID_VALUE
TEST_F(Gtsv2ExceptionTest, E3_MTooSmall_1) {
    EXPECT_EQ(CallBufferSize(handle_->get(), 1, n_,
                              dl_.data(), d_.data(), du_.data(), B_.data(), ldb_),
              ACL_SPARSE_STATUS_INVALID_VALUE);
}

// E4: m = 0 (< 3) -> ACL_SPARSE_STATUS_INVALID_VALUE
TEST_F(Gtsv2ExceptionTest, E4_MZero) {
    EXPECT_EQ(CallBufferSize(handle_->get(), 0, n_,
                              dl_.data(), d_.data(), du_.data(), B_.data(), ldb_),
              ACL_SPARSE_STATUS_INVALID_VALUE);
}

// E5: m = -1 (< 3) -> ACL_SPARSE_STATUS_INVALID_VALUE
TEST_F(Gtsv2ExceptionTest, E5_MNegative) {
    EXPECT_EQ(CallBufferSize(handle_->get(), -1, n_,
                              dl_.data(), d_.data(), du_.data(), B_.data(), ldb_),
              ACL_SPARSE_STATUS_INVALID_VALUE);
}

// E6: n = -1 (< 0) -> ACL_SPARSE_STATUS_INVALID_VALUE
// Note: n=0 is valid for gtsv2 (returns SUCCESS, early return) — tested in CSV.
TEST_F(Gtsv2ExceptionTest, E6_NNegative) {
    EXPECT_EQ(CallBufferSize(handle_->get(), m_, -1,
                              dl_.data(), d_.data(), du_.data(), B_.data(), ldb_),
              ACL_SPARSE_STATUS_INVALID_VALUE);
}

// E7: ldb = 0 (< max(1, m)) -> ACL_SPARSE_STATUS_INVALID_VALUE
TEST_F(Gtsv2ExceptionTest, E7_LdbZero) {
    EXPECT_EQ(CallBufferSize(handle_->get(), m_, n_,
                              dl_.data(), d_.data(), du_.data(), B_.data(), 0),
              ACL_SPARSE_STATUS_INVALID_VALUE);
}

// E8: ldb = m-1 (< max(1, m)) -> ACL_SPARSE_STATUS_INVALID_VALUE
TEST_F(Gtsv2ExceptionTest, E8_LdbTooSmall) {
    EXPECT_EQ(CallBufferSize(handle_->get(), m_, n_,
                              dl_.data(), d_.data(), du_.data(), B_.data(), m_ - 1),
              ACL_SPARSE_STATUS_INVALID_VALUE);
}

// E9: dl = nullptr (n>0) -> ACL_SPARSE_STATUS_INVALID_VALUE
TEST_F(Gtsv2ExceptionTest, E9_NullDl) {
    EXPECT_EQ(CallBufferSize(handle_->get(), m_, n_,
                              nullptr, d_.data(), du_.data(), B_.data(), ldb_),
              ACL_SPARSE_STATUS_INVALID_VALUE);
}

// E10: d = nullptr (n>0) -> ACL_SPARSE_STATUS_INVALID_VALUE
TEST_F(Gtsv2ExceptionTest, E10_NullD) {
    EXPECT_EQ(CallBufferSize(handle_->get(), m_, n_,
                              dl_.data(), nullptr, du_.data(), B_.data(), ldb_),
              ACL_SPARSE_STATUS_INVALID_VALUE);
}

// E11: du = nullptr (n>0) -> ACL_SPARSE_STATUS_INVALID_VALUE
TEST_F(Gtsv2ExceptionTest, E11_NullDu) {
    EXPECT_EQ(CallBufferSize(handle_->get(), m_, n_,
                              dl_.data(), d_.data(), nullptr, B_.data(), ldb_),
              ACL_SPARSE_STATUS_INVALID_VALUE);
}

// E12: B = nullptr (compute, n>0) -> ACL_SPARSE_STATUS_INVALID_VALUE
// Requires compute API (bufferSizeExt takes const B which may be NULL — see E16)
TEST_F(Gtsv2ExceptionTest, E12_NullB_compute) {
    // Obtain valid workspace size first
    size_t bufSize = 0;
    auto ret = aclsparseSgtsv2_bufferSizeExt(
        handle_->get(), m_, n_,
        dl_.data(), d_.data(), du_.data(), B_.data(), ldb_, &bufSize);
    ASSERT_EQ(ret, ACL_SPARSE_STATUS_SUCCESS);

    // Copy diagonals to device for compute API call
    auto dDl = DeviceBuffer::copyFrom(dl_.data(), m_ * sizeof(float));
    auto dD  = DeviceBuffer::copyFrom(d_.data(), m_ * sizeof(float));
    auto dDu = DeviceBuffer::copyFrom(du_.data(), m_ * sizeof(float));

    // Allocate workspace
    AlignedDeviceBuffer workspace = AlignedDeviceBuffer::allocAligned(bufSize);

    // Call compute with B=nullptr -> ACL_SPARSE_STATUS_INVALID_VALUE
    EXPECT_EQ(aclsparseSgtsv2(
        handle_->get(), m_, n_,
        reinterpret_cast<const float*>(dDl.raw()),
        reinterpret_cast<const float*>(dD.raw()),
        reinterpret_cast<const float*>(dDu.raw()),
        nullptr,  // B = nullptr
        m_, workspace.get()),
        ACL_SPARSE_STATUS_INVALID_VALUE);
}

// E13: pBuffer = nullptr (compute) -> ACL_SPARSE_STATUS_INVALID_VALUE
TEST_F(Gtsv2ExceptionTest, E13_NullPBuffer) {
    // Obtain valid workspace size first
    size_t bufSize = 0;
    auto ret = aclsparseSgtsv2_bufferSizeExt(
        handle_->get(), m_, n_,
        dl_.data(), d_.data(), du_.data(), B_.data(), ldb_, &bufSize);
    ASSERT_EQ(ret, ACL_SPARSE_STATUS_SUCCESS);

    // Copy B and diagonals to device for compute API call
    auto dB = DeviceBuffer::copyFrom(B_.data(), static_cast<int64_t>(ldb_) * n_ * sizeof(float));
    auto dDl = DeviceBuffer::copyFrom(dl_.data(), m_ * sizeof(float));
    auto dD  = DeviceBuffer::copyFrom(d_.data(), m_ * sizeof(float));
    auto dDu = DeviceBuffer::copyFrom(du_.data(), m_ * sizeof(float));

    // Call compute with pBuffer=nullptr -> ACL_SPARSE_STATUS_INVALID_VALUE
    EXPECT_EQ(aclsparseSgtsv2(
        handle_->get(), m_, n_,
        reinterpret_cast<const float*>(dDl.raw()),
        reinterpret_cast<const float*>(dD.raw()),
        reinterpret_cast<const float*>(dDu.raw()),
        reinterpret_cast<float*>(dB.get()),
        m_, nullptr),  // pBuffer = nullptr
        ACL_SPARSE_STATUS_INVALID_VALUE);
}

// E14: pBuffer not 128-byte aligned (compute) -> ACL_SPARSE_STATUS_INVALID_VALUE
TEST_F(Gtsv2ExceptionTest, E14_PBufferUnaligned) {
    // Obtain valid workspace size
    size_t bufSize = 0;
    auto ret = aclsparseSgtsv2_bufferSizeExt(
        handle_->get(), m_, n_,
        dl_.data(), d_.data(), du_.data(), B_.data(), ldb_, &bufSize);
    ASSERT_EQ(ret, ACL_SPARSE_STATUS_SUCCESS);

    // Create an aligned buffer and offset by 4 bytes to break 128-byte alignment
    if (bufSize == 0) bufSize = 256;
    auto alignedBuf = DeviceBuffer::alloc(bufSize + 128);

    // Copy B and diagonals to device
    auto dB = DeviceBuffer::copyFrom(B_.data(), static_cast<int64_t>(ldb_) * n_ * sizeof(float));
    auto dDl = DeviceBuffer::copyFrom(dl_.data(), m_ * sizeof(float));
    auto dD  = DeviceBuffer::copyFrom(d_.data(), m_ * sizeof(float));
    auto dDu = DeviceBuffer::copyFrom(du_.data(), m_ * sizeof(float));

    // Use pointer at offset 4 to break 128-byte alignment
    void* unaligned = reinterpret_cast<void*>(
        reinterpret_cast<uintptr_t>(alignedBuf.get()) + 4);

    EXPECT_EQ(aclsparseSgtsv2(
        handle_->get(), m_, n_,
        reinterpret_cast<const float*>(dDl.raw()),
        reinterpret_cast<const float*>(dD.raw()),
        reinterpret_cast<const float*>(dDu.raw()),
        reinterpret_cast<float*>(dB.get()),
        m_, unaligned),  // unaligned pBuffer
        ACL_SPARSE_STATUS_INVALID_VALUE);
}

// E15: bufferSizeInBytes = NULL (bufferSizeExt) -> ACL_SPARSE_STATUS_INVALID_VALUE
TEST_F(Gtsv2ExceptionTest, E15_NullBufferSizeInBytes) {
    EXPECT_EQ(aclsparseSgtsv2_bufferSizeExt(
        handle_->get(), m_, n_,
        dl_.data(), d_.data(), du_.data(), B_.data(), ldb_,
        nullptr),  // bufferSizeInBytes = NULL
        ACL_SPARSE_STATUS_INVALID_VALUE);
}

// E16: B = NULL (bufferSizeExt) -> ACL_SPARSE_STATUS_SUCCESS
// The query interface does not dereference B, so NULL is allowed.
TEST_F(Gtsv2ExceptionTest, E16_NullB_bufSizeAllowed) {
    size_t bufSize = 0;
    EXPECT_EQ(aclsparseSgtsv2_bufferSizeExt(
        handle_->get(), m_, n_,
        dl_.data(), d_.data(), du_.data(),
        nullptr,  // B = nullptr (allowed for bufferSizeExt)
        ldb_, &bufSize),
        ACL_SPARSE_STATUS_SUCCESS);
}

// ============================================================================
// White-box exception tests (E17 - E19)
//
// Edge cases for n=0 priority over other validation checks.
// Both the compute API and bufferSizeExt return SUCCESS for n=0 BEFORE
// checking m, ldb, or data pointers. These tests verify that priority.
// ============================================================================

// E17: bufferSizeExt with n=0 -> SUCCESS, *pBufferSizeInBytes = 0
// Covers host.cpp line 260-265: the bufferSizeExt n=0 early-return branch
// (distinct from the compute API's n=0 branch at line 203-206).
TEST_F(Gtsv2ExceptionTest, E17_BufferSizeExtN0) {
    size_t bufSize = 999;  // sentinel to verify it gets overwritten to 0
    EXPECT_EQ(aclsparseSgtsv2_bufferSizeExt(
        handle_->get(), m_, 0,  // n = 0
        dl_.data(), d_.data(), du_.data(), B_.data(), ldb_, &bufSize),
        ACL_SPARSE_STATUS_SUCCESS);
    EXPECT_EQ(bufSize, 0u)
        << "n=0 should produce bufferSize=0";
}

// E18: bufferSizeExt with n=0 AND m=2 (m<3) -> SUCCESS
// n=0 early return takes priority over m<3 check.
// Covers host.cpp line 260-265 (n=0 return) executing before line 267-271 (m<3 check).
TEST_F(Gtsv2ExceptionTest, E18_BufferSizeExtN0M2) {
    size_t bufSize = 999;
    EXPECT_EQ(aclsparseSgtsv2_bufferSizeExt(
        handle_->get(), 2, 0,  // m=2 (< 3), n=0
        dl_.data(), d_.data(), du_.data(), B_.data(), ldb_, &bufSize),
        ACL_SPARSE_STATUS_SUCCESS);
    EXPECT_EQ(bufSize, 0u);
}

// E19: compute API with n=0 AND m=2 (m<3) -> SUCCESS
// n=0 early return takes priority over ValidateGtsv2Params (which would reject m<3).
// Covers host.cpp line 203-206 (n=0 return) executing before line 208-212 (Validate).
// No device resources needed: n=0 returns before touching any GPU memory.
TEST_F(Gtsv2ExceptionTest, E19_ComputeN0M2) {
    EXPECT_EQ(aclsparseSgtsv2(
        handle_->get(), 2, 0,  // m=2 (< 3), n=0
        dl_.data(), d_.data(), du_.data(), B_.data(), ldb_,
        nullptr),  // pBuffer (not dereferenced due to n=0 early return)
        ACL_SPARSE_STATUS_SUCCESS);
}
