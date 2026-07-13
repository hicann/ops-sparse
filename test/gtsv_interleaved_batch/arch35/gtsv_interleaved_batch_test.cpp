/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// GTest + CSV parameterized test for aclsparseSgtsvInterleavedBatch (L0 + L1, iteration 2)
//
// API: aclsparseSgtsvInterleavedBatch(handle, algo, m,
//          float* dl, float* d, float* du, float* x, int batchCount, void* pBuffer)
//
// Data layout: row-major interleaved, data[row * batchCount + batch]
// Boundary conditions: dl[0][*] = 0, du[m-1][*] = 0

#include "test_common.h"

#include "../gtsv_interleaved_batch_golden.h"
#include "../gtsv_interleaved_batch_param.h"
#include "gtsv_interleaved_batch_npu_wrapper.h"

namespace sparse_test {

// ----------------------------------------------------------------------------
// Helper for exception-path tests: builds 4 device buffers of the same size
// and (optionally) an aligned 256-byte workspace buffer for the API call.
// Each TEST_F sets its own m/batchCount/algo parameters (some intentionally
// invalid) and calls aclsparseSgtsvInterleavedBatch directly, checking the
// returned status code.
// ----------------------------------------------------------------------------
struct GtsvBufSet {
    DeviceBuffer dDl, dD, dDu, dX;
    DeviceBuffer dBuf;  // default-constructed = nullptr-backed via raw()

    // Build 4 device buffers, each holding |totalElems| floats initialized to val.
    // totalElems of 0 is promoted to 1 to avoid empty device allocations.
    static GtsvBufSet Make(int totalElems, float val = 1.0f) {
        GtsvBufSet bs;
        const int n = (totalElems > 0) ? totalElems : 1;
        std::vector<float> dummy(n, val);
        bs.dDl = DeviceBuffer::copyFrom(dummy.data(), n * sizeof(float));
        bs.dD  = DeviceBuffer::copyFrom(dummy.data(), n * sizeof(float));
        bs.dDu = DeviceBuffer::copyFrom(dummy.data(), n * sizeof(float));
        bs.dX  = DeviceBuffer::copyFrom(dummy.data(), n * sizeof(float));
        return bs;
    }

    void allocAlignedWorkspace() { dBuf = DeviceBuffer::alloc(256); }

    aclsparseStatus_t call(aclsparseHandle_t h, int algo, int m, int batchCount,
                           void* pBuffer) {
        return aclsparseSgtsvInterleavedBatch(
            h, algo, m,
            static_cast<float*>(dDl.get()),
            static_cast<float*>(dD.get()),
            static_cast<float*>(dDu.get()),
            static_cast<float*>(dX.get()),
            batchCount, pBuffer);
    }

    aclsparseStatus_t call(aclsparseHandle_t h, int algo, int m, int batchCount) {
        return call(h, algo, m, batchCount, dBuf.raw());
    }
};

// ============================================================================
// GTest test fixture
// ============================================================================
class GtsvInterleavedBatchTest : public ::testing::TestWithParam<GtsvInterleavedBatchParam> {
public:
    static void SetUpTestSuite() {
        env_ = std::make_unique<AclEnvScope>();
        spHandle_ = std::make_unique<HandleManager>();
        spHandle_->setStream(env_->stream());
        goldenSelfTestOk_ = RunGoldenSelfTests();
        if (!goldenSelfTestOk_) {
            std::cerr << "[FATAL] Thomas algorithm golden self-test FAILED!\n";
        }
    }

    static void TearDownTestSuite() {
        spHandle_.reset();
        env_.reset();
    }

protected:
    inline static std::unique_ptr<AclEnvScope> env_;
    inline static std::unique_ptr<HandleManager> spHandle_;
    inline static bool goldenSelfTestOk_ = false;

    // Verifies Thomas FP64 golden matches known solution within fp64 precision
    static bool RunGoldenSelfTests() {
        bool ok = true;
        // Self-test 1: m=3, batch=2, seed=42 (diagonal dominant)
        if (!ThomasGoldenSelfTest(3, 2, 42)) {
            std::cerr << "[Golden self-test] FAIL: m=3, batch=2\n";
            ok = false;
        }
        // Self-test 2: m=5, batch=1, seed=43
        if (!ThomasGoldenSelfTest(5, 1, 43)) {
            std::cerr << "[Golden self-test] FAIL: m=5, batch=1\n";
            ok = false;
        }
        // Self-test 3: m=1, batch=1 (trivial case)
        if (!ThomasGoldenSelfTest(1, 1, 100)) {
            std::cerr << "[Golden self-test] FAIL: m=1, batch=1\n";
            ok = false;
        }
        return ok;
    }

    // Generate tridiagonal matrix data based on CSV parameters
    static TridiagMatrix GenerateTridiagMatrix(const GtsvInterleavedBatchParam& p) {
        if (p.is_diag_dominant) {
            return makeDiagDominantTridiag(p.m, p.batch_count, p.seed);
        }
        // Non-diagonal-dominant: check for identity or constant diagonal
        double loRange = p.value_hi - p.value_lo;
        if (loRange < 1e-10) {
            // Uniform constant diagonal
            float diagVal = static_cast<float>(p.value_lo);
            return makeConstantDiagTridiag(p.m, p.batch_count, diagVal);
        }
        return makeRandomTridiag(p.m, p.batch_count, p.value_lo, p.value_hi, p.seed);
    }
};

// ============================================================================
// Parameterized test case: GenericSuccess
// ============================================================================
TEST_P(GtsvInterleavedBatchTest, GenericSuccess) {
    auto p = GetParam();
    std::cout << "\n==== " << p.case_name
              << " m=" << p.m << " batch=" << p.batch_count
              << " algo=" << p.algo
              << " diag_dom=" << p.is_diag_dominant << " ====\n";

    // Pre-check golden self-test ran OK
    ASSERT_TRUE(goldenSelfTestOk_)
        << "Thomas algorithm golden reference self-test failed! Aborting.";

    // Expect success for L0 cases
    ASSERT_EQ(p.expect_result, "ACL_SPARSE_STATUS_SUCCESS");

    const int m = p.m;
    const int batchCount = p.batch_count;

    // 1. Generate tridiagonal matrix
    TridiagMatrix tri = GenerateTridiagMatrix(p);

    // 2. Generate right-hand side b
    // For identity/constant diagonal, b=x directly gives exact solution
    std::vector<float> b;
    if (p.is_diag_dominant) {
        b = makeTridiagRHS(m, batchCount, -5.0, 5.0, p.seed + 1000);
    } else {
        // Use seed-derived b for reproducibility
        b = makeTridiagRHS(m, batchCount, -5.0, 5.0, p.seed + 2000);
    }

    // 3. CPU golden: Thomas algorithm FP64
    std::vector<float> xGolden = ThomasSolveGolden(tri.dl, tri.d, tri.du, b, m, batchCount);

    // 4. NPU execution
    std::vector<float> xNpu;
    try {
        xNpu = GtsvInterleavedBatchNpu(
            spHandle_->get(), env_->stream(),
            p.algo, m,
            tri.dl, tri.d, tri.du, b,
            batchCount);
    } catch (const std::exception& e) {
        FAIL() << "[" << p.case_name << "] NPU execution threw: " << e.what();
    }

    // 5. Precision verification: MERE/MARE mode
    ASSERT_EQ(xNpu.size(), xGolden.size())
        << "[" << p.case_name << "] Size mismatch: NPU="
        << xNpu.size() << " Golden=" << xGolden.size();

    VerifyConfig cfg;
    cfg.SetMode(PrecisionMode::MERE_MARE)
       .SetMERE(p.mere_threshold)       // 2^-15 ~ 3.05e-5
       .SetMARE(p.mare_multiplier);     // 10.0 (MARE < 10 * MERE_threshold = 3.05e-4)

    EXPECT_TRUE(Verifier::verifyVector(xNpu, xGolden, cfg, p.case_name))
        << "[" << p.case_name << "] Precision verification FAILED";
}

// ============================================================================
// INSTANTIATE_TEST_SUITE_P from CSV (L0 + L1 cases)
// ============================================================================
INSTANTIATE_TEST_SUITE_P(
    GtsvInterleavedBatch,
    GtsvInterleavedBatchTest,
    ::testing::ValuesIn(GetCasesFromCsv<GtsvInterleavedBatchParam>(
        "gtsv_interleaved_batch_test.csv")),
    [](const ::testing::TestParamInfo<GtsvInterleavedBatchParam>& info) {
        return info.param.caseId();
    }
);

// ============================================================================
// Exception path tests (TEST_F, L1 iteration 2)
// These test invalid parameter validation directly via API calls,
// without golden comparison. ACL env is initialized once per suite.
// ============================================================================
class GtsvInterleavedBatchExceptionTest : public ::testing::Test {
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
};

// m=0 should return ACL_SPARSE_STATUS_INVALID_VALUE
TEST_F(GtsvInterleavedBatchExceptionTest, InvalidM0) {
    auto bs = GtsvBufSet::Make(4);
    EXPECT_EQ(bs.call(spHandle_->get(), 0, /*m=*/0, /*bc=*/4),
              ACL_SPARSE_STATUS_INVALID_VALUE);
}

// batchCount=0 should return ACL_SPARSE_STATUS_INVALID_VALUE
TEST_F(GtsvInterleavedBatchExceptionTest, InvalidBatchCount0) {
    auto bs = GtsvBufSet::Make(10);
    EXPECT_EQ(bs.call(spHandle_->get(), 0, /*m=*/10, /*bc=*/0),
              ACL_SPARSE_STATUS_INVALID_VALUE);
}

// algo=1 (valid range but not supported) should return ACL_SPARSE_STATUS_NOT_SUPPORTED
TEST_F(GtsvInterleavedBatchExceptionTest, UnsupportedAlgo1) {
    auto bs = GtsvBufSet::Make(10 * 4);
    bs.allocAlignedWorkspace();
    EXPECT_EQ(bs.call(spHandle_->get(), /*algo=*/1, /*m=*/10, /*bc=*/4),
              ACL_SPARSE_STATUS_NOT_SUPPORTED);
}

// algo=2 (valid range but not supported) should return ACL_SPARSE_STATUS_NOT_SUPPORTED
TEST_F(GtsvInterleavedBatchExceptionTest, UnsupportedAlgo2) {
    auto bs = GtsvBufSet::Make(10 * 4);
    bs.allocAlignedWorkspace();
    EXPECT_EQ(bs.call(spHandle_->get(), /*algo=*/2, /*m=*/10, /*bc=*/4),
              ACL_SPARSE_STATUS_NOT_SUPPORTED);
}

// algo=-1 (out of range) should return ACL_SPARSE_STATUS_INVALID_VALUE
TEST_F(GtsvInterleavedBatchExceptionTest, InvalidAlgoNegative) {
    auto bs = GtsvBufSet::Make(10 * 4);
    bs.allocAlignedWorkspace();
    EXPECT_EQ(bs.call(spHandle_->get(), /*algo=*/-1, /*m=*/10, /*bc=*/4),
              ACL_SPARSE_STATUS_INVALID_VALUE);
}

// algo=3 (out of valid range [0,2]) should return ACL_SPARSE_STATUS_INVALID_VALUE
TEST_F(GtsvInterleavedBatchExceptionTest, InvalidAlgo3) {
    auto bs = GtsvBufSet::Make(10 * 4);
    bs.allocAlignedWorkspace();
    EXPECT_EQ(bs.call(spHandle_->get(), /*algo=*/3, /*m=*/10, /*bc=*/4),
              ACL_SPARSE_STATUS_INVALID_VALUE);
}

// dl=nullptr should return ACL_SPARSE_STATUS_INVALID_VALUE
TEST_F(GtsvInterleavedBatchExceptionTest, NullPointerDl) {
    auto bs = GtsvBufSet::Make(4 * 1);
    bs.allocAlignedWorkspace();
    auto ret = aclsparseSgtsvInterleavedBatch(
        spHandle_->get(), 0, /*m=*/4,
        nullptr,  // dl = nullptr
        static_cast<float*>(bs.dD.get()),
        static_cast<float*>(bs.dDu.get()),
        static_cast<float*>(bs.dX.get()),
        /*bc=*/1, bs.dBuf.raw());
    EXPECT_EQ(ret, ACL_SPARSE_STATUS_INVALID_VALUE);
}

// pBuffer not 128-byte aligned (m > 1) should return ACL_SPARSE_STATUS_INVALID_VALUE
TEST_F(GtsvInterleavedBatchExceptionTest, UnalignedBuffer) {
    auto bs = GtsvBufSet::Make(4 * 1);
    bs.allocAlignedWorkspace();
    void* misalignedPtr = reinterpret_cast<char*>(bs.dBuf.raw()) + 1;
    EXPECT_EQ(bs.call(spHandle_->get(), 0, /*m=*/4, /*bc=*/1, misalignedPtr),
              ACL_SPARSE_STATUS_INVALID_VALUE);
}

}  // namespace sparse_test
