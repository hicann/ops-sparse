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
 * @file coo2csr_test.cpp
 * @brief GTest + CSV-driven test cases for aclsparseXcoo2csr (Legacy API).
 *
 * Tests COO row indices to CSR row pointers conversion.
 *
 * Test structure:
 *   - TEST_P (Coo2CsrTest)          : parameterized success-path tests from CSV
 *   - TEST_F (Coo2CsrExceptionTest) : invalid-param error tests
 *
 * Test parameters are loaded from coo2csr_test.csv (copied to build dir by
 * CMake). Entry point is shared via test/frame/test_main.cpp.
 */

#include "test_common.h"
#include "coo2csr_golden.h"
#include "coo2csr_npu_wrapper.h"
#include "coo2csr_param.h"

#include <algorithm>  // std::equal (int32 bit-exact verification)

using namespace sparse_test;

// ============================================================================
// Helper: Convert CsrMatrix to CooMatrix (inline, for diag pattern)
// ============================================================================

static CooMatrix Coo2CsrCsrToCooHelper(const CsrMatrix& csr)
{
    CooMatrix out;
    out.rows = csr.rows;
    out.cols = csr.cols;
    out.nnz = csr.nnz;
    out.rowIndices.reserve(csr.nnz);
    out.colIndices.reserve(csr.nnz);
    out.values.reserve(csr.nnz);
    for (int64_t i = 0; i < csr.rows; i++) {
        for (int32_t j = csr.rowOffsets[i]; j < csr.rowOffsets[i + 1]; j++) {
            out.rowIndices.push_back(static_cast<int32_t>(i));
            out.colIndices.push_back(csr.colIndices[j]);
            out.values.push_back(csr.values[j]);
        }
    }
    return out;
}

constexpr int kAutoNnz = -1;

// ============================================================================
// Helper: Generate COO row indices based on test pattern
//
// Supports patterns: random / diag / allsame.
// Applies idxBase offset for all patterns (including diag).
// ============================================================================

static void GenerateCooRowInd(
    const Coo2CsrParam& p,
    std::vector<int32_t>& cooRowInd,
    int& nnz)
{
    if (p.pattern == "diag") {
        auto csr = makeDiagCsr(p.m);
        auto coo = Coo2CsrCsrToCooHelper(csr);
        cooRowInd = coo.rowIndices;
        nnz = static_cast<int>(coo.nnz);
    } else if (p.pattern == "allsame") {
        int targetNnz = static_cast<int>(static_cast<double>(p.m) * static_cast<double>(p.n) * (1.0 - p.sparsity));
        targetNnz = std::max(targetNnz, 1);
        nnz = targetNnz;
        cooRowInd.assign(nnz, 0);  // 0-based，末尾统一偏移
    } else {
        if (p.sparsity >= 1.0) {
            nnz = 0;
            cooRowInd.clear();
        } else {
            SparseFillGenerator gen(p.seed);
            gen.setSparsity(p.sparsity);
            gen.setEmptyRowProb(p.empty_row_prob);
            auto coo = gen.generateCoo(p.m, p.n, kAutoNnz);
            cooRowInd = coo.rowIndices;
            nnz = static_cast<int>(coo.nnz);
        }
    }

    // Apply idxBase offset (diag and random produce 0-based indices)
    if (p.idx_base == 1) {
        for (auto& v : cooRowInd) {
            v += 1;
        }
    }
}

// ============================================================================
// Helper: Verify csrRowPtr with direct int32 bit-exact comparison.
//
// coo2csr is a pure integer index transformation. Per ops-precision-standard
// "integer compute" criteria, the int32 output must be bit-exact with the
// golden (absolute error 0 / binary identical).
//
// Why not use Verifier::verifyVector:
//   The Verifier framework's verifyVector() only accepts float* /
//   std::vector<float>, and its IntegerStrategy casts float -> int64_t to
//   compare. Routing int32 -> float -> int64 introduces a lossy intermediate
//   representation: float has only 24 bits of significand, so int32 values
//   with |v| >= 2^24 cannot be represented exactly and the comparison would
//   be incorrect for large row pointers. Comparing the int32 vectors directly
//   with std::equal guarantees bit-exactness and matches the operator's
//   pure-integer semantics.
// ============================================================================
static void VerifyCsrRowPtrInt32(
    const std::vector<int32_t>& out,
    const std::vector<int32_t>& golden,
    const std::string& caseId)
{
    if (out.size() != golden.size()) {
        std::cout << "[" << caseId << "] csrRowPtr FAILED: size mismatch out="
                  << out.size() << " golden=" << golden.size() << "\n";
        ADD_FAILURE() << "[" << caseId << "] csrRowPtr size mismatch";
        return;
    }
    bool pass = std::equal(out.begin(), out.end(), golden.begin());
    if (pass) {
        std::cout << "[" << caseId << "] csrRowPtr PASSED "
                  << "(int32 exact match, " << out.size() << " elements)\n";
    } else {
        std::cout << "[" << caseId << "] csrRowPtr FAILED (int32 mismatch)\n";
        // 仅打印前若干个不匹配位置，避免大 m 时全量打印
        int shown = 0;
        for (size_t i = 0; i < out.size() && shown < 10; i++) {
            if (out[i] != golden[i]) {
                std::cout << "  [" << i << "] out=" << out[i]
                          << " golden=" << golden[i] << "\n";
                shown++;
            }
        }
    }
    EXPECT_TRUE(pass) << "[" << caseId << "] csrRowPtr int32 mismatch";
}

// ============================================================================
// Shared test environment for all coo2csr test fixtures
// ============================================================================

class Coo2CsrTestEnv {
public:
    static void SetUp() { env_ = std::make_unique<AclEnvScope>(); }
    static void TearDown() { env_.reset(); }
    static aclrtStream stream() { return env_->stream(); }
private:
    inline static std::unique_ptr<AclEnvScope> env_;
};

// ============================================================================
// GTest parameterized fixture: Coo2CsrTest
// ============================================================================

class Coo2CsrTest : public testing::TestWithParam<Coo2CsrParam> {
public:
    static void SetUpTestSuite() { Coo2CsrTestEnv::SetUp(); }
    static void TearDownTestSuite() { Coo2CsrTestEnv::TearDown(); }

protected:
    Coo2CsrParam param_;
    aclrtStream stream_ = nullptr;

    void SetUp() override
    {
        param_ = GetParam();
        stream_ = Coo2CsrTestEnv::stream();
    }
};

// ============================================================================
// TEST_P: Success-path parameterized test
// ============================================================================

TEST_P(Coo2CsrTest, Coo2CsrSuccess) {
    const auto& p = param_;

    std::cout << "==== " << p.case_name
              << " ==== m=" << p.m << " n=" << p.n
              << " sparsity=" << p.sparsity
              << " empty_row_prob=" << p.empty_row_prob
              << " idxBase=" << p.idx_base
              << " pattern=" << p.pattern << "\n";

    // 1. Generate COO data based on pattern
    std::vector<int32_t> cooRowInd;
    int nnz = 0;
    GenerateCooRowInd(p, cooRowInd, nnz);

    std::cout << "  nnz=" << nnz << "\n";

    // 2. Compute golden reference (CPU, int32_t direct computation)
    if (cooRowInd.size() < static_cast<size_t>(nnz)) {
        FAIL() << "[" << p.case_name << "] internal: cooRowInd.size() < nnz";
    }
    auto csrRowPtrGolden = Coo2CsrGolden(cooRowInd, nnz, p.m, p.idx_base);

    // 3. Call NPU
    aclsparseIndexBase_t idxBaseEnum = (p.idx_base == 1)
        ? ACL_SPARSE_INDEX_BASE_ONE : ACL_SPARSE_INDEX_BASE_ZERO;

    HandleManager handle;
    handle.setStream(stream_);

    const int32_t* cooRowIndPtr = cooRowInd.empty() ? nullptr : cooRowInd.data();

    auto npuResult = Coo2CsrNpu(handle, stream_, cooRowIndPtr, nnz, p.m, idxBaseEnum);

    // 4. Verify API return code
    ASSERT_EQ(npuResult.computeRet, ACL_SPARSE_STATUS_SUCCESS)
        << "[" << p.case_name << "] aclsparseXcoo2csr failed with status "
        << npuResult.computeRet;

    // 5. Verify csrRowPtr — direct int32 bit-exact comparison.
    // coo2csr is a pure integer index transformation; per ops-precision-standard
    // "integer compute" criteria, the int32 output must be bit-exact with the
    // golden. See VerifyCsrRowPtrInt32 for why Verifier::toFloat is avoided.
    VerifyCsrRowPtrInt32(npuResult.csrRowPtr, csrRowPtrGolden, p.case_name);

    std::cout << "[" << p.case_name << "] PASSED (nnz=" << nnz << ")\n";
}

// ============================================================================
// Parameterized test instantiation from CSV
// ============================================================================

INSTANTIATE_TEST_SUITE_P(
    Coo2CsrCases,
    Coo2CsrTest,
    testing::ValuesIn(GetCasesFromCsv<Coo2CsrParam>("coo2csr_test.csv")),
    [](const testing::TestParamInfo<Coo2CsrParam>& info) {
        return info.param.case_name;
    }
);

// ============================================================================
// Exception test fixture: Coo2CsrExceptionTest
// ============================================================================

class Coo2CsrExceptionTest : public testing::Test {
public:
    static void SetUpTestSuite() { Coo2CsrTestEnv::SetUp(); }
    static void TearDownTestSuite() { Coo2CsrTestEnv::TearDown(); }

protected:
    static constexpr int kExcM = 4;
    static constexpr int kExcCsrLen = kExcM + 1;
    static constexpr int kInvalidIdxBase = 99;
    static constexpr double kTestSparsity = 0.5;
    static constexpr int kTestSeed = 42;

    void SetUp() override
    {
        stream_ = Coo2CsrTestEnv::stream();
        handle_ = std::make_unique<HandleManager>();
        handle_->setStream(stream_);

        auto coo = makeSparseCoo(kExcM, kExcM, kTestSparsity, kTestSeed);
        nnz_ = static_cast<int>(coo.nnz);
        cooRowInd_ = coo.rowIndices;
        csrRowPtr_.assign(kExcCsrLen, 0);
    }

    void TearDown() override
    {
        handle_.reset();
    }

    aclrtStream stream_ = nullptr;
    std::unique_ptr<HandleManager> handle_;
    int nnz_ = 0;
    std::vector<int32_t> cooRowInd_;
    std::vector<int32_t> csrRowPtr_;
};

// Exception 1: NULL handle -> ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR
TEST_F(Coo2CsrExceptionTest, NullHandle) {
    EXPECT_EQ(
        aclsparseXcoo2csr(nullptr, cooRowInd_.data(), nnz_, kExcM,
                         csrRowPtr_.data(), ACL_SPARSE_INDEX_BASE_ZERO),
        ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR);
}

// Exception 2: NULL cooRowInd with nnz > 0 -> ACL_SPARSE_STATUS_INVALID_VALUE
TEST_F(Coo2CsrExceptionTest, NullCooRowInd) {
    EXPECT_EQ(
        aclsparseXcoo2csr(handle_->get(), nullptr, nnz_, kExcM,
                         csrRowPtr_.data(), ACL_SPARSE_INDEX_BASE_ZERO),
        ACL_SPARSE_STATUS_INVALID_VALUE);
}

// Exception 3: NULL csrRowPtr -> ACL_SPARSE_STATUS_INVALID_VALUE
TEST_F(Coo2CsrExceptionTest, NullCsrRowPtr) {
    EXPECT_EQ(
        aclsparseXcoo2csr(handle_->get(), cooRowInd_.data(), nnz_, kExcM,
                         nullptr, ACL_SPARSE_INDEX_BASE_ZERO),
        ACL_SPARSE_STATUS_INVALID_VALUE);
}

// Exception 4: Invalid idxBase -> ACL_SPARSE_STATUS_INVALID_VALUE
TEST_F(Coo2CsrExceptionTest, InvalidIdxBase) {
    EXPECT_EQ(
        aclsparseXcoo2csr(handle_->get(), cooRowInd_.data(), nnz_, kExcM,
                         csrRowPtr_.data(), static_cast<aclsparseIndexBase_t>(kInvalidIdxBase)),
        ACL_SPARSE_STATUS_INVALID_VALUE);
}

// Exception 5: nnz=0 with NULL cooRowInd is legal -> ACL_SPARSE_STATUS_SUCCESS
TEST_F(Coo2CsrExceptionTest, Nnz0NullCooRowInd) {
    auto dCsrRowPtr = DeviceBuffer::alloc(kExcCsrLen * sizeof(int32_t));
    EXPECT_EQ(
        aclsparseXcoo2csr(handle_->get(), nullptr, 0, kExcM,
                         reinterpret_cast<int*>(dCsrRowPtr.get()),
                         ACL_SPARSE_INDEX_BASE_ZERO),
        ACL_SPARSE_STATUS_SUCCESS);
    EXPECT_EQ(aclrtSynchronizeStream(stream_), ACL_SUCCESS);
    std::vector<int32_t> hostCsrRowPtr(kExcCsrLen, -1);
    dCsrRowPtr.copyToHost(hostCsrRowPtr.data(), kExcCsrLen * sizeof(int32_t));
    EXPECT_EQ(hostCsrRowPtr, (std::vector<int32_t>{0, 0, 0, 0, 0}));
}

TEST_F(Coo2CsrExceptionTest, Nnz0IdxBase1DirectApi) {
    auto dCsrRowPtr = DeviceBuffer::alloc(kExcCsrLen * sizeof(int32_t));
    EXPECT_EQ(
        aclsparseXcoo2csr(handle_->get(), nullptr, 0, kExcM,
                         reinterpret_cast<int*>(dCsrRowPtr.get()),
                         ACL_SPARSE_INDEX_BASE_ONE),
        ACL_SPARSE_STATUS_SUCCESS);
    EXPECT_EQ(aclrtSynchronizeStream(stream_), ACL_SUCCESS);
    std::vector<int32_t> hostCsrRowPtr(kExcCsrLen, -1);
    dCsrRowPtr.copyToHost(hostCsrRowPtr.data(), kExcCsrLen * sizeof(int32_t));
    EXPECT_EQ(hostCsrRowPtr, (std::vector<int32_t>{1, 1, 1, 1, 1}));
}

// Exception 6: Negative nnz -> ACL_SPARSE_STATUS_INVALID_VALUE
TEST_F(Coo2CsrExceptionTest, NegativeNnz) {
    EXPECT_EQ(
        aclsparseXcoo2csr(handle_->get(), cooRowInd_.data(), -1, kExcM,
                         csrRowPtr_.data(), ACL_SPARSE_INDEX_BASE_ZERO),
        ACL_SPARSE_STATUS_INVALID_VALUE);
}

// Exception 7: Negative m -> ACL_SPARSE_STATUS_INVALID_VALUE
TEST_F(Coo2CsrExceptionTest, NegativeM) {
    EXPECT_EQ(
        aclsparseXcoo2csr(handle_->get(), cooRowInd_.data(), nnz_, -1,
                         csrRowPtr_.data(), ACL_SPARSE_INDEX_BASE_ZERO),
        ACL_SPARSE_STATUS_INVALID_VALUE);
}

// Exception 8: nnz > 0 but m == 0 -> ACL_SPARSE_STATUS_INVALID_VALUE
TEST_F(Coo2CsrExceptionTest, NnzPositiveMZero) {
    EXPECT_EQ(
        aclsparseXcoo2csr(handle_->get(), cooRowInd_.data(), nnz_, 0,
                         csrRowPtr_.data(), ACL_SPARSE_INDEX_BASE_ZERO),
        ACL_SPARSE_STATUS_INVALID_VALUE);
}

// Exception 9: handle workspace insufficient for large m
//              -> ACL_SPARSE_STATUS_INSUFFICIENT_RESOURCES
//
// The coo2csr host side allocates its rowCount[] and blockTotals[] workspace
// from the handle's effective workspace (see aclsparseGetEffectiveWorkspace).
// The default workspace created by aclsparseCreate is 4 MiB. The operator needs
//   totalNeeded = m * sizeof(int32_t) + aivCoreNum * sizeof(int32_t) bytes.
// For m = 1048576 this is 4 MiB + aivCoreNum*4 > 4 MiB, so the default handle
// workspace is insufficient. Without a user-supplied workspace (via
// aclsparseSetWorkspace), the operator must reject the call with
// INSUFFICIENT_RESOURCES rather than falling back to aclrtMalloc.
//
// This case bypasses Coo2CsrNpu (which auto-allocates a user workspace) and
// invokes the API directly, mirroring a real caller that forgot to enlarge the
// handle workspace. nnz = 1 (with idxBase = 0) is used so the operator reaches
// PrepareWorkspace instead of the nnz==0 fast path (FillCsrRowPtrZero) that
// skips the workspace check. Valid device buffers are provided so parameter
// validation passes; the operator must fail at workspace preparation before any
// kernel is launched, so the input/output buffers are not dereferenced.
TEST_F(Coo2CsrExceptionTest, InsufficientWorkspace) {
    constexpr int32_t kLargeM = 1048576;  // 2^20; default 4 MiB ws < rowCount + blockTotals
    constexpr int32_t kNnz = 1;           // nnz > 0 to bypass the nnz==0 fast path

    // Provide valid device buffers so ValidateCoo2CsrParams passes. The operator
    // returns INSUFFICIENT_RESOURCES from PrepareWorkspace before launching any
    // kernel, so these buffers are never accessed.
    const int32_t hostCooRowInd = 0;  // single COO row index = 0 (0-based)
    auto dCooRowInd = DeviceBuffer::copyFrom(&hostCooRowInd,
                                              kNnz * sizeof(int32_t));
    auto dCsrRowPtr = DeviceBuffer::alloc(
        (static_cast<size_t>(kLargeM) + 1) * sizeof(int32_t));

    EXPECT_EQ(
        aclsparseXcoo2csr(handle_->get(),
                          reinterpret_cast<const int*>(dCooRowInd.get()),
                          kNnz, kLargeM,
                          reinterpret_cast<int*>(dCsrRowPtr.get()),
                          ACL_SPARSE_INDEX_BASE_ZERO),
        ACL_SPARSE_STATUS_INSUFFICIENT_RESOURCES);
}

// 禁止定义 main 函数（由 test/frame/test_main.cpp 提供）
