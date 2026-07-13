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
 * @file csrgeam2_test.cpp
 * @brief GTest + CSV-driven test cases for aclsparseScsrgeam2 (Legacy API).
 *
 * Tests the sparse matrix addition: C = alpha * A + beta * B
 * where A, B, C are CSR-format sparse matrices.
 *
 * Test structure:
 *   - TEST_P (CsrGeam2Test)   : parameterized success-path tests from CSV
 *   - TEST_F (CsrGeam2ExceptionTest) : null-pointer / invalid-param error tests
 *
 * Test parameters are loaded from csrgeam2_test.csv (copied to build dir by
 * CMake). Entry point is shared via test/frame/test_main.cpp.
 */

#include "test_common.h"
#include "csrgeam2_golden.h"
#include "csrgeam2_npu_wrapper.h"
#include "csrgeam2_param.h"

using namespace sparse_test;

// CSR data preparation: convert fill.h 0-based CSR to desired index base
static CsrMatrix PrepareCsr(const CsrMatrix &csr0, int indexBase) {
    CsrMatrix csr = csr0;  // copy
    if (indexBase == 1) {
        for (size_t i = 0; i < csr.rowOffsets.size(); i++) {
            csr.rowOffsets[i] += 1;
        }
        for (size_t i = 0; i < csr.colIndices.size(); i++) {
            csr.colIndices[i] += 1;
        }
    }
    return csr;
}

// Shared ACL environment (one instance per test-suite, owned by the parameterized fixture).
// Non-parameterized exception tests use SetUpTestSuite too (via shared env_).
using SharedAclEnvScope = AclEnvScope;

// GTest parameterized fixture: CsrGeam2Test
class CsrGeam2Test : public testing::TestWithParam<CsrGeam2TestParam> {
public:
    static void SetUpTestSuite() {
        env_ = std::make_unique<AclEnvScope>();
    }

    static void TearDownTestSuite() {
        env_.reset();
    }

protected:
    inline static std::unique_ptr<AclEnvScope> env_;
    CsrGeam2TestParam param_;
    aclrtStream stream_ = nullptr;

    void SetUp() override {
        param_ = GetParam();
        stream_ = env_->stream();
    }
};

// Helper: Prepare input CSR matrices A and B for a parameterized test case.
static std::pair<CsrMatrix, CsrMatrix> PrepareTestInputs(const CsrGeam2TestParam &p) {
    CsrMatrix csrA_0, csrB_0;
    if (p.pattern == "diag") {
        csrA_0 = makeDiagCsr(p.m);
        csrB_0 = makeDiagCsr(p.m);
    } else {
        csrA_0 = makeSparseCsr(p.m, p.n, p.sparsity_a, p.seed_a);
        csrB_0 = makeSparseCsr(p.m, p.n, p.sparsity_b, p.seed_b);
    }
    return {PrepareCsr(csrA_0, p.index_base_a), PrepareCsr(csrB_0, p.index_base_b)};
}

// Helper: Execute the NPU three-step workflow for a parameterized test case.
static CsrGeam2NpuResult RunCsrGeam2Npu(
    HandleManager &handle, aclrtStream stream,
    const CsrMatrix &csrA, int nnzA, const CsrMatrix &csrB, int nnzB,
    const CsrGeam2TestParam &p, bool useHostMode)
{
    handle.setStream(stream);
    return CsrGeam2Npu(
        handle, stream,
        csrA.rowOffsets, csrA.colIndices, csrA.values, nnzA,
        csrB.rowOffsets, csrB.colIndices, csrB.values, nnzB,
        p.m, p.n, p.alpha, p.beta,
        p.index_base_a, p.index_base_b, p.index_base_c,
        useHostMode);
}

// Helper: Verify the NPU output against the golden reference.
static void VerifyCsrGeam2Result(
    const CsrGeam2NpuResult &npuResult, const CsrGeam2GoldenResult &golden,
    const std::string &caseName)
{
    ASSERT_EQ(npuResult.bufferSizeRet, ACL_SPARSE_STATUS_SUCCESS);
    ASSERT_EQ(npuResult.nnzRet, ACL_SPARSE_STATUS_SUCCESS);
    ASSERT_EQ(npuResult.computeRet, ACL_SPARSE_STATUS_SUCCESS);
    EXPECT_EQ(npuResult.nnzC, golden.nnzC);

    if (npuResult.nnzC == 0 && golden.nnzC == 0) {
        return;
    }

    // rowPtrC: INTEGER bit-exact
    {
        VerifyConfig intCfg;
        intCfg.SetMode(PrecisionMode::INTEGER);
        bool rowPtrPass = Verifier::verifyVector(
            Verifier::toFloat(npuResult.rowPtrC),
            Verifier::toFloat(golden.rowPtrC), intCfg, caseName + "_rowPtrC");
        EXPECT_TRUE(rowPtrPass);
    }

    // colIndC: INTEGER bit-exact
    if (npuResult.nnzC > 0 && golden.nnzC > 0) {
        VerifyConfig intCfg;
        intCfg.SetMode(PrecisionMode::INTEGER);
        bool colIndPass = Verifier::verifyVector(
            Verifier::toFloat(npuResult.colIndC),
            Verifier::toFloat(golden.colIndC), intCfg, caseName + "_colIndC");
        EXPECT_TRUE(colIndPass);
    }
}

// Test body: Success-path parameterized test
TEST_P(CsrGeam2Test, CsrGeam2Success) {
    const auto &p = param_;
    bool useHostMode = (p.pointer_mode == "HOST");

    std::cout << "==== " << p.case_name
              << " ==== m=" << p.m << " n=" << p.n
              << " alpha=" << p.alpha << " beta=" << p.beta
              << " idxA=" << p.index_base_a << " idxB=" << p.index_base_b
              << " idxC=" << p.index_base_c << " ptrMode=" << p.pointer_mode
              << " pattern=" << p.pattern << "\n";

    // Generate input CSR matrices A and B
    auto [csrA, csrB] = PrepareTestInputs(p);
    int nnzA_0 = static_cast<int>(csrA.nnz);
    int nnzB_0 = static_cast<int>(csrB.nnz);

    // Compute golden reference
    auto golden = CsrGeam2Golden(csrA, csrB, p.alpha, p.beta,
                                  p.index_base_a, p.index_base_b, p.index_base_c);
    std::cout << "  nnzA=" << nnzA_0 << " nnzB=" << nnzB_0
              << " golden.nnzC=" << golden.nnzC << "\n";

    // Call NPU three-step workflow
    HandleManager handle;
    auto npuResult = RunCsrGeam2Npu(handle, stream_, csrA, nnzA_0, csrB, nnzB_0,
                                    p, useHostMode);

    // Verify correctness
    VerifyCsrGeam2Result(npuResult, golden, p.case_name);

    if (npuResult.nnzC > 0 && golden.nnzC > 0) {
        VerifyConfig valCfg;
        valCfg.SetMode(PrecisionMode::MERE_MARE)
              .SetMERE(p.mere_threshold)
              .SetMARE(p.mare_multiplier);
        bool valPass = Verifier::verifyVector(npuResult.valuesC, golden.valuesC,
                                               valCfg, p.case_name + "_valuesC");
        EXPECT_TRUE(valPass);
    }

    std::cout << "[" << p.case_name << "] PASSED (nnzC=" << golden.nnzC << ")\n";
}

// Parameterized test instantiation from CSV (loaded via public csv_loader.h)
INSTANTIATE_TEST_SUITE_P(
    CsrGeam2Cases,
    CsrGeam2Test,
    testing::ValuesIn(GetCasesFromCsv<CsrGeam2TestParam>("csrgeam2_test.csv")),
    [](const testing::TestParamInfo<CsrGeam2TestParam> &info) {
        return info.param.case_name;
    }
);

// Exception test fixture: CsrGeam2ExceptionTest
class CsrGeam2ExceptionTest : public testing::Test {
public:
    static void SetUpTestSuite() {
        env_ = std::make_unique<AclEnvScope>();
    }

    static void TearDownTestSuite() {
        env_.reset();
    }

protected:
    inline static std::unique_ptr<AclEnvScope> env_;

    void SetUp() override {
        stream_ = env_->stream();
        handle_ = std::make_unique<HandleManager>();
        handle_->setStream(stream_);

        // Prepare a small 4x4 CSR matrix for valid parameter tests
        csr0_ = makeSparseCsr(4, 4, 0.5, 42);
        nnz_ = static_cast<int>(csr0_.nnz);

        // Create valid MatDescr
        descrA_ = std::make_unique<MatDescrGuard>();
        descrB_ = std::make_unique<MatDescrGuard>();
        descrC_ = std::make_unique<MatDescrGuard>();

        // Set host pointer mode for simpler test setup
        aclsparseSetPointerMode(handle_->get(), ACL_SPARSE_POINTER_MODE_HOST);

        alpha_ = 1.0f;
        beta_ = 1.0f;
    }

    void TearDown() override {
        descrC_.reset();
        descrB_.reset();
        descrA_.reset();
        handle_.reset();
    }

    // Inner builder class: encapsulates aclsparseScsrgeam2_bufferSizeExt call
    // with default 4x4 fixture params; fluent methods allow per-test overrides.
    class BufferSizeExtBuilder {
    public:
        explicit BufferSizeExtBuilder(CsrGeam2ExceptionTest &t)
            : t_(t), handle_(t.handle_->get()), m_(4), n_(4),
              alpha_(&t.alpha_), beta_(&t.beta_),
              descrA_(t.descrA_->cget()), descrB_(t.descrB_->cget()),
              descrC_(t.descrC_->cget()),
              nnzA_(t.nnz_), nnzB_(t.nnz_),
              valA_(t.csr0_.values.data()),
              rowPtrA_(t.csr0_.rowOffsets.data()),
              colIndA_(t.csr0_.colIndices.data()),
              valB_(t.csr0_.values.data()),
              rowPtrB_(t.csr0_.rowOffsets.data()),
              colIndB_(t.csr0_.colIndices.data()),
              bufSize_(&bufSizeLocal_),
              valC_(nullptr), rowPtrC_(nullptr), colIndC_(nullptr) {}

        // Override handle and matrix descriptors
        BufferSizeExtBuilder &WithHandle(aclsparseHandle_t h) {
            handle_ = h;
            return *this;
        }
        BufferSizeExtBuilder &WithDescrA(aclsparseMatDescr_t d) {
            descrA_ = d;
            return *this;
        }
        BufferSizeExtBuilder &WithDescrB(aclsparseMatDescr_t d) {
            descrB_ = d;
            return *this;
        }
        BufferSizeExtBuilder &WithDescrC(aclsparseMatDescr_t d) {
            descrC_ = d;
            return *this;
        }

        // Override matrix dimensions and nnz counts
        BufferSizeExtBuilder &WithDims(int m, int n) {
            m_ = m;
            n_ = n;
            return *this;
        }
        BufferSizeExtBuilder &WithNnzA(int n) {
            nnzA_ = n;
            return *this;
        }
        BufferSizeExtBuilder &WithNnzB(int n) {
            nnzB_ = n;
            return *this;
        }

        // Override scalar pointers
        BufferSizeExtBuilder &WithAlpha(const float *p) {
            alpha_ = p;
            return *this;
        }
        BufferSizeExtBuilder &WithBeta(const float *p) {
            beta_ = p;
            return *this;
        }

        // Override data pointers (explicit nullptr allowed)
        BufferSizeExtBuilder &WithValA(const float *p) {
            valA_ = p;
            return *this;
        }
        BufferSizeExtBuilder &WithRowPtrA(const int *p) {
            rowPtrA_ = p;
            return *this;
        }
        BufferSizeExtBuilder &WithColIndA(const int *p) {
            colIndA_ = p;
            return *this;
        }
        BufferSizeExtBuilder &WithValB(const float *p) {
            valB_ = p;
            return *this;
        }
        BufferSizeExtBuilder &WithRowPtrB(const int *p) {
            rowPtrB_ = p;
            return *this;
        }
        BufferSizeExtBuilder &WithColIndB(const int *p) {
            colIndB_ = p;
            return *this;
        }

        // Override workspace / output pointers
        BufferSizeExtBuilder &WithBufferSize(size_t *s) {
            bufSize_ = s;
            return *this;
        }
        BufferSizeExtBuilder &WithOutputs(float *vC, int *rC, int *cC) {
            valC_ = vC;
            rowPtrC_ = rC;
            colIndC_ = cC;
            return *this;
        }

        aclsparseStatus_t Call() {
            return aclsparseScsrgeam2_bufferSizeExt(
                handle_, m_, n_, alpha_,
                descrA_, nnzA_, valA_, rowPtrA_, colIndA_,
                beta_,
                descrB_, nnzB_, valB_, rowPtrB_, colIndB_,
                descrC_, valC_, rowPtrC_, colIndC_, bufSize_);
        }

    private:
        CsrGeam2ExceptionTest &t_;
        aclsparseHandle_t handle_;
        int m_, n_;
        const float *alpha_, *beta_;
        aclsparseMatDescr_t descrA_, descrB_, descrC_;
        int nnzA_, nnzB_;
        const float *valA_, *valB_;
        const int *rowPtrA_, *rowPtrB_;
        const int *colIndA_, *colIndB_;
        size_t bufSizeLocal_ = 0;
        size_t *bufSize_;
        float *valC_;
        int *rowPtrC_;
        int *colIndC_;
    };

    aclrtStream stream_ = nullptr;
    std::unique_ptr<HandleManager> handle_;
    CsrMatrix csr0_;
    int nnz_ = 0;
    std::unique_ptr<MatDescrGuard> descrA_;
    std::unique_ptr<MatDescrGuard> descrB_;
    std::unique_ptr<MatDescrGuard> descrC_;
    float alpha_ = 1.0f;
    float beta_ = 1.0f;
};

// Exception 1: NULL handle -> ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR
TEST_F(CsrGeam2ExceptionTest, NullHandle) {
    EXPECT_EQ(BufferSizeExtBuilder(*this).WithHandle(nullptr).Call(),
              ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR);
}

// Exception 2: NULL descrA -> ACL_SPARSE_STATUS_INVALID_VALUE
TEST_F(CsrGeam2ExceptionTest, NullDescrA) {
    EXPECT_EQ(BufferSizeExtBuilder(*this).WithDescrA(nullptr).Call(),
              ACL_SPARSE_STATUS_INVALID_VALUE);
}

// Exception 3: Invalid matrix type (SYMMETRIC) -> ACL_SPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED
TEST_F(CsrGeam2ExceptionTest, InvalidMatrixType) {
    aclsparseMatDescr_t badDescr = nullptr;
    aclsparseCreateMatDescr(&badDescr);
    aclsparseSetMatType(badDescr, ACL_SPARSE_MATRIX_TYPE_SYMMETRIC);
    aclsparseSetMatIndexBase(badDescr, ACL_SPARSE_INDEX_BASE_ZERO);

    auto ret = BufferSizeExtBuilder(*this).WithDescrA(badDescr).Call();
    EXPECT_EQ(ret, ACL_SPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED);
    aclsparseDestroyMatDescr(badDescr);
}

// Exception 4: m=0, n>0 -> ACL_SPARSE_STATUS_SUCCESS (early return for empty matrix)
TEST_F(CsrGeam2ExceptionTest, MZero_valid) {
    EXPECT_EQ(BufferSizeExtBuilder(*this).WithDims(0, 4).WithNnzA(0).Call(),
              ACL_SPARSE_STATUS_SUCCESS);
}

// Exception 5: m>0, n=0 -> ACL_SPARSE_STATUS_SUCCESS (early return for empty matrix)
TEST_F(CsrGeam2ExceptionTest, NZero_valid) {
    EXPECT_EQ(BufferSizeExtBuilder(*this).WithDims(4, 0).WithNnzB(0).Call(),
              ACL_SPARSE_STATUS_SUCCESS);
}

// Exception 6: NULL descrB -> ACL_SPARSE_STATUS_INVALID_VALUE
TEST_F(CsrGeam2ExceptionTest, NullDescrB) {
    EXPECT_EQ(BufferSizeExtBuilder(*this).WithDescrB(nullptr).Call(),
              ACL_SPARSE_STATUS_INVALID_VALUE);
}

// Exception 7: NULL descrC -> ACL_SPARSE_STATUS_INVALID_VALUE
TEST_F(CsrGeam2ExceptionTest, NullDescrC) {
    EXPECT_EQ(BufferSizeExtBuilder(*this).WithDescrC(nullptr).Call(),
              ACL_SPARSE_STATUS_INVALID_VALUE);
}

// Exception 8: m>0, NULL rowPtrA -> ACL_SPARSE_STATUS_INVALID_VALUE
TEST_F(CsrGeam2ExceptionTest, NullRowPtrA_mPositive) {
    EXPECT_EQ(BufferSizeExtBuilder(*this).WithRowPtrA(nullptr).Call(),
              ACL_SPARSE_STATUS_INVALID_VALUE);
}

// Exception 9: m>0, NULL rowPtrB -> ACL_SPARSE_STATUS_INVALID_VALUE
TEST_F(CsrGeam2ExceptionTest, NullRowPtrB_mPositive) {
    EXPECT_EQ(BufferSizeExtBuilder(*this).WithRowPtrB(nullptr).Call(),
              ACL_SPARSE_STATUS_INVALID_VALUE);
}

// Exception 10: NULL rowPtrC -> ACL_SPARSE_STATUS_INVALID_VALUE
// Tests via Xcsrgeam2Nnz where csrRowPtrC is a required output parameter.
TEST_F(CsrGeam2ExceptionTest, NullRowPtrC) {
    size_t bufSize = 0;
    auto retBuf = BufferSizeExtBuilder(*this).WithBufferSize(&bufSize).Call();
    ASSERT_EQ(retBuf, ACL_SPARSE_STATUS_SUCCESS);
    if (bufSize == 0) bufSize = 16;
    auto dWorkspace = DeviceBuffer::alloc(bufSize);

    int nnzC = 0;
    auto ret = aclsparseXcsrgeam2Nnz(
        handle_->get(), 4, 4,
        descrA_->cget(), nnz_,
        csr0_.rowOffsets.data(),
        csr0_.colIndices.data(),
        descrB_->cget(), nnz_,
        csr0_.rowOffsets.data(),
        csr0_.colIndices.data(),
        descrC_->cget(),
        nullptr,  // csrRowPtrC = null
        &nnzC,
        dWorkspace.get());
    EXPECT_EQ(ret, ACL_SPARSE_STATUS_INVALID_VALUE);
}

// Exception 11 (relaxed #13): bufferSizeExt allows NULL valA even when nnzA>0,
// matching cuSPARSE bufferSizeExt convention (workspace calculation doesn't use valA).
// The null-pointer check for valA is enforced by aclsparseScsrgeam2 (kernel launch).
TEST_F(CsrGeam2ExceptionTest, NullValA_bufferSizeExtRelaxed) {
    EXPECT_EQ(BufferSizeExtBuilder(*this).WithValA(nullptr).Call(),
              ACL_SPARSE_STATUS_SUCCESS);
}

// Exception 12 (relaxed #13): Same for valB — bufferSizeExt allows NULL valB.
TEST_F(CsrGeam2ExceptionTest, NullValB_bufferSizeExtRelaxed) {
    EXPECT_EQ(BufferSizeExtBuilder(*this).WithValB(nullptr).Call(),
              ACL_SPARSE_STATUS_SUCCESS);
}

// Exception 13: NULL alpha -> ACL_SPARSE_STATUS_INVALID_VALUE
TEST_F(CsrGeam2ExceptionTest, NullAlpha) {
    const float *nullAlpha = nullptr;
    EXPECT_EQ(BufferSizeExtBuilder(*this).WithAlpha(nullAlpha).Call(),
              ACL_SPARSE_STATUS_INVALID_VALUE);
}

// Exception 14: NULL beta -> ACL_SPARSE_STATUS_INVALID_VALUE
TEST_F(CsrGeam2ExceptionTest, NullBeta) {
    const float *nullBeta = nullptr;
    EXPECT_EQ(BufferSizeExtBuilder(*this).WithBeta(nullBeta).Call(),
              ACL_SPARSE_STATUS_INVALID_VALUE);
}

