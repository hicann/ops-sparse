/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root directory of the software repository for the full text of the License.
 */

/**
 * @file csr2gebsr_test.cpp
 * @brief GTest + CSV-driven test cases for aclsparseCsr2gebsr (Legacy API, three-step).
 *
 * Tests CSR -> GEBSR sparse format conversion.
 *
 * Test structure:
 *   - TEST_P (Csr2GebsrTest)          : parameterized success-path tests from CSV
 *   - TEST_F (Csr2GebsrExceptionTest) : null-pointer / invalid-param error tests
 *
 * Test parameters are loaded from csr2gebsr_test.csv (copied to build dir by
 * CMake). Entry point is shared via test/frame/test_main.cpp.
 */

#include "test_common.h"
#include "csr2gebsr_golden.h"
#include "csr2gebsr_npu_wrapper.h"
#include "csr2gebsr_param.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <sstream>

using namespace sparse_test;

// ============================================================================
// CSR data preparation: convert fill.h 0-based CSR to desired index base
// ============================================================================

static CsrMatrix PrepareCsr(const CsrMatrix& csr0, int indexBase) {
    CsrMatrix csr = csr0;
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

// ============================================================================
// Common base fixture: shared AclEnvScope lifecycle (eliminates duplication)
// ============================================================================

class Csr2GebsrTestBase : public testing::Test {
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

    void SetUp() override {
        stream_ = env_->stream();
    }
};

// ============================================================================
// GTest parameterized fixture: Csr2GebsrTest
// ============================================================================

class Csr2GebsrTest : public Csr2GebsrTestBase, public testing::WithParamInterface<Csr2GebsrTestParam> {
public:
    static void SetUpTestSuite() { Csr2GebsrTestBase::SetUpTestSuite(); }
    static void TearDownTestSuite() { Csr2GebsrTestBase::TearDownTestSuite(); }

protected:
    Csr2GebsrTestParam param_;

    void SetUp() override {
        Csr2GebsrTestBase::SetUp();
        param_ = GetParam();
    }
};

// ============================================================================
// Helper: Generate CSR data based on test parameters
// ============================================================================

static CsrMatrix GenerateCsr(const Csr2GebsrTestParam& p) {
    CsrMatrix csr0;

    // Determine generation method
    bool isDiag = (p.case_name.find("diag") != std::string::npos);
    bool isEmpty = (p.sparsity >= 1.0);
    bool hasEmptyRow = (p.empty_row_prob > 0.0);

    if (isDiag) {
        // Diagonal matrix (uses makeDiagCsr, ignores sparsity/empty_row_prob)
        int n = std::min(p.m, p.n);
        csr0 = makeDiagCsr(n);
        // makeDiagCsr creates n x n; for non-square diag the golden handles
        // the padding, so no extra adjustment is needed here.
    } else if (isEmpty) {
        // Empty matrix (sparsity = 1.0 -> nnz = 0)
        csr0 = makeEmptyCsr(p.m, p.n);
    } else if (hasEmptyRow) {
        // Matrix with empty rows (uses SparseFillGenerator directly)
        SparseFillGenerator gen(p.seed);
        gen.setSparsity(p.sparsity);
        gen.setEmptyRowProb(p.empty_row_prob);
        csr0 = gen.generateCsr(p.m, p.n);
    } else {
        // Standard random sparse matrix
        csr0 = makeSparseCsr(p.m, p.n, p.sparsity, p.seed);
    }

    return csr0;
}

// ============================================================================
// Sub-functions extracted from Csr2GebsrSuccess
// ============================================================================

// Prepared CSR data for golden and NPU paths
struct CsrData {
    CsrMatrix csr0;          // raw 0-based CSR from fill.h
    int nnz;                 // nonzero count
    CsrMatrix csrForGolden;  // type-round-tripped for golden comparison
};

static CsrData PrepareCsrData(const Csr2GebsrTestParam& p) {
    CsrData d;
    d.csr0 = GenerateCsr(p);
    d.nnz = static_cast<int>(d.csr0.nnz);

    // Apply type round-trip for non-FP32 dtypes
    // (ensures golden and NPU see the same effective values)
    d.csrForGolden = d.csr0;
    if (p.dtype != "FP32") {
        ApplyTypeRoundTrip(d.csrForGolden.values, p.dtype);
    }
    return d;
}

static GebsrResult ComputeGolden(const Csr2GebsrTestParam& p,
                                  const CsrMatrix& csrForGolden) {
    bool dirRow = (p.dir == "ROW");
    return Csr2GebsrGolden(
        csrForGolden,  // golden uses 0-based CSR data internally
        p.m, p.n,
        p.row_block_dim, p.col_block_dim,
        dirRow,
        0,              // golden input is always 0-based
        p.index_base_c);
}

static Csr2GebsrNpuResult RunNpu(const Csr2GebsrTestParam& p, aclrtStream stream,
                                  const CsrMatrix& csrForGolden, int nnz) {
    HandleManager handle;
    handle.setStream(stream);

    // NPU needs the CSR data with the correct index base
    // But the values should be the type-round-tripped version
    CsrMatrix csrForNpu = PrepareCsr(csrForGolden, p.index_base_a);

    bool dirRow = (p.dir == "ROW");
    return Csr2GebsrNpu(
        handle, stream,
        csrForNpu.rowOffsets, csrForNpu.colIndices, csrForNpu.values,
        p.m, p.n, nnz,
        p.row_block_dim, p.col_block_dim,
        dirRow ? ACL_SPARSE_DIRECTION_ROW : ACL_SPARSE_DIRECTION_COLUMN,
        p.index_base_a, p.index_base_c,
        p.dtype);
}

// Verify bsrRowPtrC using INTEGER bit-exact comparison
static void VerifyBsrRowPtr(const Csr2GebsrTestParam& p,
                            const Csr2GebsrNpuResult& npu,
                            const GebsrResult& golden) {
    VerifyConfig intCfg;
    intCfg.SetMode(PrecisionMode::INTEGER);
    bool rowPtrPass = Verifier::verifyVector(
        Verifier::toFloat(npu.bsrRowPtr),
        Verifier::toFloat(golden.bsrRowPtr),
        intCfg, p.case_name + "_bsrRowPtr");
    EXPECT_TRUE(rowPtrPass);
}

// Verify bsrColIndC and bsrValC when nnzb > 0
static void VerifyNonEmptyOutput(const Csr2GebsrTestParam& p,
                                 const Csr2GebsrNpuResult& npu,
                                 const GebsrResult& golden) {
    // Verify bsrRowPtrC (INTEGER bit-exact)
    VerifyBsrRowPtr(p, npu, golden);

    // Verify bsrColIndC (INTEGER bit-exact)
    {
        VerifyConfig intCfg;
        intCfg.SetMode(PrecisionMode::INTEGER);
        bool colIndPass = Verifier::verifyVector(
            Verifier::toFloat(npu.bsrColInd),
            Verifier::toFloat(golden.bsrColInd),
            intCfg, p.case_name + "_bsrColInd");
        EXPECT_TRUE(colIndPass);
    }

    // Verify bsrValC (EXACT bit-exact for float, INTEGER for int32)
    {
        VerifyConfig valCfg;
        if (p.dtype == "INT32") {
            valCfg.SetMode(PrecisionMode::INTEGER);
        } else {
            valCfg.SetMode(PrecisionMode::EXACT);
        }
        bool valPass = Verifier::verifyVector(
            npu.bsrVal, golden.bsrVal,
            valCfg, p.case_name + "_bsrVal");
        EXPECT_TRUE(valPass);
    }
}

// ============================================================================
// TEST_P: Success-path parameterized test
// ============================================================================

// Verify bufferSize value
static void VerifyBufferSize(const Csr2GebsrTestParam& p, size_t actual) {
    if (p.m == 0) {
        EXPECT_EQ(actual, 0u) << "empty matrix (m==0) should have bufferSize=0";
        return;
    }
    int mb = (p.m + p.row_block_dim - 1) / p.row_block_dim;
    int nb = (p.n + p.col_block_dim - 1) / p.col_block_dim;
    int32_t devId = 0;
    aclrtGetDevice(&devId);
    int64_t aivCoreNum = 0;
    aclrtGetDeviceInfo(static_cast<uint32_t>(devId),
                       ACL_DEV_ATTR_VECTOR_CORE_NUM, &aivCoreNum);
    size_t expected = (static_cast<size_t>(mb) + 1 +
        static_cast<size_t>(mb) * static_cast<size_t>(nb) +
        static_cast<size_t>(aivCoreNum)) * sizeof(int32_t);
    EXPECT_EQ(actual, expected) << "bufferSize mismatch";
}

TEST_P(Csr2GebsrTest, Csr2GebsrSuccess) {
    const auto& p = param_;

    std::cout << "==== " << p.case_name
              << " ==== m=" << p.m << " n=" << p.n
              << " dtype=" << p.dtype
              << " block=" << p.row_block_dim << "x" << p.col_block_dim
              << " dir=" << p.dir
              << " idxA=" << p.index_base_a << " idxC=" << p.index_base_c
              << "\n";

    auto csrData = PrepareCsrData(p);
    std::cout << "  nnz=" << csrData.nnz << "\n";

    auto golden = ComputeGolden(p, csrData.csrForGolden);
    std::cout << "  golden.nnzb=" << golden.nnzb << "\n";

#ifdef SPARSE_TEST_USE_EIGEN
    if (p.m > 0 && p.n > 0 && csrData.csrForGolden.nnz > 0) {
        bool dirRow = (p.dir == "ROW");
        Csr2GebsrEigenCrossCheck(
            csrData.csrForGolden, p.m, p.n,
            p.row_block_dim, p.col_block_dim,
            dirRow, 0, p.index_base_c, golden);
    }
#endif

    auto npuResult = RunNpu(p, stream_, csrData.csrForGolden, csrData.nnz);

    ASSERT_EQ(npuResult.bufferSizeRet, ACL_SPARSE_STATUS_SUCCESS) << "bufferSize step failed";
    ASSERT_EQ(npuResult.nnzRet, ACL_SPARSE_STATUS_SUCCESS) << "Nnz step failed";
    ASSERT_EQ(npuResult.convertRet, ACL_SPARSE_STATUS_SUCCESS) << "Convert step failed";

    VerifyBufferSize(p, npuResult.bufferSize);

    // 5. Verify nnzb
    EXPECT_EQ(npuResult.nnzb, golden.nnzb)
        << "nnzb mismatch: npu=" << npuResult.nnzb << " golden=" << golden.nnzb;

    if (npuResult.nnzb == 0 && golden.nnzb == 0) {
        // Both empty -- verify bsrRowPtr only
        VerifyBsrRowPtr(p, npuResult, golden);
        std::cout << "[" << p.case_name << "] PASSED (nnzb=0)\n";
        return;
    }

    // 6. Verify full output
    VerifyNonEmptyOutput(p, npuResult, golden);
    std::cout << "[" << p.case_name << "] PASSED (nnzb=" << golden.nnzb << ")\n";
}

// ============================================================================
// Parameterized test instantiation from CSV
// ============================================================================

INSTANTIATE_TEST_SUITE_P(
    Csr2GebsrCases,
    Csr2GebsrTest,
    testing::ValuesIn(GetCasesFromCsv<Csr2GebsrTestParam>("csr2gebsr_test.csv")),
    [](const testing::TestParamInfo<Csr2GebsrTestParam>& info) {
        return info.param.case_name;
    }
);

// ============================================================================
// Exception test fixture: Csr2GebsrExceptionTest
// ============================================================================

class Csr2GebsrExceptionTest : public Csr2GebsrTestBase {
public:
    static void SetUpTestSuite() { Csr2GebsrTestBase::SetUpTestSuite(); }
    static void TearDownTestSuite() { Csr2GebsrTestBase::TearDownTestSuite(); }

protected:
    void SetUp() override {
        Csr2GebsrTestBase::SetUp();
        handle_ = std::make_unique<HandleManager>();
        handle_->setStream(stream_);

        // Prepare a small 4x4 CSR matrix for valid parameter tests
        csr0_ = makeSparseCsr(4, 4, 0.5, 42);
        nnz_ = static_cast<int>(csr0_.nnz);

        // Create valid MatDescr
        descrA_ = std::make_unique<MatDescrGuard>();
        descrC_ = std::make_unique<MatDescrGuard>();

        // Default parameters
        m_ = 4;
        n_ = 4;
        rowBlockDim_ = 2;
        colBlockDim_ = 2;
        dir_ = ACL_SPARSE_DIRECTION_ROW;
    }

    void TearDown() override {
        descrC_.reset();
        descrA_.reset();
        handle_.reset();
    }

    // ------------------------------------------------------------------
    // Common helpers for exception tests that run bufferSize + Nnz
    // ------------------------------------------------------------------

    // Query bufferSize (common first step for all Nnz-step exception tests)
    struct BufferSizeResult {
        aclsparseStatus_t status;
        size_t bufSize;
    };
    BufferSizeResult QueryBufferSize() {
        BufferSizeResult r{};
        r.bufSize = 0;
        r.status = aclsparseScsr2gebsr_bufferSize(
            handle_->get(), dir_, m_, n_, descrA_->cget(),
            csr0_.values.data(), csr0_.rowOffsets.data(), csr0_.colIndices.data(),
            rowBlockDim_, colBlockDim_, &r.bufSize);
        return r;
    }

    // Allocate workspace buffer (bufSize clamped to minimum 16)
    DeviceBuffer AllocWorkspace(size_t bufSize) {
        if (bufSize == 0) bufSize = 16;
        return DeviceBuffer::alloc(bufSize);
    }

    // Allocate bsrRowPtrC on device
    DeviceBuffer AllocBsrRowPtr() {
        int mb = (m_ + rowBlockDim_ - 1) / rowBlockDim_;
        return DeviceBuffer::alloc((mb + 1) * sizeof(int32_t));
    }

    // Copy CSR input data to device buffers (for Convert-step tests that
    // need device pointers)
    struct DeviceCsr {
        DeviceBuffer dRowPtrA;
        DeviceBuffer dColIndA;
        DeviceBuffer dValA;
    };
    DeviceCsr CopyCsrToDevice() {
        DeviceCsr d;
        d.dRowPtrA = DeviceBuffer::copyFrom(csr0_.rowOffsets.data(),
            csr0_.rowOffsets.size() * sizeof(int32_t));
        d.dColIndA = DeviceBuffer::copyFrom(csr0_.colIndices.data(),
            csr0_.colIndices.size() * sizeof(int32_t));
        d.dValA = DeviceBuffer::copyFrom(csr0_.values.data(),
            csr0_.values.size() * sizeof(float));
        return d;
    }

    // ------------------------------------------------------------------

    // Helper: run Nnz + alloc Convert output buffers (shared by Convert-step exception tests)
    struct ConvertPrereq {
        DeviceBuffer dWorkspace;
        DeviceBuffer dBsrRowPtr;
        DeviceBuffer dBsrColInd;
        DeviceBuffer dBsrVal;
        int nnzb;
        bool skip;
    };
    ConvertPrereq PrepareConvertTest() {
        ConvertPrereq pr{};
        auto bs = QueryBufferSize();
        if (bs.status != ACL_SPARSE_STATUS_SUCCESS) { pr.skip = true; return pr; }
        pr.dWorkspace = AllocWorkspace(bs.bufSize);
        auto dCsr = CopyCsrToDevice();
        pr.dBsrRowPtr = AllocBsrRowPtr();
        pr.nnzb = 0;
        auto retNnz = aclsparseXcsr2gebsrNnz(
            handle_->get(), dir_, m_, n_, descrA_->cget(),
            reinterpret_cast<int*>(dCsr.dRowPtrA.get()),
            reinterpret_cast<int*>(dCsr.dColIndA.get()),
            descrC_->cget(), reinterpret_cast<int*>(pr.dBsrRowPtr.get()),
            rowBlockDim_, colBlockDim_, &pr.nnzb, pr.dWorkspace.get());
        if (retNnz != ACL_SPARSE_STATUS_SUCCESS) { pr.skip = true; return pr; }
        aclrtSynchronizeStream(stream_);
        if (pr.nnzb <= 0) { pr.skip = true; return pr; }
        pr.dBsrColInd = DeviceBuffer::alloc(pr.nnzb * sizeof(int32_t));
        pr.dBsrVal = DeviceBuffer::alloc(pr.nnzb * rowBlockDim_ * colBlockDim_ * sizeof(float));
        return pr;
    }

    // ------------------------------------------------------------------

    aclrtStream stream_ = nullptr;
    std::unique_ptr<HandleManager> handle_;
    CsrMatrix csr0_;
    int nnz_ = 0;
    std::unique_ptr<MatDescrGuard> descrA_;
    std::unique_ptr<MatDescrGuard> descrC_;
    int m_ = 4;
    int n_ = 4;
    int rowBlockDim_ = 2;
    int colBlockDim_ = 2;
    aclsparseDirection_t dir_ = ACL_SPARSE_DIRECTION_ROW;
};

// Exception 1: NULL handle
TEST_F(Csr2GebsrExceptionTest, NullHandle) {
    size_t bufSize = 0;
    auto ret = aclsparseScsr2gebsr_bufferSize(
        nullptr, dir_, m_, n_, descrA_->cget(),
        csr0_.values.data(), csr0_.rowOffsets.data(), csr0_.colIndices.data(),
        rowBlockDim_, colBlockDim_, &bufSize);
    EXPECT_EQ(ret, ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR);
}

// Exception 2: Invalid m (negative)
TEST_F(Csr2GebsrExceptionTest, InvalidM) {
    size_t bufSize = 0;
    auto ret = aclsparseScsr2gebsr_bufferSize(
        handle_->get(), dir_, -1, n_, descrA_->cget(),
        csr0_.values.data(), csr0_.rowOffsets.data(), csr0_.colIndices.data(),
        rowBlockDim_, colBlockDim_, &bufSize);
    EXPECT_EQ(ret, ACL_SPARSE_STATUS_INVALID_VALUE);
}

// Exception 3: Invalid n (negative)
TEST_F(Csr2GebsrExceptionTest, InvalidN) {
    size_t bufSize = 0;
    auto ret = aclsparseScsr2gebsr_bufferSize(
        handle_->get(), dir_, m_, -1, descrA_->cget(),
        csr0_.values.data(), csr0_.rowOffsets.data(), csr0_.colIndices.data(),
        rowBlockDim_, colBlockDim_, &bufSize);
    EXPECT_EQ(ret, ACL_SPARSE_STATUS_INVALID_VALUE);
}

// Exception 4: Invalid rowBlockDim (zero)
TEST_F(Csr2GebsrExceptionTest, InvalidRowBlockDim) {
    size_t bufSize = 0;
    auto ret = aclsparseScsr2gebsr_bufferSize(
        handle_->get(), dir_, m_, n_, descrA_->cget(),
        csr0_.values.data(), csr0_.rowOffsets.data(), csr0_.colIndices.data(),
        0, colBlockDim_, &bufSize);
    EXPECT_EQ(ret, ACL_SPARSE_STATUS_INVALID_VALUE);
}

// Exception 5: Invalid colBlockDim (zero)
TEST_F(Csr2GebsrExceptionTest, InvalidColBlockDim) {
    size_t bufSize = 0;
    auto ret = aclsparseScsr2gebsr_bufferSize(
        handle_->get(), dir_, m_, n_, descrA_->cget(),
        csr0_.values.data(), csr0_.rowOffsets.data(), csr0_.colIndices.data(),
        rowBlockDim_, 0, &bufSize);
    EXPECT_EQ(ret, ACL_SPARSE_STATUS_INVALID_VALUE);
}

// Exception 6: Invalid direction
TEST_F(Csr2GebsrExceptionTest, InvalidDir) {
    size_t bufSize = 0;
    auto ret = aclsparseScsr2gebsr_bufferSize(
        handle_->get(), static_cast<aclsparseDirection_t>(99),
        m_, n_, descrA_->cget(),
        csr0_.values.data(), csr0_.rowOffsets.data(), csr0_.colIndices.data(),
        rowBlockDim_, colBlockDim_, &bufSize);
    EXPECT_EQ(ret, ACL_SPARSE_STATUS_INVALID_VALUE);
}

// Exception 7: NULL descrA
TEST_F(Csr2GebsrExceptionTest, NullDescrA) {
    size_t bufSize = 0;
    auto ret = aclsparseScsr2gebsr_bufferSize(
        handle_->get(), dir_, m_, n_, nullptr,
        csr0_.values.data(), csr0_.rowOffsets.data(), csr0_.colIndices.data(),
        rowBlockDim_, colBlockDim_, &bufSize);
    EXPECT_EQ(ret, ACL_SPARSE_STATUS_INVALID_VALUE);
}

// Exception 8: NULL descrC (tested via Nnz step)
TEST_F(Csr2GebsrExceptionTest, NullDescrC) {
    auto bs = QueryBufferSize();
    ASSERT_EQ(bs.status, ACL_SPARSE_STATUS_SUCCESS);
    auto dWorkspace = AllocWorkspace(bs.bufSize);
    auto dBsrRowPtr = AllocBsrRowPtr();
    int nnzb = 0;

    auto ret = aclsparseXcsr2gebsrNnz(
        handle_->get(), dir_, m_, n_,
        descrA_->cget(),
        csr0_.rowOffsets.data(), csr0_.colIndices.data(),
        nullptr,  // descrC = null
        reinterpret_cast<int*>(dBsrRowPtr.get()),
        rowBlockDim_, colBlockDim_,
        &nnzb, dWorkspace.get());
    EXPECT_EQ(ret, ACL_SPARSE_STATUS_INVALID_VALUE);
}

// Exception 9: NULL csrRowPtrA (m > 0)
TEST_F(Csr2GebsrExceptionTest, NullRowPtr) {
    size_t bufSize = 0;
    auto ret = aclsparseScsr2gebsr_bufferSize(
        handle_->get(), dir_, m_, n_, descrA_->cget(),
        csr0_.values.data(), nullptr, csr0_.colIndices.data(),
        rowBlockDim_, colBlockDim_, &bufSize);
    EXPECT_EQ(ret, ACL_SPARSE_STATUS_INVALID_VALUE);
}

// Exception 10: NULL bsrRowPtrC (tested via Nnz step)
TEST_F(Csr2GebsrExceptionTest, NullBsrRowPtr) {
    auto bs = QueryBufferSize();
    ASSERT_EQ(bs.status, ACL_SPARSE_STATUS_SUCCESS);
    auto dWorkspace = AllocWorkspace(bs.bufSize);
    int nnzb = 0;

    auto ret = aclsparseXcsr2gebsrNnz(
        handle_->get(), dir_, m_, n_,
        descrA_->cget(),
        csr0_.rowOffsets.data(), csr0_.colIndices.data(),
        descrC_->cget(),
        nullptr,  // bsrRowPtrC = null
        rowBlockDim_, colBlockDim_,
        &nnzb, dWorkspace.get());
    EXPECT_EQ(ret, ACL_SPARSE_STATUS_INVALID_VALUE);
}

// Exception 11: Matrix type not GENERAL
TEST_F(Csr2GebsrExceptionTest, MatrixTypeNotGeneral) {
    aclsparseMatDescr_t badDescr = nullptr;
    ASSERT_EQ(aclsparseCreateMatDescr(&badDescr), ACL_SPARSE_STATUS_SUCCESS);
    aclsparseSetMatType(badDescr, ACL_SPARSE_MATRIX_TYPE_SYMMETRIC);
    aclsparseSetMatIndexBase(badDescr, ACL_SPARSE_INDEX_BASE_ZERO);

    size_t bufSize = 0;
    auto ret = aclsparseScsr2gebsr_bufferSize(
        handle_->get(), dir_, m_, n_, badDescr,
        csr0_.values.data(), csr0_.rowOffsets.data(), csr0_.colIndices.data(),
        rowBlockDim_, colBlockDim_, &bufSize);
    EXPECT_EQ(ret, ACL_SPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED);
    aclsparseDestroyMatDescr(badDescr);
}

 // Exception 12: NULL csrValA (nnz > 0, tested via Convert step)
 TEST_F(Csr2GebsrExceptionTest, NullCsrValA) {
     auto pr = PrepareConvertTest();
     if (pr.skip) { GTEST_SKIP() << "nnzb=0, null csrValA check not applicable"; }
     auto dCsr = CopyCsrToDevice();
     auto ret = aclsparseScsr2gebsr(
         handle_->get(), dir_, m_, n_, descrA_->cget(),
         nullptr,
         reinterpret_cast<int*>(dCsr.dRowPtrA.get()),
         reinterpret_cast<int*>(dCsr.dColIndA.get()),
         descrC_->cget(),
         reinterpret_cast<float*>(pr.dBsrVal.get()),
         reinterpret_cast<int*>(pr.dBsrRowPtr.get()),
         reinterpret_cast<int*>(pr.dBsrColInd.get()),
         rowBlockDim_, colBlockDim_, pr.dWorkspace.get());
     EXPECT_EQ(ret, ACL_SPARSE_STATUS_INVALID_VALUE);
 }

// Exception 13: NULL csrColIndA (nnz > 0, tested via Convert step)
TEST_F(Csr2GebsrExceptionTest, NullCsrColIndA) {
    auto pr = PrepareConvertTest();
    if (pr.skip) { GTEST_SKIP() << "nnzb=0, null csrColIndA check not applicable"; }
    auto dCsr = CopyCsrToDevice();
    auto ret = aclsparseScsr2gebsr(
        handle_->get(), dir_, m_, n_, descrA_->cget(),
        reinterpret_cast<float*>(dCsr.dValA.get()),
        reinterpret_cast<int*>(dCsr.dRowPtrA.get()),
        nullptr,
        descrC_->cget(),
        reinterpret_cast<float*>(pr.dBsrVal.get()),
        reinterpret_cast<int*>(pr.dBsrRowPtr.get()),
        reinterpret_cast<int*>(pr.dBsrColInd.get()),
        rowBlockDim_, colBlockDim_, pr.dWorkspace.get());
    EXPECT_EQ(ret, ACL_SPARSE_STATUS_INVALID_VALUE);
}

// Exception 14: NULL bsrValC (nnzb > 0, tested via Convert step)
TEST_F(Csr2GebsrExceptionTest, NullBsrValC) {
    auto pr = PrepareConvertTest();
    if (pr.skip) { GTEST_SKIP() << "nnzb=0, null bsrValC check not applicable"; }
    auto dCsr = CopyCsrToDevice();
    auto ret = aclsparseScsr2gebsr(
        handle_->get(), dir_, m_, n_, descrA_->cget(),
        reinterpret_cast<float*>(dCsr.dValA.get()),
        reinterpret_cast<int*>(dCsr.dRowPtrA.get()),
        reinterpret_cast<int*>(dCsr.dColIndA.get()),
        descrC_->cget(),
        nullptr,
        reinterpret_cast<int*>(pr.dBsrRowPtr.get()),
        reinterpret_cast<int*>(pr.dBsrColInd.get()),
        rowBlockDim_, colBlockDim_, pr.dWorkspace.get());
    EXPECT_EQ(ret, ACL_SPARSE_STATUS_INVALID_VALUE);
}

// Exception 15: NULL bsrColIndC (nnzb > 0, tested via Convert step)
TEST_F(Csr2GebsrExceptionTest, NullBsrColIndC) {
    auto pr = PrepareConvertTest();
    if (pr.skip) { GTEST_SKIP() << "nnzb=0, null bsrColIndC check not applicable"; }
    auto dCsr = CopyCsrToDevice();
    auto ret = aclsparseScsr2gebsr(
        handle_->get(), dir_, m_, n_, descrA_->cget(),
        reinterpret_cast<float*>(dCsr.dValA.get()),
        reinterpret_cast<int*>(dCsr.dRowPtrA.get()),
        reinterpret_cast<int*>(dCsr.dColIndA.get()),
        descrC_->cget(),
        reinterpret_cast<float*>(pr.dBsrVal.get()),
        reinterpret_cast<int*>(pr.dBsrRowPtr.get()),
        nullptr,
        rowBlockDim_, colBlockDim_, pr.dWorkspace.get());
    EXPECT_EQ(ret, ACL_SPARSE_STATUS_INVALID_VALUE);
}

// Exception 16: NULL pBuffer (bufferSize > 0, tested via Nnz step)
TEST_F(Csr2GebsrExceptionTest, NullPBuffer) {
    auto bs = QueryBufferSize();
    ASSERT_EQ(bs.status, ACL_SPARSE_STATUS_SUCCESS);

    if (bs.bufSize == 0) {
        GTEST_SKIP() << "bufferSize=0, null pBuffer check not applicable";
    }

    auto dBsrRowPtr = AllocBsrRowPtr();
    int nnzb = 0;

    auto ret = aclsparseXcsr2gebsrNnz(
        handle_->get(), dir_, m_, n_,
        descrA_->cget(),
        csr0_.rowOffsets.data(), csr0_.colIndices.data(),
        descrC_->cget(),
        reinterpret_cast<int*>(dBsrRowPtr.get()),
        rowBlockDim_, colBlockDim_,
        &nnzb, nullptr);  // pBuffer = null
    EXPECT_EQ(ret, ACL_SPARSE_STATUS_INVALID_VALUE);
}

// Exception 17: Invalid indexBase (value = 2)
TEST_F(Csr2GebsrExceptionTest, InvalidIndexBase) {
    aclsparseMatDescr_t badDescr = nullptr;
    ASSERT_EQ(aclsparseCreateMatDescr(&badDescr), ACL_SPARSE_STATUS_SUCCESS);
    aclsparseSetMatType(badDescr, ACL_SPARSE_MATRIX_TYPE_GENERAL);
    // Set an invalid index base by directly setting to a bad enum value
    aclsparseSetMatIndexBase(badDescr, static_cast<aclsparseIndexBase_t>(2));

    size_t bufSize = 0;
    auto ret = aclsparseScsr2gebsr_bufferSize(
        handle_->get(), dir_, m_, n_, badDescr,
        csr0_.values.data(), csr0_.rowOffsets.data(), csr0_.colIndices.data(),
        rowBlockDim_, colBlockDim_, &bufSize);
    EXPECT_EQ(ret, ACL_SPARSE_STATUS_INVALID_VALUE);
    aclsparseDestroyMatDescr(badDescr);
}

// No main() — provided by test/frame/test_main.cpp

// ============================================================================
// Additional defensive exception branch tests
//
// The following 7 exception tests cover remaining testable exception branches:
//   - NULL pBufferSizeInBytes (ValidateBufferSizeParams)
//   - m/n > INT32_MAX/2 (ValidateDimUpperBound)
//   - rowBlockDim/colBlockDim > INT32_MAX/2 (blockDim upper bound check)
//   - mb*nb > INT32_MAX (product upper bound check, m/n individually valid)
//   - ValidateNnzParams: nnzTotalDevHostPtr == nullptr (independent validation branch)
// ============================================================================

// Exception 18: NULL pBufferSizeInBytes
TEST_F(Csr2GebsrExceptionTest, NullBufferSizePtr) {
    auto ret = aclsparseScsr2gebsr_bufferSize(
        handle_->get(), dir_, m_, n_, descrA_->cget(),
        csr0_.values.data(), csr0_.rowOffsets.data(), csr0_.colIndices.data(),
        rowBlockDim_, colBlockDim_, nullptr);  // pBufferSizeInBytes = null
    EXPECT_EQ(ret, ACL_SPARSE_STATUS_INVALID_VALUE);
}

// Exception 19: m exceeds upper bound (ValidateDimUpperBound)
// kDimUpperBound = INT32_MAX/2 = 1073741823 (host.cpp:79)
TEST_F(Csr2GebsrExceptionTest, MExceedsUpperBound) {
    constexpr int kDimUpperBound = 1073741823;  // INT32_MAX / 2
    size_t bufSize = 0;
    auto ret = aclsparseScsr2gebsr_bufferSize(
        handle_->get(), dir_, kDimUpperBound + 1, n_, descrA_->cget(),
        csr0_.values.data(), csr0_.rowOffsets.data(), csr0_.colIndices.data(),
        rowBlockDim_, colBlockDim_, &bufSize);
    EXPECT_EQ(ret, ACL_SPARSE_STATUS_INVALID_VALUE);
}

// Exception 20: n exceeds upper bound (ValidateDimUpperBound)
TEST_F(Csr2GebsrExceptionTest, NExceedsUpperBound) {
    constexpr int kDimUpperBound = 1073741823;  // INT32_MAX / 2
    size_t bufSize = 0;
    auto ret = aclsparseScsr2gebsr_bufferSize(
        handle_->get(), dir_, m_, kDimUpperBound + 1, descrA_->cget(),
        csr0_.values.data(), csr0_.rowOffsets.data(), csr0_.colIndices.data(),
        rowBlockDim_, colBlockDim_, &bufSize);
    EXPECT_EQ(ret, ACL_SPARSE_STATUS_INVALID_VALUE);
}

// Exception 21: rowBlockDim exceeds upper bound (ValidateDimUpperBound)
// rowBlockDim > INT32_MAX/2 时 (m + rowBlockDim - 1) 可能溢出
TEST_F(Csr2GebsrExceptionTest, RowBlockDimExceedsUpperBound) {
    constexpr int kDimUpperBound = 1073741823;  // INT32_MAX / 2
    size_t bufSize = 0;
    auto ret = aclsparseScsr2gebsr_bufferSize(
        handle_->get(), dir_, m_, n_, descrA_->cget(),
        csr0_.values.data(), csr0_.rowOffsets.data(), csr0_.colIndices.data(),
        kDimUpperBound + 1, colBlockDim_, &bufSize);
    EXPECT_EQ(ret, ACL_SPARSE_STATUS_INVALID_VALUE);
}

// Exception 22: colBlockDim exceeds upper bound (ValidateDimUpperBound)
TEST_F(Csr2GebsrExceptionTest, ColBlockDimExceedsUpperBound) {
    constexpr int kDimUpperBound = 1073741823;  // INT32_MAX / 2
    size_t bufSize = 0;
    auto ret = aclsparseScsr2gebsr_bufferSize(
        handle_->get(), dir_, m_, n_, descrA_->cget(),
        csr0_.values.data(), csr0_.rowOffsets.data(), csr0_.colIndices.data(),
        rowBlockDim_, kDimUpperBound + 1, &bufSize);
    EXPECT_EQ(ret, ACL_SPARSE_STATUS_INVALID_VALUE);
}

// Exception 23: mb*nb exceeds INT32_MAX (ValidateDimUpperBound)
// m,n 各自合法(<=INT32_MAX/2) 但 mb*nb 溢出:
//   m=65536, n=65536, b=1x1 -> mb=65536, nb=65536, mb*nb=4294967296 > INT32_MAX
// 乘积上限校验防止 kernel 侧 markerBase int32 溢出
TEST_F(Csr2GebsrExceptionTest, MbTimesNbExceedsUpperBound) {
    size_t bufSize = 0;
    auto ret = aclsparseScsr2gebsr_bufferSize(
        handle_->get(), dir_, 65536, 65536, descrA_->cget(),
        csr0_.values.data(), csr0_.rowOffsets.data(), csr0_.colIndices.data(),
        1, 1, &bufSize);
    EXPECT_EQ(ret, ACL_SPARSE_STATUS_INVALID_VALUE);
}

// Exception 24: NULL nnzTotalDevHostPtr (ValidateNnzParams, Nnz entry)
// ValidateNnzParams 独立校验分支可测（在 LaunchNnzKernel 之前拦截）。
TEST_F(Csr2GebsrExceptionTest, NullNnzTotalDevHostPtr) {
    auto bs = QueryBufferSize();
    ASSERT_EQ(bs.status, ACL_SPARSE_STATUS_SUCCESS);
    auto dWorkspace = AllocWorkspace(bs.bufSize);
    auto dBsrRowPtr = AllocBsrRowPtr();

    auto ret = aclsparseXcsr2gebsrNnz(
        handle_->get(), dir_, m_, n_,
        descrA_->cget(),
        csr0_.rowOffsets.data(), csr0_.colIndices.data(),
        descrC_->cget(),
        reinterpret_cast<int*>(dBsrRowPtr.get()),
        rowBlockDim_, colBlockDim_,
        nullptr,  // nnzTotalDevHostPtr = null
        dWorkspace.get());
    EXPECT_EQ(ret, ACL_SPARSE_STATUS_INVALID_VALUE);
}

// ============================================================================
// pointerMode=DEVICE nnzb D2D copyback
//
// The default wrapper path uses HOST pointer mode (nnzb D2H-copied to a host
// int). This test switches the handle to DEVICE pointer mode so the operator
// D2D-copies nnzb into a device pointer, exercising:
//   - LaunchPrefixSumAndCopyback DEVICE branch for a non-empty matrix
//   - FillEmptyBsrRowPtrC DEVICE branch for m=0
// nnzb is read back from the device buffer and compared against HOST mode.
// ============================================================================
TEST_F(Csr2GebsrExceptionTest, PointerModeDeviceNnzb) {
    int nnz = static_cast<int>(csr0_.nnz);

    // HOST mode (reference truth)
    auto hostRes = Csr2GebsrNpu(*handle_, stream_,
        csr0_.rowOffsets, csr0_.colIndices, csr0_.values,
        m_, n_, nnz, rowBlockDim_, colBlockDim_,
        ACL_SPARSE_DIRECTION_ROW, 0, 0, "FP32", /*useDevicePointerMode=*/false);
    ASSERT_EQ(hostRes.nnzRet, ACL_SPARSE_STATUS_SUCCESS)
        << "HOST mode Nnz step failed";

    // DEVICE mode (D2D nnzb copyback)
    auto devRes = Csr2GebsrNpu(*handle_, stream_,
        csr0_.rowOffsets, csr0_.colIndices, csr0_.values,
        m_, n_, nnz, rowBlockDim_, colBlockDim_,
        ACL_SPARSE_DIRECTION_ROW, 0, 0, "FP32", /*useDevicePointerMode=*/true);
    ASSERT_EQ(devRes.nnzRet, ACL_SPARSE_STATUS_SUCCESS)
        << "DEVICE mode Nnz step failed";

    EXPECT_EQ(devRes.nnzb, hostRes.nnzb)
        << "DEVICE mode nnzb (" << devRes.nnzb
        << ") != HOST mode nnzb (" << hostRes.nnzb << ")";
    EXPECT_EQ(devRes.bsrRowPtr, hostRes.bsrRowPtr)
        << "DEVICE mode bsrRowPtrC differs from HOST mode";

    // m=0 + DEVICE mode exercises FillEmptyBsrRowPtrC DEVICE branch
    // (D2D copy of nnzb=0 into a device pointer).
    auto emptyCsr = makeEmptyCsr(0, n_);
    auto devEmpty = Csr2GebsrNpu(*handle_, stream_,
        emptyCsr.rowOffsets, emptyCsr.colIndices, emptyCsr.values,
        0, n_, 0, rowBlockDim_, colBlockDim_,
        ACL_SPARSE_DIRECTION_ROW, 0, 0, "FP32", /*useDevicePointerMode=*/true);
    ASSERT_EQ(devEmpty.nnzRet, ACL_SPARSE_STATUS_SUCCESS);
    EXPECT_EQ(devEmpty.nnzb, 0) << "m=0 DEVICE mode should yield nnzb=0";
    ASSERT_EQ(devEmpty.bsrRowPtr.size(), 1u);
    EXPECT_EQ(devEmpty.bsrRowPtr[0], 0)
        << "m=0 bsrRowPtrC[0] should be baseC=0 (DEVICE mode)";
}

// ============================================================================
// Nnz-side null csrColIndA with nnz=0 (legal path)
//
// The Nnz API has no nnz input parameter and therefore cannot distinguish
// "nnz=0 + null csrColIndA" (legal, kernel loop body never runs) from
// "nnz>0 + null csrColIndA" (illegal, kernel dereferences null). With nnz=0
// (all CSR row offsets equal baseA), passing null csrColIndA is legal and the
// call must succeed with nnzb=0. The nnz>0 + null csrColIndA case is a known
// API design limitation that triggers a kernel fault and is intentionally NOT
// exercised here to avoid crashing the test process.
// ============================================================================
TEST_F(Csr2GebsrExceptionTest, NnzNullCsrColIndAEmpty) {
    // Build a 4x4 CSR with nnz=0 (all row offsets equal baseA=0).
    auto emptyCsr = makeEmptyCsr(m_, n_);
    auto dRowPtr = DeviceBuffer::copyFrom(
        emptyCsr.rowOffsets.data(), emptyCsr.rowOffsets.size() * sizeof(int32_t));

    auto bs = QueryBufferSize();
    ASSERT_EQ(bs.status, ACL_SPARSE_STATUS_SUCCESS);
    auto dWorkspace = AllocWorkspace(bs.bufSize);
    auto dBsrRowPtr = AllocBsrRowPtr();

    int nnzb = -1;  // sentinel to detect whether the operator wrote it
    auto ret = aclsparseXcsr2gebsrNnz(
        handle_->get(), dir_, m_, n_,
        descrA_->cget(),
        reinterpret_cast<int*>(dRowPtr.get()),
        nullptr,  // csrColIndA = null (legal when nnz=0)
        descrC_->cget(),
        reinterpret_cast<int*>(dBsrRowPtr.get()),
        rowBlockDim_, colBlockDim_,
        &nnzb, dWorkspace.get());
    EXPECT_EQ(ret, ACL_SPARSE_STATUS_SUCCESS);
    EXPECT_EQ(nnzb, 0)
        << "nnz=0 + null csrColIndA should yield nnzb=0 (Nnz legal path)";
}

// ============================================================================
// Performance collection framework
//
// Measures end-to-end csr2gebsr three-step latency (bufferSize + Nnz + Convert)
// via aclrtSynchronizeStream-braced std::chrono timing. Warm-up 3 + measure 10
// iterations, reporting avg/min/max/median, data volume and effective bandwidth.
//
// Results are emitted to:
//   - stdout ([PERF] lines, human readable)
//   - JSON file at $CSR2GEBSR_PERF_OUT or ./csr2gebsr_perf_data.json
// ============================================================================

class Csr2GebsrPerfTest : public Csr2GebsrTestBase {
public:
    static void SetUpTestSuite() { Csr2GebsrTestBase::SetUpTestSuite(); }
    static void TearDownTestSuite() { Csr2GebsrTestBase::TearDownTestSuite(); }

protected:

    struct PerfCase {
        int m;
        int n;
        std::string dtype;
        int rbd;
        int cbd;
        double sparsity;
        const char* label;
    };

    // Estimate total bytes moved (CSR input + BSR output) for bandwidth.
    static double EstimateBytes(int m, int n, int nnz, int nnzb,
                                 int rbd, int cbd, size_t valSize) {
        int mb = (rbd > 0) ? (m + rbd - 1) / rbd : 0;
        double inBytes =
            static_cast<double>(m + 1) * sizeof(int32_t) +        // csrRowPtrA
            static_cast<double>(nnz) * sizeof(int32_t) +          // csrColIndA
            static_cast<double>(nnz) * valSize;                   // csrValA
        double outBytes =
            static_cast<double>(mb + 1) * sizeof(int32_t) +       // bsrRowPtrC
            static_cast<double>(nnzb) * sizeof(int32_t) +         // bsrColIndC
            static_cast<double>(nnzb) * rbd * cbd * valSize;      // bsrValC
        return inBytes + outBytes;
    }

    // Run the full three-step workflow once; returns elapsed microseconds
    // (or -1.0 on failure). Rebuilds CSR each call to mirror a cold input.
    double RunOnce(const PerfCase& pc, int& nnzOut, int& nnzbOut) {
        HandleManager handle;
        handle.setStream(stream_);

        auto csr0 = makeSparseCsr(pc.m, pc.n, pc.sparsity, 42);
        int nnz = static_cast<int>(csr0.nnz);

        // Force stream idle before timing window.
        aclrtSynchronizeStream(stream_);
        auto t0 = std::chrono::high_resolution_clock::now();
        auto res = Csr2GebsrNpu(handle, stream_,
            csr0.rowOffsets, csr0.colIndices, csr0.values,
            pc.m, pc.n, nnz, pc.rbd, pc.cbd,
            ACL_SPARSE_DIRECTION_ROW, 0, 0, pc.dtype, false);
        aclrtSynchronizeStream(stream_);
        auto t1 = std::chrono::high_resolution_clock::now();

        nnzOut = nnz;
        nnzbOut = res.nnzb;
        if (res.convertRet != ACL_SPARSE_STATUS_SUCCESS) {
            return -1.0;
        }
        return std::chrono::duration<double, std::micro>(t1 - t0).count();
    }

    static std::vector<PerfCase> GetPerfCases() {
        return {
            {16,   16,   "FP32", 2, 2, 0.5, "16x16_b2x2_FP32"},
            {16,   16,   "FP16", 2, 2, 0.5, "16x16_b2x2_FP16"},
            {300,  300,  "FP32", 2, 2, 0.5, "300x300_b2x2_FP32"},
            {300,  300,  "FP16", 2, 2, 0.5, "300x300_b2x2_FP16"},
            {1000, 1000, "FP32", 2, 2, 0.1, "1000x1000_b2x2_FP32"},
            {1000, 1000, "FP16", 2, 2, 0.1, "1000x1000_b2x2_FP16"},
            {1000, 10,   "FP32", 2, 2, 0.3, "1000x10_b2x2_FP32"},
            {1000, 10,   "FP16", 2, 2, 0.3, "1000x10_b2x2_FP16"},
            {10000, 10,  "FP32", 1, 1, 0.3, "10000x10_b1x1_FP32_bigmb"},
        };
    }

    void MeasureAndEmit(const PerfCase& pc, int kWarmup, int kMeasure,
                        std::ostringstream& json, size_t& emitted) {
        for (int i = 0; i < kWarmup; i++) { int a, b; RunOnce(pc, a, b); }
        int nnz = 0, nnzb = 0;
        std::vector<double> samples;
        double sum = 0.0, mn = 1e18, mx = 0.0;
        for (int i = 0; i < kMeasure; i++) {
            double t = RunOnce(pc, nnz, nnzb);
            if (t < 0) { ADD_FAILURE() << "perf case failed: " << pc.label; return; }
            samples.push_back(t); sum += t; mn = std::min(mn, t); mx = std::max(mx, t);
        }
        if (samples.empty()) return;
        double avg = sum / samples.size();
        std::sort(samples.begin(), samples.end());
        double median = samples[samples.size() / 2];
        size_t valSize = (pc.dtype == "FP16" || pc.dtype == "BF16") ? 2 : 4;
        double bytes = EstimateBytes(pc.m, pc.n, nnz, nnzb, pc.rbd, pc.cbd, valSize);
        double bwGbs = (bytes / (avg * 1e-6)) / 1e9;
        std::cout << "[PERF] " << pc.label << " nnz=" << nnz << " nnzb=" << nnzb
                  << " avg=" << avg << "us min=" << mn << "us max=" << mx
                  << "us median=" << median << "us bytes=" << bytes
                  << " bw=" << bwGbs << "GB/s\n";
        if (emitted > 0) json << "    ,\n";
        json << "    {\n      \"label\": \"" << pc.label << "\",\n"
             << "      \"m\": " << pc.m << ", \"n\": " << pc.n
             << ", \"dtype\": \"" << pc.dtype << "\", \"block\": \"" << pc.rbd << "x" << pc.cbd << "\",\n"
             << "      \"nnz\": " << nnz << ", \"nnzb\": " << nnzb << ",\n"
             << "      \"avg_us\": " << avg << ", \"min_us\": " << mn
             << ", \"max_us\": " << mx << ", \"median_us\": " << median << ",\n"
             << "      \"bytes\": " << bytes << ", \"bandwidth_gbs\": " << bwGbs << "\n    }\n";
        emitted++;
    }

    static void WriteJson(const std::ostringstream& json) {
        const char* envPath = std::getenv("CSR2GEBSR_PERF_OUT");
        std::vector<std::string> candidates;
        if (envPath && envPath[0] != '\0') candidates.emplace_back(envPath);
        candidates.emplace_back("csr2gebsr_perf_data.json");
        for (const auto& path : candidates) {
            std::ofstream f(path);
            if (f) { f << json.str(); f.close();
                std::cout << "[PERF] results written to " << path << "\n"; return; }
        }
        std::cerr << "[PERF] WARNING: could not write perf JSON; dumping:\n" << json.str();
    }
};

TEST_F(Csr2GebsrPerfTest, PerfSweep) {
    const char* warmupEnv = std::getenv("CSR2GEBSR_PERF_WARMUP");
    const char* measureEnv = std::getenv("CSR2GEBSR_PERF_MEASURE");
    const int kWarmup = warmupEnv ? std::atoi(warmupEnv) : 3;
    const int kMeasure = measureEnv ? std::atoi(measureEnv) : 10;
    const char* filterEnv = std::getenv("CSR2GEBSR_PERF_CASE");
    std::string filter = (filterEnv && filterEnv[0] != '\0') ? filterEnv : "";

    auto cases = GetPerfCases();
    std::ostringstream json;
    json << "{\n  \"op\": \"csr2gebsr\",\n  \"soc\": \"arch35\",\n"
         << "  \"warmup\": " << kWarmup << ",\n  \"measure\": " << kMeasure
         << ",\n  \"cases\": [\n";
    std::cout << "\n==================== csr2gebsr PERF SWEEP ====================\n"
              << "warmup=" << kWarmup << " measure=" << kMeasure
              << (filter.empty() ? "" : (" filter=" + filter)) << "\n";

    size_t emitted = 0;
    for (const auto& pc : cases) {
        if (!filter.empty() && filter != pc.label) continue;
        MeasureAndEmit(pc, kWarmup, kMeasure, json, emitted);
    }
    json << "  ]\n}\n";
    WriteJson(json);
    std::cout << "===============================================================\n";
}
