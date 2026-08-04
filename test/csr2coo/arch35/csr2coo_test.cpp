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
 * @file csr2coo_test.cpp
 * @brief GTest + CSV-driven test cases for aclsparseXcsr2coo (Legacy API).
 *
 * Tests CSR to COO sparse matrix row-index expansion.
 * The operation expands csrRowPtr[m+1] into cooRowInd[nnz] where each element
 * maps to its originating row (with idxBase offset).
 * Comparison is bitwise int32 (BitwiseInt32Strategy, no float intermediation).
 *
 * Legacy API signature (flat params, no SpMatDescr, no GetBufferSize):
 *   aclsparseXcsr2coo(handle, csrRowPtr, nnz, m, cooRowInd, idxBase)
 *
 * Test structure:
 *   - TEST_P (Csr2CooTest) : parameterized success-path tests from CSV (58 rows)
 *   - TEST_F (Csr2CooExceptionTest) : dedicated exception/boundary tests (19 cases)
 *
 * Entry point is shared via test/frame/test_main.cpp.
 */

#include "test_common.h"
#include "csr2coo_golden.h"
#include "csr2coo_npu_wrapper.h"
#include "csr2coo_param.h"

#include <limits>

using namespace sparse_test;

// ============================================================================
// BitwiseInt32Strategy: 直接 int32_t 位级比对（不经浮点中转）。
// csr2coo 为纯索引展开的非计算类算子，符合 ops-precision-standard non_compute.md
// 二进制一致要求。不修改公共 verify.h，在算子测试内部局部定义。
// ============================================================================

static bool VerifyInt32Bitwise(const std::vector<int32_t> &output,
                               const std::vector<int32_t> &golden,
                               const std::string &caseId)
{
    if (output.size() != golden.size()) {
        std::cout << "[" << caseId << "] FAILED: size mismatch, output=" << output.size()
                  << " golden=" << golden.size() << std::endl;
        return false;
    }
    size_t failCount = 0;
    for (size_t i = 0; i < output.size(); i++) {
        if (output[i] != golden[i]) failCount++;
    }
    bool pass = (failCount == 0);
    std::cout << "[" << caseId << "] " << (pass ? "PASSED" : "FAILED")
              << " (bitwise int32 match, " << failCount << "/" << output.size()
              << " mismatches)" << std::endl;
    return pass;
}

// ============================================================================
// Helper: Apply idxBase to CSR row offsets
// ============================================================================

static std::vector<int32_t> ApplyPtrIdxBase(
    const std::vector<int32_t> &ptrArray,
    int idxBase)
{
    if (idxBase == 0) {
        return ptrArray;
    }
    std::vector<int32_t> adjusted = ptrArray;
    for (size_t i = 0; i < adjusted.size(); i++) {
        adjusted[i] += static_cast<int32_t>(idxBase);
    }
    return adjusted;
}

// ============================================================================
// Helper: Derive CSC colPtr from CSR
// ============================================================================

static std::vector<int32_t> DeriveCscColPtr(
    const CsrMatrix &csr,
    int64_t rows,
    int64_t cols)
{
    std::vector<int32_t> colPtr(static_cast<size_t>(cols) + 1, 0);
    for (int64_t i = 0; i < rows; i++) {
        for (int32_t j = csr.rowOffsets[i]; j < csr.rowOffsets[i + 1]; j++) {
            int32_t col = csr.colIndices[j];
            colPtr[static_cast<size_t>(col) + 1]++;
        }
    }
    for (int64_t c = 0; c < cols; c++) {
        colPtr[static_cast<size_t>(c) + 1] += colPtr[static_cast<size_t>(c)];
    }
    return colPtr;
}

// ============================================================================
// Helper: Prepare CSR/CSC input — decide CSR vs CSC, set inputPtr and apiM
// ============================================================================

static void PrepareCsrInput(const Csr2CooParam &param,
                            const CsrMatrix &csr,
                            const int32_t *&inputPtr, int64_t &apiM,
                            std::vector<int32_t> &ptrStorage)
{
    if (param.pattern == "csc") {
        // CSC→COO: derive colPtr from CSR, use n as m
        auto colPtr0 = DeriveCscColPtr(csr, param.m, param.n);
        ptrStorage = ApplyPtrIdxBase(colPtr0, param.idx_base);
        inputPtr = ptrStorage.data();
        apiM = param.n;                 // CSC uses column count as m
    } else {
        // CSR→COO: use rowOffsets directly
        ptrStorage = ApplyPtrIdxBase(csr.rowOffsets, param.idx_base);
        inputPtr = ptrStorage.data();
        apiM = param.m;
    }
}

// ============================================================================
// GTest parameterized fixture: Csr2CooTest
// ============================================================================

class Csr2CooTest : public ::testing::TestWithParam<Csr2CooParam> {
public:
    static void SetUpTestSuite() {
        env_ = std::make_unique<AclEnvScope>();
    }

    static void TearDownTestSuite() {
        env_.reset();
    }

protected:
    inline static std::unique_ptr<AclEnvScope> env_;
    Csr2CooParam param_;
    aclrtStream stream_ = nullptr;

    void SetUp() override {
        param_ = GetParam();
        stream_ = env_->stream();
    }
};

// ============================================================================
// TEST_P: Success-path parameterized test
//
// Covers: random CSR, diag CSR, CSC, large shape, boundary (m=0, nnz=0),
//         extreme narrow matrices, empty row patterns, idxBase 0/1.
//
// Exception CSV rows (expect_result != SUCCESS) are not in the CSV;
// those are validated in the dedicated TEST_F (Csr2CooExceptionTest).
// ============================================================================

TEST_P(Csr2CooTest, Csr2CooSuccess) {
    const auto &p = param_;

    std::cout << "==== " << p.case_name
              << " ==== m=" << p.m << " n=" << p.n
              << " idxBase=" << p.idx_base
              << " pattern=" << p.pattern << "\n";

    // 1. Generate CSR data
    CsrMatrix csr0;
    if (p.pattern == "diag") {
        csr0 = makeDiagCsr(p.m);
    } else {
        SparseFillGenerator gen(p.seed);
        gen.setSparsity(p.sparsity);
        gen.setEmptyRowProb(p.empty_row_prob);
        csr0 = gen.generateCsr(p.m, p.n);
    }

    int64_t nnz = csr0.nnz;

    std::cout << "  nnz=" << nnz << "\n";

    // 2. Prepare input pointer and API m (handles CSR vs CSC, idxBase)
    const int32_t *inputPtr = nullptr;
    int64_t apiM = 0;
    std::vector<int32_t> ptrStorage;
    PrepareCsrInput(p, csr0, inputPtr, apiM, ptrStorage);

    // 3. Compute golden reference (format-agnostic)
    auto golden = Csr2CooGolden(ptrStorage, apiM, nnz, p.idx_base);

    // 4. Call NPU
    HandleManager handle;
    handle.setStream(stream_);

    aclsparseIndexBase_t aclIdxBase = (p.idx_base == 1)
        ? ACL_SPARSE_INDEX_BASE_ONE : ACL_SPARSE_INDEX_BASE_ZERO;

    auto result = NpuCsr2coo(handle, stream_, inputPtr, nnz, apiM, aclIdxBase);

    ASSERT_EQ(p.expect_result, "SUCCESS")
        << "[" << p.case_name << "] unexpected expect_result: " << p.expect_result;

    ASSERT_EQ(result.ret, ACL_SPARSE_STATUS_SUCCESS)
        << "[" << p.case_name << "] Csr2coo failed: " << result.ret;

    // 5. Verify (bitwise int32, 直接位级比对，不经浮点中转)
    if (nnz > 0) {
        bool pass = VerifyInt32Bitwise(result.cooRowInd, golden, p.caseId() + "_cooRowInd");
        EXPECT_TRUE(pass) << "[" << p.case_name << "] cooRowInd mismatch";
    } else {
        EXPECT_EQ(result.cooRowInd.size(), static_cast<size_t>(0))
            << "[" << p.case_name << "] Expected empty output for nnz=0";
    }

    std::cout << "[" << p.case_name << "] PASSED (nnz=" << nnz << ")\n";
}

// ============================================================================
// Parameterized test instantiation from CSV (58 success cases)
// ============================================================================

INSTANTIATE_TEST_SUITE_P(
    Csr2CooCases,
    Csr2CooTest,
    ::testing::ValuesIn(GetCasesFromCsv<Csr2CooParam>("csr2coo_test.csv")),
    [](const ::testing::TestParamInfo<Csr2CooParam> &info) {
        return info.param.case_name;
    }
);

// ============================================================================
// Helper: build device buffers, call aclsparseXcsr2coo, assert return code.
//
// Shared by the exception-path and boundary TEST_F cases below to avoid
// repeating the HandleManager → DeviceBuffer → API → EXPECT_EQ skeleton.
//   hostRowPtr/rowPtrCount : host row-pointer array (nullptr ⇒ pass nullptr)
//   nullHandle             : pass nullptr for handle
//   nullCooRowInd          : pass nullptr for cooRowInd (no device alloc)
//   syncStream             : after a SUCCESS return, sync the stream to ensure
//                            the kernel completes without crashing/OOB
// ============================================================================
static void RunCsr2cooAndCheck(
    aclrtStream stream,
    const int32_t *hostRowPtr, size_t rowPtrCount,
    int64_t nnz, int64_t m,
    bool nullHandle, bool nullCooRowInd,
    aclsparseIndexBase_t idxBase,
    aclsparseStatus_t expected,
    bool syncStream = false,
    bool compareGolden = true)
{
    std::unique_ptr<HandleManager> handle;
    if (!nullHandle) {
        handle = std::make_unique<HandleManager>();
        handle->setStream(stream);
    }

    DeviceBuffer dRowPtr;
    const int32_t *pRowPtr = nullptr;
    if (hostRowPtr != nullptr) {
        dRowPtr = DeviceBuffer::copyFrom(hostRowPtr, rowPtrCount * sizeof(int32_t));
        pRowPtr = static_cast<const int32_t *>(dRowPtr.get());
    }

    DeviceBuffer dCooRowInd;
    int32_t *pCooRowInd = nullptr;
    if (!nullCooRowInd) {
        // nnz <= 0 (validation-failure cases) still needs a valid device buffer
        // to pass as cooRowInd; allocate one element to avoid huge/zero sizes.
        size_t elemCount = (nnz > 0) ? static_cast<size_t>(nnz) : 1;
        size_t idxBytes = elemCount * sizeof(int32_t);
        dCooRowInd = DeviceBuffer::alloc(idxBytes);
        pCooRowInd = static_cast<int32_t *>(dCooRowInd.get());
        // T2: zero-init cooRowInd so unwritten elements (skipped rows) match golden.
        EXPECT_EQ(aclrtMemset(pCooRowInd, idxBytes, 0, idxBytes), ACL_SUCCESS);
    }

    auto ret = aclsparseXcsr2coo(
        nullHandle ? nullptr : handle->get(),
        pRowPtr, nnz, m, pCooRowInd, idxBase);
    EXPECT_EQ(ret, expected);

    // For inputs that pass validation and launch the kernel, sync to ensure
    // the kernel completes without crashing or writing out of bounds.
    if (syncStream && ret == ACL_SPARSE_STATUS_SUCCESS) {
        EXPECT_EQ(aclrtSynchronizeStream(stream), ACL_SUCCESS);
    }

    // T2: golden comparison for SUCCESS cases that launch the kernel.
    if (ret == ACL_SPARSE_STATUS_SUCCESS && nnz > 0 && !nullCooRowInd && hostRowPtr != nullptr && compareGolden) {
        std::vector<int32_t> ptrVec(hostRowPtr, hostRowPtr + rowPtrCount);
        int idxBaseInt = (idxBase == ACL_SPARSE_INDEX_BASE_ONE) ? 1 : 0;
        auto golden = Csr2CooGolden(ptrVec, m, nnz, idxBaseInt);

        std::vector<int32_t> npuResult(static_cast<size_t>(nnz), 0);
        dCooRowInd.copyToHost(npuResult.data(), static_cast<size_t>(nnz) * sizeof(int32_t));

        bool pass = VerifyInt32Bitwise(npuResult, golden, "abn_golden");
        EXPECT_TRUE(pass) << "golden mismatch (nnz=" << nnz << ", m=" << m << ")";
    }
}

// ============================================================================
// GTest fixture: Csr2CooExceptionTest
//
// AclEnvScope is shared across all TEST_F of this suite (SetUpTestSuite) to
// avoid repeated aclInit/aclFinalize per case (same pattern as
// gtsv2_strided_batch_test.cpp). SetUp() only fetches the stream.
//
// Covers 10 validation error-code paths plus boundary/abnormal-input cases:
//   handle == nullptr         → ACL_SPARSE_STATUS_NOT_INITIALIZED
//   nnz < 0                   → ACL_SPARSE_STATUS_INVALID_VALUE
//   m < 0                     → ACL_SPARSE_STATUS_INVALID_VALUE
//   m == 0 && nnz > 0         → ACL_SPARSE_STATUS_INVALID_VALUE
//   idxBase != 0 && != 1      → ACL_SPARSE_STATUS_INVALID_VALUE
//   m > 0 && csrRowPtr == nullptr → ACL_SPARSE_STATUS_INVALID_VALUE
//   nnz > 0 && cooRowInd == nullptr → ACL_SPARSE_STATUS_INVALID_VALUE
//   nnz > INT32_MAX           → ACL_SPARSE_STATUS_INVALID_VALUE
//   m > INT32_MAX             → ACL_SPARSE_STATUS_INVALID_VALUE
//   rowPtrBytes > UINT32_MAX  → ACL_SPARSE_STATUS_INVALID_VALUE
// ============================================================================

class Csr2CooExceptionTest : public ::testing::Test {
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

// ---------------------------------------------------------------------------
// ERR_null_handle: pass handle=nullptr → ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR
// ---------------------------------------------------------------------------

TEST_F(Csr2CooExceptionTest, NullHandle) {
    std::cout << "==== ERR_null_handle ====\n";
    int32_t rowPtr[] = {0, 1, 2, 3, 4};
    RunCsr2cooAndCheck(stream_, rowPtr, 5, 4, 4,
                       /*nullHandle=*/true, /*nullCooRowInd=*/false,
                       ACL_SPARSE_INDEX_BASE_ZERO, ACL_SPARSE_STATUS_NOT_INITIALIZED);
    std::cout << "[ERR_null_handle] PASSED (NOT_INITIALIZED)\n";
}

// ---------------------------------------------------------------------------
// ERR_null_csrRowPtr: pass csrRowPtr=nullptr, m>0 → ACL_SPARSE_STATUS_INVALID_VALUE
// ---------------------------------------------------------------------------

TEST_F(Csr2CooExceptionTest, NullCsrRowPtr) {
    std::cout << "==== ERR_null_csrRowPtr ====\n";
    RunCsr2cooAndCheck(stream_, nullptr, 0, 4, 4,
                       /*nullHandle=*/false, /*nullCooRowInd=*/false,
                       ACL_SPARSE_INDEX_BASE_ZERO, ACL_SPARSE_STATUS_INVALID_VALUE);
    std::cout << "[ERR_null_csrRowPtr] PASSED (INVALID_VALUE)\n";
}

// ---------------------------------------------------------------------------
// ERR_null_cooRowInd: pass cooRowInd=nullptr, nnz>0 → ACL_SPARSE_STATUS_INVALID_VALUE
// ---------------------------------------------------------------------------

TEST_F(Csr2CooExceptionTest, NullCooRowInd) {
    std::cout << "==== ERR_null_cooRowInd ====\n";
    int32_t rowPtr[] = {0, 1, 2, 3, 4};
    RunCsr2cooAndCheck(stream_, rowPtr, 5, 4, 4,
                       /*nullHandle=*/false, /*nullCooRowInd=*/true,
                       ACL_SPARSE_INDEX_BASE_ZERO, ACL_SPARSE_STATUS_INVALID_VALUE);
    std::cout << "[ERR_null_cooRowInd] PASSED (INVALID_VALUE)\n";
}

// ---------------------------------------------------------------------------
// ERR_nnz_negative: pass nnz=-1 → ACL_SPARSE_STATUS_INVALID_VALUE
// ---------------------------------------------------------------------------

TEST_F(Csr2CooExceptionTest, NnzNegative) {
    std::cout << "==== ERR_nnz_negative ====\n";
    int32_t rowPtr[] = {0, 0, 0, 0, 0};
    RunCsr2cooAndCheck(stream_, rowPtr, 5, -1, 4,
                       /*nullHandle=*/false, /*nullCooRowInd=*/false,
                       ACL_SPARSE_INDEX_BASE_ZERO, ACL_SPARSE_STATUS_INVALID_VALUE);
    std::cout << "[ERR_nnz_negative] PASSED (INVALID_VALUE)\n";
}

// ---------------------------------------------------------------------------
// ERR_m0_nnz_positive: pass m=0, nnz=1 → ACL_SPARSE_STATUS_INVALID_VALUE
// ---------------------------------------------------------------------------

TEST_F(Csr2CooExceptionTest, M0NnzPositive) {
    std::cout << "==== ERR_m0_nnz_positive ====\n";
    int32_t rowPtr[] = {0};  // m=0, so rowPtr length = m+1 = 1
    RunCsr2cooAndCheck(stream_, rowPtr, 1, 1, 0,
                       /*nullHandle=*/false, /*nullCooRowInd=*/false,
                       ACL_SPARSE_INDEX_BASE_ZERO, ACL_SPARSE_STATUS_INVALID_VALUE);
    std::cout << "[ERR_m0_nnz_positive] PASSED (INVALID_VALUE)\n";
}

// ---------------------------------------------------------------------------
// ERR_invalid_idxBase: pass idxBase=2 → ACL_SPARSE_STATUS_INVALID_VALUE
// ---------------------------------------------------------------------------

TEST_F(Csr2CooExceptionTest, InvalidIdxBase) {
    std::cout << "==== ERR_invalid_idxBase ====\n";
    int32_t rowPtr[] = {0, 1, 2, 3, 4};
    RunCsr2cooAndCheck(stream_, rowPtr, 5, 4, 4,
                       /*nullHandle=*/false, /*nullCooRowInd=*/false,
                       static_cast<aclsparseIndexBase_t>(2), ACL_SPARSE_STATUS_INVALID_VALUE);
    std::cout << "[ERR_invalid_idxBase] PASSED (INVALID_VALUE)\n";
}

// ---------------------------------------------------------------------------
// ERR_m_negative: pass m=-1 → ACL_SPARSE_STATUS_INVALID_VALUE
// ---------------------------------------------------------------------------

TEST_F(Csr2CooExceptionTest, MNegative) {
    std::cout << "==== ERR_m_negative ====\n";
    int32_t rowPtr[] = {0};
    RunCsr2cooAndCheck(stream_, rowPtr, 1, 0, -1,
                       /*nullHandle=*/false, /*nullCooRowInd=*/false,
                       ACL_SPARSE_INDEX_BASE_ZERO, ACL_SPARSE_STATUS_INVALID_VALUE);
    std::cout << "[ERR_m_negative] PASSED (INVALID_VALUE)\n";
}

TEST_F(Csr2CooExceptionTest, NnzExceedsInt32Max) {
    std::cout << "==== ERR_nnz_exceeds_int32_max ====\n";

    HandleManager handle;
    handle.setStream(stream_);

    int32_t rowPtr[] = {0, 1, 2, 3, 4};
    auto dRowPtr = DeviceBuffer::copyFrom(rowPtr, sizeof(rowPtr));
    auto dCooRowInd = DeviceBuffer::alloc(sizeof(int32_t));

    auto ret = aclsparseXcsr2coo(
        handle.get(),
        static_cast<const int32_t *>(dRowPtr.get()),
        static_cast<int64_t>(std::numeric_limits<int32_t>::max()) + 1,
        4,
        static_cast<int32_t *>(dCooRowInd.get()),
        ACL_SPARSE_INDEX_BASE_ZERO);
    EXPECT_EQ(ret, ACL_SPARSE_STATUS_INVALID_VALUE);

    std::cout << "[ERR_nnz_exceeds_int32_max] PASSED (INVALID_VALUE)\n";
}

TEST_F(Csr2CooExceptionTest, MExceedsInt32Max) {
    std::cout << "==== ERR_m_exceeds_int32_max ====\n";

    HandleManager handle;
    handle.setStream(stream_);

    int32_t rowPtr[] = {0, 0};
    auto dRowPtr = DeviceBuffer::copyFrom(rowPtr, sizeof(rowPtr));
    auto dCooRowInd = DeviceBuffer::alloc(sizeof(int32_t));

    auto ret = aclsparseXcsr2coo(
        handle.get(),
        static_cast<const int32_t *>(dRowPtr.get()),
        0,
        static_cast<int64_t>(std::numeric_limits<int32_t>::max()) + 1,
        static_cast<int32_t *>(dCooRowInd.get()),
        ACL_SPARSE_INDEX_BASE_ZERO);
    EXPECT_EQ(ret, ACL_SPARSE_STATUS_INVALID_VALUE);

    std::cout << "[ERR_m_exceeds_int32_max] PASSED (INVALID_VALUE)\n";
}

// ============================================================================
// Legal boundary cases: inputs that pass validation and return SUCCESS.
// These exercise the early-return / null-pointer legal paths.
// ============================================================================

// ---------------------------------------------------------------------------
// BND_m0_null_rowptr: m=0 且 csrRowPtr=nullptr 为合法边界（校验通过后 nnz==0
// 提前返回 SUCCESS，不触碰 csrRowPtr/cooRowInd）。
// ---------------------------------------------------------------------------

TEST_F(Csr2CooExceptionTest, M0NullRowPtrLegal) {
    std::cout << "==== BND_m0_null_rowptr ====\n";
    RunCsr2cooAndCheck(stream_, nullptr, 0, 0, 0,
                       /*nullHandle=*/false, /*nullCooRowInd=*/true,
                       ACL_SPARSE_INDEX_BASE_ZERO, ACL_SPARSE_STATUS_SUCCESS);
    std::cout << "[BND_m0_null_rowptr] PASSED (SUCCESS)\n";
}

// ---------------------------------------------------------------------------
// BND_nnz0_null_cooRowInd: nnz=0 且 cooRowInd=nullptr 为合法边界（nnz==0
// 提前返回 SUCCESS，不写 cooRowInd）。
// ---------------------------------------------------------------------------

TEST_F(Csr2CooExceptionTest, Nnz0NullCooRowIndLegal) {
    std::cout << "==== BND_nnz0_null_cooRowInd ====\n";
    int32_t rowPtr[] = {0, 0, 0, 0, 0};  // m=4, 全空行
    RunCsr2cooAndCheck(stream_, rowPtr, 5, 0, 4,
                       /*nullHandle=*/false, /*nullCooRowInd=*/true,
                       ACL_SPARSE_INDEX_BASE_ZERO, ACL_SPARSE_STATUS_SUCCESS);
    std::cout << "[BND_nnz0_null_cooRowInd] PASSED (SUCCESS)\n";
}

// ============================================================================
// Abnormal-input robustness cases: inputs that pass validation (so the kernel
// is launched) but violate CSR semantic preconditions. The kernel detects the
// anomaly (negative rowNnz / negative rowStart) and skips the affected row
// (continue) without crashing or writing out of bounds. These verify the
// kernel's defensive continue paths (Fix 1: unified with golden/SIMT).
// ============================================================================

// ---------------------------------------------------------------------------
// ABN_non_monotonic: csrRowPtr=[0,3,1,4] 非单调（m=3,nnz=4,idxBase=0）。
//   row1 的 rowNnz=rowPtr[2]-rowPtr[1]=1-3=-2 < 0 → 该行 continue 跳过，不崩溃。
// ---------------------------------------------------------------------------

TEST_F(Csr2CooExceptionTest, NonMonotonicRowPtr) {
    std::cout << "==== ABN_non_monotonic ====\n";
    int32_t rowPtr[] = {0, 3, 1, 4};
    RunCsr2cooAndCheck(stream_, rowPtr, 4, 4, 3,
                       /*nullHandle=*/false, /*nullCooRowInd=*/false,
                       ACL_SPARSE_INDEX_BASE_ZERO, ACL_SPARSE_STATUS_SUCCESS,
                       /*syncStream=*/true,
                       /*compareGolden=*/false);  // 非单调输入：SIMT 并行写竞态属 UB，仅验证不崩溃
    std::cout << "[ABN_non_monotonic] PASSED (SUCCESS, no crash)\n";
}

// ---------------------------------------------------------------------------
// ABN_nnz_mismatch: csrRowPtr=[0,1,2,3]（m=3）但传 nnz=5（csrRowPtr[m]-base=3≠5）。
//   Kernel 按行实际 rowNnz(=1) 写入，仅写 3 个元素，不越界，不崩溃。
// ---------------------------------------------------------------------------

TEST_F(Csr2CooExceptionTest, NnzMismatch) {
    std::cout << "==== ABN_nnz_mismatch ====\n";
    int32_t rowPtr[] = {0, 1, 2, 3};
    RunCsr2cooAndCheck(stream_, rowPtr, 4, 5, 3,
                       /*nullHandle=*/false, /*nullCooRowInd=*/false,
                       ACL_SPARSE_INDEX_BASE_ZERO, ACL_SPARSE_STATUS_SUCCESS,
                       /*syncStream=*/true);
    std::cout << "[ABN_nnz_mismatch] PASSED (SUCCESS, no crash)\n";
}

// ---------------------------------------------------------------------------
// ABN_idxbase_mismatch: idxBase=1 但 csrRowPtr[0]=0（[0,1,2,3,4], m=4, nnz=4）。
//   startLocal[0]=rowPtr[0]-idxBase=0-1=-1 < 0 → row0 continue 跳过，不崩溃。
// ---------------------------------------------------------------------------

TEST_F(Csr2CooExceptionTest, IdxBaseMismatch) {
    std::cout << "==== ABN_idxbase_mismatch ====\n";
    int32_t rowPtr[] = {0, 1, 2, 3, 4};
    RunCsr2cooAndCheck(stream_, rowPtr, 5, 4, 4,
                       /*nullHandle=*/false, /*nullCooRowInd=*/false,
                       ACL_SPARSE_INDEX_BASE_ONE, ACL_SPARSE_STATUS_SUCCESS,
                       /*syncStream=*/true);
    std::cout << "[ABN_idxbase_mismatch] PASSED (SUCCESS, no crash)\n";
}

// ---------------------------------------------------------------------------
// T1: SIMD 路径异常输入用例。现有 3 个 ABN 用例均 m>1 且 nnz/m=1 → 全走 SIMT。
// 以下用例通过 m=1 或 nnz/m>128 强制走 SIMD 路径，覆盖 ProcessRowsImpl 的
// 防御性 continue（rowNnz<0, rowStart<0, rowStart+rowNnz>nnz）。
// ---------------------------------------------------------------------------

// ABN_simd_row_exceeds_nnz: m=1 → SIMD 路径。rowPtr=[0,5], nnz=3 → rowNnz=5>nnz=3
//   → rowStart+rowNnz=5>nnz=3 → continue 跳过。golden 也跳过。输出全 0。
TEST_F(Csr2CooExceptionTest, SimdRowExceedsNnz) {
    std::cout << "==== ABN_simd_row_exceeds_nnz ====\n";
    int32_t rowPtr[] = {0, 5};
    RunCsr2cooAndCheck(stream_, rowPtr, 2, 3, 1,
                       /*nullHandle=*/false, /*nullCooRowInd=*/false,
                       ACL_SPARSE_INDEX_BASE_ZERO, ACL_SPARSE_STATUS_SUCCESS,
                       /*syncStream=*/true);
    std::cout << "[ABN_simd_row_exceeds_nnz] PASSED (SUCCESS, SIMD)\n";
}

// ABN_simd_non_monotonic: m=2, nnz=300 → nnz/m=150>128 → SIMD 路径。
//   rowPtr=[0,250,100] → row1 rowNnz=100-250=-150<0 → continue 跳过。
//   golden 也跳过 row1。输出: [0]*250 + [0]*50 (unwritten zeros)。
TEST_F(Csr2CooExceptionTest, SimdNonMonotonic) {
    std::cout << "==== ABN_simd_non_monotonic ====\n";
    int32_t rowPtr[] = {0, 250, 100};
    RunCsr2cooAndCheck(stream_, rowPtr, 3, 300, 2,
                       /*nullHandle=*/false, /*nullCooRowInd=*/false,
                       ACL_SPARSE_INDEX_BASE_ZERO, ACL_SPARSE_STATUS_SUCCESS,
                       /*syncStream=*/true);
    std::cout << "[ABN_simd_non_monotonic] PASSED (SUCCESS, SIMD)\n";
}

// ABN_simd_idxbase_mismatch: m=1 → SIMD 路径。idxBase=1 但 rowPtr=[0,3]。
//   rowStart=0-1=-1<0 → continue 跳过。golden 也跳过。输出全 0。
TEST_F(Csr2CooExceptionTest, SimdIdxBaseMismatch) {
    std::cout << "==== ABN_simd_idxbase_mismatch ====\n";
    int32_t rowPtr[] = {0, 3};
    RunCsr2cooAndCheck(stream_, rowPtr, 2, 3, 1,
                       /*nullHandle=*/false, /*nullCooRowInd=*/false,
                       ACL_SPARSE_INDEX_BASE_ONE, ACL_SPARSE_STATUS_SUCCESS,
                       /*syncStream=*/true);
    std::cout << "[ABN_simd_idxbase_mismatch] PASSED (SUCCESS, SIMD)\n";
}

// ---------------------------------------------------------------------------
// T3: B1 回归用例。连续小行累积触发 SIMD 路径 AccumulateRow 的 tail 循环。
// B1 bug: alignedEnd < writeStart 时 tail 越界写覆盖前序行数据。
// 构造 m=60 (>56 核) 使 blockSize=2，core 0 处理 rows 0-1：
//   row0 nnz=3 → accumElems=3 (非 8 对齐)
//   row1 nnz=2 → writeStart=3, writeEnd=5, alignedEnd=0 < 3 → B1 触发（已修复）
// nnz/m=128.7>128 → SIMD 路径。golden 比对验证输出正确性。
// ---------------------------------------------------------------------------
TEST_F(Csr2CooExceptionTest, B1RegressionSmallRows) {
    std::cout << "==== B1_regression_small_rows ====\n";
    int64_t m = 60;
    int64_t nnz = 5 + 58 * 133;  // = 7719, nnz/m ≈ 128.7 > 128 → SIMD
    std::vector<int32_t> rowPtr(static_cast<size_t>(m) + 1);
    rowPtr[0] = 0;
    rowPtr[1] = 3;   // row0: nnz=3
    rowPtr[2] = 5;   // row1: nnz=2
    for (int64_t i = 3; i <= m; i++) {
        rowPtr[static_cast<size_t>(i)] = static_cast<int32_t>(5 + (i - 2) * 133);
    }
    RunCsr2cooAndCheck(stream_, rowPtr.data(), static_cast<size_t>(m) + 1, nnz, m,
                       /*nullHandle=*/false, /*nullCooRowInd=*/false,
                       ACL_SPARSE_INDEX_BASE_ZERO, ACL_SPARSE_STATUS_SUCCESS,
                       /*syncStream=*/true);
    std::cout << "[B1_regression_small_rows] PASSED (SUCCESS, SIMD)\n";
}

// ---------------------------------------------------------------------------
// B7: 确定性路径边界用例（手工构造 rowPtr，不依赖随机 binomial）。
//   路径选择阈值：nnz/m <= 128 → SIMT；nnz/m > 128 → SIMD。
//   原 CSV 127/128/129 边界用例依赖 binomial_distribution 随机命中，跨 STL
//   可能偏移；此处手工构造每行精确 nnz，保证路径选择确定。
// ---------------------------------------------------------------------------

// m=10, 每行精确 127 nnz → nnz/m=127 ≤ 128 → SIMT 路径
TEST_F(Csr2CooExceptionTest, DeterministicSimtBoundary127) {
    std::cout << "==== B7_simt_boundary_127 ====\n";
    int64_t m = 10;
    int64_t nnz = 1270;
    std::vector<int32_t> rowPtr(static_cast<size_t>(m) + 1);
    for (int64_t i = 0; i <= m; i++) {
        rowPtr[static_cast<size_t>(i)] = static_cast<int32_t>(127 * i);
    }
    RunCsr2cooAndCheck(stream_, rowPtr.data(), static_cast<size_t>(m) + 1, nnz, m,
                       /*nullHandle=*/false, /*nullCooRowInd=*/false,
                       ACL_SPARSE_INDEX_BASE_ZERO, ACL_SPARSE_STATUS_SUCCESS,
                       /*syncStream=*/true);
    std::cout << "[B7_simt_boundary_127] PASSED (SUCCESS, SIMT)\n";
}

// m=10, 每行精确 128 nnz → nnz/m=128 ≤ 128 → SIMT 路径（边界）
TEST_F(Csr2CooExceptionTest, DeterministicSwitchBoundary128) {
    std::cout << "==== B7_switch_boundary_128 ====\n";
    int64_t m = 10;
    int64_t nnz = 1280;
    std::vector<int32_t> rowPtr(static_cast<size_t>(m) + 1);
    for (int64_t i = 0; i <= m; i++) {
        rowPtr[static_cast<size_t>(i)] = static_cast<int32_t>(128 * i);
    }
    RunCsr2cooAndCheck(stream_, rowPtr.data(), static_cast<size_t>(m) + 1, nnz, m,
                       /*nullHandle=*/false, /*nullCooRowInd=*/false,
                       ACL_SPARSE_INDEX_BASE_ZERO, ACL_SPARSE_STATUS_SUCCESS,
                       /*syncStream=*/true);
    std::cout << "[B7_switch_boundary_128] PASSED (SUCCESS, SIMT boundary)\n";
}

// m=10, 每行精确 129 nnz → nnz/m=129 > 128 → SIMD 路径
TEST_F(Csr2CooExceptionTest, DeterministicSimdBoundary129) {
    std::cout << "==== B7_simd_boundary_129 ====\n";
    int64_t m = 10;
    int64_t nnz = 1290;
    std::vector<int32_t> rowPtr(static_cast<size_t>(m) + 1);
    for (int64_t i = 0; i <= m; i++) {
        rowPtr[static_cast<size_t>(i)] = static_cast<int32_t>(129 * i);
    }
    RunCsr2cooAndCheck(stream_, rowPtr.data(), static_cast<size_t>(m) + 1, nnz, m,
                       /*nullHandle=*/false, /*nullCooRowInd=*/false,
                       ACL_SPARSE_INDEX_BASE_ZERO, ACL_SPARSE_STATUS_SUCCESS,
                       /*syncStream=*/true);
    std::cout << "[B7_simd_boundary_129] PASSED (SUCCESS, SIMD)\n";
}

// ---------------------------------------------------------------------------
// B8: ProcessRowChunked 多 chunk 确定性用例。
//   m=1, nnz=100000（单行超大），nnz/m=100000 > 128 → SIMD 路径。
//   rowNnz=100000 > cooChunkSize（由 UB 预算决定，通常数千），必然触发
//   ProcessRowChunked 的 while 主块多 chunk 循环。
// ---------------------------------------------------------------------------

TEST_F(Csr2CooExceptionTest, ProcessRowChunkedMultiChunk) {
    std::cout << "==== B8_process_row_chunked_multi_chunk ====\n";
    int64_t m = 1;
    int64_t nnz = 100000;
    std::vector<int32_t> rowPtr = {0, 100000};
    RunCsr2cooAndCheck(stream_, rowPtr.data(), static_cast<size_t>(m) + 1, nnz, m,
                       /*nullHandle=*/false, /*nullCooRowInd=*/false,
                       ACL_SPARSE_INDEX_BASE_ZERO, ACL_SPARSE_STATUS_SUCCESS,
                       /*syncStream=*/true);
    std::cout << "[B8_process_row_chunked_multi_chunk] PASSED (SUCCESS, SIMD multi-chunk)\n";
}

// ---------------------------------------------------------------------------
// BND_stream_nullptr: handle created (aclsparseCreate) but aclsparseSetStream
//   not called. stream defaults to nullptr (default stream), which is allowed
//   (aligned with cuSPARSE §4.2.9 and same-warehouse nnz). Kernel launches on
//   the default stream; sync default stream to verify no crash.
// ---------------------------------------------------------------------------

TEST_F(Csr2CooExceptionTest, StreamNullptr) {
    std::cout << "==== BND_stream_nullptr ====\n";

    HandleManager handle;  // creates handle, but does NOT call setStream

    int32_t rowPtr[] = {0, 1, 2, 3, 4};
    auto dRowPtr = DeviceBuffer::copyFrom(rowPtr, sizeof(rowPtr));
    auto dCooRowInd = DeviceBuffer::alloc(4 * sizeof(int32_t));
    EXPECT_EQ(aclrtMemset(dCooRowInd.get(), 4 * sizeof(int32_t), 0, 4 * sizeof(int32_t)), ACL_SUCCESS);

    auto ret = aclsparseXcsr2coo(
        handle.get(),
        static_cast<const int32_t *>(dRowPtr.get()),
        4,
        4,
        static_cast<int32_t *>(dCooRowInd.get()),
        ACL_SPARSE_INDEX_BASE_ZERO);
    EXPECT_EQ(ret, ACL_SPARSE_STATUS_SUCCESS);

    // stream is nullptr (default stream); sync default stream to ensure kernel completes
    EXPECT_EQ(aclrtSynchronizeStream(nullptr), ACL_SUCCESS);

    // T2: golden comparison
    std::vector<int32_t> ptrVec(rowPtr, rowPtr + 5);
    auto golden = Csr2CooGolden(ptrVec, 4, 4, 0);
    std::vector<int32_t> npuResult(4, 0);
    dCooRowInd.copyToHost(npuResult.data(), 4 * sizeof(int32_t));
    EXPECT_TRUE(VerifyInt32Bitwise(npuResult, golden, "stream_nullptr_golden"));

    std::cout << "[BND_stream_nullptr] PASSED (SUCCESS, default stream)\n";
}

// 禁止定义 main 函数（由 test/frame/test_main.cpp 提供）
