/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * @file spsm_test.cpp
 * @brief GTest + CSV-driven test cases for aclsparseSpSM (v2, Generic API).
 *
 * v2 rewrite: supports NON_UNIT diag, ROW/COL order, inplace, unsorted indices,
 * banded/block-diag matrix structures, singular matrix detection.
 *
 * Test structure:
 *   - TEST_P (SpsmTest)         : L0/L1/L2-S/L2-M success/NOT_SUPPORTED from CSV (94 cases)
 *   - TEST_F (SpsmExceptionTest): L2-X parameter-validation (20 cases, hardcoded)
 *
 * Entry point: test/frame/test_main.cpp (tests MUST NOT define main()).
 */

#include "test_common.h"

#include "../spsm_golden.h"
#include "../spsm_param.h"
#include "../spsm_fp16_util.h"  // SpsmFp32ToFp16Bits (L2_X18 valueType-mismatch case)
#include "spsm_npu_wrapper.h"

using namespace sparse_test;

// ============================================================================
// Helpers: data generation
// ============================================================================

// Build the base triangular CSR according to param fields (no conditioning yet):
//   matrix_struct: RANDOM / BANDED / BLOCK_DIAG
//   diag: UNIT / NON_UNIT (NON_UNIT stores explicit diagonal in [1.0, 2.0])
//   a_empty: 1 -> off-diagonal nnz=0 (UNIT: empty; NON_UNIT: diagonal only)
//   singular: 1 -> NON_UNIT with one zero diagonal element
static CsrMatrix BuildBaseCsr(const SpsmParam& p, bool isUnit) {
    const double diagLo = 1.0;
    const double diagHi = 2.0;

    // a_empty: no off-diagonal entries
    if (p.a_empty) {
        if (isUnit) {
            // UNIT + a_empty: truly empty CSR (diagonal implicit 1.0)
            return makeEmptyCsr(p.m, p.m);
        }
        // NON_UNIT + a_empty: diagonal-only CSR (m entries on diagonal)
        return makeTriangularCsrNonUnit(p.m, p.isLower(), 0.0,
                                         p.value_lo, p.value_hi,
                                         diagLo, diagHi, p.seed);
    }

    // singular: NON_UNIT with one zero diagonal element
    if (p.singular) {
        return makeSingularTriangularCsr(p.m, p.isLower(), p.density,
                                          p.value_lo, p.value_hi,
                                          diagLo, diagHi, p.seed);
    }

    if (p.isBanded()) {
        return makeBandedTriangularCsr(p.m, p.isLower(), p.bw,
                                        p.value_lo, p.value_hi,
                                        isUnit, diagLo, diagHi, p.seed);
    }
    if (p.isBlockDiag()) {
        return makeBlockDiagTriangularCsr(p.m, p.isLower(), p.blk,
                                           p.density, p.value_lo, p.value_hi,
                                           isUnit, diagLo, diagHi, p.seed);
    }

    // RANDOM
    if (isUnit) {
        return makeTriangularCsr(p.m, p.isLower(), p.density,
                                  p.value_lo, p.value_hi, p.seed);
    }
    return makeTriangularCsrNonUnit(p.m, p.isLower(), p.density,
                                     p.value_lo, p.value_hi,
                                     diagLo, diagHi, p.seed);
}

// Set triangular matrix attributes: fillMode=LOWER + diagType (UNIT or NON_UNIT).
// Extracted to eliminate duplicate code blocks across exception test cases.
static void SetTriangularAttributes(aclsparseSpMatDescr *matA, aclsparseDiagType_t diagType)
{
    aclsparseFillMode_t fillMode = ACL_SPARSE_FILL_MODE_LOWER;
    EXPECT_EQ(aclsparseSpMatSetAttribute(matA, ACL_SPARSE_SPMAT_FILL_MODE,
                                          &fillMode, sizeof(fillMode)),
              ACL_SPARSE_STATUS_SUCCESS);
    EXPECT_EQ(aclsparseSpMatSetAttribute(matA, ACL_SPARSE_SPMAT_DIAG_TYPE,
                                          &diagType, sizeof(diagType)),
              ACL_SPARSE_STATUS_SUCCESS);
}

// Scale all entries by `scale` (off-diagonal conditioning).
// NON_UNIT diagonal is re-asserted afterwards by ReassertDiagonalValues
// (diagonal in [1,2] is dominant, but scaling may shrink it).
static void ScaleOffDiagonal(CsrMatrix& csr, double scale) {
    for (auto& val : csr.values) {
        val = static_cast<float>(static_cast<double>(val) * scale);
    }
}

// RNG seed salt for ReassertDiagonalValues (decouples diag re-generation from
// the base matrix seed, avoiding correlated off-diag/diag RNG streams).
constexpr uint32_t kDiagReassertSeedSalt = 0xA5A5u;

// Re-assert diagonal values for NON_UNIT after scaling (may have shrunk them).
static void ReassertDiagonalValues(CsrMatrix& csr, const SpsmParam& p) {
    std::mt19937 rng(p.seed ^ kDiagReassertSeedSalt);
    std::uniform_real_distribution<float> diagDist(1.0f, 2.0f);
    for (int i = 0; i < p.m; ++i) {
        int rs = csr.rowOffsets[i];
        int re = csr.rowOffsets[i + 1];
        for (int k = rs; k < re; ++k) {
            if (csr.colIndices[k] == i) {
                csr.values[k] = diagDist(rng);
                break;
            }
        }
    }
}

// Conditioning: scale off-diagonal values to keep diagonal dominance.
// Target: E[sum of |off-diag| per row] <= 0.5.
//   RANDOM : expected nnz/row = m * density -> scale = 5/(m*v)
//   BANDED : expected nnz/row = bw * 0.5    -> scale = 2/(bw*v)
//            (BANDED was previously excluded, which left large off-diag row
//             sums for bw=8 and inflated |X| up to ~78, amplifying FP32
//             cancellation in trans=T solves — L1_25 precision boundary.)
//   BLOCK_DIAG : excluded (block-local, naturally well-conditioned).
static void ApplyConditioning(CsrMatrix& csr, const SpsmParam& p, bool isUnit) {
    const double v = std::max(std::abs(p.value_lo), std::abs(p.value_hi));
    if (v <= 0.0 || p.isBlockDiag()) {
        return;
    }
    double scale;
    if (p.isBanded()) {
        // BANDED: expected nnz/row = bw * 0.5 (50% density within band).
        // Guard bw<=0 explicitly (makeBandedTriangularCsr with bw=0 yields a
        // diagonal-only matrix); without this the 2.0/(bw*v) division relies on
        // IEEE 754 inf -> std::min(1.0, inf)=1.0 -> no-op, which is fragile.
        if (p.bw <= 0) {
            return;
        }
        scale = std::min(1.0, 2.0 / (static_cast<double>(p.bw) * v));
    } else {
        // RANDOM
        scale = std::min(1.0, 5.0 / (static_cast<double>(p.m) * v));
    }
    if (scale >= 1.0) {
        return;
    }
    ScaleOffDiagonal(csr, scale);
    if (!isUnit) {
        ReassertDiagonalValues(csr, p);
    }
}

// Build the triangular CSR matrix A according to param fields:
//   matrix_struct: RANDOM / BANDED / BLOCK_DIAG
//   diag: UNIT / NON_UNIT (NON_UNIT stores explicit diagonal in [1.0, 2.0])
//   a_empty: 1 -> off-diagonal nnz=0 (UNIT: empty; NON_UNIT: diagonal only)
//   singular: 1 -> NON_UNIT with one zero diagonal element
//   unsorted: 1 -> row-internal colInd shuffle (applied after generation)
//   index_base: ONE -> idxBase=ONE descriptor transmitted to NPU; colInd stays
//               0-based (opA=N kernel uses as-is; opA=T host Csr2Csc normalizes).
//               golden always receives 0-based colInd.
//
// Conditioning: scale off-diagonal values for large m to keep diagonal dominance.
static CsrMatrix GenerateTriangularA(const SpsmParam& p) {
    const bool isUnit = p.isUnitDiag();
    CsrMatrix csr = BuildBaseCsr(p, isUnit);
    // singular / a_empty paths must keep their early-return semantics:
    // skip conditioning so makeSingularTriangularCsr's zero diagonal
    // elements are not overwritten by ReassertDiagonalValues.
    if (!p.singular && !p.a_empty) {
        ApplyConditioning(csr, p, isUnit);
    }

    // unsorted: Shuffle colInd within each row
    if (p.isUnsorted()) {
        shuffleRowInternal(csr, p.seed);
    }

    return csr;
}

// Build dense RHS B (layout per order, ldb x n, FP32). b_zero -> all zeros.
// COL order: B[i,j] at B[j*ldb + i] (column-major)
// ROW order: B[i,j] at B[i*ldb + j] (row-major)
static std::vector<float> GenerateB(const SpsmParam& p) {
    // Buffer size: COL order -> ldb * n; ROW order -> ldb * m (v2 §3.10).
    const int64_t count = p.isRowOrder()
        ? static_cast<int64_t>(p.ldb) * p.m
        : static_cast<int64_t>(p.ldb) * p.n;
    if (p.b_zero) {
        return std::vector<float>(static_cast<size_t>(count), 0.0f);
    }
    std::vector<float> B(static_cast<size_t>(count), 0.0f);
    if (p.m <= 0 || p.n <= 0) return B;
    std::mt19937 rng(p.seed + 1000);
    std::uniform_real_distribution<float> dist(static_cast<float>(p.value_lo),
                                                 static_cast<float>(p.value_hi));
    for (int j = 0; j < p.n; ++j) {
        for (int i = 0; i < p.m; ++i) {
            int64_t off = p.isRowOrder()
                ? static_cast<int64_t>(i) * p.ldb + j
                : static_cast<int64_t>(j) * p.ldb + i;
            B[off] = dist(rng);
        }
    }
    return B;
}

static void PrintCaseInfo(const SpsmParam& p) {
    std::cout << "==== " << p.case_name
              << " ==== m=" << p.m << " n=" << p.n << " ldb=" << p.ldb
              << " dtype=" << p.dtype << " uplo=" << p.uplo << " trans=" << p.trans
              << " opB=" << p.opB
              << " diag=" << p.diag << " order=" << p.order
              << " alpha=" << p.alpha << " density=" << p.density
              << " seed=" << p.seed
              << " struct=" << p.matrix_struct << " bw=" << p.bw << " blk=" << p.blk
              << " inplace=" << p.inplace << " unsorted=" << p.unsorted
              << " b_zero=" << p.b_zero << " a_empty=" << p.a_empty
              << " singular=" << p.singular << " index_base=" << p.index_base
              << " expect=" << p.expect_result
              << "\n";
}

// ============================================================================
// GTest parameterized fixture: SpsmTest
// ============================================================================
class SpsmTest : public testing::TestWithParam<SpsmParam> {
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

    SpsmParam param_;

    void SetUp() override { param_ = GetParam(); }

    // 1. Generate triangular A (CSR) and dense B (per order layout).
    //
    // Golden receives 0-based colInd (Eigen requires 0-based indices).
    //
    // indexBase handling per opA path:
    //   opA=N: kernel reads colInd and subtracts td_.indexBase
    //          (`j = colIndLocal.GetValue(k) - td_.indexBase`). host
    //          trusts the descriptor base directly (descriptorBase=1 -> effectiveBase=1,
    //          kernel subtracts 1; descriptorBase=0 -> effectiveBase=0, no
    //          subtraction). No colInd scanning heuristic.
    //   opA=T: host Csr2CscHost normalizes ONE->ZERO during the transpose
    //          (Csr2CscNormalizeColInd subtracts indexBase), so the transposed
    //          CSR is 0-based and td_.indexBase is set to 0 (kernel does not
    //          subtract again).
    // GenerateTriangularA always produces 0-based colInd, which matches both
    // NPU (opA=N with ZERO descriptor -> no subtract / opA=T host-normalized)
    // and golden (0-based). The idxBase=ONE descriptor is still transmitted to
    // the NPU wrapper to verify the operator accepts it; with 0-based data +
    // ONE descriptor, opA=N kernel subtracts 1 (yielding -1 colInd, exercising
    // the descriptor-trust path), opA=T host Csr2Csc normalizes ONE->ZERO.
    //
    // L3 whitebox (one_based_data=1): transmit REAL 1-based colInd to NPU
    // (csrA += 1 on every colInd), while golden keeps the 0-based copy.
    //   opA=N path: host SpsmDetectEffectiveIndexBase trusts descriptor
    //     (ONE -> effectiveBase=1) -> kernel subtracts indexBase=1 to
    //     normalize back to 0-based. Correct result.
    //   opA=T path: host SpsmCsr2CscHost normalizes ONE->ZERO (colInd -= 1)
    //     inside the transpose routine (Csr2CscNormalizeColInd).
    // This covers the "real 1-based normalization" branches that the default
    // (0-based data + ZERO descriptor) path does not exercise.
    void PrepareInputs(CsrMatrix& csrA, std::vector<float>& B_host,
                       CsrMatrix& csrAForGolden) const {
        const auto& p = param_;
        csrA = GenerateTriangularA(p);
        B_host = GenerateB(p);
        csrAForGolden = csrA;
        if (p.isOneBasedData()) {
            applyIndexBaseOne(csrA);  // csrA becomes 1-based; golden stays 0-based
        }
    }

    // 2. Golden reference (Eigen FP64 triangular solve).
    //    Computed lazily: only needed for success-path verification.
    // opB=T is an early-reject path (BufferSize returns NOT_SUPPORTED before any
    // solve). Skip golden computation (no X to verify) — only the NPU return
    // code is checked via expect_result=NOT_SUPPORTED below.
    void RunGolden(const CsrMatrix& csrAForGolden, const std::vector<float>& B_host,
                   bool isOpBReject, SpsmGoldenResult& golden) const {
        // Skip golden computation when there is no X to verify:
        //   - opB=T: early-reject path (BufferSize returns NOT_SUPPORTED).
        //   - expect_result=NOT_SUPPORTED: only the NPU return code is checked
        //     (e.g. singular matrix), golden is never compared.
        if (isOpBReject || param_.expect_result == "NOT_SUPPORTED") {
            return;
        }
        const auto& p = param_;
        golden = SpsmGolden(
            csrAForGolden.rowOffsets, csrAForGolden.colIndices, csrAForGolden.values,
            p.m, p.n, p.ldb, B_host, p.alpha, p.isLower(), p.isTranspose(),
            p.isUnitDiag(), p.isRowOrder());
    }

    // 3. NPU solve (three-stage Generic API, with opB).
    SpsmNpuResult RunNpu(const CsrMatrix& csrA,
                         const std::vector<float>& B_host) const {
        const auto& p = param_;
        aclsparseOperation_t opA = p.isTranspose() ? ACL_SPARSE_OP_TRANSPOSE
                                                     : ACL_SPARSE_OP_NON_TRANSPOSE;
        aclsparseOperation_t opB = p.isOpBTranspose() ? ACL_SPARSE_OP_TRANSPOSE
                                                        : ACL_SPARSE_OP_NON_TRANSPOSE;
        aclsparseFillMode_t uplo = p.isLower() ? ACL_SPARSE_FILL_MODE_LOWER
                                                : ACL_SPARSE_FILL_MODE_UPPER;
        aclsparseDiagType_t diagType = p.isUnitDiag() ? ACL_SPARSE_DIAG_TYPE_UNIT
                                                        : ACL_SPARSE_DIAG_TYPE_NON_UNIT;
        aclsparseOrder_t order = p.isRowOrder() ? ACL_SPARSE_ORDER_ROW
                                                  : ACL_SPARSE_ORDER_COL;
        aclsparseIndexBase_t idxBase = p.isIndexBaseOne() ? ACL_SPARSE_INDEX_BASE_ONE
                                                             : ACL_SPARSE_INDEX_BASE_ZERO;
        return SpsmNpu(*spHandle_, env_->stream(),
                       csrA.rowOffsets, csrA.colIndices, csrA.values,
                       static_cast<int32_t>(csrA.nnz),
                       B_host, p.m, p.n, p.ldb, p.alpha,
                       opA, opB, uplo, diagType, order,
                       p.isInplace(), idxBase);
    }

    // 4. Assert NOT_SUPPORTED path (singular matrix / opB=T reject).
    void AssertNotSupported(const SpsmNpuResult& npu) const {
        const auto& p = param_;
        bool rejected = (npu.bufferSizeRet == ACL_SPARSE_STATUS_NOT_SUPPORTED ||
                         npu.analysisRet == ACL_SPARSE_STATUS_NOT_SUPPORTED ||
                         npu.solveRet == ACL_SPARSE_STATUS_NOT_SUPPORTED);
        EXPECT_TRUE(rejected) << p.case_name
                              << " expected NOT_SUPPORTED but got"
                              << " buf=" << npu.bufferSizeRet
                              << " analysis=" << npu.analysisRet
                              << " solve=" << npu.solveRet;
        std::cout << "[" << p.case_name << "] NOT_SUPPORTED path confirmed\n";
    }

    // 5. Verify: compare the m x n valid region (layout per order).
    //    MixedTolerance precision verification.
    void VerifyResults(const SpsmNpuResult& npu,
                       const SpsmGoldenResult& golden) const {
        const auto& p = param_;
        sparse_test::VerifyConfig cfg;
        // FP32 only (computeType=ACL_FLOAT); MIXED_TOLERANCE with FP32 defaults.
        sparse_test::applyMixedTolerance(cfg, ACL_FLOAT, golden.X.data(), golden.X.size());

        // Extract valid m x n region per order layout.
        std::vector<float> X_npu, X_golden;
        X_npu.reserve(static_cast<size_t>(p.m) * p.n);
        X_golden.reserve(static_cast<size_t>(p.m) * p.n);
        for (int j = 0; j < p.n; ++j) {
            for (int i = 0; i < p.m; ++i) {
                int64_t off = p.isRowOrder()
                    ? static_cast<int64_t>(i) * p.ldb + j
                    : static_cast<int64_t>(j) * p.ldb + i;
                X_npu.push_back(npu.X[off]);
                X_golden.push_back(golden.X[off]);
            }
        }

        bool pass = sparse_test::Verifier::verifyVector(X_npu, X_golden, cfg, p.case_name);
        // Performance capture: per-stage latency + level metadata.
        std::cout << "[" << p.case_name << "] Perf(ms): BufferSize=" << npu.timingMs[0]
                  << " Analysis=" << npu.timingMs[1]
                  << " Solve=" << npu.timingMs[2]
                  << " Total=" << npu.timingMs[3]
                  << " L=" << npu.levelCount
                  << " kChunk=" << npu.kChunkSize
                  << " maxRowLen=" << npu.maxRowLen << "\n";
        EXPECT_TRUE(pass);
        std::cout << "[" << p.case_name << "] " << (pass ? "PASSED" : "FAILED") << "\n";
    }
};

// ============================================================================
// Test body: parameterized test (L0/L1/L2-S/L2-M)
// ============================================================================
TEST_P(SpsmTest, SpsmSuccess) {
    PrintCaseInfo(param_);

    CsrMatrix csrA, csrAForGolden;
    std::vector<float> B_host;
    PrepareInputs(csrA, B_host, csrAForGolden);

    const bool isOpBReject = param_.isOpBTranspose();
    SpsmGoldenResult golden;
    RunGolden(csrAForGolden, B_host, isOpBReject, golden);

    SpsmNpuResult npu = RunNpu(csrA, B_host);

    // Assert return codes per expect_result.
    if (param_.expect_result == "NOT_SUPPORTED") {
        AssertNotSupported(npu);
        return;
    }
    ASSERT_EQ(npu.bufferSizeRet, ACL_SPARSE_STATUS_SUCCESS);
    ASSERT_EQ(npu.analysisRet, ACL_SPARSE_STATUS_SUCCESS);
    ASSERT_EQ(npu.solveRet, ACL_SPARSE_STATUS_SUCCESS);

    VerifyResults(npu, golden);
}

// ============================================================================
// Parameterized test instantiation from CSV
// ============================================================================
INSTANTIATE_TEST_SUITE_P(
    SpsmCases,
    SpsmTest,
    testing::ValuesIn(GetCasesFromCsv<SpsmParam>("spsm_test.csv")),
    [](const testing::TestParamInfo<SpsmParam>& info) {
        return info.param.case_name;
    });

// ============================================================================
// Exception test fixture: SpsmExceptionTest (L2-X, 20 cases)
// Tests aclsparseSpSMBufferSize (host-side validation stage) with opB.
// ============================================================================
class SpsmExceptionTest : public testing::Test {
public:
    static void SetUpTestSuite() {
        env_ = std::make_unique<AclEnvScope>();
    }
    static void TearDownTestSuite() { env_.reset(); }

protected:
    inline static std::unique_ptr<AclEnvScope> env_;

    aclrtStream stream_ = nullptr;
    std::unique_ptr<HandleManager> handle_;

    // Default valid inputs (m=16, n=8, ldb=16, FP32, N, LOWER, UNIT, COL).
    int m_ = 16;
    int n_ = 8;
    int ldb_ = 16;
    CsrMatrix csrA_;
    std::vector<float> B_;
    float alpha_ = 1.0f;

    std::unique_ptr<DeviceBuffer> dRowPtr_;
    std::unique_ptr<DeviceBuffer> dColInd_;
    std::unique_ptr<DeviceBuffer> dCsrVals_;
    std::unique_ptr<DeviceBuffer> dB_;
    std::unique_ptr<DeviceBuffer> dC_;

    void SetUp() override {
        stream_ = env_->stream();
        handle_ = std::make_unique<HandleManager>();
        handle_->setStream(stream_);

        csrA_ = makeTriangularCsr(m_, true, 0.2, -1.0, 1.0, 100);
        B_ = makeFullColMajor(m_, n_, ldb_, 200, -1.0, 1.0);

        dRowPtr_ = std::make_unique<DeviceBuffer>(
            DeviceBuffer::copyFrom(csrA_.rowOffsets.data(), (m_ + 1) * sizeof(int32_t)));
        dColInd_ = std::make_unique<DeviceBuffer>(
            DeviceBuffer::copyFrom(csrA_.colIndices.data(), csrA_.nnz * sizeof(int32_t)));
        dCsrVals_ = std::make_unique<DeviceBuffer>(
            DeviceBuffer::copyFrom(csrA_.values.data(), csrA_.nnz * sizeof(float)));
        dB_ = std::make_unique<DeviceBuffer>(
            DeviceBuffer::copyFrom(B_.data(), static_cast<size_t>(ldb_) * n_ * sizeof(float)));
        dC_ = std::make_unique<DeviceBuffer>(
            DeviceBuffer::alloc(static_cast<size_t>(ldb_) * n_ * sizeof(float)));
    }

    void TearDown() override {
        dC_.reset();
        dB_.reset();
        dCsrVals_.reset();
        dColInd_.reset();
        dRowPtr_.reset();
        handle_.reset();
    }

    struct ValidDescrs {
        SpMatManager matA;
        DnMatManager matB;
        DnMatManager matC;
        SpsmDescrGuard spsmDescr;
    };

    std::unique_ptr<ValidDescrs> MakeValidDescrs() {
        auto v = std::make_unique<ValidDescrs>();
        v->matA = SpMatManager::createCsr(
            m_, m_, static_cast<int64_t>(csrA_.nnz),
            dRowPtr_->get(), dColInd_->get(), dCsrVals_->get(),
            ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_32I,
            ACL_SPARSE_INDEX_BASE_ZERO, ACL_FLOAT);
        {
            aclsparseFillMode_t fillMode = ACL_SPARSE_FILL_MODE_LOWER;
            EXPECT_EQ(aclsparseSpMatSetAttribute(v->matA.get(), ACL_SPARSE_SPMAT_FILL_MODE,
                                                  &fillMode, sizeof(fillMode)),
                      ACL_SPARSE_STATUS_SUCCESS);
            aclsparseDiagType_t diagType = ACL_SPARSE_DIAG_TYPE_UNIT;
            EXPECT_EQ(aclsparseSpMatSetAttribute(v->matA.get(), ACL_SPARSE_SPMAT_DIAG_TYPE,
                                                  &diagType, sizeof(diagType)),
                      ACL_SPARSE_STATUS_SUCCESS);
        }
        v->matB = DnMatManager::createConst(
            m_, n_, ldb_, dB_->raw(), ACL_FLOAT, ACL_SPARSE_ORDER_COL);
        v->matC = DnMatManager::create(
            m_, n_, ldb_, dC_->get(), ACL_FLOAT, ACL_SPARSE_ORDER_COL);
        return v;
    }

    // Call aclsparseSpSMBufferSize with full control over each argument (v2: opB).
    aclsparseStatus_t CallBufferSize(
        aclsparseHandle_t handle,
        aclsparseOperation_t opA,
        aclsparseOperation_t opB,
        const void* alpha,
        aclsparseConstSpMatDescr_t matA,
        aclsparseConstDnMatDescr_t matB,
        aclsparseDnMatDescr_t matC,
        aclDataType computeType,
        aclsparseSpSMAlg_t alg,
        aclsparseSpSMDescr_t spsmDescr) {
        size_t bufferSize = 0;
        return aclsparseSpSMBufferSize(
            handle, opA, opB, alpha, matA, matB, matC,
            computeType, alg, spsmDescr, &bufferSize);
    }

    // Convenience: default opB=NON_TRANSPOSE, alg=DEFAULT.
    aclsparseStatus_t CallBufferSize(
        aclsparseHandle_t handle,
        aclsparseOperation_t opA,
        const void* alpha,
        aclsparseConstSpMatDescr_t matA,
        aclsparseConstDnMatDescr_t matB,
        aclsparseDnMatDescr_t matC,
        aclDataType computeType,
        aclsparseSpSMDescr_t spsmDescr) {
        return CallBufferSize(handle, opA, ACL_SPARSE_OP_NON_TRANSPOSE, alpha,
                              matA, matB, matC, computeType,
                              ACL_SPARSE_SPSM_ALG_DEFAULT, spsmDescr);
    }
};

// L2_X1: null handle -> ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR
TEST_F(SpsmExceptionTest, NullHandle) {
    auto d = MakeValidDescrs();
    EXPECT_EQ(CallBufferSize(nullptr, ACL_SPARSE_OP_NON_TRANSPOSE, &alpha_,
                              d->matA.cget(), d->matB.cget(), d->matC.get(),
                              ACL_FLOAT, d->spsmDescr.get()),
              ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR);
}

// L2_X2: null matA -> ACL_SPARSE_STATUS_INVALID_VALUE
TEST_F(SpsmExceptionTest, NullMatA) {
    auto d = MakeValidDescrs();
    EXPECT_EQ(CallBufferSize(handle_->get(), ACL_SPARSE_OP_NON_TRANSPOSE, &alpha_,
                              nullptr, d->matB.cget(), d->matC.get(),
                              ACL_FLOAT, d->spsmDescr.get()),
              ACL_SPARSE_STATUS_INVALID_VALUE);
}

// L2_X3: null matB -> ACL_SPARSE_STATUS_INVALID_VALUE
TEST_F(SpsmExceptionTest, NullMatB) {
    auto d = MakeValidDescrs();
    EXPECT_EQ(CallBufferSize(handle_->get(), ACL_SPARSE_OP_NON_TRANSPOSE, &alpha_,
                              d->matA.cget(), nullptr, d->matC.get(),
                              ACL_FLOAT, d->spsmDescr.get()),
              ACL_SPARSE_STATUS_INVALID_VALUE);
}

// L2_X4: null matC -> ACL_SPARSE_STATUS_INVALID_VALUE
TEST_F(SpsmExceptionTest, NullMatC) {
    auto d = MakeValidDescrs();
    EXPECT_EQ(CallBufferSize(handle_->get(), ACL_SPARSE_OP_NON_TRANSPOSE, &alpha_,
                              d->matA.cget(), d->matB.cget(), nullptr,
                              ACL_FLOAT, d->spsmDescr.get()),
              ACL_SPARSE_STATUS_INVALID_VALUE);
}

// L2_X5: null alpha -> ACL_SPARSE_STATUS_INVALID_VALUE
TEST_F(SpsmExceptionTest, NullAlpha) {
    auto d = MakeValidDescrs();
    EXPECT_EQ(CallBufferSize(handle_->get(), ACL_SPARSE_OP_NON_TRANSPOSE, nullptr,
                              d->matA.cget(), d->matB.cget(), d->matC.get(),
                              ACL_FLOAT, d->spsmDescr.get()),
              ACL_SPARSE_STATUS_INVALID_VALUE);
}

// L2_X6: unsupported computeType (INT8) -> ACL_SPARSE_STATUS_NOT_SUPPORTED
TEST_F(SpsmExceptionTest, UnsupportedComputeType_INT8) {
    auto d = MakeValidDescrs();
    EXPECT_EQ(CallBufferSize(handle_->get(), ACL_SPARSE_OP_NON_TRANSPOSE, &alpha_,
                              d->matA.cget(), d->matB.cget(), d->matC.get(),
                              ACL_INT8, d->spsmDescr.get()),
              ACL_SPARSE_STATUS_NOT_SUPPORTED);
}

// L2_X7: unsupported computeType (FP16) -> ACL_SPARSE_STATUS_NOT_SUPPORTED (C1: v2 no FP16)
TEST_F(SpsmExceptionTest, UnsupportedComputeType_FP16) {
    auto d = MakeValidDescrs();
    EXPECT_EQ(CallBufferSize(handle_->get(), ACL_SPARSE_OP_NON_TRANSPOSE, &alpha_,
                              d->matA.cget(), d->matB.cget(), d->matC.get(),
                              ACL_FLOAT16, d->spsmDescr.get()),
              ACL_SPARSE_STATUS_NOT_SUPPORTED);
}

// L2_X8: unsupported opA (CONJUGATE_TRANSPOSE) -> ACL_SPARSE_STATUS_NOT_SUPPORTED
TEST_F(SpsmExceptionTest, UnsupportedOpA_Conjugate) {
    auto d = MakeValidDescrs();
    EXPECT_EQ(CallBufferSize(handle_->get(), ACL_SPARSE_OP_CONJUGATE_TRANSPOSE,
                              ACL_SPARSE_OP_NON_TRANSPOSE, &alpha_,
                              d->matA.cget(), d->matB.cget(), d->matC.get(),
                              ACL_FLOAT, ACL_SPARSE_SPSM_ALG_DEFAULT, d->spsmDescr.get()),
              ACL_SPARSE_STATUS_NOT_SUPPORTED);
}

// L2_X9: unsupported opB (TRANSPOSE) -> ACL_SPARSE_STATUS_NOT_SUPPORTED (v2 new)
TEST_F(SpsmExceptionTest, UnsupportedOpB_Transpose) {
    auto d = MakeValidDescrs();
    EXPECT_EQ(CallBufferSize(handle_->get(), ACL_SPARSE_OP_NON_TRANSPOSE,
                              ACL_SPARSE_OP_TRANSPOSE, &alpha_,
                              d->matA.cget(), d->matB.cget(), d->matC.get(),
                              ACL_FLOAT, ACL_SPARSE_SPSM_ALG_DEFAULT, d->spsmDescr.get()),
              ACL_SPARSE_STATUS_NOT_SUPPORTED);
}

// L2_X10: unsupported alg -> ACL_SPARSE_STATUS_NOT_SUPPORTED
// Cast an invalid alg value to exercise the rejection path.
TEST_F(SpsmExceptionTest, UnsupportedAlg) {
    auto d = MakeValidDescrs();
    aclsparseSpSMAlg_t badAlg = static_cast<aclsparseSpSMAlg_t>(99);
    EXPECT_EQ(CallBufferSize(handle_->get(), ACL_SPARSE_OP_NON_TRANSPOSE,
                              ACL_SPARSE_OP_NON_TRANSPOSE, &alpha_,
                              d->matA.cget(), d->matB.cget(), d->matC.get(),
                              ACL_FLOAT, badAlg, d->spsmDescr.get()),
              ACL_SPARSE_STATUS_NOT_SUPPORTED);
}

// L2_X11: m=0 -> ACL_SPARSE_STATUS_INVALID_VALUE (or rejected at descriptor layer)
TEST_F(SpsmExceptionTest, MZero) {
    auto emptyA = makeEmptyCsr(0, 0);
    auto dRowPtr = DeviceBuffer::copyFrom(emptyA.rowOffsets.data(), sizeof(int32_t));
    auto dColInd = DeviceBuffer::alloc(sizeof(int32_t));
    auto dCsrVals = DeviceBuffer::alloc(sizeof(float));
    SpsmDescrGuard spsmDescr;
    try {
        SpMatManager matA = SpMatManager::createCsr(
            0, 0, 0, dRowPtr.get(), dColInd.get(), dCsrVals.get(),
            ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_32I,
            ACL_SPARSE_INDEX_BASE_ZERO, ACL_FLOAT);
        SetTriangularAttributes(matA.get(), ACL_SPARSE_DIAG_TYPE_UNIT);
        DnMatManager matB = DnMatManager::createConst(
            0, n_, ldb_, dB_->raw(), ACL_FLOAT, ACL_SPARSE_ORDER_COL);
        DnMatManager matC = DnMatManager::create(
            0, n_, ldb_, dC_->get(), ACL_FLOAT, ACL_SPARSE_ORDER_COL);
        auto ret = CallBufferSize(handle_->get(), ACL_SPARSE_OP_NON_TRANSPOSE, &alpha_,
                                  matA.cget(), matB.cget(), matC.get(),
                                  ACL_FLOAT, spsmDescr.get());
        // m=0 is invalid input regardless of order (rejected as INVALID_VALUE
        // at descriptor creation via aclsparseCreateDnMat rows<=0, or at
        // BufferSize if the descriptor layer ever accepts it).
        EXPECT_EQ(ret, ACL_SPARSE_STATUS_INVALID_VALUE);
    } catch (const std::runtime_error&) {
        SUCCEED() << "0-dimension descriptor rejected at creation layer";
    }
}

// L2_X12: n=0 -> ACL_SPARSE_STATUS_INVALID_VALUE (or rejected at descriptor layer)
TEST_F(SpsmExceptionTest, NZero) {
    auto d = MakeValidDescrs();
    try {
        DnMatManager matB0 = DnMatManager::createConst(
            m_, 0, ldb_, dB_->raw(), ACL_FLOAT, ACL_SPARSE_ORDER_COL);
        DnMatManager matC0 = DnMatManager::create(
            m_, 0, ldb_, dC_->get(), ACL_FLOAT, ACL_SPARSE_ORDER_COL);
        auto ret = CallBufferSize(handle_->get(), ACL_SPARSE_OP_NON_TRANSPOSE, &alpha_,
                                  d->matA.cget(), matB0.cget(), matC0.get(),
                                  ACL_FLOAT, d->spsmDescr.get());
        // n=0 is invalid input: aclsparseCreateDnMat rejects cols<=0
        // (INVALID_VALUE at descriptor creation). Assert the single expected
        // outcome instead of the previous dual-value (INVALID_VALUE|SUCCESS).
        EXPECT_EQ(ret, ACL_SPARSE_STATUS_INVALID_VALUE);
    } catch (const std::runtime_error&) {
        SUCCEED() << "0-width dense matrix rejected at creation layer";
    }
}

// L2_X13: A non-square (rows != cols) -> ACL_SPARSE_STATUS_INVALID_VALUE
TEST_F(SpsmExceptionTest, ANonSquare) {
    auto nonSquareA = makeSparseCsr(m_, m_ / 2, 0.8, 100);
    auto dRowPtr = DeviceBuffer::copyFrom(nonSquareA.rowOffsets.data(),
                                           (m_ + 1) * sizeof(int32_t));
    auto dColInd = DeviceBuffer::copyFrom(nonSquareA.colIndices.data(),
                                           nonSquareA.nnz * sizeof(int32_t));
    auto dCsrVals = DeviceBuffer::copyFrom(nonSquareA.values.data(),
                                             nonSquareA.nnz * sizeof(float));
    SpMatManager matA = SpMatManager::createCsr(
        m_, m_ / 2, static_cast<int64_t>(nonSquareA.nnz),
        dRowPtr.get(), dColInd.get(), dCsrVals.get(),
        ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_32I,
        ACL_SPARSE_INDEX_BASE_ZERO, ACL_FLOAT);
    SetTriangularAttributes(matA.get(), ACL_SPARSE_DIAG_TYPE_UNIT);
    SpsmDescrGuard spsmDescr;
    // MakeValidDescrs provides valid matB/matC; its d->matA (square) is unused —
    // this case tests the non-square matA above, not d->matA.
    auto d = MakeValidDescrs();
    EXPECT_EQ(CallBufferSize(handle_->get(), ACL_SPARSE_OP_NON_TRANSPOSE, &alpha_,
                              matA.cget(), d->matB.cget(), d->matC.get(),
                              ACL_FLOAT, spsmDescr.get()),
              ACL_SPARSE_STATUS_INVALID_VALUE);
}

// L2_X14: B rows != m -> ACL_SPARSE_STATUS_INVALID_VALUE
TEST_F(SpsmExceptionTest, BRowsMismatch) {
    auto d = MakeValidDescrs();
    auto dBmismatch = DeviceBuffer::alloc(static_cast<size_t>(ldb_) * n_ * sizeof(float));
    DnMatManager matBbad = DnMatManager::create(
        m_ / 2, n_, ldb_, dBmismatch.get(), ACL_FLOAT, ACL_SPARSE_ORDER_COL);
    EXPECT_EQ(CallBufferSize(handle_->get(), ACL_SPARSE_OP_NON_TRANSPOSE, &alpha_,
                              d->matA.cget(), matBbad.cget(), d->matC.get(),
                              ACL_FLOAT, d->spsmDescr.get()),
              ACL_SPARSE_STATUS_INVALID_VALUE);
}

// L2_X15: null spsmDescr -> ACL_SPARSE_STATUS_INVALID_VALUE
TEST_F(SpsmExceptionTest, NullSpsmDescr) {
    auto d = MakeValidDescrs();
    EXPECT_EQ(CallBufferSize(handle_->get(), ACL_SPARSE_OP_NON_TRANSPOSE, &alpha_,
                              d->matA.cget(), d->matB.cget(), d->matC.get(),
                              ACL_FLOAT, nullptr),
              ACL_SPARSE_STATUS_INVALID_VALUE);
}

// L2_X16: B cols != C cols -> ACL_SPARSE_STATUS_INVALID_VALUE
TEST_F(SpsmExceptionTest, BColsMismatch) {
    auto d = MakeValidDescrs();
    // matC with cols = n_/2 (mismatch with matB.cols = n_)
    auto dCmismatch = DeviceBuffer::alloc(static_cast<size_t>(ldb_) * (n_ / 2) * sizeof(float));
    DnMatManager matCbad = DnMatManager::create(
        m_, n_ / 2, ldb_, dCmismatch.get(), ACL_FLOAT, ACL_SPARSE_ORDER_COL);
    EXPECT_EQ(CallBufferSize(handle_->get(), ACL_SPARSE_OP_NON_TRANSPOSE, &alpha_,
                              d->matA.cget(), d->matB.cget(), matCbad.get(),
                              ACL_FLOAT, d->spsmDescr.get()),
              ACL_SPARSE_STATUS_INVALID_VALUE);
}

// L2_X17: ldb < m -> ACL_SPARSE_STATUS_INVALID_VALUE
// Build matB/matC with ld < rows; descriptor creation may reject or BufferSize rejects.
TEST_F(SpsmExceptionTest, LdbLtRows) {
    auto d = MakeValidDescrs();
    try {
        DnMatManager matBbad = DnMatManager::create(
            m_, n_, m_ / 2, dB_->raw(), ACL_FLOAT, ACL_SPARSE_ORDER_COL);
        DnMatManager matCbad = DnMatManager::create(
            m_, n_, m_ / 2, dC_->get(), ACL_FLOAT, ACL_SPARSE_ORDER_COL);
        auto ret = CallBufferSize(handle_->get(), ACL_SPARSE_OP_NON_TRANSPOSE, &alpha_,
                                  d->matA.cget(), matBbad.cget(), matCbad.get(),
                                  ACL_FLOAT, d->spsmDescr.get());
        // order=COL, ld(8) < rows(16): invalid after the order-aware ld fix
        // (COL requires ld>=rows). Rejected as INVALID_VALUE at descriptor
        // creation (aclsparseCreateDnMat) and/or at BufferSize. This is a
        // genuinely illegal scenario, so assert the single expected outcome.
        EXPECT_EQ(ret, ACL_SPARSE_STATUS_INVALID_VALUE);
    } catch (const std::runtime_error&) {
        SUCCEED() << "ldb < rows rejected at descriptor creation layer";
    }
}

// L2_X18: valueType mismatch (matA.valueType=FP16, computeType=FP32)
//         -> ACL_SPARSE_STATUS_NOT_SUPPORTED
TEST_F(SpsmExceptionTest, ValueTypeMismatch) {
    // Build a CSR with FP16 valueType descriptor (but FP32 data).
    auto dRowPtr = DeviceBuffer::copyFrom(csrA_.rowOffsets.data(), (m_ + 1) * sizeof(int32_t));
    auto dColInd = DeviceBuffer::copyFrom(csrA_.colIndices.data(), csrA_.nnz * sizeof(int32_t));
    // FP16 data: 2 bytes per element
    std::vector<uint16_t> hValsFp16(csrA_.nnz);
    for (int64_t i = 0; i < csrA_.nnz; ++i) {
        hValsFp16[i] = SpsmFp32ToFp16Bits(csrA_.values[i]);
    }
    auto dCsrValsFp16 = DeviceBuffer::copyFrom(hValsFp16.data(), csrA_.nnz * sizeof(uint16_t));
    SpMatManager matA = SpMatManager::createCsr(
        m_, m_, static_cast<int64_t>(csrA_.nnz),
        dRowPtr.get(), dColInd.get(), dCsrValsFp16.get(),
        ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_32I,
        ACL_SPARSE_INDEX_BASE_ZERO, ACL_FLOAT16);
    SetTriangularAttributes(matA.get(), ACL_SPARSE_DIAG_TYPE_UNIT);
    auto d = MakeValidDescrs();
    SpsmDescrGuard spsmDescr;
    EXPECT_EQ(CallBufferSize(handle_->get(), ACL_SPARSE_OP_NON_TRANSPOSE, &alpha_,
                              matA.cget(), d->matB.cget(), d->matC.get(),
                              ACL_FLOAT, spsmDescr.get()),
              ACL_SPARSE_STATUS_NOT_SUPPORTED);
}

// L2_X19: singular matrix (NON_UNIT zero diagonal) -> NOT_SUPPORTED at Analysis/Solve.
// Builds a NON_UNIT CSR with a zero diagonal element and runs the full pipeline.
TEST_F(SpsmExceptionTest, SingularMatrix_NonUnit) {
    // Build a 16x16 NON_UNIT LOWER triangular CSR with one zero diagonal.
    CsrMatrix singA = makeSingularTriangularCsr(m_, true, 0.2, -1.0, 1.0,
                                                 1.0, 2.0, 100);
    auto dRowPtr = DeviceBuffer::copyFrom(singA.rowOffsets.data(), (m_ + 1) * sizeof(int32_t));
    auto dColInd = DeviceBuffer::copyFrom(singA.colIndices.data(), singA.nnz * sizeof(int32_t));
    auto dCsrVals = DeviceBuffer::copyFrom(singA.values.data(), singA.nnz * sizeof(float));
    auto dB = DeviceBuffer::copyFrom(B_.data(), static_cast<size_t>(ldb_) * n_ * sizeof(float));
    auto dC = DeviceBuffer::alloc(static_cast<size_t>(ldb_) * n_ * sizeof(float));

    SpMatManager matA = SpMatManager::createCsr(
        m_, m_, static_cast<int64_t>(singA.nnz),
        dRowPtr.get(), dColInd.get(), dCsrVals.get(),
        ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_32I,
        ACL_SPARSE_INDEX_BASE_ZERO, ACL_FLOAT);
    SetTriangularAttributes(matA.get(), ACL_SPARSE_DIAG_TYPE_NON_UNIT);
    DnMatManager matB = DnMatManager::createConst(
        m_, n_, ldb_, dB.raw(), ACL_FLOAT, ACL_SPARSE_ORDER_COL);
    DnMatManager matC = DnMatManager::create(
        m_, n_, ldb_, dC.get(), ACL_FLOAT, ACL_SPARSE_ORDER_COL);
    SpsmDescrGuard spsmDescr;

    // BufferSize should succeed (no singular check at this stage).
    size_t bufferSize = 0;
    auto bufRet = aclsparseSpSMBufferSize(
        handle_->get(), ACL_SPARSE_OP_NON_TRANSPOSE, ACL_SPARSE_OP_NON_TRANSPOSE,
        &alpha_, matA.cget(), matB.cget(), matC.get(),
        ACL_FLOAT, ACL_SPARSE_SPSM_ALG_DEFAULT, spsmDescr.get(), &bufferSize);
    ASSERT_EQ(bufRet, ACL_SPARSE_STATUS_SUCCESS);
    if (bufferSize == 0) bufferSize = 16;
    auto dBuffer = DeviceBuffer::alloc(bufferSize);

    // Analysis should detect singular and return NOT_SUPPORTED.
    auto analysisRet = aclsparseSpSMAnalysis(
        handle_->get(), ACL_SPARSE_OP_NON_TRANSPOSE, ACL_SPARSE_OP_NON_TRANSPOSE,
        &alpha_, matA.cget(), matB.cget(), matC.get(),
        ACL_FLOAT, ACL_SPARSE_SPSM_ALG_DEFAULT, spsmDescr.get(), dBuffer.get());
    EXPECT_EQ(analysisRet, ACL_SPARSE_STATUS_NOT_SUPPORTED);
}

// L2_X20: unsorted indices accepted -> SUCCESS (positive exception case).
TEST_F(SpsmExceptionTest, UnsortedIndicesAccepted) {
    // Build a UNIT LOWER triangular CSR, then shuffle colInd within rows.
    CsrMatrix csr = makeTriangularCsr(m_, true, 0.2, -1.0, 1.0, 100);
    shuffleRowInternal(csr, 999);
    auto dRowPtr = DeviceBuffer::copyFrom(csr.rowOffsets.data(), (m_ + 1) * sizeof(int32_t));
    auto dColInd = DeviceBuffer::copyFrom(csr.colIndices.data(), csr.nnz * sizeof(int32_t));
    auto dCsrVals = DeviceBuffer::copyFrom(csr.values.data(), csr.nnz * sizeof(float));

    SpMatManager matA = SpMatManager::createCsr(
        m_, m_, static_cast<int64_t>(csr.nnz),
        dRowPtr.get(), dColInd.get(), dCsrVals.get(),
        ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_32I,
        ACL_SPARSE_INDEX_BASE_ZERO, ACL_FLOAT);
    SetTriangularAttributes(matA.get(), ACL_SPARSE_DIAG_TYPE_UNIT);
    auto d = MakeValidDescrs();
    SpsmDescrGuard spsmDescr;
    // BufferSize with unsorted indices should succeed (not rejected).
    auto ret = CallBufferSize(handle_->get(), ACL_SPARSE_OP_NON_TRANSPOSE, &alpha_,
                              matA.cget(), d->matB.cget(), d->matC.get(),
                              ACL_FLOAT, spsmDescr.get());
    EXPECT_EQ(ret, ACL_SPARSE_STATUS_SUCCESS);
}
