/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * SpMVOp GTest entry — parameterized from spmv_op_test.csv (L0 + L1).
 */

#include <gtest/gtest.h>

#include "test_common.h"
#include "cann_ops_sparse.h"

#include "../spmv_op_param.h"
#include "../spmv_op_golden.h"
#include "spmv_op_npu_wrapper.h"

#include <algorithm>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace sparse_test {

// ---------------------------------------------------------------------------
// GTest fixture: shared ACL env + aclsparse handle (one per process)
// ---------------------------------------------------------------------------
class SpMVOpTest : public ::testing::TestWithParam<SpMVOpParam> {
public:
    static void SetUpTestSuite() {
        env_        = std::make_unique<AclEnvScope>();
        spHandle_   = std::make_unique<HandleManager>();
        spHandle_->setStream(env_->stream());
    }

    static void TearDownTestSuite() {
        spHandle_.reset();
        env_.reset();
    }

    static aclrtStream Stream() { return env_->stream(); }
    static HandleManager& SpHandle() { return *spHandle_; }

protected:
    inline static std::unique_ptr<AclEnvScope> env_;
    inline static std::unique_ptr<HandleManager> spHandle_;
};

// ---------------------------------------------------------------------------
// Helper: build CSR matrix for test case
// ---------------------------------------------------------------------------
static CsrMatrix BuildCsr(const SpMVOpParam& p) {
    if (p.pattern == "diag") {
        return makeDiagCsr(p.m, 1.0f);
    }
    if (p.pattern == "empty" || p.m <= 0 || p.k <= 0) {
        return makeEmptyCsr(p.m, p.k);
    }
    // "random" or "full" pattern
    SparseFillGenerator gen(p.seed);
    gen.setSparsity(p.sparsity);
    gen.setEmptyRowProb(p.empty_row_prob);
    gen.setValueRange(-5.0, 5.0);
    CsrMatrix csr = gen.generateCsr(p.m, p.k);

    if (p.unsorted) {
        csr = ShuffleCsrIndices(csr, p.seed);
    }
    return csr;
}

// ---------------------------------------------------------------------------
// TEST_P: parameterized test (called once per CSV row)
// ---------------------------------------------------------------------------
TEST_P(SpMVOpTest, ExecuteSpmvOp) {
    const auto& p = GetParam();
    const std::string caseId = p.caseId();

    std::cout << "\n==== [" << caseId << "] m=" << p.m << " k=" << p.k
              << " alpha=" << p.alpha << " beta=" << p.beta
              << " ptrMode=" << p.pointer_mode << " alg=" << p.alg
              << " pattern=" << p.pattern << " unsrt=" << p.unsorted << " ====\n";

    // Skip expected-failure cases (not exercised in L0; placeholder for L1)
    if (!p.expectSuccess()) {
        std::cout << "[" << caseId << "] SKIPPED (expect_result=" << p.expect_result << ")\n";
        GTEST_SKIP();
    }

    // ------------------------------------------------------------------
    // 1. Build CSR matrix A
    // ------------------------------------------------------------------
    CsrMatrix csr = BuildCsr(p);
    std::cout << "[" << caseId << "] CSR: rows=" << csr.rows
              << " cols=" << csr.cols << " nnz=" << csr.nnz << "\n";

    // ------------------------------------------------------------------
    // 2. Build dense vectors X (length = k) and Y (length = m)
    // ------------------------------------------------------------------
    std::vector<float> x = (p.k > 0) ? makeDenseFloat(p.k, -3.0, 3.0, p.seed + 1)
                                     : std::vector<float>{};
    std::vector<float> y = (p.m > 0) ? makeDenseFloat(p.m, -3.0, 3.0, p.seed + 2)
                                     : std::vector<float>{};

    // ------------------------------------------------------------------
    // 3. Compute CPU golden (Eigen FP64)
    // ------------------------------------------------------------------
    std::vector<float> zGolden = SpMVOpGolden(csr, x, y, p.alpha, p.beta);

    // ------------------------------------------------------------------
    // 4. Run NPU wrapper
    // ------------------------------------------------------------------
    aclsparseIndexType_t rowIdxType = ParseRowIndexType(p.row_offset_type);
    aclsparseSpMVOpAlg_t algEnum    = ParseSpMVOpAlg(p.alg);

    std::vector<float> zNpu;
    try {
        zNpu = SpMVOpNpuWrapper(
            SpHandle(), Stream(),
            csr, x, y,
            p.alpha, p.beta,
            p.pointer_mode, rowIdxType, algEnum,
            p.alias_z_y, caseId);
    } catch (const std::exception& e) {
        ADD_FAILURE() << "[" << caseId << "] NPU wrapper threw: " << e.what();
        return;
    }

    // ------------------------------------------------------------------
    // 5. Verify (skip for zero-length output)
    // ------------------------------------------------------------------
    if (zGolden.empty()) {
        std::cout << "[" << caseId << "] Golden is empty (m<=0), PASS.\n";
        return;
    }

    VerifyConfig cfg;
    cfg.SetMode(PrecisionMode::MERE_MARE)
       .SetMERE(p.mere_threshold)
        .SetMARE(p.mare_multiplier);

    bool pass = Verifier::verifyVector(zNpu, zGolden, cfg, caseId);
    EXPECT_TRUE(pass) << "[" << caseId << "] Verification FAILED";
}

// ---------------------------------------------------------------------------
// Parameterised instantiation (CSV loaded as "spmv_op_test.csv")
// ---------------------------------------------------------------------------
INSTANTIATE_TEST_SUITE_P(
    SpMVOp,
    SpMVOpTest,
    ::testing::ValuesIn(GetCasesFromCsv<SpMVOpParam>("spmv_op_test.csv")),
    [](const ::testing::TestParamInfo<SpMVOpParam>& info) {
        return info.param.caseId();
    }
);

}  // namespace sparse_test

// Note: main() is provided by test/frame/test_main.cpp (shared GTest entry point).
