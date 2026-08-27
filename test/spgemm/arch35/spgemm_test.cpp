/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * SpGEMM GTest entry — parameterized from spgemm_test.csv (L0 + L1).
 */

#include <gtest/gtest.h>

#include "test_common.h"
#include "cann_ops_sparse.h"

#include "../spgemm_param.h"
#include "../spgemm_golden.h"
#include "spgemm_npu_wrapper.h"

#include <algorithm>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace sparse_test {

// ---------------------------------------------------------------------------
// GTest fixture: shared ACL env + aclsparse handle (one per process)
// ---------------------------------------------------------------------------
class SpGEMMTest : public ::testing::TestWithParam<SpGEMMParam> {
public:
    static void SetUpTestSuite() {
        env_      = std::make_unique<AclEnvScope>();
        spHandle_ = std::make_unique<HandleManager>();
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
static CsrMatrix BuildCsr(int64_t rows, int64_t cols, double sparsity,
                          uint32_t seed, const std::string& pattern = "random")
{
    if (pattern == "diag" && rows == cols && rows > 0) {
        return makeDiagCsr(rows, 1.0f);
    }
    if (rows <= 0 || cols <= 0 || sparsity >= 1.0) {
        return makeEmptyCsr(rows, cols);
    }
    SparseFillGenerator gen(seed);
    gen.setSparsity(sparsity);
    gen.setValueRange(-5.0, 5.0);
    return gen.generateCsr(rows, cols);
}

// ---------------------------------------------------------------------------
// Helper: verify CSR structure + values
// ---------------------------------------------------------------------------
static bool VerifyCsrResult(
    const CsrMatrix& got, const CsrMatrix& golden,
    const VerifyConfig& cfg, const std::string& caseId)
{
    // Structure check: rowPtr
    if (got.rowOffsets.size() != golden.rowOffsets.size()) {
        std::cout << "[" << caseId << "] FAIL: rowPtr size mismatch (got="
                  << got.rowOffsets.size() << " expect=" << golden.rowOffsets.size() << ")\n";
        return false;
    }
    bool structMatch = true;
    for (size_t i = 0; i < got.rowOffsets.size(); i++) {
        if (got.rowOffsets[i] != golden.rowOffsets[i]) {
            structMatch = false;
            break;
        }
    }
    if (!structMatch) {
        std::cout << "[" << caseId << "] FAIL: rowPtr mismatch\n";
        std::cout << "  got:    ";
        for (size_t i = 0; i < std::min(got.rowOffsets.size(), static_cast<size_t>(10)); i++)
            std::cout << got.rowOffsets[i] << " ";
        std::cout << "\n  expect: ";
        for (size_t i = 0; i < std::min(golden.rowOffsets.size(), static_cast<size_t>(10)); i++)
            std::cout << golden.rowOffsets[i] << " ";
        std::cout << "\n";
        return false;
    }

    // Structure check: nnz
    if (got.nnz != golden.nnz) {
        std::cout << "[" << caseId << "] FAIL: nnz mismatch (got=" << got.nnz
                  << " expect=" << golden.nnz << ")\n";
        return false;
    }

    // Structure check: colInd
    if (got.colIndices != golden.colIndices) {
        std::cout << "[" << caseId << "] FAIL: colInd mismatch\n";
        return false;
    }

    // Values check
    if (got.nnz == 0) {
        std::cout << "[" << caseId << "] nnz=0, structure PASS\n";
        return true;
    }

    return Verifier::verifyVector(got.values, golden.values, cfg, caseId);
}

// ---------------------------------------------------------------------------
// Helper: quantize CSR values through target dtype round-trip.
// For fp16/bf16 the NPU receives dtype-quantized inputs, so the golden must
// use the same quantized values to avoid penalising input quantization error.
// ---------------------------------------------------------------------------
static CsrMatrix QuantizeCsrValues(const CsrMatrix& csr, aclDataType dtype) {
    if (dtype == ACL_FLOAT || csr.nnz == 0) return csr;
    CsrMatrix out = csr;
    auto bytes = ConvertValues(csr.values, dtype);
    out.values = ConvertToFloat(bytes.data(), csr.nnz, dtype);
    return out;
}

// ---------------------------------------------------------------------------
// TEST_P: parameterized test (called once per CSV row)
// ---------------------------------------------------------------------------
TEST_P(SpGEMMTest, ExecuteSpgemm) {
    const auto& p = GetParam();
    const std::string caseId = p.caseId();

    std::cout << "\n==== [" << caseId << "] m=" << p.m << " k=" << p.k
              << " n=" << p.n << " alpha=" << p.alpha << " beta=" << p.beta
              << " dtype=" << p.dtype << " alg=" << p.alg << " ====\n";

    if (!p.expectSuccess()) {
        std::cout << "[" << caseId << "] SKIPPED (expect_result=" << p.expect_result << ")\n";
        GTEST_SKIP();
    }

    // 1. Build CSR matrices A and B
    CsrMatrix csrA = BuildCsr(p.m, p.k, p.sparsity_a, p.seed);
    CsrMatrix csrB = BuildCsr(p.k, p.n, p.sparsity_b, p.seed + 1000);
    std::cout << "[" << caseId << "] A: rows=" << csrA.rows << " cols=" << csrA.cols
              << " nnz=" << csrA.nnz << " | B: rows=" << csrB.rows << " cols=" << csrB.cols
              << " nnz=" << csrB.nnz << "\n";

    // 2. Compute CPU golden — quantize inputs through dtype for fp16/bf16
    //    so the golden uses the same precision the NPU actually receives.
    CsrMatrix goldenA = QuantizeCsrValues(csrA, p.aclType());
    CsrMatrix goldenB = QuantizeCsrValues(csrB, p.aclType());
    CsrMatrix golden = SpGEMMGolden(goldenA, goldenB, p.alpha, p.beta);
    std::cout << "[" << caseId << "] Golden nnzC=" << golden.nnz << "\n";

    // 3. Run NPU wrapper
    aclsparseSpGEMMAlg_t algEnum = ParseSpGEMMAlg(p.alg);
    CsrMatrix npuResult;
    try {
        npuResult = SpGEMMNpuWrapper(
            SpHandle(), Stream(),
            csrA, csrB,
            p.alpha, p.beta,
            p.aclType(), algEnum, caseId);
    } catch (const std::exception& e) {
        ADD_FAILURE() << "[" << caseId << "] NPU wrapper threw: " << e.what();
        return;
    }

    // 4. Verify
    VerifyConfig cfg;
    cfg.SetMode(PrecisionMode::MERE_MARE)
       .SetMERE(p.mere_threshold)
       .SetMARE(p.mare_multiplier);

    bool pass = VerifyCsrResult(npuResult, golden, cfg, caseId);
    EXPECT_TRUE(pass) << "[" << caseId << "] Verification FAILED";
}

// ---------------------------------------------------------------------------
// Parameterised instantiation (CSV loaded as "spgemm_test.csv")
// ---------------------------------------------------------------------------
INSTANTIATE_TEST_SUITE_P(
    SpGEMM,
    SpGEMMTest,
    ::testing::ValuesIn(GetCasesFromCsv<SpGEMMParam>("spgemm_test.csv")),
    [](const ::testing::TestParamInfo<SpGEMMParam>& info) {
        return info.param.caseId();
    }
);

}  // namespace sparse_test
