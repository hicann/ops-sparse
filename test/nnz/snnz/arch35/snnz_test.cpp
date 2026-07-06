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
 * @file snnz_test.cpp
 * @brief GTest + CSV-driven test cases for aclsparseSnnz (Legacy API).
 *
 * aclsparseSnnz counts non-zero elements per row or per column in a dense
 * column-major float matrix, plus the total non-zero count.
 *
 * Test parameters are loaded from snnz_test.csv at static initialization time.
 * Verification uses the test framework's Verifier with INTEGER precision mode.
 */

#include "sparse_test.h"
#include "fill.h"
#include "verify.h"
#include "descriptor_manager.h"
#include "snnz_golden.h"
#include "snnz_param.h"
#include "snnz_npu_wrapper.h"

#include "acl/acl.h"
#include "cann_ops_sparse.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace sparse_test;

// ============================================================================
// Matrix generation dispatcher
// (Uses fill.h's column-major dense matrix generators)
// ============================================================================

static std::vector<float> GenerateMatrix(const NnzTestParam &p) {
    int lda = (p.m > 0) ? p.m : 1;

    if (p.distribution == "allzero") {
        return makeZeroColMajor(p.m, p.n, lda);
    } else if (p.distribution == "allnonzero") {
        return makeFullColMajor(p.m, p.n, lda, p.seed);
    } else if (p.distribution == "diag") {
        return makeDiagColMajor(p.m, p.n, lda, p.seed);
    } else if (p.distribution == "extreme") {
        return makeExtremelySparseColMajor(p.m, p.n, lda, p.seed);
    } else {
        return makeDenseColMajor(p.m, p.n, lda, p.density, p.seed);
    }
}

// ============================================================================
// Global ACL environment (uses framework's AclEnvScope)
// ============================================================================

class AclTestEnvironment : public testing::Environment {
public:
    void SetUp() override {
        env_ = std::make_unique<AclEnvScope>();
    }

    void TearDown() override {
        env_.reset();
    }

    aclrtStream stream() const { return env_->stream(); }

private:
    std::unique_ptr<AclEnvScope> env_;
};

static AclTestEnvironment *g_acl_env = nullptr;

// ============================================================================
// GTest parameterized fixture
// ============================================================================

class NnzTest : public testing::TestWithParam<NnzTestParam> {
protected:
    void SetUp() override {
        param_ = GetParam();
        stream_ = g_acl_env->stream();
    }

    NnzTestParam param_;
    aclrtStream stream_ = nullptr;
};

// ============================================================================
// Test body
// ============================================================================

TEST_P(NnzTest, BitExact) {
    const auto &p = param_;
    bool dirRow = (p.dir == "ROW");
    aclsparseDirection_t dirA = dirRow ? ACL_SPARSE_DIRECTION_ROW : ACL_SPARSE_DIRECTION_COLUMN;
    int totalUnits = dirRow ? p.m : p.n;
    int lda = (p.m > 0) ? p.m : 1;

    std::cout << "==== " << p.case_name << " ==== m=" << p.m << " n=" << p.n
              << " dir=" << p.dir << " dist=" << p.distribution
              << " density=" << p.density << " lda=" << lda << "\n";

    auto A = GenerateMatrix(p);

    // Edge case: m=0 or n=0 -> skip NPU call, verify golden produces zero
    if (p.m <= 0 || p.n <= 0) {
        std::vector<int32_t> goldenPerUnit;
        int32_t goldenTotal = 0;
        ReferenceNnz(A.data(), p.m, p.n, lda, dirRow, goldenPerUnit, goldenTotal);
        ASSERT_EQ(goldenTotal, 0) << "Golden total should be 0 for empty matrix";
        std::cout << "[" << p.case_name << "] PASSED (edge case, total=0)\n";
        return;
    }

    // 1. Compute golden
    std::vector<int32_t> goldenPerUnit;
    int32_t goldenTotal = 0;
    ReferenceNnz(A.data(), p.m, p.n, lda, dirRow, goldenPerUnit, goldenTotal);

    // 2. Prepare host output buffers
    std::vector<int32_t> hostNnzPerUnit(totalUnits, 0);
    int32_t hostNnzTotal = 0;

    // 3. Handle (RAII from descriptor_manager.h)
    HandleManager handle;
    handle.setStream(stream_);

    // 4. NPU call (wrapper handles device memory, MatDescr, sync, D2H)
    bool useHostPtrMode = (p.pointer_mode == "HOST");
    auto ret = aclsparseSnnz_npu(handle.get(), dirA, p.m, p.n,
                                  A.data(), lda,
                                  hostNnzPerUnit.data(), totalUnits,
                                  &hostNnzTotal, stream_, useHostPtrMode);
    ASSERT_EQ(ret, ACL_SPARSE_STATUS_SUCCESS) << "aclsparseSnnz_npu failed, status=" << ret;

    // 5. Verification using framework's Verifier with INTEGER mode
    VerifyConfig cfg;
    cfg.SetMode(PrecisionMode::INTEGER);

    auto outputFloat = Verifier::toFloat(hostNnzPerUnit);
    auto goldenFloat = Verifier::toFloat(goldenPerUnit);
    bool perUnitPass = Verifier::verifyVector(outputFloat, goldenFloat, cfg, p.case_name);
    EXPECT_TRUE(perUnitPass) << "nnzPerRowColumn verification failed";

    // Verify nnzTotal using Verifier::verifyScalar
    bool totalPass = Verifier::verifyScalar(
        static_cast<float>(hostNnzTotal), static_cast<float>(goldenTotal), cfg, p.case_name + "_total");
    EXPECT_TRUE(totalPass) << "nnzTotal verification failed";

    if (perUnitPass && totalPass) {
        std::cout << "[" << p.case_name << "] PASSED (exact match, total="
                  << goldenTotal << ")\n";
    }
}

// ============================================================================
// Instantiate test suite from CSV
// ============================================================================

static std::vector<NnzTestParam> g_test_cases = LoadTestCasesFromCsv();

INSTANTIATE_TEST_SUITE_P(
    NnzCases,
    NnzTest,
    testing::ValuesIn(g_test_cases),
    [](const testing::TestParamInfo<NnzTestParam> &info) {
        return info.param.case_name;
    }
);

// ============================================================================
// main
// ============================================================================

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    g_acl_env = new AclTestEnvironment();
    testing::AddGlobalTestEnvironment(g_acl_env);
    return RUN_ALL_TESTS();
}
