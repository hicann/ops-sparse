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
 * @file gather_test.cpp
 * @brief GTest + CSV-driven test cases for aclsparseGather.
 *
 * Gather: X.values[i] = Y[X.indices[i] - idxBase] for i = 0 .. nnz-1.
 *
 * Test parameters are loaded from gather_test.csv (copied to build dir by CMake).
 * Verification uses the test framework's Verifier with appropriate precision mode.
 *
 * Entry point is shared via test/frame/test_main.cpp.
 */

#include <gtest/gtest.h>
#include "acl/acl_base_rt.h"
#include "cann_ops_sparse.h"
#include "fill.h"
#include "test_common.h"
#include "gather_golden.h"
#include "gather_npu_wrapper.h"
#include "gather_param.h"

#include <random>
#include <vector>

using namespace sparse_test;

class GatherTest : public testing::TestWithParam<GatherTestParam> {
public:
    static void SetUpTestSuite() { env_ = std::make_unique<AclEnvScope>(); }

    static void TearDownTestSuite() { env_.reset(); }

protected:
    inline static std::unique_ptr<AclEnvScope> env_;
    GatherTestParam param_;
    aclrtStream stream_ = nullptr;

    void SetUp() override
    {
        param_ = GetParam();
        stream_ = env_->stream();
    }
};

TEST_P(GatherTest, Gather)
{
    const auto& p = param_;

    std::cout << "==== " << p.case_name << " ==== vec_size=" << p.vec_size << " nnz=" << p.nnz << "\n";

    std::mt19937 rng(p.seed);
    HandleManager handle;
    handle.setStream(stream_);

    VerifyConfig cfg;
    cfg.SetMode(PrecisionMode::EXACT);
    bool pass = false;
    int match_count = 0;

    // c++17 don't support template lambda
    auto match_case = [&match_count, &p, &rng, &handle, &cfg, &pass](
                          aclDataType aclFloat_t, auto float_v, aclsparseIndexType_t aclIndex_t, auto int_v) -> void {
        using float_t = decltype(float_v);
        using int_t = decltype(int_v);
        if (p.value_type == aclFloat_t && p.idx_type == aclIndex_t) {
            match_count += 1;
            std::uniform_int_distribution<int_t> idxDist(p.idx_base, p.vec_size - 1 + p.idx_base);
            auto YHost = makeDense<float_t>(p.vec_size, -2.0, 2.0, p.seed);
            std::vector<int_t> indices(p.nnz);
            for (auto& i : indices) {
                i = idxDist(rng);
            }
            std::vector<float_t> golden = GatherGolden<float_t, int_t>(YHost, indices, p.idx_base);
            std::vector<float_t> output =
                GatherNpu<float_t, int_t>(handle, YHost, indices, p.value_type, p.idx_type, p.idx_base);
            // Verifier::verifyVector 仅支持 float32 类型
            pass = Verifier::verifyVector(
                std::vector<float>(output.begin(), output.end()), std::vector<float>(golden.begin(), golden.end()), cfg,
                p.case_name);
        }
    };
    // bf16 type don't exist on host
    match_case(ACL_FLOAT, float{}, ACL_SPARSE_INDEX_32I, int32_t{});
    match_case(ACL_FLOAT, float{}, ACL_SPARSE_INDEX_64I, int64_t{});
    match_case(ACL_FLOAT16, _Float16{}, ACL_SPARSE_INDEX_32I, int32_t{});
    match_case(ACL_FLOAT16, _Float16{}, ACL_SPARSE_INDEX_64I, int64_t{});
    match_case(ACL_DOUBLE, double{}, ACL_SPARSE_INDEX_32I, int32_t{});
    match_case(ACL_DOUBLE, double{}, ACL_SPARSE_INDEX_64I, int64_t{});
    ASSERT_EQ(match_count, 1);

    EXPECT_TRUE(pass) << "Gather verification failed";

    if (pass) {
        std::cout << "[" << p.case_name << "] PASSED\n";
    } else {
        std::cout << "[" << p.case_name << "] FAILED\n";
    }
}

INSTANTIATE_TEST_SUITE_P(
    GatherCases, GatherTest, testing::ValuesIn(GetCasesFromCsv<GatherTestParam>("gather_test.csv")),
    [](const testing::TestParamInfo<GatherTestParam>& info) { return info.param.case_name; });
