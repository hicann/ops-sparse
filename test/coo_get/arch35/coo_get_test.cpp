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
 * @file coo_get_test.cpp
 * @brief GTest + CSV-driven tests for aclsparseCreateCoo / aclsparseCooGet (arch35).
 *
 * CooGet 对标 cuSPARSE cusparseCooGet：CreateCoo -> CooGet 闭环，读回字段
 * 应与 Create 传入一致。CooGet 是纯 Host 侧描述符访问器，不启动 NPU kernel；
 * 测试侧建设备缓冲存入描述符，CooGet 读回字段后断言与 Create 传入一致。
 *
 *   - TEST_P (CooGetTest)        : 10 个 CSV 驱动的成功路径用例（FP16/FP32,
 *                                  idx0/idx1, 方阵/非方阵/不对齐/大 shape）
 *   - TEST_F (CooGetExceptionTest): null descr / 非 COO 格式 / 64I 索引等异常路径
 */

#include "test_common.h"

#include "../coo_get_param.h"
#include "coo_get_npu_wrapper.h"

using namespace sparse_test;

// ============================================================================
// GTest parameterized fixture: CooGetTest
// ============================================================================
class CooGetTest : public testing::TestWithParam<CooGetParam> {
public:
    static void SetUpTestSuite() {
        env_ = std::make_unique<AclEnvScope>();
    }
    static void TearDownTestSuite() {
        env_.reset();
    }

protected:
    inline static std::unique_ptr<AclEnvScope> env_;
    CooGetParam param_;

    void SetUp() override { param_ = GetParam(); }
};

// ============================================================================
// Test body: success-path parameterized test
// ============================================================================
TEST_P(CooGetTest, CooGetRoundTrip) {
    const auto &p = param_;
    std::cout << "==== " << p.case_name << " ==== rows=" << p.rows << " cols=" << p.cols
              << " nnz=" << p.nnz << " dtype=" << p.dtype << " idx_base=" << p.idx_base << "\n";
    ASSERT_EQ(p.expect_result, "SUCCESS");

    // 1. 生成 COO 输入数据（host 侧仅生成输入，不在 CPU 做算子计算）
    std::vector<int32_t> hostRowInd, hostColInd;
    GenerateCooData(p.rows, p.cols, p.nnz, p.idx_base, p.seed,
                     hostRowInd, hostColInd);

    // 2. NPU 执行：CreateCoo -> CooGet accessor 验证
    auto npu = CooGetNpu(env_->stream(), hostRowInd, hostColInd,
                         p.rows, p.cols, p.nnz, p.idx_base, p.isFp16());

    ASSERT_EQ(npu.createRet, ACL_SPARSE_STATUS_SUCCESS)
        << "aclsparseCreateCoo failed";
    ASSERT_EQ(npu.getRet, ACL_SPARSE_STATUS_SUCCESS) << "aclsparseCooGet failed";

    // 3. 断言 Get 读回的标量元数据 == Create 传入
    EXPECT_EQ(npu.rows, p.rows);
    EXPECT_EQ(npu.cols, p.cols);
    EXPECT_EQ(npu.nnz, p.nnz);
    EXPECT_EQ(npu.idxType, ACL_SPARSE_INDEX_32I);
    EXPECT_EQ(npu.idxBase, (p.idx_base == 1) ? ACL_SPARSE_INDEX_BASE_ONE
                                              : ACL_SPARSE_INDEX_BASE_ZERO);
    EXPECT_EQ(npu.valueType, p.isFp16() ? ACL_FLOAT16 : ACL_FLOAT);

    // 4. 断言 Get 读回的设备指针 == Create 传入的设备指针（accessor 语义）
    EXPECT_TRUE(npu.devPtrMatch) << "CooGet device pointer mismatch";

    std::cout << "[" << p.case_name << "] PASSED\n";
    SUCCEED();
}

INSTANTIATE_TEST_SUITE_P(
    CooGetCases,
    CooGetTest,
    testing::ValuesIn(GetCasesFromCsv<CooGetParam>("coo_get_test.csv")),
    [](const testing::TestParamInfo<CooGetParam> &info) {
        return info.param.case_name;
    });

// ============================================================================
// Exception test fixture: CooGetExceptionTest
// ============================================================================
class CooGetExceptionTest : public testing::Test {
public:
    static void SetUpTestSuite() { env_ = std::make_unique<AclEnvScope>(); }
    static void TearDownTestSuite() { env_.reset(); }

protected:
    inline static std::unique_ptr<AclEnvScope> env_;
    std::vector<int32_t> rowInd_{0, 1};
    std::vector<int32_t> colInd_{0, 1};
    std::vector<float> values_{1.0f, 2.0f};

    void SetUp() override {}
};

// E1: nullptr descriptor handle -> ACL_SPARSE_STATUS_INVALID_VALUE
TEST_F(CooGetExceptionTest, NullSpMatDescrHandle) {
    int64_t r, c, n;
    void *ri, *ci, *v;
    aclsparseIndexType_t it;
    aclsparseIndexBase_t ib;
    aclDataType vt;
    EXPECT_EQ(aclsparseCooGet(nullptr, &r, &c, &n, &ri, &ci, &v, &it, &ib, &vt),
              ACL_SPARSE_STATUS_INVALID_VALUE);
}

// E2: nullptr out handle for CreateCoo -> ACL_SPARSE_STATUS_INVALID_VALUE
TEST_F(CooGetExceptionTest, CreateCooNullOutHandle) {
    EXPECT_EQ(aclsparseCreateCoo(nullptr, 2, 2, 2, rowInd_.data(), colInd_.data(),
                                 values_.data(), ACL_SPARSE_INDEX_32I,
                                 ACL_SPARSE_INDEX_BASE_ZERO, ACL_FLOAT),
              ACL_SPARSE_STATUS_INVALID_VALUE);
}

// E3: 64-bit index type is supported (same as cuSPARSE)
TEST_F(CooGetExceptionTest, CreateCooIndex64Supported) {
    aclsparseSpMatDescr_t descr = nullptr;
    ASSERT_EQ(aclsparseCreateCoo(&descr, 2, 2, 2, rowInd_.data(), colInd_.data(),
                                 values_.data(), ACL_SPARSE_INDEX_64I,
                                 ACL_SPARSE_INDEX_BASE_ZERO, ACL_FLOAT),
              ACL_SPARSE_STATUS_SUCCESS);
    EXPECT_NE(descr, nullptr);
    int64_t r = -1, c = -1, n = -1;
    void *ri = nullptr;
    void *ci = nullptr;
    void *v = nullptr;
    aclsparseIndexType_t it = ACL_SPARSE_INDEX_32I;
    aclsparseIndexBase_t ib;
    aclDataType vt;
    EXPECT_EQ(aclsparseCooGet(descr, &r, &c, &n, &ri, &ci, &v, &it, &ib, &vt),
              ACL_SPARSE_STATUS_SUCCESS);
    EXPECT_EQ(it, ACL_SPARSE_INDEX_64I);
    aclsparseDestroySpMat(descr);
}

// E4: negative dims -> ACL_SPARSE_STATUS_INVALID_VALUE
TEST_F(CooGetExceptionTest, CreateCooNegativeDim) {
    aclsparseSpMatDescr_t descr = nullptr;
    EXPECT_EQ(aclsparseCreateCoo(&descr, -1, 2, 2, rowInd_.data(), colInd_.data(),
                                 values_.data(), ACL_SPARSE_INDEX_32I,
                                 ACL_SPARSE_INDEX_BASE_ZERO, ACL_FLOAT),
              ACL_SPARSE_STATUS_INVALID_VALUE);
    if (descr != nullptr) {
        aclsparseDestroySpMat(descr);
    }
}

// E5: CooGet on a non-COO descriptor -> ACL_SPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED
TEST_F(CooGetExceptionTest, CooGetOnNonCooDescr) {
    // 先构造一个 CSR 描述符
    std::vector<int32_t> rowOff{0, 1, 2};
    aclsparseSpMatDescr_t csrDescr = nullptr;
    ASSERT_EQ(aclsparseCreateCsr(&csrDescr, 2, 2, 2, rowOff.data(), colInd_.data(),
                                 values_.data(), ACL_SPARSE_INDEX_32I,
                                 ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_BASE_ZERO,
                                 ACL_FLOAT),
              ACL_SPARSE_STATUS_SUCCESS);
    int64_t r, c, n;
    void *ri, *ci, *v;
    aclsparseIndexType_t it;
    aclsparseIndexBase_t ib;
    aclDataType vt;
    EXPECT_EQ(aclsparseCooGet(csrDescr, &r, &c, &n, &ri, &ci, &v, &it, &ib, &vt),
              ACL_SPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED);
    aclsparseDestroySpMat(csrDescr);
}

// E6: full const variant success (aclsparseConstCooGet round-trip)
TEST_F(CooGetExceptionTest, ConstCooGetRoundTrip) {
    aclsparseConstSpMatDescr_t descr = nullptr;
    ASSERT_EQ(aclsparseCreateConstCoo(&descr, 2, 2, 2, rowInd_.data(), colInd_.data(),
                                      values_.data(), ACL_SPARSE_INDEX_32I,
                                      ACL_SPARSE_INDEX_BASE_ONE, ACL_FLOAT),
              ACL_SPARSE_STATUS_SUCCESS);
    int64_t r = -1, c = -1, n = -1;
    const void *ri = nullptr;
    const void *ci = nullptr;
    const void *v = nullptr;
    aclsparseIndexType_t it;
    aclsparseIndexBase_t ib;
    aclDataType vt;
    EXPECT_EQ(aclsparseConstCooGet(descr, &r, &c, &n, &ri, &ci, &v, &it, &ib, &vt),
              ACL_SPARSE_STATUS_SUCCESS);
    EXPECT_EQ(r, 2);
    EXPECT_EQ(c, 2);
    EXPECT_EQ(n, 2);
    EXPECT_EQ(ib, ACL_SPARSE_INDEX_BASE_ONE);
    EXPECT_EQ(vt, ACL_FLOAT);
    EXPECT_EQ(ri, rowInd_.data());
    EXPECT_EQ(ci, colInd_.data());
    EXPECT_EQ(v, values_.data());
    EXPECT_EQ(it, ACL_SPARSE_INDEX_32I);
    aclsparseDestroySpMat(descr);
}
