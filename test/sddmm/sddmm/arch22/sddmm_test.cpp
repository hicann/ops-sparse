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

#include "test_common.h"
#include "sddmm_golden.h"
#include "sddmm_npu_wrapper.h"
#include "sddmm_param.h"

#include <random>

using namespace sparse_test;

static std::vector<double> Fp16ToDoubles(const std::vector<uint16_t>& v) {
    std::vector<double> out(v.size());
    for (size_t i = 0; i < v.size(); i++) {
        out[i] = static_cast<double>(Fp16BitsToFp32(v[i]));
    }
    return out;
}

static std::vector<double> QuantizeDoublesViaFp16(const std::vector<double>& v) {
    return Fp16ToDoubles(DoublesToFp16(v));
}

template <typename T>
static std::vector<T> RepackRowToCol(const std::vector<T>& rowBuf,
                                      int64_t rows, int64_t cols) {
    if (rows <= 0 || cols <= 0) {
        return rowBuf;
    }
    std::vector<T> colBuf(rowBuf.size());
    for (int64_t i = 0; i < rows; ++i) {
        for (int64_t j = 0; j < cols; ++j) {
            colBuf[static_cast<size_t>(j) * static_cast<size_t>(rows) +
                   static_cast<size_t>(i)] =
                rowBuf[static_cast<size_t>(i) * static_cast<size_t>(cols) +
                       static_cast<size_t>(j)];
        }
    }
    return colBuf;
}

struct SddmmInputs {
    SddmmCsr csrC;
    std::vector<double> Xf64;
    std::vector<double> Yf64;
    int64_t nnz = 0;
};

static SddmmInputs GenerateSddmmInputs(const SddmmTestParam& p) {
    SddmmInputs in;
    in.csrC = MakeSddmmSparsity(p.m, p.n, p.sparsity_ratio,
                                p.value_lo, p.value_hi, p.random_seed);
    in.nnz = in.csrC.nnz;

    std::mt19937 rngX(p.random_seed + 1);
    std::mt19937 rngY(p.random_seed + 2);
    std::uniform_real_distribution<double> dist(p.value_lo, p.value_hi);

    in.Xf64.resize(static_cast<size_t>(p.m) * static_cast<size_t>(p.k));
    in.Yf64.resize(static_cast<size_t>(p.n) * static_cast<size_t>(p.k));
    for (size_t i = 0; i < in.Xf64.size(); i++) {
        in.Xf64[i] = dist(rngX);
    }
    for (size_t i = 0; i < in.Yf64.size(); i++) {
        in.Yf64[i] = dist(rngY);
    }
    return in;
}

static std::vector<double> ComputeGoldenExpect(const SddmmTestParam& p,
                                                const SddmmCsr& csrC,
                                                const std::vector<double>& Xf64,
                                                const std::vector<double>& Yf64,
                                                aclDataType dtype,
                                                aclsparseOperation_t opX,
                                                aclsparseOperation_t opY) {
    std::vector<double> goldenX = Xf64;
    std::vector<double> goldenY = Yf64;
    SddmmCsr goldenCsr = csrC;
    double goldenAlpha = p.alpha;
    double goldenBeta = p.beta;
    if (dtype == ACL_FLOAT16) {
        goldenX = QuantizeDoublesViaFp16(Xf64);
        goldenY = QuantizeDoublesViaFp16(Yf64);
        goldenCsr.values = QuantizeDoublesViaFp16(csrC.values);
        goldenAlpha = static_cast<double>(Fp16BitsToFp32(Fp32ToFp16Bits(static_cast<float>(p.alpha))));
        goldenBeta = static_cast<double>(Fp16BitsToFp32(Fp32ToFp16Bits(static_cast<float>(p.beta))));
    }
    return SddmmGolden(p.m, p.n, p.k, goldenX, goldenY, goldenCsr,
                       goldenAlpha, goldenBeta, opX, opY);
}

static SddmmNpuResult RunNpuSddmm(const SddmmTestParam& p, aclrtStream stream,
                                   const SddmmCsr& csrC,
                                   const std::vector<double>& Xf64,
                                   const std::vector<double>& Yf64,
                                   aclDataType dtype, aclDataType computeType,
                                   aclsparseSDDMMAlg_t alg,
                                   aclsparseOperation_t opX, aclsparseOperation_t opY,
                                   aclsparseOrder_t orderX, aclsparseOrder_t orderY,
                                   int64_t nnz) {
    HandleManager handle;
    float alphaF = static_cast<float>(p.alpha);
    float betaF = static_cast<float>(p.beta);
    const bool xTransposed = (opX == ACL_SPARSE_OP_TRANSPOSE);
    const int64_t xRows = xTransposed ? p.k : p.m;
    const int64_t xCols = xTransposed ? p.m : p.k;
    const bool yTransposed = (opY == ACL_SPARSE_OP_TRANSPOSE);
    const int64_t yRows = yTransposed ? p.n : p.k;
    const int64_t yCols = yTransposed ? p.k : p.n;

    SddmmNpuResult npuResult;
    if (dtype == ACL_FLOAT) {
        auto hCInit = DoublesToFp32(csrC.values);
        auto hX = DoublesToFp32(Xf64);
        auto hY = DoublesToFp32(Yf64);
        if (orderX == ACL_SPARSE_ORDER_COL) {
            hX = RepackRowToCol(hX, xRows, xCols);
        }
        if (orderY == ACL_SPARSE_ORDER_COL) {
            hY = RepackRowToCol(hY, yRows, yCols);
        }
        npuResult = SddmmNpu<float>(handle, stream, p.m, p.n, p.k,
                                     opX, opY, alphaF, betaF,
                                     dtype, computeType, alg, orderX, orderY,
                                     csrC.rowOffsets, csrC.colIndices,
                                     hCInit, hX, hY, nnz);
    } else {
        auto hCInit = DoublesToFp16(csrC.values);
        auto hX = DoublesToFp16(Xf64);
        auto hY = DoublesToFp16(Yf64);
        if (orderX == ACL_SPARSE_ORDER_COL) {
            hX = RepackRowToCol(hX, xRows, xCols);
        }
        if (orderY == ACL_SPARSE_ORDER_COL) {
            hY = RepackRowToCol(hY, yRows, yCols);
        }
        npuResult = SddmmNpu<uint16_t>(handle, stream, p.m, p.n, p.k,
                                        opX, opY, alphaF, betaF,
                                        dtype, computeType, alg, orderX, orderY,
                                        csrC.rowOffsets, csrC.colIndices,
                                        hCInit, hX, hY, nnz);
    }
    return npuResult;
}

class SddmmArch22Test : public testing::TestWithParam<SddmmTestParam> {
public:
    static void SetUpTestSuite() {
        env_ = std::make_unique<AclEnvScope>();
    }
    static void TearDownTestSuite() {
        env_.reset();
    }
protected:
    inline static std::unique_ptr<AclEnvScope> env_;
    SddmmTestParam param_;
    aclrtStream stream_ = nullptr;

    void SetUp() override {
        param_ = GetParam();
        stream_ = env_->stream();
    }
};

TEST_P(SddmmArch22Test, SddmmFunctional) {
    const auto& p = param_;

    aclDataType dtype = ParseDtype(p.dtype);
    aclDataType computeType = ParseDtype(p.compute_type);
    aclsparseOperation_t opX = ParseOperation(p.op_x);
    aclsparseOperation_t opY = ParseOperation(p.op_y);
    aclsparseOrder_t orderX = ParseOrder(p.order_x);
    aclsparseOrder_t orderY = ParseOrder(p.order_y);
    aclsparseSDDMMAlg_t alg = ParseSddmmAlg(p.alg);

    SddmmInputs inputs = GenerateSddmmInputs(p);
    int64_t nnz = inputs.nnz;

    std::vector<double> expectFp64 = ComputeGoldenExpect(p, inputs.csrC, inputs.Xf64,
                                                          inputs.Yf64, dtype, opX, opY);

    SddmmNpuResult npuResult = RunNpuSddmm(p, stream_, inputs.csrC, inputs.Xf64,
                                            inputs.Yf64, dtype, computeType, alg,
                                            opX, opY, orderX, orderY, nnz);

    ASSERT_EQ(npuResult.bufferSizeRet, ACL_SPARSE_STATUS_SUCCESS);
    ASSERT_EQ(npuResult.preprocessRet, ACL_SPARSE_STATUS_SUCCESS);
    ASSERT_EQ(npuResult.executeRet, ACL_SPARSE_STATUS_SUCCESS);

    if (nnz > 0) {
        ASSERT_EQ(npuResult.valuesOut.size(), static_cast<size_t>(nnz));
        std::vector<float> goldenFloat(static_cast<size_t>(nnz));
        std::vector<float> npuFloat(static_cast<size_t>(nnz));
        for (int64_t i = 0; i < nnz; i++) {
            float gv = static_cast<float>(expectFp64[static_cast<size_t>(i)]);
            if (dtype == ACL_FLOAT16) {
                if (gv > 65504.0f) { gv = 65504.0f; }
                if (gv < -65504.0f) { gv = -65504.0f; }
            }
            goldenFloat[static_cast<size_t>(i)] = gv;
            npuFloat[static_cast<size_t>(i)] =
                static_cast<float>(npuResult.valuesOut[static_cast<size_t>(i)]);
        }
        VerifyConfig cfg;
        applyMixedTolerance(cfg, dtype, goldenFloat.data(), goldenFloat.size());
        bool pass = Verifier::verifyVector(npuFloat.data(), goldenFloat.data(),
                                           static_cast<size_t>(nnz), 1, cfg, p.case_name);
        EXPECT_TRUE(pass) << "Precision verification failed for " << p.case_name;
    }
}

INSTANTIATE_TEST_SUITE_P(
    SddmmArch22Cases,
    SddmmArch22Test,
    testing::ValuesIn(GetCasesFromCsv<SddmmTestParam>("sddmm_test.csv")),
    [](const testing::TestParamInfo<SddmmTestParam>& info) {
        return info.param.case_name;
    }
);

class SddmmArch22ExceptionTest : public testing::Test {
public:
    static void SetUpTestSuite() {
        env_ = std::make_unique<AclEnvScope>();
    }
    static void TearDownTestSuite() {
        env_.reset();
    }
protected:
    inline static std::unique_ptr<AclEnvScope> env_;
};

TEST_F(SddmmArch22ExceptionTest, E1_NullHandle) {
    size_t bufSize = 0;
    auto ret = aclsparseSDDMMBufferSize(
        nullptr, ACL_SPARSE_OP_NON_TRANSPOSE, ACL_SPARSE_OP_TRANSPOSE,
        nullptr, nullptr, nullptr, nullptr, nullptr,
        ACL_FLOAT, ACL_SPARSE_SDDMM_ALG_DEFAULT, &bufSize);
    EXPECT_NE(ret, ACL_SPARSE_STATUS_SUCCESS);
}

TEST_F(SddmmArch22ExceptionTest, E2_UnsupportedComputeType) {
    HandleManager handle;
    handle.setStream(env_->stream());
    float alpha = 1.0f, beta = 0.0f;
    int64_t m = 4, n = 3, k = 2, nnz = 5;
    std::vector<int32_t> rowOff = {0, 2, 3, 4, 5};
    std::vector<int32_t> colInd = {0, 2, 1, 0, 2};
    std::vector<float> vals(5, 0.0f);
    std::vector<float> hX(m * k, 1.0f);
    std::vector<float> hY(n * k, 1.0f);

    DeviceBuffer dRowOff = DeviceBuffer::copyFrom(rowOff.data(), (m + 1) * sizeof(int32_t));
    DeviceBuffer dColInd = DeviceBuffer::copyFrom(colInd.data(), nnz * sizeof(int32_t));
    DeviceBuffer dVals = DeviceBuffer::copyFrom(vals.data(), nnz * sizeof(float));
    DeviceBuffer dX = DeviceBuffer::copyFrom(hX.data(), m * k * sizeof(float));
    DeviceBuffer dY = DeviceBuffer::copyFrom(hY.data(), n * k * sizeof(float));

    auto matX = DnMatManager::createConst(m, k, k, dX.raw(), ACL_FLOAT, ACL_SPARSE_ORDER_ROW);
    auto matY = DnMatManager::createConst(n, k, k, dY.raw(), ACL_FLOAT, ACL_SPARSE_ORDER_ROW);
    auto matC = SpMatManager::createCsr(m, n, nnz, dRowOff.get(), dColInd.get(), dVals.get(),
                                        ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_32I,
                                        ACL_SPARSE_INDEX_BASE_ZERO, ACL_FLOAT);
    size_t bufSize = 0;
    auto ret = aclsparseSDDMMBufferSize(
        handle.get(), ACL_SPARSE_OP_NON_TRANSPOSE, ACL_SPARSE_OP_TRANSPOSE,
        &alpha, matX.cget(), matY.cget(), &beta, matC.get(),
        ACL_INT32, ACL_SPARSE_SDDMM_ALG_DEFAULT, &bufSize);
    EXPECT_EQ(ret, ACL_SPARSE_STATUS_NOT_SUPPORTED);
}

TEST_F(SddmmArch22ExceptionTest, E3_DimensionMismatch) {
    HandleManager handle;
    handle.setStream(env_->stream());
    float alpha = 1.0f, beta = 0.0f;
    int64_t m = 4, n = 3, k = 2, nnz = 5;
    std::vector<int32_t> rowOff = {0, 2, 3, 4, 5};
    std::vector<int32_t> colInd = {0, 2, 1, 0, 2};
    std::vector<float> vals(5, 0.0f);
    std::vector<float> hX(m * k, 1.0f);
    std::vector<float> hY(n * k, 1.0f);

    DeviceBuffer dRowOff = DeviceBuffer::copyFrom(rowOff.data(), (m + 1) * sizeof(int32_t));
    DeviceBuffer dColInd = DeviceBuffer::copyFrom(colInd.data(), nnz * sizeof(int32_t));
    DeviceBuffer dVals = DeviceBuffer::copyFrom(vals.data(), nnz * sizeof(float));
    DeviceBuffer dX = DeviceBuffer::copyFrom(hX.data(), m * k * sizeof(float));
    DeviceBuffer dY = DeviceBuffer::copyFrom(hY.data(), n * k * sizeof(float));

    auto matX = DnMatManager::createConst(m, k + 1, k + 1, dX.raw(), ACL_FLOAT, ACL_SPARSE_ORDER_ROW);
    auto matY = DnMatManager::createConst(n, k, k, dY.raw(), ACL_FLOAT, ACL_SPARSE_ORDER_ROW);
    auto matC = SpMatManager::createCsr(m, n, nnz, dRowOff.get(), dColInd.get(), dVals.get(),
                                        ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_32I,
                                        ACL_SPARSE_INDEX_BASE_ZERO, ACL_FLOAT);
    size_t bufSize = 0;
    auto ret = aclsparseSDDMMBufferSize(
        handle.get(), ACL_SPARSE_OP_NON_TRANSPOSE, ACL_SPARSE_OP_TRANSPOSE,
        &alpha, matX.cget(), matY.cget(), &beta, matC.get(),
        ACL_FLOAT, ACL_SPARSE_SDDMM_ALG_DEFAULT, &bufSize);
    EXPECT_EQ(ret, ACL_SPARSE_STATUS_INVALID_VALUE);
}
