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
 * @file gebsr2gebsc_test.cpp
 * @brief GTest + CSV-driven test cases for aclsparseSgebsr2gebsc.
 *
 * Tests GEBSR to GEBSC sparse matrix format conversion (block-level transpose).
 *
 * Test structure:
 *   - TEST_P (Gebsr2GebscTest)          : parameterized success-path tests from CSV
 *   - TEST_F (Gebsr2GebscExceptionTest) : invalid-param error tests
 *
 * Test parameters are loaded from gebsr2gebsc_test.csv (copied to build dir by
 * CMake). Entry point is shared via test/frame/test_main.cpp.
 */

#include <gtest/gtest.h>
#include <memory>
#include <random>
#include <vector>

#include "test_common.h"
#include "descriptor_manager.h"
#include "types.h"
#include "gebsr2gebsc_param.h"
#include "gebsr2gebsc_golden.h"
#include "gebsr2gebsc_npu_wrapper.h"

using namespace sparse_test;

namespace {

aclsparseDirection_t ParseDir(const std::string& s) {
    return (s == "COLUMN") ? ACL_SPARSE_DIRECTION_COLUMN : ACL_SPARSE_DIRECTION_ROW;
}

aclsparseAction_t ParseAction(const std::string& s) {
    return (s == "NUMERIC") ? ACL_SPARSE_ACTION_NUMERIC : ACL_SPARSE_ACTION_SYMBOLIC;
}

CsrMatrix MakeBlockCsr(const Gebsr2GebscParam& p) {
    if (p.mb <= 0 || p.nb <= 0) {
        CsrMatrix empty;
        empty.rows = p.mb;
        empty.cols = p.nb;
        empty.rowOffsets.assign(p.mb + 1, 0);
        return empty;
    }
    SparseFillGenerator gen(p.seed);
    gen.setSparsity(p.sparsity);
    gen.setEmptyRowProb(p.empty_row_prob);
    CsrMatrix csr0 = gen.generateCsr(p.mb, p.nb);
    if (p.idx_base == 1) {
        for (size_t i = 0; i < csr0.rowOffsets.size(); i++) {
            csr0.rowOffsets[i] += 1;
        }
        for (size_t i = 0; i < csr0.colIndices.size(); i++) {
            csr0.colIndices[i] += 1;
        }
    }
    return csr0;
}

size_t ValSizeFromType(const std::string& t) {
    if (t == "FP32" || t == "INT32") return 4;
    if (t == "FP16" || t == "BF16") return 2;
    if (t == "INT8") return 1;
    return 4;
}

inline void StoreFloat(uint8_t* dst, float v) {
    union { float f; uint32_t u; } cvt;
    cvt.f = v;
    dst[0] = static_cast<uint8_t>(cvt.u);
    dst[1] = static_cast<uint8_t>(cvt.u >> 8);
    dst[2] = static_cast<uint8_t>(cvt.u >> 16);
    dst[3] = static_cast<uint8_t>(cvt.u >> 24);
}

inline void StoreU16(uint8_t* dst, uint16_t v) {
    dst[0] = static_cast<uint8_t>(v);
    dst[1] = static_cast<uint8_t>(v >> 8);
}

inline void StoreU8(uint8_t* dst, uint8_t v) {
    dst[0] = v;
}

inline uint16_t FloatToHalf(float f) {
    union { float f; uint32_t u; } cvt;
    cvt.f = f;
    uint32_t s = (cvt.u >> 16) & 0x8000;
    int32_t e = ((cvt.u >> 23) & 0xFF) - 127 + 15;
    uint32_t m = cvt.u & 0x7FFFFF;
    if (e <= 0) { return static_cast<uint16_t>(s); }
    if (e >= 31) { return static_cast<uint16_t>(s | 0x7C00); }
    return static_cast<uint16_t>(s | (e << 10) | (m >> 13));
}

inline uint16_t FloatToBf16(float f) {
    union { float f; uint32_t u; } cvt;
    cvt.f = f;
    return static_cast<uint16_t>(cvt.u >> 16);
}

std::vector<uint8_t> MakeBlockValBytes(const CsrMatrix& csr, int blockSizeA,
    size_t valSize, uint32_t seed, const std::string& valType)
{
    std::mt19937 rng(seed + 1);
    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
    size_t totalBytes = static_cast<size_t>(csr.nnz) * blockSizeA * valSize;
    std::vector<uint8_t> buf(totalBytes);

    for (int64_t k = 0; k < csr.nnz; k++) {
        size_t base = static_cast<size_t>(k) * blockSizeA * valSize;
        for (int elem = 0; elem < blockSizeA; elem++) {
            float fv = dist(rng);
            size_t off = base + static_cast<size_t>(elem) * valSize;
            if (valType == "FP32" || valType == "INT32") {
                int32_t iv = static_cast<int32_t>(fv * 100);
                StoreFloat(&buf[off], static_cast<float>(iv));
            } else if (valType == "FP16") {
                StoreU16(&buf[off], FloatToHalf(fv));
            } else if (valType == "BF16") {
                StoreU16(&buf[off], FloatToBf16(fv));
            } else if (valType == "INT8") {
                StoreU8(&buf[off], static_cast<uint8_t>(static_cast<int>(fv * 50)));
            }
        }
    }
    return buf;
}

inline bool VerifyColPtr(const Gebsr2GebscNpuResult& npu, const Gebsr2GebscGoldenResult& golden,
    const std::string& caseId)
{
    if (npu.bscColPtr.size() != golden.bscColPtr.size()) return false;
    for (size_t i = 0; i < npu.bscColPtr.size(); i++) {
        if (npu.bscColPtr[i] != golden.bscColPtr[i]) {
            std::cout << "[" << caseId << "] bscColPtr[" << i << "] mismatch: "
                      << npu.bscColPtr[i] << " vs " << golden.bscColPtr[i] << std::endl;
            return false;
        }
    }
    return true;
}

inline bool VerifyRowInd(const Gebsr2GebscNpuResult& npu, const Gebsr2GebscGoldenResult& golden,
    const std::string& caseId)
{
    if (npu.bscRowInd.size() != golden.bscRowInd.size()) return false;
    for (size_t i = 0; i < npu.bscRowInd.size(); i++) {
        if (npu.bscRowInd[i] != golden.bscRowInd[i]) {
            std::cout << "[" << caseId << "] bscRowInd[" << i << "] mismatch: "
                      << npu.bscRowInd[i] << " vs " << golden.bscRowInd[i] << std::endl;
            return false;
        }
    }
    return true;
}

inline bool VerifyVal(const Gebsr2GebscNpuResult& npu, const Gebsr2GebscGoldenResult& golden,
    const std::string& caseId, bool isNumeric)
{
    if (isNumeric) {
        if (npu.bscVal.size() != golden.bscVal.size()) return false;
        for (size_t i = 0; i < npu.bscVal.size(); i++) {
            if (npu.bscVal[i] != golden.bscVal[i]) {
                std::cout << "[" << caseId << "] bscVal[" << i << "] mismatch: "
                          << static_cast<int>(npu.bscVal[i]) << " vs "
                          << static_cast<int>(golden.bscVal[i]) << std::endl;
                return false;
            }
        }
    } else {
        for (size_t i = 0; i < npu.bscVal.size(); i++) {
            if (npu.bscVal[i] != 0xDE) return false;
        }
    }
    return true;
}

}  // namespace

class Gebsr2GebscAclEnv : public testing::Test {
protected:
    static std::unique_ptr<AclEnvScope> env_;
    static std::unique_ptr<HandleManager> handle_;

    static void SetUpTestSuite() {
        env_ = std::make_unique<AclEnvScope>();
        handle_ = std::make_unique<HandleManager>();
        handle_->setStream(env_->stream());
    }

    static void TearDownTestSuite() {
        handle_.reset();
        env_.reset();
    }
};

std::unique_ptr<AclEnvScope> Gebsr2GebscAclEnv::env_;
std::unique_ptr<HandleManager> Gebsr2GebscAclEnv::handle_;

class Gebsr2GebscTest : public Gebsr2GebscAclEnv,
                        public testing::WithParamInterface<Gebsr2GebscParam> {
};

TEST_P(Gebsr2GebscTest, Gebsr2GebscConversion) {
    auto p = GetParam();
    std::string caseId = p.case_name;

    CsrMatrix csr = MakeBlockCsr(p);
    int nnzb = static_cast<int>(csr.nnz);
    int blockSizeA = p.row_block_dim_a * p.col_block_dim_a;
    int blockSizeC = p.row_block_dim_c * p.col_block_dim_c;
    size_t valSize = ValSizeFromType(p.val_type);

    auto blockValBytes = MakeBlockValBytes(csr, blockSizeA, valSize, p.seed, p.val_type);

    auto golden = Gebsr2GebscGolden(
        csr, blockValBytes,
        p.mb, p.nb,
        p.row_block_dim_a, p.col_block_dim_a,
        p.row_block_dim_c, p.col_block_dim_c,
        p.idx_base,
        (p.dir_a == "COLUMN") ? 1 : 0,
        (p.copy_values == "NUMERIC") ? 1 : 0,
        static_cast<int>(valSize));

    auto npuResult = Gebsr2GebscNpu(
        *handle_, env_->stream(),
        p.val_type,
        p.mb, p.nb, nnzb,
        (nnzb > 0) ? blockValBytes.data() : nullptr,
        csr.rowOffsets.data(),
        (nnzb > 0) ? csr.colIndices.data() : nullptr,
        p.row_block_dim_a, p.col_block_dim_a,
        p.row_block_dim_c, p.col_block_dim_c,
        ParseAction(p.copy_values),
        static_cast<aclsparseIndexBase_t>(p.idx_base),
        ParseDir(p.dir_a));

    ASSERT_EQ(npuResult.bufferSizeRet, ACL_SPARSE_STATUS_SUCCESS)
        << "[" << caseId << "] bufferSize failed";
    ASSERT_EQ(npuResult.computeRet, ACL_SPARSE_STATUS_SUCCESS)
        << "[" << caseId << "] compute failed";

    EXPECT_TRUE(VerifyColPtr(npuResult, golden, caseId))
        << "[" << caseId << "] bscColPtr verification failed";

    if (nnzb > 0) {
        EXPECT_TRUE(VerifyRowInd(npuResult, golden, caseId))
            << "[" << caseId << "] bscRowInd verification failed";
        EXPECT_TRUE(VerifyVal(npuResult, golden, caseId, p.copy_values == "NUMERIC"))
            << "[" << caseId << "] bscVal verification failed";
    }
}

INSTANTIATE_TEST_SUITE_P(
    Gebsr2GebscCsv,
    Gebsr2GebscTest,
    testing::ValuesIn(GetCasesFromCsv<Gebsr2GebscParam>("gebsr2gebsc_test.csv")),
    [](const testing::TestParamInfo<Gebsr2GebscParam>& info) {
        return info.param.case_name;
    });

// ---------------------------------------------------------------------------
// Exception tests
// ---------------------------------------------------------------------------

class Gebsr2GebscExceptionTest : public Gebsr2GebscAclEnv {
};

TEST_F(Gebsr2GebscExceptionTest, NullHandle) {
    size_t bufSize = 0;
    auto ret = aclsparseGebsr2gebsc_bufferSize(
        nullptr, 4, 4, 4, nullptr, nullptr, nullptr, 2, 2,
        ACL_SPARSE_DIRECTION_ROW, ACL_FLOAT, &bufSize);
    EXPECT_EQ(ret, ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR);
}

TEST_F(Gebsr2GebscExceptionTest, NullBufferSizePtr) {
    auto ret = aclsparseGebsr2gebsc_bufferSize(
        handle_->get(), 4, 4, 4, nullptr, nullptr, nullptr, 2, 2,
        ACL_SPARSE_DIRECTION_ROW, ACL_FLOAT, nullptr);
    EXPECT_EQ(ret, ACL_SPARSE_STATUS_INVALID_VALUE);
}

TEST_F(Gebsr2GebscExceptionTest, NegativeMb) {
    size_t bufSize = 0;
    auto ret = aclsparseGebsr2gebsc_bufferSize(
        handle_->get(), -1, 4, 4, nullptr, nullptr, nullptr, 2, 2,
        ACL_SPARSE_DIRECTION_ROW, ACL_FLOAT, &bufSize);
    EXPECT_EQ(ret, ACL_SPARSE_STATUS_INVALID_VALUE);
}

TEST_F(Gebsr2GebscExceptionTest, NegativeNnzb) {
    size_t bufSize = 0;
    auto ret = aclsparseGebsr2gebsc_bufferSize(
        handle_->get(), 4, 4, -1, nullptr, nullptr, nullptr, 2, 2,
        ACL_SPARSE_DIRECTION_ROW, ACL_FLOAT, &bufSize);
    EXPECT_EQ(ret, ACL_SPARSE_STATUS_INVALID_VALUE);
}

TEST_F(Gebsr2GebscExceptionTest, ZeroBlockDim) {
    size_t bufSize = 0;
    auto ret = aclsparseGebsr2gebsc_bufferSize(
        handle_->get(), 4, 4, 4, nullptr, nullptr, nullptr, 0, 2,
        ACL_SPARSE_DIRECTION_ROW, ACL_FLOAT, &bufSize);
    EXPECT_EQ(ret, ACL_SPARSE_STATUS_INVALID_VALUE);
}

TEST_F(Gebsr2GebscExceptionTest, InvalidDirA) {
    size_t bufSize = 0;
    auto ret = aclsparseGebsr2gebsc_bufferSize(
        handle_->get(), 4, 4, 4, nullptr, nullptr, nullptr, 2, 2,
        static_cast<aclsparseDirection_t>(99), ACL_FLOAT, &bufSize);
    EXPECT_EQ(ret, ACL_SPARSE_STATUS_INVALID_VALUE);
}

TEST_F(Gebsr2GebscExceptionTest, UnsupportedBlockDimCombo) {
    int rowPtr[5] = {0, 1, 2, 3, 4};
    int colInd[4] = {0, 1, 2, 3};
    float val[16] = {0};
    int bscColPtr[5] = {0};
    int bscRowInd[4] = {0};
    float bscVal[16] = {0};

    auto ret = aclsparseGebsr2gebsc(
        handle_->get(), 4, 4, 4, val, rowPtr, colInd, 2, 3,
        bscVal, bscColPtr, bscRowInd, 3, 3,
        ACL_SPARSE_ACTION_NUMERIC, ACL_SPARSE_INDEX_BASE_ZERO,
        ACL_SPARSE_DIRECTION_ROW, ACL_FLOAT, nullptr);
    EXPECT_EQ(ret, ACL_SPARSE_STATUS_NOT_SUPPORTED);
}

TEST_F(Gebsr2GebscExceptionTest, NullBscColPtr) {
    int rowPtr[5] = {0, 1, 2, 3, 4};
    int colInd[4] = {0, 1, 2, 3};
    float val[16] = {0};
    int bscRowInd[4] = {0};
    float bscVal[16] = {0};

    auto ret = aclsparseGebsr2gebsc(
        handle_->get(), 4, 4, 4, val, rowPtr, colInd, 2, 2,
        bscVal, nullptr, bscRowInd, 2, 2,
        ACL_SPARSE_ACTION_NUMERIC, ACL_SPARSE_INDEX_BASE_ZERO,
        ACL_SPARSE_DIRECTION_ROW, ACL_FLOAT, nullptr);
    EXPECT_EQ(ret, ACL_SPARSE_STATUS_INVALID_VALUE);
}
