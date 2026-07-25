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
 * @file csr2csc_ex2_test.cpp
 * @brief GTest + CSV-driven test cases for aclsparseCsr2cscEx2 (Legacy API).
 *
 * Tests CSR to CSC sparse matrix format conversion (transpose).
 *
 * Test structure:
 *   - TEST_P (Csr2CscEx2Test)          : parameterized success-path tests from CSV
 *   - TEST_F (Csr2CscEx2ExceptionTest) : invalid-param error tests
 *
 * Test parameters are loaded from csr2csc_ex2_test.csv (copied to build dir by
 * CMake). Entry point is shared via test/frame/test_main.cpp.
 */

#include <cstdint>

#include "test_common.h"
#include "csr2csc_ex2_golden.h"
#include "csr2csc_ex2_npu_wrapper.h"
#include "csr2csc_ex2_param.h"

using namespace sparse_test;

// ============================================================================
// Helper: Float <-> half/bf16 conversion (bit manipulation)
// ============================================================================

// 位重解释 float -> uint32（联合体方式，避免使用 memcpy 类危险函数）
static uint32_t FloatToBits(float f) {
    union {
        float f;
        uint32_t u;
    } cvt;
    cvt.f = f;
    return cvt.u;
}

// 将小端字节序写入字节缓冲区（按位操作，避免 memcpy）
static void StoreBytes(uint8_t* dst, uint32_t bits, size_t numBytes) {
    for (size_t i = 0; i < numBytes; i++) {
        dst[i] = static_cast<uint8_t>((bits >> (i * 8)) & 0xFFu);
    }
}

static uint16_t FloatToHalf(float f) {
    uint32_t x = FloatToBits(f);
    uint32_t sign = (x >> 16) & 0x8000;
    int32_t  exp  = ((x >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = (x >> 13) & 0x3FF;
    if (exp <= 0) return static_cast<uint16_t>(sign);
    if (exp >= 31) return static_cast<uint16_t>(sign | 0x7C00);
    return static_cast<uint16_t>(sign | (exp << 10) | mant);
}

static uint16_t FloatToBf16(float f) {
    return static_cast<uint16_t>(FloatToBits(f) >> 16);
}

// ============================================================================
// Helper: Parse aclDataType from CSV string
// ============================================================================

static aclDataType ParseValType(const std::string& s) {
    if (s == "INT8")  return ACL_INT8;
    if (s == "FP16")  return ACL_FLOAT16;
    if (s == "BF16")  return ACL_BF16;
    if (s == "FP32")  return ACL_FLOAT;
    return ACL_FLOAT;
}

// 解析 CSV expect_result 列为 aclsparseStatus_t，用于断言（支持未来异常用例）
static aclsparseStatus_t ParseExpectResult(const std::string& s) {
    if (s == "SUCCESS")        return ACL_SPARSE_STATUS_SUCCESS;
    if (s == "INVALID_VALUE")  return ACL_SPARSE_STATUS_INVALID_VALUE;
    if (s == "NOT_SUPPORTED")  return ACL_SPARSE_STATUS_NOT_SUPPORTED;
    if (s == "INSUFFICIENT_RESOURCES") return ACL_SPARSE_STATUS_INSUFFICIENT_RESOURCES;
    // 默认按成功处理（向后兼容现有 SUCCESS 用例）
    return ACL_SPARSE_STATUS_SUCCESS;
}

// ============================================================================
// Helper: Prepare CSR data with desired index base
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
// Helper: Convert float values to target type byte buffer
// ============================================================================

static std::vector<uint8_t> ConvertValuesToBytes(const CsrMatrix& csr,
                                                  aclDataType valType) {
    int nnz = static_cast<int>(csr.nnz);
    size_t valSize = AclDataTypeSize(valType);
    if (valSize == 0) {
        return {};
    }
    std::vector<uint8_t> buf(static_cast<size_t>(nnz) * valSize);

    for (int k = 0; k < nnz; k++) {
        float fv = csr.values[k];
        switch (valType) {
            case ACL_INT8: {
                int8_t v = static_cast<int8_t>(fv);
                StoreBytes(&buf[k * valSize], static_cast<uint8_t>(v), sizeof(v));
                break;
            }
            case ACL_FLOAT16: {
                uint16_t v = FloatToHalf(fv);
                StoreBytes(&buf[k * valSize], v, sizeof(v));
                break;
            }
            case ACL_BF16: {
                uint16_t v = FloatToBf16(fv);
                StoreBytes(&buf[k * valSize], v, sizeof(v));
                break;
            }
            case ACL_FLOAT:
            default: {
                StoreBytes(&buf[k * valSize], FloatToBits(fv), sizeof(fv));
                break;
            }
        }
    }
    return buf;
}

// ============================================================================
// Helper: Convert float buffer to typed vector
// ============================================================================

static std::vector<int8_t> FloatToInt8(const std::vector<float>& fv) {
    std::vector<int8_t> out(fv.size());
    for (size_t i = 0; i < fv.size(); i++) out[i] = static_cast<int8_t>(fv[i]);
    return out;
}

static std::vector<uint16_t> FloatToHalfVec(const std::vector<float>& fv) {
    std::vector<uint16_t> out(fv.size());
    for (size_t i = 0; i < fv.size(); i++) out[i] = FloatToHalf(fv[i]);
    return out;
}

static std::vector<uint16_t> FloatToBf16Vec(const std::vector<float>& fv) {
    std::vector<uint16_t> out(fv.size());
    for (size_t i = 0; i < fv.size(); i++) out[i] = FloatToBf16(fv[i]);
    return out;
}

// ============================================================================
// Helpers for TEST_P (split out to keep method body under 50 nbnc lines)
// ============================================================================

static Csr2CscGoldenResult ComputeGolden(
    const CsrMatrix& csr, aclDataType valType, int idxBase)
{
    switch (valType) {
        case ACL_INT8: {
            auto typedVals = FloatToInt8(csr.values);
            return Csr2CscGolden<int8_t>(csr, typedVals, idxBase);
        }
        case ACL_FLOAT16: {
            auto typedVals = FloatToHalfVec(csr.values);
            return Csr2CscGolden<uint16_t>(csr, typedVals, idxBase);
        }
        case ACL_BF16: {
            auto typedVals = FloatToBf16Vec(csr.values);
            return Csr2CscGolden<uint16_t>(csr, typedVals, idxBase);
        }
        case ACL_FLOAT:
        default: {
            return Csr2CscGolden<float>(csr, csr.values, idxBase);
        }
    }
}

static void VerifyCsr2CscColPtr(const std::vector<int32_t> &actual,
                                const std::vector<int32_t> &expected,
                                const std::string &caseId)
{
    VerifyConfig cfg;
    cfg.SetMode(PrecisionMode::EXACT);
    bool ok = Verifier::verifyVector(
        Verifier::toFloat<int32_t>(actual),
        Verifier::toFloat<int32_t>(expected),
        cfg, caseId + "_cscColPtr");
    EXPECT_TRUE(ok);
}

static void VerifyCsr2CscRowInd(const std::vector<int32_t> &actual,
                                const std::vector<int32_t> &expected,
                                const std::string &caseId)
{
    VerifyConfig cfg;
    cfg.SetMode(PrecisionMode::EXACT);
    bool ok = Verifier::verifyVector(
        Verifier::toFloat<int32_t>(actual),
        Verifier::toFloat<int32_t>(expected),
        cfg, caseId + "_cscRowInd");
    EXPECT_TRUE(ok);
}

// 位重解释 uint32 -> float（联合体方式，与 FloatToBits 互逆）
static float BitsToFloat(uint32_t u) {
    union {
        uint32_t u;
        float f;
    } cvt;
    cvt.u = u;
    return cvt.f;
}

// 从小端字节序缓冲区读取位模式（按位操作，避免 memcpy）
static uint32_t LoadBytes(const uint8_t* src, size_t numBytes) {
    uint32_t bits = 0;
    for (size_t i = 0; i < numBytes; i++) {
        bits |= static_cast<uint32_t>(src[i]) << (i * 8);
    }
    return bits;
}

static float HalfToFloat(uint16_t h) {
    uint32_t sign = (static_cast<uint32_t>(h) & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1Fu;
    uint32_t mant = h & 0x3FFu;
    uint32_t bits;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {
            // half 非规格化数：左移归一化到隐含 1 位
            int32_t shifts = 0;
            while ((mant & 0x400u) == 0) {
                mant <<= 1;
                shifts++;
            }
            mant &= 0x3FFu;
            bits = sign | (static_cast<uint32_t>(113 - shifts) << 23) | (mant << 13);
        }
    } else if (exp == 0x1Fu) {
        bits = sign | 0x7F800000u | (mant << 13);  // Inf / NaN
    } else {
        bits = sign | ((exp - 15 + 127) << 23) | (mant << 13);
    }
    return BitsToFloat(bits);
}

static float Bf16ToFloat(uint16_t b) {
    return BitsToFloat(static_cast<uint32_t>(b) << 16);
}

// ============================================================================
// Helper: Reinterpret target-type byte buffer as float vector
// ============================================================================
// dtype -> float 转换对非 NaN 值为单射，配合 PrecisionMode::EXACT 与原字节级
// 比对等价。

static std::vector<float> BytesToFloatVec(const std::vector<uint8_t>& bytes,
                                           aclDataType valType) {
    size_t valSize = AclDataTypeSize(valType);
    if (valSize == 0) {
        return {};
    }
    size_t count = bytes.size() / valSize;
    std::vector<float> out(count);
    for (size_t i = 0; i < count; i++) {
        const uint8_t* src = &bytes[i * valSize];
        switch (valType) {
            case ACL_INT8:
                out[i] = static_cast<float>(static_cast<int8_t>(src[0]));
                break;
            case ACL_FLOAT16:
                out[i] = HalfToFloat(static_cast<uint16_t>(LoadBytes(src, sizeof(uint16_t))));
                break;
            case ACL_BF16:
                out[i] = Bf16ToFloat(static_cast<uint16_t>(LoadBytes(src, sizeof(uint16_t))));
                break;
            case ACL_FLOAT:
            default:
                out[i] = BitsToFloat(LoadBytes(src, sizeof(float)));
                break;
        }
    }
    return out;
}

static void VerifyCsr2CscVal(const Csr2CscNpuResult& npuResult,
                              const Csr2CscGoldenResult& golden,
                              const std::string& caseId,
                              aclsparseAction_t copyValues, int nnz,
                              aclDataType valType)
{
    if (copyValues != ACL_SPARSE_ACTION_NUMERIC) {
        for (size_t i = 0; i < npuResult.cscVal.size(); i++) {
            EXPECT_EQ(npuResult.cscVal[i], static_cast<uint8_t>(0xDE))
                << "SYMBOLIC mode modified cscVal at byte " << i;
        }
        return;
    }
    if (nnz == 0) return;
    VerifyConfig valCfg;
    valCfg.SetMode(PrecisionMode::EXACT);
    bool ok = Verifier::verifyVector(
        BytesToFloatVec(npuResult.cscVal, valType),
        BytesToFloatVec(golden.cscVal, valType),
        valCfg, caseId + "_cscVal");
    EXPECT_TRUE(ok);
}

// ============================================================================
// GTest parameterized fixture: Csr2CscEx2Test
// ============================================================================

class Csr2CscEx2Test : public testing::TestWithParam<Csr2CscEx2Param> {
public:
    static void SetUpTestSuite() {
        env_ = std::make_unique<AclEnvScope>();
    }

    static void TearDownTestSuite() {
        env_.reset();
    }

protected:
    inline static std::unique_ptr<AclEnvScope> env_;
    Csr2CscEx2Param param_;
    aclrtStream stream_ = nullptr;

    void SetUp() override {
        param_ = GetParam();
        stream_ = env_->stream();
    }
};

// ============================================================================
// TEST_P: Success-path parameterized test
// ============================================================================

TEST_P(Csr2CscEx2Test, Csr2CscEx2Success) {
    const auto& p = param_;

    std::cout << "==== " << p.case_name
              << " ==== m=" << p.m << " n=" << p.n
              << " valType=" << p.val_type
              << " copyValues=" << p.copy_values
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

    int nnz = static_cast<int>(csr0.nnz);
    std::cout << "  nnz=" << nnz << "\n";

    // 2. Apply index base
    CsrMatrix csr = PrepareCsr(csr0, p.idx_base);

    // 3. Parse API parameters
    aclDataType valType = ParseValType(p.val_type);
    aclsparseAction_t copyValues = (p.copy_values == "SYMBOLIC")
        ? ACL_SPARSE_ACTION_SYMBOLIC : ACL_SPARSE_ACTION_NUMERIC;
    aclsparseIndexBase_t idxBase = (p.idx_base == 1)
        ? ACL_SPARSE_INDEX_BASE_ONE : ACL_SPARSE_INDEX_BASE_ZERO;

    // 4. Convert values to target type byte buffer
    std::vector<uint8_t> csrValBytes = ConvertValuesToBytes(csr, valType);

    // 5. Compute golden reference (per-type dispatch)
    Csr2CscGoldenResult golden = ComputeGolden(csr, valType, p.idx_base);

    // 6. Call NPU
    HandleManager handle;
    handle.setStream(stream_);

    auto npuResult = Csr2CscNpu(
        handle, stream_, p.m, p.n, nnz,
        csrValBytes.data(),
        csr.rowOffsets.data(),
        csr.colIndices.data(),
        valType, copyValues, idxBase);

    // 7. Verify API return codes (使用 CSV expect_result 列断言，支持未来异常用例)
    aclsparseStatus_t expectedStatus = ParseExpectResult(p.expect_result);
    ASSERT_EQ(npuResult.bufferSizeRet, expectedStatus)
        << "[" << p.case_name << "] bufferSize failed";
    ASSERT_EQ(npuResult.computeRet, expectedStatus)
        << "[" << p.case_name << "] compute failed";

    // 8-10. Verify outputs
    VerifyCsr2CscColPtr(npuResult.cscColPtr, golden.cscColPtr, p.case_name);
    VerifyCsr2CscRowInd(npuResult.cscRowInd, golden.cscRowInd, p.case_name);
    VerifyCsr2CscVal(npuResult, golden, p.case_name, copyValues, nnz, valType);

    std::cout << "[" << p.case_name << "] PASSED (nnz=" << nnz << ")\n";
}

// ============================================================================
// Parameterized test instantiation from CSV
// ============================================================================

INSTANTIATE_TEST_SUITE_P(
    Csr2CscEx2Cases,
    Csr2CscEx2Test,
    testing::ValuesIn(GetCasesFromCsv<Csr2CscEx2Param>("csr2csc_ex2_test.csv")),
    [](const testing::TestParamInfo<Csr2CscEx2Param>& info) {
        return info.param.case_name;
    }
);

// ============================================================================
// Exception test fixture: Csr2CscEx2ExceptionTest
// ============================================================================

class Csr2CscEx2ExceptionTest : public testing::Test {
public:
    static void SetUpTestSuite() {
        env_ = std::make_unique<AclEnvScope>();
    }

    static void TearDownTestSuite() {
        env_.reset();
    }

protected:
    inline static std::unique_ptr<AclEnvScope> env_;

    void SetUp() override {
        stream_ = env_->stream();
        handle_ = std::make_unique<HandleManager>();
        handle_->setStream(stream_);

        // Prepare a small 4x4 CSR matrix for valid parameter defaults
        csr0_ = makeSparseCsr(4, 4, 0.5, 42);
        nnz_ = static_cast<int>(csr0_.nnz);

        // Prepare dummy host buffers for parameter validation tests
        dummyRowPtr_.assign(5, 0);
        dummyColInd_.assign(std::max(nnz_, 1), 0);
        dummyVal_.assign(std::max(nnz_, 1) * sizeof(float), 0);
        dummyOutPtr_.assign(5, 0);
        dummyOutInd_.assign(std::max(nnz_, 1), 0);
        dummyOutVal_.assign(std::max(nnz_, 1) * sizeof(float), 0);
    }

    void TearDown() override {
        handle_.reset();
    }

    aclsparseStatus_t CallBufferSize(
        aclsparseHandle_t h,
        int m, int n, int nnz,
        const void* csrVal, const int* csrRowPtr, const int* csrColInd,
        void* cscVal, int* cscColPtr, int* cscRowInd,
        aclDataType valType, aclsparseAction_t copyValues,
        aclsparseIndexBase_t idxBase, aclsparseCsr2CscAlg_t alg,
        size_t* bufSize)
    {
        return aclsparseCsr2cscEx2_bufferSize(
            h, m, n, nnz, csrVal, csrRowPtr, csrColInd,
            cscVal, cscColPtr, cscRowInd,
            valType, copyValues, idxBase, alg, bufSize);
    }

    aclsparseStatus_t CallCompute(
        aclsparseHandle_t h,
        int m, int n, int nnz,
        const void* csrVal, const int* csrRowPtr, const int* csrColInd,
        void* cscVal, int* cscColPtr, int* cscRowInd,
        aclDataType valType, aclsparseAction_t copyValues,
        aclsparseIndexBase_t idxBase)
    {
        return aclsparseCsr2cscEx2(
            h, m, n, nnz, csrVal, csrRowPtr, csrColInd,
            cscVal, cscColPtr, cscRowInd,
            valType, copyValues, idxBase,
            ACL_SPARSE_CSR2CSC_ALG_DEFAULT, nullptr);
    }

    aclrtStream stream_ = nullptr;
    std::unique_ptr<HandleManager> handle_;
    CsrMatrix csr0_;
    int nnz_ = 0;
    std::vector<int32_t> dummyRowPtr_;
    std::vector<int32_t> dummyColInd_;
    std::vector<uint8_t> dummyVal_;
    std::vector<int32_t> dummyOutPtr_;
    std::vector<int32_t> dummyOutInd_;
    std::vector<uint8_t> dummyOutVal_;
    size_t bufSize_ = 0;
};

// Exception 1: NULL handle -> ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR
TEST_F(Csr2CscEx2ExceptionTest, NullHandle) {
    EXPECT_EQ(CallBufferSize(
        nullptr, 4, 4, nnz_,
        dummyVal_.data(), dummyRowPtr_.data(), dummyColInd_.data(),
        dummyOutVal_.data(), dummyOutPtr_.data(), dummyOutInd_.data(),
        ACL_FLOAT, ACL_SPARSE_ACTION_NUMERIC,
        ACL_SPARSE_INDEX_BASE_ZERO, ACL_SPARSE_CSR2CSC_ALG_DEFAULT,
        &bufSize_),
        ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR);
}

// Exception 2: Invalid valType (unsupported dtype) -> ACL_SPARSE_STATUS_NOT_SUPPORTED
TEST_F(Csr2CscEx2ExceptionTest, InvalidValType) {
    // Use a clearly invalid enum value (not a supported dtype)
    aclDataType invalidType = static_cast<aclDataType>(999);
    EXPECT_EQ(CallBufferSize(
        handle_->get(), 4, 4, nnz_,
        dummyVal_.data(), dummyRowPtr_.data(), dummyColInd_.data(),
        dummyOutVal_.data(), dummyOutPtr_.data(), dummyOutInd_.data(),
        invalidType, ACL_SPARSE_ACTION_NUMERIC,
        ACL_SPARSE_INDEX_BASE_ZERO, ACL_SPARSE_CSR2CSC_ALG_DEFAULT,
        &bufSize_),
        ACL_SPARSE_STATUS_NOT_SUPPORTED);
}

// Exception 3: Invalid copyValues -> ACL_SPARSE_STATUS_INVALID_VALUE
TEST_F(Csr2CscEx2ExceptionTest, InvalidCopyValues) {
    EXPECT_EQ(CallBufferSize(
        handle_->get(), 4, 4, nnz_,
        dummyVal_.data(), dummyRowPtr_.data(), dummyColInd_.data(),
        dummyOutVal_.data(), dummyOutPtr_.data(), dummyOutInd_.data(),
        ACL_FLOAT, static_cast<aclsparseAction_t>(999),
        ACL_SPARSE_INDEX_BASE_ZERO, ACL_SPARSE_CSR2CSC_ALG_DEFAULT,
        &bufSize_),
        ACL_SPARSE_STATUS_INVALID_VALUE);
}

// Exception 4: Invalid idxBase -> ACL_SPARSE_STATUS_INVALID_VALUE
TEST_F(Csr2CscEx2ExceptionTest, InvalidIdxBase) {
    EXPECT_EQ(CallBufferSize(
        handle_->get(), 4, 4, nnz_,
        dummyVal_.data(), dummyRowPtr_.data(), dummyColInd_.data(),
        dummyOutVal_.data(), dummyOutPtr_.data(), dummyOutInd_.data(),
        ACL_FLOAT, ACL_SPARSE_ACTION_NUMERIC,
        static_cast<aclsparseIndexBase_t>(999), ACL_SPARSE_CSR2CSC_ALG_DEFAULT,
        &bufSize_),
        ACL_SPARSE_STATUS_INVALID_VALUE);
}

// Exception 5: m < 0 -> ACL_SPARSE_STATUS_INVALID_VALUE
TEST_F(Csr2CscEx2ExceptionTest, NegativeM) {
    EXPECT_EQ(CallBufferSize(
        handle_->get(), -1, 4, nnz_,
        dummyVal_.data(), dummyRowPtr_.data(), dummyColInd_.data(),
        dummyOutVal_.data(), dummyOutPtr_.data(), dummyOutInd_.data(),
        ACL_FLOAT, ACL_SPARSE_ACTION_NUMERIC,
        ACL_SPARSE_INDEX_BASE_ZERO, ACL_SPARSE_CSR2CSC_ALG_DEFAULT,
        &bufSize_),
        ACL_SPARSE_STATUS_INVALID_VALUE);
}

// Exception 6: n < 0 -> ACL_SPARSE_STATUS_INVALID_VALUE
TEST_F(Csr2CscEx2ExceptionTest, NegativeN) {
    EXPECT_EQ(CallBufferSize(
        handle_->get(), 4, -1, nnz_,
        dummyVal_.data(), dummyRowPtr_.data(), dummyColInd_.data(),
        dummyOutVal_.data(), dummyOutPtr_.data(), dummyOutInd_.data(),
        ACL_FLOAT, ACL_SPARSE_ACTION_NUMERIC,
        ACL_SPARSE_INDEX_BASE_ZERO, ACL_SPARSE_CSR2CSC_ALG_DEFAULT,
        &bufSize_),
        ACL_SPARSE_STATUS_INVALID_VALUE);
}

// Exception 7: nnz < 0 -> ACL_SPARSE_STATUS_INVALID_VALUE
TEST_F(Csr2CscEx2ExceptionTest, NegativeNnz) {
    EXPECT_EQ(CallBufferSize(
        handle_->get(), 4, 4, -1,
        dummyVal_.data(), dummyRowPtr_.data(), dummyColInd_.data(),
        dummyOutVal_.data(), dummyOutPtr_.data(), dummyOutInd_.data(),
        ACL_FLOAT, ACL_SPARSE_ACTION_NUMERIC,
        ACL_SPARSE_INDEX_BASE_ZERO, ACL_SPARSE_CSR2CSC_ALG_DEFAULT,
        &bufSize_),
        ACL_SPARSE_STATUS_INVALID_VALUE);
}

// Compute Exception 1: NULL handle -> ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR
TEST_F(Csr2CscEx2ExceptionTest, ComputeNullHandle) {
    EXPECT_EQ(CallCompute(
        nullptr, 4, 4, 4, nullptr, nullptr, nullptr,
        nullptr, nullptr, nullptr,
        ACL_FLOAT, ACL_SPARSE_ACTION_NUMERIC, ACL_SPARSE_INDEX_BASE_ZERO),
        ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR);
}

// Compute Exception 2: Invalid valType -> ACL_SPARSE_STATUS_NOT_SUPPORTED
TEST_F(Csr2CscEx2ExceptionTest, ComputeInvalidValType) {
    aclDataType invalidType = static_cast<aclDataType>(999);
    EXPECT_EQ(CallCompute(
        handle_->get(), 4, 4, 4, nullptr, nullptr, nullptr,
        nullptr, nullptr, nullptr,
        invalidType, ACL_SPARSE_ACTION_NUMERIC, ACL_SPARSE_INDEX_BASE_ZERO),
        ACL_SPARSE_STATUS_NOT_SUPPORTED);
}

// Compute Exception 3: NULL cscColPtr -> ACL_SPARSE_STATUS_INVALID_VALUE
TEST_F(Csr2CscEx2ExceptionTest, ComputeNullCscColPtr) {
    EXPECT_EQ(CallCompute(
        handle_->get(), 4, 4, 4, nullptr, nullptr, nullptr,
        nullptr, nullptr, nullptr,
        ACL_FLOAT, ACL_SPARSE_ACTION_NUMERIC, ACL_SPARSE_INDEX_BASE_ZERO),
        ACL_SPARSE_STATUS_INVALID_VALUE);
}

// Compute Exception 4: m < 0 -> ACL_SPARSE_STATUS_INVALID_VALUE
TEST_F(Csr2CscEx2ExceptionTest, ComputeNegativeM) {
    EXPECT_EQ(CallCompute(
        handle_->get(), -1, 4, 4, nullptr, nullptr, nullptr,
        nullptr, nullptr, nullptr,
        ACL_FLOAT, ACL_SPARSE_ACTION_NUMERIC, ACL_SPARSE_INDEX_BASE_ZERO),
        ACL_SPARSE_STATUS_INVALID_VALUE);
}

// Compute Exception 5: m=0 with nnz>0 -> ACL_SPARSE_STATUS_INVALID_VALUE
TEST_F(Csr2CscEx2ExceptionTest, ComputeZeroMWithNnz) {
    EXPECT_EQ(CallCompute(
        handle_->get(), 0, 4, 5, nullptr, nullptr, nullptr,
        nullptr, nullptr, nullptr,
        ACL_FLOAT, ACL_SPARSE_ACTION_NUMERIC, ACL_SPARSE_INDEX_BASE_ZERO),
        ACL_SPARSE_STATUS_INVALID_VALUE);
}

// ALG1 bufferSize returns success
TEST_F(Csr2CscEx2ExceptionTest, Alg1SameAsDefault) {
    EXPECT_EQ(CallBufferSize(
        handle_->get(), 4, 4, 4,
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
        ACL_FLOAT, ACL_SPARSE_ACTION_NUMERIC, ACL_SPARSE_INDEX_BASE_ZERO,
        ACL_SPARSE_CSR2CSC_ALG1, &bufSize_),
        ACL_SPARSE_STATUS_SUCCESS);
    EXPECT_GT(bufSize_, 0u);
}

// 禁止定义 main 函数（由 test/frame/test_main.cpp 提供）
