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
#include "spmma_compress_npu_wrapper.h"
#include "aclsparselt_mat_descriptor_internal.h"
#include "aclsparselt_matmul_descriptor_internal.h"
#include "aclsparselt_matmul_plan_internal.h"
#include <gtest/gtest.h>
#include <limits>

using namespace sparse_test;

static std::vector<CompressParam> LoadCompressCases()
{
    auto cases = GetCasesFromCsv<CompressParam>("spmma_compress_test.csv");
    if (cases.empty())
        throw std::runtime_error("compression CSV missing or empty");
    return cases;
}
class CompressTestEnvironment : public ::testing::Test {
public:
    static void SetUpTestSuite()
    {
        env = std::make_unique<AclEnvScope>();
    }
    static void TearDownTestSuite()
    {
        env.reset();
    }

protected:
    static std::unique_ptr<AclEnvScope> env;
};
std::unique_ptr<AclEnvScope> CompressTestEnvironment::env;

class CompressDeviceTest : public CompressTestEnvironment, public ::testing::WithParamInterface<CompressParam> {
protected:
    void SetUp() override
    {
        ASSERT_NE(env, nullptr) << "ACL suite initialization failed; compression body will not run";
    }
};
TEST_P(CompressDeviceTest, ExactBytesAndBounds)
{
    RunCompressCase(GetParam(), env->stream());
}
INSTANTIATE_TEST_SUITE_P(Csv, CompressDeviceTest, ::testing::ValuesIn(LoadCompressCases()),
    [](const ::testing::TestParamInfo<CompressParam>& info) { return info.param.name; });

static csv_map ValidCompressConfig()
{
    return {{"case_name", "config_validation"}, {"dtype", "FP32"}, {"order", "ROW"}, {"op", "N"}, {"side", "A"},
        {"pattern", "mixed"}, {"rows", "32"}, {"cols", "64"}, {"ld", "64"}, {"stride", "0"}, {"batches", "1"},
        {"level", "0"}, {"default_stream", "0"}, {"workspace_sentinel", "0"}};
}

TEST(CompressConfigTest, RejectsMissingFields)
{
    const auto valid = ValidCompressConfig();
    for (const auto& field : valid) {
        auto row = valid;
        row.erase(field.first);
        CompressParam param;
        SCOPED_TRACE(field.first);
        EXPECT_THROW(param.fillCustom(row), std::invalid_argument);
    }
}

TEST(CompressConfigTest, RejectsUnknownChoices)
{
    const csv_map invalid{{"dtype", "FLOAT32"}, {"order", "COLL"}, {"op", "NN"}, {"side", "C"}, {"pattern", "speical"},
        {"default_stream", "ture"}, {"workspace_sentinel", "2"}};
    for (const auto& field : invalid) {
        auto row = ValidCompressConfig();
        row[field.first] = field.second;
        CompressParam param;
        SCOPED_TRACE(field.first);
        try {
            param.fillCustom(row);
            FAIL() << "invalid configuration accepted";
        } catch (const std::invalid_argument& error) {
            const std::string message = error.what();
            EXPECT_NE(message.find("config_validation"), std::string::npos);
            EXPECT_NE(message.find(field.first), std::string::npos);
        }
    }
}

TEST(CompressConfigTest, RejectsMalformedNumbers)
{
    for (const auto* field : {"rows", "cols", "ld", "stride", "batches", "level"}) {
        for (const auto* value : {"", "32oops", "99999999999999999999999"}) {
            auto row = ValidCompressConfig();
            row[field] = value;
            CompressParam param;
            SCOPED_TRACE(std::string(field) + "=" + value);
            EXPECT_THROW(param.fillCustom(row), std::invalid_argument);
        }
    }
}

TEST(CompressConfigTest, RejectsUnsafeStorage)
{
    const std::array<std::pair<const char*, const char*>, 10> invalid{
        {{"rows", "0"}, {"rows", "-8"}, {"cols", "9"}, {"ld", "56"}, {"stride", "-1"}, {"stride", "8"},
            {"batches", "0"}, {"batches", "2147483648"}, {"level", "-1"}, {"ld", "9223372036854775800"}}};
    for (const auto& field : invalid) {
        auto row = ValidCompressConfig();
        row[field.first] = field.second;
        CompressParam param;
        SCOPED_TRACE(std::string(field.first) + "=" + field.second);
        EXPECT_THROW(param.fillCustom(row), std::invalid_argument);
    }
    auto row = ValidCompressConfig();
    row["batches"] = "2";
    row["stride"] = "9223372036854775800";
    CompressParam param;
    EXPECT_THROW(param.fillCustom(row), std::invalid_argument);
}

TEST(CompressOracleTest, HandWrittenPositionCodes)
{
    const std::array<std::array<int, 3>, 6> expected{
        {{{0, 1, 4}}, {{0, 2, 8}}, {{0, 3, 12}}, {{1, 2, 9}}, {{1, 3, 13}}, {{2, 3, 14}}}};
    for (const auto& item : expected)
        EXPECT_EQ(PositionCode({item[0], item[1]}, 4), item[2]);
    EXPECT_EQ(PositionCode({0}, 2), 4);
    EXPECT_EQ(PositionCode({1}, 2), 14);
}
TEST(CompressOracleTest, HandWrittenBytesAndZeroFill)
{
    CompressParam p;
    p.dtype = "INT8";
    p.rows = 1;
    p.cols = 8;
    p.ld = 8;
    EXPECT_EQ(CompressGolden(p, {0, 0x80, 0, 0x7f, 0, 0, 0, 0}), (std::vector<uint8_t>{0x80, 0x7f, 0, 0, 0x4d}));
    p.dtype = "FP32";
    p.cols = 4;
    p.ld = 4;
    const std::vector<uint8_t> input{0, 0, 0, 0x80, 1, 0, 0x80, 0x3f, 0, 0, 0, 0x80, 0, 0, 0, 0};
    EXPECT_EQ(CompressGolden(p, input), (std::vector<uint8_t>{1, 0, 0x80, 0x3f, 0, 0, 0, 0x80, 0x4e}));
}
TEST(CompressOracleTest, RawBitsAreNotArithmetic)
{
    for (const std::string dtype : {"FP32", "FP16", "BF16", "INT8"}) {
        CompressParam p;
        p.dtype = dtype;
        for (uint32_t bits : SpecialBits(dtype)) {
            std::vector<uint8_t> bytes(p.bytes());
            PutBits(bytes, 0, p.bytes(), bits);
            EXPECT_TRUE(RawNonzero(bytes.data(), p.bytes(), dtype == "INT8"));
        }
        std::vector<uint8_t> zero(p.bytes(), 0);
        EXPECT_FALSE(RawNonzero(zero.data(), p.bytes(), dtype == "INT8"));
        if (dtype != "INT8") {
            zero.back() = 0x80;
            EXPECT_FALSE(RawNonzero(zero.data(), p.bytes(), false));
        }
    }
}
TEST(CompressOracleTest, ByteVerifierNegativeControls)
{
    std::vector<uint8_t> expected{0, 0x80, 0x7f, 0xff, 4, 0x5a};
    EXPECT_TRUE(CompressExact(expected, expected, "negative-control baseline"));
    for (size_t offset : {size_t{0}, size_t{4}, size_t{5}}) {
        auto broken = expected;
        broken[offset] ^= 1;
        EXPECT_FALSE(CompressExact(broken, expected, "intentional byte corruption"));
    }
}

TEST(CompressHostNullTest, HandlesAndPlans)
{
    size_t size = 123;
    size_t workspace = 456;
    aclsparseLtHandle_t empty = nullptr;
    EXPECT_EQ(aclsparseLtSpMMACompressedSize(nullptr, nullptr, &size, &workspace), ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR);
    EXPECT_EQ(aclsparseLtSpMMACompressedSize(&empty, nullptr, &size, &workspace), ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR);
    EXPECT_EQ(aclsparseLtSpMMACompress(nullptr, nullptr, nullptr, nullptr, nullptr, nullptr),
        ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR);
    EXPECT_EQ(aclsparseLtSpMMACompress(&empty, nullptr, nullptr, nullptr, nullptr, nullptr),
        ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR);
    EXPECT_EQ(size, 123);
    EXPECT_EQ(workspace, 456);
}

class CompressValidationTest : public CompressTestEnvironment {
protected:
    CompressParam param;
    std::unique_ptr<CompressPlan> ctx;
    void SetUp() override
    {
        ASSERT_NE(env, nullptr) << "ACL suite initialization failed; validation body will not run";
        ctx = std::make_unique<CompressPlan>(param);
    }
    aclsparseLtMatDescriptor* sparse()
    {
        return ctx->a.value;
    }
    void QueryReject(aclsparseStatus_t status = ACL_SPARSE_STATUS_INVALID_VALUE)
    {
        size_t size = 123;
        size_t workspace = 456;
        EXPECT_EQ(aclsparseLtSpMMACompressedSize(&ctx->handle.value, &ctx->plan.value, &size, &workspace), status);
        EXPECT_EQ(size, 123);
        EXPECT_EQ(workspace, 456);
    }
};
TEST_F(CompressValidationTest, NullPlanAndOutputs)
{
    size_t size = 123;
    size_t workspace = 456;
    aclsparseLtMatmulPlan_t empty = nullptr;
    EXPECT_EQ(aclsparseLtSpMMACompressedSize(&ctx->handle.value, nullptr, &size, &workspace),
        ACL_SPARSE_STATUS_INVALID_VALUE);
    EXPECT_EQ(
        aclsparseLtSpMMACompressedSize(&ctx->handle.value, &empty, &size, &workspace), ACL_SPARSE_STATUS_INVALID_VALUE);
    EXPECT_EQ(aclsparseLtSpMMACompressedSize(&ctx->handle.value, &ctx->plan.value, nullptr, &workspace),
        ACL_SPARSE_STATUS_INVALID_VALUE);
    EXPECT_EQ(aclsparseLtSpMMACompressedSize(&ctx->handle.value, &ctx->plan.value, &size, nullptr),
        ACL_SPARSE_STATUS_INVALID_VALUE);
    EXPECT_EQ(size, 123);
    EXPECT_EQ(workspace, 456);
}
TEST_F(CompressValidationTest, MissingInternalDescriptors)
{
    auto md = ctx->plan.value->matmulDescr;
    ctx->plan.value->matmulDescr = nullptr;
    QueryReject();
    ctx->plan.value->matmulDescr = md;
    auto a = md->matA;
    md->matA = nullptr;
    QueryReject();
    md->matA = a;
    auto b = md->matB;
    md->matB = nullptr;
    QueryReject();
    md->matB = b;
    auto c = md->matC;
    md->matC = nullptr;
    QueryReject();
    md->matC = c;
    auto d = md->matD;
    md->matD = nullptr;
    QueryReject();
    md->matD = d;
}
TEST_F(CompressValidationTest, SparseSideAndEnums)
{
    sparse()->isStructured = false;
    QueryReject();
    sparse()->isStructured = true;
    ctx->b.value->isStructured = true;
    QueryReject();
    ctx->b.value->isStructured = false;
    sparse()->order = static_cast<aclsparseOrder_t>(99);
    QueryReject();
    sparse()->order = ACL_SPARSE_ORDER_ROW;
    sparse()->sparsity = static_cast<aclsparseLtSparsity_t>(99);
    QueryReject();
    sparse()->sparsity = ACL_SPARSE_LT_SPARSITY_50_PERCENT;
    ctx->md.value->opA = static_cast<aclsparseOperation_t>(99);
    QueryReject();
    ctx->md.value->opA = ACL_SPARSE_OP_NON_TRANSPOSE;
    sparse()->alignment = 3;
    QueryReject();
    sparse()->alignment = 16;
    sparse()->valueType = ACL_INT32;
    QueryReject(ACL_SPARSE_STATUS_NOT_SUPPORTED);
}
TEST_F(CompressValidationTest, ShapeStrideAndBatch)
{
    const auto original = *sparse();
    for (int64_t rows : {int64_t{0}, int64_t{-8}, int64_t{9}}) {
        sparse()->rows = rows;
        QueryReject();
        *sparse() = original;
    }
    sparse()->cols = 0;
    QueryReject();
    *sparse() = original;
    for (int64_t ld : {int64_t{0}, int64_t{56}, int64_t{65}}) {
        sparse()->ld = ld;
        QueryReject();
        *sparse() = original;
    }
    sparse()->numBatches = 0;
    QueryReject();
    *sparse() = original;
    sparse()->numBatches = -1;
    QueryReject();
    *sparse() = original;
    sparse()->batchStride = -1;
    QueryReject();
    *sparse() = original;
    sparse()->numBatches = 3;
    sparse()->batchStride = 8;
    QueryReject();
    *sparse() = original;
}
TEST_F(CompressValidationTest, DimensionAndWideLeadingDimensionBoundaries)
{
    const auto original = *sparse();
    for (const auto dtype : {ACL_FLOAT, ACL_FLOAT16, ACL_BF16, ACL_INT8}) {
        const int64_t alignment = dtype == ACL_FLOAT ? 8 : dtype == ACL_INT8 ? 32 : 16;
        const size_t bytes = dtype == ACL_FLOAT ? 4 : dtype == ACL_INT8 ? 1 : 2;
        const int group = dtype == ACL_FLOAT ? 2 : 4;
        const int64_t maximum = std::numeric_limits<int32_t>::max() / alignment * alignment;
        for (bool wide : {false, true}) {
            *sparse() = original;
            sparse()->valueType = dtype;
            sparse()->rows = wide ? alignment : maximum;
            sparse()->cols = wide ? maximum : alignment;
            sparse()->ld = sparse()->cols;
            size_t size = 0;
            size_t workspace = 1;
            EXPECT_EQ(aclsparseLtSpMMACompressedSize(&ctx->handle.value, &ctx->plan.value, &size, &workspace),
                ACL_SPARSE_STATUS_SUCCESS);
            const size_t elements = static_cast<size_t>(maximum) * alignment;
            EXPECT_EQ(size, elements / 2 * bytes + elements / group / 2);
            EXPECT_EQ(workspace, 0);
            if (wide) {
                sparse()->cols += alignment;
                sparse()->ld = sparse()->cols;
            } else
                sparse()->rows += alignment;
            QueryReject();
        }
    }
    *sparse() = original;
    sparse()->ld = int64_t{1} << 33;
    size_t size = 0;
    size_t workspace = 1;
    EXPECT_EQ(aclsparseLtSpMMACompressedSize(&ctx->handle.value, &ctx->plan.value, &size, &workspace),
        ACL_SPARSE_STATUS_SUCCESS);
    EXPECT_EQ(size, param.compressedBytes());
    EXPECT_EQ(workspace, 0);
    *sparse() = original;
}
TEST_F(CompressValidationTest, OverflowWithoutAllocation)
{
    const auto original = *sparse();
    const uint64_t limit = std::numeric_limits<int64_t>::max();
    // 分别检查矩阵跨度中的乘法溢出，以及乘积合法但加上末行宽度 C 后溢出。
    sparse()->ld = std::numeric_limits<int64_t>::max() / 8 * 8;
    QueryReject();
    *sparse() = original;
    sparse()->rows = 8;
    sparse()->ld = static_cast<int64_t>(limit / 7 / 8 * 8);
    QueryReject();
    *sparse() = original;
    // 分别检查 (batches-1)*stride 的乘法溢出，以及再加 matrixSpan 时的加法溢出。
    sparse()->numBatches = 4;
    sparse()->batchStride = std::numeric_limits<int64_t>::max() / 8 * 8;
    QueryReject();
    *sparse() = original;
    sparse()->numBatches = 2;
    sparse()->batchStride = std::numeric_limits<int64_t>::max() - 7;
    QueryReject();
    *sparse() = original;
    // 输入跨度的元素数合法，但乘以 FP32 元素字节数后溢出。
    sparse()->numBatches = 2;
    sparse()->batchStride = int64_t{1} << 61;
    QueryReject();
    *sparse() = original;
    // 广播输入占用较小，输出元素总数仍可能溢出。
    sparse()->rows = 2147483640;
    sparse()->cols = 8;
    sparse()->ld = 8;
    sparse()->numBatches = std::numeric_limits<int32_t>::max();
    QueryReject();
    *sparse() = original;
    // 在输入跨度合法时，检查 values 字节数及 values+metadata 总字节数的溢出。
    sparse()->rows = 2147483640;
    sparse()->cols = 8;
    sparse()->ld = 8;
    const uint64_t elementsPerBatch = uint64_t{2147483640} * 8;
    sparse()->numBatches = static_cast<int32_t>(limit / 2 / elementsPerBatch + 1);
    QueryReject();
    // 用相邻批次数检查输出总字节数的合法上界和溢出边界。
    const uint64_t bytesPerBatch = elementsPerBatch * 2 + elementsPerBatch / 4;
    sparse()->numBatches = static_cast<int32_t>(limit / bytesPerBatch);
    size_t size = 0;
    size_t workspace = 1;
    EXPECT_EQ(aclsparseLtSpMMACompressedSize(&ctx->handle.value, &ctx->plan.value, &size, &workspace),
        ACL_SPARSE_STATUS_SUCCESS);
    EXPECT_EQ(size, bytesPerBatch * sparse()->numBatches);
    EXPECT_EQ(workspace, 0);
    sparse()->numBatches += 1;
    QueryReject();
    *sparse() = original;
}
TEST_F(CompressValidationTest, NullMisalignedOverlappingAndOverflowingPointers)
{
    auto check = [&](const void* in, void* out) {
        EXPECT_EQ(aclsparseLtSpMMACompress(&ctx->handle.value, &ctx->plan.value, in, out, nullptr, env->stream()),
            ACL_SPARSE_STATUS_INVALID_VALUE);
    };
    auto input = reinterpret_cast<void*>(uintptr_t{0x10000});
    auto output = reinterpret_cast<void*>(uintptr_t{0x100000});
    check(nullptr, output);
    check(input, nullptr);
    check(reinterpret_cast<void*>(uintptr_t{0x10001}), output);
    check(input, reinterpret_cast<void*>(uintptr_t{0x100001}));
    check(input, input);
    check(input, reinterpret_cast<void*>(uintptr_t{0x10010}));
    check(reinterpret_cast<void*>(std::numeric_limits<uintptr_t>::max() - 15), output);
    check(input, reinterpret_cast<void*>(std::numeric_limits<uintptr_t>::max() - 15));
}

class CompressPinnedHost {
public:
    void* value = nullptr;
    explicit CompressPinnedHost(size_t bytes)
    {
        CompressAclCheck(aclrtMallocHost(&value, bytes), "MallocHost");
    }
    ~CompressPinnedHost()
    {
        if (value)
            aclrtFreeHost(value);
    }
    CompressPinnedHost(const CompressPinnedHost&) = delete;
    CompressPinnedHost& operator=(const CompressPinnedHost&) = delete;
};
TEST_F(CompressValidationTest, ExplicitStreamDependencyAndNonblockingObservation)
{
    const auto dense = MakeCompressInput(param);
    const auto golden = CompressGolden(param, dense);
    auto initial = dense;
    std::fill(initial.begin(), initial.end(), 0);
    CompressBuffers buffers(param, initial);
    std::copy(dense.begin(), dense.end(), buffers.inputBefore.begin() + COMPRESS_GUARD);
    CompressPinnedHost pinned(dense.size());
    std::copy(dense.begin(), dense.end(), static_cast<uint8_t*>(pinned.value));
    constexpr size_t workBytes = 64 * 1024 * 1024;
    auto source = DeviceBuffer::alloc(workBytes);
    auto destination = DeviceBuffer::alloc(workBytes);
    CompressAclCheck(aclrtMemset(source.get(), workBytes, 0, workBytes), "Memset precursor");
    bool observed = false;
    for (int attempt = 0; attempt < 2 && !observed; ++attempt) {
        CompressEvent start;
        CompressEvent before;
        CompressEvent after;
        CompressAclCheck(aclrtRecordEvent(start.value, env->stream()), "RecordStart");
        const int copies = attempt == 0 ? 128 : 512;
        for (int i = 0; i < copies; ++i)
            CompressAclCheck(aclrtMemcpyAsync(destination.get(), workBytes, source.get(), workBytes,
                ACL_MEMCPY_DEVICE_TO_DEVICE, env->stream()),
                "Precursor D2D");
        CompressAclCheck(aclrtMemcpyAsync(buffers.dense(), dense.size(), pinned.value, dense.size(),
            ACL_MEMCPY_HOST_TO_DEVICE, env->stream()),
            "Dependency H2D");
        CompressAclCheck(aclrtRecordEvent(before.value, env->stream()), "RecordBefore");
        CompressCheck(aclsparseLtSpMMACompress(&ctx->handle.value, &ctx->plan.value, buffers.dense(),
            buffers.compressed(), nullptr, env->stream()),
            "Stream Compress");
        aclrtEventRecordedStatus status = ACL_EVENT_RECORDED_STATUS_COMPLETE;
        CompressAclCheck(aclrtQueryEventStatus(before.value, &status), "QueryBefore");
        observed = status == ACL_EVENT_RECORDED_STATUS_NOT_READY;
        CompressAclCheck(aclrtRecordEvent(after.value, env->stream()), "RecordAfter");
        CompressAclCheck(aclrtSynchronizeStream(env->stream()), "Synchronize dependency");
        float milliseconds = 0;
        CompressAclCheck(aclrtEventElapsedTime(&milliseconds, start.value, before.value), "Precursor time");
        RecordProperty("precursor_ms_attempt" + std::to_string(attempt), std::to_string(milliseconds));
        CheckCompressResult(param, buffers, golden);
    }
    RecordProperty("nonblocking_observed", observed ? "true" : "inconclusive");
    if (!observed)
        GTEST_SKIP() << "Dependency result passed; bounded event observation was inconclusive";
}
