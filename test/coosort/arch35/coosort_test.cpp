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
 * @file coosort_test.cpp
 * @brief GTest tests for aclsparseXcoosort (Legacy API, arch35 / ascend950).
 *
 * Covers aclsparseXcoosort_bufferSizeExt / aclsparseXcoosortByRow /
 * aclsparseXcoosortByColumn.
 *
 * Sorting semantics - DUAL KEY lexicographic stable sort:
 *   ByRow    : primary=row, secondary=col
 *   ByColumn : primary=col, secondary=row
 * P semantics: caller presets P = 0:1:(nnz-1); after sort
 *   sortedRows[i] = origRows[P[i]], sortedCols[i] = origCols[P[i]].
 * cooRowsA / cooColsA are reordered in-place.
 *
 * Coosort is a single-step flow (no CSV / no param / no npu_wrapper).
 * TEST_F fixture style, with a self-contained main().
 *
 * 共 27 个用例：覆盖单核 ByRow/ByColumn、双键稳定性、
 * 随机输入、CUDA 语义锚点、参数异常、nnz=0、INT_MAX workspace 查询、
 * 多核多 run/跨核归并/每核多次 Sort，以及 bufferSizeExt 不带 P 的语义。
 */

#include "sparse_test.h"
#include "fill.h"
#include "verify.h"
#include "descriptor_manager.h"
#include "coosort_golden.h"

#include "acl/acl.h"
#include "cann_ops_sparse.h"
#include "coosort_tiling_data.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

using namespace sparse_test;

// ============================================================================
// Global ACL environment (framework AclEnvScope RAII).
// ============================================================================
class AclTestEnvironment : public testing::Environment {
public:
    void SetUp() override
    {
        env_ = std::make_unique<AclEnvScope>();
    }
    void TearDown() override
    {
        env_.reset();
    }
    aclrtStream stream() const
    {
        return env_->stream();
    }

private:
    std::unique_ptr<AclEnvScope> env_;
};

static AclTestEnvironment *g_acl_env = nullptr;

// ============================================================================
// Test fixture.
// ============================================================================
class CoosortTest : public testing::Test {
public:
    // Exposed to the free helper RunSortAndVerify (test-internal utilities).
    aclsparseHandle_t handle()
    {
        return handle_->get();
    }
    aclrtStream stream() const
    {
        return stream_;
    }

protected:
    void SetUp() override
    {
        stream_ = g_acl_env->stream();
        handle_ = std::make_unique<HandleManager>();
        handle_->setStream(stream_);
    }

    void TearDown() override
    {
        handle_.reset();
    }

    aclrtStream stream_ = nullptr;
    std::unique_ptr<HandleManager> handle_;
};

// ============================================================================
// Helpers
// ============================================================================

// Verify that a vector<int32_t> equals the expected, element-wise EXACT.
::testing::AssertionResult AssertIntVecEq(
    const std::vector<int32_t> &actual, const std::vector<int32_t> &expected, const char *name)
{
    if (actual.size() != expected.size()) {
        return ::testing::AssertionFailure()
               << name << " size mismatch: actual=" << actual.size() << " expected=" << expected.size();
    }
    for (size_t i = 0; i < actual.size(); i++) {
        if (actual[i] != expected[i]) {
            return ::testing::AssertionFailure()
                   << name << "[" << i << "] mismatch: actual=" << actual[i] << " expected=" << expected[i];
        }
    }
    return ::testing::AssertionSuccess();
}

// Build an identity permutation P = 0:1:(nnz-1).
static std::vector<int32_t> IdentityPerm(int nnz)
{
    std::vector<int32_t> p(nnz);
    std::iota(p.begin(), p.end(), 0);
    return p;
}

// Run a full success-path coosort flow and verify against the host golden.
//   sortByRow=true  -> aclsparseXcoosortByRow
//   sortByRow=false -> aclsparseXcoosortByColumn
// rowsHost/colsHost are the ORIGINAL input (copied; not mutated by caller).
::testing::AssertionResult RunSortAndVerify(CoosortTest *t, int m, int n, const std::vector<int32_t> &rowsHost,
    const std::vector<int32_t> &colsHost, bool sortByRow, const std::string &caseId)
{
    const int nnz = static_cast<int>(rowsHost.size());
    if (nnz != static_cast<int>(colsHost.size())) {
        return ::testing::AssertionFailure() << caseId << " rows/cols size mismatch";
    }

    // 1. Host golden from the ORIGINAL input (read-only).
    CoosortGoldenResult gold = coosortGolden(rowsHost, colsHost, sortByRow);

    // 2. Copy inputs to device.
    auto dRows = DeviceBuffer::copyFrom(rowsHost.data(), static_cast<size_t>(nnz) * sizeof(int32_t));
    auto dCols = DeviceBuffer::copyFrom(colsHost.data(), static_cast<size_t>(nnz) * sizeof(int32_t));
    std::vector<int32_t> pHost = IdentityPerm(nnz);
    auto dP = DeviceBuffer::copyFrom(pHost.data(), static_cast<size_t>(nnz) * sizeof(int32_t));

    // 3. Query buffer size.
    size_t bufSize = 0;
    auto retBuf = aclsparseXcoosort_bufferSizeExt(t->handle(),
        m,
        n,
        nnz,
        reinterpret_cast<const int *>(dRows.raw()),
        reinterpret_cast<const int *>(dCols.raw()),
        &bufSize);
    if (retBuf != ACL_SPARSE_STATUS_SUCCESS) {
        return ::testing::AssertionFailure() << caseId << " bufferSizeExt failed: " << retBuf;
    }
    if (nnz > 0 && bufSize == 0) {
        return ::testing::AssertionFailure() << caseId << " bufferSizeExt returned 0 for nnz>0";
    }

    // 4. Allocate workspace + run sort.
    auto dBuf = DeviceBuffer::alloc(bufSize);
    aclsparseStatus_t ret;
    if (sortByRow) {
        ret = aclsparseXcoosortByRow(t->handle(),
            m,
            n,
            nnz,
            reinterpret_cast<int *>(dRows.get()),
            reinterpret_cast<int *>(dCols.get()),
            reinterpret_cast<int *>(dP.get()),
            dBuf.get());
    } else {
        ret = aclsparseXcoosortByColumn(t->handle(),
            m,
            n,
            nnz,
            reinterpret_cast<int *>(dRows.get()),
            reinterpret_cast<int *>(dCols.get()),
            reinterpret_cast<int *>(dP.get()),
            dBuf.get());
    }
    if (ret != ACL_SPARSE_STATUS_SUCCESS) {
        return ::testing::AssertionFailure() << caseId << " sort failed: " << ret;
    }

    // 5. Synchronize (kernel is async) + D2H.
    auto syncRet = aclrtSynchronizeStream(t->stream());
    if (syncRet != ACL_SUCCESS) {
        return ::testing::AssertionFailure() << caseId << " aclrtSynchronizeStream failed: " << syncRet;
    }

    std::vector<int32_t> outRows(nnz), outCols(nnz), outP(nnz);
    dRows.copyToHost(outRows.data(), static_cast<size_t>(nnz) * sizeof(int32_t));
    dCols.copyToHost(outCols.data(), static_cast<size_t>(nnz) * sizeof(int32_t));
    dP.copyToHost(outP.data(), static_cast<size_t>(nnz) * sizeof(int32_t));

    // 6. EXACT element-wise comparison (sort is bit-deterministic).
    auto r1 = AssertIntVecEq(outRows, gold.rows, "rows");
    if (!r1)
        return ::testing::AssertionFailure() << caseId << " " << r1.message();
    auto r2 = AssertIntVecEq(outCols, gold.cols, "cols");
    if (!r2)
        return ::testing::AssertionFailure() << caseId << " " << r2.message();
    auto r3 = AssertIntVecEq(outP, gold.P, "P");
    if (!r3)
        return ::testing::AssertionFailure() << caseId << " " << r3.message();

    return ::testing::AssertionSuccess();
}

// ============================================================================
// Success-path tests
// ============================================================================

// 1. ByRowBasic - small hardcoded matrix.
TEST_F(CoosortTest, ByRowBasic)
{
    const int m = 4, n = 4, nnz = 6;
    std::vector<int32_t> rows = {2, 0, 3, 0, 1, 2};
    std::vector<int32_t> cols = {1, 3, 2, 1, 0, 3};
    EXPECT_TRUE(RunSortAndVerify(this, m, n, rows, cols, /*sortByRow=*/true, "ByRowBasic"));
}

// 2. ByColumnBasic - small hardcoded matrix.
TEST_F(CoosortTest, ByColumnBasic)
{
    const int m = 4, n = 4, nnz = 6;
    std::vector<int32_t> rows = {2, 0, 3, 0, 1, 2};
    std::vector<int32_t> cols = {1, 3, 2, 1, 0, 3};
    EXPECT_TRUE(RunSortAndVerify(this, m, n, rows, cols, /*sortByRow=*/false, "ByColumnBasic"));
}

// 3. ByRowStableDuplicateKey - duplicate primary key, verify stability on
//    the secondary key (NOT input order; dual-key re-sorts by col).
TEST_F(CoosortTest, ByRowStableDuplicateKey)
{
    const int m = 3, n = 5, nnz = 6;
    // Three row=0 entries with cols out of order: must end up col-ascending.
    std::vector<int32_t> rows = {0, 0, 0, 1, 1, 2};
    std::vector<int32_t> cols = {4, 1, 3, 2, 0, 9};
    EXPECT_TRUE(RunSortAndVerify(this, m, n, rows, cols, /*sortByRow=*/true, "ByRowStableDuplicateKey"));
}

// 4-8. ByRowRandom x5 seed - nnz covers 12/49/103/215/388.
//      Use fill.h makeSparseCoo to generate random COO matrices. The exact
//      nnz is driven by sparsity on a chosen shape; we tune (m,n,sparsity,seed)
//      so the produced nnz lands near each target. The golden recomputes from
//      the actual generated arrays, so the exact nnz need not match precisely.
TEST_F(CoosortTest, ByRowRandom_seed42)
{
    // target nnz ~ 12
    auto coo = makeSparseCoo(/*rows=*/8, /*cols=*/8, /*sparsity=*/0.81, /*seed=*/42);
    ASSERT_GT(coo.nnz, 0);
    EXPECT_TRUE(RunSortAndVerify(this,
        static_cast<int>(coo.rows),
        static_cast<int>(coo.cols),
        coo.rowIndices,
        coo.colIndices,
        /*sortByRow=*/true,
        "ByRowRandom_seed42"));
}

TEST_F(CoosortTest, ByRowRandom_seed43)
{
    // target nnz ~ 49
    auto coo = makeSparseCoo(/*rows=*/16, /*cols=*/16, /*sparsity=*/0.81, /*seed=*/43);
    ASSERT_GT(coo.nnz, 0);
    EXPECT_TRUE(RunSortAndVerify(this,
        static_cast<int>(coo.rows),
        static_cast<int>(coo.cols),
        coo.rowIndices,
        coo.colIndices,
        /*sortByRow=*/true,
        "ByRowRandom_seed43"));
}

TEST_F(CoosortTest, ByRowRandom_seed44)
{
    // target nnz ~ 103
    auto coo = makeSparseCoo(/*rows=*/24, /*cols=*/24, /*sparsity=*/0.82, /*seed=*/44);
    ASSERT_GT(coo.nnz, 0);
    EXPECT_TRUE(RunSortAndVerify(this,
        static_cast<int>(coo.rows),
        static_cast<int>(coo.cols),
        coo.rowIndices,
        coo.colIndices,
        /*sortByRow=*/true,
        "ByRowRandom_seed44"));
}

TEST_F(CoosortTest, ByRowRandom_seed45)
{
    // target nnz ~ 215
    auto coo = makeSparseCoo(/*rows=*/32, /*cols=*/32, /*sparsity=*/0.79, /*seed=*/45);
    ASSERT_GT(coo.nnz, 0);
    EXPECT_TRUE(RunSortAndVerify(this,
        static_cast<int>(coo.rows),
        static_cast<int>(coo.cols),
        coo.rowIndices,
        coo.colIndices,
        /*sortByRow=*/true,
        "ByRowRandom_seed45"));
}

TEST_F(CoosortTest, ByRowRandom_seed46)
{
    // target nnz ~ 388
    auto coo = makeSparseCoo(/*rows=*/44, /*cols=*/44, /*sparsity=*/0.80, /*seed=*/46);
    ASSERT_GT(coo.nnz, 0);
    EXPECT_TRUE(RunSortAndVerify(this,
        static_cast<int>(coo.rows),
        static_cast<int>(coo.cols),
        coo.rowIndices,
        coo.colIndices,
        /*sortByRow=*/true,
        "ByRowRandom_seed46"));
}

// 9. ByColumnRandom
TEST_F(CoosortTest, ByColumnRandom)
{
    auto coo = makeSparseCoo(/*rows=*/20, /*cols=*/20, /*sparsity=*/0.80, /*seed=*/47);
    ASSERT_GT(coo.nnz, 0);
    EXPECT_TRUE(RunSortAndVerify(this,
        static_cast<int>(coo.rows),
        static_cast<int>(coo.cols),
        coo.rowIndices,
        coo.colIndices,
        /*sortByRow=*/false,
        "ByColumnRandom"));
}

// 10. CudaConsistency_DualKey - cuda-measured semantic anchor.
//     Input  rows={1,0,0,1,0} cols={9,7,3,2,5}
//     ByRow   rows={0,0,0,1,1} cols={3,5,7,2,9} P={2,4,1,3,0}
//     This must pass regardless of kernel implementation; it protects the
//     dual-key cuda semantics.
TEST_F(CoosortTest, CudaConsistency_DualKey)
{
    const int m = 2, n = 10, nnz = 5;
    std::vector<int32_t> rows = {1, 0, 0, 1, 0};
    std::vector<int32_t> cols = {9, 7, 3, 2, 5};

    // Expected (cuda-measured).
    std::vector<int32_t> expRows = {0, 0, 0, 1, 1};
    std::vector<int32_t> expCols = {3, 5, 7, 2, 9};
    std::vector<int32_t> expP = {2, 4, 1, 3, 0};

    // Sanity: host golden must agree with the cuda anchor.
    CoosortGoldenResult gold = coosortGolden(rows, cols, /*sortByRow=*/true);
    ASSERT_TRUE(AssertIntVecEq(gold.rows, expRows, "gold.rows"));
    ASSERT_TRUE(AssertIntVecEq(gold.cols, expCols, "gold.cols"));
    ASSERT_TRUE(AssertIntVecEq(gold.P, expP, "gold.P"));

    // Run on device.
    auto dRows = DeviceBuffer::copyFrom(rows.data(), static_cast<size_t>(nnz) * sizeof(int32_t));
    auto dCols = DeviceBuffer::copyFrom(cols.data(), static_cast<size_t>(nnz) * sizeof(int32_t));
    auto pHost = IdentityPerm(nnz);
    auto dP = DeviceBuffer::copyFrom(pHost.data(), static_cast<size_t>(nnz) * sizeof(int32_t));

    size_t bufSize = 0;
    ASSERT_EQ(aclsparseXcoosort_bufferSizeExt(handle(),
                  m,
                  n,
                  nnz,
                  reinterpret_cast<const int *>(dRows.raw()),
                  reinterpret_cast<const int *>(dCols.raw()),
                  &bufSize),
        ACL_SPARSE_STATUS_SUCCESS);
    ASSERT_GT(bufSize, 0u);

    auto dBuf = DeviceBuffer::alloc(bufSize);
    ASSERT_EQ(aclsparseXcoosortByRow(handle(),
                  m,
                  n,
                  nnz,
                  reinterpret_cast<int *>(dRows.get()),
                  reinterpret_cast<int *>(dCols.get()),
                  reinterpret_cast<int *>(dP.get()),
                  dBuf.get()),
        ACL_SPARSE_STATUS_SUCCESS);
    ASSERT_EQ(aclrtSynchronizeStream(stream()), ACL_SUCCESS);

    std::vector<int32_t> outRows(nnz), outCols(nnz), outP(nnz);
    dRows.copyToHost(outRows.data(), static_cast<size_t>(nnz) * sizeof(int32_t));
    dCols.copyToHost(outCols.data(), static_cast<size_t>(nnz) * sizeof(int32_t));
    dP.copyToHost(outP.data(), static_cast<size_t>(nnz) * sizeof(int32_t));

    EXPECT_TRUE(AssertIntVecEq(outRows, expRows, "rows"));
    EXPECT_TRUE(AssertIntVecEq(outCols, expCols, "cols"));
    EXPECT_TRUE(AssertIntVecEq(outP, expP, "P"));
}

// ============================================================================
// Exception-path tests (return-code only, no actual sort performed).
// Parameter validation cases.
// ============================================================================

// 11. NullHandle -> ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR
TEST_F(CoosortTest, NullHandle)
{
    int dummyRows = 0, dummyCols = 0, dummyP = 0;
    size_t bufSize = 0;
    EXPECT_EQ(aclsparseXcoosort_bufferSizeExt(nullptr, 4, 4, 1, &dummyRows, &dummyCols, &bufSize),
        ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR);
    EXPECT_EQ(aclsparseXcoosortByRow(nullptr, 4, 4, 1, &dummyRows, &dummyCols, &dummyP, &bufSize),
        ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR);
    EXPECT_EQ(aclsparseXcoosortByColumn(nullptr, 4, 4, 1, &dummyRows, &dummyCols, &dummyP, &bufSize),
        ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR);
}

// 12. NullBufferSize -> ACL_SPARSE_STATUS_INVALID_VALUE
TEST_F(CoosortTest, NullBufferSize)
{
    int dummyRows = 0, dummyCols = 0;
    EXPECT_EQ(aclsparseXcoosort_bufferSizeExt(handle(), 4, 4, 1, &dummyRows, &dummyCols, nullptr),
        ACL_SPARSE_STATUS_INVALID_VALUE);
}

// 13. InvalidM (m <= 0) -> ACL_SPARSE_STATUS_INVALID_VALUE
TEST_F(CoosortTest, InvalidM)
{
    int dummyRows = 0, dummyCols = 0;
    size_t bufSize = 0;
    EXPECT_EQ(aclsparseXcoosort_bufferSizeExt(handle(), 0, 4, 1, &dummyRows, &dummyCols, &bufSize),
        ACL_SPARSE_STATUS_INVALID_VALUE);
    EXPECT_EQ(aclsparseXcoosort_bufferSizeExt(handle(), -1, 4, 1, &dummyRows, &dummyCols, &bufSize),
        ACL_SPARSE_STATUS_INVALID_VALUE);
}

// 14. InvalidN (n <= 0) -> ACL_SPARSE_STATUS_INVALID_VALUE
TEST_F(CoosortTest, InvalidN)
{
    int dummyRows = 0, dummyCols = 0;
    size_t bufSize = 0;
    EXPECT_EQ(aclsparseXcoosort_bufferSizeExt(handle(), 4, 0, 1, &dummyRows, &dummyCols, &bufSize),
        ACL_SPARSE_STATUS_INVALID_VALUE);
    EXPECT_EQ(aclsparseXcoosort_bufferSizeExt(handle(), 4, -3, 1, &dummyRows, &dummyCols, &bufSize),
        ACL_SPARSE_STATUS_INVALID_VALUE);
}

// 15. NullRows (nnz > 0, cooRowsA == nullptr) -> ACL_SPARSE_STATUS_INVALID_VALUE
TEST_F(CoosortTest, NullRows)
{
    int dummyCols = 0;
    size_t bufSize = 0;
    EXPECT_EQ(aclsparseXcoosort_bufferSizeExt(handle(), 4, 4, 1, nullptr, &dummyCols, &bufSize),
        ACL_SPARSE_STATUS_INVALID_VALUE);
}

// 16. NullCols (nnz > 0, cooColsA == nullptr) -> ACL_SPARSE_STATUS_INVALID_VALUE
TEST_F(CoosortTest, NullCols)
{
    int dummyRows = 0;
    size_t bufSize = 0;
    EXPECT_EQ(aclsparseXcoosort_bufferSizeExt(handle(), 4, 4, 1, &dummyRows, nullptr, &bufSize),
        ACL_SPARSE_STATUS_INVALID_VALUE);
}

// 17. NullP (nnz > 0, P == nullptr) -> ACL_SPARSE_STATUS_INVALID_VALUE
TEST_F(CoosortTest, NullP)
{
    const int m = 4, n = 4, nnz = 4;
    std::vector<int32_t> rows = {1, 0, 2, 1};
    std::vector<int32_t> cols = {3, 2, 0, 1};
    auto dRows = DeviceBuffer::copyFrom(rows.data(), rows.size() * sizeof(int32_t));
    auto dCols = DeviceBuffer::copyFrom(cols.data(), cols.size() * sizeof(int32_t));

    size_t bufSize = 0;
    ASSERT_EQ(aclsparseXcoosort_bufferSizeExt(handle(),
                  m,
                  n,
                  nnz,
                  reinterpret_cast<const int *>(dRows.raw()),
                  reinterpret_cast<const int *>(dCols.raw()),
                  &bufSize),
        ACL_SPARSE_STATUS_SUCCESS);
    auto dBuf = DeviceBuffer::alloc(bufSize);

    EXPECT_EQ(aclsparseXcoosortByRow(handle(),
                  m,
                  n,
                  nnz,
                  reinterpret_cast<int *>(dRows.get()),
                  reinterpret_cast<int *>(dCols.get()),
                  nullptr,
                  dBuf.get()),
        ACL_SPARSE_STATUS_INVALID_VALUE);
}

// 18. NullPBuffer (pBuffer == nullptr) -> ACL_SPARSE_STATUS_INVALID_VALUE
TEST_F(CoosortTest, NullPBuffer)
{
    const int m = 4, n = 4, nnz = 4;
    std::vector<int32_t> rows = {1, 0, 2, 1};
    std::vector<int32_t> cols = {3, 2, 0, 1};
    auto pHost = IdentityPerm(nnz);
    auto dRows = DeviceBuffer::copyFrom(rows.data(), rows.size() * sizeof(int32_t));
    auto dCols = DeviceBuffer::copyFrom(cols.data(), cols.size() * sizeof(int32_t));
    auto dP = DeviceBuffer::copyFrom(pHost.data(), pHost.size() * sizeof(int32_t));

    EXPECT_EQ(aclsparseXcoosortByRow(handle(),
                  m,
                  n,
                  nnz,
                  reinterpret_cast<int *>(dRows.get()),
                  reinterpret_cast<int *>(dCols.get()),
                  reinterpret_cast<int *>(dP.get()),
                  nullptr),
        ACL_SPARSE_STATUS_INVALID_VALUE);
}

// 19. NnzZero -> bufferSizeExt returns 0; ByRow/ByColumn return SUCCESS.
TEST_F(CoosortTest, NnzZero)
{
    const int m = 4, n = 4, nnz = 0;
    // rows/cols/P may be nullptr when nnz == 0.
    size_t bufSize = 99;  // sentinel
    EXPECT_EQ(
        aclsparseXcoosort_bufferSizeExt(handle(), m, n, nnz, nullptr, nullptr, &bufSize), ACL_SPARSE_STATUS_SUCCESS);
    EXPECT_EQ(bufSize, 0u);

    EXPECT_EQ(
        aclsparseXcoosortByRow(handle(), m, n, nnz, nullptr, nullptr, nullptr, nullptr), ACL_SPARSE_STATUS_SUCCESS);
    EXPECT_EQ(
        aclsparseXcoosortByColumn(handle(), m, n, nnz, nullptr, nullptr, nullptr, nullptr), ACL_SPARSE_STATUS_SUCCESS);
}

// 20. 使用 INT_MAX 验证超大 nnz 的 ping-pong workspace 大小计算；
//     这里只查询容量，不分配对应规模的数据，也不启动 kernel。
TEST_F(CoosortTest, BufferSizeExtIntMax)
{
    const int nnz = std::numeric_limits<int>::max();
    int dummy = 0;
    size_t bufSize = 0;

    EXPECT_EQ(
        aclsparseXcoosort_bufferSizeExt(handle(), 1, 1, nnz, &dummy, &dummy, &bufSize), ACL_SPARSE_STATUS_SUCCESS);
    size_t halfBytes = static_cast<size_t>(nnz) * 3U * sizeof(int32_t);
    size_t expected =
        (halfBytes * 2U + 8U * sizeof(int32_t) + ::kCoosortAlign - 1U) & ~(static_cast<size_t>(::kCoosortAlign) - 1U);
    EXPECT_EQ(bufSize, expected);
}

// 21. nnz 超过单核分发阈值，验证多核 ByRow 结果与 Host golden 一致。
TEST_F(CoosortTest, LargeNnz_MultiCore_ByRow)
{
    const int rows = 96, cols = 96;
    auto coo = makeSparseCoo(rows, cols, /*sparsity=*/0.72, /*seed=*/101);
    ASSERT_GT(coo.nnz, 0);
    ASSERT_GT(static_cast<uint32_t>(coo.nnz), kCoosortSingleCoreMaxNnz)
        << "Expected nnz > " << kCoosortSingleCoreMaxNnz << " to exercise multi-core path, got nnz=" << coo.nnz;
    EXPECT_TRUE(RunSortAndVerify(this,
        static_cast<int>(coo.rows),
        static_cast<int>(coo.cols),
        coo.rowIndices,
        coo.colIndices,
        /*sortByRow=*/true,
        "LargeNnz_MultiCore_ByRow"));
}

// 22. nnz 超过单核分发阈值，验证多核 ByColumn。
TEST_F(CoosortTest, LargeNnz_MultiCore_ByColumn)
{
    const int rows = 96, cols = 96;
    auto coo = makeSparseCoo(rows, cols, /*sparsity=*/0.72, /*seed=*/102);
    ASSERT_GT(coo.nnz, 0);
    ASSERT_GT(static_cast<uint32_t>(coo.nnz), kCoosortSingleCoreMaxNnz);
    EXPECT_TRUE(RunSortAndVerify(this,
        static_cast<int>(coo.rows),
        static_cast<int>(coo.cols),
        coo.rowIndices,
        coo.colIndices,
        /*sortByRow=*/false,
        "LargeNnz_MultiCore_ByColumn"));
}

// 23. nnz 约 5000，覆盖两个 Phase 1 run 之间的 merge-path 归并。
TEST_F(CoosortTest, LargeNnz_MultiCore_Merge)
{
    const int rows = 96, cols = 96;
    auto coo = makeSparseCoo(rows, cols, /*sparsity=*/0.44, /*seed=*/103);
    ASSERT_GT(coo.nnz, 0);
    ASSERT_GT(static_cast<uint32_t>(coo.nnz), kCoosortSingleCoreMaxNnz);
    EXPECT_TRUE(RunSortAndVerify(this,
        static_cast<int>(coo.rows),
        static_cast<int>(coo.cols),
        coo.rowIndices,
        coo.colIndices,
        /*sortByRow=*/true,
        "LargeNnz_MultiCore_Merge"));
}

// 24. 多核路径包含重复双键，验证全局稳定性。
TEST_F(CoosortTest, LargeNnz_MultiCore_Stable)
{
    const int rows = 72, cols = 72;
    auto coo = makeSparseCoo(rows, cols, /*sparsity=*/0.40, /*seed=*/104);
    ASSERT_GT(coo.nnz, 0);
    ASSERT_GT(static_cast<uint32_t>(coo.nnz), kCoosortSingleCoreMaxNnz);
    EXPECT_TRUE(RunSortAndVerify(this,
        static_cast<int>(coo.rows),
        static_cast<int>(coo.cols),
        coo.rowIndices,
        coo.colIndices,
        /*sortByRow=*/true,
        "LargeNnz_MultiCore_Stable"));
}

// 25. nnz 刚超过单核分发阈值，覆盖多核路径的阈值边界。
TEST_F(CoosortTest, LargeNnz_MultiCore_BarelyAbove)
{
    const int rows = 72, cols = 72;
    auto coo = makeSparseCoo(rows, cols, /*sparsity=*/0.59, /*seed=*/105);
    ASSERT_GT(coo.nnz, 0);
    ASSERT_GT(static_cast<uint32_t>(coo.nnz), kCoosortSingleCoreMaxNnz)
        << "Expected nnz > " << kCoosortSingleCoreMaxNnz << ", got nnz=" << coo.nnz;
    EXPECT_TRUE(RunSortAndVerify(this,
        static_cast<int>(coo.rows),
        static_cast<int>(coo.cols),
        coo.rowIndices,
        coo.colIndices,
        /*sortByRow=*/true,
        "LargeNnz_MultiCore_BarelyAbove"));
}

// 26. nnz 大于“每个 AIV 核只处理一个理论最大 run”所能覆盖的总量，
//     保证 runCount > coreNum，覆盖每核在 Phase 1 循环执行多次 Sort。
TEST_F(CoosortTest, VeryLargeNnz_MultiRunPerCore)
{
    int64_t coreNum = 0;
    int64_t ubSize = 0;
    ASSERT_EQ(aclrtGetDeviceInfo(TEST_DEVICE_ID, ACL_DEV_ATTR_VECTOR_CORE_NUM, &coreNum), ACL_SUCCESS);
    ASSERT_EQ(aclrtGetDeviceInfo(TEST_DEVICE_ID, ACL_DEV_ATTR_UBUF_PER_VECTOR_CORE, &ubSize), ACL_SUCCESS);
    ASSERT_GT(coreNum, 0);
    ASSERT_GT(ubSize, 0);

    constexpr uint32_t kAlignElems = 8U;
    constexpr uint32_t kPhaseOneBytesPerElem = 48U;
    uint64_t maxRunUpperBound = (static_cast<uint64_t>(ubSize) / kPhaseOneBytesPerElem / kAlignElems) * kAlignElems;
    ASSERT_GT(maxRunUpperBound, 0U);
    uint64_t nnz64 = static_cast<uint64_t>(coreNum) * maxRunUpperBound + 17U;
    ASSERT_LE(nnz64, static_cast<uint64_t>(std::numeric_limits<int>::max()));
    const int nnz = static_cast<int>(nnz64);
    const int rows = 4096;
    const int cols = 4096;

    std::vector<int32_t> rowIndices(nnz);
    std::vector<int32_t> colIndices(nnz);
    for (int i = 0; i < nnz; ++i) {
        uint64_t index = static_cast<uint64_t>(i);
        rowIndices[i] = static_cast<int32_t>((index * 37U + index / 97U) % rows);
        colIndices[i] = static_cast<int32_t>((index * 101U + index / 53U) % cols);
    }

    EXPECT_TRUE(RunSortAndVerify(this,
        rows,
        cols,
        rowIndices,
        colIndices,
        /*sortByRow=*/true,
        "VeryLargeNnz_MultiRunPerCore"));
}

// 27. bufferSizeExt 不带 P 参数；即使尚未准备排列数组，也应正常返回空间大小。
TEST_F(CoosortTest, BufferSizeExtNoP)
{
    const int m = 4, n = 4, nnz = 5;
    std::vector<int32_t> rows = {1, 0, 0, 1, 0};
    std::vector<int32_t> cols = {9, 7, 3, 2, 5};
    auto dRows = DeviceBuffer::copyFrom(rows.data(), rows.size() * sizeof(int32_t));
    auto dCols = DeviceBuffer::copyFrom(cols.data(), cols.size() * sizeof(int32_t));

    size_t bufSize = 0;
    EXPECT_EQ(aclsparseXcoosort_bufferSizeExt(handle(),
                  m,
                  n,
                  nnz,
                  reinterpret_cast<const int *>(dRows.raw()),
                  reinterpret_cast<const int *>(dCols.raw()),
                  &bufSize),
        ACL_SPARSE_STATUS_SUCCESS);
    // Reasonable lower bound: nnz*12 aligned to 128.
    EXPECT_GE(bufSize, coosortExpectedBufferSizeMin(nnz));
}

// ============================================================================
// main()
// ============================================================================
int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);
    g_acl_env = new AclTestEnvironment();
    testing::AddGlobalTestEnvironment(g_acl_env);
    return RUN_ALL_TESTS();
}
