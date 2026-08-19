/**
 * ----------------------------------------------------------------------------------------------------------
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ----------------------------------------------------------------------------------------------------------
 */

/**
 * @file prune_test.cpp
 * @brief GTest + CSV-driven test for aclsparseLtSpMMAPrune (prune-only).
 *
 * Formula: A_pruned = Prune_2:4(A)
 *
 * Test structure:
 *   - TEST_P (PruneTest) : parameterized success-path tests from CSV
 *       1. host data generation (range-bounded random, FP32 -> T)
 *       2. CPU golden reference (prune_golden.h: SpMMAPruneGolden)
 *       3. NPU wrapper (prune-only aclsparseLt chain, stops at SpMMAPrune)
 *       4. D2H gather of A_pruned
 *       5. Element-wise verification (prune is deterministic -> exact match
 *          in FP32 domain via MIXED_TOLERANCE, which for exact-equal inputs
 *          trivially passes)
 *
 * Per-case NPU elapsed time (SpMMAPrune, synchronised) is collected via
 * aclrtEvent and reported for performance profiling.
 *
 * Entry point is shared via test/frame/test_main.cpp; this file MUST NOT
 * define main().
 */

#include "test_common.h"

#include "prune_golden.h"
#include "prune_npu_wrapper.h"
#include "prune_param.h"

#include <chrono>

using namespace sparse_test;

namespace {

class PruneAclEnvScopeMixin {
protected:
    inline static std::unique_ptr<AclEnvScope> env_;
    aclrtStream stream_ = nullptr;
    static void InitEnv() {
        env_ = std::make_unique<AclEnvScope>();
    }
    static void FiniEnv() { env_.reset(); }
    void SetUpStream() { stream_ = env_->stream(); }
};

class PruneTestFixture : public testing::Test, public PruneAclEnvScopeMixin {
public:
    static void SetUpTestSuite() { InitEnv(); }
    static void TearDownTestSuite() { FiniEnv(); }
protected:
    void SetUp() override { SetUpStream(); }
};

template <typename P>
class PruneParamFixture : public testing::TestWithParam<P>, public PruneAclEnvScopeMixin {
public:
    static void SetUpTestSuite() { InitEnv(); }
    static void TearDownTestSuite() { FiniEnv(); }
protected:
    void SetUp() override { SetUpStream(); }
};

inline void VerifyPruneMixedTolerance(const std::vector<float>& npuF,
                                        const std::vector<float>& goldenF,
                                        aclDataType aclDtype, const std::string& caseId)
{
    ASSERT_EQ(npuF.size(), goldenF.size());
    VerifyConfig cfg;
    applyMixedTolerance(cfg, aclDtype, goldenF.data(), goldenF.size());
    if (aclDtype == ACL_INT8) {
        cfg.mixedRtol = 0.0;
        cfg.mixedAtol = 0.0;
    }
    EXPECT_TRUE(Verifier::verifyVector(npuF, goldenF, cfg, caseId));
}

// Deterministic seed per case for reproducible random inputs.
inline uint32_t SeedForCase(int32_t caseId) { return 2000u + static_cast<uint32_t>(caseId); }

// GenFp32Matrix / ToStorageFp32 / ToStorageFp16 / GenTestMatrix
// 定义在 prune_test_util.h 中，通过 `using namespace sparse_test` 可用。

template <typename T>
inline std::vector<T> PreparePruneTestData(const PruneParam& p)
{
    return MatmulGenTestMatrix<T>(p.m, p.k, p.range_low, p.range_high, SeedForCase(p.case_id));
}

inline void ReportPrunePerf(const PruneParam& p,
                             double msGen, double msGolden, double msNpuWall,
                             const PruneNpuResult& npu)
{
    std::cout << "[" << p.caseId() << "] perf: datagen=" << msGen << "ms"
              << " golden=" << msGolden << "ms"
              << " npu_wall=" << msNpuWall << "ms"
              << " npu_device=" << npu.npuMs << "ms"
              << " ws=" << npu.workspaceSize << "B"
              << " (" << p.m << "x" << p.k
              << " " << p.dtype << " alg" << p.alg_config_id
              << " splitk" << p.split_k
              << " " << (p.isTile() ? "TILE" : "STRIP")
              << " " << (p.isTranspose() ? "T" : "N")
              << (p.isColOrder() ? "COL" : "ROW")
              << " side=" << (p.isSparseA() ? "A" : "B") << ")\n";
}

// Templated end-to-end runner shared by both dtype branches and A/B sparse sides.
// p.m/p.k denote the structured matrix shape (A-sparse: A is (m,k);
// B-sparse: B is (m,k) = (logical k, logical n)). p.op is opA (A-sparse) or
// opB (B-sparse). The prune golden formula is symmetric in (m, k, trans, order),
// so the same SpMMAPruneGolden call covers both sides.
template <typename T>
void RunOneCase(const PruneParam& p, aclrtStream stream) {
    const auto t0 = std::chrono::steady_clock::now();

    // 1. Host data generation for the structured matrix (A-sparse -> A; B-sparse -> B).
    std::vector<T> hInput = PreparePruneTestData<T>(p);
    const auto t1 = std::chrono::steady_clock::now();

    // 2. CPU golden reference (prune-only, dtype + direction + pruneAlg aware).
    std::vector<T> hGoldenPruned;
    SpMMAPruneGolden<T>(hInput, hGoldenPruned, p.m, p.k, p.alg_config_id, p.split_k,
                         p.isTranspose(), !p.isColOrder(), p.pruneAlg);
    const auto t2 = std::chrono::steady_clock::now();

    // 3. NPU execution (prune-only aclsparseLt chain).
    std::vector<T> hNpuPruned;
    const aclsparseOperation_t op = p.isTranspose() ? ACL_SPARSE_OP_TRANSPOSE
                                                    : ACL_SPARSE_OP_NON_TRANSPOSE;
    const aclsparseOrder_t order = p.isColOrder() ? ACL_SPARSE_ORDER_COL
                                                     : ACL_SPARSE_ORDER_ROW;
    const aclsparseLtPruneAlg_t pruneAlg = p.isTile()
        ? ACLSPARSELT_PRUNE_SPMMA_TILE
        : ACLSPARSELT_PRUNE_SPMMA_STRIP;
    auto npu = p.isSparseA()
        ? PruneNpu<T>(stream, hInput, hNpuPruned, p.m, p.k, p.alg_config_id, p.split_k,
                       op, order, pruneAlg)
        : PruneNpuBSparse<T>(stream, hInput, hNpuPruned, p.m, p.k, p.alg_config_id, p.split_k,
                              op, order, pruneAlg);
    const auto t3 = std::chrono::steady_clock::now();

    // 4. Assert the prune API returned SUCCESS.
    //    (prune is independent of alg_config_id / split_k — no AlgSetAttribute
    //    step in the decoupled prune-only test chain.)
    ASSERT_EQ(npu.pruneRet, ACL_SPARSE_STATUS_SUCCESS)
        << "SpMMAPrune failed for " << p.caseId();

    // 5. Performance report (before verification so timing is always printed).
    double msGen = std::chrono::duration<double, std::milli>(t1 - t0).count();
    double msGolden = std::chrono::duration<double, std::milli>(t2 - t1).count();
    double msNpuWall = std::chrono::duration<double, std::milli>(t3 - t2).count();
    ReportPrunePerf(p, msGen, msGolden, msNpuWall, npu);

    // 6. Verification in FP32 domain (last — shared helper).
    // Prune is deterministic: the NPU pruned matrix should match the CPU
    // golden pruned matrix exactly (same elements zeroed, same values kept).
    std::vector<float> npuF = PruneToFloat(hNpuPruned);
    std::vector<float> goldenF = PruneToFloat(hGoldenPruned);
    VerifyPruneMixedTolerance(npuF, goldenF, NpuDtypeTrait<T>::kAclDtype, p.caseId());
}

}  // namespace

// =============================================================================
// Parameterized fixture
// =============================================================================

class PruneTest : public PruneParamFixture<PruneParam> {
};

TEST_P(PruneTest, FunctionalPrecision) {
    const auto& p = GetParam();
    std::cout << "\n==== " << p.caseId() << " ===="
              << " m=" << p.m << " k=" << p.k
              << " dtype=" << p.dtype
              << " alg=" << p.alg_config_id << " split_k=" << p.split_k
              << " pruneAlg=" << (p.isTile() ? "TILE" : "STRIP")
              << " op=" << (p.isTranspose() ? "T" : "N")
              << " order=" << (p.isColOrder() ? "COL" : "ROW")
              << " range=[" << p.range_low << "," << p.range_high << "]"
              << " level=" << p.level << "\n";

    if (p.isFp16()) {
        RunOneCase<uint16_t>(p, stream_);
    } else if (p.isBf16()) {
        RunOneCase<bf16_bits_t>(p, stream_);
    } else if (p.isInt8()) {
        RunOneCase<int8_t>(p, stream_);
    } else {
        RunOneCase<float>(p, stream_);
    }
}

INSTANTIATE_TEST_SUITE_P(
    PruneCases,
    PruneTest,
    testing::ValuesIn(GetCasesFromCsv<PruneParam>("prune_test.csv")),
    [](const testing::TestParamInfo<PruneParam>& info) {
        return info.param.caseId();
    }
);

// =============================================================================
// Exception-path tests for aclsparseLtSpMMAPrune.
// Verifies parameter validation: null handle, null descriptor, null d_in/d_out,
// invalid pruneAlg, and null stream.
// =============================================================================

class PruneExceptionTest : public PruneTestFixture {
};

// Helper to create a valid matmul descriptor for prune tests is unnecessary —
// each test builds its own descriptors via RAII guards for clarity.

// Shared descriptor chain for prune exception tests PE4/PE5/PE6.
// These three tests all create the same handle + dA + dAPruned + matA + matB +
// matC + matD + matmulDesc, differing only in the SpMMAPrune call parameters.
// Extracted into PruneTestCtx to eliminate the triplicated descriptor creation
// chain. Members are initialized in declaration order so each guard's
// constructor parameters (which reference earlier members) are valid.
struct PruneTestCtx {
    SparseLtHandleGuard handle;
    sparse_test::DeviceBuffer dA;
    sparse_test::DeviceBuffer dAPruned;
    MatDescrGuard matA;
    MatDescrGuard matB;
    MatDescrGuard matC;
    MatDescrGuard matD;
    MatmulDescrGuard md;

    PruneTestCtx()
        : handle(),
          dA(sparse_test::DeviceBuffer::alloc(128 * 128 * sizeof(float))),
          dAPruned(sparse_test::DeviceBuffer::alloc(128 * 128 * sizeof(float)))
    {
        matA.initStructured(handle.ptr(), 128, 128, 128, 16, ACL_FLOAT,
                            ACL_SPARSE_ORDER_ROW, ACL_SPARSE_LT_SPARSITY_50_PERCENT);
        matB.initDense(handle.ptr(), 128, 128, 128, 16, ACL_FLOAT,
                       ACL_SPARSE_ORDER_ROW);
        matC.initDense(handle.ptr(), 128, 128, 128, 16, ACL_FLOAT,
                       ACL_SPARSE_ORDER_ROW);
        matD.initDense(handle.ptr(), 128, 128, 128, 16, ACL_FLOAT,
                       ACL_SPARSE_ORDER_ROW);
        md.init(handle.ptr(), ACL_SPARSE_OP_NON_TRANSPOSE,
                ACL_SPARSE_OP_NON_TRANSPOSE,
                matA.ptr(), matB.ptr(), matC.ptr(), matD.ptr(), ACL_SPARSE_COMPUTE_32F);
    }
};

// PE1: null handle -> expected non-SUCCESS.
TEST_F(PruneExceptionTest, NullHandle) {
    auto ret = aclsparseLtSpMMAPrune(
        nullptr, nullptr, nullptr, nullptr, ACLSPARSELT_PRUNE_SPMMA_STRIP, stream_);
    EXPECT_NE(ret, ACL_SPARSE_STATUS_SUCCESS);
}

// PE2: null matmulDescr -> expected non-SUCCESS.
TEST_F(PruneExceptionTest, NullDescriptor) {
    SparseLtHandleGuard handle;
    sparse_test::DeviceBuffer dA = sparse_test::DeviceBuffer::alloc(128 * 128 * sizeof(float));
    sparse_test::DeviceBuffer dAPruned = sparse_test::DeviceBuffer::alloc(128 * 128 * sizeof(float));
    auto ret = aclsparseLtSpMMAPrune(
        handle.ptr(), nullptr, dA.get(), dAPruned.get(), ACLSPARSELT_PRUNE_SPMMA_STRIP, stream_);
    EXPECT_NE(ret, ACL_SPARSE_STATUS_SUCCESS);
}

// PE3: null d_in -> expected non-SUCCESS (descriptors no longer bind values,
// so d_in must be passed explicitly to SpMMAPrune).
TEST_F(PruneExceptionTest, NullDinAndMatAValues) {
    SparseLtHandleGuard handle;
    // Create descriptor (values are no longer bound at init time).
    MatDescrGuard matA;
    matA.initStructured(handle.ptr(), 128, 128, 128, 16, ACL_FLOAT,
                        ACL_SPARSE_ORDER_ROW, ACL_SPARSE_LT_SPARSITY_50_PERCENT);
    MatDescrGuard matB;
    matB.initDense(handle.ptr(), 128, 128, 128, 16, ACL_FLOAT,
                   ACL_SPARSE_ORDER_ROW);
    MatDescrGuard matC;
    matC.initDense(handle.ptr(), 128, 128, 128, 16, ACL_FLOAT,
                   ACL_SPARSE_ORDER_ROW);
    MatDescrGuard matD;
    matD.initDense(handle.ptr(), 128, 128, 128, 16, ACL_FLOAT,
                   ACL_SPARSE_ORDER_ROW);
    MatmulDescrGuard md;
    md.init(handle.ptr(), ACL_SPARSE_OP_NON_TRANSPOSE,
            ACL_SPARSE_OP_NON_TRANSPOSE,
            matA.ptr(), matB.ptr(), matC.ptr(), matD.ptr(), ACL_SPARSE_COMPUTE_32F);
    sparse_test::DeviceBuffer dAPruned = sparse_test::DeviceBuffer::alloc(128 * 128 * sizeof(float));
    auto ret = aclsparseLtSpMMAPrune(
        handle.ptr(), matmulDescCptr(md), nullptr, dAPruned.get(), ACLSPARSELT_PRUNE_SPMMA_STRIP, stream_);
    EXPECT_NE(ret, ACL_SPARSE_STATUS_SUCCESS);
}

// PE4: null d_out -> expected non-SUCCESS.
TEST_F(PruneExceptionTest, NullDout) {
    PruneTestCtx ctx;
    auto ret = aclsparseLtSpMMAPrune(
        ctx.handle.ptr(), matmulDescCptr(ctx.md), ctx.dA.get(), nullptr,
        ACLSPARSELT_PRUNE_SPMMA_STRIP, stream_);
    EXPECT_NE(ret, ACL_SPARSE_STATUS_SUCCESS);
}

// PE5: [TILE] TILE mode is now a supported algorithm (no longer NOT_SUPPORTED).
// The original UnsupportedPruneAlgTile test expected NOT_SUPPORTED; with TILE
// implementation complete, this test is removed. TILE normal-path coverage is
// provided by CSV cases 23-43. TILE exception paths (null d_out / null handle)
// are covered by TileNullDout and TileNullHandle below.

// PE6: null stream -> expected SUCCESS (default stream is valid in ACL runtime).
// ACL runtime treats nullptr/0 as the default stream, which is
// valid for all aclrtMemcpy / kernel-launch APIs. The prune implementation
// (validate_prune_params) correctly accepts null stream. This aligns with the
// matmul WB_L2_43_Streams0Null fix (null/0 stream is a valid default stream).
TEST_F(PruneExceptionTest, NullStream) {
    PruneTestCtx ctx;
    auto ret = aclsparseLtSpMMAPrune(
        ctx.handle.ptr(), matmulDescCptr(ctx.md), ctx.dA.get(), ctx.dAPruned.get(),
        ACLSPARSELT_PRUNE_SPMMA_STRIP, nullptr);
    EXPECT_EQ(ret, ACL_SPARSE_STATUS_SUCCESS);
}

// =============================================================================
// [TILE] Exception-path tests with pruneAlg=TILE.
// Verifies that TILE mode also validates parameters (null d_out / null handle).
// =============================================================================

// TILE + null d_out -> expected non-SUCCESS.
// Confirms TILE path also goes through resolve_prune_buffers validation.
TEST_F(PruneExceptionTest, TileNullDout) {
    PruneTestCtx ctx;
    auto ret = aclsparseLtSpMMAPrune(
        ctx.handle.ptr(), matmulDescCptr(ctx.md), ctx.dA.get(), nullptr,
        ACLSPARSELT_PRUNE_SPMMA_TILE, stream_);
    EXPECT_NE(ret, ACL_SPARSE_STATUS_SUCCESS);
}

// TILE + null handle -> expected non-SUCCESS.
TEST_F(PruneExceptionTest, TileNullHandle) {
    auto ret = aclsparseLtSpMMAPrune(
        nullptr, nullptr, nullptr, nullptr,
        ACLSPARSELT_PRUNE_SPMMA_TILE, stream_);
    EXPECT_NE(ret, ACL_SPARSE_STATUS_SUCCESS);
}

// =============================================================================
// [TILE] Property test: TILE 2D structural sparsity.
// TILE pruning applies a row/col joint constraint: within each TS×TS tile,
// every row keeps exactly KEEP nonzeros AND every column keeps exactly KEEP
// nonzeros. STRIP pruning only enforces the per-row constraint (KEEP nonzeros
// per group of TS within each row); the column distribution is unconstrained.
//
// This test verifies the core value proposition of TILE mode — the 2D
// structural sparsity that enables hardware-aware structured matmul:
//   1. TILE output: each 4×4 tile has exactly 2 nonzeros per row and per column.
//   2. STRIP output: each row group of 4 has exactly 2 nonzeros (row constraint
//      satisfied), but the per-column count within a 4×4 tile is NOT required
//      to be 2 (column constraint not enforced).
//
// This is a CPU-only property test (no NPU execution needed).
// =============================================================================

namespace {

inline int32_t CountRowNnzInTile(const std::vector<float>& mat, int32_t k,
                                  int32_t baseRow, int32_t baseCol, int32_t ts)
{
    int32_t nnz = 0;
    for (int32_t c = 0; c < ts; ++c) {
        if (mat[static_cast<size_t>(baseRow) * k + (baseCol + c)] != 0.0f) {
            ++nnz;
        }
    }
    return nnz;
}

inline int32_t CountColNnzInTile(const std::vector<float>& mat, int32_t k,
                                  int32_t baseRow, int32_t baseCol, int32_t ts)
{
    int32_t nnz = 0;
    for (int32_t r = 0; r < ts; ++r) {
        if (mat[static_cast<size_t>(baseRow + r) * k + baseCol] != 0.0f) {
            ++nnz;
        }
    }
    return nnz;
}

inline int32_t CountTileTotalNnz(const std::vector<float>& mat, int32_t k,
                                  int32_t baseRow, int32_t baseCol, int32_t ts)
{
    int32_t nnz = 0;
    for (int32_t r = 0; r < ts; ++r) {
        nnz += CountRowNnzInTile(mat, k, baseRow + r, baseCol, ts);
    }
    return nnz;
}

inline void VerifyTile2DConstraint(const std::vector<float>& tileF,
                                    int32_t m, int32_t k, int32_t ts, int32_t keep)
{
    if (ts <= 0) { return; }
    const int32_t safeTs = (ts > 0) ? ts : 1;
    int32_t tileChecked = 0;
    for (int32_t ti = 0; ti < m / safeTs; ++ti) {
        for (int32_t tj = 0; tj < k / safeTs; ++tj) {
            const int32_t br = ti * safeTs;
            const int32_t bc = tj * safeTs;
            for (int32_t r = 0; r < safeTs; ++r) {
                EXPECT_EQ(CountRowNnzInTile(tileF, k, br + r, bc, safeTs), keep)
                    << "TILE row constraint violated at tile(" << ti << "," << tj
                    << ") row " << r;
            }
            for (int32_t c = 0; c < safeTs; ++c) {
                EXPECT_EQ(CountColNnzInTile(tileF, k, br, bc + c, safeTs), keep)
                    << "TILE column constraint violated at tile(" << ti << "," << tj
                    << ") col " << c;
            }
            EXPECT_EQ(CountTileTotalNnz(tileF, k, br, bc, safeTs), keep * safeTs)
                << "TILE total nnz violated at tile(" << ti << "," << tj << ")";
            ++tileChecked;
        }
    }
    std::cout << "[Property] TILE: checked " << tileChecked << " tiles, all satisfy "
              << "2D constraint (2 per row + 2 per col).\n";
}

inline void VerifyStripRowConstraint(const std::vector<float>& stripF,
                                      int32_t m, int32_t k, int32_t ts, int32_t keep)
{
    if (ts <= 0) { return; }
    const int32_t safeTs = (ts > 0) ? ts : 1;
    for (int32_t i = 0; i < m; ++i) {
        for (int32_t gj = 0; gj < k / safeTs; ++gj) {
            EXPECT_EQ(CountRowNnzInTile(stripF, k, i, gj * safeTs, safeTs), keep)
                << "STRIP row constraint violated at row " << i << " group " << gj;
        }
    }
}

inline int32_t CountStripColViolations(const std::vector<float>& stripF,
                                        int32_t m, int32_t k, int32_t ts, int32_t keep)
{
    if (ts <= 0) { return 0; }
    const int32_t safeTs = (ts > 0) ? ts : 1;
    int32_t violations = 0;
    for (int32_t ti = 0; ti < m / safeTs; ++ti) {
        for (int32_t tj = 0; tj < k / safeTs; ++tj) {
            const int32_t br = ti * safeTs;
            const int32_t bc = tj * safeTs;
            for (int32_t c = 0; c < safeTs; ++c) {
                if (CountColNnzInTile(stripF, k, br, bc + c, safeTs) != keep) {
                    ++violations;
                    break;
                }
            }
        }
    }
    return violations;
}

}  // namespace

class PrunePropertyTest : public PruneTestFixture {
};

TEST_F(PrunePropertyTest, TileStructuralSparsity) {
    const int32_t m = 16;
    const int32_t k = 16;
    const int32_t ts = 4;
    const int32_t keep = 2;
    auto hA = GenTestMatrix<uint16_t>(m, k, -1.0f, 1.0f, 7000u);

    std::vector<uint16_t> hStripPruned;
    std::vector<uint16_t> hTilePruned;
    SpMMAPruneGolden<uint16_t>(hA, hStripPruned, m, k, 0, 1, false, true, "STRIP");
    SpMMAPruneGolden<uint16_t>(hA, hTilePruned, m, k, 0, 1, false, true, "TILE");

    auto stripF = PruneToFloat(hStripPruned);
    auto tileF = PruneToFloat(hTilePruned);

    VerifyTile2DConstraint(tileF, m, k, ts, keep);
    VerifyStripRowConstraint(stripF, m, k, ts, keep);

    int32_t stripColViolations = CountStripColViolations(stripF, m, k, ts, keep);
    const int32_t safeTs = (ts > 0) ? ts : 1;
    std::cout << "[Property] STRIP: row constraint satisfied (2 per group per row).\n"
              << "[Property] STRIP: column constraint violated in " << stripColViolations
              << " of " << (m / safeTs) * (k / safeTs) << " tiles (expected: violations likely"
              << " since STRIP has no column constraint).\n";
}
