/**
 * ----------------------------------------------------------------------------------------------------------
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ----------------------------------------------------------------------------------------------------------
 */

/*!
 * \file aclsparselt_internal.h
 * \brief sparseLt shared internal header: TilingData struct + constants
 *        (shared CXX/ASC), host-only helper functions (CXX only), and
 *        host-only descriptor internal structs (CXX only).
 *
 * Included by both host.cpp (CXX) and kernel.cpp (ASC). The TilingData struct,
 * constants and splt_load_tiling above the __CCE_AICORE__ guard are compiled by
 * both; the host-only helpers and descriptor structs below the guard are skipped
 * by the ASC compiler (host-only, pull in acl/acl.h + cann_ops_sparse*.h).
 *
 * Merged from the former single-folder header + the anonymous-namespace
 * helpers in the former host.cpp + splt_load_tiling from the former kernel.cpp so the
 * old single-folder layout can be split into shared/matmul/prune/alg_set_attr
 * sub-directories that all reuse this common internal header.
 */

#ifndef ACLSPARSELT_INTERNAL_H
#define ACLSPARSELT_INTERNAL_H


#include <cstdint>
#include <cfloat>

#ifndef __gm__
#define __gm__
#endif

// Data type encoding for host-side kernel dispatch (matches SpmmDtype convention).
#define SPLT_DTYPE_INVALID (-1)
#define SPLT_DTYPE_FP32 0
#define SPLT_DTYPE_FP16 1
// v2: BF16 / INT8 input dtypes; INT32 only used for outDataType field
// (INT8 input -> INT32 output path), never appears in dataType field.
#define SPLT_DTYPE_BF16 2
#define SPLT_DTYPE_INT8 3
#define SPLT_DTYPE_INT32 4

// Element byte sizes by dtype (used in workspace / UB capacity calculations).
#define SPLT_FP32_BYTES 4
#define SPLT_HALF_BYTES 2   // FP16 / BF16
#define SPLT_INT8_BYTES 1
// L1 budget (bytes) used for kL1 segment computation: kL1Size = SPLT_KL1_BUDGET_BYTES / elemSize.
#define SPLT_KL1_BUDGET_BYTES 512
// UB 32-byte alignment boundary (DataCopyPad requirement).
#define SPLT_UB_ALIGN_BYTES 32

// Workspace layout (all regions 64B aligned):
//   [ tilingOff .. aPrunedOff )  : AclsparseltTilingData
//   [ aPrunedOff .. tempOff )    : A_pruned (m * k * sizeof(T))
//   [ tempOff .. end )           : temp_result (splitK * m * n * sizeof(float))
#define SPLT_WS_HEADER_BYTES 64
#define SPLT_WS_ALIGN 64

// Hardware constants (DAV-3510).
#define SPLT_CORE_NUM 32
// L0C cube edge is always 16 on DAV-3510 (see tensor-api-reference §1.4).
#define SPLT_L0C_C0 16
// L0A/L0B buffer size (64KB each on DAV-3510). Used for L0 double-buffer offset.
#define SPLT_L0A_SIZE (64 * 1024)
// Half L0A for ping-pong double buffer (32KB per slot).
#define SPLT_HALF_L0_SIZE (SPLT_L0A_SIZE / 2)
// GM output (temp, FP32) C0 must match L0C C0 (=16) for correct Fixpipe
// ND output conversion. Previously used 8 (32/sizeof(float)), but the
// Fixpipe converts L0C NZ fractals (C0=16) to GM NDExt — a C0 mismatch
// causes incorrect fractal splitting that scrambles ~0.005% of elements.
#define SPLT_GM_FLOAT_C0 16

#define SPLT_PRUNE_ROW_TILE 16
// TILE mode constants.
// Tile size: FP16=4 (4x4 tile), FP32=2 (2x2 tile).
#define SPLT_TILE_SIZE_FP16 4
#define SPLT_TILE_SIZE_FP32 2
// Number of valid configurations per tile.
#define SPLT_TILE_CONFIGS_4X4 90   // 4x4, 2-per-row, 2-per-col
#define SPLT_TILE_CONFIGS_2X2 2    // 2x2, 1-per-row, 1-per-col
// Prune algorithm constants (mirror aclsparseLtPruneAlg_t enum for kernel use).
// Defined here because cann_ops_sparseLt.h is host-only (not ASC-compilable).
#define SPLT_PRUNE_ALG_TILE 0
#define SPLT_PRUNE_ALG_STRIP 1
// unitFlag values for Mmad (matching SDK blaze common_utils.h:67-68).
// FINAL_ACCUMULATION = 3: last K iteration, triggers M->FIX sync internally.
// NON_FINAL_ACCUMULATION = 2: intermediate K iterations, tells hardware more
// accumulations are coming. Using 0 (as before) is undefined behavior and
// causes occasional pipeline sync failures producing outlier elements.
#define SPLT_FINAL_ACCUMULATION 3
#define SPLT_NON_FINAL_ACCUMULATION 2

// TilingData read by every kernel from workspace[tilingOff].
// POD struct — copied H2D by host before each kernel launch.
typedef struct AclsparseltTilingData {
    // matrix dimensions
    int32_t m;
    int32_t n;
    int32_t k;
    // leading dimension of matrix A (may be > k when padded).
    // Prune kernel uses ld for row offsets (row*ld) instead of row*k to handle
    // padded matrices correctly. Matmul path ignores ld (A_pruned is contiguous).
    int32_t ld;
    // cube tiling parameters
    int32_t baseM;
    int32_t baseN;
    int32_t baseK;
    // [REFACTOR-KLOOP] K-dimension L1 tile size for the outer kL1 loop.
    // The K-loop is split into two layers: outer kL1 (GM→L1, kL1Size chunk)
    // and inner kL0 (L1→L0, baseK chunk). kL1Size = 512/sizeof(T) aligned to
    // baseK (256 for FP16, 128 for FP32). L1 capacity constraint:
    // 2*(baseM*kL1Size + kL1Size*baseN)*sizeof(T) <= 512KB (DAV-3510 L1 total).
    int32_t kL1Size;
    // split-k
    int32_t splitK;
    int32_t kSegLen;        // ceil(k / splitK) aligned up to baseK
    // algorithm config
    int32_t algConfigId;    // 0 / 1
    // data type (SPLT_DTYPE_*)
    int32_t dataType;
    // v2: output dtype (SPLT_DTYPE_*), drives INT8 epilogue dispatch.
    //   FP32/FP16/BF16: outDataType == dataType.
    //   INT8: outDataType == SPLT_DTYPE_INT8(3) or SPLT_DTYPE_INT32(4).
    int32_t outDataType;
    int32_t pruneAlongRow;
    // pruneAlg: 0=TILE (ACLSPARSELT_PRUNE_SPMMA_TILE), 1=STRIP (ACLSPARSELT_PRUNE_SPMMA_STRIP).
    // Consumed by the prune kernel to dispatch between TILE and STRIP paths.
    // Default 0 (TILE) when zero-initialized; prune host explicitly sets this.
    int32_t pruneAlg;
    // sparseTrans/transB flags: 1 = op(A)/op(B) is transpose.
    // sparseTrans is consumed by the prune kernel to read the physical (k,m) layout
    // and produce A_pruned as (m,k) row-major. transB is consumed by the matmul
    // kernel to declare the GM B tensor as (n,k) and transpose during CopyGM2L1.
    int32_t sparseTrans;
    int32_t transB;
    // scalars (filled at Matmul time; prune/compress ignore them)
    float alpha;
    float beta;
    // vector scaling flags (0=scalar, 1=per-row device pointer)
    int32_t alphaVectorScaling;
    int32_t betaVectorScaling;
    // device pointers to alpha/beta vectors (float[M]), valid when scaling==1
    uint64_t alphaDevPtr;
    uint64_t betaDevPtr;
    // multi-core
    int32_t coreNum;        // 32
    int32_t usedCoreNum;
    // workspace byte offsets
    // Use int64_t for offsets to prevent overflow on large
    // workspaces (m*k*sizeof(T) can exceed 2GB for large FP32 shapes).
    int64_t tilingOffset;
    int64_t aPrunedOffset;
    int64_t tempResultOffset;
    // —— bias + activation epilogue 字段（追加在末尾，向后兼容）——
    uint64_t biasDevPtr;       // bias device 指针，0=无 bias
    int64_t  biasStride;       // batch 间 bias 步长，0=广播
    int32_t  activationType;   // 0=无, 1=ReLU, 2=GeLU（统一编码，替代分离的 relu/gelu）
    float    reluUpperBound;   // 默认 FLT_MAX
    float    reluThreshold;    // 默认 0.0f
    float    geluScaling;      // 默认 1.0f
    int32_t  biasChunkMode;    // 0=全量加载, 1=per-chunk 降级（UB 超限时）
    // —— batch 字段（追加在末尾，向后兼容）——
    int32_t  numBatches;       // batch 数量，默认 1
    int64_t  batchStrideA;     // A 的 batch 步长（元素数）
    int64_t  batchStrideB;     // B 的 batch 步长
    int64_t  batchStrideC;     // C 的 batch 步长
    int64_t  batchStrideD;     // D 的 batch 步长
    // Prune trans path chunk size (columns per chunk).
    // Computed by host based on UB capacity. When m > chunkM, the trans path
    // splits the m-dimension into chunks of chunkM columns each. When m <= chunkM,
    // single chunk (= m), behavior unchanged from original. chunkM is aligned to
    // 32B/elemSize so that alignedChunkM == chunkM (exact UB row stride).
    // 0 = not set (non-trans paths or default); kernel treats 0 as "use m".
    int32_t  chunkM;
} AclsparseltTilingData;


// ============================================================================
// Device-side tiling loader (ASC only).
//
// Reads the TilingData POD struct from GM workspace[tilingOff] into a local
// copy. Marked __aicore__ inline so it is only usable from kernel code; the
// host compiler (CXX) skips this section via the __CCE_AICORE__ guard below.
// ============================================================================
#ifdef __CCE_AICORE__
__aicore__ inline AclsparseltTilingData splt_load_tiling(GM_ADDR tilingGm)
{
    __gm__ AclsparseltTilingData* gmTd = reinterpret_cast<__gm__ AclsparseltTilingData*>(tilingGm);
    AclsparseltTilingData td;
    td.m = gmTd->m;
    td.n = gmTd->n;
    td.k = gmTd->k;
    td.ld = gmTd->ld;  // load leading dimension for prune kernel
    td.baseM = gmTd->baseM;
    td.baseN = gmTd->baseN;
    td.baseK = gmTd->baseK;
    td.kL1Size = gmTd->kL1Size;
    td.splitK = gmTd->splitK;
    td.kSegLen = gmTd->kSegLen;
    td.algConfigId = gmTd->algConfigId;
    td.dataType = gmTd->dataType;
    td.outDataType = gmTd->outDataType;  // v2: load output dtype for epilogue dispatch
    td.pruneAlongRow = gmTd->pruneAlongRow;
    td.pruneAlg = gmTd->pruneAlg;
    td.sparseTrans = gmTd->sparseTrans;
    td.transB = gmTd->transB;
    td.alpha = gmTd->alpha;
    td.beta = gmTd->beta;
    td.alphaVectorScaling = gmTd->alphaVectorScaling;
    td.betaVectorScaling = gmTd->betaVectorScaling;
    td.alphaDevPtr = gmTd->alphaDevPtr;
    td.betaDevPtr = gmTd->betaDevPtr;
    td.coreNum = gmTd->coreNum;
    td.usedCoreNum = gmTd->usedCoreNum;
    td.tilingOffset = gmTd->tilingOffset;
    td.aPrunedOffset = gmTd->aPrunedOffset;
    td.tempResultOffset = gmTd->tempResultOffset;
    // bias + activation 字段
    td.biasDevPtr = gmTd->biasDevPtr;
    td.biasStride = gmTd->biasStride;
    td.activationType = gmTd->activationType;
    td.reluUpperBound = gmTd->reluUpperBound;
    td.reluThreshold = gmTd->reluThreshold;
    td.geluScaling = gmTd->geluScaling;
    td.biasChunkMode = gmTd->biasChunkMode;
    // batch 字段
    td.numBatches = gmTd->numBatches;
    td.batchStrideA = gmTd->batchStrideA;
    td.batchStrideB = gmTd->batchStrideB;
    td.batchStrideC = gmTd->batchStrideC;
    td.batchStrideD = gmTd->batchStrideD;
    td.chunkM = gmTd->chunkM;
    return td;
}
#endif // __CCE_AICORE__

// ============================================================================
// Host-side descriptor internal structs + helpers (not exposed publicly).
//
// Mirrors cuSPARSELt opaque-pointer pattern: external headers
// (cann_ops_sparseLt.h) only forward-declare the structs; internal layout lives
// here and is visible only to the library impl (host.cpp).
//
// Guarded by __CCE_AICORE__: kernel.cpp (ASC) includes this header for the
// TilingData struct above; the descriptor section is host-only (CXX) and must
// not be compiled by the ASC compiler (it pulls in acl/acl.h +
// cann_ops_sparse*.h which are not available / not compatible on the device
// side).
// ============================================================================
#ifndef __CCE_AICORE__
#include "aclsparselt_mat_descriptor_internal.h"
#include "aclsparselt_matmul_descriptor_internal.h"
#include "aclsparselt_matmul_alg_selection_internal.h"
#include "aclsparselt_matmul_plan_internal.h"
#include <acl/acl.h>
#include "cann_ops_sparse.h"
#include "cann_ops_sparseLt.h"
#include "log/log.h"

// Centralized log tag for all OP_LOGE calls in the sparseLt
// library. Previously the string literal "aclsparseLt" was hardcoded in every
// OP_LOGE call site (79 occurrences across 5 files). Defining it once here as
// a compile-time constant ensures consistency and makes future tag changes a
// single-line edit.
constexpr const char* kSparseLtLogTag = "aclsparseLt";

// Named constant for ALG_CONFIG_MAX_ID, shared between
// alg_set_attribute_host.cpp (set/get) and aclsparselt_host.cpp (validate).
// algConfigId 0 and 1 are valid (2 configs total). Platform-driven; future
// versions may expand — change this single constant instead of magic 0/1.
constexpr int32_t kAlgConfigMaxId = 2;

// aclsparseLtMatDescriptor / aclsparseLtMatmulDescriptor structs are defined
// in upstream's common/aclsparselt_mat_descriptor_internal.h and
// common/aclsparselt_matmul_descriptor_internal.h (included above). This
// avoids duplicate definitions and ensures binary compatibility with
// upstream's DenseDescriptorInit / StructuredDescriptorInit / MatmulDescriptorInit.

// ============================================================================
// Inline helper functions — compute PR-specific derived fields from the
// upstream-compatible matmul descriptor. These replace the former cached
// fields (transA, transB, isSparseA, hasStructuredSparsity, m, n, k) that
// were removed when the struct was aligned to upstream's layout.
// ============================================================================
inline bool md_transA(const aclsparseLtMatmulDescriptor* md) {
    return md->opA == ACL_SPARSE_OP_TRANSPOSE;
}
inline bool md_transB(const aclsparseLtMatmulDescriptor* md) {
    return md->opB == ACL_SPARSE_OP_TRANSPOSE;
}
inline bool md_isSparseA(const aclsparseLtMatmulDescriptor* md) {
    return md->matA->isStructured;
}
inline bool md_hasStructuredSparsity(const aclsparseLtMatmulDescriptor* md) {
    return md->matA->isStructured || md->matB->isStructured;
}
inline int32_t md_m(const aclsparseLtMatmulDescriptor* md) {
    return md_transA(md) ? static_cast<int32_t>(md->matA->cols)
                         : static_cast<int32_t>(md->matA->rows);
}
inline int32_t md_k(const aclsparseLtMatmulDescriptor* md) {
    return md_transA(md) ? static_cast<int32_t>(md->matA->rows)
                         : static_cast<int32_t>(md->matA->cols);
}
inline int32_t md_n(const aclsparseLtMatmulDescriptor* md) {
    return md_transB(md) ? static_cast<int32_t>(md->matB->rows)
                         : static_cast<int32_t>(md->matB->cols);
}

// ============================================================================
// Vector scaling flag encoding into matD->alignment high bits.
//
// alphaVectorScaling and betaVectorScaling are set via aclsparseLtMatmulDescSetAttribute
// (which receives the matmul descriptor, not the algSelection). The upstream-compatible
// matmul descriptor struct has no field for these, so we encode them into the high 2 bits
// of matD->alignment (uint32_t).
//
// Rationale: alignment is validated as a non-zero multiple of 16 (low 4 bits = 0) and
// practical values are small (16..4096), so bits 30-31 are always free. Encoding:
//   bit 31 (0x80000000): alphaVectorScaling
//   bit 30 (0x40000000): betaVectorScaling
//
// This replaces the former global side-map (get_matmul_desc_ext_map) which was keyed by
// raw descriptor pointer and suffered from address-reuse pollution: after Destroy+realloc
// the new descriptor could inherit the previous occupant's flags. With alignment encoding,
// FillMatDescriptor resets alignment to the user-provided value (clearing high bits) on
// every DescriptorInit, so stale flags are naturally cleared.
//
// NOTE: This is thread-unsafe (single-threaded C API library, like cuSPARSELt).
// ============================================================================
inline void encode_scaling_flags(aclsparseLtMatmulDescriptor* md, bool alphaVec, bool betaVec)
{
    if (md == nullptr || md->matD == nullptr) { return; }
    uint32_t& a = md->matD->alignment;
    a &= 0x3FFFFFFFu;  // clear top 2 bits
    if (alphaVec) { a |= 0x80000000u; }
    if (betaVec)  { a |= 0x40000000u; }
}

inline bool decode_alpha_scaling(const aclsparseLtMatmulDescriptor* md)
{
    return (md != nullptr && md->matD != nullptr) && ((md->matD->alignment & 0x80000000u) != 0u);
}

inline bool decode_beta_scaling(const aclsparseLtMatmulDescriptor* md)
{
    return (md != nullptr && md->matD != nullptr) && ((md->matD->alignment & 0x40000000u) != 0u);
}

// Safe handle -> internal struct casts. The external API exposes const
// descriptor types (aclsparseLtConstMatmulDescriptor_t etc.); internal code
// needs mutable access. to_xxx_internal accepts the const type and
// performs the const_cast centrally, so business code does not scatter
// const_cast calls.
inline aclsparseLtMatDescriptor_t to_mat_internal(aclsparseLtConstMatDescriptor_t d) {
    return const_cast<aclsparseLtMatDescriptor_t>(d);
}
inline aclsparseLtMatmulDescriptor_t to_matmul_internal(aclsparseLtConstMatmulDescriptor_t d) {
    return const_cast<aclsparseLtMatmulDescriptor_t>(d);
}

// ============================================================================
// Host-only helper functions (formerly anonymous-namespace helpers in the
// former host.cpp). Declared inline/static to avoid ODR issues
// when multiple host TUs include this header.
// ============================================================================

inline int32_t dtype_from_acl(aclDataType t)
{
    if (t == ACL_FLOAT) { return SPLT_DTYPE_FP32; }
    if (t == ACL_FLOAT16) { return SPLT_DTYPE_FP16; }
    // v2: BF16 / INT8 input dtypes; INT32 only used for matD output type (INT8 path).
    if (t == ACL_BF16) { return SPLT_DTYPE_BF16; }
    if (t == ACL_INT8) { return SPLT_DTYPE_INT8; }
    if (t == ACL_INT32) { return SPLT_DTYPE_INT32; }
    return SPLT_DTYPE_INVALID;
}

inline int64_t align_up(int64_t v, int64_t a)
{
    if (a == 0) {
        return v;
    }
    return ((v + a - 1) / a) * a;
}

// Resolve cube tiling (baseM, baseN, baseK) from dtype + algConfigId (design §6.2).
struct CubeTiling { int32_t baseM; int32_t baseN; int32_t baseK; };
inline CubeTiling resolve_cube_tiling(int32_t dataType, int32_t algConfigId)
{
    if (dataType == SPLT_DTYPE_FP32) {
        if (algConfigId == 1) { return {32, 128, 16}; }
        // baseK=64: fewer K-loop iterations reduce MTE2→MTE1 sync overhead
        // and improve L1 double-buffer utilization. Validated by the full
        // ltmatmul test suite (algConfigId 0/1 × all dtype paths).
        return {64, 64, 64};
    }
    // v2: BF16 reuses FP16 tiling (sizeof=2, C0=16).
    if (dataType == SPLT_DTYPE_BF16) {
        if (algConfigId == 1) { return {64, 256, 32}; }
        return {128, 128, 64};
    }
    // v2: INT8 tiling (C0=32, sizeof=1; L1 usage verified in design §5.4).
    if (dataType == SPLT_DTYPE_INT8) {
        if (algConfigId == 1) { return {64, 256, 32}; }
        return {128, 128, 64};
    }
    // FP16
    if (algConfigId == 1) { return {64, 256, 32}; }
    return {128, 128, 64};
}

// get_cube_core_num is the only consumer of
// tiling/platform/platform_ascendc.h. Moving the implementation to
// aclsparselt_host.cpp (which already includes the platform header) keeps
// this header lightweight — other host TUs (alg_set_attribute_host.cpp,
// prune_host.cpp) that include this header no longer transitively pull in
// the heavy platform_ascendc.h. Only the declaration lives here.
uint32_t get_cube_core_num();

// get_ub_size queries the platform UB capacity via
// PlatformAscendC::GetCoreMemSize(CoreMemType::UB). Implemented in
// aclsparselt_host.cpp (which includes platform_ascendc.h). Returns the
// per-core Unified Buffer size in bytes, or a hardcoded fallback if the
// platform API is unavailable.
uint64_t get_ub_size();

// Compute workspace byte offsets (design §5.4).
struct WsLayout {
    int64_t tilingOff;
    int64_t aPrunedOff;
    int64_t tempOff;
    int64_t totalBytes;
};

inline WsLayout compute_ws_layout(int32_t m, int32_t n, int32_t k, int32_t splitK,
                                  int32_t dataType, int32_t numBatches)
{
    // v2: four-branch elemSize (FP32=4, INT8=1, INT32=4, FP16/BF16=2).
    const int64_t elemSize =
        (dataType == SPLT_DTYPE_FP32) ? SPLT_FP32_BYTES :
        (dataType == SPLT_DTYPE_INT8) ? SPLT_INT8_BYTES :
        (dataType == SPLT_DTYPE_INT32) ? SPLT_FP32_BYTES : SPLT_HALF_BYTES;  // FP16/BF16
    // numBatches 至少为 1（防御性，fill_tiling_dims 保证 >= 1）
    const int64_t batches = (numBatches > 0) ? static_cast<int64_t>(numBatches) : 1;
    WsLayout w;
    w.tilingOff = SPLT_WS_HEADER_BYTES;
    w.aPrunedOff = align_up(w.tilingOff + sizeof(AclsparseltTilingData), SPLT_WS_ALIGN);
    w.tempOff = align_up(w.aPrunedOff + static_cast<int64_t>(m) * static_cast<int64_t>(k) * elemSize, SPLT_WS_ALIGN);
    // When splitK==1, the fused kernel path (SpltFusedMatmulCubeImpl +
    // SpltFusedEpilogueImpl) writes directly L0C->UB->GM, bypassing the temp
    // buffer entirely. Skip the temp_result allocation (splitK*m*n*4 bytes) to
    // avoid a large block of unused workspace memory. tempOff is still computed
    // (points past A_pruned) so td.tempResultOffset remains a valid offset, but
    // totalBytes stops at tempOff — the host never allocates or memsets temp.
    // The fused path never reads td.tempResultOffset, so this is safe.
    if (splitK == 1) {
        w.totalBytes = w.tempOff;
    } else {
        // temp buffer 须包含所有 batch 的部分和
        w.totalBytes = align_up(w.tempOff + batches * static_cast<int64_t>(splitK) * static_cast<int64_t>(m) *
                                static_cast<int64_t>(n) * SPLT_FP32_BYTES, SPLT_WS_ALIGN);
    }
    // A_pruned buffer is always allocated here because compute_ws_layout
    // runs at tiling compute time, before the caller invokes aclsparseLtMatmul (where
    // matA is provided). When the Matmul caller passes a non-null matA (already
    // pruned A, e.g. caller's d_out from SpMMAPrune), the host routes matA
    // directly to the kernel and the workspace A_pruned region is unused. We
    // cannot avoid this allocation at tiling compute time since matA availability is
    // unknown until Matmul. The wasted memory is m*k*sizeof(T), modest relative
    // to the temp buffer (splitK*m*n*4) that is already saved by D7 when
    // splitK==1.
    return w;
}

// Copy host TilingData to device workspace[tilingOff].
inline aclsparseStatus_t push_tiling_to_device(void* workspace, int64_t tilingOff,
                                               const AclsparseltTilingData& td)
{
    if (workspace == nullptr) {
        OP_LOGE(kSparseLtLogTag, "workspace is null when pushing tiling");
        return ACL_SPARSE_STATUS_INSUFFICIENT_RESOURCES;
    }
    void* devTiling = static_cast<uint8_t*>(workspace) + tilingOff;
    aclError ret = aclrtMemcpy(devTiling, sizeof(td), &td, sizeof(td),
                               ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != ACL_SUCCESS) {
        OP_LOGE(kSparseLtLogTag, "aclrtMemcpy tiling failed, aclErr=%d", (int)ret);
        return ACL_SPARSE_STATUS_EXECUTION_FAILED;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

inline void fill_cube_tiling_dims(AclsparseltTilingData& td,
                                  const aclsparseLtMatmulAlgSelection* asel,
                                  const CubeTiling& ct, int32_t dt)
{
    const auto* md = asel->matmulDescr;
    td.m = md_m(md);
    td.n = md_n(md);
    td.k = md_k(md);
    // matmul path: A_pruned is stored contiguously (ld == k).
    // Prune path sets td.ld = matA->ld separately in prune_host.cpp.
    td.ld = md_k(md);
    td.baseM = ct.baseM;
    td.baseN = ct.baseN;
    td.baseK = ct.baseK;
    // [REFACTOR-KLOOP] kL1Size: SPLT_KL1_BUDGET_BYTES/elemSize, aligned up to baseK.
    // v2: four-branch elemSize (FP32=4, INT8=1, FP16/BF16=2).
    const int32_t elemSize =
        (dt == SPLT_DTYPE_FP32) ? SPLT_FP32_BYTES :
        (dt == SPLT_DTYPE_INT8) ? SPLT_INT8_BYTES : SPLT_HALF_BYTES;  // FP16/BF16
    int32_t kL1Sz = SPLT_KL1_BUDGET_BYTES / elemSize;
    kL1Sz = ((kL1Sz + ct.baseK - 1) / ct.baseK) * ct.baseK;
    td.kL1Size = kL1Sz;
}

// splitK / splitKMode are passed in (queried via AlgGetAttribute by the
// caller fill_tiling_dims) rather than read from asel directly, so that the
// tiling path consumes alg attributes exclusively through the public API.
inline void fill_splitk_tiling_dims(AclsparseltTilingData& td,
                                     const aclsparseLtMatmulAlgSelection* asel,
                                     const CubeTiling& ct,
                                     int32_t splitK, int32_t splitKMode)
{
    const auto* md = asel->matmulDescr;
    const int32_t k = md_k(md);
    // [SPLIT_K_MODE] ONE_KERNEL absorbs splitK as 1 (fused GEMM+reduction);
    // TWO_KERNELS preserves user splitK (matmul+epilogue two-kernel path).
    const int32_t effectiveSplitK =
        (splitKMode == ACLSPARSELT_SPLIT_K_MODE_ONE_KERNEL) ? 1 : splitK;
    // Warn when ONE_KERNEL mode silently ignores splitK > 1.
    if (splitKMode == ACLSPARSELT_SPLIT_K_MODE_ONE_KERNEL && splitK > 1) {
        OP_LOGI(kSparseLtLogTag, "Tiling: splitKMode=ONE_KERNEL, splitK=%d ignored (effectiveSplitK=1)",
                splitK);
    }
    td.splitK = effectiveSplitK;
    // kSegLen = ceil(k / effectiveSplitK), aligned up to baseK for cube efficiency.
    // kernel clamps to td.k at segment tail.
    int32_t kSeg = (k + effectiveSplitK - 1) / effectiveSplitK;
    kSeg = ((kSeg + ct.baseK - 1) / ct.baseK) * ct.baseK;
    if (effectiveSplitK == 1 && kSeg > k) { kSeg = k; }
    td.kSegLen = kSeg;
}

inline void fill_sparse_tiling_dims(AclsparseltTilingData& td,
                                     const aclsparseLtMatmulDescriptor* md)
{
    const bool hasStructuredSparsity = md_hasStructuredSparsity(md);
    const bool isSparseA = md_isSparseA(md);
    const bool transA = md_transA(md);
    const bool transB = md_transB(md);
    // pruneAlongRow: dispatch by isSparseA to use the sparse
    // matrix's order/trans. v2: dense×dense sets pruneAlongRow=0.
    if (!hasStructuredSparsity) {
        td.pruneAlongRow = 0;
    } else {
        const bool isRowOrder = (isSparseA ? md->matA->order : md->matB->order)
                                == ACL_SPARSE_ORDER_ROW;
        td.pruneAlongRow = ((isSparseA ? transA : transB) != isRowOrder) ? 1 : 0;
    }
    td.sparseTrans = transA ? 1 : 0;
    // A-sparse: prune transposes A, matmul must not.
    // B-sparse: A is dense (not pruned), matmul must handle transA via sparseTrans.
    if (hasStructuredSparsity && isSparseA) {
        td.sparseTrans = 0;
    }
    // B-sparse: prune always outputs B_pruned in (k, n) row-major layout
    // (TransRowOrder path transposes during prune; AlongRow/AlongCol paths
    // read B in (k,n) and output (k,n) unchanged). The matmul must treat
    // B_pruned as NDExt(k, n) — no additional transpose.
    if (hasStructuredSparsity && !isSparseA) {
        td.transB = 0;
    } else {
        td.transB = transB ? 1 : 0;
    }
}

// Fill tiling dimension fields (shared by PlanInit).
// Derived fields are computed from md (matmulDescr) at call time, not cached in asel.
//
// algConfigId/splitK/splitKMode are queried via AlgGetAttribute API;
// other fields (m/n/k/transA etc.) are derived from md.
//
// The AlgGetAttribute calls are extracted into query_alg_attributes()
// [codecheck: oversized function].

struct AlgAttrValues {
    int32_t algConfigId = 0;
    int32_t splitK = 1;
    int32_t splitKMode = 0;
};

// Query algConfigId / splitK / splitKMode via the public AlgGetAttribute API.
// On failure, fall back to safe defaults and log the error.
inline AlgAttrValues query_alg_attributes(aclsparseLtConstHandle_t handle,
                                          const aclsparseLtMatmulAlgSelection* asel)
{
    AlgAttrValues v;
    aclsparseLtConstMatmulAlgSelection_t algSelForApi = asel;
    aclsparseStatus_t st = aclsparseLtMatmulAlgGetAttribute(
        handle, &algSelForApi, ACLSPARSELT_MATMUL_ALG_CONFIG_ID,
        &v.algConfigId, sizeof(int32_t));
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        OP_LOGE(kSparseLtLogTag, "fill_tiling_dims: get algConfigId failed (%d), using default 0",
                static_cast<int32_t>(st));
        v.algConfigId = 0;
    }
    st = aclsparseLtMatmulAlgGetAttribute(
        handle, &algSelForApi, ACLSPARSELT_MATMUL_SPLIT_K,
        &v.splitK, sizeof(int32_t));
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        OP_LOGE(kSparseLtLogTag, "fill_tiling_dims: get splitK failed (%d), using default 1",
                static_cast<int32_t>(st));
        v.splitK = 1;
    }
    st = aclsparseLtMatmulAlgGetAttribute(
        handle, &algSelForApi, ACLSPARSELT_MATMUL_SPLIT_K_MODE,
        &v.splitKMode, sizeof(int32_t));
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        OP_LOGE(kSparseLtLogTag, "fill_tiling_dims: get splitKMode failed (%d), using default 0",
                static_cast<int32_t>(st));
        v.splitKMode = 0;
    }
    return v;
}

/**
 * @brief 填充 bias + activation 默认值到 tiling data。
 * prepare_matmul_pointers 会用 descriptor 实际值覆盖这些默认值。
 */
inline void fill_bias_activation_defaults(AclsparseltTilingData& td)
{
    td.biasDevPtr = 0;           // 默认无 bias
    td.biasStride = 0;
    td.activationType = 0;      // 默认无 activation
    td.reluUpperBound = FLT_MAX;
    td.reluThreshold = 0.0f;
    td.geluScaling = 1.0f;
    td.biasChunkMode = 0;       // 默认全量加载（下方根据 UB 预算计算降级标志）
}

/**
 * @brief 计算 biasChunkMode 降级标志。
 *
 * 三向量全量加载 + tile buffer 超 UB 时降级为 per-chunk。
 * 仅融合路径（splitK==1）需要降级（非融合路径 CHUNK=256 固定小 buffer，不超限）。
 */
inline void compute_bias_chunk_mode(AclsparseltTilingData& td,
                                     const aclsparseLtMatmulDescriptor* md,
                                     int32_t dt, const CubeTiling& ct, int32_t m)
{
    const bool biasEnabled = (md->biasPointer != nullptr);
    const bool needVecScale = (td.alphaVectorScaling == 1 || td.betaVectorScaling == 1);
    if (biasEnabled && needVecScale && td.splitK == 1) {
        const uint64_t ubSize = get_ub_size();
        const uint64_t maxTileBytes =
            2 * static_cast<uint64_t>(ct.baseM) * static_cast<uint64_t>(ct.baseN) * sizeof(float)
            + 2048;  // accBuf + dFp32Buf + small bufs
        // biasVecBuf (FP32) + biasLoadBuf (FP16/BF16 only, not FP32/INT8)
        const bool needBiasLoadBuf = (dt != SPLT_DTYPE_FP32 && dt != SPLT_DTYPE_INT8);
        const uint64_t vecBytes =
            (td.alphaVectorScaling ? static_cast<uint64_t>(m) * 4 : 0) +
            (td.betaVectorScaling ? static_cast<uint64_t>(m) * 4 : 0) +
            (biasEnabled ? static_cast<uint64_t>(m) * 4 : 0) +
            (biasEnabled && needBiasLoadBuf
                ? static_cast<uint64_t>(m) * static_cast<uint64_t>(SPLT_HALF_BYTES)
                : 0);
        if (vecBytes + maxTileBytes > ubSize) {
            td.biasChunkMode = 1;
        }
    }
}

inline void fill_tiling_dims(AclsparseltTilingData& td,
                             aclsparseLtConstHandle_t handle,
                             const aclsparseLtMatmulDescriptor* md,
                             const aclsparseLtMatmulAlgSelection* asel,
                             const WsLayout& wsl, int32_t coreNum)
{
    const int32_t dt = dtype_from_acl(md->matA->valueType);
    const AlgAttrValues v = query_alg_attributes(handle, asel);
    const CubeTiling ct = resolve_cube_tiling(dt, v.algConfigId);
    fill_cube_tiling_dims(td, asel, ct, dt);
    fill_splitk_tiling_dims(td, asel, ct, v.splitK, v.splitKMode);
    td.algConfigId = v.algConfigId;
    td.dataType = dt;
    // v2: outDataType derived from matD->valueType (INT8 path: INT8 or INT32).
    td.outDataType = dtype_from_acl(md->matD->valueType);
    fill_sparse_tiling_dims(td, md);
    td.alpha = 1.0f;
    td.beta = 0.0f;
    td.alphaVectorScaling = decode_alpha_scaling(md) ? 1 : 0;
    td.betaVectorScaling = decode_beta_scaling(md) ? 1 : 0;
    td.alphaDevPtr = 0;
    td.betaDevPtr = 0;
    td.coreNum = static_cast<int32_t>(coreNum);
    const int32_t m = md_m(md);
    const int32_t n = md_n(md);
    // 从 mat descriptor 读取 batch 配置（SetAttribute 在 PlanInit 之前调用）
    const int32_t numBatches = (md->matA != nullptr && md->matA->numBatches > 0) ? md->matA->numBatches : 1;
    td.numBatches = numBatches;
    td.batchStrideA = (md->matA != nullptr) ? md->matA->batchStride : 0;
    td.batchStrideB = (md->matB != nullptr) ? md->matB->batchStride : 0;
    td.batchStrideC = (md->matC != nullptr) ? md->matC->batchStride : 0;
    td.batchStrideD = (md->matD != nullptr) ? md->matD->batchStride : 0;
    const int64_t totalMNTiles = static_cast<int64_t>((m + ct.baseM - 1) / ct.baseM) *
                                  static_cast<int64_t>((n + ct.baseN - 1) / ct.baseN);
    // usedCoreNum 基于 per-batch tile 数（不含 numBatches）。
    // batch 循环在 kernel 内部，每个 block 处理所有 batch 的同一组 tile。
    // 此前 totalTiles 乘以 numBatches 导致 usedCoreNum 被放大，多余 block 在
    // 每 batch 的 tile 循环中空转（rawBlockId >= totalTiles 直接 continue）。
    // 改为 per-batch tile 数后，核数与 kernel 侧 totalTiles 一致，无空转。
    const int64_t perBatchTotalTiles = totalMNTiles * static_cast<int64_t>(td.splitK);
    td.usedCoreNum = (perBatchTotalTiles < static_cast<int64_t>(coreNum))
                         ? static_cast<int32_t>(perBatchTotalTiles)
                         : static_cast<int32_t>(coreNum);
    if (td.usedCoreNum <= 0) { td.usedCoreNum = 1; }
    td.tilingOffset = wsl.tilingOff;
    td.aPrunedOffset = wsl.aPrunedOff;
    td.tempResultOffset = wsl.tempOff;
    fill_bias_activation_defaults(td);
    compute_bias_chunk_mode(td, md, dt, ct, m);
}

#endif // __CCE_AICORE__

#endif // ACLSPARSELT_INTERNAL_H
