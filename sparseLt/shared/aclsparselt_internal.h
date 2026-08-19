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

/*!
 * \file aclsparselt_internal.h
 * \brief sparseLt shared internal header (prune-only trimmed copy).
 *
 * Contains the TilingData struct + constants shared by host (CXX) and kernel
 * (ASC), and a small set of host-only helpers consumed exclusively by the
 * prune path (md_transA / md_isSparseA / dtype_from_acl / get_cube_core_num /
 * get_ub_size). Matmul / algSet / algGet helpers and structs are intentionally
 * excluded — this fork ships prune only.
 *
 * Included by both host.cpp (CXX) and kernel.cpp (ASC). The TilingData struct
 * and constants above the __CCE_AICORE__ guard are compiled by both; the
 * host-only helpers below the guard are skipped by the ASC compiler.
 */

#ifndef ACLSPARSELT_INTERNAL_H
#define ACLSPARSELT_INTERNAL_H


#include <cstdint>

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

// Element byte sizes by dtype (used in UB capacity calculations).
#define SPLT_FP32_BYTES 4
#define SPLT_HALF_BYTES 2   // FP16 / BF16
#define SPLT_INT8_BYTES 1
// UB 32-byte alignment boundary (DataCopyPad requirement).
#define SPLT_UB_ALIGN_BYTES 32

// Hardware constants (DAV-3510).
#define SPLT_CORE_NUM 32

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

// TilingData forwarded by value through the kernel launch args block.
// POD struct — full field set retained for struct layout stability.
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
    // K-dimension L1 tile size. Prune does not consume; retained for layout stability.
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
    // [TRANSPOSE] sparseTrans/transB flags: 1 = op(A)/op(B) is transpose.
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
} AclsparseltTilingData;


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
#include <acl/acl.h>
#include "cann_ops_sparse.h"
#include "cann_ops_sparseLt.h"
#include "log/log.h"

// Centralized log tag for all OP_LOGE calls in the sparseLt
// library. Previously the string literal "aclsparseLt" was hardcoded in every
// OP_LOGE call site. Defining it once here as a compile-time constant ensures
// consistency and makes future tag changes a single-line edit.
constexpr const char* kSparseLtLogTag = "aclsparseLt";

// aclsparseLtMatDescriptor / aclsparseLtMatmulDescriptor 定义复用自
// common/aclsparselt_mat_descriptor_internal.h 和
// common/aclsparselt_matmul_descriptor_internal.h（上方 include）。

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

// Safe handle -> internal struct casts. The external API exposes const
// descriptor types (aclsparseLtConstMatmulDescriptor_t etc.); internal code
// needs mutable access. to_xxx_internal accepts the const type and
// performs the const_cast centrally, so business code does not scatter
// const_cast calls.
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

// get_cube_core_num is the only consumer of
// tiling/platform/platform_ascendc.h. Moving the implementation to
// aclsparselt_host.cpp (which already includes the platform header) keeps
// this header lightweight — other host TUs (prune_host.cpp) that include this
// header no longer transitively pull in the heavy platform_ascendc.h. Only the
// declaration lives here.
uint32_t get_cube_core_num();

// get_ub_size queries the platform UB capacity via
// PlatformAscendC::GetCoreMemSize(CoreMemType::UB). Implemented in
// aclsparselt_host.cpp (which includes platform_ascendc.h). Returns the
// per-core Unified Buffer size in bytes, or a hardcoded fallback if the
// platform API is unavailable.
uint64_t get_ub_size();

#endif // __CCE_AICORE__

#endif // ACLSPARSELT_INTERNAL_H
