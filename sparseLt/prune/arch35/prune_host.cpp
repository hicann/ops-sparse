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
 * \file prune_host.cpp
 * \brief LtMatmulSpMMAPrune host-side API implementation.
 *
 * Launches the prune kernel: A (d_in) -> A_pruned (d_out).
 *
 * Tiling is passed by value through the kernel launch args block
 * (the AclsparseltTilingData POD struct is forwarded by the launcher into the
 * <<<>>> invocation). This aligns Prune with the cuSPARSELt SpMMAPrune
 * properties:
 *   - D5 "requires no extra storage": no aclrtMalloc'd tilingGm buffer.
 *   - D4 "supports asynchronous execution with respect to stream": no
 *     aclrtSynchronizeStream; the API returns immediately after launching the
 *     kernel on the stream (the args block is managed by the ACL runtime for
 *     the lifetime of the launched task, so freeing a host-owned buffer is not
 *     a concern).
 * The matmul path conveys tiling through the user-provided workspace; Prune has
 * no workspace parameter (matching the NVIDIA signature), so the args block is
 * the natural channel for the small (~84 B) tiling struct.
 */

// Include order: C standard → C++ → CANN → local.
#include <cstdint>

#include "acl/acl_rt.h"
#include "log/log.h"
#include "cann_ops_sparseLt.h"
#include "shared/aclsparselt_internal.h"
#include "prune/arch35/prune_kernel.h"

// ============================================================================
// SpMMAPrune — launch prune kernel: A (d_in) -> A_pruned (d_out).
//
// Split into three focused helpers:
//   1. validate_prune_params  — null/enum checks + core param derivation
//   2. resolve_prune_buffers  — UB capacity + buffer pointer/alignment checks
//   3. compute_prune_tiling   — AclsparseltTilingData struct construction
// The main function is a thin orchestration: validate → resolve → tile → launch.
// ============================================================================

// ----------------------------------------------------------------------------
// validate_prune_params — validate API inputs and derive core parameters.
// On success, outputs md / dt / m / k / pruneAlongRow / pruneAlgOut.
// ----------------------------------------------------------------------------
static aclsparseStatus_t derive_bsparse_params(
    aclsparseLtMatmulDescriptor_t md, int32_t& dt,
    int32_t& m, int32_t& k, int32_t& pruneAlongRow)
{
    // B-sparse: B is the (k, n) structured matrix; prune operates on B.
    // Map B's (rows=k, cols=n) to the kernel's (m, k) naming convention.
    dt = dtype_from_acl(md->matB->valueType);
    if (dt == SPLT_DTYPE_INVALID) {
        OP_LOGE(kSparseLtLogTag, "Prune: unsupported dtype (only FP32/FP16/BF16/INT8)");
        return ACL_SPARSE_STATUS_NOT_SUPPORTED;
    }
    m = md_k(md);   // B rows = logical k
    k = md_n(md);   // B cols = logical n
    const bool isRowOrder = (md->matB->order == ACL_SPARSE_ORDER_ROW);
    pruneAlongRow = (md_transB(md) != isRowOrder) ? 1 : 0;
    return ACL_SPARSE_STATUS_SUCCESS;
}

static aclsparseStatus_t validate_prune_params(
    aclsparseLtConstHandle_t handle, aclsparseLtConstMatmulDescriptor_t* matmulDescr,
    aclsparseLtPruneAlg_t pruneAlg, aclrtStream stream,
    aclsparseLtMatmulDescriptor_t& md, int32_t& dt,
    int32_t& m, int32_t& k, int32_t& pruneAlongRow, int32_t& pruneAlgOut)
{
    if (handle == nullptr || *handle == nullptr) {
        OP_LOGE(kSparseLtLogTag, "Prune: handle is null");
        return ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR;
    }
    if (matmulDescr == nullptr || *matmulDescr == nullptr) {
        OP_LOGE(kSparseLtLogTag, "Prune: matmulDescr is null");
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    // stream 可为 nullptr（ACL 默认流），不做非空校验。
    if (pruneAlg != ACLSPARSELT_PRUNE_SPMMA_STRIP && pruneAlg != ACLSPARSELT_PRUNE_SPMMA_TILE) {
        OP_LOGE(kSparseLtLogTag, "Prune: unsupported pruneAlg %d",
                static_cast<int>(pruneAlg));
        return ACL_SPARSE_STATUS_NOT_SUPPORTED;
    }
    pruneAlgOut = static_cast<int32_t>(pruneAlg);


    md = to_matmul_internal(*matmulDescr);
    if (md->matA == nullptr || md->matB == nullptr) {
        OP_LOGE(kSparseLtLogTag, "Prune: matA/matB is null");
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }

    // Derive prune parameters from the matmul descriptor (no plan needed).
    // isSparseA: A structured -> prune A (existing path).
    //            B structured -> prune B (matmulDescr implicit detection).
    if (md_isSparseA(md)) {
        dt = dtype_from_acl(md->matA->valueType);
        if (dt == SPLT_DTYPE_INVALID) {
            OP_LOGE(kSparseLtLogTag, "Prune: unsupported dtype (only FP32/FP16/BF16/INT8)");
            return ACL_SPARSE_STATUS_NOT_SUPPORTED;
        }
        m = md_m(md);
        k = md_k(md);
        const bool isRowOrder = (md->matA->order == ACL_SPARSE_ORDER_ROW);
        pruneAlongRow = (md_transA(md) != isRowOrder) ? 1 : 0;
    } else {
        return derive_bsparse_params(md, dt, m, k, pruneAlongRow);
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

static aclsparseStatus_t check_ub_capacity(
    aclsparseLtMatmulDescriptor_t md, int32_t dt,
    int32_t m, int32_t k, int32_t pruneAlongRow, int32_t pruneAlg)
{
    const int64_t elemSize =
        (dt == SPLT_DTYPE_FP32) ? SPLT_FP32_BYTES :
        (dt == SPLT_DTYPE_INT8) ? SPLT_INT8_BYTES : SPLT_HALF_BYTES;
    const bool isTile = (pruneAlg == ACLSPARSELT_PRUNE_SPMMA_TILE);
    const bool sparseTrans = md_isSparseA(md) ? md_transA(md) : md_transB(md);
    const bool isTransRowPath = (pruneAlongRow == 0 && sparseTrans);
    const int32_t rowDim = isTransRowPath ? m : k;
    const int64_t rowBytes = static_cast<int64_t>(rowDim) * elemSize;
    const int64_t SPLT_UB_BYTES = static_cast<int64_t>(get_ub_size());
    const int32_t tileSize = isTile
        ? ((dt == SPLT_DTYPE_FP32) ? SPLT_TILE_SIZE_FP32 : SPLT_TILE_SIZE_FP16)
        : 1;
    const int64_t alignedRowBytes = (rowBytes + (SPLT_UB_ALIGN_BYTES - 1)) /
                                    SPLT_UB_ALIGN_BYTES * SPLT_UB_ALIGN_BYTES;
    const int64_t dataBytes = (pruneAlongRow != 0)
        ? alignedRowBytes * static_cast<int64_t>(tileSize)
        : alignedRowBytes * static_cast<int64_t>(SPLT_PRUNE_ROW_TILE);
    const bool hasVecBuf = (pruneAlongRow != 0) || (isTransRowPath && !isTile);
    const int64_t vecBufBytes = hasVecBuf
        ? 2 * static_cast<int64_t>(rowDim + 64) * static_cast<int64_t>(sizeof(float))
        : 0;
    const int64_t transExtraBytes = isTransRowPath
        ? static_cast<int64_t>(SPLT_PRUNE_ROW_TILE) * elemSize
          + static_cast<int64_t>(SPLT_PRUNE_ROW_TILE) *
            static_cast<int64_t>((elemSize == SPLT_INT8_BYTES) ? 32 : 16) * elemSize
        : 0;
    const int64_t ubBytes = dataBytes + vecBufBytes + transExtraBytes;
    if (ubBytes > SPLT_UB_BYTES) {
        OP_LOGE(kSparseLtLogTag, "Prune: %s=%d exceeds UB capacity (ubBytes=%lld > %lld, alongRow=%d)",
                isTransRowPath ? "m" : "k", rowDim,
                static_cast<long long>(ubBytes), static_cast<long long>(SPLT_UB_BYTES),
                pruneAlongRow);
        return ACL_SPARSE_STATUS_NOT_SUPPORTED;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

// ----------------------------------------------------------------------------
// resolve_prune_buffers — validate UB capacity and resolve data pointers.
// On success, outputs aIn (source) / aPruned (destination).
// ----------------------------------------------------------------------------
static aclsparseStatus_t resolve_prune_buffers(
    const void* d_in, void* d_out, aclsparseLtMatmulDescriptor_t md,
    int32_t dt, int32_t m, int32_t k, int32_t pruneAlongRow, int32_t pruneAlg,
    const void*& aIn, void*& aPruned)
{
    aclsparseStatus_t ubSt = check_ub_capacity(md, dt, m, k, pruneAlongRow, pruneAlg);
    if (ubSt != ACL_SPARSE_STATUS_SUCCESS) { return ubSt; }

    if (d_in == nullptr) {
        OP_LOGE(kSparseLtLogTag, "Prune: d_in is null (must be passed explicitly)");
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    aIn = d_in;
    if (d_out == nullptr) {
        OP_LOGE(kSparseLtLogTag, "Prune: d_out is null (no workspace fallback)");
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    // 转置路径（sparseTrans=1, pruneAlongRow=0）不支持 in-place：输出布局 (m,k) 与
    // 输入布局 (k,m) 不同，多核并行写入会覆盖后续 block 的读取区域。
    const bool sparseTrans = md_isSparseA(md) ? md_transA(md) : md_transB(md);
    const bool isTransRowPath = (pruneAlongRow == 0 && sparseTrans);
    if (isTransRowPath && d_in == d_out) {
        OP_LOGE(kSparseLtLogTag, "Prune: in-place (d_in == d_out) not supported for transpose path");
        return ACL_SPARSE_STATUS_NOT_SUPPORTED;
    }
    if ((reinterpret_cast<uintptr_t>(aIn) % 16) != 0 ||
        (reinterpret_cast<uintptr_t>(d_out) % 16) != 0) {
        OP_LOGE(kSparseLtLogTag, "Prune: d_in/d_out not 16-byte aligned");
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    aPruned = d_out;
    return ACL_SPARSE_STATUS_SUCCESS;
}

// ----------------------------------------------------------------------------
// compute_prune_tiling — build the AclsparseltTilingData struct for the kernel.
// The prune kernel consumes m, k, pruneAlongRow, pruneAlg from tiling.
// Other fields are zeroed — they are irrelevant to the prune path.
// ----------------------------------------------------------------------------
static AclsparseltTilingData compute_prune_tiling(
    aclsparseLtMatmulDescriptor_t md, int32_t dt, int32_t m,
    int32_t k, int32_t pruneAlongRow, int32_t pruneAlg)
{
    AclsparseltTilingData td{};
    td.m = m;
    td.k = k;
    // Pass leading dimension to kernel so row offsets use row*ld
    // instead of row*k. When ld > k (padded matrix), this prevents incorrect
    // memory access from the 2nd row onward. md->matA->ld is int64_t; truncate
    // to int32_t (same width as m/k — matrices fit in int32_t per existing code).
    // ld from the structured matrix: A-sparse -> matA->ld; B-sparse -> matB->ld.
    const int64_t sparseLd = md_isSparseA(md) ? md->matA->ld : md->matB->ld;
    td.ld = static_cast<int32_t>(sparseLd);
    // The minimum ld depends on which kernel path runs:
    //   pruneAlongRow=0, sparseTrans=1 → SpltPruneTransRowOrder: physical
    //     matrix is (k, m) row-major, each physical row has m elements → ld ≥ m.
    //   All other paths: physical matrix has rows of k elements → ld ≥ k.
    const bool sparseTrans = md_isSparseA(md) ? md_transA(md) : md_transB(md);
    const bool isTransRowPath = (pruneAlongRow == 0 && sparseTrans);
    const int32_t minLd = isTransRowPath ? td.m : td.k;
    if (td.ld < minLd) { td.ld = minLd; }
    td.pruneAlongRow = pruneAlongRow;
    td.pruneAlg = pruneAlg;
    td.sparseTrans = sparseTrans ? 1 : 0;
    td.dataType = dt;
    td.coreNum = static_cast<int32_t>(get_cube_core_num());
    // [TILE] TILE alongRow uses TS rows/block (vs 1 for STRIP alongRow).
    const int32_t tileSize = (pruneAlg == ACLSPARSELT_PRUNE_SPMMA_TILE)
        ? ((dt == SPLT_DTYPE_FP32) ? SPLT_TILE_SIZE_FP32 : SPLT_TILE_SIZE_FP16)
        : 1;
    // For alongCol paths, physical row count = sparseTrans ? k : m.
    // When sparseTrans=1, the physical matrix is (k, m) row-major, so k is the
    // physical row count (m is logical row count, used for alongRow paths).
    const int32_t pruneTiles = (pruneAlongRow != 0)
        ? ((m + tileSize - 1) / tileSize)
        : ((td.sparseTrans ? k : m) + SPLT_PRUNE_ROW_TILE - 1) / SPLT_PRUNE_ROW_TILE;
    td.usedCoreNum = (pruneTiles < static_cast<int32_t>(get_cube_core_num()))
        ? pruneTiles
        : static_cast<int32_t>(get_cube_core_num());
    if (td.usedCoreNum <= 0) { td.usedCoreNum = 1; }
    return td;
}

// ============================================================================
// Public API entry point.
// ============================================================================
extern "C" aclsparseStatus_t aclsparseLtSpMMAPrune(
    aclsparseLtConstHandle_t handle,
    aclsparseLtConstMatmulDescriptor_t* matmulDescr,
    const void* d_in,
    void* d_out,
    aclsparseLtPruneAlg_t pruneAlg,
    aclrtStream stream)
{
    aclsparseLtMatmulDescriptor_t md = nullptr;
    int32_t dt = 0;
    int32_t m = 0;
    int32_t k = 0;
    int32_t pruneAlongRow = 0;
    int32_t pruneAlgInt = 0;
    aclsparseStatus_t st = validate_prune_params(handle, matmulDescr, pruneAlg, stream,
                                                   md, dt, m, k, pruneAlongRow, pruneAlgInt);
    if (st != ACL_SPARSE_STATUS_SUCCESS) { return st; }

    const void* aIn = nullptr;
    void* aPruned = nullptr;
    st = resolve_prune_buffers(d_in, d_out, md, dt, m, k, pruneAlongRow, pruneAlgInt, aIn, aPruned);
    if (st != ACL_SPARSE_STATUS_SUCCESS) { return st; }

    AclsparseltTilingData td = compute_prune_tiling(md, dt, m, k, pruneAlongRow, pruneAlgInt);

    // Launch the prune kernel with tiling passed by value through
    // the kernel launch args block. No aclrtMalloc'd tilingGm buffer (D5: no
    // extra storage) and no aclrtSynchronizeStream (D4: async execution). The
    // ACL runtime owns the args block for the lifetime of the launched task,
    // so the host may return immediately after enqueueing the kernel.
    splt_prune_kernel_launch(reinterpret_cast<GM_ADDR>(const_cast<void*>(aIn)),
                             reinterpret_cast<GM_ADDR>(aPruned), &td,
                             td.dataType, static_cast<uint32_t>(td.usedCoreNum), stream);

    return ACL_SPARSE_STATUS_SUCCESS;
}
