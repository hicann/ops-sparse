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
 * \file matmul_host.cpp
 * \brief aclsparseLtMatmul host-side API implementation.
 *
 * Launches the cube matmul + epilogue (splitK>1) or fused matmul+epilogue
 * (splitK==1) device kernels via extern "C" launchers declared in
 * matmul/arch35/matmul_kernel.h. Fills alpha/beta into the tiling POD and
 * pushes it to device workspace[tilingOff] before launching.
 */

// Include order: C standard → C++ → CANN → local.
#include <cstdint>

#include "acl/acl.h"
#include "acl/acl_rt.h"
#include "log/log.h"
#include "cann_ops_sparseLt.h"
#include "aclsparselt_handle_internal.h"
#include "shared/aclsparselt_internal.h"
#include "matmul/arch35/matmul_kernel.h"

// ============================================================================
// Matmul — launch cube matmul + epilogue. Fills alpha/beta into tiling.
// Split aclsparseLtMatmul into validate_pointer_alignment +
// validate_matmul_params + prepare_matmul_pointers + launch_matmul_kernels to
// reduce cyclomatic complexity (was 35, R5 limit 20) and NBNC (was 80, R7
// limit 50). The validation logic, error codes, log messages and kernel launch
// order are unchanged. The public extern "C" signature is preserved.
// ============================================================================

// Bundled resolved state for the matmul kernel launch: produced by
// prepare_matmul_pointers and consumed by launch_matmul_kernels.
struct MatmulLaunchArgs {
    AclsparseltTilingData td;   // copy of baseTD with alpha/beta filled
    void* aPruned;
    void* bMat;
    void* temp;
    void* cMat;
    void* dMat;
    void* tilingGm;
    int32_t dt;
};

// validate_pointer_alignment — 16-byte alignment check for all data pointers.
static aclsparseStatus_t validate_pointer_alignment(const void* workspace,
                                                    const void* matA, const void* matB,
                                                    const void* matC, const void* matD)
{
    // Validate 16-byte alignment of data pointers (NVIDIA
    // requires all pointers aligned to 16 bytes).
    auto checkAlign = [](const void* p) -> bool {
        return (reinterpret_cast<uintptr_t>(p) % 16) == 0;
    };
    if (!checkAlign(workspace) ||
        (matA != nullptr && !checkAlign(matA)) ||
        (matB != nullptr && !checkAlign(matB)) ||
        (matC != nullptr && !checkAlign(matC)) ||
        (matD != nullptr && !checkAlign(matD))) {
        OP_LOGE(kSparseLtLogTag, "Matmul: data pointer not 16-byte aligned");
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

// validate_matmul_params — all null/alignment/transpose checks. outMd/outBaseTD
// return the plan internals extracted for downstream prepare_matmul_pointers.
static aclsparseStatus_t validate_matmul_params(
    const aclsparseLtHandle_t* handle,
    aclsparseLtConstMatmulPlan_t* plan,
    aclrtStream* streams, int32_t numStreams,
    const void* workspace,
    const void* matA, const void* matB, const void* matC, const void* matD,
    aclsparseLtMatmulDescriptor*& outMd,
    AclsparseltTilingData*& outBaseTD,
    aclrtStream& outStream)
{
    // Return HANDLE_IS_NULLPTR for null handle, consistent
    // with all other APIs (was INVALID_VALUE before).
    if (handle == nullptr || *handle == nullptr) {
        OP_LOGE(kSparseLtLogTag, "Matmul: handle is null");
        return ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR;
    }
    if (plan == nullptr || *plan == nullptr) {
        OP_LOGE(kSparseLtLogTag, "Matmul: plan is null");
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    // Validate plan internal pointers — a partially
    // initialized or corrupted plan would otherwise cause null dereference.
    auto* md = (*plan)->matmulDescr;
    auto* baseTd = (*plan)->tilingData;
    if (md == nullptr || baseTd == nullptr) {
        OP_LOGE(kSparseLtLogTag, "Matmul: plan internals null (matmulDescr/tilingData)");
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    // Validate streams: the streams pointer must be non-null with numStreams > 0
    // (multi-stream not yet supported, only streams[0] is used). streams[0]
    // itself may be nullptr — the ACL runtime treats 0/nullptr as the default
    // stream, which is valid for all aclrtMemcpy / kernel-launch APIs.
    // The previous non-null check on streams[0] incorrectly
    // rejected the default stream, causing every CSV case that passes stream=0
    // (the test framework's default stream) to fail with INVALID_VALUE.
    if (streams == nullptr || numStreams <= 0) {
        OP_LOGE(kSparseLtLogTag, "Matmul: streams is null or numStreams<=0");
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    outStream = streams[0];
    if (workspace == nullptr) {
        OP_LOGE(kSparseLtLogTag, "Matmul: workspace is null");
        return ACL_SPARSE_STATUS_INSUFFICIENT_RESOURCES;
    }
    aclsparseStatus_t st = validate_pointer_alignment(workspace, matA, matB, matC, matD);
    if (st != ACL_SPARSE_STATUS_SUCCESS) { return st; }
    // [TRANSPOSE] transA and transB are now supported end-to-end:
    //   - transA is handled by the prune kernel (reads physical (k,m) layout,
    //     outputs A_pruned as (m,k) row-major). The matmul kernel reads A_pruned
    //     unchanged (always (m,k)).
    //   - transB is passed to the matmul kernel via td.transB; the kernel
    //     declares GM B as (n,k) and transposes during CopyGM2L1 (NDExt→ZN).
    outMd = md;
    outBaseTD = baseTd;
    return ACL_SPARSE_STATUS_SUCCESS;
}

// prepare_matmul_pointers — tiling copy with alpha/beta fill, push to device,
// and pointer resolution (matA fallback, bMat/cMat/dMat fallback to descriptor
// values). The fallback order and the null check after fallback are preserved
// exactly (constraint: the null check after fallback is critical).
static aclsparseStatus_t prepare_matmul_pointers(
    const aclsparseLtMatmulDescriptor* md,
    const AclsparseltTilingData* baseTd,
    const void* alpha, const void* beta,
    const void* matA, const void* matB, const void* matC, void* matD,
    void* workspace,
    MatmulLaunchArgs& args)
{
    // build a tiling copy with alpha/beta filled in.
    args.td = *baseTd;
    if (baseTd->alphaVectorScaling) {
        args.td.alphaVectorScaling = 1;
        args.td.alphaDevPtr = reinterpret_cast<uint64_t>(alpha);
        args.td.alpha = 1.0f;
    } else {
        args.td.alphaVectorScaling = 0;
        args.td.alphaDevPtr = 0;
        args.td.alpha = (alpha != nullptr) ? *static_cast<const float*>(alpha) : 1.0f;
    }
    if (baseTd->betaVectorScaling) {
        args.td.betaVectorScaling = 1;
        args.td.betaDevPtr = reinterpret_cast<uint64_t>(beta);
        args.td.beta = 0.0f;
    } else {
        args.td.betaVectorScaling = 0;
        args.td.betaDevPtr = 0;
        args.td.beta = (beta != nullptr) ? *static_cast<const float*>(beta) : 0.0f;
    }
    args.dt = baseTd->dataType;

    aclsparseStatus_t st = push_tiling_to_device(workspace, args.td.tilingOffset, args.td);
    if (st != ACL_SPARSE_STATUS_SUCCESS) { return st; }

    // Honor matA per cuSPARSELt Matmul contract: when matA is
    // non-null, read the (already pruned) A from matA (e.g. caller's d_out from
    // SpMMAPrune); only fall back to workspace[aPrunedOffset] when matA is null.
    // Workspace is still required for the temp result (splitK>1) and tiling.
    //
    // matA==null fallback usage prerequisite:
    // The workspace[aPrunedOffset] region is NOT auto-populated by the plan or
    // matmul host. It is only filled when the user previously called
    // aclsparseLtSpMMAPrune with d_out pointing into workspace at exactly
    // aPrunedOffset. If the user calls Matmul with matA==null without first
    // populating that workspace region (via SpMMAPrune or equivalent), the
    // matmul kernel will read uninitialized device memory, producing undefined
    // results. There is no aPrunedValid flag in the current design; the caller
    // bears responsibility for ensuring the workspace A_pruned region is valid
    // before invoking Matmul with matA==null. The tested path always passes
    // matA non-null (the pruned d_out from SpMMAPrune), so this fallback exists
    // for API-completeness and is not exercised by current tests.
    //
    // Defensive runtime check on the matA==null fallback path.
    // Although workspace is validated non-null earlier in validate_matmul_params,
    // add an explicit guard here so the fallback path is self-contained and
    // safe against future refactoring that might move or relax the earlier
    // check. If matA is null and workspace is also null, the aPruned region
    // cannot possibly be valid (no buffer to read from), so return
    // INVALID_VALUE instead of dereferencing null. When workspace is non-null,
    // keep the current fallback behavior (caller is responsible for having
    // populated the aPruned region via SpMMAPrune).
    if (matA == nullptr) {
        // v2: dense×dense path (hasStructuredSparsity==false) — matA is the
        // original dense A and must be passed explicitly (no workspace fallback,
        // since there is no prune front-end to populate workspace[aPrunedOffset]).
        if (!md_hasStructuredSparsity(md)) {
            OP_LOGE(kSparseLtLogTag, "Matmul: matA is null (dense×dense requires explicit dense A)");
            return ACL_SPARSE_STATUS_INVALID_VALUE;
        }
        // B-sparse: matA is the dense matrix and must be passed
        // explicitly; no workspace fallback (unlike A-sparse where A_pruned
        // can live in workspace[aPrunedOffset]).
        if (!md_isSparseA(md) || workspace == nullptr) {
            OP_LOGE(kSparseLtLogTag, "Matmul: matA is null (B-sparse requires explicit dense A; A-sparse requires workspace fallback)");
            return ACL_SPARSE_STATUS_INVALID_VALUE;
        }
        args.aPruned = static_cast<uint8_t*>(workspace) + args.td.aPrunedOffset;
    } else {
        args.aPruned = const_cast<void*>(matA);
    }
    // B/C/D must be passed explicitly to Matmul (aligned with cuSPARSELt, which
    // requires d_A/d_B/d_C/d_D pointers). Descriptors no longer bind values,
    // so there is no fallback — null pointers return INVALID_VALUE.
    args.bMat = const_cast<void*>(matB);
    // temp only used when splitK > 1; fused path (splitK==1) bypasses
    // temp entirely (tempResultOffset points to workspace end, never dereferenced).
    args.temp = (args.td.splitK > 1)
        ? static_cast<uint8_t*>(workspace) + args.td.tempResultOffset
        : nullptr;
    args.cMat = const_cast<void*>(matC);
    args.dMat = matD;
    if (args.bMat == nullptr || args.cMat == nullptr || args.dMat == nullptr) {
        OP_LOGE(kSparseLtLogTag, "Matmul: matB/matC/matD is null (must be passed explicitly)");
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    args.tilingGm = static_cast<uint8_t*>(workspace) + args.td.tilingOffset;
    return ACL_SPARSE_STATUS_SUCCESS;
}

// launch_matmul_kernels — fused (splitK==1) vs splitK>1 (aclrtMemsetAsync ->
// matmul -> epilogue) kernel launch. Launch order is preserved exactly
// (constraint: splitK>1 path memset -> matmul -> epilogue order is critical).
// v2: epilogue/fused launchers take outDataType (drives INT8 INT8-vs-INT32 dispatch).
static aclsparseStatus_t launch_matmul_kernels(const MatmulLaunchArgs& args, aclrtStream stream)
{
    // Defense-in-depth: reject unsupported dtype before reaching
    // the launcher. An explicit guard here ensures an illegal dtype cannot
    // reach the kernel dispatch switch.
    if (args.dt != SPLT_DTYPE_FP32 && args.dt != SPLT_DTYPE_FP16 &&
        args.dt != SPLT_DTYPE_BF16 && args.dt != SPLT_DTYPE_INT8) {
        OP_LOGE(kSparseLtLogTag, "Matmul: invalid dataType=%d (only FP32/FP16/BF16/INT8)", (int)args.dt);
        return ACL_SPARSE_STATUS_NOT_SUPPORTED;
    }
    if (args.td.splitK == 1) {
        // [OPT-P2] Fused path: single __mix__(1,2) kernel.
        // AIC: matmul -> Fixpipe L0C->UB -> CrossCore notify AIV
        // AIV: Vector(alpha*acc+beta*C+Cast) -> DataCopyPad UB->GM(D)
        splt_fused_matmul_kernel_launch(reinterpret_cast<GM_ADDR>(args.aPruned),
                                        reinterpret_cast<GM_ADDR>(args.bMat),
                                        reinterpret_cast<GM_ADDR>(args.cMat),
                                        reinterpret_cast<GM_ADDR>(args.dMat),
                                        reinterpret_cast<GM_ADDR>(args.tilingGm),
                                        args.dt, args.td.outDataType,
                                        static_cast<uint32_t>(args.td.usedCoreNum), stream);
        return ACL_SPARSE_STATUS_SUCCESS;
    }
    // Slow path: two-segment (cube->temp->epilogue reduce) for splitK>1.
    {
        // Use size_t for the temp buffer byte count to avoid
        // int32/int64 overflow on large matrices (m*n*splitK*sizeof(float)).
        const size_t tempBytes = static_cast<size_t>(args.td.splitK) *
                                 static_cast<size_t>(args.td.m) *
                                 static_cast<size_t>(args.td.n) * SPLT_FP32_BYTES;
        // Guard against unreasonable allocation (cap at 8 GiB temp buffer).
        constexpr size_t SPLT_TEMP_BYTES_CAP = static_cast<size_t>(8) * 1024 * 1024 * 1024;
        if (tempBytes > SPLT_TEMP_BYTES_CAP) {
            OP_LOGE(kSparseLtLogTag, "Matmul: temp buffer overflow, tempBytes=%zu > cap=%zu",
                    tempBytes, SPLT_TEMP_BYTES_CAP);
            return ACL_SPARSE_STATUS_INVALID_VALUE;
        }
        aclError memsetRet = aclrtMemsetAsync(args.temp, tempBytes, 0,
                                               tempBytes, stream);
        if (memsetRet != ACL_SUCCESS) {
            OP_LOGE(kSparseLtLogTag, "Matmul: aclrtMemsetAsync temp failed, aclErr=%d", (int)memsetRet);
            return ACL_SPARSE_STATUS_EXECUTION_FAILED;
        }
    }
    splt_matmul_kernel_launch(reinterpret_cast<GM_ADDR>(args.aPruned),
                              reinterpret_cast<GM_ADDR>(args.bMat),
                              reinterpret_cast<GM_ADDR>(args.temp),
                              reinterpret_cast<GM_ADDR>(args.tilingGm), args.dt,
                              static_cast<uint32_t>(args.td.usedCoreNum), stream);
    splt_epilogue_kernel_launch(reinterpret_cast<GM_ADDR>(args.temp),
                                reinterpret_cast<GM_ADDR>(args.cMat),
                                reinterpret_cast<GM_ADDR>(args.dMat),
                                reinterpret_cast<GM_ADDR>(args.tilingGm), args.dt,
                                args.td.outDataType,
                                static_cast<uint32_t>(args.td.usedCoreNum), stream);
    return ACL_SPARSE_STATUS_SUCCESS;
}

extern "C" aclsparseStatus_t aclsparseLtMatmul(
    const aclsparseLtHandle_t* handle,
    aclsparseLtConstMatmulPlan_t* plan,
    const void* alpha,
    const void* matA, const void* matB,
    const void* beta,
    const void* matC, void* matD,
    void* workspace,
    aclrtStream* streams, int32_t numStreams)
{
    aclsparseLtMatmulDescriptor* md = nullptr;
    AclsparseltTilingData* baseTd = nullptr;
    aclrtStream stream = nullptr;
    aclsparseStatus_t st = validate_matmul_params(handle, plan, streams, numStreams, workspace,
                                                   matA, matB, matC, matD, md, baseTd, stream);
    if (st != ACL_SPARSE_STATUS_SUCCESS) { return st; }

    MatmulLaunchArgs args;
    st = prepare_matmul_pointers(md, baseTd, alpha, beta, matA, matB, matC, matD, workspace, args);
    if (st != ACL_SPARSE_STATUS_SUCCESS) { return st; }

    return launch_matmul_kernels(args, stream);
}
