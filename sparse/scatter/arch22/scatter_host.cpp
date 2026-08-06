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

#include "cann_ops_sparse.h"
#include "aclsparse_host_utils.h"   // CHECK_ACL 宏
#include "aclsparse_descr_internal.h"
#include "aclsparse_handle_internal.h"
#include "scatter.h"
#include "scatter_kernel.h"

// ---------------------------------------------------------------------------
// Scatter execution  (cuSPARSE mode: no D2H, no dedup, no sort)
//
// Data path: raw device indices/values -> Tiling -> Kernel launch
// Behavior matches cuSPARSE: deterministic only if indices are distinct.
// ---------------------------------------------------------------------------

extern "C" aclsparseStatus_t aclsparseScatter(
    aclsparseHandle_t handle,
    aclsparseConstSpVecDescr_t vecX,
    aclsparseDnVecDescr_t vecY)
{
    if (handle == nullptr || vecX == nullptr || vecY == nullptr) {
        return ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR;
    }

    auto *handleInner  = reinterpret_cast<aclsparseContext *>(handle);
    auto *spVecInner   = ScatterToSpVecInner(vecX);
    auto *dnVecInner   = reinterpret_cast<aclsparseDnVecDescr *>(vecY);

    // 索引类型校验：kernel 按 int32_t 读取 indices，仅支持 ACL_SPARSE_INDEX_32I。
    // ACL_SPARSE_INDEX_64I 暂未支持（参考 spmv 的 idxType 校验模式）。
    if (spVecInner->idxType != ACL_SPARSE_INDEX_32I) {
        OP_LOGE("aclsparse", "aclsparseScatter: unsupported idxType %d (only ACL_SPARSE_INDEX_32I)",
                spVecInner->idxType);
        return ACL_SPARSE_STATUS_NOT_SUPPORTED;
    }

    int64_t nnz = spVecInner->nnz;
    int64_t N   = dnVecInner->nums;

    // nnz == 0: nothing to scatter, y unchanged
    if (nnz == 0) {
        return ACL_SPARSE_STATUS_SUCCESS;
    }

    // 入参安全校验（合并）：arch22 kernel 按 float 读取 values、ZERO-base 直接索引
    // yGm[targetIdx]、size 不得超 y 长度、indices/values 不得为空。任一不满足即拒绝，
    // 避免 kernel 侧越界/解引用空指针。放在 nnz==0 早退之后——空稀疏向量（nnz==0）
    // indices/values 可合法为 nullptr、size 无需校验，不应被本块拦截。
    // arch35 在 tiling 里带 idxBase/valueType 由 kernel 分发，支持多类型，故该校验仅限 arch22。
    if (spVecInner->valueType != ACL_FLOAT ||
        spVecInner->idxBase != ACL_SPARSE_INDEX_BASE_ZERO ||
        spVecInner->size > static_cast<uint64_t>(N) ||
        spVecInner->indices == nullptr ||
        spVecInner->values == nullptr) {
        OP_LOGE("aclsparse", "aclsparseScatter: invalid params (valueType=%d idxBase=%d size=%lu nums=%lu indices=%p values=%p)",
                spVecInner->valueType, spVecInner->idxBase, spVecInner->size,
                dnVecInner->nums, spVecInner->indices, spVecInner->values);
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }

    // Raw device pointers — data stays on device, no D2H round-trip
    void *dVal = spVecInner->values;
    void *dIdx = spVecInner->indices;
    void *dY   = dnVecInner->values;

    // -----------------------------------------------------------------------
    // Tiling: power-of-2 tile size for all kernel versions
    // -----------------------------------------------------------------------
    int32_t deviceId = 0;
    CHECK_ACL(aclrtGetDevice(&deviceId));

    int64_t coreCount = 1;
    CHECK_ACL(aclrtGetDeviceInfo(deviceId, ACL_DEV_ATTR_VECTOR_CORE_NUM, &coreCount));

    uint32_t availableCores = static_cast<uint32_t>(coreCount);
    uint32_t blockNum = (static_cast<uint32_t>(nnz) < availableCores)
                            ? static_cast<uint32_t>(nnz)
                            : availableCores;

    // Clamp to fixed-array capacity: coreNnzOffset/coreNnzCount in
    // ScatterTilingData are SCATTER_MAX_CORE_NUM-element arrays. Without this
    // guard, devices with >64 vector cores and nnz >64 would yield blockNum >64,
    // causing out-of-bounds writes on the host stack array and out-of-bounds
    // reads on kernel-side blockIdx >= 64.
    if (blockNum > SCATTER_MAX_CORE_NUM) {
        blockNum = SCATTER_MAX_CORE_NUM;
    }

    uint32_t baseCount = static_cast<uint32_t>(nnz) / blockNum;
    uint32_t remainder = static_cast<uint32_t>(nnz) % blockNum;
    uint32_t maxPerCore = baseCount + (remainder > 0 ? 1 : 0);

    // Tile size: largest power of 2 <= maxPerCore (min 8)
    uint32_t tileNn = SCATTER_TILE_NN_MAX;   // 4096
    while (tileNn > maxPerCore && tileNn > 8) {
        tileNn >>= 1;
    }

    ScatterTilingData tilingData{};
    tilingData.nnz      = static_cast<uint32_t>(nnz);
    tilingData.blockNum = blockNum;
    tilingData.tileNn        = tileNn;
    tilingData.tileNnAligned8 = ((tileNn + 7) / 8) * 8;
    tilingData.tileNnAligned4 = ((tileNn + 3) / 4) * 4;
    tilingData.yLen          = static_cast<uint64_t>(N) * sizeof(float);

    uint32_t off = 0;
    for (uint32_t c = 0; c < blockNum; c++) {
        tilingData.coreNnzOffset[c] = off;
        uint32_t count = baseCount + (c < remainder ? 1 : 0);
        tilingData.coreNnzCount[c] = count;
        off += count;
    }

    // -----------------------------------------------------------------------
    // Kernel launch with raw device pointers
    //
    // Tiling 随 kernel 启动参数一起下发（const 引用），Host 侧不再单独分配
    // device 内存存 tiling，对齐 arch35 scatter 与 spmv arch22 的实现模式。
    // -----------------------------------------------------------------------
    scatter_kernel_do(dVal, dIdx, dY, tilingData, blockNum, handleInner->stream);

    return ACL_SPARSE_STATUS_SUCCESS;
}
