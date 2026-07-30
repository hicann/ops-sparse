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

/*!
 * \file gather_host.cpp
 * \brief aclsparseGather Host 侧实现：参数校验 + Kernel launch。
 *
 * Gather:  X.values[i] = Y[X.indices[i] - idxBase]  for i = 0 .. nnz-1.
 */

#include <algorithm>
#include <cstdint>
#include "acl/acl_base_rt.h"
#include "cann_ops_sparse.h"
#include "aclsparse_handle_internal.h"
#include "aclsparse_descr_internal.h"
#include "aclsparse_host_utils.h"
#include "gather_tiling_data.h"
#include "gather_kernel.h"

namespace {

bool supportedType(aclDataType t) { return t == ACL_FLOAT || t == ACL_FLOAT16 || t == ACL_BF16 || t == ACL_DOUBLE; }

// ===========================================================================
// 参数校验
// ===========================================================================
static aclsparseStatus_t ValidateGatherParams(
    aclsparseHandle_t handle, aclsparseConstDnVecDescr_t vecY, aclsparseSpVecDescr_t vecX)
{
    if (handle == nullptr) {
        OP_LOGE("aclsparseGather", "handle is nullptr");
        return ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR;
    }
    if (vecY == nullptr) {
        OP_LOGE("aclsparseGather", "vecY is nullptr");
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (vecX == nullptr) {
        OP_LOGE("aclsparseGather", "vecX is nullptr");
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (vecX->valueType != vecY->valueType) {
        OP_LOGE("aclsparseGather", "value type mismatch: X=%d, Y=%d", vecX->valueType, vecY->valueType);
        return ACL_SPARSE_STATUS_NOT_SUPPORTED;
    }
    if (!supportedType(vecY->valueType)) {
        OP_LOGE("aclsparseGather", "unsupported value type: %d", vecY->valueType);
        return ACL_SPARSE_STATUS_NOT_SUPPORTED;
    }
    if (vecY->nums < vecX->size) {
        OP_LOGE("aclsparseGather", "vecY->nums(%lu) < vecX->size(%lu)", vecY->nums, vecX->size);
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

// ===========================================================================
// Kernel launch
// ===========================================================================
static aclsparseStatus_t LaunchGatherKernel(
    aclsparseHandle_t handle, aclsparseConstDnVecDescr_t vecY, aclsparseSpVecDescr_t vecX)
{
    GatherTilingData tiling{};
    tiling.nnz = vecX->nnz;
    tiling.idxBase = vecX->idxBase;
    tiling.valType = vecY->valueType;
    tiling.idxType = vecX->idxType;

    uint32_t maxCoreNum = GetAivCoreCount();
    CHECK_RET(maxCoreNum > 0, OP_LOGE("aclsparseGather", "GetAivCoreCount returned 0");
              return ACL_SPARSE_STATUS_INTERNAL_ERROR);

    tiling.numBlocks = std::min<uint64_t>(CeilDiv<uint64_t>(tiling.nnz, kGatherMaxThreadsPerBlock), maxCoreNum);

    OP_LOGD(
        "aclsparseGather", "Tiling: nnz=%ld, idxBase=%d, valType=%d, idxType=%d, numBlocks=%u", tiling.nnz,
        tiling.idxBase, tiling.valType, tiling.idxType, tiling.numBlocks);

    gather_kernel_do(
        reinterpret_cast<GM_ADDR>(vecX->indices), reinterpret_cast<GM_ADDR>(const_cast<void*>(vecY->values)),
        reinterpret_cast<GM_ADDR>(vecX->values), tiling, handle->stream);

    return ACL_SPARSE_STATUS_SUCCESS;
}

} // namespace

// ============================================================================
// Public API
// ============================================================================
extern "C" {

aclsparseStatus_t aclsparseGather(aclsparseHandle_t handle, aclsparseConstDnVecDescr_t vecY, aclsparseSpVecDescr_t vecX)
{
    aclsparseStatus_t st = ValidateGatherParams(handle, vecY, vecX);
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        return st;
    }

    if (vecX->nnz == 0) {
        OP_LOGD("aclsparseGather", "nnz=0, nothing to gather");
        return ACL_SPARSE_STATUS_SUCCESS;
    }

    return LaunchGatherKernel(handle, vecY, vecX);
}

} // extern "C"
