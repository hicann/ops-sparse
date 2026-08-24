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
 * \file alg_set_attribute_host.cpp
 * \brief aclsparseLtMatmulAlgSetAttribute host-side API implementation.
 *
 * This API mutates the aclsparseLtMatmulAlgSelection struct
 * (algConfigId / splitK / searchIterations / splitKBuffers). It is a pure
 * host-side metadata operation and does not launch any kernel.
 *
 * aclsparseLtMatmulAlgGetAttribute (the query companion) lives in
 * sparseLt/alg_get_attribute/arch35/alg_get_attribute_host.cpp.
 */

// Include order: C standard → C++ → CANN → local.
#include <cstdint>

#include "log/log.h"
#include "cann_ops_sparseLt.h"
#include "aclsparselt_handle_internal.h"
#include "shared/aclsparselt_internal.h"


// ============================================================================
// ★ aclsparseLtMatmulAlgSetAttribute — the namesake API of this operator.
// ============================================================================
// Each switch-case body extracted into a static helper so the
// main function's cyclomatic complexity and NBNC stay under the codecheck cap.

static aclsparseStatus_t set_alg_config_id(aclsparseLtMatmulAlgSelection_t algSel,
                                           const void* attrValue, size_t attrValueSize)
{
    if (attrValueSize != sizeof(int32_t)) {
        OP_LOGE(kSparseLtLogTag, "ALG_CONFIG_ID: size mismatch, got %zu", attrValueSize);
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    int32_t v = *static_cast<const int32_t*>(attrValue);
    if (v < 0 || v >= kAlgConfigMaxId) {
        OP_LOGE(kSparseLtLogTag, "ALG_CONFIG_ID: invalid value %d (expect 0/1)", v);
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    algSel->algConfigId = v;
    return ACL_SPARSE_STATUS_SUCCESS;
}

static aclsparseStatus_t set_split_k(aclsparseLtMatmulAlgSelection_t algSel,
                                     const void* attrValue, size_t attrValueSize)
{
    if (attrValueSize != sizeof(int32_t)) {
        OP_LOGE(kSparseLtLogTag, "SPLIT_K: size mismatch, got %zu", attrValueSize);
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    int32_t v = *static_cast<const int32_t*>(attrValue);
    if (v < 1) {
        OP_LOGE(kSparseLtLogTag, "SPLIT_K: invalid value %d (must be >= 1)", v);
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    int32_t k = md_k(algSel->matmulDescr);
    if (k > 0 && v > k) {
        OP_LOGE(kSparseLtLogTag, "SPLIT_K: invalid value %d (must be <= K=%d)",
                v, k);
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    algSel->splitK = v;
    return ACL_SPARSE_STATUS_SUCCESS;
}

static aclsparseStatus_t set_search_iterations(aclsparseLtMatmulAlgSelection_t algSel,
                                               const void* attrValue, size_t attrValueSize)
{
    if (attrValueSize != sizeof(int32_t)) {
        OP_LOGE(kSparseLtLogTag, "SEARCH_ITERATIONS: size mismatch");
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    int32_t v = *static_cast<const int32_t*>(attrValue);
    if (v <= 0) {
        OP_LOGE(kSparseLtLogTag, "SEARCH_ITERATIONS: invalid value %d (must be > 0)", v);
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    algSel->searchIterations = v;
    return ACL_SPARSE_STATUS_SUCCESS;
}

static aclsparseStatus_t set_split_k_mode(aclsparseLtMatmulAlgSelection_t algSel,
                                          const void* attrValue, size_t attrValueSize)
{
    // [SPLIT_K_MODE] splitKMode 现已参与 dispatch:
    //   ONE_KERNEL(0)   -> effectiveSplitK=1, fused __mix__(1,2) kernel (K-loop 全长)
    //   TWO_KERNELS(1)  -> 保留 splitK, matmul + epilogue 两 kernel 路径
    // 见 fill_tiling_dims 的 effectiveSplitK 计算.
    if (attrValueSize != sizeof(int32_t)) {
        OP_LOGE(kSparseLtLogTag, "SPLIT_K_MODE: size mismatch, got %zu", attrValueSize);
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    int32_t v = *static_cast<const int32_t*>(attrValue);
    if (v < 0 || v > static_cast<int32_t>(ACLSPARSELT_SPLIT_K_MODE_TWO_KERNELS)) {
        OP_LOGE(kSparseLtLogTag, "SPLIT_K_MODE: invalid value %d", v);
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    algSel->splitKMode = v;
    return ACL_SPARSE_STATUS_SUCCESS;
}

static aclsparseStatus_t set_split_k_buffers(aclsparseLtMatmulAlgSelection_t algSel,
                                             const void* attrValue, size_t attrValueSize)
{
    // [SPLIT_K_BUFFERS] 下界校验在此处; 上界 [0, splitK-1] 也在此处校验
    // (当 splitK 已被设置时). 调用方在 tiling 计算时复核.
    // Added upper bound check: splitKBuffers >= splitK is rejected
    // at SetAttribute time when splitK is already set (L2_11 expects this).
    // When splitK is still the default (1), only buffers >= 1 is rejected,
    // which is correct since [0, 0] is the only valid range for splitK=1.
    if (attrValueSize != sizeof(int32_t)) {
        OP_LOGE(kSparseLtLogTag, "SPLIT_K_BUFFERS: size mismatch, got %zu", attrValueSize);
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    int32_t v = *static_cast<const int32_t*>(attrValue);
    if (v < 0) {
        OP_LOGE(kSparseLtLogTag, "SPLIT_K_BUFFERS: invalid value %d (must be >= 0)", v);
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (v >= algSel->splitK) {
        OP_LOGE(kSparseLtLogTag, "SPLIT_K_BUFFERS: invalid value %d (must be < splitK=%d)",
                v, algSel->splitK);
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    algSel->splitKBuffers = v;
    return ACL_SPARSE_STATUS_SUCCESS;
}

extern "C" aclsparseStatus_t aclsparseLtMatmulAlgSetAttribute(
    const aclsparseLtHandle_t* handle,
    aclsparseLtMatmulAlgSelection_t* algSelection,
    aclsparseLtMatmulAlgAttribute_t attr,
    const void* attrValue,
    size_t attrValueSize)
{
    if (handle == nullptr || *handle == nullptr) {
        OP_LOGE(kSparseLtLogTag, "AlgSetAttribute: handle is null");
        return ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR;
    }
    if (algSelection == nullptr || *algSelection == nullptr || attrValue == nullptr) {
        OP_LOGE(kSparseLtLogTag, "AlgSetAttribute: algSelection/attrValue is null");
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    auto* algSel = *algSelection;
    switch (attr) {
        case ACLSPARSELT_MATMUL_ALG_CONFIG_ID:
            return set_alg_config_id(algSel, attrValue, attrValueSize);
        case ACLSPARSELT_MATMUL_SPLIT_K:
            return set_split_k(algSel, attrValue, attrValueSize);
        case ACLSPARSELT_MATMUL_SEARCH_ITERATIONS:
            return set_search_iterations(algSel, attrValue, attrValueSize);
        case ACLSPARSELT_MATMUL_SPLIT_K_MODE:
            return set_split_k_mode(algSel, attrValue, attrValueSize);
        case ACLSPARSELT_MATMUL_ALG_CONFIG_MAX_ID:
            // [OPT-P3] read-only attribute: cannot be set.
            OP_LOGE(kSparseLtLogTag, "ALG_CONFIG_MAX_ID is read-only");
            return ACL_SPARSE_STATUS_NOT_SUPPORTED;
        case ACLSPARSELT_MATMUL_SPLIT_K_BUFFERS:
            return set_split_k_buffers(algSel, attrValue, attrValueSize);
        default:
            OP_LOGE(kSparseLtLogTag, "AlgSetAttribute: unsupported attr %d", static_cast<int32_t>(attr));
            return ACL_SPARSE_STATUS_NOT_SUPPORTED;
    }
}
