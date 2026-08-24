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
 * \file alg_get_attribute_host.cpp
 * \brief aclsparseLtMatmulAlgGetAttribute host-side API implementation.
 *
 * This API queries the aclsparseLtMatmulAlgSelection struct
 * (algConfigId / splitK / searchIterations / splitKBuffers). It is a pure
 * host-side metadata operation and does not launch any kernel.
 */

// Include order: C standard → C++ → CANN → local.
#include <cstdint>

#include "log/log.h"
#include "cann_ops_sparseLt.h"
#include "aclsparselt_handle_internal.h"
#include "shared/aclsparselt_internal.h"

// ============================================================================
// ★ aclsparseLtMatmulAlgGetAttribute — query current alg selection attributes.
// [OPT-P3] Aligned with cuSPARSELt cusparseLtMatmulAlgGetAttribute.
// ============================================================================
// Each switch-case body extracted into a static helper so the
// main function's NBNC stays under the codecheck cap. Output logic, error codes
// and log messages are unchanged.

static aclsparseStatus_t get_alg_config_id(aclsparseLtConstMatmulAlgSelection_t algSel,
                                           void* attrValue, size_t attrValueSize)
{
    if (attrValueSize != sizeof(int32_t)) {
        OP_LOGE(kSparseLtLogTag, "ALG_CONFIG_ID: size mismatch, got %zu", attrValueSize);
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    *static_cast<int32_t*>(attrValue) = algSel->algConfigId;
    return ACL_SPARSE_STATUS_SUCCESS;
}

static aclsparseStatus_t get_alg_config_max_id(void* attrValue, size_t attrValueSize)
{
    if (attrValueSize != sizeof(int32_t)) {
        OP_LOGE(kSparseLtLogTag, "ALG_CONFIG_MAX_ID: size mismatch, got %zu", attrValueSize);
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    *static_cast<int32_t*>(attrValue) = kAlgConfigMaxId;
    return ACL_SPARSE_STATUS_SUCCESS;
}

static aclsparseStatus_t get_search_iterations(aclsparseLtConstMatmulAlgSelection_t algSel,
                                               void* attrValue, size_t attrValueSize)
{
    if (attrValueSize != sizeof(int32_t)) {
        OP_LOGE(kSparseLtLogTag, "SEARCH_ITERATIONS: size mismatch, got %zu", attrValueSize);
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    *static_cast<int32_t*>(attrValue) = algSel->searchIterations;
    return ACL_SPARSE_STATUS_SUCCESS;
}

static aclsparseStatus_t get_split_k(aclsparseLtConstMatmulAlgSelection_t algSel,
                                      void* attrValue, size_t attrValueSize)
{
    if (attrValueSize != sizeof(int32_t)) {
        OP_LOGE(kSparseLtLogTag, "SPLIT_K: size mismatch, got %zu", attrValueSize);
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    *static_cast<int32_t*>(attrValue) = algSel->splitK;
    return ACL_SPARSE_STATUS_SUCCESS;
}

static aclsparseStatus_t get_split_k_mode(aclsparseLtConstMatmulAlgSelection_t algSel,
                                          void* attrValue, size_t attrValueSize)
{
    // [SPLIT_K_MODE] Return the stored splitKMode value.
    if (attrValueSize != sizeof(int32_t)) {
        OP_LOGE(kSparseLtLogTag, "SPLIT_K_MODE: size mismatch, got %zu", attrValueSize);
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    *static_cast<int32_t*>(attrValue) = algSel->splitKMode;
    return ACL_SPARSE_STATUS_SUCCESS;
}

static aclsparseStatus_t get_split_k_buffers(aclsparseLtConstMatmulAlgSelection_t algSel,
                                             void* attrValue, size_t attrValueSize)
{
    // [SPLIT_K_BUFFERS] Stored value; TWO_KERNELS uses splitK-1 buffers, ONE_KERNEL uses no temp.
    if (attrValueSize != sizeof(int32_t)) {
        OP_LOGE(kSparseLtLogTag, "SPLIT_K_BUFFERS: size mismatch, got %zu", attrValueSize);
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    *static_cast<int32_t*>(attrValue) = algSel->splitKBuffers;
    return ACL_SPARSE_STATUS_SUCCESS;
}

extern "C" aclsparseStatus_t aclsparseLtMatmulAlgGetAttribute(
    const aclsparseLtHandle_t* handle,
    aclsparseLtConstMatmulAlgSelection_t* algSelection,
    aclsparseLtMatmulAlgAttribute_t attr,
    void* attrValue,
    size_t attrValueSize)
{
    if (handle == nullptr || *handle == nullptr) {
        OP_LOGE(kSparseLtLogTag, "AlgGetAttribute: handle is null");
        return ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR;
    }
    if (algSelection == nullptr || *algSelection == nullptr || attrValue == nullptr) {
        OP_LOGE(kSparseLtLogTag, "AlgGetAttribute: algSelection/attrValue is null");
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    // get_* helpers accept aclsparseLtConstMatmulAlgSelection_t directly,
    // so no const_cast is needed in the read-only AlgGetAttribute path.
    switch (attr) {
        case ACLSPARSELT_MATMUL_ALG_CONFIG_ID:
            return get_alg_config_id(*algSelection, attrValue, attrValueSize);
        case ACLSPARSELT_MATMUL_ALG_CONFIG_MAX_ID:
            return get_alg_config_max_id(attrValue, attrValueSize);
        case ACLSPARSELT_MATMUL_SEARCH_ITERATIONS:
            return get_search_iterations(*algSelection, attrValue, attrValueSize);
        case ACLSPARSELT_MATMUL_SPLIT_K:
            return get_split_k(*algSelection, attrValue, attrValueSize);
        case ACLSPARSELT_MATMUL_SPLIT_K_MODE:
            return get_split_k_mode(*algSelection, attrValue, attrValueSize);
        case ACLSPARSELT_MATMUL_SPLIT_K_BUFFERS:
            return get_split_k_buffers(*algSelection, attrValue, attrValueSize);
        default:
            OP_LOGE(kSparseLtLogTag, "AlgGetAttribute: unsupported attr %d", static_cast<int32_t>(attr));
            return ACL_SPARSE_STATUS_NOT_SUPPORTED;
    }
}
