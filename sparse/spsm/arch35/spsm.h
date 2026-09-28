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

#ifndef SPSM_H_
#define SPSM_H_
#include <array>
#include <cstdint>
#include "cann_ops_sparse.h"
#include "spsm_tiling_data.h"
#include "spsm_plan.h"

#define SPSM_ORDER_ROW 0
#define SPSM_ORDER_COL 1
#define SPSM_FILL_LOWER 0
#define SPSM_FILL_UPPER 1
#define SPSM_DIAG_NON_UNIT 0
#define SPSM_DIAG_UNIT 1

// Fixed-size host metadata; all matrix-dependent storage is in device workspace.
struct aclsparseSpSMDescr
{
    bool sized = false;
    std::array<uint64_t, 32> signature { };
    SpsmPlanTiling plan { };
    aclsparseHandle_t owner = nullptr;
    aclsparseConstSpMatDescr_t inputA = nullptr;
    aclsparseConstDnMatDescr_t inputB = nullptr;
    aclsparseDnMatDescr_t outputC = nullptr;
    const void* alpha = nullptr;
    const void* boundB = nullptr;
    void* boundC = nullptr;
    aclrtStream stream = nullptr;
    int32_t device = -1;
    aclsparsePointerMode_t pointerMode = ACL_SPARSE_POINTER_MODE_HOST;
    bool analyzed = false;
    void* buffer = nullptr;
    aclsparseOperation_t opA = ACL_SPARSE_OP_NON_TRANSPOSE;
    aclsparseOperation_t opB = ACL_SPARSE_OP_NON_TRANSPOSE;
    int64_t cachedBufferSize = 0;
    SpsmTilingData cachedTiling { };
};

inline const SpsmTilingData& GetSpsmCachedTiling(const aclsparseSpSMDescr* d)
{
    return d->cachedTiling;
}
#endif // SPSM_H_
