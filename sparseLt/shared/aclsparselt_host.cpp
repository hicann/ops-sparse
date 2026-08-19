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
 * \file aclsparselt_host.cpp
 * \brief sparseLt shared host-side helpers (prune-only trimmed copy).
 *
 * Implements only the platform query helpers (get_cube_core_num / get_ub_size)
 * consumed by prune_host.cpp. All matmul / algSet / algGet / PlanInit /
 * Destroy extern "C" entry points are intentionally excluded — this fork
 * ships prune only.
 */

// Include order: C standard → C++ → CANN → local.
#include <cstdint>

#include "acl/acl.h"
#include "acl/acl_rt.h"
#include "log/log.h"

#ifndef __CCE_AICORE__
#include "tiling/platform/platform_ascendc.h"
#endif

#include "cann_ops_sparseLt.h"
#include "aclsparselt_handle_internal.h"
#include "shared/aclsparselt_internal.h"

// ============================================================================
// get_cube_core_num implementation — queries the platform API for the AIC
// core count, falling back to SPLT_CORE_NUM (32) when unavailable.
// ============================================================================
uint32_t get_cube_core_num()
{
    auto* plat = platform_ascendc::PlatformAscendCManager::GetInstance();
    if (plat == nullptr) { return SPLT_CORE_NUM; }
    uint32_t aic = plat->GetCoreNumAic();
    return (aic > 0) ? aic : static_cast<uint32_t>(SPLT_CORE_NUM);
}

// Query per-core UB capacity from the platform API instead of
// hardcoding 248KB. PlatformAscendC::GetCoreMemSize(CoreMemType::UB) returns
// the actual UB size for the current SoC. Falls back to 248 * 1024 (Ascend950
// UB capacity) when the platform API is unavailable (e.g. null instance or
// zero returned).
uint64_t get_ub_size()
{
    auto* plat = platform_ascendc::PlatformAscendCManager::GetInstance();
    if (plat == nullptr) { return 248 * 1024; }
    uint64_t ubSize = 0;
    plat->GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    return (ubSize > 0) ? ubSize : static_cast<uint64_t>(248 * 1024);
}
