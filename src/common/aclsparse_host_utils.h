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
 * \file aclsparse_host_utils.h
 * \brief Host-side utilities for ops-sparse.
 */

#pragma once

#include <cstdint>
#include <limits>
#include <type_traits>

#include "log/log.h"
#include "tiling/platform/platform_ascendc.h"

#ifndef CHECK_RET
#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)
#endif

template <typename R = uint32_t, typename T1, typename T2>
static inline R CeilDiv(T1 a, T2 b)
{
    static_assert(
        std::is_arithmetic<T1>::value && std::is_arithmetic<T2>::value,
        "CeilDiv arguments must be arithmetic types");
    static_assert(std::is_unsigned<R>::value,
        "CeilDiv return type R must be unsigned to ensure safe overflow detection");
    auto ra = static_cast<R>(a);
    auto rb = static_cast<R>(b);
    if (rb == 0) {
        return std::numeric_limits<R>::max();
    }
    if (ra + rb - 1 < ra) {
        return std::numeric_limits<R>::max();
    }
    return (ra + rb - 1) / rb;
}

static inline uint32_t GetAivCoreCount()
{
    auto *platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    if (platform == nullptr) {
        OP_LOGE("aclsparse", "PlatformAscendCManager is null, return 0");
        return 0;
    }
    return platform->GetCoreNumAiv();
}

static inline uint32_t GetAicCoreCount()
{
    auto *platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    if (platform == nullptr) {
        OP_LOGE("aclsparse", "PlatformAscendCManager is null, return 0");
        return 0;
    }
    return platform->GetCoreNumAic();
}
