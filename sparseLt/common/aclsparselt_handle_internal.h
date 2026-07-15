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
 * \file aclsparselt_handle_internal.h
 * \brief ops-sparseLt handle 内部结构定义（不对外暴露）。
 */

#pragma once

#include <acl/acl.h>
#include <cstddef>
#include <cstdint>
#include "cann_ops_sparseLt.h"

/* ========== 版本信息 ========== */
#define ACLSPARSELT_VERSION_MAJOR 0
#define ACLSPARSELT_VERSION_MINOR 1
#define ACLSPARSELT_VERSION_PATCH 0

/**
 * @brief 将版本号 (major, minor, patch) 编码为整型。
 *
 * 编码规则：MAJOR * 10000 + MINOR * 100 + PATCH，
 * 例如 0.1.0 -> 100、1.0.0 -> 10000。
 */
#define ACLSPARSELT_MAKE_VERSION(major, minor, patch) \
    ((major) * 10000 + (minor) * 100 + (patch))

#define ACLSPARSELT_VERSION \
    ACLSPARSELT_MAKE_VERSION(ACLSPARSELT_VERSION_MAJOR, ACLSPARSELT_VERSION_MINOR, ACLSPARSELT_VERSION_PATCH)

/**
 * @brief ops-sparseLt handle 内部结构体
 *
 * 持有 sparseLt 库上下文（设备属性、系统信息等）。
 * 仅在实现文件中可见，对外部用户完全不可见。
 */
struct aclsparseLtContext {
    aclrtStream stream = nullptr;       ///< 当前 stream，用于 kernel 执行
    int32_t device_id = 0;              ///< 当前 NPU 设备 ID
    bool initialized = false;           ///< 是否已初始化
};
