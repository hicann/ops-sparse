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

#ifndef SPVV_KERNEL_H_
#define SPVV_KERNEL_H_

#include <cstdint>
#include <acl/acl_base.h>
#include "spvv_tiling_data.h"

// Host-side launcher for the spvv kernel. Defined in spvv_kernel.cpp, called
// from spvv_host.cpp. Tiling is passed by const reference (no host-side copy);
// the <<<>>> launch copies it into kernel parameter space by value.
void spvv_kernel_do(void* xIndices, void* xValues, void* y,
    void* output, const SpvvTilingData &tiling, uint32_t blockNum,
    aclDataType valueType, void* stream);

#endif // SPVV_KERNEL_H_
