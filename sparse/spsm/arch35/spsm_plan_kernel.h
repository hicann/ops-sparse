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

#ifndef SPSM_PLAN_KERNEL_H_
#define SPSM_PLAN_KERNEL_H_
#include "spsm_plan.h"
#ifndef GM_ADDR
#define GM_ADDR uint8_t*
#endif

void spsm_plan_kernel_do(GM_ADDR ptr, GM_ADDR idx, GM_ADDR values, GM_ADDR dense, GM_ADDR alpha, GM_ADDR workspace,
    const SpsmPlanTiling& tiling, SpsmPhase phase, int64_t width, uint32_t blocks, void* stream);
#endif // SPSM_PLAN_KERNEL_H_
