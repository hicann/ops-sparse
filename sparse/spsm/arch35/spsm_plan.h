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

#ifndef SPSM_PLAN_H_
#define SPSM_PLAN_H_

#include <cstdint>

// All offsets are bytes relative to the caller's device workspace. This POD is
// passed by value to kernels; no host-side matrix data is stored in the plan.
struct SpsmPlanTiling
{
    int32_t m = 0, n = 0, nnz = 0;
    int32_t lowWidth = 0; // RHS columns sharing the reusable rounding-tail buffer
    int32_t lowRows = 0; // Power-of-two dependency window, or m for a full panel
    int32_t format = 0; // CSR=0, CSC=1, COO=2
    int32_t base = 0, opA = 0, opB = 0, upper = 0, unit = 0, components = 1;
    int32_t orderB = 0, orderC = 0, deviceAlpha = 0, levels = 0, maxRowLen = 0;
    float alphaReal = 1.0f, alphaImag = 0.0f;
    int64_t ldb = 0, ldc = 0;
    int64_t row = 0, col = 0, perm = 0, val = 0, diag = 0;
    int64_t rowLevel = 0, levelPtr = 0, levelIdx = 0, scratch = 0;
    int64_t key0 = 0, key1 = 0, candidateDiag = 0, low = 0, reduce = 0;
    int64_t lowCapacity = 0;
};

enum class SpsmPhase : int32_t
{
    CHECK,
    DECODE,
    MERGE,
    CANONICAL,
    ANALYZE,
    SNAPSHOT,
    SNAPSHOT_ROOTS,
    PREPARE_PANEL,
    PREPARE_PANEL_ROOTS,
    SOLVE,
    SOLVE_LEVELS,
    SOLVE_DIAGONAL,
    SCATTER,
    CHECK_GENERAL,
    CHECK_DIAGONAL,
    COMMIT_GENERAL,
    COMMIT_DIAGONAL
};

struct SpsmDeviceSummary
{
    int32_t error = 0; // 1=invalid structure; 2=missing/zero diagonal
    int32_t levels = 0;
    int32_t maxRowLen = 0;
    int32_t maxDependencyDistance = 0;
    int32_t reserved[12] { };
};
static_assert(sizeof(SpsmDeviceSummary) == 64, "Analysis transfers only a fixed summary");

#endif // SPSM_PLAN_H_
