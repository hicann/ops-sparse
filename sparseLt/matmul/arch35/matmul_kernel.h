/**
 * ----------------------------------------------------------------------------------------------------------
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ----------------------------------------------------------------------------------------------------------
 */

/*!
 * \file matmul_kernel.h
 * \brief Matmul / epilogue / fused kernel launcher declarations
 *        (ASC side defines, CXX side includes).
 *
 * host.cpp (CXX) includes this header and calls launchers as plain C functions.
 * kernel.cpp (ASC) defines the launchers and wraps the <<<>>> invocation.
 * host.cpp MUST NOT write <<<>>> directly.
 */

#ifndef SPLT_MATMUL_KERNEL_H
#define SPLT_MATMUL_KERNEL_H

#include <cstdint>

// Define GM_ADDR for the CXX (host) TU so the launcher
// declarations can use GM_ADDR uniformly. On the ASC (kernel) TU, GM_ADDR is
// already defined by the compiler — the #ifndef guard skips redefinition.
// This mirrors the spsm_kernel.h pattern and eliminates the (GM_ADDR) casts in
// <<<>>> calls that triggered the "cce_global attribute ignored" warning.
#ifndef GM_ADDR
#define GM_ADDR uint8_t *
#endif

#ifdef __cplusplus
extern "C" {
#endif

// Cube matmul: M(m,k) x N(k,n) -> temp (splitK*m*n, FP32).
extern void splt_matmul_kernel_launch(
    GM_ADDR aPrunedGm, GM_ADDR bGm, GM_ADDR tempGm,
    GM_ADDR tilingGm, int32_t dataType, uint32_t blockDim, void *stream);

// Epilogue: D = alpha * reduce(temp) + beta * C.
// v2: outDataType added — INT8 path selects INT8 (saturation Cast) vs INT32 (direct write).
extern void splt_epilogue_kernel_launch(
    GM_ADDR tempGm, GM_ADDR cGm, GM_ADDR dGm,
    GM_ADDR tilingGm, int32_t dataType, int32_t outDataType, uint32_t blockDim, void *stream);

// [OPT-P2] Fused matmul+epilogue kernel for splitK==1
// AIC: matmul -> Fixpipe L0C->UB -> CrossCore notify AIV
// AIV: CrossCore wait -> Vector(alpha*acc+beta*C+Cast) -> DataCopyPad UB->GM(D)
// v2: outDataType added — INT8 path selects INT8 (saturation Cast) vs INT32 (direct write).
extern void splt_fused_matmul_kernel_launch(
    GM_ADDR aPrunedGm, GM_ADDR bGm,
    GM_ADDR cGm, GM_ADDR dGm,
    GM_ADDR tilingGm, int32_t dataType, int32_t outDataType, uint32_t blockDim, void *stream);

#ifdef __cplusplus
}
#endif

#endif // SPLT_MATMUL_KERNEL_H
