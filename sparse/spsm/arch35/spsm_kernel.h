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

#ifndef SPSM_KERNEL_H_
#define SPSM_KERNEL_H_

#include <cstdint>
#include "spsm_tiling_data.h"

#ifndef GM_ADDR
#define GM_ADDR uint8_t *
#endif

// Host-side kernel launch dispatchers (defined in spsm_kernel.cpp, C linkage).
// spsm_host.cpp 通过本头文件引入签名, 禁止以 extern 前向声明方式声明。
#ifdef __cplusplus
extern "C" {
#endif

// Solve kernel 启动器 (Solve 阶段调用)。
// 多核 (blockDim=numCores)，单 kernel 多 pass 逐 level 求解。异步 launch。
// TilingData 由 host by-value (const 引用) 传入, kernel 侧 by value 接收。
void spsm_solve_kernel_do(GM_ADDR csrRowOffsets, GM_ADDR csrColInd,
                          GM_ADDR csrValues, GM_ADDR matB, GM_ADDR matC,
                          GM_ADDR workspaceGM, const SpsmTilingData& tiling,
                          uint32_t blockDim, void *stream);

// vector transpose kernel 启动器 (COL<->ROW 转置, DataCopyPad 跨步搬运)。
// direction=0: COL->ROW (src[k*ld+i] -> dst[i*n+k])
// direction=1: ROW->COL (src[i*n+k] -> dst[k*ld+i])
// 多核 (blockDim=GetSpsmBlockDim), 每 block 用 DataCopyPad 跨步搬运 (vector SIMD), 全异步。
void spsm_transpose_kernel_do(GM_ADDR src, GM_ADDR dst, int32_t m, int32_t n,
                              int32_t ld, int32_t direction, uint32_t blockDim,
                              void *stream);

#ifdef __cplusplus
}
#endif

#endif // SPSM_KERNEL_H_
