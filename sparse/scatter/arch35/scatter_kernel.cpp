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
 * \file scatter_kernel.cpp
 * \brief aclsparseScatter SIMT kernel 实现（仅 arch35/DAV-3510 可用）。
 *
 * 三层结构：
 *   Layer 1: __simt_vf__ 计算函数 ScatterSimtCompute（grid-stride loop 覆盖 nnz 维度）
 *   Layer 2: __global__ 调度器（按 idxType/valType 分发到 6 个模板特化）
 *   Layer 3: kernel_do 启动器（<<<>>> 异步 launch）
 *
 * 16 字节对齐约束：X.indices / X.values / Y.values 需 16 字节对齐，
 * 由描述符创建侧保证，kernel 侧不校验。
 */

#include <cstdint>
#include "kernel_operator.h"      // KERNEL_TASK_TYPE_DEFAULT, half, bfloat16_t
#include "simt_api/asc_simt.h"   // __simt_vf__, asc_vf_call, threadIdx/blockDim 等
#include "scatter_kernel.h"

/// Y[X.indices[i] - idxBase] = X.values[i]，grid-stride loop。
/// idxBase 统一用减法（0 无偏移，1 减 1）避免分支；重复索引 last-write-wins；
/// 越界索引行为未定义（与 cuSPARSE 一致）。
template <typename ValT, typename IdxT>
__simt_vf__ __aicore__ __launch_bounds__(kScatterMaxThreadsPerBlock) inline void
ScatterSimtCompute(
    uint64_t nnz,
    __gm__ const IdxT *indices,
    __gm__ const ValT *values,
    __gm__ ValT *yVec,
    IdxT idxBase)
{
    const uint64_t init = static_cast<uint64_t>(blockIdx.x) * static_cast<uint64_t>(blockDim.x)
                          + static_cast<uint64_t>(threadIdx.x);
    const uint64_t step = static_cast<uint64_t>(blockDim.x) * static_cast<uint64_t>(gridDim.x);
    for (uint64_t i = init; i < nnz; i += step) {
        const IdxT idx = indices[i] - idxBase;
        yVec[idx] = values[i];
    }
}

template <typename IdxT>
__aicore__ inline void DispatchScatterByValType(
    const ScatterTilingData &tiling,
    GM_ADDR gmIndices, GM_ADDR gmValues, GM_ADDR gmYVec)
{
    if (tiling.valType == SCATTER_VAL_FP32) {
        asc_vf_call<ScatterSimtCompute<float, IdxT>>(
            dim3{kScatterMaxThreadsPerBlock},
            tiling.nnz,
            (__gm__ const IdxT *)gmIndices,
            (__gm__ const float *)gmValues,
            (__gm__ float *)gmYVec,
            IdxT(tiling.idxBase));
    } else if (tiling.valType == SCATTER_VAL_FP16) {
        asc_vf_call<ScatterSimtCompute<half, IdxT>>(
            dim3{kScatterMaxThreadsPerBlock},
            tiling.nnz,
            (__gm__ const IdxT *)gmIndices,
            (__gm__ const half *)gmValues,
            (__gm__ half *)gmYVec,
            IdxT(tiling.idxBase));
    } else {  // SCATTER_VAL_BF16
        asc_vf_call<ScatterSimtCompute<bfloat16_t, IdxT>>(
            dim3{kScatterMaxThreadsPerBlock},
            tiling.nnz,
            (__gm__ const IdxT *)gmIndices,
            (__gm__ const bfloat16_t *)gmValues,
            (__gm__ bfloat16_t *)gmYVec,
            IdxT(tiling.idxBase));
    }
}

extern "C" __global__ __aicore__ void scatter_kernel(
    GM_ADDR gmIndices, GM_ADDR gmValues, GM_ADDR gmYVec,
    const ScatterTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

    if (tiling.idxType == SCATTER_IDX_I32) {
        DispatchScatterByValType<int32_t>(tiling, gmIndices, gmValues, gmYVec);
    } else {  // SCATTER_IDX_I64
        DispatchScatterByValType<int64_t>(tiling, gmIndices, gmValues, gmYVec);
    }
}

extern "C" void scatter_kernel_do(
    GM_ADDR indices, GM_ADDR values, GM_ADDR yVec,
    const ScatterTilingData &tiling, uint32_t numBlocks, void *stream)
{
    scatter_kernel<<<numBlocks, nullptr, stream>>>(indices, values, yVec, tiling);
}
