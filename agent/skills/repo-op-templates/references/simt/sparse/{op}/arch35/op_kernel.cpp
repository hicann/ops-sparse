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

// TEMPLATE: SIMT Kernel 实现（Sparse 算子，仅 arch35 可用）
// 标准三层结构：
//   Layer 1: __simt_vf__ 计算函数（线程级并行，grid-stride loop）
//   Layer 2: __global__ 调度器（读 tiling → asc_vf_call 分发）
//   Layer 3: _kernel_do 启动器（<<<>>> 语法）
//
// 与 SIMD/RegBase 的区别：
//   - 无 TPipe/TQue/DataCopyPad，直接通过 __gm__ 指针访问 GM
//   - 线程级并行：threadIdx.x / blockDim.x / blockIdx.x / gridDim.x
//   - 同步：asc_syncthreads()（线程间屏障）
//   - 无类封装，使用自由函数

#include <cstdint>
#include "kernel_operator.h"
#include "simt_api/asc_simt.h"
#include "{{op}}_kernel.h"

// TEMPLATE: 计算函数
// - __simt_vf__ __aicore__ LAUNCH_BOUND(SIMT_MAX_THREAD_NUM) 装饰器（必需）
// - 参数为标量 + __gm__ 指针（不用 GlobalTensor）
// - grid-stride loop
template </* bool COMPILE_TIME_FLAG */>
__simt_vf__ __aicore__ LAUNCH_BOUND(SIMT_MAX_THREAD_NUM) inline void {{Op}}SimtCompute(
    /* 标量参数: uint32_t rows, uint32_t cols, float alpha, float beta, ... */
    /* GM 指针: __gm__ const int32_t* rowPtr, __gm__ const int32_t* colInd,
       __gm__ const float* values, __gm__ const float* xVec, __gm__ float* yVec, ... */)
{
    // TEMPLATE: grid-stride loop — 每个线程处理一个或多个输出行
    for (uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
         row < /* totalRows */;
         row += gridDim.x * blockDim.x) {

        // 稀疏算子 SIMT 处理流程（以 SpMV CSR 为例）：
        //
        // int32_t rowStart = rowPtr[row];
        // int32_t rowEnd   = rowPtr[row + 1];
        // float acc = 0.0f;
        // for (int32_t j = rowStart; j < rowEnd; ++j) {
        //     acc += values[j] * xVec[colInd[j]];
        // }
        // yVec[row] = alpha * acc + beta * yVec[row];
    }
}

// TEMPLATE: __global__ 调度器
// - KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY)
// - tiling 通过 by value 方式从 host 传入
// - 按运行时条件（格式、dtype）选择不同模板特化的 asc_vf_call
// - **强制**使用 `extern "C"` 禁止 C++ name mangling
extern "C" __global__ __aicore__ void {{op}}_kernel(
    /* GM_ADDR 各参数, */ const {{Op}}TilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

    // TEMPLATE: 按运行时参数分发到不同模板特化
    // 示例 A — 单一路径（无编译期分支）：
    //   asc_vf_call<{{Op}}SimtCompute>(dim3{tiling.nthreads, 1, 1}, /* tiling 参数 */);
    //
    // 示例 B — 按格式分发：
    //   if (tiling.format == ACL_SPARSE_FORMAT_CSR) {
    //       asc_vf_call<{{Op}}SimtComputeCSR>(dim3{tiling.nthreads, 1, 1}, ...);
    //   } else if (tiling.format == ACL_SPARSE_FORMAT_COO) {
    //       asc_vf_call<{{Op}}SimtComputeCOO>(dim3{tiling.nthreads, 1, 1}, ...);
    //   }
    //
    // 示例 C — 按 trans 分发：
    //   if (tiling.trans == ACL_SPARSE_OP_N) {
    //       asc_vf_call<{{Op}}SimtComputeN>(dim3{tiling.nthreads, 1, 1}, ...);
    //   } else {
    //       asc_vf_call<{{Op}}SimtComputeT>(dim3{tiling.nthreads, 1, 1}, ...);
    //   }
}

// Kernel 启动器（host 侧调用）
// Tiling 通过 const 引用从 host 传入，kernel launch 时自动拷贝至 kernel 函数参数（by value）
void {{op}}_kernel_do(
    /* GM_ADDR 各数据指针, */ uint32_t numBlocks,
    const {{Op}}TilingData& tiling, void* stream)
{
    {{op}}_kernel<<<numBlocks, nullptr, stream>>>(/* ..., */ tiling);
}
