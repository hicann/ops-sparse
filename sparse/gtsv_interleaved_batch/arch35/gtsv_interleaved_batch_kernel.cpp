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
 * \file gtsv_interleaved_batch_kernel.cpp
 * \brief aclsparseSgtsvInterleavedBatch SIMT kernel (Thomas algorithm).
 *
 * 层次结构：
 *   kernel_do → kernel<<<>>> → Launcher::Process() → asc_vf_call<GtsvSimtCompute>
 *
 * SIMT 模型：每个 thread 独立处理 1 个 batch 的 Thomas 算法。
 * 线程局部状态 (d_prev, b_prev, x_next) 在寄存器中贯穿整个 m 行扫描，
 * 跨行依赖不经过 workspace GM 读写（SIMD 版本的瓶颈）。
 *
 * Workspace 只存 d'[i] (i=1..m-1) 供 backward 使用（m × batchCount × sizeof(float)
 * 而非 SIMD 版本的 2 × m × batchCount × sizeof(float)）。
 *
 * b'[i] 直接覆写到 x[i]（in-place，原 b 不再使用）。
 *
 * Data layout: row-major interleaved, data[row * batchCount + batch].
 */

#include <cstdint>
#include "kernel_operator.h"
#include "simt_api/asc_simt.h"
#include "gtsv_interleaved_batch_kernel.h"

namespace {
// 奇异矩阵保护阈值：若 Thomas 算法前向/后向消元过程中 d' 的绝对值小
// 于此阈值，则 early-return，保留 x 为当前（可能是部分）状态，避免
// 0/0 除法引发 NaN/Inf 在后续行中累积传播。此值远小于 FP32 最小正
// 规数 (≈ 1.2e-38) 的平方根，不会截断任何正常矩阵。
constexpr float kSingularEps = 1e-20f;
} // namespace

// ---------------------------------------------------------------------------
// Backward substitution helper. Called from GtsvProcessOneBatch after the
// forward sweep has completed d'[1..m-1] / b'[1..m-1] write-back.
// |d_prev| >= kSingularEps for m >= 2 is guaranteed by forward (which bails
// out early otherwise).
//
// State on entry:
//   d_prev = d'[m-1]  (computed in forward loop; still in register)
//   b_prev = b'[m-1]
// Workspace/GM layout at entry:
//   xGm[i*bc + myBatch]  == b'[i]   for i >= 1  (in-place over input b[i])
//   wsGm[i*bc + myBatch] == d'[i]   for i >= 1
//   xGm[0*bc + myBatch]  == b[0]    (forward never touches row 0)
// ---------------------------------------------------------------------------
__simt_callee__ inline void
GtsvBackwardSub(
    __gm__ float *dGm,
    __gm__ float *duGm,
    __gm__ float *xGm,
    __gm__ float *wsGm,
    int32_t m, int32_t bc, uint32_t myBatch,
    float d_prev, float b_prev)
{
    float x_next;

    // row m-1: x[m-1] = b'[m-1] / d'[m-1] (both still in registers)
    {
        const int64_t idxLast = (int64_t)(m - 1) * bc + myBatch;
        x_next = b_prev / d_prev;
        xGm[idxLast] = x_next;
    }

    // row m-2..1
    // 性能优化：用递减 idx 取代循环内 (int64_t)i * bc，
    // 每次迭代只需一次 64-bit 整数减法。
    int64_t idx = static_cast<int64_t>(m - 2) * bc + myBatch;
    for (int32_t i = m - 2; i >= 1; i--) {
        // x[idx] 当前持有 b'[i] (forward 写入)
        // ws[idx] 当前持有 d'[i] (forward 写入)
        const float b_prime_i = xGm[idx];
        const float d_prime_i = wsGm[idx];
        const float du_i      = duGm[idx];
        // 奇异保护：若 d_prime_i 过小，保留已写回的 x[i+1..m-1]，early return
        if (d_prime_i > -kSingularEps && d_prime_i < kSingularEps) return;
        const float x_i = (b_prime_i - du_i * x_next) / d_prime_i;
        x_next = x_i;
        xGm[idx] = x_i;  // 最终 x[i] 覆写 b'[i]
        idx -= bc;
    }

    // row 0: x[0] = (b'[0] - du[0] * x[1]) / d'[0]
    //   d'[0] = d[0] (forward 不修改 d[0]，直接读)
    //   b'[0] = 原 b[0] = x[0] (forward 不覆写 x[0]，因为 row 0 的 b'[0] 就等于 b[0])
    {
        const float b_prime_0 = xGm[myBatch];
        const float du_0      = duGm[myBatch];
        const float d_prime_0 = dGm[myBatch];
        if (d_prime_0 > -kSingularEps && d_prime_0 < kSingularEps) return;
        const float x_0 = (b_prime_0 - du_0 * x_next) / d_prime_0;
        xGm[myBatch] = x_0;
    }
}

// ---------------------------------------------------------------------------
// Per-batch Thomas solver helper (forward + backward on a single batch
// identified by myBatch in interleaved layout). Called from the grid-stride
// entry below after thread-to-batch mapping.
//
// Early-returns (with this batch marked undefined by contract) when the
// diagonal element |d[0]| or any forward-sweep |d'[i]| falls below
// kSingularEps. In that case the x output for this batch stays at whatever
// partial state the forward sweep produced — input b if forward never
// wrote anything, otherwise the last successfully-written row preserved.
//
// State at GtsvBackwardSub call time:
//   d_prev = d'[m-1]  (computed in forward loop; still in register)
//   b_prev = b'[m-1]
// Workspace/GM layout at end of forward:
//   xGm[i*bc + myBatch]  == b'[i]   for i >= 1  (in-place over input b[i])
//   wsGm[i*bc + myBatch] == d'[i]   for i >= 1
//   xGm[0*bc + myBatch]  == b[0]    (forward never touches row 0)
// ---------------------------------------------------------------------------
__simt_callee__ inline void
GtsvProcessOneBatch(
    __gm__ float *dlGm,
    __gm__ float *dGm,
    __gm__ float *duGm,
    __gm__ float *xGm,
    __gm__ float *wsGm,
    int32_t m, int32_t bc, uint32_t myBatch)
{
    // m=1 边界：直接 x[0] = b[0] / d[0]
    if (m == 1) {
        const float d0 = dGm[myBatch];
        // 奇异保护：若 |d0| 过小则不除，保留 x[0] 为原 b[0]（避免 NaN）
        if (d0 > -kSingularEps && d0 < kSingularEps) return;
        xGm[myBatch] = xGm[myBatch] / d0;
        return;
    }

    // Forward sweep (row 0..m-1)
    // 线程局部寄存器保存 d'[i-1] 和 b'[i-1]，无需 GM 跨行传递。
    float d_prev = dGm[myBatch];              // d'[0] = d[0]
    float b_prev = xGm[myBatch];              // b'[0] = b[0] (原 x 持有 b)
    // x[0] 不变：x[0] 持有 b[0]，forward 后持有 b'[0]（值相同）
    // 奇异保护：若 |d_prev| 过小，前向消元无法继续，保留 x 在初始 b 状态
    if (d_prev > -kSingularEps && d_prev < kSingularEps) return;

    // 性能优化：用递增 idx 取代循环内 (int64_t)i * bc，
    // 每次迭代只需一次 64-bit 整数加法（而非乘 + 加）。
    int64_t idx = static_cast<int64_t>(bc) + myBatch;   // row 1
    int64_t idxPrev = static_cast<int64_t>(myBatch);     // row 0
    for (int32_t i = 1; i < m; i++) {
        const float dl_i  = dlGm[idx];
        const float d_i   = dGm[idx];
        const float du_p  = duGm[idxPrev];
        const float b_i   = xGm[idx];

        const float w = dl_i / d_prev;
        d_prev = d_i - w * du_p;
        b_prev = b_i - w * b_prev;

        // 奇异保护：d_prev 过小则后续消元失败；保留 x[1..i-1] 已写的前向状态，early return
        if (d_prev > -kSingularEps && d_prev < kSingularEps) return;

        wsGm[idx] = d_prev;  // 持久化 d'[i] 到 workspace，供 backward 读取
        xGm[idx]  = b_prev;  // in-place: b'[i] 写入 x[i]，覆盖原 b[i]

        idxPrev = idx;
        idx += bc;
    }

    // Backward substitution (row m-1..0) — 拆分到 helper 以控制 NBNC
    GtsvBackwardSub(dGm, duGm, xGm, wsGm, m, bc, myBatch, d_prev, b_prev);
}

// ---------------------------------------------------------------------------
// SIMT VF 函数入口。thread-to-batch 映射后委托给 GtsvProcessOneBatch。
//
// 采用 grid-stride：host 侧 numBlocks 可能 cap 到物理 AIV 核数（远小于
// batchCount / kGtsvBatchPerBlock），此时单线程顺序处理多个 batch，确保
// 所有 batch 都被处理。gridStrideInBatches = numBlocks * kGtsvBatchPerBlock，
// 由 Launcher 在 Init 时预计算，避免 kernel 内重复乘。
// ---------------------------------------------------------------------------
__simt_vf__ __aicore__ __launch_bounds__(kGtsvBatchPerBlock) inline void
GtsvSimtCompute(
    __gm__ float *dlGm,
    __gm__ float *dGm,
    __gm__ float *duGm,
    __gm__ float *xGm,
    __gm__ float *wsGm,
    int32_t m, int32_t batchCount, int32_t gridStrideInBatches)
{
    // myBatch = blockIdx.x * kGtsvBatchPerBlock + threadIdx.x，然后按
    // gridStrideInBatches 递增遍历。batchCount <= gridStrideInBatches 时
    // 每个 active thread 只执行一次（等价于原来的 1:1 映射）。
    for (uint32_t myBatch = blockIdx.x * kGtsvBatchPerBlock + threadIdx.x;
         myBatch < static_cast<uint32_t>(batchCount);
         myBatch += static_cast<uint32_t>(gridStrideInBatches)) {
        GtsvProcessOneBatch(dlGm, dGm, duGm, xGm, wsGm, m, batchCount, myBatch);
    }
}

// ---------------------------------------------------------------------------
// Launcher 类：管理 GM 指针与 grid-stride 信息，调用 asc_vf_call 启动 VF。
// ---------------------------------------------------------------------------
class GtsvInterleavedBatchSimtLauncher {
public:
    __aicore__ inline void Init(
        GM_ADDR dl, GM_ADDR d, GM_ADDR du, GM_ADDR x, GM_ADDR workspace,
        int32_t m, int32_t batchCount, int32_t numBlocks)
    {
        dl_ = (__gm__ float *)dl;
        d_  = (__gm__ float *)d;
        du_ = (__gm__ float *)du;
        x_  = (__gm__ float *)x;
        ws_ = (__gm__ float *)workspace;
        m_ = m;
        batchCount_ = batchCount;
        gridStrideInBatches_ = static_cast<int32_t>(
            static_cast<uint32_t>(numBlocks) * kGtsvBatchPerBlock);
    }

    __aicore__ inline void Process()
    {
        asc_vf_call<GtsvSimtCompute>(dim3{kGtsvBatchPerBlock},
            dl_, d_, du_, x_, ws_, m_, batchCount_, gridStrideInBatches_);
    }

private:
    __gm__ float *dl_{nullptr};
    __gm__ float *d_{nullptr};
    __gm__ float *du_{nullptr};
    __gm__ float *x_{nullptr};
    __gm__ float *ws_{nullptr};
    int32_t m_{0};
    int32_t batchCount_{0};
    int32_t gridStrideInBatches_{0};
};

// ---------------------------------------------------------------------------
// __global__ 入口：由 kernel_do 通过 <<<numBlocks, 0, stream>>> 启动。
// ---------------------------------------------------------------------------
extern "C" __global__ __aicore__ void gtsv_interleaved_batch_kernel(
    GM_ADDR dl, GM_ADDR d, GM_ADDR du, GM_ADDR x, GM_ADDR workspace,
    const GtsvInterleavedBatchTilingData tiling, int32_t numBlocks)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    GtsvInterleavedBatchSimtLauncher launcher;
    launcher.Init(dl, d, du, x, workspace, tiling.m, tiling.batchCount, numBlocks);
    launcher.Process();
}

// ---------------------------------------------------------------------------
// kernel_do：Host 侧调用入口
// ---------------------------------------------------------------------------
void gtsv_interleaved_batch_kernel_do(
    GM_ADDR dl, GM_ADDR d, GM_ADDR du, GM_ADDR x,
    GM_ADDR workspace,
    const GtsvInterleavedBatchTilingData &tiling,
    uint32_t numBlocks,
    void *stream)
{
    gtsv_interleaved_batch_kernel<<<numBlocks, 0, stream>>>(
        dl, d, du, x, workspace, tiling, static_cast<int32_t>(numBlocks));
}
