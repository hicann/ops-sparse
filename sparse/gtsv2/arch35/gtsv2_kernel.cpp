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
 * \file gtsv2_kernel.cpp
 * \brief aclsparseSgtsv2 SIMT kernel (Thomas algorithm with partial pivoting).
 *
 * 参考 gtsv2_nopivot / gtsv_interleaved_batch 的 SIMT 实现模式。
 * 每个 thread 处理一个 RHS 列，直接读写 GM。
 *
 * 与 gtsv2_nopivot 的关键区别：
 *   - 前向消元带部分选主元（比较 |d'[i-1]| 与 |dl[i]|，必要时交换行）
 *   - 主元交换时产生两类 fill-in，均需存储到 workspace：
 *       du'[i]   = -mult * du[i]   （第一上对角线 fill-in）
 *       du2'[i-1] = 原 du[i]        （第二上对角线 fill-in，行交换将 old du[i]
 *                                     推到 row i-1 的第二上对角线）
 *   - workspace 为 kGtsv2WorkspaceArraysPerRhs*m floats/RHS（d' + du' + du2'），是 nopivot 的 3 倍
 *   - 回代使用修改后的 du' 和 du2'（含 fill-in）
 *
 * Launcher 层: Init → Process → asc_vf_call<Gtsv2SimtCompute>
 * SIMT 层:     Gtsv2SimtCompute → Gtsv2ProcessOneRhs
 */

#include <cstdint>
#include "kernel_operator.h"
#include "simt_api/asc_simt.h"
#include "gtsv2_kernel.h"

// ---------------------------------------------------------------------------
// 绝对值内联实现（CP2.2 建议：使用内联比较，消除 fabsf 依赖不确定性）
// 与仓内"无 math 函数依赖"的 SIMT 风格一致。
// ---------------------------------------------------------------------------
static __simt_callee__ inline float Gtsv2Fabsf(float x)
{
    return x < 0.0f ? -x : x;
}

// ---------------------------------------------------------------------------
// 每个 RHS 列的带主元 Thomas 算法
//
// 算法（0-indexed，对齐 LAPACK dgtsv）：
//
// 前向消元（step i = 1..m-1，消去 dl[i]）：
//   比较 |pivot_d|（= d'[i-1]）与 |dl[i]|：
//     若 |pivot_d| >= |dl[i]|：不交换，mult = dl[i]/pivot_d，标准消元
//     否则：交换行 i-1 与 i，mult = pivot_d/dl[i]，du 和 du2 均产生 fill-in
//       du'[i]    = -mult * du[i]   （第一上对角线 fill-in）
//       du2'[i-1] = 原 du[i]         （第二上对角线 fill-in：行交换将 old du[i]
//                                      推到 row i-1 的第二上对角线）
//
// 关键不变式（CP2.2 建议澄清）：
//   - "step j 修改 du[j]" 指修改寄存器中的 du 局部副本（cur_du / pivot_du），
//     GM 中的 du 数组始终只读。修改后的 du' 值写入 workspace，不从 GM 重读。
//   - dl[i] 始终读 GM 原始值：step j 仅修改 dl[j]（dl[j] = du[j+1]），
//     不影响 dl[i]（i > j）。
//   - d[i+1] / B[i+1] 预读均读 GM 原始值：forward 阶段仅写 row j-1（j <= i），
//     不影响 row i+1。
//
// GM layout:
//   dl/d/du: [m] — 所有 RHS 列共享（只读）
//   B:        [ldb * n] column-major — 第 rhsIdx 列 = B[rhsIdx * ldb : rhsIdx * ldb + m]
//   ws:       [3 * m] per RHS column — wsD[0..m-1] + wsDu[0..m-1] + wsDu2[0..m-1]
//
// 拆分约定（Gtsv2ProcessOneRhs 调度，m>=2 时进入 Forward/Backward）：
//   - d'/du'/du2' 全量写入 workspace（含末行），供 Backward 读取
//   - B' 全量写入 GM（含末行 b'[m-1]），Backward 从 GM 读回
//   - 两子函数之间不传递寄存器状态，仅经 workspace/GM 通信
// ---------------------------------------------------------------------------

// ---- Phase 1: Forward elimination with partial pivoting ----
// 比较主元、行交换、消元、du2 fill-in 存储；末行 b'[m-1] 写回 GM 供回代读取。
__simt_callee__ inline void Gtsv2ForwardSweep(
    __gm__ const float *dlGm, __gm__ const float *dGm, __gm__ const float *duGm,
    __gm__ float *BGm, __gm__ float *wsGm, int32_t m, int64_t colOff, int64_t wsOff)
{
    // Initialize pivot registers (row 0)
    float pivot_d = dGm[0];            // d'[0] = d[0]
    float pivot_du = duGm[0];          // du'[0] = du[0]
    float pivot_b = BGm[colOff];       // b'[0] = b[0]

    // Pre-read row 1
    float cur_d = dGm[1];              // d[1] (original)
    float cur_du = duGm[1];            // du[1] (original)
    float cur_b = BGm[colOff + 1];     // b[1] (original)

    for (int32_t i = 1; i < m; i++) {
        float dl_i = dlGm[i];          // always read original from GM

        // Row i-1 final state + row i next-pivot state
        float storeD, storeDu, storeDu2, storeB, nextD, nextDu, nextB;

        if (Gtsv2Fabsf(pivot_d) >= Gtsv2Fabsf(dl_i)) {
            // —— No swap: standard Thomas forward step ——
            float mult = dl_i / pivot_d;
            storeD = pivot_d;                      // d'[i-1] unchanged
            storeDu = pivot_du;                     // du'[i-1] unchanged
            storeDu2 = 0.0f;                        // no fill-in on 2nd super-diagonal
            storeB = pivot_b;                       // b'[i-1] unchanged
            nextD = cur_d - mult * pivot_du;        // d'[i]
            nextDu = cur_du;                        // du'[i] = du[i] (unchanged)
            nextB = cur_b - mult * pivot_b;         // b'[i]
        } else {
            // —— Swap rows i-1 and i ——
            // Row swap pushes old du[i] onto 2nd super-diagonal (du2) of row i-1.
            float mult = pivot_d / dl_i;
            storeD = dl_i;                          // d'[i-1] = dl[i] (new pivot)
            storeDu = cur_d;                        // du'[i-1] = old d[i]
            storeDu2 = cur_du;                      // du2'[i-1] = old du[i] (FILL-IN)
            storeB = cur_b;                         // b'[i-1] = old b'[i] (swapped)
            nextD = pivot_du - mult * cur_d;        // d'[i]
            nextDu = -mult * cur_du;                // du'[i] = FILL-IN (1st super-diagonal)
            nextB = pivot_b - mult * cur_b;         // b'[i]
        }

        // Store row i-1 final state (step i+1 won't modify row i-1)
        wsGm[wsOff + (i - 1)] = storeD;             // wsD[i-1]
        wsGm[wsOff + m + (i - 1)] = storeDu;        // wsDu[i-1]
        wsGm[wsOff + 2LL * m + (i - 1)] = storeDu2;   // wsDu2[i-1]
        BGm[colOff + (i - 1)] = storeB;             // b'[i-1]

        // Advance: row i becomes next pivot
        pivot_d = nextD;
        pivot_du = nextDu;
        pivot_b = nextB;

        // Pre-read row i+1 (if within bounds)
        if (i + 1 < m) {
            cur_d = dGm[i + 1];
            cur_du = duGm[i + 1];
            cur_b = BGm[colOff + (i + 1)];
        }
    }

    // Store last row (row m-1): d'/du'/du2' to workspace, b' to GM
    wsGm[wsOff + (m - 1)] = pivot_d;                // wsD[m-1] = d'[m-1]
    wsGm[wsOff + m + (m - 1)] = pivot_du;           // wsDu[m-1] = du[m-1] (user-constrained 0, unused in backward)
    wsGm[wsOff + 2LL * m + (m - 1)] = 0.0f;           // wsDu2[m-1] = 0 (no 2nd super-diagonal)
    BGm[colOff + (m - 1)] = pivot_b;                // b'[m-1] (供 Backward 读取)
}

// ---- Phase 2: Backward substitution ----
// x[m-1] = b'[m-1] / d'[m-1]（均从 GM/workspace 读取）
// x[i] = (b'[i] - du'[i]*x[i+1] - du2'[i]*x[i+2]) / d'[i]  for i = m-2 downto 0
//   du2 term applies only when i+2 < m (x[i+2] exists); matches golden back-sub.
// Note: no singularity guard — zero pivot produces Inf/NaN (by design)
__simt_callee__ inline void Gtsv2BackwardSubst(
    __gm__ float *BGm, __gm__ float *wsGm,
    int32_t m, int64_t colOff, int64_t wsOff)
{
    float xNext = BGm[colOff + (m - 1)] / wsGm[wsOff + (m - 1)];  // x[m-1]
    float xNext2 = 0.0f;                                          // placeholder for x[m]
    BGm[colOff + (m - 1)] = xNext;

    for (int32_t i = m - 2; i >= 0; i--) {
        float bPrime = BGm[colOff + i];                // b'[i] (forward wrote this)
        float dPrime = wsGm[wsOff + i];                // d'[i]
        float duPrime = wsGm[wsOff + m + i];           // du'[i] (may contain fill-in)
        float numerator = bPrime - duPrime * xNext;
        if (i + 2 < m) {
            float du2Prime = wsGm[wsOff + 2LL * m + i];  // du2'[i] (2nd super-diagonal fill-in)
            numerator -= du2Prime * xNext2;
        }
        float xI = numerator / dPrime;
        xNext2 = xNext;
        xNext = xI;
        BGm[colOff + i] = xI;                          // overwrite b'[i] with x[i]
    }
}

// ---- Dispatcher: 每个 RHS 列的带主元 Thomas 算法入口 ----
// Host 侧已校验 m >= 3，kernel 运行时 m 始终 >= 3，直接调用 Forward + Backward。
__simt_callee__ inline void Gtsv2ProcessOneRhs(
    __gm__ const float *dlGm, __gm__ const float *dGm, __gm__ const float *duGm,
    __gm__ float *BGm, __gm__ float *wsGm, int32_t m, int32_t ldb, int32_t rhsIdx)
{
    int64_t colOff = static_cast<int64_t>(rhsIdx) * ldb;
    // Workspace per RHS: kGtsv2WorkspaceArraysPerRhs*m floats. wsOff = rhsIdx * kGtsv2WorkspaceArraysPerRhs * m
    //   wsD[0..m-1]   = ws[wsOff + 0     .. wsOff + m - 1]   (modified d')
    //   wsDu[0..m-1]  = ws[wsOff + m     .. wsOff + 2*m - 1] (modified du',含 fill-in)
    //   wsDu2[0..m-1] = ws[wsOff + 2*m   .. wsOff + 3*m - 1] (2nd super-diagonal du2', swap fill-in)
    int64_t wsOff = static_cast<int64_t>(rhsIdx) * kGtsv2WorkspaceArraysPerRhs * static_cast<int64_t>(m);
    Gtsv2ForwardSweep(dlGm, dGm, duGm, BGm, wsGm, m, colOff, wsOff);
    Gtsv2BackwardSubst(BGm, wsGm, m, colOff, wsOff);
}

// ---------------------------------------------------------------------------
// SIMT VF 入口: 每个 thread 处理一个 RHS 列 (grid-stride)
// ---------------------------------------------------------------------------
__simt_vf__ __aicore__ __launch_bounds__(kGtsv2PivotThreadsPerBlock) inline void
Gtsv2SimtCompute(
    __gm__ const float *dlGm,
    __gm__ const float *dGm,
    __gm__ const float *duGm,
    __gm__ float *BGm,
    __gm__ float *wsGm,
    int32_t m, int32_t n, int32_t ldb, int32_t gridStrideInRhs)
{
    for (uint32_t rhsIdx = blockIdx.x * kGtsv2PivotThreadsPerBlock + threadIdx.x;
         rhsIdx < static_cast<uint32_t>(n);
         rhsIdx += static_cast<uint32_t>(gridStrideInRhs)) {
        Gtsv2ProcessOneRhs(dlGm, dGm, duGm, BGm, wsGm, m, ldb,
                           static_cast<int32_t>(rhsIdx));
    }
}

// ---------------------------------------------------------------------------
// Launcher: 管理 GM 指针，调用 asc_vf_call 启动 VF
// ---------------------------------------------------------------------------
class Gtsv2Launcher {
public:
    __aicore__ inline void Init(
        GM_ADDR dl, GM_ADDR d, GM_ADDR du, GM_ADDR B,
        GM_ADDR workspace,
        const Gtsv2TilingData &tiling)
    {
        dlGm_ = reinterpret_cast<__gm__ const float *>(dl);
        dGm_  = reinterpret_cast<__gm__ const float *>(d);
        duGm_ = reinterpret_cast<__gm__ const float *>(du);
        BGm_  = reinterpret_cast<__gm__ float *>(B);
        wsGm_ = reinterpret_cast<__gm__ float *>(workspace);
        m_ = tiling.m;
        n_ = tiling.n;
        ldb_ = tiling.ldb;
        gridStrideInRhs_ = static_cast<int32_t>(
            static_cast<uint32_t>(tiling.numBlocks) * kGtsv2PivotThreadsPerBlock);
    }

    __aicore__ inline void Process()
    {
        asc_vf_call<Gtsv2SimtCompute>(dim3{kGtsv2PivotThreadsPerBlock},
            dlGm_, dGm_, duGm_, BGm_, wsGm_, m_, n_, ldb_, gridStrideInRhs_);
    }

private:
    __gm__ const float *dlGm_{nullptr};
    __gm__ const float *dGm_{nullptr};
    __gm__ const float *duGm_{nullptr};
    __gm__ float *BGm_{nullptr};
    __gm__ float *wsGm_{nullptr};
    int32_t m_{0};
    int32_t n_{0};
    int32_t ldb_{0};
    int32_t gridStrideInRhs_{0};
};

// ---------------------------------------------------------------------------
// __global__ 入口
// ---------------------------------------------------------------------------
extern "C" __global__ __aicore__ void gtsv2_kernel(
    GM_ADDR dl, GM_ADDR d, GM_ADDR du, GM_ADDR B,
    GM_ADDR workspace,
    const Gtsv2TilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    Gtsv2Launcher launcher;
    launcher.Init(dl, d, du, B, workspace, tiling);
    launcher.Process();
}

void gtsv2_kernel_do(
    GM_ADDR dl, GM_ADDR d, GM_ADDR du, GM_ADDR B,
    GM_ADDR workspace,
    const Gtsv2TilingData &tiling,
    uint32_t numBlocks,
    void *stream)
{
    gtsv2_kernel<<<numBlocks, nullptr, stream>>>(
        dl, d, du, B, workspace, tiling);
}
