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
 * \file matmul_kernel.cpp
 * \brief sparseLt matmul 设备侧 kernel + extern "C" 启动器。
 *
 * kernel 列表：
 *   - matmul_kernel         (AIC_ONLY, Te::Mmad tensor_api): M(m,k) x N(k,n) -> temp (FP32)
 *   - epilogue_kernel       (AIV_ONLY, 低层 AscendC Vector API): D = alpha*reduce(temp) + beta*C
 *   - fused_matmul_kernel   (__mix__(1,2)): splitK==1 时的 matmul+epilogue 融合
 *
 * 启动器在 ASC 编译单元内封装 <<<>>>；host.cpp 以普通 C 函数形式调用。
 * 与 prune_kernel（tiling 经启动参数块按值传入）不同，
 * matmul/epilogue/fused 各 kernel 通过 splt_load_tiling(tilingGm) 从 GM 加载 tiling。
 * TilingData 结构体与 splt_load_tiling 定义在 shared/aclsparselt_internal.h。
 */


#include <cstdint>

#include "log/log.h"

#include "kernel_operator.h"
#include "tensor_api/tensor.h"

#include "shared/aclsparselt_internal.h"
#include "matmul_kernel.h"

#ifdef __CCE_AICORE__
static constexpr const char* kSparseLtLogTag = "aclsparseLt";
#endif

using namespace AscendC;
// tensor_api（Te 命名空间）的符号统一以显式 Te:: 前缀引用，
// 以避免 AscendC（kernel_operator.h）与 AscendC::Te（tensor_api）之间
// LocalTensor/GlobalTensor 的二义性。禁止添加 "using namespace AscendC::Te"。

// ============================================================================
// v2：L0C 累加类型映射。
// FP32/FP16/BF16 以 float 累加（L0C=float）；INT8 以 int32_t 累加。
// 用于在构造处（cube + fused）推导 L0C 张量类型。
// ============================================================================
template <typename T> struct SpltL0CTypeTrait { using type = float; };  // FP32/FP16/BF16
template <> struct SpltL0CTypeTrait<int8_t> { using type = int32_t; };  // INT8

// ============================================================================
// bias dtype 映射（依据 cuSPARSELt 规范）：
// "bias 向量的数据类型与矩阵 C 相同，唯一例外是 INT8 输入、INT8/INT32 输出、
//  INT32 计算的场景——此时 bias 的数据类型为 FP32。"
//   FP32 路径：bias = FP32（= C dtype）→ 直接加载，无需 Cast
//   FP16 路径：bias = FP16（= C dtype）→ 加载 FP16，Cast 到 FP32
//   BF16 路径：bias = BF16（= C dtype）→ 加载 BF16，Cast 到 FP32
//   INT8 路径：bias = FP32（cuSPARSELt 规范）→ 直接加载，无需 Cast
// ============================================================================
template <typename T> struct SpltBiasDTypeTrait { using type = T; };  // FP32/FP16/BF16: bias = C dtype
template <> struct SpltBiasDTypeTrait<int8_t> { using type = float; };  // INT8: bias = FP32
template <typename T> using SpltBiasDType = typename SpltBiasDTypeTrait<T>::type;

// 逐元素标量 Cast 循环（GetValue/SetValue）。对支持 static_cast 的
// 类型（FP16、INT8 等）使用 static_cast。
template <typename DstT, typename SrcT>
__aicore__ inline void SpltScalarCastLoop(LocalTensor<DstT>& dst, LocalTensor<SrcT> src,
                                           int32_t count)
{
    for (int32_t i = 0; i < count; ++i) {
        dst.SetValue(i, static_cast<DstT>(src.GetValue(i)));
    }
}

// ----------------------------------------------------------------------------
// float→int8_t Cast 辅助函数。
// DAV-3510 的 Cast API 不支持 float→int8_t 直接转换
// （CastImpl 的 cast_round_all / cast_none 列表中没有 Tuple<int8_t, float>）。
// release 模式下 ASCENDC_ASSERT 为空操作，Cast 会静默不执行——
// 目的缓冲区未被写入，产生垃圾数据。
// 受支持的路径：float→half（CAST_ROUND，Tuple<half,float> 在 cast_round_all 中）
// → int8_t（CAST_ROUND，Tuple<int8_t,half> 在 cast_round_all 中）。两步均为
// 硬件 Vector Cast。half 中间结果可放入一个很小的 TBuf
// （count * sizeof(half) 字节）。数值已由调用方预先 clamp 到 [-128,127]
// （Mins/Maxs），且 half 能精确表示该范围内的所有整数。
// ----------------------------------------------------------------------------
__aicore__ inline void SpltCastFp32ToInt8(
    LocalTensor<int8_t>& dst, const LocalTensor<float>& src,
    TBuf<TPosition::VECCALC>& halfBuf, int32_t count)
{
    LocalTensor<half> halfUB = halfBuf.Get<half>();
    Cast(halfUB, src, AscendC::RoundMode::CAST_ROUND, count);
    PipeBarrier<PIPE_V>();
    Cast(dst, halfUB, AscendC::RoundMode::CAST_ROUND, count);
    PipeBarrier<PIPE_V>();
}

// ----------------------------------------------------------------------------
// [OPT-BF16-VEC] 向量化 FP32→BF16 Cast。
// DAV-3510 CANN 9.1.0 支持 Vector Cast<bfloat16_t, float, CAST_ROUND>
// （Tuple<bfloat16_t, float> 在 cast_round_all 中）。单次 Vector Cast
// 调用即可达到与 FP16 路径相当的性能特征。
// ----------------------------------------------------------------------------
__aicore__ inline void SpltCastFp32ToBf16Vec(
    LocalTensor<bfloat16_t>& dst, const LocalTensor<float>& src, int32_t count)
{
    Cast(dst, src, AscendC::RoundMode::CAST_ROUND, count);
    PipeBarrier<PIPE_V>();
}

// [OPT-BF16-VEC] 向量化 BF16→FP32 Cast（SpltCastFp32ToBf16Vec 的逆操作）。
// 支持 Cast<float, bfloat16_t, CAST_NONE>（Tuple<float, bfloat16_t>
// 在 cast_none 中）。BF16→FP32 是加宽 Cast（无精度损失 → CAST_NONE）。
__aicore__ inline void SpltCastBf16ToFp32Vec(
    LocalTensor<float>& dst, const LocalTensor<bfloat16_t>& src, int32_t count)
{
    Cast(dst, src, AscendC::RoundMode::CAST_NONE, count);
    PipeBarrier<PIPE_V>();
}

// ============================================================================
// cube matmul kernel 的共享辅助函数。
// 提取以消除重复代码（#9、#10），并降低 SpltMatmulCubeImpl 与
// SpltFusedMatmulCubeImpl 的 NBNC/圈复杂度。
// 关键约束：SetFlag/WaitFlag 的位置与原始内联代码完全一致——
// 禁止重排或合并任何 flag 操作。
// ============================================================================
struct L1BufferConfig {
    int64_t aL1Off[2];
    int64_t bL1Off[2];
};

// [REFACTOR-KLOOP] L1 双缓冲初始化：buffer 大小按 kL1Size（而非
// baseK）计算，使外层 kL1 循环每次向 L1 加载更大的 K 块。预置
// MTE1_MTE2(0/1) 使首个 WaitFlag 可通过；M_MTE1(0/1) 用于 L0 乒乓；
// FIX_M(0) 用于跨 tile 的 L0C 复用握手。
template <typename T>
__aicore__ inline L1BufferConfig InitL1DoubleBuffer(int32_t baseM, int32_t baseN, int32_t kL1Size)
{
    const int64_t aL1Bytes = static_cast<int64_t>(baseM) * kL1Size * sizeof(T);
    const int64_t bL1Bytes = static_cast<int64_t>(kL1Size) * baseN * sizeof(T);
    L1BufferConfig cfg;
    cfg.aL1Off[0] = 0;
    cfg.aL1Off[1] = aL1Bytes + bL1Bytes;
    cfg.bL1Off[0] = aL1Bytes;
    cfg.bL1Off[1] = 2 * aL1Bytes + bL1Bytes;
    SetFlag<HardEvent::MTE1_MTE2>(0);
    SetFlag<HardEvent::MTE1_MTE2>(1);
    SetFlag<HardEvent::M_MTE1>(0);
    SetFlag<HardEvent::M_MTE1>(1);
    SetFlag<HardEvent::FIX_M>(0);
    return cfg;
}

// kernel 结束时排空 cube 侧全部 flag：
// - L1 乒乓：MTE1_MTE2(0/1) + M_MTE1(0/1) —— L1 双缓冲同步
// - L0C 复用：FIX_M(0) —— 跨 tile 的 L0C 复用握手（非双缓冲 flag）
__aicore__ inline void DrainCubeFlags()
{
    WaitFlag<HardEvent::MTE1_MTE2>(0);
    WaitFlag<HardEvent::MTE1_MTE2>(1);
    WaitFlag<HardEvent::M_MTE1>(0);
    WaitFlag<HardEvent::M_MTE1>(1);
    WaitFlag<HardEvent::FIX_M>(0);
}

// ============================================================================
// [REFACTOR-KLOOP] 两层 K 循环：外层 kL1（GM→L1）+ 内层 kL0（L1→L0+Mmad）。
// 参考：cann-samples streamk/main.asc §269-355。
//   kL1 循环上 L1 双缓冲乒乓（l1BufId = kl1Idx & 1）：
//     MTE1_MTE2(0/1) —— GM→L1 加载同步；M_MTE1(0/1) —— L1→L0 拷贝同步。
//   kL0 循环上 L0 双缓冲乒乓（l0BufId = kl0Idx & 1）。
//   FIX_M(0) 是独立的跨 tile L0C 复用握手 flag，
//   不属于 L1/L0 双缓冲机制。
// 关键约束：SetFlag/WaitFlag 位置遵循 streamk —— 禁止重排或合并。
// ============================================================================

// GM → L1：将 A(curM, curKL1) + B(curKL1, curN) 的一个 kL1 块从 GM 加载到 L1。
// L1 张量由调用方（kL1 循环）创建并按引用传入，
// 使其在后续 kL0 循环切片时仍然存活。
// transB 完全在 GM 层处理：transB=1 时，GM B 声明为
// DNExt(k, n)（列主序）。CopyGM2L1 自动选择 DN2ZN，
// 在分形转换过程中完成转置。此处无需分支。
template <typename T, typename GmSliceA, typename GmSliceB, typename L1ATensor, typename L1BTensor>
__aicore__ inline void SpltGmToL1(
    const GmSliceA& gmBlockARow, const GmSliceB& gmBlockBCol,
    L1ATensor& tensorAL1, L1BTensor& tensorBL1,
    int32_t curM, int32_t curN, int32_t curKL1, int32_t kPosL1, int32_t l1BufId)
{
    WaitFlag<HardEvent::MTE1_MTE2>(l1BufId);
    auto gmBlockA = gmBlockARow.Slice(Te::MakeCoord(0, kPosL1), Te::MakeShape(curM, curKL1));
    auto gmBlockB = gmBlockBCol.Slice(Te::MakeCoord(kPosL1, 0), Te::MakeShape(curKL1, curN));
    Te::Copy(Te::MakeCopy(Te::CopyGM2L1{}), tensorAL1, gmBlockA);
    Te::Copy(Te::MakeCopy(Te::CopyGM2L1{}), tensorBL1, gmBlockB);
    SetFlag<HardEvent::MTE2_MTE1>(l1BufId);
    WaitFlag<HardEvent::MTE2_MTE1>(l1BufId);
}

// 单步 L1→L0 + Mmad：从 L1 的 kOffL1 处切出 (curM, curKL0)，拷贝到 L0
// （经 l0BufId 偏移 = SPLT_HALF_L0_SIZE * l0BufId 双缓冲），再执行 Mmad。
// MTE1_M 同步确保 Mmad 前 L1→L0 拷贝已完成。M_MTE1 释放 L0。
template <typename T, typename L1ATensor, typename L1BTensor, typename L0CTensor>
__aicore__ inline void SpltL0AndMmadStep(
    const L1ATensor& tensorAL1, const L1BTensor& tensorBL1, L0CTensor& tensorL0C,
    int32_t curM, int32_t curN, int32_t curKL0, int32_t kOffL1,
    int32_t l0BufId, bool isFinalAcc, bool& cmatrixInitVal)
{
    WaitFlag<HardEvent::M_MTE1>(l0BufId);
    const int64_t l0Off = static_cast<int64_t>(SPLT_HALF_L0_SIZE) * l0BufId;
    auto tensorAL0 = Te::MakeTensor(Te::MakeMemPtr<Te::Location::L0A, T>(l0Off),
                                Te::MakeFrameLayout<Te::NZLayoutPtn, Te::LayoutTraitDefault<T>>(curM, curKL0));
    auto tensorBL0 = Te::MakeTensor(Te::MakeMemPtr<Te::Location::L0B, T>(l0Off),
                                Te::MakeFrameLayout<Te::ZNLayoutPtn, Te::LayoutTraitDefault<T>>(curKL0, curN));
    auto tensorAL1Tile = tensorAL1.Slice(Te::MakeCoord(0, kOffL1), Te::MakeShape(curM, curKL0));
    auto tensorBL1Tile = tensorBL1.Slice(Te::MakeCoord(kOffL1, 0), Te::MakeShape(curKL0, curN));
    Te::Copy(Te::MakeCopy(Te::CopyL12L0A{}), tensorAL0, tensorAL1Tile);
    Te::Copy(Te::MakeCopy(Te::CopyL12L0B{}), tensorBL0, tensorBL1Tile);
    SetFlag<HardEvent::MTE1_M>(l0BufId);
    WaitFlag<HardEvent::MTE1_M>(l0BufId);
    const uint8_t unitFlag = isFinalAcc ? static_cast<uint8_t>(SPLT_FINAL_ACCUMULATION)
                                        : static_cast<uint8_t>(SPLT_NON_FINAL_ACCUMULATION);
    Te::MmadParams params{static_cast<uint16_t>(curM), static_cast<uint16_t>(curN),
                          static_cast<uint16_t>(curKL0), unitFlag, cmatrixInitVal};
    constexpr auto mmadAtom = Te::MakeMmad(Te::MmadOperation{}, Te::MmadTraitDefault{});
    Te::Mmad(mmadAtom.with(params), tensorL0C, tensorAL0, tensorBL0);
    SetFlag<HardEvent::M_MTE1>(l0BufId);
    cmatrixInitVal = false;
}

// 内层 kL0 循环：从 L1 缓冲按 baseK 大小的块迭代，每块执行 L1→L0 + Mmad。
// L0 双缓冲乒乓（l0BufId = kl0Idx & 1）。isFinalAcc 仅在最后一个 kL1
// 迭代的最后一个 kL0 迭代时为 true。
template <typename T, typename L1ATensor, typename L1BTensor, typename L0CTensor>
__aicore__ inline void SpltKL0Loop(
    const L1ATensor& tensorAL1, const L1BTensor& tensorBL1, L0CTensor& tensorL0C,
    int32_t curM, int32_t curN, int32_t curKL1, int32_t baseK,
    bool isLastKL1, bool& cmatrixInitVal)
{
    const int32_t kL0IterNum = (curKL1 + baseK - 1) / baseK;
    for (int32_t kl0Idx = 0; kl0Idx < kL0IterNum; ++kl0Idx) {
        const int32_t l0BufId = kl0Idx & 0x1;
        const int32_t curKL0 = (kl0Idx + 1 == kL0IterNum) ? (curKL1 - kl0Idx * baseK) : baseK;
        const int32_t kOffL1 = kl0Idx * baseK;
        const bool isFinalAcc = isLastKL1 && (kl0Idx + 1 == kL0IterNum);
        SpltL0AndMmadStep<T>(tensorAL1, tensorBL1, tensorL0C,
                             curM, curN, curKL0, kOffL1, l0BufId, isFinalAcc, cmatrixInitVal);
    }
}

// 外层 kL1 循环驱动：GM→L1（kL1 块）→ kL0 循环（baseK 分块）→ 释放 L1。
// L1 双缓冲乒乓（l1BufId = kl1Idx & 1）。kSegStart 对 K 位置施加偏移：
// splitK>1（非融合）时为 kSeg*kSegLen，splitK==1（融合）时为 0。
// 取代原先单层的 SpltKLoopDriver。
template <typename T, typename GmSliceA, typename GmSliceB, typename L0CTensor>
__aicore__ inline void SpltKL1Loop(
    const GmSliceA& gmBlockARow, const GmSliceB& gmBlockBCol, L0CTensor& tensorL0C,
    int32_t curM, int32_t curN, int32_t k, int32_t kSegLen, int32_t kSegStart,
    int32_t baseK, int32_t kL1Size, bool& cmatrixInitVal, const L1BufferConfig& l1cfg)
{
    const int32_t kL1TileNum = (kSegLen + kL1Size - 1) / kL1Size;
    for (int32_t kl1Idx = 0; kl1Idx < kL1TileNum; ++kl1Idx) {
        const int32_t l1BufId = kl1Idx & 0x1;
        const int32_t kPosL1 = kSegStart + kl1Idx * kL1Size;
        int32_t curKL1 = (kl1Idx + 1 == kL1TileNum) ? (kSegLen - kl1Idx * kL1Size) : kL1Size;
        if (kPosL1 + curKL1 > k) { curKL1 = k - kPosL1; }
        if (curKL1 <= 0) { break; }
        const bool isLastKL1 = (kl1Idx + 1 == kL1TileNum) || (kPosL1 + curKL1 >= k);
        auto tensorAL1 = Te::MakeTensor(Te::MakeMemPtr<Te::Location::L1, T>(l1cfg.aL1Off[l1BufId]),
                                    Te::MakeFrameLayout<Te::NZLayoutPtn, Te::LayoutTraitDefault<T>>(curM, curKL1));
        auto tensorBL1 = Te::MakeTensor(Te::MakeMemPtr<Te::Location::L1, T>(l1cfg.bL1Off[l1BufId]),
                                    Te::MakeFrameLayout<Te::ZNLayoutPtn, Te::LayoutTraitDefault<T>>(curKL1, curN));
        SpltGmToL1<T>(gmBlockARow, gmBlockBCol, tensorAL1, tensorBL1,
                      curM, curN, curKL1, kPosL1, l1BufId);
        SpltKL0Loop<T>(tensorAL1, tensorBL1, tensorL0C,
                       curM, curN, curKL1, baseK, isLastKL1, cmatrixInitVal);
        SetFlag<HardEvent::MTE1_MTE2>(l1BufId);
    }
}

// SpltMatmulCubeImpl 的 tile 循环驱动：m/n tile 遍历 + GM 切片计算 +
// K 循环 + Fixpipe L0C->GM。提取以降低 SpltMatmulCubeImpl 的 NBNC
// （#3，72 -> 目标 <=50）。flag 操作保持不变。
template <typename T, typename GmTensorA, typename GmTensorB>
__aicore__ inline void SpltMatmulCubeTileLoop(
    const AclsparseltTilingData& td,
    GmTensorA& gmA, GmTensorB& gmB, GM_ADDR tempGm,
    const L1BufferConfig& l1cfg, int32_t blockId, int32_t blockNum,
    int32_t nTiles, int64_t totalMNTiles, int64_t totalTiles)
{
    const int32_t m = td.m;
    const int32_t n = td.n;
    const int32_t k = td.k;
    const int32_t baseM = td.baseM;
    const int32_t baseN = td.baseN;
    const int32_t baseK = td.baseK;
    const int32_t kSegLen = td.kSegLen;

    for (int32_t tileIdx = blockId; tileIdx < totalTiles; tileIdx += blockNum) {
        const int32_t kSeg = static_cast<int32_t>(static_cast<int64_t>(tileIdx) / totalMNTiles);
        const int32_t mnTile = static_cast<int32_t>(static_cast<int64_t>(tileIdx) % totalMNTiles);
        const int32_t mTile = mnTile / nTiles;
        const int32_t nTile = mnTile % nTiles;

        const int32_t mPos = mTile * baseM;
        const int32_t nPos = nTile * baseN;
        const int32_t kSegStart = kSeg * kSegLen;
        const int32_t curM = (mPos + baseM <= m) ? baseM : (m - mPos);
        const int32_t curN = (nPos + baseN <= n) ? baseN : (n - nPos);

        // 跳过起始位置超出实际 k 的 kSeg tile。
        if (kSegStart >= k) {
            continue;
        }

        // 两级 Slice，与 cann-samples matmul_kernel_swat.h 一致。
        auto gmBlockARow = gmA.Slice(Te::MakeCoord(mPos, 0), Te::MakeShape(curM, k));
        // B 切片对 transB 与非 transB 相同：transB 时 GM B 声明为
        // DNExt(k,n)（(n,k) 物理数据的列主序视图），因此按 (0, nPos, k, curN)
        // 切片对两条路径均成立。
        auto gmBlockBCol = gmB.Slice(Te::MakeCoord(0, nPos), Te::MakeShape(k, curN));

        // 本 kSeg 的 GM 输出切片：tempSeg 位于 [kSeg*m*n]。
        // v2：temp 元素类型跟随 L0C 类型（FP32/FP16/BF16 为 float，
        // INT8 为 int32_t）。sizeof(float) == sizeof(int32_t) == 4，
        // 字节偏移计算完全一致；仅类型化指针不同。
        using L0CType = typename SpltL0CTypeTrait<T>::type;
        __gm__ uint8_t* tempBase = reinterpret_cast<__gm__ uint8_t*>(tempGm) +
                                   static_cast<int64_t>(kSeg) * m * n * sizeof(L0CType);
        auto gmCSeg = Te::MakeTensor(Te::MakeMemPtr<Te::Location::GM>(reinterpret_cast<__gm__ L0CType*>(tempBase)),
                                 Te::MakeFrameLayout<Te::NDExtLayoutPtn, SPLT_GM_FLOAT_C0>(m, n));

        auto tensorL0C = Te::MakeTensor(Te::MakeMemPtr<Te::Location::L0C, typename SpltL0CTypeTrait<T>::type>(0),
                                    Te::MakeFrameLayout<Te::NZLayoutPtn, SPLT_L0C_C0>(curM, curN));

        // 写 L0C 前等待前一个 tile 的 Fixpipe 完成。
        WaitFlag<HardEvent::FIX_M>(0);
        bool cmatrixInitVal = true;
        SpltKL1Loop<T>(gmBlockARow, gmBlockBCol, tensorL0C,
                       curM, curN, k, kSegLen, kSegStart, baseK, td.kL1Size,
                       cmatrixInitVal, l1cfg);

        // ---- L0C -> GM (Fixpipe).
        auto gmBlockC = gmCSeg.Slice(Te::MakeCoord(mPos, nPos), Te::MakeShape(curM, curN));
        Te::FixpipeParams fixpParams;
        fixpParams.unitFlag = static_cast<uint8_t>(SPLT_FINAL_ACCUMULATION);
        auto copyL0C2GM = Te::MakeCopy(Te::CopyL0C2GM{});
        Te::Copy(copyL0C2GM.with(fixpParams), gmBlockC, tensorL0C);
        SetFlag<HardEvent::FIX_M>(0);
    }
}

// SpltMatmulCubeImpl 的 transB 分发：构造 gmB（依据 transB 选择
// DNExt 或 NDExt）并调用 SpltMatmulCubeTileLoop。消除重复的
// transB 分支代码（4 个分支 -> 2 个 + 辅助函数）。
template <typename T, typename GmTensorA>
__aicore__ inline void SpltMatmulCubeDispatchB(
    const AclsparseltTilingData& td, GmTensorA& gmA, GM_ADDR bGm, GM_ADDR tempGm,
    const L1BufferConfig& l1cfg, int32_t blockId, int32_t blockNum,
    int32_t nTiles, int64_t totalMNTiles, int64_t totalTiles)
{
    const int32_t k = td.k;
    const int32_t n = td.n;
    if (td.transB != 0) {
        auto gmB = Te::MakeTensor(Te::MakeMemPtr<Te::Location::GM>(reinterpret_cast<__gm__ T*>(bGm)),
                              Te::MakeFrameLayout<Te::DNExtLayoutPtn, Te::LayoutTraitDefault<T>>(k, n));
        SpltMatmulCubeTileLoop<T>(td, gmA, gmB, tempGm, l1cfg, blockId, blockNum,
                                   nTiles, totalMNTiles, totalTiles);
    } else {
        auto gmB = Te::MakeTensor(Te::MakeMemPtr<Te::Location::GM>(reinterpret_cast<__gm__ T*>(bGm)),
                              Te::MakeFrameLayout<Te::NDExtLayoutPtn, Te::LayoutTraitDefault<T>>(k, n));
        SpltMatmulCubeTileLoop<T>(td, gmA, gmB, tempGm, l1cfg, blockId, blockNum,
                                   nTiles, totalMNTiles, totalTiles);
    }
}

// ============================================================================
// CUBE MATMUL KERNEL（AIC_ONLY，Te::Mmad tensor_api）
//   M(m,k) x N(k,n) -> temp(splitK*m*n, FP32)
//   每 tile 单缓冲，K 循环在 L0C 中累加。正确性优先。
// ============================================================================
// GM 输入张量布局使用 Te::LayoutTraitDefault<T>，其 C0 由数据类型推导
// （32/sizeof(T)：fp16=16，fp32=8），因此无需显式 C0 参数。
template <typename T>
__aicore__ inline void SpltMatmulCubeImpl(GM_ADDR aPrunedGm, GM_ADDR bGm,
                                          GM_ADDR tempGm, GM_ADDR tilingGm)
{
    const AclsparseltTilingData td = splt_load_tiling(tilingGm);
    const int32_t m = td.m;
    const int32_t n = td.n;
    const int32_t k = td.k;
    const int32_t baseM = td.baseM;
    const int32_t baseN = td.baseN;
    const int32_t splitK = td.splitK;
    // batch 参数
    const int32_t numBatches = (td.numBatches > 0) ? td.numBatches : 1;

    const int32_t blockId = static_cast<int32_t>(GetBlockIdx());
    const int32_t blockNum = static_cast<int32_t>(GetBlockNum());

    const int32_t mTiles = (m + baseM - 1) / baseM;
    const int32_t nTiles = (n + baseN - 1) / baseN;
    // 使用 int64_t 防止 mTiles*nTiles 接近
    // INT32_MAX（大 m、n）时溢出。
    const int64_t totalMNTiles = static_cast<int64_t>(mTiles) * static_cast<int64_t>(nTiles);
    // 每个 batch 内的 totalTiles（batch 循环在外层）
    const int64_t totalTiles = totalMNTiles * static_cast<int64_t>(splitK);
    if (totalTiles <= 0) { return; }

    // GM 基础张量（ND 行主序入口）。
    // GM 侧 Te::NDExtLayoutPtn 必须使用与 dtype 匹配的 C0
    // （32/sizeof(T)：fp16=16，fp32=8）。默认 Te::LayoutTraitDefault<> 以
    // uint16_t 推导 => C0=16，对 fp32 是错误的，会导致 CopyGmToCbufMultiND2Nz
    // 按错误的内块大小打包 NZ 分形，数据错乱。
    // 使用 Te::LayoutTraitDefault<T>，它同时设置数据类型与 C0（=32/sizeof(T)）。
    // sparseTrans=1：dense×dense + transA。物理 A 为
    // (k,m) 行主序；声明为 DNExt(m,k)（列主序视图），使 CopyGM2L1
    // 在分形转换时完成转置 —— 与 transB 同一机制。
    // sparseTrans=0：标准 NDExt(m,k)（sparse 路径 prune 已完成转置）。
    // transB=1：GM B 物理布局为 (n, k) 行主序。声明为
    // DNExt(k, n)（列主序）—— CopyGM2L1 自动选择 DN2ZN 完成转置。
    // DNExt 与 NDExt 产生不同的张量类型，因此 tile 循环调用必须在
    // 调用点分支（按 GmTensorA/GmTensorB 类型做模板分发）。
    const L1BufferConfig l1cfg = InitL1DoubleBuffer<T>(baseM, baseN, td.kL1Size);
    SetMMLayoutTransform(true);
    // batch 外循环（GM 指针按 batchStride 偏移）
    using L0CType = typename SpltL0CTypeTrait<T>::type;
    const int64_t tempBatchStride = static_cast<int64_t>(splitK) * m * n * sizeof(L0CType);
    for (int32_t b = 0; b < numBatches; ++b) {
        // GM 指针按 batchStride 偏移（batchStride 以元素数为单位）
        __gm__ T* aBase = reinterpret_cast<__gm__ T*>(aPrunedGm)
                          + static_cast<int64_t>(b) * td.batchStrideA;
        GM_ADDR bBatchGm = bGm + static_cast<int64_t>(b) * td.batchStrideB * static_cast<int64_t>(sizeof(T));
        GM_ADDR tempBatchGm = tempGm + static_cast<int64_t>(b) * tempBatchStride;
        if (td.sparseTrans != 0) {
            auto gmA = Te::MakeTensor(Te::MakeMemPtr<Te::Location::GM>(aBase),
                                  Te::MakeFrameLayout<Te::DNExtLayoutPtn, Te::LayoutTraitDefault<T>>(m, k));
            SpltMatmulCubeDispatchB<T>(td, gmA, bBatchGm, tempBatchGm, l1cfg, blockId, blockNum,
                                        nTiles, totalMNTiles, totalTiles);
        } else {
            auto gmA = Te::MakeTensor(Te::MakeMemPtr<Te::Location::GM>(aBase),
                                  Te::MakeFrameLayout<Te::NDExtLayoutPtn, Te::LayoutTraitDefault<T>>(m, k));
            SpltMatmulCubeDispatchB<T>(td, gmA, bBatchGm, tempBatchGm, l1cfg, blockId, blockNum,
                                        nTiles, totalMNTiles, totalTiles);
        }
    }

    // [OPT-P1] 排空所有残留 flag（L1 两个乒乓槽位 + L0 + L0C）。
    DrainCubeFlags();

    // 恢复默认布局转换模式（与 cann-samples
    // matmul_kernel_swat.h:162 一致）。
    SetMMLayoutTransform(false);
}

// ============================================================================
// EPILOGUE KERNEL（AIV_ONLY）
//   D = alpha * reduce_k(temp) + beta * C
//   temp 为 FP32（splitK 部分和）；C/D 为 T（FP16 或 FP32）。
//   以 DataCopyPad（GM<->UB）+ UB 内标量 GetValue/SetValue 取代原来的
//   标量 __gm__ 访问。标量 GM 访问（__gm__ T* 下标）在
//   API 黑名单中（ascendc-api-best-practices），在 AIV 核上会静默丢数据
//   —— temp 值正确但 D 全为零。
// ----------------------------------------------------------------------------
// 共享的 int32 域 splitK 累加，供 SpltEpilogueInt32AccPath（步骤 1-2）与
// SpltEpilogueChunkSplitKReduce 的 int32-temp 分支使用。
// 加载第一个 splitK 段到 accIntUB，再用向量 Add<int32_t> 累加其余段
// （原地操作：dst==src0；arch35/DAV_3510 支持）。
// 同步语义：
//   - 每次 DataCopyPad 加载后 MTE2_V 握手；
//   - 每次 Add 后 PipeBarrier<PIPE_V> + V_MTE2 握手 —— 在下一段
//     DataCopyPad 覆写 tempIntUB 前完成 WAR 防护。
// GmT 为 GM temp 元素类型（int32_t）；模板化使一个辅助函数同时服务
// GlobalTensor<int32_t> 与 GlobalTensor<TempType> 两类调用点。
// ----------------------------------------------------------------------------
template <typename GmT>
__aicore__ inline void SpltSplitKAccInt32(
    LocalTensor<int32_t>& accIntUB, LocalTensor<int32_t>& tempIntUB,
    GlobalTensor<GmT>& tempGM,
    int64_t base, int32_t count, int32_t splitK, int64_t totalElem,
    DataCopyExtParams& copyAcc, DataCopyPadExtParams<GmT>& padAcc)
{
    // 约束：两条调用路径（SpltEpilogueInt32AccPath 与 SpltEpilogueChunkSplitKReduce
    // 的 int32 分支）均以 ReinterpretCast<int32_t> 复用 float UB 后传入本函数，
    // 隐含 GmT == int32_t 的假设；SpltL0CTypeTrait 未来扩展新 temp 类型时，
    // 在此编译期拦截，防止静默出错。
    static_assert(std::is_same_v<GmT, int32_t>,
                  "SpltSplitKAccInt32 仅支持 int32_t 的 temp 元素类型");
    // 1. 加载第一个 splitK 段（int32 GM -> int32 UB）
    DataCopyPad(accIntUB, tempGM[base], copyAcc, padAcc);
    SetFlag<HardEvent::MTE2_V>(0);
    WaitFlag<HardEvent::MTE2_V>(0);

    // 2. 向量 int32 累加其余段
    for (int32_t s = 1; s < splitK; ++s) {
        DataCopyPad(tempIntUB, tempGM[static_cast<uint64_t>(s) * totalElem + base],
                    copyAcc, padAcc);
        SetFlag<HardEvent::MTE2_V>(0);
        WaitFlag<HardEvent::MTE2_V>(0);
        Add<int32_t>(accIntUB, accIntUB, tempIntUB, count);
        PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_MTE2>(0);
        WaitFlag<HardEvent::V_MTE2>(0);
    }
}

// splitK epilogue 的 INT8 + alpha=1 + beta=0 快速路径。
// 使用 int32 累加（LocalTensor 上的 Add<int32_t>）而非 float Cast+Add，
// 对大累加值保持精确的 int32 精度。
// [OPT-INT32-VEC] splitK 累加现使用 Vector Add<int32_t>（原地，
// dst==src0）取代标量 GetValue/SetValue 循环。arch35（DAV_3510）支持
// Add<int32_t>。所有 GM<->UB 搬运均使用 DataCopyPad。
// [OPT-INT8-VEC] INT8 输出现使用向量化 Cast+clamp（取代标量 clamp 循环）：
//   Cast int32→float（Vector）→ Mins/Maxs clamp（Vector）→ SpltCastFp32ToInt8（Vector）。
// float 中间结果对 clamp 后 [-128,127] 内的值是精确的。
// flag 操作与既有 SpltEpilogueProcessChunk 模式完全一致。
template <typename OutType>
__aicore__ inline void SpltEpilogueInt32AccPath(
    LocalTensor<float>& accUB, LocalTensor<float>& tempUB,
    TBuf<TPosition::VECCALC>& dTBuf, TBuf<TPosition::VECCALC>& halfBuf,
    GlobalTensor<int32_t>& tempGM, GlobalTensor<OutType>& dGM,
    int64_t base, int32_t count, int32_t splitK, int64_t totalElem)
{
    DataCopyExtParams copyAcc{1, 0, 0, 0, 0};
    DataCopyPadExtParams<int32_t> padAcc{false, 0, 0, int32_t(0)};
    copyAcc.blockLen = static_cast<uint32_t>(count * sizeof(int32_t));

    // 将 float 缓冲重解释为 int32（sizeof(float) == sizeof(int32_t) == 4）
    LocalTensor<int32_t> accIntUB = accUB.ReinterpretCast<int32_t>();
    LocalTensor<int32_t> tempIntUB = tempUB.ReinterpretCast<int32_t>();

    // 1+2. 加载第一个 splitK 段并在 int32 域向量累加其余段
    //      （共享辅助函数，亦被 SpltEpilogueChunkSplitKReduce 的
    //      int32-temp 分支使用）。
    SpltSplitKAccInt32(accIntUB, tempIntUB, tempGM, base, count, splitK, totalElem,
                       copyAcc, padAcc);

    // 3. 输出
    if constexpr (std::is_same_v<OutType, int32_t>) {
        // INT32 输出：int32 UB 直接写 GM（无 Cast）
        SetFlag<HardEvent::V_MTE3>(0);
        WaitFlag<HardEvent::V_MTE3>(0);
        DataCopyPad(dGM[base], accIntUB, copyAcc);
    } else {
        // [OPT-INT8-VEC] INT8 输出：向量化 Cast int32→float→clamp→int8。
        // tempUB（float）复用为 Cast 目的缓冲 —— 上方 splitK 循环完成后
        // 它不再参与累加。
        LocalTensor<int8_t> dUB = dTBuf.Get<int8_t>();
        Cast(tempUB, accIntUB, AscendC::RoundMode::CAST_NONE, count);
        PipeBarrier<PIPE_V>();
        Mins(tempUB, tempUB, 127.0f, count);
        Maxs(tempUB, tempUB, -128.0f, count);
        SpltCastFp32ToInt8(dUB, tempUB, halfBuf, count);
        DataCopyExtParams copyOut{1, 0, 0, 0, 0};
        copyOut.blockLen = static_cast<uint32_t>(count * sizeof(int8_t));
        SetFlag<HardEvent::V_MTE3>(0);
        WaitFlag<HardEvent::V_MTE3>(0);
        DataCopyPad(dGM[base], dUB, copyOut);
    }

    // 4. 跨迭代同步（与既有 SpltEpilogueProcessChunk 模式一致）
    SetFlag<HardEvent::MTE3_MTE2>(0);
    WaitFlag<HardEvent::MTE3_MTE2>(0);
}

__aicore__ inline float SpltGetScalarFromUB(LocalTensor<float>& ubTensor, int32_t index)
{
    return ubTensor.GetValue(static_cast<uint32_t>(index));
}

// 将完整 bias 向量从 GM 加载到 FP32 UB（count 个元素）。
// BiasType=float（FP32/INT8）：直接 DataCopyPad 到 FP32 UB。
// BiasType=half（FP16）：DataCopyPad 原始数据到 biasRawBuf，再 Vector Cast 到 FP32 UB。
// BiasType=bfloat16_t（BF16）：DataCopyPad 原始数据到 biasRawBuf，再走 SpltCastBf16ToFp32Vec。
// BiasType != float 时 biasRawBuf 必须预先分配 count * sizeof(BiasType) 字节。
template <typename BiasType>
__aicore__ inline void SpltLoadBiasFullToFp32(
    LocalTensor<float>& biasFp32UB,
    TBuf<TPosition::VECCALC>& biasRawBuf,
    GlobalTensor<BiasType>& biasGM,
    int32_t count)
{
    if constexpr (std::is_same_v<BiasType, float>) {
        // FP32/INT8 路径：GM 中 bias 为 FP32，直接加载
        DataCopyExtParams copyBias{1, static_cast<uint32_t>(count * sizeof(float)), 0, 0, 0};
        DataCopyPadExtParams<float> padBias{false, 0, 0, 0.0f};
        DataCopyPad(biasFp32UB, biasGM[0], copyBias, padBias);
        SetFlag<HardEvent::MTE2_V>(0);
        WaitFlag<HardEvent::MTE2_V>(0);
    } else {
        // FP16/BF16 路径：加载原始 bias，再 Vector Cast 到 FP32
        LocalTensor<BiasType> biasRawUB = biasRawBuf.Get<BiasType>();
        DataCopyExtParams copyBias{1, static_cast<uint32_t>(count * sizeof(BiasType)), 0, 0, 0};
        DataCopyPadExtParams<BiasType> padBias{false, 0, 0, BiasType(0)};
        DataCopyPad(biasRawUB, biasGM[0], copyBias, padBias);
        SetFlag<HardEvent::MTE2_V>(0);
        WaitFlag<HardEvent::MTE2_V>(0);
        if constexpr (std::is_same_v<BiasType, bfloat16_t>) {
            // BF16：使用既有 Vector Cast 辅助函数（Cast<float, bfloat16_t, CAST_NONE>）
            SpltCastBf16ToFp32Vec(biasFp32UB, biasRawUB, count);
        } else {
            // FP16：直接 Vector Cast（Cast<float, half, CAST_NONE>）
            Cast(biasFp32UB, biasRawUB, AscendC::RoundMode::CAST_NONE, count);
            PipeBarrier<PIPE_V>();
        }
    }
}

// 分配 bias UB 缓冲，并按需将完整 bias 向量从 GM 加载到 FP32 UB。
// 由 SpltEpilogueImpl（非融合）与 SpltFusedEpilogueImpl（融合）共享，
// 消除重复的 bias 初始化代码（#2）。两条路径构建相同的
// GlobalTensor + TBuf + LocalTensor + 条件调用 SpltLoadBiasFullToFp32。
// biasRawBuf 仅在 BiasType 非 float 时分配（FP16/BF16 的 Cast 中间缓冲）。
template <typename BiasType>
__aicore__ inline void SpltInitBiasBuffers(
    GlobalTensor<BiasType>& biasVecGM,
    TBuf<TPosition::VECCALC>& biasVecBuf,
    TBuf<TPosition::VECCALC>& biasRawBuf,
    LocalTensor<float>& biasVecUB,
    TPipe& pipe,
    uint64_t biasDevPtr, int64_t biasStride, int32_t m)
{
    biasVecGM.SetGlobalBuffer(reinterpret_cast<__gm__ BiasType*>(biasDevPtr),
                               static_cast<uint64_t>(m));
    pipe.InitBuffer(biasVecBuf, static_cast<uint32_t>(static_cast<size_t>(m) * sizeof(float)));
    biasVecUB = biasVecBuf.Get<float>();
    if constexpr (!std::is_same_v<BiasType, float>) {
        pipe.InitBuffer(biasRawBuf, static_cast<uint32_t>(static_cast<size_t>(m) * sizeof(BiasType)));
    }
    if (biasStride == 0) {
        // 所有 batch 共用同一 bias
        SpltLoadBiasFullToFp32<BiasType>(biasVecUB, biasRawBuf, biasVecGM, m);
    }
}

// 从 SpltEpilogueProcessChunk 提取的子辅助函数，用于降低
// NBNC（139->约45）、圈复杂度（36->约10）与嵌套深度（6->3）。
// flag 操作与原始内联代码完全一致。

// splitK 加载 + 累加：加载第一段并累加其余 splitK 段。
// int32-temp 路径（INT8 输入）在 int32 域累加，对加载目标缓冲执行
// Add<int32_t>（与 SpltEpilogueInt32AccPath 及 float 分支同模式），
// 加载循环结束后仅做一次 int32->float Cast。MTE2 加载后紧跟逐段
// Cast 并不安全：在 DAV-3510 上 Cast 的完成既不被 MTE2_V 事件握手
// 覆盖，也不被 PipeBarrier<PIPE_V> 覆盖，因此 Cast 可能在下一段
// DataCopyPad 已覆写 tempUB 之后才执行 —— Cast 会读到陈旧的
// （前一段的）内容，reduce 会得到 2*partial0 / 2*partial1。最终 Cast
// 读取的 accIntUB 从不被 MTE2 写入，故不受该竞争影响。循环后的单次
// Cast 对 |sum| < 2^24 也是精确的，且与 golden 参考一致（int32 累加、
// 一次舍入）；逐段 float 累加则会引入每段一次的舍入误差。
template <typename T, typename TempType>
__aicore__ inline void SpltEpilogueChunkSplitKReduce(
    LocalTensor<float>& accUB, LocalTensor<float>& tempUB,
    GlobalTensor<TempType>& tempGM,
    int64_t base, int32_t count, int32_t splitK, int64_t totalElem,
    DataCopyExtParams& copyAcc, DataCopyPadExtParams<TempType>& padAcc)
{
    if constexpr (std::is_same_v<TempType, float>) {
        DataCopyPad(accUB, tempGM[base], copyAcc, padAcc);
        SetFlag<HardEvent::MTE2_V>(0);
        WaitFlag<HardEvent::MTE2_V>(0);
        for (int32_t s = 1; s < splitK; ++s) {
            DataCopyPad(tempUB, tempGM[static_cast<uint64_t>(s) * totalElem + base], copyAcc, padAcc);
            SetFlag<HardEvent::MTE2_V>(0);
            WaitFlag<HardEvent::MTE2_V>(0);
            Add(accUB, accUB, tempUB, static_cast<int32_t>(count));
            SetFlag<HardEvent::V_MTE2>(0);
            WaitFlag<HardEvent::V_MTE2>(0);
        }
    } else {
        LocalTensor<int32_t> accIntUB = accUB.ReinterpretCast<int32_t>();
        LocalTensor<int32_t> tempIntUB = tempUB.ReinterpretCast<int32_t>();
        SpltSplitKAccInt32(accIntUB, tempIntUB, tempGM, base, count, splitK, totalElem,
                           copyAcc, padAcc);
        // 所有加载消费完毕后做单次 int32->float Cast：accIntUB 仅由
        // V（本 kernel）写入，不存在残留的跨流水竞争。
        Cast(accUB, accIntUB, AscendC::RoundMode::CAST_NONE, static_cast<int32_t>(count));
        PipeBarrier<PIPE_V>();
    }
}

// 施加 alpha 缩放：D = alpha * acc（标量或逐行向量）。
template <typename T>
__aicore__ inline void SpltEpilogueChunkApplyAlpha(
    LocalTensor<float>& dFp32UB, LocalTensor<float>& accUB, LocalTensor<float>& alphaVecUB,
    int64_t base, int32_t count, int32_t n, int32_t alphaVectorScaling, float alpha)
{
    if (n == 0) { return; }
    if (alphaVectorScaling == 1) {
        int32_t offset = 0;
        while (offset < count) {
            int32_t row = static_cast<int32_t>((base + offset) / n);
            int64_t rowEndFlat = static_cast<int64_t>(row + 1) * n;
            int64_t chunkEnd = base + count;
            int32_t elemEnd = static_cast<int32_t>((rowEndFlat < chunkEnd) ? rowEndFlat : chunkEnd);
            int32_t elemCount = elemEnd - static_cast<int32_t>(base + offset);
            float alpha_r = SpltGetScalarFromUB(alphaVecUB, row);
            Muls(dFp32UB[offset], accUB[offset], alpha_r, elemCount);
            PipeBarrier<PIPE_V>();
            offset += elemCount;
        }
    } else {
        Muls(dFp32UB, accUB, alpha, static_cast<int32_t>(count));
    }
}

// 以逐行向量缩放方式施加 beta*C（betaVectorScaling 路径）。
template <typename T>
__aicore__ inline void SpltEpilogueChunkApplyBetaVec(
    LocalTensor<float>& dFp32UB, LocalTensor<float>& tempUB,
    TBuf<TPosition::VECCALC>& cTBuf, TBuf<TPosition::VECCALC>& cFp32Buf,
    GlobalTensor<T>& cGM,
    int64_t base, int32_t count, int32_t n,
    DataCopyExtParams& copyC, DataCopyPadExtParams<T>& padC,
    LocalTensor<float>& betaVecUB)
{
    if (n == 0) { return; }
    LocalTensor<T> cUB = cTBuf.Get<T>();
    DataCopyPad(cUB, cGM[base], copyC, padC);
    SetFlag<HardEvent::MTE2_V>(0);
    WaitFlag<HardEvent::MTE2_V>(0);
    int32_t offset = 0;
    while (offset < count) {
        int32_t row = static_cast<int32_t>((base + offset) / n);
        int64_t rowEndFlat = static_cast<int64_t>(row + 1) * n;
        int64_t chunkEnd = base + count;
        int32_t elemEnd = static_cast<int32_t>((rowEndFlat < chunkEnd) ? rowEndFlat : chunkEnd);
        int32_t elemCount = elemEnd - static_cast<int32_t>(base + offset);
        float beta_r = SpltGetScalarFromUB(betaVecUB, row);
        if constexpr (std::is_same_v<T, float>) {
            Muls(tempUB[offset], cUB[offset], beta_r, elemCount);
        } else {
            LocalTensor<float> cFp32UB = cFp32Buf.Get<float>();
            if constexpr (std::is_same_v<T, __bf16>) {
                SpltCastBf16ToFp32Vec(cFp32UB, cUB[offset], elemCount);
            } else if constexpr (std::is_same_v<T, int8_t>) {
                SpltScalarCastLoop<float, T>(cFp32UB, cUB[offset], elemCount);
                PipeBarrier<PIPE_V>();
            } else {
                Cast(cFp32UB, cUB[offset], AscendC::RoundMode::CAST_NONE, elemCount);
            }
            PipeBarrier<PIPE_V>();
            Muls(tempUB[offset], cFp32UB, beta_r, elemCount);
        }
        PipeBarrier<PIPE_V>();
        Add(dFp32UB[offset], dFp32UB[offset], tempUB[offset], elemCount);
        offset += elemCount;
    }
}

// 以标量 beta 施加 beta*C（非向量缩放路径）。
template <typename T>
__aicore__ inline void SpltEpilogueChunkApplyBetaScalar(
    LocalTensor<float>& dFp32UB, LocalTensor<float>& tempUB,
    TBuf<TPosition::VECCALC>& cTBuf, TBuf<TPosition::VECCALC>& cFp32Buf,
    GlobalTensor<T>& cGM,
    int64_t base, int32_t count,
    DataCopyExtParams& copyC, DataCopyPadExtParams<T>& padC, float beta)
{
    LocalTensor<T> cUB = cTBuf.Get<T>();
    DataCopyPad(cUB, cGM[base], copyC, padC);
    SetFlag<HardEvent::MTE2_V>(0);
    WaitFlag<HardEvent::MTE2_V>(0);
    if constexpr (std::is_same_v<T, float>) {
        Muls(tempUB, cUB, beta, static_cast<int32_t>(count));
    } else {
        LocalTensor<float> cFp32UB = cFp32Buf.Get<float>();
        if constexpr (std::is_same_v<T, __bf16>) {
            SpltCastBf16ToFp32Vec(cFp32UB, cUB, static_cast<int32_t>(count));
        } else if constexpr (std::is_same_v<T, int8_t>) {
            SpltScalarCastLoop<float, T>(cFp32UB, cUB, static_cast<int32_t>(count));
            PipeBarrier<PIPE_V>();
        } else {
            Cast(cFp32UB, cUB, AscendC::RoundMode::CAST_NONE, static_cast<int32_t>(count));
        }
        Muls(tempUB, cFp32UB, beta, static_cast<int32_t>(count));
    }
    Add(dFp32UB, dFp32UB, tempUB, static_cast<int32_t>(count));
}

// ============================================================================
// bias + activation 的 epilogue 辅助函数。
// epilogue 链：SplitK reduce -> alpha -> beta*C -> bias -> activation -> Cast -> 写 D
// bias 和 activation 在 FP32 域（dFp32UB）上操作。
// ============================================================================

// 非融合（chunk 式）路径中向 dFp32UB 施加 bias。
// bias 按行广播：chunk 可能跨行，逐行取出 bias[r] 标量，对该行片段做 Adds。
// 仿 SpltEpilogueChunkApplyAlpha 的 per-row 分段模式。
__aicore__ inline void SpltEpilogueChunkApplyBias(
    LocalTensor<float>& dFp32UB, LocalTensor<float>& biasVecUB,
    int64_t base, int32_t count, int32_t n)
{
    if (n == 0) { return; }
    int32_t offset = 0;
    while (offset < count) {
        int32_t row = static_cast<int32_t>((base + offset) / n);
        int64_t rowEndFlat = static_cast<int64_t>(row + 1) * n;
        int64_t chunkEnd = base + count;
        int32_t elemEnd = static_cast<int32_t>((rowEndFlat < chunkEnd) ? rowEndFlat : chunkEnd);
        int32_t elemCount = elemEnd - static_cast<int32_t>(base + offset);
        float bias_r = SpltGetScalarFromUB(biasVecUB, row);
        Adds(dFp32UB[offset], dFp32UB[offset], bias_r, elemCount);
        PipeBarrier<PIPE_V>();
        offset += elemCount;
    }
}

// 施加 ReLU 激活：D = min(upperBound, max(threshold, D))。
// 在 FP32 域（dFp32UB）操作。count 为元素数。
__aicore__ inline void SpltApplyReLU(
    LocalTensor<float>& dFp32UB, int32_t count,
    float reluThreshold, float reluUpperBound)
{
    Maxs(dFp32UB, dFp32UB, reluThreshold, count);
    PipeBarrier<PIPE_V>();
    Mins(dFp32UB, dFp32UB, reluUpperBound, count);
    PipeBarrier<PIPE_V>();
}

// 施加 GeLU 激活（sigmoid 等价实现，11 步，含 Exp clamp）。
// 输入 x = dFp32UB，输出回写 dFp32UB。geluTemp 复用 tempBuf。
// 逐行执行（per-row, count=curN），geluTemp 需要一行大小 buffer。
// 步骤 6 clamp z<=20 防 Exp 溢出（不改变 golden 语义，sigmoid(20)≈1.0）。
// LocalTensor 按值传递（轻量级 handle，类似指针，避免临时对象无法绑定到非 const 引用）。
__aicore__ inline void SpltApplyGeLU(
    LocalTensor<float> dFp32UB, LocalTensor<float> geluTemp,
    int32_t count, float geluScaling)
{
    // 步骤 1：geluTemp = x²
    Mul(geluTemp, dFp32UB, dFp32UB, count);
    PipeBarrier<PIPE_V>();
    // 步骤 2：geluTemp = 0.044715 * x²
    Muls(geluTemp, geluTemp, 0.044715f, count);
    PipeBarrier<PIPE_V>();
    // 步骤 3：geluTemp = 0.044715 * x³
    Mul(geluTemp, geluTemp, dFp32UB, count);
    PipeBarrier<PIPE_V>();
    // 步骤 4：geluTemp = x + 0.044715 * x³
    Add(geluTemp, geluTemp, dFp32UB, count);
    PipeBarrier<PIPE_V>();
    // 步骤 5：z = sqrt(8/π) * (x + 0.044715 * x³)
    Muls(geluTemp, geluTemp, 1.5957691f, count);
    PipeBarrier<PIPE_V>();
    // 步骤 6：clamp z <= 20.0f 防 Exp 溢出
    Mins(geluTemp, geluTemp, 20.0f, count);
    PipeBarrier<PIPE_V>();
    // 步骤 7：exp(z)
    Exp(geluTemp, geluTemp, count);
    PipeBarrier<PIPE_V>();
    // 步骤 8：x = geluScaling * x
    Muls(dFp32UB, dFp32UB, geluScaling, count);
    PipeBarrier<PIPE_V>();
    // 步骤 9：x = geluScaling * x * exp(z)
    Mul(dFp32UB, dFp32UB, geluTemp, count);
    PipeBarrier<PIPE_V>();
    // 步骤 10：geluTemp = 1 + exp(z)
    Adds(geluTemp, geluTemp, 1.0f, count);
    PipeBarrier<PIPE_V>();
    // 步骤 11：D = geluScaling * x * exp(z) / (1 + exp(z))
    Div(dFp32UB, dFp32UB, geluTemp, count);
    PipeBarrier<PIPE_V>();
}

// Cast FP32 -> OutType 并写回 GM（UB -> GM 经 DataCopyPad）。
template <typename OutType>
__aicore__ inline void SpltEpilogueChunkStoreOutput(
    LocalTensor<float>& dFp32UB, TBuf<TPosition::VECCALC>& dTBuf,
    TBuf<TPosition::VECCALC>& halfBuf, GlobalTensor<OutType>& dGM,
    int64_t base, int32_t count,
    DataCopyExtParams& copyAcc, DataCopyExtParams& copyOut)
{
    if constexpr (std::is_same_v<OutType, float>) {
        SetFlag<HardEvent::V_MTE3>(0);
        WaitFlag<HardEvent::V_MTE3>(0);
        DataCopyPad(dGM[base], dFp32UB, copyAcc);
    } else {
        LocalTensor<OutType> dUB = dTBuf.Get<OutType>();
        if constexpr (std::is_same_v<OutType, int8_t>) {
            Mins(dFp32UB, dFp32UB, 127.0f, static_cast<int32_t>(count));
            Maxs(dFp32UB, dFp32UB, -128.0f, static_cast<int32_t>(count));
            SpltCastFp32ToInt8(dUB, dFp32UB, halfBuf, static_cast<int32_t>(count));
        } else if constexpr (std::is_same_v<OutType, int32_t>) {
            Cast(dUB, dFp32UB, AscendC::RoundMode::CAST_ROUND, static_cast<int32_t>(count));
        } else if constexpr (std::is_same_v<OutType, __bf16>) {
            SpltCastFp32ToBf16Vec(dUB, dFp32UB, static_cast<int32_t>(count));
        } else {
            Cast(dUB, dFp32UB, AscendC::RoundMode::CAST_ROUND, static_cast<int32_t>(count));
        }
        SetFlag<HardEvent::V_MTE3>(0);
        WaitFlag<HardEvent::V_MTE3>(0);
        DataCopyPad(dGM[base], dUB, copyOut);
    }
}

// 处理 epilogue 循环的一个 chunk（splitK reduce + alpha/beta + 输出）。
// 从 SpltEpilogueImpl 提取以降低 NBNC。所有 SetFlag/WaitFlag 保持不变。
// v2：模板化为 (T, OutType)。AccType 恒为 float（非快速路径下 INT8 的
// int32 temp 在加载时 Cast 为 float）。INT8 输出在 Cast 前用
// Mins/Maxs clamp 到 [-128,127]（CAST_SATURATE 不可用）。
// 新增 halfBuf 参数：float→int8_t 需要两步 Cast。
// 新增 alpha=1+beta=0 时的 int32 快速路径分发（不经过 float）。
template <typename T, typename OutType>
__aicore__ inline void SpltEpilogueProcessChunk(
    LocalTensor<float>& accUB, LocalTensor<float>& tempUB, LocalTensor<float>& dFp32UB,
    TBuf<TPosition::VECCALC>& cTBuf, TBuf<TPosition::VECCALC>& cFp32Buf,
    TBuf<TPosition::VECCALC>& dTBuf, TBuf<TPosition::VECCALC>& halfBuf,
    GlobalTensor<typename SpltL0CTypeTrait<T>::type>& tempGM,
    GlobalTensor<T>& cGM, GlobalTensor<OutType>& dGM,
    int64_t base, int32_t count, int32_t splitK, float alpha, float beta,
    int64_t totalElem, int32_t n,
    int32_t alphaVectorScaling, int32_t betaVectorScaling,
    LocalTensor<float>& alphaVecUB, LocalTensor<float>& betaVecUB,
    LocalTensor<float>& biasVecUB,
    uint64_t biasDevPtr, int32_t activationType,
    float reluThreshold, float reluUpperBound, float geluScaling)
{
    using TempType = typename SpltL0CTypeTrait<T>::type;
    // INT32 快速路径：alpha=1+beta=0+无 bias+无 activation 时直接 int32 累加输出
    // 有 bias 或 activation 时必须走 FP32 域（bias Adds / GeLU Exp 等在 FP32 域操作）
    if constexpr (!std::is_same_v<TempType, float>) {
        if (alphaVectorScaling == 0 && alpha == 1.0f && beta == 0.0f
            && biasDevPtr == 0 && activationType == 0) {
            SpltEpilogueInt32AccPath<OutType>(accUB, tempUB, dTBuf, halfBuf, tempGM, dGM,
                                              base, count, splitK, totalElem);
            return;
        }
    }
    DataCopyExtParams copyAcc{1, 0, 0, 0, 0};
    DataCopyPadExtParams<TempType> padAcc{false, 0, 0, TempType(0)};
    DataCopyExtParams copyOut{1, 0, 0, 0, 0};
    DataCopyPadExtParams<OutType> padOut{false, 0, 0, OutType(0)};
    DataCopyExtParams copyC{1, 0, 0, 0, 0};
    DataCopyPadExtParams<T> padC{false, 0, 0, T(0)};
    copyAcc.blockLen = static_cast<uint32_t>(count * sizeof(TempType));
    copyOut.blockLen = static_cast<uint32_t>(count * sizeof(OutType));
    copyC.blockLen = static_cast<uint32_t>(count * sizeof(T));

    SpltEpilogueChunkSplitKReduce<T, TempType>(accUB, tempUB, tempGM,
                                                base, count, splitK, totalElem, copyAcc, padAcc);
    SpltEpilogueChunkApplyAlpha<T>(dFp32UB, accUB, alphaVecUB,
                                    base, count, n, alphaVectorScaling, alpha);
    if (betaVectorScaling == 1 || beta != 0.0f) {
        if (betaVectorScaling == 1) {
            SpltEpilogueChunkApplyBetaVec<T>(dFp32UB, tempUB, cTBuf, cFp32Buf,
                                              cGM, base, count, n, copyC, padC, betaVecUB);
        } else {
            SpltEpilogueChunkApplyBetaScalar<T>(dFp32UB, tempUB, cTBuf, cFp32Buf,
                                                 cGM, base, count, copyC, padC, beta);
        }
    }
    // ⑤ bias → 累加到 dFp32UB
    if (biasDevPtr != 0) {
        SpltEpilogueChunkApplyBias(dFp32UB, biasVecUB, base, count, n);
    }
    // ⑥ activation
    if (activationType == 1) {
        SpltApplyReLU(dFp32UB, static_cast<int32_t>(count), reluThreshold, reluUpperBound);
    } else if (activationType == 2) {
        // 非融合路径：chunk 级批量执行 GeLU，geluTemp 复用 tempUB
        SpltApplyGeLU(dFp32UB, tempUB, static_cast<int32_t>(count), geluScaling);
    }
    SpltEpilogueChunkStoreOutput<OutType>(dFp32UB, dTBuf, halfBuf, dGM,
                                               base, count, copyAcc, copyOut);
    SetFlag<HardEvent::MTE3_MTE2>(0);
    WaitFlag<HardEvent::MTE3_MTE2>(0);
}

// 共享的缩放辅助函数：从 SpltEpilogueImpl 与 SpltFusedEpilogueImpl
// 提取，以消除重复代码（alphaVecGM/betaVecGM 构建与 UB 加载）。
// flag 操作完全保持不变。

// 向量缩放启用时，构建 alpha/beta 向量的 global buffer。
__aicore__ inline void SpltInitScalingGM(
    GlobalTensor<float>& alphaVecGM, GlobalTensor<float>& betaVecGM,
    uint64_t alphaDevPtr, uint64_t betaDevPtr, int32_t m,
    int32_t alphaVectorScaling, int32_t betaVectorScaling)
{
    if (alphaVectorScaling == 1) {
        alphaVecGM.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(alphaDevPtr),
                                    static_cast<uint64_t>(m));
    }
    if (betaVectorScaling == 1) {
        betaVecGM.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(betaDevPtr),
                                   static_cast<uint64_t>(m));
    }
}

// 向量缩放启用时，分配 UB 并从 GM 加载 alpha/beta 向量。
__aicore__ inline void SpltLoadScalingUB(
    TBuf<TPosition::VECCALC>& alphaVecBuf, TBuf<TPosition::VECCALC>& betaVecBuf,
    LocalTensor<float>& alphaVecUB, LocalTensor<float>& betaVecUB,
    GlobalTensor<float>& alphaVecGM, GlobalTensor<float>& betaVecGM,
    TPipe& pipe, int32_t m,
    int32_t alphaVectorScaling, int32_t betaVectorScaling)
{
    if (alphaVectorScaling == 1) {
        pipe.InitBuffer(alphaVecBuf, static_cast<uint32_t>(static_cast<size_t>(m) * sizeof(float)));
        alphaVecUB = alphaVecBuf.Get<float>();
        DataCopyExtParams copyAlpha{1, static_cast<uint32_t>(static_cast<size_t>(m) * sizeof(float)), 0, 0, 0};
        DataCopyPadExtParams<float> padAlpha{false, 0, 0, 0.0f};
        DataCopyPad(alphaVecUB, alphaVecGM[0], copyAlpha, padAlpha);
        SetFlag<HardEvent::MTE2_V>(0);
        WaitFlag<HardEvent::MTE2_V>(0);
    }
    if (betaVectorScaling == 1) {
        pipe.InitBuffer(betaVecBuf, static_cast<uint32_t>(static_cast<size_t>(m) * sizeof(float)));
        betaVecUB = betaVecBuf.Get<float>();
        DataCopyExtParams copyBeta{1, static_cast<uint32_t>(static_cast<size_t>(m) * sizeof(float)), 0, 0, 0};
        DataCopyPadExtParams<float> padBeta{false, 0, 0, 0.0f};
        DataCopyPad(betaVecUB, betaVecGM[0], copyBeta, padBeta);
        SetFlag<HardEvent::MTE2_V>(0);
        WaitFlag<HardEvent::MTE2_V>(0);
    }
}

// SpltEpilogueImpl 的 TBuf 初始化（按 CHUNK 分配）。
template <typename T, typename OutType>
__aicore__ inline void SpltEpilogueInitBuffers(
    TPipe& pipe,
    TBuf<TPosition::VECCALC>& accBuf, TBuf<TPosition::VECCALC>& tempBuf,
    TBuf<TPosition::VECCALC>& dFp32Buf, TBuf<TPosition::VECCALC>& dTBuf,
    TBuf<TPosition::VECCALC>& cTBuf, TBuf<TPosition::VECCALC>& cFp32Buf,
    TBuf<TPosition::VECCALC>& halfBuf,
    float beta, int32_t betaVectorScaling)
{
    using TempType = typename SpltL0CTypeTrait<T>::type;
    constexpr int32_t CHUNK = 256;
    pipe.InitBuffer(accBuf, CHUNK * sizeof(float));
    pipe.InitBuffer(tempBuf, CHUNK * sizeof(float));
    pipe.InitBuffer(dFp32Buf, CHUNK * sizeof(float));
    if constexpr (!std::is_same_v<TempType, float>) {
        pipe.InitBuffer(dTBuf, CHUNK * sizeof(float));
    }
    if constexpr (std::is_same_v<OutType, int8_t>) {
        pipe.InitBuffer(halfBuf, CHUNK * sizeof(half));
    }
    // [LOW-8] 移除 BF16 的 halfBuf 分配：SpltCastFp32ToBf16Vec 不使用
    // halfBuf（Vector Cast<bfloat16_t,float> 无需中间缓冲）。
    if (betaVectorScaling == 1 || beta != 0.0f) {
        pipe.InitBuffer(cTBuf, CHUNK * sizeof(T));
        if constexpr (!std::is_same_v<T, float>) {
            pipe.InitBuffer(cFp32Buf, CHUNK * sizeof(float));
        }
    }
    if constexpr (!std::is_same_v<OutType, float> && std::is_same_v<TempType, float>) {
        pipe.InitBuffer(dTBuf, CHUNK * sizeof(OutType));
    }
}

// 为 SpltEpilogueBatchLoop 设置逐 batch 的 GM 指针。
// 将 temp/c/d/bias GM 按 batch stride 偏移；biasStride != 0 时重载 bias。
template <typename T, typename OutType>
__aicore__ inline void SpltEpilogueOffsetBatchGM(
    const AclsparseltTilingData& td,
    GlobalTensor<typename SpltL0CTypeTrait<T>::type>& tempGM,
    GlobalTensor<T>& cGM, GlobalTensor<OutType>& dGM,
    GlobalTensor<SpltBiasDType<T>>& biasVecGM,
    LocalTensor<float>& biasVecUB, TBuf<TPosition::VECCALC>& biasRawBuf,
    GM_ADDR tempGm, GM_ADDR cGm, GM_ADDR dGm,
    int32_t b, int32_t m, int64_t totalElem)
{
    using TempType = typename SpltL0CTypeTrait<T>::type;
    using BiasType = SpltBiasDType<T>;
    const int64_t tempBatchElems = static_cast<int64_t>(td.splitK) * totalElem;
    tempGM.SetGlobalBuffer(reinterpret_cast<__gm__ TempType*>(tempGm)
                           + static_cast<int64_t>(b) * tempBatchElems,
                           static_cast<uint64_t>(tempBatchElems));
    cGM.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(cGm)
                        + static_cast<int64_t>(b) * td.batchStrideC,
                        static_cast<uint64_t>(totalElem));
    dGM.SetGlobalBuffer(reinterpret_cast<__gm__ OutType*>(dGm)
                        + static_cast<int64_t>(b) * td.batchStrideD,
                        static_cast<uint64_t>(totalElem));
    if (td.biasDevPtr != 0 && td.biasStride != 0) {
        biasVecGM.SetGlobalBuffer(reinterpret_cast<__gm__ BiasType*>(td.biasDevPtr)
                                  + static_cast<int64_t>(b) * td.biasStride,
                                  static_cast<uint64_t>(m));
        SpltLoadBiasFullToFp32<BiasType>(biasVecUB, biasRawBuf, biasVecGM, m);
    }
}

// SpltEpilogueImpl 的 batch 循环：遍历 batch，设置逐 batch GM 指针、
// 按需重载 bias，并执行 chunk 处理循环。
// 提取以降低 SpltEpilogueImpl 的 NBNC（#8，原 86，目标 <=50）。
// flag 操作与循环顺序完全保持不变。
template <typename T, typename OutType>
__aicore__ inline void SpltEpilogueBatchLoop(
    const AclsparseltTilingData& td,
    GlobalTensor<typename SpltL0CTypeTrait<T>::type>& tempGM,
    GlobalTensor<T>& cGM, GlobalTensor<OutType>& dGM,
    GlobalTensor<SpltBiasDType<T>>& biasVecGM,
    LocalTensor<float>& biasVecUB, TBuf<TPosition::VECCALC>& biasRawBuf,
    LocalTensor<float>& accUB, LocalTensor<float>& tempUB, LocalTensor<float>& dFp32UB,
    TBuf<TPosition::VECCALC>& cTBuf, TBuf<TPosition::VECCALC>& cFp32Buf,
    TBuf<TPosition::VECCALC>& dTBuf, TBuf<TPosition::VECCALC>& halfBuf,
    LocalTensor<float>& alphaVecUB, LocalTensor<float>& betaVecUB,
    GM_ADDR tempGm, GM_ADDR cGm, GM_ADDR dGm, int32_t m,
    int32_t alphaVectorScaling, int32_t betaVectorScaling, float beta)
{
    const int32_t n = td.n;
    const int32_t splitK = td.splitK;
    const float alpha = td.alpha;
    const uint64_t biasDevPtr = td.biasDevPtr;
    const int32_t activationType = td.activationType;
    const float reluThreshold = td.reluThreshold;
    const float reluUpperBound = td.reluUpperBound;
    const float geluScaling = td.geluScaling;
    const int32_t numBatches = (td.numBatches > 0) ? td.numBatches : 1;
    const int32_t blockId = static_cast<int32_t>(GetBlockIdx());
    const int32_t blockNum = static_cast<int32_t>(GetBlockNum());
    const int64_t totalElem = static_cast<int64_t>(m) * static_cast<int64_t>(n);
    constexpr int32_t CHUNK = 256;

    for (int32_t b = 0; b < numBatches; ++b) {
        SpltEpilogueOffsetBatchGM<T, OutType>(td, tempGM, cGM, dGM, biasVecGM,
                                              biasVecUB, biasRawBuf,
                                              tempGm, cGm, dGm, b, m, totalElem);

        for (int64_t base = static_cast<int64_t>(blockId) * CHUNK; base < totalElem;
             base += static_cast<int64_t>(blockNum) * CHUNK) {
            int32_t count = (base + CHUNK <= totalElem) ? CHUNK : static_cast<int32_t>(totalElem - base);
            SpltEpilogueProcessChunk<T, OutType>(accUB, tempUB, dFp32UB, cTBuf, cFp32Buf, dTBuf, halfBuf,
                                        tempGM, cGM, dGM, base, count, splitK, alpha, beta,
                                        totalElem, n, alphaVectorScaling, betaVectorScaling,
                                        alphaVecUB, betaVecUB,
                                        biasVecUB, biasDevPtr, activationType,
                                        reluThreshold, reluUpperBound, geluScaling);
        }
    }
}

// ============================================================================
// v2：SpltEpilogueImpl 模板化为 (T, OutType)。AccType 恒为 float
// （INT8 的 int32 temp 在加载时 Cast 为 float）。非 INT8 路径：OutType == T。
template <typename T, typename OutType = T>
__aicore__ inline void SpltEpilogueImpl(GM_ADDR tempGm, GM_ADDR cGm, GM_ADDR dGm, GM_ADDR tilingGm)
{
    using TempType = typename SpltL0CTypeTrait<T>::type;  // float 或 int32_t
    using BiasType = SpltBiasDType<T>;  // bias dtype：FP32/FP16/BF16 时为 T，INT8 时为 float
    const AclsparseltTilingData td = splt_load_tiling(tilingGm);
    const int32_t m = td.m;
    const float beta = td.beta;
    const int32_t alphaVectorScaling = td.alphaVectorScaling;
    const int32_t betaVectorScaling = td.betaVectorScaling;
    const uint64_t biasDevPtr = td.biasDevPtr;
    const int64_t biasStride = td.biasStride;

    GlobalTensor<TempType> tempGM;
    GlobalTensor<T> cGM;
    GlobalTensor<OutType> dGM;

    GlobalTensor<float> alphaVecGM;
    GlobalTensor<float> betaVecGM;
    SpltInitScalingGM(alphaVecGM, betaVecGM, td.alphaDevPtr, td.betaDevPtr,
                       m, alphaVectorScaling, betaVectorScaling);

    GlobalTensor<BiasType> biasVecGM;
    TBuf<TPosition::VECCALC> biasVecBuf;
    TBuf<TPosition::VECCALC> biasRawBuf;
    LocalTensor<float> biasVecUB;
    TPipe pipe;
    if (biasDevPtr != 0) {
        SpltInitBiasBuffers<BiasType>(biasVecGM, biasVecBuf, biasRawBuf, biasVecUB,
                                       pipe, biasDevPtr, biasStride, m);
    }

    TBuf<TPosition::VECCALC> accBuf, tempBuf, cTBuf, cFp32Buf, dFp32Buf, dTBuf, halfBuf;
    TBuf<TPosition::VECCALC> alphaVecBuf, betaVecBuf;
    SpltEpilogueInitBuffers<T, OutType>(pipe, accBuf, tempBuf, dFp32Buf, dTBuf,
                                         cTBuf, cFp32Buf, halfBuf, beta, betaVectorScaling);
    LocalTensor<float> alphaVecUB;
    LocalTensor<float> betaVecUB;
    SpltLoadScalingUB(alphaVecBuf, betaVecBuf, alphaVecUB, betaVecUB,
                       alphaVecGM, betaVecGM, pipe, m,
                       alphaVectorScaling, betaVectorScaling);

    LocalTensor<float> accUB = accBuf.Get<float>();
    LocalTensor<float> tempUB = tempBuf.Get<float>();
    LocalTensor<float> dFp32UB = dFp32Buf.Get<float>();

    SpltEpilogueBatchLoop<T, OutType>(td, tempGM, cGM, dGM, biasVecGM, biasVecUB, biasRawBuf,
        accUB, tempUB, dFp32UB, cTBuf, cFp32Buf, dTBuf, halfBuf, alphaVecUB, betaVecUB,
        tempGm, cGm, dGm, m, alphaVectorScaling, betaVectorScaling, beta);
}

// ============================================================================
// [OPT-P2] FUSED MATMUL + EPILOGUE KERNEL（__mix__(1,2)）
//   splitK==1 路径：AIC 执行 matmul -> Fixpipe L0C->UB -> CrossCore 通知 AIV
//   AIV 执行 CrossCore 等待 -> Vector(alpha*acc+beta*C+Cast) -> DataCopyPad UB->GM(D)
//   消除 GM temp 往返 + 1 次 kernel 启动。
//
// 架构：__mix__(1,2) = 每 block 1 个 AIC + 2 个 AIV。AIC 与 AIV 位于
// 不同物理核、各有独立 UB。CopyL0C2UB 依据 subBlockId 将完整 tile 数据
// 路由到目标 AIV 的 UB。tile 在 AIV0（subBlockId=false）与 AIV1
// （subBlockId=true）间交替以均衡负载。
// 简化起见不做乒乓（每 AIV 单 UB 槽位）；正确性优先。
// ============================================================================

// CrossCore 同步 flag 常量（块内 MODE 4，blaze-sync-patterns §7）。
static constexpr uint8_t  SPLT_SYNC_MODE_4      = 4;    // 块内模式
static constexpr uint16_t SPLT_AIV0_SYNC_AIC    = 4;   // AIV0->AIC
static constexpr uint16_t SPLT_AIC_SYNC_AIV0    = 6;   // AIC->AIV0
static constexpr uint16_t SPLT_AIV1_SYNC_AIC    = 4 + 16;  // AIV1->AIC（FLAG_ID_MAX 偏移）
static constexpr uint16_t SPLT_AIC_SYNC_AIV1    = 6 + 16;  // AIC->AIV1

// ----------------------------------------------------------------------------
// AIC 侧：K 循环 + Fixpipe L0C->UB（subBlockId 交替）+ CrossCore
// ----------------------------------------------------------------------------
// SpltFusedMatmulCubeImpl 的 tile 循环驱动：m/n tile 遍历 + K 循环 +
// Fixpipe L0C->UB + CrossCore 同步。提取以降低 SpltFusedMatmulCubeImpl
// 的 NBNC（#4，76 -> 目标 <=50），并与 SpltMatmulCubeTileLoop 共享
// SpltKL1Loop（消除重复 #1）。flag 操作保持不变。
template <typename T, typename GmTensorA, typename GmTensorB>
__aicore__ inline void SpltFusedMatmulCubeTileLoop(
    const AclsparseltTilingData& td,
    GmTensorA& gmA, GmTensorB& gmB,
    const L1BufferConfig& l1cfg, int32_t blockId, int32_t blockNum,
    int32_t nTiles, int64_t totalTiles, uint32_t& localTileIdx)
{
    const int32_t m = td.m;
    const int32_t n = td.n;
    const int32_t k = td.k;
    const int32_t baseM = td.baseM;
    const int32_t baseN = td.baseN;
    const int32_t baseK = td.baseK;
    const int32_t kSegLen = td.kSegLen;

    for (int32_t tileIdx = blockId; static_cast<int64_t>(tileIdx) < totalTiles; tileIdx += blockNum) {
        const int32_t mTile = static_cast<int32_t>(static_cast<int64_t>(tileIdx) / nTiles);
        const int32_t nTile = static_cast<int32_t>(static_cast<int64_t>(tileIdx) % nTiles);
        const int32_t mPos = mTile * baseM;
        const int32_t nPos = nTile * baseN;
        const int32_t curM = (mPos + baseM <= m) ? baseM : (m - mPos);
        const int32_t curN = (nPos + baseN <= n) ? baseN : (n - nPos);
        const uint64_t curMAlign = static_cast<uint64_t>((curM + 1) & ~1);
        // UB nAlign 必须用 L0C C0（=16），不能用 dtype C0（FP32 为 8）。
        // Fixpipe 将 L0C NZ 分形（C0=16）写入 UB NDExt。UB 中的行 stride
        // 为 CeilAlign(curN, 16) 个元素。若 curNAlign 使用 dtype C0（8），行 stride
        // 与 AIV 期望值（CeilAlign(curN, 16)）不匹配，导致非 16 对齐 curN 的
        // 数据错位（如 curN=36 -> AIC stride=40，AIV stride=48）。
        const uint64_t curNAlign = static_cast<uint64_t>((curN + SPLT_L0C_C0 - 1) / SPLT_L0C_C0) * SPLT_L0C_C0;

        auto gmBlockARow = gmA.Slice(Te::MakeCoord(mPos, 0), Te::MakeShape(curM, k));
        // B 切片对两条路径相同（DNExt GM 布局已处理 transB）。
        auto gmBlockBCol = gmB.Slice(Te::MakeCoord(0, nPos), Te::MakeShape(k, curN));
        // v2：L0C 类型跟随 SpltL0CTypeTrait（FP32/FP16/BF16 为 float，INT8 为 int32_t）。
        auto tensorL0C = Te::MakeTensor(Te::MakeMemPtr<Te::Location::L0C, typename SpltL0CTypeTrait<T>::type>(0),
                                    Te::MakeFrameLayout<Te::NZLayoutPtn, SPLT_L0C_C0>(curM, curN));

        WaitFlag<HardEvent::FIX_M>(0);
        bool cmatrixInitVal = true;
        // kSegStart=0：融合路径 splitK==1，无 K 段偏移。
        SpltKL1Loop<T>(gmBlockARow, gmBlockBCol, tensorL0C,
                       curM, curN, k, kSegLen, 0, baseK, td.kL1Size,
                       cmatrixInitVal, l1cfg);

        // ---- [OPT-P2] Fixpipe L0C -> UB（经 subBlockId 路由到目标 AIV）----
        // 交替规则：localTileIdx 偶数 -> AIV0（subBlockId=false），奇数 -> AIV1。
        const bool toAiv1 = (localTileIdx & 0x1) == 1;
        const uint16_t aivWaitFlag = toAiv1 ? SPLT_AIV1_SYNC_AIC : SPLT_AIV0_SYNC_AIC;
        const uint16_t aivNotifyFlag = toAiv1 ? SPLT_AIC_SYNC_AIV1 : SPLT_AIC_SYNC_AIV0;

        // 等待目标 AIV 释放 UB（反向依赖，首轮由 AIV 预置）。
        CrossCoreWaitFlag<SPLT_SYNC_MODE_4, PIPE_FIX>(aivWaitFlag);

        // UB 目的张量位于偏移 0（单槽位，无乒乓）。
        // Te::NDExtLayoutPtn 的 C0 必须与 L0C C0（DAV-3510 上 =16）一致，
        // 不能用 dtype C0（FP32 为 32/sizeof(float)=8）。不匹配会导致
        // Fixpipe 产生错误的分形拆分，输出错乱。
        // v2：UB 张量类型跟随 L0C 类型（INT8 为 int32_t）。
        auto layoutUB = Te::MakeFrameLayout<Te::NDExtLayoutPtn, SPLT_L0C_C0>(curMAlign, curNAlign);
        auto ubTensor = Te::MakeTensor(Te::MakeMemPtr<Te::Location::UB, typename SpltL0CTypeTrait<T>::type>(0), layoutUB);

        Te::FixpipeParams fixpParams;
        fixpParams.unitFlag = static_cast<uint8_t>(SPLT_FINAL_ACCUMULATION);
        fixpParams.subBlockId = toAiv1;   // 路由到 AIV0 或 AIV1
        auto copyL0C2UB = Te::MakeCopy(Te::CopyL0C2UB{});
        Te::Copy(copyL0C2UB.with(fixpParams), ubTensor, tensorL0C);

        // 通知目标 AIV 数据已就绪。
        CrossCoreSetFlag<SPLT_SYNC_MODE_4, PIPE_FIX>(aivNotifyFlag);

        SetFlag<HardEvent::FIX_M>(0);
        localTileIdx++;
    }
}

// SpltFusedMatmulCubeImpl 的 transB 分发：构造 gmB（依据 transB 选择
// DNExt 或 NDExt）并调用 SpltFusedMatmulCubeTileLoop。
// 消除重复的 transB 分支代码（4 个分支 -> 2 个 + 辅助函数）。
template <typename T, typename GmTensorA>
__aicore__ inline void SpltFusedMatmulCubeDispatchB(
    const AclsparseltTilingData& td, GmTensorA& gmA, GM_ADDR bGm,
    const L1BufferConfig& l1cfg, int32_t blockId, int32_t blockNum,
    int32_t nTiles, int64_t totalTiles, uint32_t& localTileIdx)
{
    const int32_t k = td.k;
    const int32_t n = td.n;
    if (td.transB != 0) {
        auto gmB = Te::MakeTensor(Te::MakeMemPtr<Te::Location::GM>(reinterpret_cast<__gm__ T*>(bGm)),
                              Te::MakeFrameLayout<Te::DNExtLayoutPtn, Te::LayoutTraitDefault<T>>(k, n));
        SpltFusedMatmulCubeTileLoop<T>(td, gmA, gmB, l1cfg, blockId, blockNum,
                                       nTiles, totalTiles, localTileIdx);
    } else {
        auto gmB = Te::MakeTensor(Te::MakeMemPtr<Te::Location::GM>(reinterpret_cast<__gm__ T*>(bGm)),
                              Te::MakeFrameLayout<Te::NDExtLayoutPtn, Te::LayoutTraitDefault<T>>(k, n));
        SpltFusedMatmulCubeTileLoop<T>(td, gmA, gmB, l1cfg, blockId, blockNum,
                                       nTiles, totalTiles, localTileIdx);
    }
}


template <typename T>
__aicore__ inline void SpltFusedMatmulCubeImpl(GM_ADDR aPrunedGm, GM_ADDR bGm,
                                               GM_ADDR dGm, GM_ADDR tilingGm)
{
    const AclsparseltTilingData td = splt_load_tiling(tilingGm);
    const int32_t m = td.m;
    const int32_t n = td.n;
    const int32_t k = td.k;
    const int32_t baseM = td.baseM;
    const int32_t baseN = td.baseN;
    // batch 参数
    const int32_t numBatches = (td.numBatches > 0) ? td.numBatches : 1;

    const int32_t blockId = static_cast<int32_t>(GetBlockIdx());
    const int32_t blockNum = static_cast<int32_t>(GetBlockNum());
    const int32_t mTiles = (m + baseM - 1) / baseM;
    const int32_t nTiles = (n + baseN - 1) / baseN;
    // 使用 int64_t 防止 mTiles*nTiles 接近
    // INT32_MAX（大 m、n）时溢出。与非融合路径一致。
    // 每个 batch 内的 totalTiles（batch 循环在外层）
    const int64_t totalTiles = static_cast<int64_t>(mTiles) * static_cast<int64_t>(nTiles);
    if (totalTiles <= 0) { return; }

    // sparseTrans 处理：理由见 SpltMatmulCubeImpl。
    // DNExt/NDExt 产生不同类型，因此 A 与 B 均需在调用点分支。
    const L1BufferConfig l1cfg = InitL1DoubleBuffer<T>(baseM, baseN, td.kL1Size);
    SetMMLayoutTransform(true);
    // batch 外循环（GM 指针按 batchStride 偏移）
    // localTileIdx 每 batch 重置为 0，与 AIV 侧 SpltFusedEpilogueTileLoop
    // 的 per-batch 重置对齐。此前 localTileIdx 在循环外声明并跨 batch 累加，
    // 导致 batch 边界处 AIC 奇偶翻转而 AIV 重置为 0，下一个 batch 首个 tile 的
    // AIC→AIV 路由 flag 与 AIV 等待 flag 错配 → CrossCore 死锁。
    for (int32_t b = 0; b < numBatches; ++b) {
        uint32_t localTileIdx = 0;  // 每 batch 重置，与 AIV 侧对齐
        // GM 指针按 batchStride 偏移（batchStride 以元素数为单位）
        __gm__ T* aBase = reinterpret_cast<__gm__ T*>(aPrunedGm)
                          + static_cast<int64_t>(b) * td.batchStrideA;
        GM_ADDR bBatchGm = bGm + static_cast<int64_t>(b) * td.batchStrideB * static_cast<int64_t>(sizeof(T));
        if (td.sparseTrans != 0) {
            auto gmA = Te::MakeTensor(Te::MakeMemPtr<Te::Location::GM>(aBase),
                                  Te::MakeFrameLayout<Te::DNExtLayoutPtn, Te::LayoutTraitDefault<T>>(m, k));
            SpltFusedMatmulCubeDispatchB<T>(td, gmA, bBatchGm, l1cfg, blockId, blockNum,
                                            nTiles, totalTiles, localTileIdx);
        } else {
            auto gmA = Te::MakeTensor(Te::MakeMemPtr<Te::Location::GM>(aBase),
                                  Te::MakeFrameLayout<Te::NDExtLayoutPtn, Te::LayoutTraitDefault<T>>(m, k));
            SpltFusedMatmulCubeDispatchB<T>(td, gmA, bBatchGm, l1cfg, blockId, blockNum,
                                            nTiles, totalTiles, localTileIdx);
        }
        // CrossCore drain 每 batch 执行：等待本 batch 各 AIV 释放 UB。
        // 若只 wait 最后一个 tile 的目标 AIV，另一路 AIV 的"最后释放" flag
        // 会残留置位；下一 batch 该 AIV 又在首个 tile 前预发一次释放
        // SetFlag（见 SpltFusedEpilogueTileLoop），两代 set 叠加使 AIC 在
        // 下一个 batch 的 tile WaitFlag 消费到旧 set —— AIC 在 AIV 尚未
        // 就绪时就 CopyL0C2UB 覆写 UB，破坏该 batch 首批 tile 的数据
        // （batch>1 且多 tile 时表现为部分 tile 首行数据陈旧）。按 blaze
        // ctor/dtor 预发排空模式，每 batch 结束把本 batch 实际参与的两路
        // AIV 的释放 flag 全部排空，保证 CrossCore set/wait 在 batch 边界
        // 严格配对。
        if (localTileIdx >= 2) {
            // 两路 AIV 都处理过 tile：排空 AIV0 与 AIV1 的最后释放。
            CrossCoreWaitFlag<SPLT_SYNC_MODE_4, PIPE_FIX>(SPLT_AIV0_SYNC_AIC);
            CrossCoreWaitFlag<SPLT_SYNC_MODE_4, PIPE_FIX>(SPLT_AIV1_SYNC_AIC);
        } else if (localTileIdx == 1) {
            // 仅 1 个 tile（localTileIdx=0 → 必为 AIV0）：只排空 AIV0。
            CrossCoreWaitFlag<SPLT_SYNC_MODE_4, PIPE_FIX>(SPLT_AIV0_SYNC_AIC);
        }
    }

    // 排空 L1 双缓冲 flag。
    DrainCubeFlags();

    SetMMLayoutTransform(false);
}

// ----------------------------------------------------------------------------
// SpltFusedEpilogueImpl 的共享输出辅助函数。
// 提取以消除重复代码（#11）并降低 NBNC/复杂度/嵌套深度。
// 关键约束：SetFlag/WaitFlag 位置完全保持不变。
// v2：模板化为 AccType（FP32/FP16/BF16 为 float，INT8 为 int32_t）与
// OutType（输出 dtype）。INT8 输出用 CAST_SATURATE；INT32 用 ROUND。
// ----------------------------------------------------------------------------

// FP32 输出：V_MTE3 同步 + 批量/逐行 DataCopyPad（OutType=float 时调用）。
__aicore__ inline void SpltFusedEpilogueOutputFp32(
    const LocalTensor<float>& srcUB, GlobalTensor<float>& dGM,
    int32_t mPos, int32_t nPos, int32_t curM, int32_t curN,
    uint64_t curNAlign, int32_t n)
{
    SetFlag<HardEvent::V_MTE3>(0);
    WaitFlag<HardEvent::V_MTE3>(0);
    if (curN % 8 == 0) {
        const uint32_t srcStride = static_cast<uint32_t>(
            (curNAlign - curN) * sizeof(float) / 32);
        const uint32_t dstStride = static_cast<uint32_t>(
            (n - curN) * sizeof(float));
        DataCopyExtParams copyBatch{
            static_cast<uint16_t>(curM),
            static_cast<uint32_t>(curN * sizeof(float)),
            srcStride, dstStride, 0};
        DataCopyPad(dGM[static_cast<int64_t>(mPos) * n + nPos], srcUB, copyBatch);
    } else {
        DataCopyExtParams copyOut{1, 0, 0, 0, 0};
        for (int32_t r = 0; r < curM; ++r) {
            copyOut.blockLen = static_cast<uint32_t>(curN * sizeof(float));
            DataCopyPad(dGM[static_cast<int64_t>(mPos + r) * n + nPos],
                        srcUB[r * curNAlign], copyOut);
        }
    }
}

// 非 FP32 输出：逐行 Cast + V_MTE3 + DataCopyPad + MTE3_V。
// v2：INT8 输出 —— 先 Mins/Maxs clamp 再 Cast ROUND（CAST_SATURATE 不可用）。
// [OPT-BF16-VEC] BF16 输出 —— Vector Cast<bfloat16_t, float, CAST_ROUND>。
// dFp32Buf 用作 INT8 clamp 的 float 中间缓冲（与输出 dTBuf 分开）。
// 新增 halfBuf 参数：float→int8_t 需要两步
// Cast（float→half→int8_t），因为 DAV-3510 不支持 float→int8 直转。
template <typename OutType>
__aicore__ inline void SpltFusedEpilogueOutputCast(
    const LocalTensor<float>& srcUB, TBuf<TPosition::VECCALC>& dTBuf,
    TBuf<TPosition::VECCALC>& dFp32Buf, TBuf<TPosition::VECCALC>& halfBuf,
    GlobalTensor<OutType>& dGM,
    int32_t mPos, int32_t nPos, int32_t curM, int32_t curN,
    uint64_t curNAlign, int32_t n)
{
    LocalTensor<OutType> dUB = dTBuf.Get<OutType>();
    DataCopyExtParams copyOut{1, 0, 0, 0, 0};
    for (int32_t r = 0; r < curM; ++r) {
        if constexpr (std::is_same_v<OutType, int8_t>) {
            // INT8：float 先 clamp 到 [-128, 127]，再经 float→half→int8 Cast 为 int8。
            // DAV-3510 不支持 float→int8_t 直接
            // Cast；使用 SpltCastFp32ToInt8（两步 Vector Cast）。
            LocalTensor<float> clampUB = dFp32Buf.Get<float>();
            Mins(clampUB, srcUB[r * curNAlign], 127.0f, curN);
            Maxs(clampUB, clampUB, -128.0f, curN);
            SpltCastFp32ToInt8(dUB, clampUB, halfBuf, curN);
        } else if constexpr (std::is_same_v<OutType, __bf16>) {
            // [OPT-BF16-VEC] 向量化 FP32→BF16 Cast（取代标量循环）。
            // SpltCastFp32ToBf16Vec 直接使用 Vector Cast<bfloat16_t,float>；
            // 无需中间缓冲（dFp32Buf/halfBuf）。
            SpltCastFp32ToBf16Vec(dUB, srcUB[r * curNAlign], curN);
        } else {
            Cast(dUB, srcUB[r * curNAlign], AscendC::RoundMode::CAST_ROUND, curN);
        }
        SetFlag<HardEvent::V_MTE3>(0);
        WaitFlag<HardEvent::V_MTE3>(0);
        copyOut.blockLen = static_cast<uint32_t>(curN * sizeof(OutType));
        DataCopyPad(dGM[static_cast<int64_t>(mPos + r) * n + nPos], dUB, copyOut);
        SetFlag<HardEvent::MTE3_V>(0);
        WaitFlag<HardEvent::MTE3_V>(0);
    }
}

// 融合 INT8→INT32 快速路径的 int32 UB→GM 直接输出。
// 无 Cast、无 float 中间量 —— 对大累加值保持精确的 int32 精度
// （float 在 [-2^24, 2^24] 之外损失精度）。
// 逻辑镜像 SpltFusedEpilogueOutputFp32（sizeof(int32_t)==sizeof(float)==4）。
__aicore__ inline void SpltFusedEpilogueOutputInt32Direct(
    const LocalTensor<int32_t>& srcUB, GlobalTensor<int32_t>& dGM,
    int32_t mPos, int32_t nPos, int32_t curM, int32_t curN,
    uint64_t curNAlign, int32_t n)
{
    SetFlag<HardEvent::V_MTE3>(0);
    WaitFlag<HardEvent::V_MTE3>(0);
    if (curN % 8 == 0) {
        const uint32_t srcStride = static_cast<uint32_t>(
            (curNAlign - curN) * sizeof(int32_t) / 32);
        const uint32_t dstStride = static_cast<uint32_t>(
            (n - curN) * sizeof(int32_t));
        DataCopyExtParams copyBatch{
            static_cast<uint16_t>(curM),
            static_cast<uint32_t>(curN * sizeof(int32_t)),
            srcStride, dstStride, 0};
        DataCopyPad(dGM[static_cast<int64_t>(mPos) * n + nPos], srcUB, copyBatch);
    } else {
        DataCopyExtParams copyOut{1, 0, 0, 0, 0};
        for (int32_t r = 0; r < curM; ++r) {
            copyOut.blockLen = static_cast<uint32_t>(curN * sizeof(int32_t));
            DataCopyPad(dGM[static_cast<int64_t>(mPos + r) * n + nPos],
                        srcUB[r * curNAlign], copyOut);
        }
    }
}

// [OPT-INT8-VEC] 融合 INT8→INT8 快速路径的向量化 int32→int8 clamp。
// 以 Vector 操作取代标量 GetValue/SetValue 循环（128×128 需 2.1ms）：
//   1. Cast int32→float（Vector，CAST_NONE —— [-2^24, 2^24] 内精确）
//   2. Mins/Maxs clamp 到 [-128, 127]（Vector）
//   3. SpltCastFp32ToInt8（float→half→int8，两步 Vector Cast）
// float 中间结果对 [-128,127] 内（clamp 后）的所有 int32 值精确，
// 且 Cast int32→float 在 [-2^24, 2^24] 内精确，覆盖所有实际
// 累加结果（alpha=1+beta=0 快速路径，无缩放）。
// flag 模式镜像 SpltFusedEpilogueOutputCast（逐行 V_MTE3 + MTE3_V）。
// dFp32Buf/halfBuf 已在 SpltFusedEpilogueImpl 中为 INT8 快速路径预分配。
__aicore__ inline void SpltFusedEpilogueOutputInt8VecClamp(
    const LocalTensor<int32_t>& srcUB, TBuf<TPosition::VECCALC>& dTBuf,
    TBuf<TPosition::VECCALC>& dFp32Buf, TBuf<TPosition::VECCALC>& halfBuf,
    GlobalTensor<int8_t>& dGM,
    int32_t mPos, int32_t nPos, int32_t curM, int32_t curN,
    uint64_t curNAlign, int32_t n)
{
    LocalTensor<int8_t> dUB = dTBuf.Get<int8_t>();
    LocalTensor<float> clampUB = dFp32Buf.Get<float>();
    DataCopyExtParams copyOut{1, 0, 0, 0, 0};
    for (int32_t r = 0; r < curM; ++r) {
        // 1. 本行 Cast int32 → float
        Cast(clampUB, srcUB[r * curNAlign], AscendC::RoundMode::CAST_NONE, curN);
        PipeBarrier<PIPE_V>();
        // 2. clamp 到 [-128, 127]（先 Mins 后 Maxs，均为 Vector）
        Mins(clampUB, clampUB, 127.0f, curN);
        Maxs(clampUB, clampUB, -128.0f, curN);
        // 3. 经 half Cast float → int8（SpltCastFp32ToInt8 执行 float→half→int8）
        SpltCastFp32ToInt8(dUB, clampUB, halfBuf, curN);
        // 4. 本行输出到 GM（flag 模式与 SpltFusedEpilogueOutputCast 一致）
        SetFlag<HardEvent::V_MTE3>(0);
        WaitFlag<HardEvent::V_MTE3>(0);
        copyOut.blockLen = static_cast<uint32_t>(curN * sizeof(int8_t));
        DataCopyPad(dGM[static_cast<int64_t>(mPos + r) * n + nPos], dUB, copyOut);
        SetFlag<HardEvent::MTE3_V>(0);
        WaitFlag<HardEvent::MTE3_V>(0);
    }
}

// 快速路径：alpha=1、beta=0 -> acc 直接输出到 GM。
// v2：INT8 路径（AccType=int32_t）需要 Cast int32 -> float -> OutType
// （INT8 输出：经 Mins/Maxs clamp；INT32 输出：AccType==OutType 时 int32 直接写）。
// 新增 curMAlign 参数；INT8 的 Cast int32->float 必须覆盖整个
// tile（curMAlign * curNAlign 个元素），而不能只覆盖 curN（一行）。用 curN 会使
// accFp32 第 1 行及以后未初始化，造成约 94% 输出不匹配（15424/16384）。
// 新增 halfBuf 参数用于 float→half→int8 两步 Cast。
// INT8 快速路径现采用 int32 直接处理（不做 float Cast）：
//   - INT32 输出：int32 UB→GM 直接 DataCopyPad
//   - INT8 输出：[OPT-INT8-VEC] 向量化 Cast int32→float→clamp→int8
// float 路径仅保留给通用路径（alpha!=1 || beta!=0，容差允许）。
template <typename AccType, typename OutType>
__aicore__ inline void SpltFusedEpilogueFastPath(
    LocalTensor<AccType>& accUB, GlobalTensor<OutType>& dGM,
    TBuf<TPosition::VECCALC>& dTBuf, TBuf<TPosition::VECCALC>& dFp32Buf,
    TBuf<TPosition::VECCALC>& halfBuf,
    int32_t mPos, int32_t nPos, int32_t curM, int32_t curN,
    uint64_t curMAlign, uint64_t curNAlign, int32_t n)
{
    if constexpr (std::is_same_v<AccType, float>) {
        // FP32/FP16/BF16：acc 为 float，Cast（或直接）得到 OutType。
        if constexpr (std::is_same_v<OutType, float>) {
            SpltFusedEpilogueOutputFp32(accUB, dGM, mPos, nPos, curM, curN, curNAlign, n);
        } else {
            SpltFusedEpilogueOutputCast<OutType>(accUB, dTBuf, dFp32Buf, halfBuf, dGM,
                                                 mPos, nPos, curM, curN, curNAlign, n);
        }
    } else {
        // INT8 快速路径：int32 直接处理，跳过 float Cast。
        if constexpr (std::is_same_v<OutType, int32_t>) {
            // INT32 输出：int32 UB→GM 直接写（无 Cast、无 float 中间量）
            SpltFusedEpilogueOutputInt32Direct(accUB, dGM, mPos, nPos, curM, curN, curNAlign, n);
        } else {
            // INT8 输出：向量化 int32→float→clamp→int8（取代标量循环）
            SpltFusedEpilogueOutputInt8VecClamp(accUB, dTBuf, dFp32Buf, halfBuf, dGM,
                                                mPos, nPos, curM, curN, curNAlign, n);
        }
    }
}

// beta*C 累加循环（通用路径子步骤）。
// v2：C dtype == OutType（validate_descriptors 强制 C 与 D 一致）。
template <typename AccType, typename OutType, typename CType = OutType>
__aicore__ inline void SpltFusedEpilogueBetaC(
    LocalTensor<float>& dFp32UB, GlobalTensor<CType>& cGM,
    TBuf<TPosition::VECCALC>& cTBuf, TBuf<TPosition::VECCALC>& cFp32Buf,
    TBuf<TPosition::VECCALC>& tempBuf,
    int32_t mPos, int32_t nPos, int32_t curM, int32_t curN,
    uint64_t curNAlign, int32_t n, float beta,
    int32_t betaVectorScaling, LocalTensor<float>& betaVecUB)
{
    LocalTensor<CType> cUB = cTBuf.Get<CType>();
    LocalTensor<float> tempUB = tempBuf.Get<float>();
    DataCopyExtParams copyC{1, 0, 0, 0, 0};
    DataCopyPadExtParams<CType> padC{false, 0, 0, CType(0)};
    for (int32_t r = 0; r < curM; ++r) {
        copyC.blockLen = static_cast<uint32_t>(curN * sizeof(CType));
        DataCopyPad(cUB, cGM[static_cast<int64_t>(mPos + r) * n + nPos], copyC, padC);
        SetFlag<HardEvent::MTE2_V>(0);
        WaitFlag<HardEvent::MTE2_V>(0);
        float beta_r = (betaVectorScaling == 1)
            ? SpltGetScalarFromUB(betaVecUB, mPos + r) : beta;
        if constexpr (std::is_same_v<CType, float>) {
            Muls(tempUB, cUB, beta_r, curN);
        } else {
            LocalTensor<float> cFp32UB = cFp32Buf.Get<float>();
            // [OPT-BF16-VEC] BF16 使用向量化位操作 Cast（取代标量）。
            // tempBuf 复用为 Cast<uint32,uint16>+ShiftLeft 的 uint32 临时缓冲。
            // INT8 仍用标量 Cast：DAV-3510 Vector Cast
            // 不支持 int8_t→float（静默空操作，同 float→int8_t）。
            if constexpr (std::is_same_v<CType, __bf16>) {
                SpltCastBf16ToFp32Vec(cFp32UB, cUB, curN);
            } else if constexpr (std::is_same_v<CType, int8_t>) {
                SpltScalarCastLoop<float, CType>(cFp32UB, cUB, curN);
                PipeBarrier<PIPE_V>();
            } else {
                Cast(cFp32UB, cUB, AscendC::RoundMode::CAST_NONE, curN);
            }
            Muls(tempUB, cFp32UB, beta_r, curN);
        }
        Add(dFp32UB[r * curNAlign], dFp32UB[r * curNAlign], tempUB, curN);
        SetFlag<HardEvent::V_MTE2>(0);
        WaitFlag<HardEvent::V_MTE2>(0);
    }
}

// ----------------------------------------------------------------------------
// 从 SpltFusedEpilogueGeneralPath 提取的子辅助函数，降低 CCN（原 26）
// 与 NBNC（原 95）。每个辅助函数保持精确的 flag/pipe 操作。
// ----------------------------------------------------------------------------

// 对累加结果施加 alpha 缩放 -> dFp32UB。
// 同时处理标量 alpha 与逐行向量 alpha，覆盖 float 与
// int32 两种累加类型（INT8 需在 Muls 前 Cast int32->float）。
template <typename AccType>
__aicore__ inline void SpltFusedEpilogueApplyAlpha(
    LocalTensor<float>& dFp32UB, LocalTensor<AccType>& accUB,
    LocalTensor<float>& alphaVecUB,
    int32_t mPos, int32_t curM, int32_t curN,
    uint64_t curMAlign, uint64_t curNAlign,
    float alpha, int32_t alphaVectorScaling)
{
    if (alphaVectorScaling == 1) {
        if constexpr (std::is_same_v<AccType, float>) {
            for (int32_t r = 0; r < curM; ++r) {
                float alpha_r = SpltGetScalarFromUB(alphaVecUB, mPos + r);
                Muls(dFp32UB[r * curNAlign], accUB[r * curNAlign], alpha_r, curN);
            }
            PipeBarrier<PIPE_V>();
        } else {
            Cast(dFp32UB, accUB, AscendC::RoundMode::CAST_NONE,
                 static_cast<int32_t>(curMAlign * curNAlign));
            PipeBarrier<PIPE_V>();
            for (int32_t r = 0; r < curM; ++r) {
                float alpha_r = SpltGetScalarFromUB(alphaVecUB, mPos + r);
                Muls(dFp32UB[r * curNAlign], dFp32UB[r * curNAlign], alpha_r, curN);
            }
            PipeBarrier<PIPE_V>();
        }
    } else {
        if constexpr (std::is_same_v<AccType, float>) {
            Muls(dFp32UB, accUB, alpha, static_cast<int32_t>(curMAlign * curNAlign));
        } else {
            // INT8：int32 acc 先 Cast -> float，再 Muls 乘 alpha。
            Cast(dFp32UB, accUB, AscendC::RoundMode::CAST_NONE,
                 static_cast<int32_t>(curMAlign * curNAlign));
            PipeBarrier<PIPE_V>();
            Muls(dFp32UB, dFp32UB, alpha, static_cast<int32_t>(curMAlign * curNAlign));
        }
    }
}

// 从 GM 加载非 FP32 的 bias chunk 并 Cast 到 FP32。
// 供 SpltFusedEpilogueApplyBias 用于 FP16/BF16 bias 类型。
template <typename BiasType>
__aicore__ inline void SpltCastBiasChunkToFp32(
    LocalTensor<BiasType>& biasRawChunkUB, LocalTensor<float>& biasChunkUB,
    GlobalTensor<BiasType>& biasVecGM, int32_t mPos, int32_t curM)
{
    DataCopyExtParams copyBiasChunk{1,
        static_cast<uint32_t>(curM * sizeof(BiasType)), 0, 0, 0};
    DataCopyPadExtParams<BiasType> padBiasChunk{false, 0, 0, BiasType(0)};
    DataCopyPad(biasRawChunkUB, biasVecGM[mPos], copyBiasChunk, padBiasChunk);
    SetFlag<HardEvent::MTE2_V>(0);
    WaitFlag<HardEvent::MTE2_V>(0);
    if constexpr (std::is_same_v<BiasType, bfloat16_t>) {
        SpltCastBf16ToFp32Vec(biasChunkUB, biasRawChunkUB, curM);
    } else {
        Cast(biasChunkUB, biasRawChunkUB, AscendC::RoundMode::CAST_NONE, curM);
        PipeBarrier<PIPE_V>();
    }
}

// 融合通用路径中向 dFp32UB 施加 bias。
// 同时处理逐 chunk（biasChunkMode==1：从 GM 加载 curM 行）与
// 全量加载（biasChunkMode==0：biasVecUB 已加载）两种模式。
// FP16/BF16 bias 的原始数据在逐行 Adds 前先 Cast 到 FP32。
template <typename CType>
__aicore__ inline void SpltFusedEpilogueApplyBias(
    LocalTensor<float>& dFp32UB,
    TBuf<TPosition::VECCALC>& tempBuf, TBuf<TPosition::VECCALC>& biasRawBuf,
    LocalTensor<float>& biasVecUB, GlobalTensor<SpltBiasDType<CType>>& biasVecGM,
    int32_t mPos, int32_t curM, int32_t curN, uint64_t curNAlign,
    uint64_t biasDevPtr, int32_t biasChunkMode)
{
    using BiasType = SpltBiasDType<CType>;
    if (biasDevPtr == 0) { return; }
    if (biasChunkMode == 1) {
        if constexpr (std::is_same_v<BiasType, float>) {
            // FP32/INT8：GM 中 bias 为 FP32，直接加载到 tempBuf
            LocalTensor<float> biasChunkUB = tempBuf.Get<float>();
            DataCopyExtParams copyBiasChunk{1,
                static_cast<uint32_t>(curM * sizeof(float)), 0, 0, 0};
            DataCopyPadExtParams<float> padBiasChunk{false, 0, 0, 0.0f};
            DataCopyPad(biasChunkUB, biasVecGM[mPos], copyBiasChunk, padBiasChunk);
            SetFlag<HardEvent::MTE2_V>(0);
            WaitFlag<HardEvent::MTE2_V>(0);
            for (int32_t r = 0; r < curM; ++r) {
                float bias_r = SpltGetScalarFromUB(biasChunkUB, r);
                Adds(dFp32UB[r * curNAlign], dFp32UB[r * curNAlign], bias_r, curN);
            }
            PipeBarrier<PIPE_V>();
        } else {
            // FP16/BF16：加载原始 bias，Cast 到 FP32，逐行 Adds
            LocalTensor<BiasType> biasRawChunkUB = biasRawBuf.Get<BiasType>();
            LocalTensor<float> biasChunkUB = tempBuf.Get<float>();
            SpltCastBiasChunkToFp32<BiasType>(biasRawChunkUB, biasChunkUB, biasVecGM, mPos, curM);
            for (int32_t r = 0; r < curM; ++r) {
                float bias_r = SpltGetScalarFromUB(biasChunkUB, r);
                Adds(dFp32UB[r * curNAlign], dFp32UB[r * curNAlign], bias_r, curN);
            }
            PipeBarrier<PIPE_V>();
        }
    } else {
        // 全量加载：biasVecUB 已加载 (FP32), per-row GetValue + Adds
        for (int32_t r = 0; r < curM; ++r) {
            float bias_r = SpltGetScalarFromUB(biasVecUB, mPos + r);
            Adds(dFp32UB[r * curNAlign], dFp32UB[r * curNAlign], bias_r, curN);
        }
        PipeBarrier<PIPE_V>();
    }
}

// 融合通用路径中向 dFp32UB 施加激活（ReLU 或 GeLU）。
// ReLU：整 tile 批量执行。GeLU：逐行执行，tempBuf 复用。
__aicore__ inline void SpltFusedEpilogueApplyActivation(
    LocalTensor<float>& dFp32UB, TBuf<TPosition::VECCALC>& tempBuf,
    int32_t curM, int32_t curN, uint64_t curMAlign, uint64_t curNAlign,
    int32_t activationType,
    float reluThreshold, float reluUpperBound, float geluScaling)
{
    if (activationType == 1) {
        // ReLU: 全 tile 批量执行
        SpltApplyReLU(dFp32UB, static_cast<int32_t>(curMAlign * curNAlign),
                       reluThreshold, reluUpperBound);
    } else if (activationType == 2) {
        // GeLU: 逐行执行，geluTemp 复用 tempBuf（一行大小 = maxNAlign×4）
        LocalTensor<float> geluTemp = tempBuf.Get<float>();
        for (int32_t r = 0; r < curM; ++r) {
            SpltApplyGeLU(dFp32UB[r * curNAlign], geluTemp, curN, geluScaling);
        }
    }
}

// 通用路径：D = alpha*acc + beta*C + bias + activation -> 输出。
// v2：INT8 路径（AccType=int32_t）需在 Muls 前 Cast int32 -> float。
// 新增 halfBuf 参数用于 float→half→int8 两步 Cast。
// bias + activation 插入在 beta*C 之后、Cast+output 之前。
template <typename AccType, typename OutType, typename CType = OutType>
__aicore__ inline void SpltFusedEpilogueGeneralPath(
    LocalTensor<AccType>& accUB, GlobalTensor<CType>& cGM, GlobalTensor<OutType>& dGM,
    TBuf<TPosition::VECCALC>& dFp32Buf, TBuf<TPosition::VECCALC>& cTBuf,
    TBuf<TPosition::VECCALC>& cFp32Buf, TBuf<TPosition::VECCALC>& tempBuf,
    TBuf<TPosition::VECCALC>& dTBuf, TBuf<TPosition::VECCALC>& halfBuf,
    TBuf<TPosition::VECCALC>& biasRawBuf,
    int32_t mPos, int32_t nPos, int32_t curM, int32_t curN,
    uint64_t curMAlign, uint64_t curNAlign, int32_t n,
    float alpha, float beta,
    int32_t alphaVectorScaling, int32_t betaVectorScaling,
    LocalTensor<float>& alphaVecUB, LocalTensor<float>& betaVecUB,
    LocalTensor<float>& biasVecUB, GlobalTensor<SpltBiasDType<CType>>& biasVecGM,
    uint64_t biasDevPtr, int32_t biasChunkMode,
    int32_t activationType,
    float reluThreshold, float reluUpperBound, float geluScaling)
{
    using BiasType = SpltBiasDType<CType>;
    LocalTensor<float> dFp32UB = dFp32Buf.Get<float>();
    SpltFusedEpilogueApplyAlpha<AccType>(dFp32UB, accUB, alphaVecUB,
        mPos, curM, curN, curMAlign, curNAlign, alpha, alphaVectorScaling);
    if (betaVectorScaling == 1 || beta != 0.0f) {
        SpltFusedEpilogueBetaC<AccType, OutType, CType>(dFp32UB, cGM, cTBuf, cFp32Buf, tempBuf,
                                  mPos, nPos, curM, curN, curNAlign, n, beta,
                                  betaVectorScaling, betaVecUB);
    }
    // ③ bias → 累加到 dFp32UB
    SpltFusedEpilogueApplyBias<CType>(dFp32UB, tempBuf, biasRawBuf, biasVecUB, biasVecGM,
        mPos, curM, curN, curNAlign, biasDevPtr, biasChunkMode);
    // ④ activation
    SpltFusedEpilogueApplyActivation(dFp32UB, tempBuf, curM, curN, curMAlign, curNAlign,
        activationType, reluThreshold, reluUpperBound, geluScaling);
    if constexpr (std::is_same_v<OutType, float>) {
        SpltFusedEpilogueOutputFp32(dFp32UB, dGM, mPos, nPos, curM, curN, curNAlign, n);
    } else {
        SpltFusedEpilogueOutputCast<OutType>(dFp32UB, dTBuf, dFp32Buf, halfBuf, dGM,
                                            mPos, nPos, curM, curN, curNAlign, n);
    }
}

// ----------------------------------------------------------------------------
// AIV 侧：CrossCore 等待 -> Vector(alpha*acc+beta*C+Cast) -> DataCopyPad UB->GM
// 每个 AIV 处理分派给自己的 tile（与另一 AIV 交替）。
// ----------------------------------------------------------------------------
// SpltFusedEpilogueImpl 的 tile 循环驱动：逐 tile 的
// CrossCore 等待 -> FastPath/GeneralPath -> CrossCore 通知遍历。提取
// 以降低 SpltFusedEpilogueImpl 的 NBNC（#5，73 -> 目标 <=50）。flag 操作
// 完全保持不变。TBufs/TPipe 归调用方（Impl）所有并按引用传入；
// 其生命周期覆盖整个循环。
// v2：模板化为 (AccType, OutType) —— AccType 来自 SpltL0CTypeTrait<T>，
// OutType 为输出 dtype（非 INT8 时等于 T；INT8 时为 int8_t/int32_t）。
// 新增 halfBuf 参数用于 float→half→int8 两步 Cast。
template <typename AccType, typename OutType, typename CType = OutType>
__aicore__ inline void SpltFusedEpilogueTileLoop(
    const AclsparseltTilingData& td,
    int32_t rawBlockId, int32_t blockNum, uint32_t subBlockIdx,
    uint16_t aivWaitFlag, uint16_t aivNotifyFlag,
    int32_t nTiles, int64_t totalTiles,
    GlobalTensor<CType>& cGM, GlobalTensor<OutType>& dGM,
    TBuf<TPosition::VECCALC>& accBuf, TBuf<TPosition::VECCALC>& dFp32Buf,
    TBuf<TPosition::VECCALC>& dTBuf, TBuf<TPosition::VECCALC>& cTBuf,
    TBuf<TPosition::VECCALC>& cFp32Buf, TBuf<TPosition::VECCALC>& tempBuf,
    TBuf<TPosition::VECCALC>& halfBuf, TBuf<TPosition::VECCALC>& biasRawBuf,
    float alpha, float beta,
    int32_t alphaVectorScaling, int32_t betaVectorScaling,
    LocalTensor<float>& alphaVecUB, LocalTensor<float>& betaVecUB,
    LocalTensor<float>& biasVecUB, GlobalTensor<SpltBiasDType<CType>>& biasVecGM,
    uint64_t biasDevPtr, int32_t biasChunkMode,
    int32_t activationType,
    float reluThreshold, float reluUpperBound, float geluScaling)
{
    const int32_t m = td.m;
    const int32_t n = td.n;
    const int32_t baseM = td.baseM;
    const int32_t baseN = td.baseN;

    uint32_t localTileIdx = 0;
    bool ubReleased = false;  // 逐 batch 的首次释放标记
    for (int32_t tileIdx = rawBlockId; static_cast<int64_t>(tileIdx) < totalTiles; tileIdx += blockNum) {
        const bool myTile = (localTileIdx & 0x1) == subBlockIdx;

        const int32_t mTile = static_cast<int32_t>(static_cast<int64_t>(tileIdx) / nTiles);
        const int32_t nTile = static_cast<int32_t>(static_cast<int64_t>(tileIdx) % nTiles);
        const int32_t mPos = mTile * baseM;
        const int32_t nPos = nTile * baseN;
        const int32_t curM = (mPos + baseM <= m) ? baseM : (m - mPos);
        const int32_t curN = (nPos + baseN <= n) ? baseN : (n - nPos);
        const uint64_t curMAlign = static_cast<uint64_t>((curM + 1) & ~1);
        const uint64_t curNAlign = static_cast<uint64_t>((curN + SPLT_L0C_C0 - 1) / SPLT_L0C_C0) * SPLT_L0C_C0;

        // 仅对本 AIV 分派到的 tile 参与同步。
        // AIC 只通知并等待目标 AIV。
        if (!myTile) {
            localTileIdx++;
            continue;
        }

        // 在本 AIV 于该 batch 的首个 tile 前先释放一次。仅当本 AIV
        // 确实拥有 tile 时才释放，使 CrossCore set/wait 对在 batch 边界
        // 严格配对（某 batch 内 tile 数为零的 AIV 不会发出多余的释放，
        // 导致 AIC 下一 batch 的 wait 错序消费）。
        if (!ubReleased) {
            CrossCoreSetFlag<SPLT_SYNC_MODE_4, PIPE_MTE3>(aivNotifyFlag);
            ubReleased = true;
        }

        // 1. 等待 AIC Fixpipe 完成（数据已就绪于 UB）。
        CrossCoreWaitFlag<SPLT_SYNC_MODE_4, PIPE_V>(aivWaitFlag);

        // 2. 处理整个 tile（本 AIV 处理全部 curM 行）。
        LocalTensor<AccType> accUB = accBuf.Get<AccType>();

        // FastPath 条件更新：有 bias 或 activation 时走 GeneralPath（方式 A，见 §3.2）
        const bool isFastPath = (alphaVectorScaling == 0 && alpha == 1.0f && beta == 0.0f
                                 && biasDevPtr == 0 && activationType == 0);
        if (isFastPath) {
            // 快速路径：alpha=1、beta=0、无 bias、无 activation -> Cast acc -> OutType 并写 GM。
            // 传入 curMAlign 使 INT8 Cast 覆盖整个 tile。
            // 传入 halfBuf 用于 float→half→int8 两步 Cast。
            SpltFusedEpilogueFastPath<AccType, OutType>(accUB, dGM, dTBuf, dFp32Buf, halfBuf,
                                         mPos, nPos, curM, curN, curMAlign, curNAlign, n);
        } else {
            // 通用路径：D = alpha*acc + beta*C + bias + activation。
            SpltFusedEpilogueGeneralPath<AccType, OutType, CType>(accUB, cGM, dGM,
                dFp32Buf, cTBuf, cFp32Buf, tempBuf, dTBuf, halfBuf, biasRawBuf,
                mPos, nPos, curM, curN, curMAlign, curNAlign, n,
                alpha, beta, alphaVectorScaling, betaVectorScaling,
                alphaVecUB, betaVecUB,
                biasVecUB, biasVecGM, biasDevPtr, biasChunkMode,
                activationType, reluThreshold, reluUpperBound, geluScaling);
        }

        // 3. 通知 AIC：本 AIV 已用完该 UB 槽位。
        CrossCoreSetFlag<SPLT_SYNC_MODE_4, PIPE_MTE3>(aivNotifyFlag);
        localTileIdx++;
    }
}

// SpltFusedEpilogueImpl 的 TBuf 初始化（按 tile 分配）。
// 提取以消除重复的缓冲初始化代码并降低 NBNC。
template <typename T, typename OutType>
__aicore__ inline void SpltFusedEpilogueInitBuffers(
    TPipe& pipe,
    TBuf<TPosition::VECCALC>& accBuf, TBuf<TPosition::VECCALC>& dFp32Buf,
    TBuf<TPosition::VECCALC>& dTBuf, TBuf<TPosition::VECCALC>& cTBuf,
    TBuf<TPosition::VECCALC>& cFp32Buf, TBuf<TPosition::VECCALC>& tempBuf,
    TBuf<TPosition::VECCALC>& halfBuf,
    uint64_t maxMAlign, uint64_t maxNAlign,
    float alpha, float beta,
    int32_t alphaVectorScaling, int32_t betaVectorScaling,
    uint64_t biasDevPtr, int32_t biasChunkMode, int32_t activationType)
{
    using AccType = typename SpltL0CTypeTrait<T>::type;
    // FastPath 条件——有 bias 或 activation 时走 GeneralPath
    const bool isFastPath = (alphaVectorScaling == 0 && alpha == 1.0f && beta == 0.0f
                             && biasDevPtr == 0 && activationType == 0);
    pipe.InitBuffer(accBuf, static_cast<uint32_t>(maxMAlign * maxNAlign * sizeof(AccType)));
    if (!isFastPath) {
        pipe.InitBuffer(dFp32Buf, static_cast<uint32_t>(maxMAlign * maxNAlign * sizeof(float)));
    }
    if (isFastPath && !std::is_same_v<AccType, float>) {
        pipe.InitBuffer(dFp32Buf, static_cast<uint32_t>(maxMAlign * maxNAlign * sizeof(float)));
    }
    // [LOW-8] 移除 BF16 快速路径的 dFp32Buf 分配：SpltCastFp32ToBf16Vec
    // （BF16 时在 SpltFusedEpilogueOutputCast 中调用）不使用 dFp32Buf。
    if constexpr (!std::is_same_v<OutType, float>) {
        pipe.InitBuffer(dTBuf, static_cast<uint32_t>(maxNAlign * sizeof(OutType)));
    }
    if constexpr (std::is_same_v<OutType, int8_t>) {
        pipe.InitBuffer(halfBuf, static_cast<uint32_t>(maxNAlign * sizeof(half)));
    }
    // tempBuf/cTBuf/cFp32Buf 分配条件扩展
    // tempBuf: beta*C 中间量 | biasChunkUB（biasChunkMode==1，dtype=BiasType）| geluTemp（activationType==2）
    // cTBuf: C 加载（beta!=0）
    // cFp32Buf: C Cast 中间缓冲（beta!=0，T!=float）
    const bool needTempBuf = (betaVectorScaling == 1 || beta != 0.0f)
        || (biasDevPtr != 0 && biasChunkMode == 1)
        || (activationType == 2);
    const bool needCTBuf = (betaVectorScaling == 1 || beta != 0.0f);
    if (needCTBuf) {
        pipe.InitBuffer(cTBuf, static_cast<uint32_t>(maxNAlign * sizeof(T)));
    }
    if (needTempBuf) {
        pipe.InitBuffer(tempBuf, static_cast<uint32_t>(maxNAlign * sizeof(float)));
    }
    if ((betaVectorScaling == 1 || beta != 0.0f) && !std::is_same_v<T, float>) {
        pipe.InitBuffer(cFp32Buf, static_cast<uint32_t>(maxNAlign * sizeof(float)));
    }
}

// 为 SpltFusedEpilogueImpl 构建 cGM/dGM global buffer。
// 提取以降低 SpltFusedEpilogueImpl 的 NBNC（53 -> 约45）。
template <typename T, typename OutType>
__aicore__ inline void SpltFusedEpilogueInitGM(
    GlobalTensor<T>& cGM, GlobalTensor<OutType>& dGM,
    GM_ADDR cGm, GM_ADDR dGm, int32_t m, int32_t n)
{
    cGM.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(cGm), static_cast<uint64_t>(m) * n);
    dGM.SetGlobalBuffer(reinterpret_cast<__gm__ OutType*>(dGm), static_cast<uint64_t>(m) * n);
}

// 为 SpltFusedEpilogueBatchLoop 设置逐 batch 的 GM 指针。
// 将 c/d/bias GM 按 batch stride 偏移；biasChunkMode==0 时重载完整 bias。
template <typename T, typename OutType>
__aicore__ inline void SpltFusedEpilogueOffsetBatchGM(
    const AclsparseltTilingData& td,
    GlobalTensor<T>& cGM, GlobalTensor<OutType>& dGM,
    GlobalTensor<SpltBiasDType<T>>& biasVecGM,
    LocalTensor<float>& biasVecUB, TBuf<TPosition::VECCALC>& biasRawBuf,
    GM_ADDR cGm, GM_ADDR dGm, int32_t b, int32_t m, uint64_t totalElem)
{
    using BiasType = SpltBiasDType<T>;
    cGM.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(cGm)
                        + static_cast<int64_t>(b) * td.batchStrideC, totalElem);
    dGM.SetGlobalBuffer(reinterpret_cast<__gm__ OutType*>(dGm)
                        + static_cast<int64_t>(b) * td.batchStrideD, totalElem);
    if (td.biasDevPtr != 0 && td.biasStride != 0) {
        biasVecGM.SetGlobalBuffer(reinterpret_cast<__gm__ BiasType*>(td.biasDevPtr)
                                  + static_cast<int64_t>(b) * td.biasStride,
                                  static_cast<uint64_t>(m));
        if (td.biasChunkMode == 0) {
            SpltLoadBiasFullToFp32<BiasType>(biasVecUB, biasRawBuf, biasVecGM, m);
        }
    }
}

// SpltFusedEpilogueImpl 的 batch 循环：遍历 batch，设置逐 batch GM
// 指针，按需重载 bias，并执行 tile 处理循环。
// 提取以降低 SpltFusedEpilogueImpl 的 NBNC（#9，原 95，目标 <=50）。
// flag 操作与 CrossCore 同步完全保持不变。
template <typename T, typename OutType>
__aicore__ inline void SpltFusedEpilogueBatchLoop(
    const AclsparseltTilingData& td,
    GlobalTensor<T>& cGM, GlobalTensor<OutType>& dGM,
    GlobalTensor<SpltBiasDType<T>>& biasVecGM,
    LocalTensor<float>& biasVecUB, TBuf<TPosition::VECCALC>& biasRawBuf,
    TBuf<TPosition::VECCALC>& accBuf, TBuf<TPosition::VECCALC>& dFp32Buf,
    TBuf<TPosition::VECCALC>& dTBuf, TBuf<TPosition::VECCALC>& cTBuf,
    TBuf<TPosition::VECCALC>& cFp32Buf, TBuf<TPosition::VECCALC>& tempBuf,
    TBuf<TPosition::VECCALC>& halfBuf,
    LocalTensor<float>& alphaVecUB, LocalTensor<float>& betaVecUB,
    GM_ADDR cGm, GM_ADDR dGm)
{
    using AccType = typename SpltL0CTypeTrait<T>::type;
    const int32_t m = td.m;
    const int32_t n = td.n;
    const int32_t baseM = td.baseM;
    const int32_t baseN = td.baseN;
    const float alpha = td.alpha;
    const float beta = td.beta;
    const int32_t alphaVectorScaling = td.alphaVectorScaling;
    const int32_t betaVectorScaling = td.betaVectorScaling;
    const uint64_t biasDevPtr = td.biasDevPtr;
    const int32_t biasChunkMode = td.biasChunkMode;
    const int32_t activationType = td.activationType;
    const float reluThreshold = td.reluThreshold;
    const float reluUpperBound = td.reluUpperBound;
    const float geluScaling = td.geluScaling;
    const int32_t numBatches = (td.numBatches > 0) ? td.numBatches : 1;
    const int32_t blockId = static_cast<int32_t>(GetBlockIdx());
    const int32_t blockNum = static_cast<int32_t>(GetBlockNum());
    const uint32_t taskRation = GetTaskRation();
    const int32_t rawBlockId = blockId / static_cast<int32_t>(taskRation);
    const uint32_t subBlockIdx = GetSubBlockIdx();
    const uint16_t aivWaitFlag = (subBlockIdx == 1) ? SPLT_AIC_SYNC_AIV1 : SPLT_AIC_SYNC_AIV0;
    const uint16_t aivNotifyFlag = (subBlockIdx == 1) ? SPLT_AIV1_SYNC_AIC : SPLT_AIV0_SYNC_AIC;
    const int32_t mTiles = (m + baseM - 1) / baseM;
    const int32_t nTiles = (n + baseN - 1) / baseN;
    const int64_t totalTiles = static_cast<int64_t>(mTiles) * static_cast<int64_t>(nTiles);
    const uint64_t totalElem = static_cast<uint64_t>(m) * static_cast<uint64_t>(n);

    for (int32_t b = 0; b < numBatches; ++b) {
        SpltFusedEpilogueOffsetBatchGM<T, OutType>(td, cGM, dGM, biasVecGM,
                                                   biasVecUB, biasRawBuf,
                                                   cGm, dGm, b, m, totalElem);
        // 无 tile 的 block 跳过 tile 循环（也不参与任何 CrossCore 交互）。
        // 预释放 SetFlag 由 SpltFusedEpilogueTileLoop 在首个 myTile 前按需
        // 预发，见该函数内注释。
        if (static_cast<int64_t>(rawBlockId) >= totalTiles) { continue; }
        SpltFusedEpilogueTileLoop<AccType, OutType, T>(td, rawBlockId, blockNum, subBlockIdx,
                                     aivWaitFlag, aivNotifyFlag,
                                     nTiles, totalTiles, cGM, dGM,
                                     accBuf, dFp32Buf, dTBuf, cTBuf, cFp32Buf, tempBuf, halfBuf, biasRawBuf,
                                     alpha, beta,
                                     alphaVectorScaling, betaVectorScaling,
                                     alphaVecUB, betaVecUB,
                                     biasVecUB, biasVecGM, biasDevPtr, biasChunkMode,
                                     activationType, reluThreshold, reluUpperBound, geluScaling);
    }
}

// SpltFusedEpilogueImpl 的 bias 缓冲初始化，按 biasChunkMode 分发。
// 从 SpltFusedEpilogueImpl 提取以降低 NBNC（#2）。
//   biasChunkMode == 0：经 SpltInitBiasBuffers 全量加载 bias（驻留 UB）。
//   biasChunkMode == 1：chunk 模式 bias 驻留 GM（逐 tile 惰性加载），
//                       仅分配 Cast 中间缓冲（biasRawBuf）。
template <typename BiasType>
__aicore__ inline void SpltFusedEpilogueInitBias(
    GlobalTensor<BiasType>& biasVecGM,
    TBuf<TPosition::VECCALC>& biasVecBuf,
    TBuf<TPosition::VECCALC>& biasRawBuf,
    LocalTensor<float>& biasVecUB,
    TPipe& pipe,
    uint64_t biasDevPtr, int64_t biasStride, int32_t biasChunkMode,
    int32_t m, uint64_t maxMAlign)
{
    if (biasDevPtr == 0) { return; }
    if (biasChunkMode == 0) {
        SpltInitBiasBuffers<BiasType>(biasVecGM, biasVecBuf, biasRawBuf, biasVecUB,
                                       pipe, biasDevPtr, biasStride, m);
    } else if (biasChunkMode == 1) {
        biasVecGM.SetGlobalBuffer(reinterpret_cast<__gm__ BiasType*>(biasDevPtr),
                                   static_cast<uint64_t>(m));
        if constexpr (!std::is_same_v<BiasType, float>) {
            pipe.InitBuffer(biasRawBuf, static_cast<uint32_t>(maxMAlign * sizeof(BiasType)));
        }
    }
}

// v2：SpltFusedEpilogueImpl 模板化为 (T, OutType)。AccType 经
// SpltL0CTypeTrait<T> 推导。非 INT8 路径：OutType == T（向后兼容）。
template <typename T, typename OutType = T>
__aicore__ inline void SpltFusedEpilogueImpl(GM_ADDR cGm, GM_ADDR dGm, GM_ADDR tilingGm)
{
    using AccType = typename SpltL0CTypeTrait<T>::type;
    using BiasType = SpltBiasDType<T>;
    const AclsparseltTilingData td = splt_load_tiling(tilingGm);
    const int32_t m = td.m;
    const int32_t baseM = td.baseM;
    const int32_t baseN = td.baseN;
    const int32_t alphaVectorScaling = td.alphaVectorScaling;
    const int32_t betaVectorScaling = td.betaVectorScaling;
    const uint64_t biasDevPtr = td.biasDevPtr;
    const int64_t biasStride = td.biasStride;
    const int32_t biasChunkMode = td.biasChunkMode;
    const uint32_t taskRation = GetTaskRation();
    if (taskRation == 0) { return; }
    const int64_t totalTiles = static_cast<int64_t>((m + baseM - 1) / baseM) *
                               static_cast<int64_t>((td.n + baseN - 1) / baseN);
    if (totalTiles <= 0) { return; }
    GlobalTensor<T> cGM;
    GlobalTensor<OutType> dGM;
    GlobalTensor<float> alphaVecGM, betaVecGM;
    SpltInitScalingGM(alphaVecGM, betaVecGM, td.alphaDevPtr, td.betaDevPtr,
                       m, alphaVectorScaling, betaVectorScaling);
    GlobalTensor<BiasType> biasVecGM;
    TBuf<TPosition::VECCALC> biasVecBuf;
    TBuf<TPosition::VECCALC> biasRawBuf;
    LocalTensor<float> biasVecUB;
    TPipe pipe;
    const uint64_t maxMAlign = static_cast<uint64_t>((baseM + 1) & ~1);
    const uint64_t maxNAlign = static_cast<uint64_t>((baseN + SPLT_L0C_C0 - 1) / SPLT_L0C_C0) * SPLT_L0C_C0;
    TBuf<TPosition::VECCALC> accBuf, dFp32Buf, dTBuf, cTBuf, cFp32Buf, tempBuf, halfBuf;
    TBuf<TPosition::VECCALC> alphaVecBuf, betaVecBuf;
    SpltFusedEpilogueInitBuffers<T, OutType>(pipe, accBuf, dFp32Buf, dTBuf, cTBuf,
        cFp32Buf, tempBuf, halfBuf, maxMAlign, maxNAlign, td.alpha, td.beta,
        alphaVectorScaling, betaVectorScaling,
        biasDevPtr, biasChunkMode, td.activationType);
    // bias UB 的分配 + 加载放在 accBuf 之后（accBuf 必须保持在
    // UB 偏移 0，以匹配 AIC 的 CopyL0C2UB 目标地址）。
    SpltFusedEpilogueInitBias<BiasType>(biasVecGM, biasVecBuf, biasRawBuf, biasVecUB,
                                         pipe, biasDevPtr, biasStride, biasChunkMode,
                                         m, maxMAlign);
    LocalTensor<float> alphaVecUB, betaVecUB;
    SpltLoadScalingUB(alphaVecBuf, betaVecBuf, alphaVecUB, betaVecUB,
                       alphaVecGM, betaVecGM, pipe, m,
                       alphaVectorScaling, betaVectorScaling);
    SpltFusedEpilogueBatchLoop<T, OutType>(td, cGM, dGM, biasVecGM, biasVecUB, biasRawBuf,
        accBuf, dFp32Buf, dTBuf, cTBuf, cFp32Buf, tempBuf, halfBuf,
        alphaVecUB, betaVecUB, cGm, dGm);
}

// ============================================================================
// [OPT-P2] 融合 matmul+epilogue kernel 入口（splitK==1 路径）。
// v2：新增 BF16 / INT8 / INT8_I32 入口。
// ============================================================================
extern "C" __global__ __aicore__ __mix__(1, 2) void splt_fused_matmul_kernel_fp16(
    GM_ADDR aPrunedGm, GM_ADDR bGm, GM_ADDR cGm, GM_ADDR dGm, GM_ADDR tilingGm)
{
    if ASCEND_IS_AIC {
        SpltFusedMatmulCubeImpl<__fp16>(aPrunedGm, bGm, dGm, tilingGm);
    } else {
        SpltFusedEpilogueImpl<__fp16>(cGm, dGm, tilingGm);
    }
}

extern "C" __global__ __aicore__ __mix__(1, 2) void splt_fused_matmul_kernel_fp32(
    GM_ADDR aPrunedGm, GM_ADDR bGm, GM_ADDR cGm, GM_ADDR dGm, GM_ADDR tilingGm)
{
    if ASCEND_IS_AIC {
        SpltFusedMatmulCubeImpl<float>(aPrunedGm, bGm, dGm, tilingGm);
    } else {
        SpltFusedEpilogueImpl<float>(cGm, dGm, tilingGm);
    }
}

// v2：BF16 融合 —— 重新启用。
// [OPT-BF16-VEC] BF16 Cast 使用 Vector Cast<bfloat16_t, float>（CANN 9.1.0
// 的 cast_round_all 已支持）。cube Mmad（tensor_api）将 BF16 走
// NORMAL 路径（与 dtype 无关）。
extern "C" __global__ __aicore__ __mix__(1, 2) void splt_fused_matmul_kernel_bf16(
    GM_ADDR aPrunedGm, GM_ADDR bGm, GM_ADDR cGm, GM_ADDR dGm, GM_ADDR tilingGm)
{
    if ASCEND_IS_AIC {
        SpltFusedMatmulCubeImpl<__bf16>(aPrunedGm, bGm, dGm, tilingGm);
    } else {
        SpltFusedEpilogueImpl<__bf16>(cGm, dGm, tilingGm);
    }
}

// v2：INT8 融合 —— INT8 输出（饱和 Cast）。
extern "C" __global__ __aicore__ __mix__(1, 2) void splt_fused_matmul_kernel_int8(
    GM_ADDR aPrunedGm, GM_ADDR bGm, GM_ADDR cGm, GM_ADDR dGm, GM_ADDR tilingGm)
{
    if ASCEND_IS_AIC {
        SpltFusedMatmulCubeImpl<int8_t>(aPrunedGm, bGm, dGm, tilingGm);
    } else {
        SpltFusedEpilogueImpl<int8_t, int8_t>(cGm, dGm, tilingGm);
    }
}

// v2：INT8 融合 —— INT32 输出（经 ROUND Cast 的 int32 直接写）。
extern "C" __global__ __aicore__ __mix__(1, 2) void splt_fused_matmul_kernel_int8_i32(
    GM_ADDR aPrunedGm, GM_ADDR bGm, GM_ADDR cGm, GM_ADDR dGm, GM_ADDR tilingGm)
{
    if ASCEND_IS_AIC {
        SpltFusedMatmulCubeImpl<int8_t>(aPrunedGm, bGm, dGm, tilingGm);
    } else {
        SpltFusedEpilogueImpl<int8_t, int32_t>(cGm, dGm, tilingGm);
    }
}

// ============================================================================
// kernel 入口（每种 dtype 一个），由 host 启动器分发。
// v2：新增 BF16 / INT8 入口。INT8 matmul（cube）为 INT8->INT8 与
// INT8->INT32 共享（cube 只写 int32 temp；输出类型分发
// 在 epilogue/fused 中完成）。
// ============================================================================

// --- matmul (cube) ---
extern "C" __global__ __aicore__ void splt_matmul_kernel_fp16(
    GM_ADDR aPrunedGm, GM_ADDR bGm, GM_ADDR tempGm, GM_ADDR tilingGm)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    SpltMatmulCubeImpl<__fp16>(aPrunedGm, bGm, tempGm, tilingGm);
}
extern "C" __global__ __aicore__ void splt_matmul_kernel_fp32(
    GM_ADDR aPrunedGm, GM_ADDR bGm, GM_ADDR tempGm, GM_ADDR tilingGm)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    SpltMatmulCubeImpl<float>(aPrunedGm, bGm, tempGm, tilingGm);
}
// v2：BF16 cube —— 重新启用。
// cube Mmad（tensor_api）将 BF16 走 NORMAL 路径（与 dtype 无关）；
// cube-only kernel 不涉及 Vector Cast。L0C=float 由 SpltL0CTypeTrait 推导。
extern "C" __global__ __aicore__ void splt_matmul_kernel_bf16(
    GM_ADDR aPrunedGm, GM_ADDR bGm, GM_ADDR tempGm, GM_ADDR tilingGm)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    SpltMatmulCubeImpl<__bf16>(aPrunedGm, bGm, tempGm, tilingGm);
}
// v2：INT8 cube（经 SpltL0CTypeTrait 得 L0C=int32_t；INT8->INT8 / INT8->INT32 共享）。
extern "C" __global__ __aicore__ void splt_matmul_kernel_int8(
    GM_ADDR aPrunedGm, GM_ADDR bGm, GM_ADDR tempGm, GM_ADDR tilingGm)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    SpltMatmulCubeImpl<int8_t>(aPrunedGm, bGm, tempGm, tilingGm);
}

// --- epilogue ---
extern "C" __global__ __aicore__ void splt_epilogue_kernel_fp16(
    GM_ADDR tempGm, GM_ADDR cGm, GM_ADDR dGm, GM_ADDR tilingGm)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    SpltEpilogueImpl<__fp16>(tempGm, cGm, dGm, tilingGm);
}
extern "C" __global__ __aicore__ void splt_epilogue_kernel_fp32(
    GM_ADDR tempGm, GM_ADDR cGm, GM_ADDR dGm, GM_ADDR tilingGm)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    SpltEpilogueImpl<float>(tempGm, cGm, dGm, tilingGm);
}
// v2：BF16 epilogue —— 重新启用。
// [OPT-BF16-VEC] BF16 Cast（FP32<->BF16）使用 Vector Cast（CANN 9.1.0
// 在 cast_round_all 中支持 Cast<bfloat16_t,float>，在 cast_none 中
// 支持 Cast<float,bfloat16_t>）。
extern "C" __global__ __aicore__ void splt_epilogue_kernel_bf16(
    GM_ADDR tempGm, GM_ADDR cGm, GM_ADDR dGm, GM_ADDR tilingGm)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    SpltEpilogueImpl<__bf16>(tempGm, cGm, dGm, tilingGm);
}
// v2：INT8 epilogue —— INT8 输出（饱和 Cast）。
extern "C" __global__ __aicore__ void splt_epilogue_kernel_int8(
    GM_ADDR tempGm, GM_ADDR cGm, GM_ADDR dGm, GM_ADDR tilingGm)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    SpltEpilogueImpl<int8_t, int8_t>(tempGm, cGm, dGm, tilingGm);
}
// v2：INT8 epilogue —— INT32 输出（int32 直接写）。
extern "C" __global__ __aicore__ void splt_epilogue_kernel_int8_i32(
    GM_ADDR tempGm, GM_ADDR cGm, GM_ADDR dGm, GM_ADDR tilingGm)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    SpltEpilogueImpl<int8_t, int32_t>(tempGm, cGm, dGm, tilingGm);
}

// ============================================================================
// host 侧启动器（ASC 编译单元）：封装 <<<>>>。host.cpp 以 C 函数形式调用。
// v2：switch 四分支分发（FP32/FP16/BF16/INT8）+ epilogue/fused 的
// INT8 outDataType 子分支（INT8 与 INT32 输出）。
// ============================================================================
extern "C" void splt_matmul_kernel_launch(
    GM_ADDR aPrunedGm, GM_ADDR bGm, GM_ADDR tempGm,
    GM_ADDR tilingGm, int32_t dataType, uint32_t blockDim, void *stream)
{
    switch (dataType) {
        case SPLT_DTYPE_FP32:
            splt_matmul_kernel_fp32<<<blockDim, nullptr, stream>>>(
                aPrunedGm, bGm, tempGm, tilingGm);
            break;
        case SPLT_DTYPE_FP16:
            splt_matmul_kernel_fp16<<<blockDim, nullptr, stream>>>(
                aPrunedGm, bGm, tempGm, tilingGm);
            break;
        case SPLT_DTYPE_BF16:
            splt_matmul_kernel_bf16<<<blockDim, nullptr, stream>>>(
                aPrunedGm, bGm, tempGm, tilingGm);
            break;
        case SPLT_DTYPE_INT8:
            splt_matmul_kernel_int8<<<blockDim, nullptr, stream>>>(
                aPrunedGm, bGm, tempGm, tilingGm);
            break;
        default:
            OP_LOGE(kSparseLtLogTag, "unreachable: unknown dtype in matmul launcher\n"); return;
    }
}

// v2：新增 outDataType —— INT8 路径选择 INT8（饱和 Cast）或 INT32（直接写）。
extern "C" void splt_epilogue_kernel_launch(
    GM_ADDR tempGm, GM_ADDR cGm, GM_ADDR dGm,
    GM_ADDR tilingGm, int32_t dataType, int32_t outDataType, uint32_t blockDim, void *stream)
{
    switch (dataType) {
        case SPLT_DTYPE_FP32:
            splt_epilogue_kernel_fp32<<<blockDim, nullptr, stream>>>(
                tempGm, cGm, dGm, tilingGm);
            break;
        case SPLT_DTYPE_FP16:
            splt_epilogue_kernel_fp16<<<blockDim, nullptr, stream>>>(
                tempGm, cGm, dGm, tilingGm);
            break;
        case SPLT_DTYPE_BF16:
            splt_epilogue_kernel_bf16<<<blockDim, nullptr, stream>>>(
                tempGm, cGm, dGm, tilingGm);
            break;
        case SPLT_DTYPE_INT8:
            if (outDataType == SPLT_DTYPE_INT32) {
                splt_epilogue_kernel_int8_i32<<<blockDim, nullptr, stream>>>(
                    tempGm, cGm, dGm, tilingGm);
            } else {
                splt_epilogue_kernel_int8<<<blockDim, nullptr, stream>>>(
                    tempGm, cGm, dGm, tilingGm);
            }
            break;
        default:
            OP_LOGE(kSparseLtLogTag, "unreachable: unknown dtype in epilogue launcher\n"); return;
    }
}

// [OPT-P2] 融合 matmul+epilogue 启动器（splitK==1 路径）。
// v2：新增 outDataType —— INT8 路径选择 INT8 或 INT32 输出入口。
extern "C" void splt_fused_matmul_kernel_launch(
    GM_ADDR aPrunedGm, GM_ADDR bGm,
    GM_ADDR cGm, GM_ADDR dGm,
    GM_ADDR tilingGm, int32_t dataType, int32_t outDataType, uint32_t blockDim, void *stream)
{
    switch (dataType) {
        case SPLT_DTYPE_FP32:
            splt_fused_matmul_kernel_fp32<<<blockDim, nullptr, stream>>>(
                aPrunedGm, bGm, cGm, dGm, tilingGm);
            break;
        case SPLT_DTYPE_FP16:
            splt_fused_matmul_kernel_fp16<<<blockDim, nullptr, stream>>>(
                aPrunedGm, bGm, cGm, dGm, tilingGm);
            break;
        case SPLT_DTYPE_BF16:
            splt_fused_matmul_kernel_bf16<<<blockDim, nullptr, stream>>>(
                aPrunedGm, bGm, cGm, dGm, tilingGm);
            break;
        case SPLT_DTYPE_INT8:
            if (outDataType == SPLT_DTYPE_INT32) {
                splt_fused_matmul_kernel_int8_i32<<<blockDim, nullptr, stream>>>(
                    aPrunedGm, bGm, cGm, dGm, tilingGm);
            } else {
                splt_fused_matmul_kernel_int8<<<blockDim, nullptr, stream>>>(
                    aPrunedGm, bGm, cGm, dGm, tilingGm);
            }
            break;
        default:
            OP_LOGE(kSparseLtLogTag, "unreachable: unknown dtype in fused matmul launcher\n"); return;
    }
}
