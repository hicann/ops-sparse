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
 * \brief sparseLt matmul device-side kernels + extern "C" launchers.
 *
 * Kernels:
 *   - matmul_kernel         (AIC_ONLY, Te::Mmad tensor_api): M(m,k) x N(k,n) -> temp (FP32)
 *   - epilogue_kernel       (AIV_ONLY, low-level AscendC Vector API): D = alpha*reduce(temp) + beta*C
 *   - fused_matmul_kernel   (__mix__(1,2)): splitK==1 fused matmul+epilogue
 *
 * Launchers wrap <<<>>> in the ASC translation unit; host.cpp calls them as plain C functions.
 * Unlike prune_kernel (which receives tiling by value via launch args),
 * matmul/epilogue/fused kernels load tiling from GM via splt_load_tiling(tilingGm).
 * The TilingData struct and splt_load_tiling are defined in shared/aclsparselt_internal.h.
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
// tensor_api (Te namespace) symbols are referenced with explicit Te:: prefix
// to avoid LocalTensor/GlobalTensor ambiguity between AscendC (kernel_operator.h)
// and AscendC::Te (tensor_api). Do NOT add "using namespace AscendC::Te".

// ============================================================================
// v2: L0C accumulator type mapping.
// FP32/FP16/BF16 accumulate in float (L0C=float); INT8 accumulates in int32_t.
// Used to derive the L0C tensor type at construction sites (cube + fused).
// ============================================================================
template <typename T> struct SpltL0CTypeTrait { using type = float; };  // FP32/FP16/BF16
template <> struct SpltL0CTypeTrait<int8_t> { using type = int32_t; };  // INT8

// ============================================================================
// Bias dtype mapping (per cuSPARSELt spec):
// "The data type of the bias vector is the same as the matrix C except the
//  following case: INT8 input, INT8/INT32 output, INT32 compute — in which
//  the data type of the bias is FP32."
//   FP32 path: bias = FP32 (= C dtype) → load directly, no Cast
//   FP16 path: bias = FP16 (= C dtype) → load FP16, Cast to FP32
//   BF16 path: bias = BF16 (= C dtype) → load BF16, Cast to FP32
//   INT8 path: bias = FP32 (cuSPARSELt spec) → load directly, no Cast
// ============================================================================
template <typename T> struct SpltBiasDTypeTrait { using type = T; };  // FP32/FP16/BF16: bias = C dtype
template <> struct SpltBiasDTypeTrait<int8_t> { using type = float; };  // INT8: bias = FP32
template <typename T> using SpltBiasDType = typename SpltBiasDTypeTrait<T>::type;

// Element-wise scalar Cast loop (GetValue/SetValue). Uses static_cast for
// types that support it (FP16, INT8, etc.).
template <typename DstT, typename SrcT>
__aicore__ inline void SpltScalarCastLoop(LocalTensor<DstT>& dst, LocalTensor<SrcT> src,
                                           int32_t count)
{
    for (int32_t i = 0; i < count; ++i) {
        dst.SetValue(i, static_cast<DstT>(src.GetValue(i)));
    }
}

// ----------------------------------------------------------------------------
// float→int8_t Cast helper.
// DAV-3510 Cast API does NOT support direct float→int8_t conversion
// (CastImpl cast_round_all / cast_none lists have no Tuple<int8_t, float>).
// In release mode the ASCENDC_ASSERT is a no-op, so Cast silently does
// nothing — the destination buffer is left unwritten, producing garbage.
// Supported path: float→half (CAST_ROUND, Tuple<half,float> in cast_round_all)
// → int8_t (CAST_ROUND, Tuple<int8_t,half> in cast_round_all). Both steps
// are hardware Vector Cast. The half intermediate fits in a small TBuf
// (count * sizeof(half) bytes). Values are pre-clamped to [-128,127] by the
// caller (Mins/Maxs), and half represents all integers in that range exactly.
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
// [OPT-BF16-VEC] Vectorized FP32→BF16 Cast.
// DAV-3510 CANN 9.1.0 supports Vector Cast<bfloat16_t, float, CAST_ROUND>
// (Tuple<bfloat16_t, float> is in cast_round_all). This single Vector Cast
// call matches the FP16 path's performance characteristics.
// ----------------------------------------------------------------------------
__aicore__ inline void SpltCastFp32ToBf16Vec(
    LocalTensor<bfloat16_t>& dst, const LocalTensor<float>& src, int32_t count)
{
    Cast(dst, src, AscendC::RoundMode::CAST_ROUND, count);
    PipeBarrier<PIPE_V>();
}

// [OPT-BF16-VEC] Vectorized BF16→FP32 Cast (inverse of SpltCastFp32ToBf16Vec).
// Cast<float, bfloat16_t, CAST_NONE> is supported (Tuple<float, bfloat16_t>
// in cast_none). BF16→FP32 is a widening Cast (no precision loss → CAST_NONE).
__aicore__ inline void SpltCastBf16ToFp32Vec(
    LocalTensor<float>& dst, const LocalTensor<bfloat16_t>& src, int32_t count)
{
    Cast(dst, src, AscendC::RoundMode::CAST_NONE, count);
    PipeBarrier<PIPE_V>();
}

// ============================================================================
// Shared helpers for cube matmul kernels.
// Extracted to eliminate duplicate code (#9, #10) and reduce NBNC/complexity
// of SpltMatmulCubeImpl and SpltFusedMatmulCubeImpl.
// CRITICAL: SetFlag/WaitFlag positions are preserved exactly from the original
// inline code — do not reorder or merge any flag operations.
// ============================================================================
struct L1BufferConfig {
    int64_t aL1Off[2];
    int64_t bL1Off[2];
};

// [REFACTOR-KLOOP] L1 double-buffer init: buffer sizes based on kL1Size (not
// baseK) so the outer kL1 loop loads a larger K block to L1. Preset
// MTE1_MTE2(0/1) so the first WaitFlag passes; M_MTE1(0/1) for L0 ping-pong;
// FIX_M(0) for cross-tile L0C reuse handshake.
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

// Drain all cube flags at kernel end:
// - L1 ping-pong: MTE1_MTE2(0/1) + M_MTE1(0/1) — L1 double-buffer sync
// - L0C reuse:    FIX_M(0) — cross-tile L0C reuse handshake (not a double-buffer flag)
__aicore__ inline void DrainCubeFlags()
{
    WaitFlag<HardEvent::MTE1_MTE2>(0);
    WaitFlag<HardEvent::MTE1_MTE2>(1);
    WaitFlag<HardEvent::M_MTE1>(0);
    WaitFlag<HardEvent::M_MTE1>(1);
    WaitFlag<HardEvent::FIX_M>(0);
}

// ============================================================================
// [REFACTOR-KLOOP] Two-layer K-loop: outer kL1 (GM→L1) + inner kL0 (L1→L0+Mmad).
// Reference: cann-samples streamk/main.asc §269-355.
//   L1 double-buffer ping-pong on kL1 loop (l1BufId = kl1Idx & 1):
//     MTE1_MTE2(0/1) — GM→L1 load sync; M_MTE1(0/1) — L1→L0 copy sync.
//   L0  double-buffer ping-pong on kL0 loop (l0BufId = kl0Idx & 1).
//   FIX_M(0) is a separate cross-tile L0C reuse handshake flag,
//   not part of the L1/L0 double-buffer mechanism.
// CRITICAL: SetFlag/WaitFlag positions follow streamk — do not reorder or merge.
// ============================================================================

// GM → L1: load one kL1 block of A(curM, curKL1) + B(curKL1, curN) from GM to L1.
// The L1 tensors are created by the caller (kL1 loop) and passed by reference so
// they persist for the subsequent kL0 loop slicing.
// transB is handled entirely at the GM level: when transB=1, GM B
// is declared as DNExt(k, n) (column-major). CopyGM2L1 auto-selects DN2ZN which
// transposes during the fractal conversion. No branching needed here.
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

// Single L1→L0 + Mmad step: slice (curM, curKL0) from L1 at kOffL1, copy to L0
// (double-buffered via l0BufId offset = SPLT_HALF_L0_SIZE * l0BufId), then Mmad.
// MTE1_M sync ensures L1→L0 copy completes before Mmad. M_MTE1 releases L0.
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

// Inner kL0 loop: iterate baseK-sized chunks from the L1 buffer, L1→L0 + Mmad
// per chunk. L0 double-buffer ping-pong (l0BufId = kl0Idx & 1). isFinalAcc is
// true only on the last kL0 iteration of the last kL1 iteration.
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

// Outer kL1 loop driver: GM→L1 (kL1 block) → kL0 loop (baseK chunks) → release L1.
// L1 double-buffer ping-pong (l1BufId = kl1Idx & 1). kSegStart offsets the K
// position: kSeg*kSegLen for splitK>1 (non-fused), 0 for splitK==1 (fused).
// Replaces the former single-layer SpltKLoopDriver.
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

// Tile-loop driver for SpltMatmulCubeImpl: the m/n tile
// traversal + GM slice computation + K-loop + Fixpipe L0C->GM. Extracted to
// reduce NBNC of SpltMatmulCubeImpl (#3, 72 -> target<=50). Flag ops preserved.
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

        // Skip kSeg tiles whose start is beyond actual k.
        if (kSegStart >= k) {
            continue;
        }

        // Two-level Slice matching cann-samples matmul_kernel_swat.h.
        auto gmBlockARow = gmA.Slice(Te::MakeCoord(mPos, 0), Te::MakeShape(curM, k));
        // B slice is the same for transB and non-transB: GM B is
        // declared as DNExt(k,n) when transB (column-major view of (n,k) physical
        // data), so slicing (0, nPos, k, curN) works for both paths.
        auto gmBlockBCol = gmB.Slice(Te::MakeCoord(0, nPos), Te::MakeShape(k, curN));

        // GM output slice for this kSeg: tempSeg at [kSeg*m*n].
        // v2: temp element type follows L0C type (float for FP32/FP16/BF16,
        // int32_t for INT8). sizeof(float) == sizeof(int32_t) == 4, so the
        // byte offset arithmetic is identical; only the typed pointer differs.
        using L0CType = typename SpltL0CTypeTrait<T>::type;
        __gm__ uint8_t* tempBase = reinterpret_cast<__gm__ uint8_t*>(tempGm) +
                                   static_cast<int64_t>(kSeg) * m * n * sizeof(L0CType);
        auto gmCSeg = Te::MakeTensor(Te::MakeMemPtr<Te::Location::GM>(reinterpret_cast<__gm__ L0CType*>(tempBase)),
                                 Te::MakeFrameLayout<Te::NDExtLayoutPtn, SPLT_GM_FLOAT_C0>(m, n));

        auto tensorL0C = Te::MakeTensor(Te::MakeMemPtr<Te::Location::L0C, typename SpltL0CTypeTrait<T>::type>(0),
                                    Te::MakeFrameLayout<Te::NZLayoutPtn, SPLT_L0C_C0>(curM, curN));

        // Wait for previous tile's Fixpipe before writing L0C.
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

// Dispatch transB for SpltMatmulCubeImpl: constructs gmB (DNExt
// or NDExt based on transB) and calls SpltMatmulCubeTileLoop. Eliminates
// duplicate transB branch code (4 branches -> 2 + helper).
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
// CUBE MATMUL KERNEL (AIC_ONLY, Te::Mmad tensor_api)
//   M(m,k) x N(k,n) -> temp(splitK*m*n, FP32)
//   Single-buffer per tile, K-loop accumulates in L0C. Correctness first.
// ============================================================================
// The GM input tensor layout uses Te::LayoutTraitDefault<T> which derives C0 from the data type
// (32/sizeof(T): fp16=16, fp32=8), so no explicit C0 parameter is needed.
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
    // Use int64_t to prevent overflow when mTiles*nTiles
    // approaches INT32_MAX (large m, n).
    const int64_t totalMNTiles = static_cast<int64_t>(mTiles) * static_cast<int64_t>(nTiles);
    // 每个 batch 内的 totalTiles（batch 循环在外层）
    const int64_t totalTiles = totalMNTiles * static_cast<int64_t>(splitK);
    if (totalTiles <= 0) { return; }

    // GM base tensors (ND row-major entry).
    // GM-side Te::NDExtLayoutPtn MUST use dtype-correct C0
    // (32/sizeof(T): fp16=16, fp32=8). Default Te::LayoutTraitDefault<> uses
    // uint16_t => C0=16, which is WRONG for fp32 and causes CopyGmToCbufMultiND2Nz
    // to pack NZ fractals with the wrong inner-block size, scrambling data.
    // Use Te::LayoutTraitDefault<T> which sets both data type AND C0 (=32/sizeof(T)).
    // sparseTrans=1: dense×dense + transA. Physical A is
    // (k,m) row-major; declare as DNExt(m,k) (column-major view) so CopyGM2L1
    // transposes during the fractal conversion — same mechanism as transB.
    // sparseTrans=0: standard NDExt(m,k) (prune already transposed for sparse path).
    // transB=1: GM B physical is (n, k) row-major. Declare as
    // DNExt(k, n) (column-major) — CopyGM2L1 auto-selects DN2ZN which transposes.
    // DNExt and NDExt produce different tensor types, so the tile loop call must
    // branch at the call site (template dispatch on GmTensorA/GmTensorB types).
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

    // [OPT-P1] Drain all leftover flags (both L1 ping-pong slots + L0 + L0C).
    DrainCubeFlags();

    // Restore default layout transform mode (matches cann-samples
    // matmul_kernel_swat.h:162).
    SetMMLayoutTransform(false);
}

// ============================================================================
// EPILOGUE KERNEL (AIV_ONLY)
//   D = alpha * reduce_k(temp) + beta * C
//   temp is FP32 (splitK partial sums); C/D are T (FP16 or FP32).
//   Replaced scalar __gm__ access with DataCopyPad (GM<->UB)
//   + UB-local scalar GetValue/SetValue. Scalar GM access (__gm__ T* indexing)
//   is on the API blacklist (ascendc-api-best-practices) and silently drops
//   data on AIV cores — temp had correct values but D had zeros.
// INT8 + alpha=1 + beta=0 fast path for splitK epilogue.
// Uses int32 accumulation (Add<int32_t> on LocalTensor) instead of float Cast+Add,
// preserving exact int32 precision for large accumulation values.
// [OPT-INT32-VEC] splitK accumulation now uses Vector Add<int32_t> (in-place,
// dst==src0) instead of scalar GetValue/SetValue loop. Add<int32_t> is supported
// on arch35 (DAV_3510). DataCopyPad is used for all GM<->UB transfers.
// [OPT-INT8-VEC] INT8 output now uses vectorized Cast+clamp (replaces scalar clamp loop):
//   Cast int32→float (Vector) → Mins/Maxs clamp (Vector) → SpltCastFp32ToInt8 (Vector).
// The float intermediate is exact for post-clamp values in [-128,127].
// Flag operations mirror the existing SpltEpilogueProcessChunk pattern exactly.
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

    // Reinterpret float buffers as int32 (sizeof(float) == sizeof(int32_t) == 4)
    LocalTensor<int32_t> accIntUB = accUB.ReinterpretCast<int32_t>();
    LocalTensor<int32_t> tempIntUB = tempUB.ReinterpretCast<int32_t>();

    // 1. Load first splitK segment (int32 GM -> int32 UB)
    DataCopyPad(accIntUB, tempGM[base], copyAcc, padAcc);
    SetFlag<HardEvent::MTE2_V>(0);
    WaitFlag<HardEvent::MTE2_V>(0);

    // 2. Vector int32 accumulate remaining segments
    //    [OPT-INT32-VEC] Add<int32_t> (in-place: dst==src0) replaces the scalar
    //    GetValue/SetValue loop. PipeBarrier<PIPE_V> ensures the Vector pipeline
    //    completes before the V_MTE2 flag is set (WAR on tempIntUB).
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

    // 3. Output
    if constexpr (std::is_same_v<OutType, int32_t>) {
        // INT32 output: direct int32 UB -> GM (no Cast)
        SetFlag<HardEvent::V_MTE3>(0);
        WaitFlag<HardEvent::V_MTE3>(0);
        DataCopyPad(dGM[base], accIntUB, copyAcc);
    } else {
        // [OPT-INT8-VEC] INT8 output: vectorized Cast int32→float→clamp→int8.
        // tempUB (float) is reused as the Cast destination — it's no longer needed
        // for accumulation after the splitK loop above completed.
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

    // 4. Cross-iteration sync (matches existing SpltEpilogueProcessChunk pattern)
    SetFlag<HardEvent::MTE3_MTE2>(0);
    WaitFlag<HardEvent::MTE3_MTE2>(0);
}

__aicore__ inline float SpltGetScalarFromUB(LocalTensor<float>& ubTensor, int32_t index)
{
    return ubTensor.GetValue(static_cast<uint32_t>(index));
}

// Load full bias vector from GM to FP32 UB (count elements).
// BiasType=float (FP32/INT8): direct DataCopyPad to FP32 UB.
// BiasType=half (FP16): DataCopyPad raw to biasRawBuf, then Vector Cast to FP32 UB.
// BiasType=bfloat16_t (BF16): DataCopyPad raw to biasRawBuf, then SpltCastBf16ToFp32Vec.
// biasRawBuf must be pre-allocated with count * sizeof(BiasType) bytes when BiasType != float.
template <typename BiasType>
__aicore__ inline void SpltLoadBiasFullToFp32(
    LocalTensor<float>& biasFp32UB,
    TBuf<TPosition::VECCALC>& biasRawBuf,
    GlobalTensor<BiasType>& biasGM,
    int32_t count)
{
    if constexpr (std::is_same_v<BiasType, float>) {
        // FP32/INT8 path: bias is FP32 in GM, load directly
        DataCopyExtParams copyBias{1, static_cast<uint32_t>(count * sizeof(float)), 0, 0, 0};
        DataCopyPadExtParams<float> padBias{false, 0, 0, 0.0f};
        DataCopyPad(biasFp32UB, biasGM[0], copyBias, padBias);
        SetFlag<HardEvent::MTE2_V>(0);
        WaitFlag<HardEvent::MTE2_V>(0);
    } else {
        // FP16/BF16 path: load raw bias, then Vector Cast to FP32
        LocalTensor<BiasType> biasRawUB = biasRawBuf.Get<BiasType>();
        DataCopyExtParams copyBias{1, static_cast<uint32_t>(count * sizeof(BiasType)), 0, 0, 0};
        DataCopyPadExtParams<BiasType> padBias{false, 0, 0, BiasType(0)};
        DataCopyPad(biasRawUB, biasGM[0], copyBias, padBias);
        SetFlag<HardEvent::MTE2_V>(0);
        WaitFlag<HardEvent::MTE2_V>(0);
        if constexpr (std::is_same_v<BiasType, bfloat16_t>) {
            // BF16: use existing Vector Cast helper (Cast<float, bfloat16_t, CAST_NONE>)
            SpltCastBf16ToFp32Vec(biasFp32UB, biasRawUB, count);
        } else {
            // FP16: direct Vector Cast (Cast<float, half, CAST_NONE>)
            Cast(biasFp32UB, biasRawUB, AscendC::RoundMode::CAST_NONE, count);
            PipeBarrier<PIPE_V>();
        }
    }
}

// Allocate bias UB buffers and optionally load full bias vector from GM to FP32 UB.
// Shared by SpltEpilogueImpl (non-fused) and SpltFusedEpilogueImpl (fused) to
// eliminate duplicate bias init code (#2). Both paths set up the same
// GlobalTensor + TBuf + LocalTensor + conditional SpltLoadBiasFullToFp32.
// biasRawBuf is only allocated for non-float BiasType (FP16/BF16 Cast intermediate).
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

// Sub-helpers extracted from SpltEpilogueProcessChunk to reduce
// NBNC (139->~45), cyclomatic complexity (36->~10), and depth (6->3).
// Flag operations are preserved exactly from the original inline code.

// SplitK load + accumulate: loads first segment and accumulates remaining splitK segments.
template <typename T, typename TempType>
__aicore__ inline void SpltEpilogueChunkSplitKReduce(
    LocalTensor<float>& accUB, LocalTensor<float>& tempUB, LocalTensor<float>& dFp32UB,
    GlobalTensor<TempType>& tempGM,
    int64_t base, int32_t count, int32_t splitK, int64_t totalElem,
    DataCopyExtParams& copyAcc, DataCopyPadExtParams<TempType>& padAcc)
{
    if constexpr (std::is_same_v<TempType, float>) {
        DataCopyPad(accUB, tempGM[base], copyAcc, padAcc);
        SetFlag<HardEvent::MTE2_V>(0);
        WaitFlag<HardEvent::MTE2_V>(0);
    } else {
        LocalTensor<TempType> tempIntUB = tempUB.ReinterpretCast<TempType>();
        DataCopyPad(tempIntUB, tempGM[base], copyAcc, padAcc);
        SetFlag<HardEvent::MTE2_V>(0);
        WaitFlag<HardEvent::MTE2_V>(0);
        Cast(accUB, tempIntUB, AscendC::RoundMode::CAST_NONE, static_cast<int32_t>(count));
        PipeBarrier<PIPE_V>();
    }
    for (int32_t s = 1; s < splitK; ++s) {
        if constexpr (std::is_same_v<TempType, float>) {
            DataCopyPad(tempUB, tempGM[static_cast<uint64_t>(s) * totalElem + base], copyAcc, padAcc);
            SetFlag<HardEvent::MTE2_V>(0);
            WaitFlag<HardEvent::MTE2_V>(0);
            Add(accUB, accUB, tempUB, static_cast<int32_t>(count));
        } else {
            LocalTensor<TempType> tempIntUB = tempUB.ReinterpretCast<TempType>();
            DataCopyPad(tempIntUB, tempGM[static_cast<uint64_t>(s) * totalElem + base], copyAcc, padAcc);
            SetFlag<HardEvent::MTE2_V>(0);
            WaitFlag<HardEvent::MTE2_V>(0);
            Cast(dFp32UB, tempIntUB, AscendC::RoundMode::CAST_NONE, static_cast<int32_t>(count));
            PipeBarrier<PIPE_V>();
            Add(accUB, accUB, dFp32UB, static_cast<int32_t>(count));
        }
        SetFlag<HardEvent::V_MTE2>(0);
        WaitFlag<HardEvent::V_MTE2>(0);
    }
}

// Apply alpha scaling: D = alpha * acc (scalar or per-row vector).
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

// Apply beta*C with per-row vector scaling (betaVectorScaling path).
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

// Apply beta*C with scalar beta (non-vector-scaling path).
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
// bias + activation epilogue helpers.
// Epilogue chain: SplitK reduce -> alpha -> beta*C -> bias -> activation -> Cast -> store D
// bias 和 activation 在 FP32 域（dFp32UB）上操作。
// ============================================================================

// Apply bias to dFp32UB in the non-fused (chunk-based) path.
// bias is per-row broadcast: chunk 可能跨行，逐行取出 bias[r] 标量，对该行片段做 Adds。
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

// Apply ReLU activation: D = min(upperBound, max(threshold, D)).
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

// Apply GeLU activation (sigmoid equivalent, 11 steps with Exp clamp).
// 输入 x = dFp32UB，输出回写 dFp32UB。geluTemp 复用 tempBuf。
// 逐行执行（per-row, count=curN），geluTemp 需要一行大小 buffer。
// Step 6 clamp z<=20 防 Exp 溢出（不改变 golden 语义，sigmoid(20)≈1.0）。
// LocalTensor 按值传递（轻量级 handle，类似指针，避免临时对象无法绑定到非 const 引用）。
__aicore__ inline void SpltApplyGeLU(
    LocalTensor<float> dFp32UB, LocalTensor<float> geluTemp,
    int32_t count, float geluScaling)
{
    // Step 1: geluTemp = x²
    Mul(geluTemp, dFp32UB, dFp32UB, count);
    PipeBarrier<PIPE_V>();
    // Step 2: geluTemp = 0.044715 * x²
    Muls(geluTemp, geluTemp, 0.044715f, count);
    PipeBarrier<PIPE_V>();
    // Step 3: geluTemp = 0.044715 * x³
    Mul(geluTemp, geluTemp, dFp32UB, count);
    PipeBarrier<PIPE_V>();
    // Step 4: geluTemp = x + 0.044715 * x³
    Add(geluTemp, geluTemp, dFp32UB, count);
    PipeBarrier<PIPE_V>();
    // Step 5: z = sqrt(8/π) * (x + 0.044715 * x³)
    Muls(geluTemp, geluTemp, 1.5957691f, count);
    PipeBarrier<PIPE_V>();
    // Step 6: clamp z <= 20.0f 防 Exp 溢出
    Mins(geluTemp, geluTemp, 20.0f, count);
    PipeBarrier<PIPE_V>();
    // Step 7: exp(z)
    Exp(geluTemp, geluTemp, count);
    PipeBarrier<PIPE_V>();
    // Step 8: x = geluScaling * x
    Muls(dFp32UB, dFp32UB, geluScaling, count);
    PipeBarrier<PIPE_V>();
    // Step 9: x = geluScaling * x * exp(z)
    Mul(dFp32UB, dFp32UB, geluTemp, count);
    PipeBarrier<PIPE_V>();
    // Step 10: geluTemp = 1 + exp(z)
    Adds(geluTemp, geluTemp, 1.0f, count);
    PipeBarrier<PIPE_V>();
    // Step 11: D = geluScaling * x * exp(z) / (1 + exp(z))
    Div(dFp32UB, dFp32UB, geluTemp, count);
    PipeBarrier<PIPE_V>();
}

// Cast FP32 -> OutType and store to GM (UB -> GM via DataCopyPad).
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

// Process one chunk of the epilogue loop (splitK reduce + alpha/beta + output).
// Extracted from SpltEpilogueImpl to reduce NBNC. All SetFlag/WaitFlag preserved.
// v2: templated on (T, OutType). AccType is always float (INT8 int32 temp is
// Cast to float on load — for the non-fast-path). INT8 output
// uses Mins/Maxs to clamp to [-128,127] before Cast (CAST_SATURATE unavailable).
// Added halfBuf param: float→int8_t requires two-step Cast.
// Added int32 fast path dispatch for alpha=1+beta=0 (no float).
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
    // INT32 fast path: alpha=1+beta=0+无 bias+无 activation 时直接 int32 累加输出
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

    SpltEpilogueChunkSplitKReduce<T, TempType>(accUB, tempUB, dFp32UB, tempGM,
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

// Shared scaling helpers: extracted from SpltEpilogueImpl and
// SpltFusedEpilogueImpl to eliminate duplicate code (alphaVecGM/betaVecGM setup
// and UB load). Flag operations preserved exactly.

// Set up alpha/beta vector global buffers when vector scaling is enabled.
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

// Allocate UB and load alpha/beta vectors from GM when vector scaling is enabled.
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

// Initialize TBufs for SpltEpilogueImpl (CHUNK-based allocation).
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
    // [LOW-8] BF16 halfBuf allocation removed: SpltCastFp32ToBf16Vec does not
    // use halfBuf (Vector Cast<bfloat16_t,float> needs no intermediate buffer).
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

// Set per-batch GM pointers for SpltEpilogueBatchLoop.
// Offsets temp/c/d/bias GM by batch stride; reloads bias when biasStride != 0.
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

// Batch loop for SpltEpilogueImpl: iterates batches, sets per-batch GM pointers,
// reloads bias if needed, and runs the chunk processing loop.
// Extracted to reduce NBNC of SpltEpilogueImpl (#8, was 86, target<=50).
// Flag operations and loop order preserved exactly.
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
// v2: SpltEpilogueImpl templated on (T, OutType). AccType is always float
// (INT8 int32 temp is Cast to float on load). Non-INT8 path: OutType == T.
template <typename T, typename OutType = T>
__aicore__ inline void SpltEpilogueImpl(GM_ADDR tempGm, GM_ADDR cGm, GM_ADDR dGm, GM_ADDR tilingGm)
{
    using TempType = typename SpltL0CTypeTrait<T>::type;  // float or int32_t
    using BiasType = SpltBiasDType<T>;  // bias dtype: T for FP32/FP16/BF16, float for INT8
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
// [OPT-P2] FUSED MATMUL + EPILOGUE KERNEL (__mix__(1,2))
//   splitK==1 path: AIC does matmul -> Fixpipe L0C->UB -> CrossCore notify AIV
//   AIV does CrossCore wait -> Vector(alpha*acc+beta*C+Cast) -> DataCopyPad UB->GM(D)
//   Eliminates GM temp round-trip + 1 kernel launch.
//
// Architecture: __mix__(1,2) = 1 AIC + 2 AIV per block. AIC and AIVs are on
// separate physical cores with separate UBs. CopyL0C2UB with subBlockId routes
// the full tile data to the target AIV's UB. Tiles alternate between AIV0
// (subBlockId=false) and AIV1 (subBlockId=true) for load balancing.
// No ping-pong (single UB slot per AIV) for simplicity; correctness first.
// ============================================================================

// CrossCore sync flag constants (intra-block MODE 4, blaze-sync-patterns §7).
static constexpr uint8_t  SPLT_SYNC_MODE_4      = 4;    // intra-block mode
static constexpr uint16_t SPLT_AIV0_SYNC_AIC    = 4;   // AIV0->AIC
static constexpr uint16_t SPLT_AIC_SYNC_AIV0    = 6;   // AIC->AIV0
static constexpr uint16_t SPLT_AIV1_SYNC_AIC    = 4 + 16;  // AIV1->AIC (FLAG_ID_MAX offset)
static constexpr uint16_t SPLT_AIC_SYNC_AIV1    = 6 + 16;  // AIC->AIV1

// ----------------------------------------------------------------------------
// AIC side: K-loop + Fixpipe L0C->UB (subBlockId alternation) + CrossCore
// ----------------------------------------------------------------------------
// Tile-loop driver for SpltFusedMatmulCubeImpl: the m/n tile
// traversal + K-loop + Fixpipe L0C->UB + CrossCore sync. Extracted to reduce
// NBNC of SpltFusedMatmulCubeImpl (#4, 76 -> target<=50) and share SpltKL1Loop
// with SpltMatmulCubeTileLoop (eliminates duplicate #1). Flag ops preserved.
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
        // UB nAlign must use L0C C0 (=16), NOT the dtype C0 (8 for FP32).
        // The Fixpipe writes L0C NZ fractals (C0=16) to UB NDExt. The row stride in UB
        // is CeilAlign(curN, 16) elements. If curNAlign uses dtype C0 (8), the row stride
        // mismatches what the AIV expects (CeilAlign(curN, 16)), causing data misalignment
        // for non-16-aligned curN (e.g. curN=36 -> AIC stride=40, AIV stride=48).
        const uint64_t curNAlign = static_cast<uint64_t>((curN + SPLT_L0C_C0 - 1) / SPLT_L0C_C0) * SPLT_L0C_C0;

        auto gmBlockARow = gmA.Slice(Te::MakeCoord(mPos, 0), Te::MakeShape(curM, k));
        // B slice is the same for both paths (DNExt GM layout handles transB).
        auto gmBlockBCol = gmB.Slice(Te::MakeCoord(0, nPos), Te::MakeShape(k, curN));
        // v2: L0C type follows SpltL0CTypeTrait (float for FP32/FP16/BF16, int32_t for INT8).
        auto tensorL0C = Te::MakeTensor(Te::MakeMemPtr<Te::Location::L0C, typename SpltL0CTypeTrait<T>::type>(0),
                                    Te::MakeFrameLayout<Te::NZLayoutPtn, SPLT_L0C_C0>(curM, curN));

        WaitFlag<HardEvent::FIX_M>(0);
        bool cmatrixInitVal = true;
        // kSegStart=0: fused path has splitK==1, no K-segment offset.
        SpltKL1Loop<T>(gmBlockARow, gmBlockBCol, tensorL0C,
                       curM, curN, k, kSegLen, 0, baseK, td.kL1Size,
                       cmatrixInitVal, l1cfg);

        // ---- [OPT-P2] Fixpipe L0C -> UB (target AIV via subBlockId) ----
        // Alternate: even localTileIdx -> AIV0 (subBlockId=false), odd -> AIV1.
        const bool toAiv1 = (localTileIdx & 0x1) == 1;
        const uint16_t aivWaitFlag = toAiv1 ? SPLT_AIV1_SYNC_AIC : SPLT_AIV0_SYNC_AIC;
        const uint16_t aivNotifyFlag = toAiv1 ? SPLT_AIC_SYNC_AIV1 : SPLT_AIC_SYNC_AIV0;

        // Wait for target AIV to release UB (reverse dep, first round preset by AIV).
        CrossCoreWaitFlag<SPLT_SYNC_MODE_4, PIPE_FIX>(aivWaitFlag);

        // UB destination tensor at offset 0 (single slot, no ping-pong).
        // Te::NDExtLayoutPtn C0 must match L0C C0 (=16 on DAV-3510),
        // NOT the dtype C0 (32/sizeof(float)=8 for FP32). Mismatch causes the
        // Fixpipe to produce wrong fractal splitting, scrambling the output.
        // v2: UB tensor type follows L0C type (int32_t for INT8).
        auto layoutUB = Te::MakeFrameLayout<Te::NDExtLayoutPtn, SPLT_L0C_C0>(curMAlign, curNAlign);
        auto ubTensor = Te::MakeTensor(Te::MakeMemPtr<Te::Location::UB, typename SpltL0CTypeTrait<T>::type>(0), layoutUB);

        Te::FixpipeParams fixpParams;
        fixpParams.unitFlag = static_cast<uint8_t>(SPLT_FINAL_ACCUMULATION);
        fixpParams.subBlockId = toAiv1;   // route to AIV0 or AIV1
        auto copyL0C2UB = Te::MakeCopy(Te::CopyL0C2UB{});
        Te::Copy(copyL0C2UB.with(fixpParams), ubTensor, tensorL0C);

        // Notify target AIV that data is ready.
        CrossCoreSetFlag<SPLT_SYNC_MODE_4, PIPE_FIX>(aivNotifyFlag);

        SetFlag<HardEvent::FIX_M>(0);
        localTileIdx++;
    }
}

// Dispatch transB for SpltFusedMatmulCubeImpl: constructs gmB
// (DNExt or NDExt based on transB) and calls SpltFusedMatmulCubeTileLoop.
// Eliminates duplicate transB branch code (4 branches -> 2 + helper).
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
    // Use int64_t to prevent overflow when mTiles*nTiles
    // approaches INT32_MAX (large m, n). Consistent with non-Fused path.
    // 每个 batch 内的 totalTiles（batch 循环在外层）
    const int64_t totalTiles = static_cast<int64_t>(mTiles) * static_cast<int64_t>(nTiles);
    if (totalTiles <= 0) { return; }

    // sparseTrans handling: see SpltMatmulCubeImpl for rationale.
    // DNExt/NDExt produce different types, so branch at call site for both A and B.
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
        // CrossCore drain 每 batch 执行：等待本 batch 最后一个 tile 的
        // 目标 AIV 释放 UB（AIV 处理完后 SetFlag，AIC 这里 WaitFlag 消费）。
        // 每 batch 独立 drain，避免残留 flag 跨 batch 累积。
        if (localTileIdx > 0) {
            const bool lastToAiv1 = ((localTileIdx - 1) & 0x1) == 1;
            const uint16_t lastAivWaitFlag = lastToAiv1 ? SPLT_AIV1_SYNC_AIC : SPLT_AIV0_SYNC_AIC;
            CrossCoreWaitFlag<SPLT_SYNC_MODE_4, PIPE_FIX>(lastAivWaitFlag);
        }
    }

    // Drain L1 double-buffer flags.
    DrainCubeFlags();

    SetMMLayoutTransform(false);
}

// ----------------------------------------------------------------------------
// Shared output helpers for SpltFusedEpilogueImpl.
// Extracted to eliminate duplicate code (#11) and reduce NBNC/complexity/depth.
// CRITICAL: SetFlag/WaitFlag positions preserved exactly.
// v2: templated on AccType (float for FP32/FP16/BF16, int32_t for INT8) and
// OutType (output dtype). INT8 output uses CAST_SATURATE; INT32 uses ROUND.
// ----------------------------------------------------------------------------

// FP32 output: V_MTE3 sync + batch/per-row DataCopyPad (called when OutType=float).
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

// Non-FP32 output: Cast per-row + V_MTE3 + DataCopyPad + MTE3_V.
// v2: INT8 output — clamp via Mins/Maxs then Cast ROUND (CAST_SATURATE unavailable).
// [OPT-BF16-VEC] BF16 output — Vector Cast<bfloat16_t, float, CAST_ROUND>.
// dFp32Buf is used as the float temp for INT8 clamping (separate from dTBuf output).
// Added halfBuf param: float→int8_t requires a two-step
// Cast (float→half→int8_t) because DAV-3510 does not support direct float→int8.
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
            // INT8: clamp float to [-128, 127] then Cast to int8 via float→half→int8.
            // Direct float→int8_t Cast is not supported by
            // DAV-3510; use SpltCastFp32ToInt8 (two-step Vector Cast).
            LocalTensor<float> clampUB = dFp32Buf.Get<float>();
            Mins(clampUB, srcUB[r * curNAlign], 127.0f, curN);
            Maxs(clampUB, clampUB, -128.0f, curN);
            SpltCastFp32ToInt8(dUB, clampUB, halfBuf, curN);
        } else if constexpr (std::is_same_v<OutType, __bf16>) {
            // [OPT-BF16-VEC] Vectorized FP32→BF16 Cast (replaces scalar loop).
            // SpltCastFp32ToBf16Vec uses Vector Cast<bfloat16_t,float> directly;
            // no intermediate buffer (dFp32Buf/halfBuf) is needed.
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

// Direct int32 UB→GM output for fused INT8→INT32 fast path.
// No Cast, no float intermediate — preserves exact int32 precision for large
// accumulation values (float loses precision outside [-2^24, 2^24]).
// Logic mirrors SpltFusedEpilogueOutputFp32 (sizeof(int32_t)==sizeof(float)==4).
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

// [OPT-INT8-VEC] Vectorized int32→int8 clamp for fused INT8→INT8 fast path.
// Replaces the scalar GetValue/SetValue loop (2.1ms for 128×128) with Vector ops:
//   1. Cast int32→float (Vector, CAST_NONE — exact for [-2^24, 2^24])
//   2. Mins/Maxs clamp to [-128, 127] (Vector)
//   3. SpltCastFp32ToInt8 (float→half→int8, two-step Vector Cast)
// The float intermediate is exact for all int32 values in [-128,127] (post-clamp),
// and Cast int32→float is exact within [-2^24, 2^24] which covers all practical
// accumulation results (alpha=1+beta=0 fast path, no scaling).
// Flag pattern mirrors SpltFusedEpilogueOutputCast (per-row V_MTE3 + MTE3_V).
// dFp32Buf/halfBuf are pre-allocated in SpltFusedEpilogueImpl for INT8 fast path.
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
        // 1. Cast int32 → float for this row
        Cast(clampUB, srcUB[r * curNAlign], AscendC::RoundMode::CAST_NONE, curN);
        PipeBarrier<PIPE_V>();
        // 2. Clamp to [-128, 127] (Mins then Maxs, both Vector)
        Mins(clampUB, clampUB, 127.0f, curN);
        Maxs(clampUB, clampUB, -128.0f, curN);
        // 3. Cast float → int8 via half (SpltCastFp32ToInt8 does float→half→int8)
        SpltCastFp32ToInt8(dUB, clampUB, halfBuf, curN);
        // 4. Output row to GM (flag pattern matches SpltFusedEpilogueOutputCast)
        SetFlag<HardEvent::V_MTE3>(0);
        WaitFlag<HardEvent::V_MTE3>(0);
        copyOut.blockLen = static_cast<uint32_t>(curN * sizeof(int8_t));
        DataCopyPad(dGM[static_cast<int64_t>(mPos + r) * n + nPos], dUB, copyOut);
        SetFlag<HardEvent::MTE3_V>(0);
        WaitFlag<HardEvent::MTE3_V>(0);
    }
}

// Fast path: alpha=1, beta=0 -> output acc directly to GM.
// v2: INT8 path (AccType=int32_t) requires Cast int32 -> float -> OutType
// (INT8 output: clamp via Mins/Maxs; INT32 output: direct int32 write when AccType==OutType).
// Added curMAlign param; the INT8 Cast int32->float must cover the
// full tile (curMAlign * curNAlign elements), not just curN (one row). Using curN left
// rows 1+ of accFp32 uninitialized, causing ~94% output mismatches (15424/16384).
// Added halfBuf param for float→half→int8 two-step Cast.
// INT8 fast path now uses direct int32 processing (no float Cast):
//   - INT32 output: direct DataCopyPad int32 UB→GM
//   - INT8 output: [OPT-INT8-VEC] vectorized Cast int32→float→clamp→int8
// Float path retained only for general path (alpha!=1 || beta!=0, tolerance allows).
template <typename AccType, typename OutType>
__aicore__ inline void SpltFusedEpilogueFastPath(
    LocalTensor<AccType>& accUB, GlobalTensor<OutType>& dGM,
    TBuf<TPosition::VECCALC>& dTBuf, TBuf<TPosition::VECCALC>& dFp32Buf,
    TBuf<TPosition::VECCALC>& halfBuf,
    int32_t mPos, int32_t nPos, int32_t curM, int32_t curN,
    uint64_t curMAlign, uint64_t curNAlign, int32_t n)
{
    if constexpr (std::is_same_v<AccType, float>) {
        // FP32/FP16/BF16: acc is float, Cast (or direct) to OutType.
        if constexpr (std::is_same_v<OutType, float>) {
            SpltFusedEpilogueOutputFp32(accUB, dGM, mPos, nPos, curM, curN, curNAlign, n);
        } else {
            SpltFusedEpilogueOutputCast<OutType>(accUB, dTBuf, dFp32Buf, halfBuf, dGM,
                                                 mPos, nPos, curM, curN, curNAlign, n);
        }
    } else {
        // INT8 fast path: process int32 directly, skip float Cast.
        if constexpr (std::is_same_v<OutType, int32_t>) {
            // INT32 output: direct int32 UB→GM (no Cast, no float intermediate)
            SpltFusedEpilogueOutputInt32Direct(accUB, dGM, mPos, nPos, curM, curN, curNAlign, n);
        } else {
            // INT8 output: vectorized clamp int32→float→clamp→int8 (replaces scalar loop)
            SpltFusedEpilogueOutputInt8VecClamp(accUB, dTBuf, dFp32Buf, halfBuf, dGM,
                                                mPos, nPos, curM, curN, curNAlign, n);
        }
    }
}

// Beta*C accumulation loop (general path sub-step).
// v2: C dtype == OutType (validate_descriptors enforces C matches D).
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
            // [OPT-BF16-VEC] BF16 uses vectorized bit-operation Cast (replaces scalar).
            // tempBuf reused as uint32 tmp for Cast<uint32,uint16>+ShiftLeft.
            // INT8 still uses scalar Cast: DAV-3510 Vector Cast
            // does not support int8_t→float (silent no-op, same as float→int8_t).
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
// Sub-helpers extracted from SpltFusedEpilogueGeneralPath to reduce CCN (was 26)
// and NBNC (was 95). Each helper preserves the exact flag/pipe operations.
// ----------------------------------------------------------------------------

// Apply alpha scaling to accumulator -> dFp32UB.
// Handles both scalar alpha and per-row vector alpha, for both float and
// int32 accumulators (INT8 requires Cast int32->float before Muls).
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
            // INT8: Cast int32 acc -> float, then Muls by alpha.
            Cast(dFp32UB, accUB, AscendC::RoundMode::CAST_NONE,
                 static_cast<int32_t>(curMAlign * curNAlign));
            PipeBarrier<PIPE_V>();
            Muls(dFp32UB, dFp32UB, alpha, static_cast<int32_t>(curMAlign * curNAlign));
        }
    }
}

// Load non-FP32 bias chunk from GM and Cast to FP32.
// Used by SpltFusedEpilogueApplyBias for FP16/BF16 bias types.
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

// Apply bias to dFp32UB in the fused general path.
// Handles both per-chunk (biasChunkMode==1: load curM rows from GM) and
// full-load (biasChunkMode==0: biasVecUB already loaded) modes.
// For FP16/BF16 bias, raw data is Cast to FP32 before per-row Adds.
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
            // FP32/INT8: bias is FP32 in GM, load directly to tempBuf
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
            // FP16/BF16: load raw bias, Cast to FP32, per-row Adds
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

// Apply activation (ReLU or GeLU) to dFp32UB in the fused general path.
// ReLU: batch execution over full tile. GeLU: per-row with tempBuf reuse.
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

// General path: D = alpha*acc + beta*C + bias + activation -> output.
// v2: INT8 path (AccType=int32_t) requires Cast int32 -> float before Muls.
// Added halfBuf param for float→half→int8 two-step Cast.
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
// AIV side: CrossCore wait -> Vector(alpha*acc+beta*C+Cast) -> DataCopyPad UB->GM
// Each AIV processes tiles assigned to it (alternating with the other AIV).
// ----------------------------------------------------------------------------
// Tile-loop driver for SpltFusedEpilogueImpl: the per-tile
// CrossCore wait -> FastPath/GeneralPath -> CrossCore notify traversal. Extracted
// to reduce NBNC of SpltFusedEpilogueImpl (#5, 73 -> target<=50). Flag ops
// preserved exactly. TBufs/TPipe are owned by the caller (Impl) and passed by
// reference; their lifetime spans the loop.
// v2: templated on (AccType, OutType) — AccType from SpltL0CTypeTrait<T>,
// OutType is the output dtype (equals T for non-INT8; int8_t/int32_t for INT8).
// Added halfBuf param for float→half→int8 two-step Cast.
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

        // Only participate in sync for tiles assigned to this AIV.
        // The AIC only notifies and waits for the target AIV.
        if (!myTile) {
            localTileIdx++;
            continue;
        }

        // 1. Wait for AIC Fixpipe to complete (data ready in UB).
        CrossCoreWaitFlag<SPLT_SYNC_MODE_4, PIPE_V>(aivWaitFlag);

        // 2. Process the full tile (this AIV handles ALL curM rows).
        LocalTensor<AccType> accUB = accBuf.Get<AccType>();

        // FastPath 条件更新：有 bias 或 activation 时走 GeneralPath（方式 A，见 §3.2）
        const bool isFastPath = (alphaVectorScaling == 0 && alpha == 1.0f && beta == 0.0f
                                 && biasDevPtr == 0 && activationType == 0);
        if (isFastPath) {
            // Fast path: alpha=1, beta=0, no bias, no activation -> Cast acc -> OutType and write to GM.
            // Pass curMAlign so INT8 Cast covers the full tile.
            // Pass halfBuf for float→half→int8 two-step Cast.
            SpltFusedEpilogueFastPath<AccType, OutType>(accUB, dGM, dTBuf, dFp32Buf, halfBuf,
                                         mPos, nPos, curM, curN, curMAlign, curNAlign, n);
        } else {
            // General path: D = alpha*acc + beta*C + bias + activation.
            SpltFusedEpilogueGeneralPath<AccType, OutType, CType>(accUB, cGM, dGM,
                dFp32Buf, cTBuf, cFp32Buf, tempBuf, dTBuf, halfBuf, biasRawBuf,
                mPos, nPos, curM, curN, curMAlign, curNAlign, n,
                alpha, beta, alphaVectorScaling, betaVectorScaling,
                alphaVecUB, betaVecUB,
                biasVecUB, biasVecGM, biasDevPtr, biasChunkMode,
                activationType, reluThreshold, reluUpperBound, geluScaling);
        }

        // 3. Notify AIC that this AIV is done with the UB slot.
        CrossCoreSetFlag<SPLT_SYNC_MODE_4, PIPE_MTE3>(aivNotifyFlag);
        localTileIdx++;
    }
}

// Initialize TBufs for SpltFusedEpilogueImpl (tile-based allocation).
// Extracted to eliminate duplicate buffer init code and reduce NBNC.
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
    // [LOW-8] BF16 fast-path dFp32Buf allocation removed: SpltCastFp32ToBf16Vec
    // (called in SpltFusedEpilogueOutputCast for BF16) does not use dFp32Buf.
    if constexpr (!std::is_same_v<OutType, float>) {
        pipe.InitBuffer(dTBuf, static_cast<uint32_t>(maxNAlign * sizeof(OutType)));
    }
    if constexpr (std::is_same_v<OutType, int8_t>) {
        pipe.InitBuffer(halfBuf, static_cast<uint32_t>(maxNAlign * sizeof(half)));
    }
    // tempBuf/cTBuf/cFp32Buf 分配条件扩展
    // tempBuf: beta*C temp | biasChunkUB (biasChunkMode==1, dtype=BiasType) | geluTemp (activationType==2)
    // cTBuf: C load (beta!=0)
    // cFp32Buf: C Cast (beta!=0, T!=float)
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

// Set up cGM/dGM global buffers for SpltFusedEpilogueImpl.
// Extracted to reduce NBNC of SpltFusedEpilogueImpl (53 -> ~45).
template <typename T, typename OutType>
__aicore__ inline void SpltFusedEpilogueInitGM(
    GlobalTensor<T>& cGM, GlobalTensor<OutType>& dGM,
    GM_ADDR cGm, GM_ADDR dGm, int32_t m, int32_t n)
{
    cGM.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(cGm), static_cast<uint64_t>(m) * n);
    dGM.SetGlobalBuffer(reinterpret_cast<__gm__ OutType*>(dGm), static_cast<uint64_t>(m) * n);
}

// Set per-batch GM pointers for SpltFusedEpilogueBatchLoop.
// Offsets c/d/bias GM by batch stride; reloads full bias when biasChunkMode==0.
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

// Batch loop for SpltFusedEpilogueImpl: iterates batches, sets per-batch GM
// pointers, reloads bias if needed, and runs the tile processing loop.
// Extracted to reduce NBNC of SpltFusedEpilogueImpl (#9, was 95, target<=50).
// Flag operations and CrossCore sync preserved exactly.
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
        // 无 tile 的 block 跳过 CrossCoreSetFlag，避免残留 flag
        // 影响下一个 batch 的同步。
        if (static_cast<int64_t>(rawBlockId) >= totalTiles) { continue; }
        CrossCoreSetFlag<SPLT_SYNC_MODE_4, PIPE_MTE3>(aivNotifyFlag);
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

// Initialize bias buffers for SpltFusedEpilogueImpl, dispatching on
// biasChunkMode. Extracted from SpltFusedEpilogueImpl to reduce NBNC (#2).
//   biasChunkMode == 0: full bias load via SpltInitBiasBuffers (UB-resident).
//   biasChunkMode == 1: chunk-mode GM-resident bias (lazy per-tile load),
//                       only allocates the Cast intermediate (biasRawBuf).
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

// v2: SpltFusedEpilogueImpl templated on (T, OutType). AccType derived via
// SpltL0CTypeTrait<T>. Non-INT8 path: OutType == T (backward compat).
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
    // Bias UB allocation + loading AFTER accBuf (accBuf must stay at
    // UB offset 0 to match AIC's CopyL0C2UB target).
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
// [OPT-P2] Fused matmul+epilogue kernel entries (splitK==1 path).
// v2: added BF16 / INT8 / INT8_I32 entries.
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

// v2: BF16 fused — re-enabled.
// [OPT-BF16-VEC] BF16 Cast uses Vector Cast<bfloat16_t, float> (CANN 9.1.0
// supports it in cast_round_all). Cube Mmad (tensor_api) routes BF16 through
// the NORMAL path (dtype-agnostic).
extern "C" __global__ __aicore__ __mix__(1, 2) void splt_fused_matmul_kernel_bf16(
    GM_ADDR aPrunedGm, GM_ADDR bGm, GM_ADDR cGm, GM_ADDR dGm, GM_ADDR tilingGm)
{
    if ASCEND_IS_AIC {
        SpltFusedMatmulCubeImpl<__bf16>(aPrunedGm, bGm, dGm, tilingGm);
    } else {
        SpltFusedEpilogueImpl<__bf16>(cGm, dGm, tilingGm);
    }
}

// v2: INT8 fused — INT8 output (saturation Cast).
extern "C" __global__ __aicore__ __mix__(1, 2) void splt_fused_matmul_kernel_int8(
    GM_ADDR aPrunedGm, GM_ADDR bGm, GM_ADDR cGm, GM_ADDR dGm, GM_ADDR tilingGm)
{
    if ASCEND_IS_AIC {
        SpltFusedMatmulCubeImpl<int8_t>(aPrunedGm, bGm, dGm, tilingGm);
    } else {
        SpltFusedEpilogueImpl<int8_t, int8_t>(cGm, dGm, tilingGm);
    }
}

// v2: INT8 fused — INT32 output (direct int32 write via ROUND Cast).
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
// Kernel entry points (one per dtype), dispatched by host launchers.
// v2: added BF16 / INT8 entries. INT8 matmul (cube) is shared between
// INT8->INT8 and INT8->INT32 (cube only writes int32 temp; output type
// dispatch happens in epilogue/fused).
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
// v2: BF16 cube — re-enabled.
// Cube Mmad (tensor_api) routes BF16 through the NORMAL path (dtype-agnostic);
// no Vector Cast is involved in the cube-only kernel. L0C=float via SpltL0CTypeTrait.
extern "C" __global__ __aicore__ void splt_matmul_kernel_bf16(
    GM_ADDR aPrunedGm, GM_ADDR bGm, GM_ADDR tempGm, GM_ADDR tilingGm)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    SpltMatmulCubeImpl<__bf16>(aPrunedGm, bGm, tempGm, tilingGm);
}
// v2: INT8 cube (L0C=int32_t via SpltL0CTypeTrait; shared by INT8->INT8 / INT8->INT32).
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
// v2: BF16 epilogue — re-enabled.
// [OPT-BF16-VEC] BF16 Cast (FP32<->BF16) uses Vector Cast (CANN 9.1.0
// supports Cast<bfloat16_t,float> in cast_round_all and Cast<float,bfloat16_t>
// in cast_none).
extern "C" __global__ __aicore__ void splt_epilogue_kernel_bf16(
    GM_ADDR tempGm, GM_ADDR cGm, GM_ADDR dGm, GM_ADDR tilingGm)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    SpltEpilogueImpl<__bf16>(tempGm, cGm, dGm, tilingGm);
}
// v2: INT8 epilogue — INT8 output (saturation Cast).
extern "C" __global__ __aicore__ void splt_epilogue_kernel_int8(
    GM_ADDR tempGm, GM_ADDR cGm, GM_ADDR dGm, GM_ADDR tilingGm)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    SpltEpilogueImpl<int8_t, int8_t>(tempGm, cGm, dGm, tilingGm);
}
// v2: INT8 epilogue — INT32 output (direct int32 write).
extern "C" __global__ __aicore__ void splt_epilogue_kernel_int8_i32(
    GM_ADDR tempGm, GM_ADDR cGm, GM_ADDR dGm, GM_ADDR tilingGm)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    SpltEpilogueImpl<int8_t, int32_t>(tempGm, cGm, dGm, tilingGm);
}

// ============================================================================
// Host-side launchers (ASC TU): wrap <<<>>>. host.cpp calls these as C funcs.
// v2: switch four-branch dispatch (FP32/FP16/BF16/INT8) + INT8 outDataType
// sub-branch for epilogue/fused (INT8 vs INT32 output).
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

// v2: outDataType added — INT8 path selects INT8 (saturation Cast) vs INT32 (direct write).
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

// [OPT-P2] Fused matmul+epilogue launcher (splitK==1 path).
// v2: outDataType added — INT8 path selects INT8 vs INT32 output entry.
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
