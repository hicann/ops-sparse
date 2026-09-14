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
 * \file spgemm_merge_emit.h
 * \brief SpGEMM arch22 融合归并的发射块（自 spgemm_merge.h 切出）。
 *
 * 承载 SpgemmMergeFusedRowImpl 快路径的五个发射实现：
 *   - SpgemmMergeEmitVec7Body    da == 7 实型向量发射体
 *   - SpgemmMergeEmitSegMajorOne 段主序标量发射的单段循环体
 *   - SpgemmMergeEmitLayerMajor  层主序 + 2 路 ILP 标量发射
 *   - SpgemmMergeEmitVec8        da 为 8 倍数、实型的向量发射块
 *   - SpgemmMergeEmitVecC64      complex64 的向量发射块
 */

// 本文件是 spgemm_merge.h 的内部片段，只允许由 spgemm_merge.h 在原定义位置
// #include，不是独立可用的头文件。
#ifndef SPGEMM_ARCH22_MERGE_H
#error "spgemm_merge_{emit,row}.h must only be included from spgemm_merge.h"
#endif

#ifndef SPGEMM_ARCH22_MERGE_EMIT_H
#define SPGEMM_ARCH22_MERGE_EMIT_H

/**
 * da == 7 的实型向量发射体，从 SpgemmMergeFusedRowImpl 的 da=7 向量块内层外提。
 * 谓词已通过时发射本行。必须 always_inline。
 */
template <uint32_t DA_FIXED, typename T, typename VEC> [[gnu::always_inline]] __aicore__ inline void SpgemmMergeEmitVec7Body(
    const AscendC::GlobalTensor<T> &bValGm, const SpgemmValueOps<T> &ops,
    const VEC &vec, uint32_t vecSlotOff, const AscendC::GlobalTensor<T> &aValGm,
    int32_t aBase, const int32_t (&head)[DA_FIXED], int32_t hdStep, int32_t stride, uint32_t segLen, uint32_t total, uint32_t pad, uint32_t minCur)
{
    // 跨行 V→S 令牌：Init 预置一个、每个发射行消一个置一个、Process 收尾消掉。
    AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID6);
    // 视图全部由 vec.off 按分区表派生，不新增 ctx 成员。
    // 分区步长用编译期常量 kSpgemmVecEmitCap。
    constexpr uint32_t vcap = kSpgemmVecEmitCap; const AscendC::LocalTensor<float> poolF = vec.off.template ReinterpretCast<float>();
    const AscendC::LocalTensor<uint32_t> mod7U = vec.offU[kSpgemmVec7Mod * vcap];
    const AscendC::LocalTensor<int32_t> lay7 = vec.off[kSpgemmVec7Layer * vcap];
    const AscendC::LocalTensor<int32_t> lay7B = vec.off[kSpgemmVec7LayerB * vcap];
    const AscendC::LocalTensor<int32_t> colS = vec.off[kSpgemmVec7Col * vcap];
    const AscendC::LocalTensor<float> valS = poolF[kSpgemmVec7Val * vcap];
    const AscendC::LocalTensor<uint32_t> shiftU = vec.offU[kSpgemmVec7Shift * vcap + pad * 64u];
    const AscendC::LocalTensor<int32_t> mod7I = vec.off[kSpgemmVec7ModIdx * vcap];
    const AscendC::LocalTensor<float> bA = poolF[kSpgemmVec7Base * vcap + 16u];
    // B 值置换落点：与 bA 同分区块，互不重叠。
    static_assert(kSpgemmVec7Base * kSpgemmVecEmitCap * sizeof(int32_t) + 128u + 64u * sizeof(float) <= (kSpgemmVec7Base + 1u) * kSpgemmVecEmitCap *
                      sizeof(int32_t),
                  "bT staging must stay inside slot 14");
    const AscendC::LocalTensor<float> bT = poolF[kSpgemmVec7Base * vcap + 32u];
    // A 值搬入用到的两个类型视图。
    constexpr uint32_t kBaseByte = kSpgemmVec7Base * vcap * sizeof(int32_t);
    const AscendC::LocalTensor<T> bAT = vec.off.template ReinterpretCast<T>() [(kBaseByte + 64u) / sizeof(T)];
    const AscendC::LocalTensor<T> bANarrow = vec.off.template ReinterpretCast<T>()[kBaseByte / sizeof(T)];


    // 落点：本行所在 8 元素块的起址（32B 对齐）。
    const AscendC::LocalTensor<int32_t> dstCol = vec.baseCol[vecSlotOff - pad];
    const AscendC::LocalTensor<float> dstVal = vec.baseDst[vecSlotOff - pad];
    // (1) B 值窗口一次搬入
    constexpr uint32_t kElem = SpgemmValueOps<T>::kDeferCast ? sizeof(T) : sizeof(float); SpgemmVecLoadBWindow<T, VEC>(vec, bValGm, minCur, total);
    // (1b) A 值搬入（DataCopyPad，与 B 窗口同一批 MTE2）。
    {
        AscendC::DataCopyExtParams cpa{
            1, static_cast<uint32_t>(DA_FIXED * kElem), 0, 0, 0};
        if constexpr (SpgemmValueOps<T>::kDeferCast) {
            AscendC::DataCopyPadExtParams<T> pada{false, 0, 0, 0}; AscendC::DataCopyPad(bANarrow, aValGm[static_cast<uint64_t>(aBase)], cpa, pada);
        } else {
            AscendC::DataCopyPadExtParams<T> pada{false, 0, 0, 0}; AscendC::DataCopyPad(bAT, aValGm[static_cast<uint64_t>(aBase)], cpa, pada);
        }
    }

    // (2) off / outCol 解析计算（无标量 UB 写）。
    const int32_t cnt = static_cast<int32_t>(total); AscendC::Muls(vec.off, mod7I, static_cast<int32_t>(segLen * sizeof(float)), cnt);
    AscendC::Add(vec.off, vec.off, lay7B, cnt); AscendC::Muls(colS, mod7I, hdStep, cnt); AscendC::Muls(vec.layerCol, lay7, stride, cnt);
    AscendC::Add(colS, colS, vec.layerCol, cnt); AscendC::Adds(colS, colS, head[0], cnt);
    // valS 预置 0，供 MulAddDst 当累加器（落在 MTE2 影子里）。
    AscendC::Duplicate(valS, 0.0f, cnt);
    // (4) 置换 + 乘 + alpha
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID6); AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID6);
    if constexpr (SpgemmValueOps<T>::kDeferCast) {
        AscendC::Cast(vec.tile, vec.narrow, AscendC::RoundMode::CAST_NONE, total);
        // A 值无损展宽。
        AscendC::Cast(bA, bANarrow, AscendC::RoundMode::CAST_NONE, DA_FIXED);
    }
    // A 值按 j%7 广播（必须在 MTE2→V 之后）。
    AscendC::Gather(vec.aTile, bA, mod7U, 0u, total); AscendC::Gather(bT, vec.tile, vec.offU, 0u, total);
    // MulAddDst: dst = src0·src1 + dst (valS 已置 0)。
    AscendC::MulAddDst(valS, bT, vec.aTile, cnt);
    if (ops.alphaRe != 1.0f) {
        AscendC::Muls(valS, valS, ops.alphaRe, cnt);
    }
    // (5) 落点：位模式 mask 的 level-0 Gather。
    uint64_t pmask[2];
    pmask[0] = (total >= 64u) ? ~static_cast<uint64_t>(0) : (((static_cast<uint64_t>(1) << total) - static_cast<uint64_t>(1)) << pad); pmask[1] = 0;
    AscendC::Gather(dstCol, colS, shiftU, 0u, pmask, static_cast<uint8_t>(1), static_cast<uint16_t>(8));
    AscendC::Gather(dstVal, valS, shiftU, 0u, pmask, static_cast<uint8_t>(1), static_cast<uint16_t>(8));
    // 复位 mask，避免后续 level-0 调用继承。
    AscendC::ResetMask(); AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID6);
}


/**
 * 轮转标量发射（段主序）的单段循环体。
 * 必须 always_inline。
 */
template <uint32_t DA_FIXED, typename T> [[gnu::always_inline]] __aicore__ inline void SpgemmMergeEmitSegMajorOne(
    const AscendC::GlobalTensor<T> &bValGm, const SpgemmValueOps<T> &ops, const SpgemmSegments &seg, const uint32_t (&rank)[DA_FIXED],
    const int32_t (&head)[DA_FIXED], uint32_t s, const AscendC::LocalTensor<int32_t> &outCol, const AscendC::LocalTensor<T> &outVal,
    const AscendC::LocalTensor<float> &wideVal, int32_t stride, uint32_t segLen)
{
    const uint32_t i = rank[s]; const uint32_t basePos = static_cast<uint32_t>(seg.cur[s]);
    // slot/col 强度削减：递推取代运行时乘法。
    uint32_t slot = i; int32_t col = head[s]; uint32_t t = 0;
    for (; t + 1 < segLen; t += 2) {
        float sumR0 = 0.0f, sumI0 = 0.0f; float sumR1 = 0.0f, sumI1 = 0.0f; float bR0 = 0.0f, bI0 = 0.0f; float bR1 = 0.0f, bI1 = 0.0f;
        ops.LoadRaw(bValGm, basePos + t, bR0, bI0); ops.LoadRaw(bValGm, basePos + t + 1, bR1, bI1);
        ops.Accumulate(seg.valRe[s], seg.valIm[s], bR0, bI0, sumR0, sumI0); ops.Accumulate(seg.valRe[s], seg.valIm[s], bR1, bI1, sumR1, sumI1);
        ops.ApplyAlpha(sumR0, sumI0); ops.ApplyAlpha(sumR1, sumI1); const uint32_t slot0 = slot; const uint32_t slot1 = slot + DA_FIXED;
        outCol.SetValue(slot0, col); outCol.SetValue(slot1, col + stride); ops.StoreValDispatch(outVal, wideVal, slot0, sumR0, sumI0);
        ops.StoreValDispatch(outVal, wideVal, slot1, sumR1, sumI1); slot += 2 * DA_FIXED; col += 2 * stride;
    }
    if (t < segLen) {
        float sumRe = 0.0f; float sumIm = 0.0f; float bRe = 0.0f; float bIm = 0.0f; ops.LoadRaw(bValGm, basePos + t, bRe, bIm);
        ops.Accumulate(seg.valRe[s], seg.valIm[s], bRe, bIm, sumRe, sumIm); ops.ApplyAlpha(sumRe, sumIm); outCol.SetValue(slot, col);
        ops.StoreValDispatch(outVal, wideVal, slot, sumRe, sumIm);
    }
}


/**
 * 层主序发射的非复数臂。预置换 ord → 内层顺序读、无依赖索引链。
 * 必须 always_inline。
 */
template <uint32_t DA_FIXED, typename T> [[gnu::always_inline]] __aicore__ inline void SpgemmMergeEmitLayerMajorReal(
    const AscendC::GlobalTensor<T> &bValGm, const SpgemmValueOps<T> &ops, const SpgemmSegments &seg, const uint32_t (&rank)[DA_FIXED],
    const int32_t (&head)[DA_FIXED], const AscendC::LocalTensor<int32_t> &outCol, const AscendC::LocalTensor<T> &outVal,
    const AscendC::LocalTensor<float> &wideVal, int32_t stride, uint32_t segLen)
{
    uint32_t ordCur[DA_FIXED]; float ordValRe[DA_FIXED]; int32_t ordFirst[DA_FIXED];
    uint32_t ord[DA_FIXED];
    for (uint32_t s = 0; s < DA_FIXED; s++) {
        ord[rank[s]] = s;
    }
    for (uint32_t i = 0; i < DA_FIXED; i++) {
        const uint32_t s = ord[i]; ordCur[i] = static_cast<uint32_t>(seg.cur[s]); ordValRe[i] = seg.valRe[s]; ordFirst[i] = head[s];
    }
    for (uint32_t t = 0; t < segLen; t++) {
        const int32_t colBase = static_cast<int32_t>(t) * stride; const uint32_t rowBase = t * DA_FIXED; uint32_t i = 0;
        for (; i + 1 < DA_FIXED; i += 2) {
            float sumR0 = 0.0f, sumI0 = 0.0f; float sumR1 = 0.0f, sumI1 = 0.0f; float bR0 = 0.0f, bI0 = 0.0f; float bR1 = 0.0f, bI1 = 0.0f;
            ops.LoadRaw(bValGm, ordCur[i] + t, bR0, bI0); ops.LoadRaw(bValGm, ordCur[i + 1] + t, bR1, bI1);
            ops.Accumulate(ordValRe[i], 0.0f, bR0, bI0, sumR0, sumI0); ops.Accumulate(ordValRe[i + 1], 0.0f, bR1, bI1, sumR1, sumI1);
            ops.ApplyAlpha(sumR0, sumI0); ops.ApplyAlpha(sumR1, sumI1); outCol.SetValue(rowBase + i, ordFirst[i] + colBase);
            outCol.SetValue(rowBase + i + 1, ordFirst[i + 1] + colBase); ops.StoreValDispatch(outVal, wideVal, rowBase + i, sumR0, sumI0);
            ops.StoreValDispatch(outVal, wideVal, rowBase + i + 1, sumR1, sumI1);
        }
        if (i < DA_FIXED) {
            float sumRe = 0.0f; float sumIm = 0.0f; float bRe = 0.0f; float bIm = 0.0f; ops.LoadRaw(bValGm, ordCur[i] + t, bRe, bIm);
            ops.Accumulate(ordValRe[i], 0.0f, bRe, bIm, sumRe, sumIm); ops.ApplyAlpha(sumRe, sumIm);
            outCol.SetValue(rowBase + i, ordFirst[i] + colBase); ops.StoreValDispatch(outVal, wideVal, rowBase + i, sumRe, sumIm);
        }
    }
}

/**
 * 层主序发射的 complex64 臂。内层循环不做预置换与强度削减（寄存器压力敏感）。
 * 必须 always_inline。
 */
template <uint32_t DA_FIXED, typename T> [[gnu::always_inline]] __aicore__ inline void SpgemmMergeEmitLayerMajorC64(
    const AscendC::GlobalTensor<T> &bValGm, const SpgemmValueOps<T> &ops, const SpgemmSegments &seg, const uint32_t (&rank)[DA_FIXED],
    const int32_t (&head)[DA_FIXED], const AscendC::LocalTensor<int32_t> &outCol, const AscendC::LocalTensor<T> &outVal,
    const AscendC::LocalTensor<float> &wideVal, int32_t stride, uint32_t segLen)
{
    uint32_t ord[DA_FIXED];
    for (uint32_t s = 0; s < DA_FIXED; s++) {
        ord[rank[s]] = s;
    }
    for (uint32_t t = 0; t < segLen; t++) {
        const int32_t colBase = static_cast<int32_t>(t) * stride; const uint32_t rowBase = t * DA_FIXED; uint32_t i = 0;
        for (; i + 1 < DA_FIXED; i += 2) {
            const uint32_t s0 = ord[i]; const uint32_t s1 = ord[i + 1]; float sumR0 = 0.0f, sumI0 = 0.0f; float sumR1 = 0.0f, sumI1 = 0.0f;
            float bR0 = 0.0f, bI0 = 0.0f; float bR1 = 0.0f, bI1 = 0.0f; ops.LoadRaw(bValGm, static_cast<uint32_t>(seg.cur[s0]) + t, bR0, bI0);
            ops.LoadRaw(bValGm, static_cast<uint32_t>(seg.cur[s1]) + t, bR1, bI1);
            ops.Accumulate(seg.valRe[s0], seg.valIm[s0], bR0, bI0, sumR0, sumI0);
            ops.Accumulate(seg.valRe[s1], seg.valIm[s1], bR1, bI1, sumR1, sumI1); ops.ApplyAlpha(sumR0, sumI0); ops.ApplyAlpha(sumR1, sumI1);
            outCol.SetValue(rowBase + i, head[s0] + colBase); outCol.SetValue(rowBase + i + 1, head[s1] + colBase);
            ops.StoreValDispatch(outVal, wideVal, rowBase + i, sumR0, sumI0); ops.StoreValDispatch(outVal, wideVal, rowBase + i + 1, sumR1, sumI1);
        }
        if (i < DA_FIXED) {
            const uint32_t s = ord[i]; float sumRe = 0.0f; float sumIm = 0.0f; float bRe = 0.0f; float bIm = 0.0f;
            ops.LoadRaw(bValGm, static_cast<uint32_t>(seg.cur[s]) + t, bRe, bIm);
            ops.Accumulate(seg.valRe[s], seg.valIm[s], bRe, bIm, sumRe, sumIm); ops.ApplyAlpha(sumRe, sumIm);
            outCol.SetValue(rowBase + i, head[s] + colBase); ops.StoreValDispatch(outVal, wideVal, rowBase + i, sumRe, sumIm);
        }
    }
}

/**
 * 轮转标量发射（层主序 + 2 路 ILP）。按类型编译期分叉为实数/复数两臂。
 * 必须 always_inline。
 */
template <uint32_t DA_FIXED, typename T> [[gnu::always_inline]] __aicore__ inline void SpgemmMergeEmitLayerMajor(
    const AscendC::GlobalTensor<T> &bValGm, const SpgemmValueOps<T> &ops, const SpgemmSegments &seg, const uint32_t (&rank)[DA_FIXED],
    const int32_t (&head)[DA_FIXED], const AscendC::LocalTensor<int32_t> &outCol, const AscendC::LocalTensor<T> &outVal,
    const AscendC::LocalTensor<float> &wideVal, int32_t stride, uint32_t segLen)
{
    if constexpr (!SpgemmValueTraits<T>::kIsComplex) {
        SpgemmMergeEmitLayerMajorReal<DA_FIXED, T>(bValGm, ops, seg, rank, head, outCol, outVal, wideVal, stride, segLen);
    } else {
        SpgemmMergeEmitLayerMajorC64<DA_FIXED, T>(bValGm, ops, seg, rank, head, outCol, outVal, wideVal, stride, segLen);
    }
}


/**
 * da 为 8 倍数、实型向量发射块的发射体。谓词已通过时发射本行。
 * 必须 always_inline。
 */
template <uint32_t DA_FIXED, typename T, typename VEC> [[gnu::always_inline]] __aicore__ inline void SpgemmMergeEmitVec8Body(
    const AscendC::GlobalTensor<T> &bValGm, const SpgemmValueOps<T> &ops,
    const AscendC::LocalTensor<int32_t> &outCol, const AscendC::LocalTensor<T> &outVal, const AscendC::LocalTensor<float> &wideVal, const VEC &vec,
    const AscendC::GlobalTensor<T> &aValGm, int32_t aBase, const int32_t (&head)[DA_FIXED], int32_t hdStep, int32_t stride, uint32_t segLen,
    uint32_t total, uint32_t minCur)
{
        // 推迟到本行才等上一个发射行的向量读（跨行 V→S 令牌协议）。
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID6);
        // 段号表 iPat8[j] = j % 8（行不变，Init 建一次）。
        const AscendC::LocalTensor<int32_t> iPat8 = vec.off[kSpgemmVec8ModIdx * kSpgemmVecEmitCap];
        // 值的落点：窄类型走 fp32 暂存，fp32 直接走 outVal。
        const AscendC::LocalTensor<float> &vdst = SpgemmVecEmitDst<T>(outVal, wideVal);
        // (1) B 值窗口一次搬入
        constexpr uint32_t kElem = SpgemmValueOps<T>::kDeferCast ? sizeof(T) : sizeof(float);
        SpgemmVecLoadBWindow<T, VEC>(vec, bValGm, minCur, total);
        // (2) off / outCol 解析计算 + A 值 DataCopyPad 搬入
        {
            AscendC::DataCopyExtParams cpa{
                1, static_cast<uint32_t>(DA_FIXED * kElem), 0, 0, 0};
            AscendC::DataCopyPadExtParams<T> pada{false, 0, 0, 0};
            if constexpr (SpgemmValueOps<T>::kDeferCast) {
                const AscendC::LocalTensor<T> aRawT = vec.off.template ReinterpretCast<T>()
                        [kSpgemmVec8ARaw * kSpgemmVecEmitCap * sizeof(int32_t) / sizeof(T)];
                AscendC::DataCopyPad(aRawT, aValGm[static_cast<uint64_t>(aBase)], cpa, pada);
            } else {
                AscendC::DataCopyPad(vec.aTile.template ReinterpretCast<T>(), aValGm[static_cast<uint64_t>(aBase)], cpa, pada);
            }
        }
        // (3) off / outCol 解析建表（取代倍增循环）。
        {
            static_assert(DA_FIXED >= 8u && (DA_FIXED & 7u) == 0u,
                          "level-0 broadcast needs da to be a 32B multiple");
            const int32_t cnt = static_cast<int32_t>(total);
            AscendC::Muls(vec.off, iPat8, static_cast<int32_t>(segLen * sizeof(float)), cnt); AscendC::Add(vec.off, vec.off, vec.layerBytes, cnt);
            AscendC::Muls(vec.layerCol, vec.layerIdx, stride, cnt); AscendC::Muls(outCol, iPat8, hdStep, cnt);
            AscendC::Add(outCol, outCol, vec.layerCol, cnt); AscendC::Adds(outCol, outCol, head[0], cnt);

        }

        // (4) 置换 + 乘 + alpha
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID6); AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID6);
        // 窄类型无损展宽到 fp32。
        if constexpr (SpgemmValueOps<T>::kDeferCast) {
            AscendC::Cast(vec.tile, vec.narrow, AscendC::RoundMode::CAST_NONE, total);
            // A 值无损展宽。
            const AscendC::LocalTensor<T> aRawT = vec.off.template ReinterpretCast<T>()
                    [kSpgemmVec8ARaw * kSpgemmVecEmitCap * sizeof(int32_t) / sizeof(T)];
            AscendC::Cast(vec.aTile, aRawT, AscendC::RoundMode::CAST_NONE, DA_FIXED);
        }
        // aTile 的 level-0 广播复制（必须在 MTE2→V 之后）。
        {
            constexpr uint16_t kBlk = DA_FIXED / 8u;
            const AscendC::UnaryRepeatParams repA{
                1, 1, static_cast<uint8_t>(kBlk), 0};
            AscendC::Muls<float>(vec.aTile, vec.aTile, 1.0f, DA_FIXED, static_cast<uint8_t>(segLen), repA);
        }
        AscendC::Gather(vdst, vec.tile, vec.offU, 0u, total); AscendC::Mul(vdst, vdst, vec.aTile, total); AscendC::Adds(vdst, vdst, 0.0f, total);
        if (ops.alphaRe != 1.0f) {
            AscendC::Muls(vdst, vdst, ops.alphaRe, total);
        }
        // 只发不等（配对 WaitFlag 在下一个发射行的块首）。
        AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID6);
}

/**
 * da 为 8 倍数、实型的向量发射块。命中则本行发射完毕返回 true，否则 false。
 * 必须 always_inline。
 */
template <uint32_t DA_FIXED, typename T, typename VEC> [[gnu::always_inline]] __aicore__ inline bool SpgemmMergeEmitVec8(
    const AscendC::GlobalTensor<T> &bValGm, const SpgemmValueOps<T> &ops, const SpgemmSegments &seg, const AscendC::LocalTensor<int32_t> &outCol,
    const AscendC::LocalTensor<T> &outVal, const AscendC::LocalTensor<float> &wideVal, const VEC &vec, uint32_t vecSlotOff,
    const AscendC::GlobalTensor<T> &aValGm, int32_t aBase, const int32_t (&head)[DA_FIXED], int32_t stride, uint32_t segLen, uint32_t total)
{
    if constexpr (VEC::kUsable && !SpgemmValueOps<T>::kIsComplex &&
                  DA_FIXED >= 8u && (DA_FIXED & 7u) == 0u) {
        if (vec.enabled && ((vecSlotOff & 7u) == 0u) && total <= vec.cap &&
            (total & 7u) == 0u && aBase >= 0) {
            // 逐段等距判据（比 min/max 跨度更强，为解析化服务）+ 段首等差判据。
            const uint32_t minCur = static_cast<uint32_t>(seg.cur[0]); const int32_t hdStep = head[1] - head[0];
            uint32_t spanBad = (hdStep > 0) ? 0u : 1u;
            {
                uint32_t prev = minCur;
                for (uint32_t s = 1; s < DA_FIXED; s++) {
                    const uint32_t c = static_cast<uint32_t>(seg.cur[s]); spanBad |= (c - prev - segLen);
                    spanBad |= static_cast<uint32_t>(head[s] - head[s - 1] - hdStep); prev = c;
                }
            }
            if (spanBad == 0u) {
                SpgemmMergeEmitVec8Body<DA_FIXED, T, VEC>(bValGm, ops, outCol, outVal,
                    wideVal, vec, aValGm, aBase, head, hdStep, stride, segLen, total, minCur);
                return true;
            }
        }
    }
    return false;
}

/**
 * complex64 向量发射块的发射体。谓词通过后的直线发射码。
 * 必须 always_inline。
 */
template <uint32_t DA_FIXED, typename T, typename VEC> [[gnu::always_inline]] __aicore__ inline void SpgemmMergeEmitVecC64Body(
    const AscendC::GlobalTensor<T> &bValGm, const SpgemmValueOps<T> &ops,
    const AscendC::LocalTensor<int32_t> &outCol, const AscendC::LocalTensor<T> &outVal,
    const VEC &vec, const AscendC::GlobalTensor<T> &aValGm, int32_t aBase,
    const int32_t (&head)[DA_FIXED], int32_t hdStep, int32_t stride, uint32_t segLen, uint32_t total, uint32_t minCur)
{
        // 跨行 V→S 令牌（同实数路径）。
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID6);
        // 视图由 vec.off 按 cap 整数倍下标派生。
        constexpr uint32_t cap = kSpgemmVecEmitCap; const AscendC::LocalTensor<float> poolF = vec.off.template ReinterpretCast<float>();
        const AscendC::LocalTensor<float> cTile = poolF[8u * cap]; const AscendC::LocalTensor<float> aIm = poolF[11u * cap];
        const AscendC::LocalTensor<float> resR = poolF[12u * cap]; const AscendC::LocalTensor<float> resI = poolF[13u * cap];
        const AscendC::LocalTensor<float> w1 = poolF[14u * cap]; const AscendC::LocalTensor<float> w2 = poolF[15u * cap];
        const AscendC::LocalTensor<uint32_t> ilvU = vec.offU[18u * cap];
        // A 行值的原始窗口与行不变广播索引表。
        const AscendC::LocalTensor<float> aRaw = poolF[5u * cap]; const AscendC::LocalTensor<uint32_t> bcastU = vec.offU[4u * cap];
        // off / outCol 的行不变基表。
        const AscendC::LocalTensor<int32_t> iPat = vec.off[3u * cap]; const AscendC::LocalTensor<int32_t> layI = vec.off[6u * cap];
        const AscendC::LocalTensor<int32_t> layB8 = vec.off[7u * cap]; const AscendC::LocalTensor<int32_t> tmpI = vec.off[10u * cap];
        const AscendC::LocalTensor<float> vdst = outVal.template ReinterpretCast<float>();
        // (1) B 值窗口搬入（float 视图）。
        AscendC::GlobalTensor<float> bValF;
        bValF.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(bValGm.GetPhyAddr(static_cast<uint64_t>(minCur))), 2u * total);
        AscendC::DataCopyExtParams cpv{
            1, static_cast<uint32_t>(2u * total * sizeof(float)), 0, 0, 0};
        AscendC::DataCopyPadExtParams<float> padv{false, 0, 0, 0}; AscendC::DataCopyPad(cTile, bValF, cpv, padv);
        // (1b) A 行值窗口（与 B 同一批 MTE2）。
        {
            AscendC::GlobalTensor<float> aRowF;
            aRowF.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(aValGm.GetPhyAddr(static_cast<uint64_t>(aBase))), 2u * DA_FIXED);
            AscendC::DataCopyExtParams cpa{
                1, static_cast<uint32_t>(2u * DA_FIXED * sizeof(float)), 0, 0, 0};
            AscendC::DataCopyPad(aRaw, aRowF, cpa, padv);
        }
        // (2) off / outCol 解析计算。
        AscendC::Muls(vec.off, iPat, static_cast<int32_t>(segLen * 2u * sizeof(float)), static_cast<int32_t>(total));
        AscendC::Add(vec.off, vec.off, layB8, static_cast<int32_t>(total)); AscendC::Muls(tmpI, layI, stride, static_cast<int32_t>(total));
        AscendC::Muls(outCol, iPat, hdStep, static_cast<int32_t>(total)); AscendC::Add(outCol, outCol, tmpI, static_cast<int32_t>(total));
        AscendC::Adds(outCol, outCol, head[0], static_cast<int32_t>(total));
        // (4) 取值 → 复乘 → sum → alpha → 交错写出
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID6); AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID6);
        // A 实/虚部去交错 + 按层广播。
        AscendC::Gather(vec.aTile, aRaw, bcastU, 0u, total); AscendC::Gather(aIm, aRaw, bcastU, static_cast<uint32_t>(sizeof(float)), total);
        AscendC::Gather(w1, cTile, vec.offU, 0u, total);  // bRe
        AscendC::Gather(w2, cTile, vec.offU, static_cast<uint32_t>(sizeof(float)),
                        total);                          // bIm
        AscendC::Mul(resR, vec.aTile, w1, total);         // aRe·bRe
        AscendC::Mul(resI, vec.aTile, w2, total);         // aRe·bIm
        AscendC::Mul(w2, aIm, w2, total);                 // aIm·bIm
        AscendC::Mul(w1, aIm, w1, total);                 // aIm·bRe
        AscendC::Sub(resR, resR, w2, total);              // pRe
        AscendC::Add(resI, resI, w1, total);              // pIm
        AscendC::Adds(resR, resR, 0.0f, total);           // sum = 0 + p
        AscendC::Adds(resI, resI, 0.0f, total);
        // alpha == (1,0) 时跳过。
        if (ops.alphaRe != 1.0f || ops.alphaIm != 0.0f) {
            AscendC::Muls(w1, resI, ops.alphaRe, total);      // C
            AscendC::Muls(w2, resR, ops.alphaIm, total);      // D
            AscendC::Muls(resR, resR, ops.alphaRe, total);    // A
            AscendC::Muls(resI, resI, ops.alphaIm, total);    // B
            AscendC::Sub(resR, resR, resI, total);            // outRe = A − B
            AscendC::Add(resI, w1, w2, total);                // outIm = C + D
        }
        AscendC::Gather(vdst, resR, ilvU, 0u, 2u * total); AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID6);
}

/**
 * complex64 的向量发射块。命中则返回 true，否则 false。
 * 必须 always_inline。
 */
template <uint32_t DA_FIXED, typename T, typename VEC> [[gnu::always_inline]] __aicore__ inline bool SpgemmMergeEmitVecC64(
    const AscendC::GlobalTensor<T> &bValGm, const SpgemmValueOps<T> &ops, const SpgemmSegments &seg, const AscendC::LocalTensor<int32_t> &outCol,
    const AscendC::LocalTensor<T> &outVal, const VEC &vec, uint32_t vecSlotOff, const AscendC::GlobalTensor<T> &aValGm, int32_t aBase,
    const int32_t (&head)[DA_FIXED], int32_t hdStep, int32_t stride, uint32_t segLen, uint32_t total)
{
    if constexpr (VEC::kUsable && SpgemmValueOps<T>::kIsComplex &&
                  DA_FIXED >= 8u && (DA_FIXED & 7u) == 0u) {
        if (vec.enabled && ((vecSlotOff & 7u) == 0u) && total <= kSpgemmVecEmitCap &&
            (total & 7u) == 0u && aBase >= 0) {
            // min/max 由建段循环顺带折出。
            const uint32_t minCur = static_cast<uint32_t>(seg.curMin); const uint32_t maxCur = static_cast<uint32_t>(seg.curMax);
            if (maxCur - minCur == (DA_FIXED - 1) * segLen) {
                SpgemmMergeEmitVecC64Body<DA_FIXED, T, VEC>(bValGm, ops, outCol, outVal,
                    vec, aValGm, aBase, head, hdStep, stride, segLen, total, minCur);
                return true;
            }
        }
    }
    return false;
}


#endif  // SPGEMM_ARCH22_MERGE_EMIT_H
