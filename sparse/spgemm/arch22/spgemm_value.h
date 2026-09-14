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
 * \file spgemm_value.h
 * \brief SpGEMM arch22 数值语义实现：dtype traits、读值提升、复乘累加、写回转换。
 *
 * 符号阶段（T1 融合路径）与数值阶段共用本文件，确保累加顺序与 Cast 时机一致，
 * 保证 bit-wise 确定性。
 *
 * 精度策略：
 *   fp32       —— fp32 累加
 *   fp16/bf16  —— fp32 中累加，末尾 Cast 回目标类型
 *   complex64  —— 实虚双路 fp32 累加，复乘按 (ar·br − ai·bi, ar·bi + ai·br) 展开
 *
 * 显式零保留：本文件不含任何 "值为零则丢弃" 的判断，结构完全由坐标模式决定。
 */

#ifndef SPGEMM_ARCH22_VALUE_H
#define SPGEMM_ARCH22_VALUE_H

#include "kernel_operator.h"

namespace {

/**
 * dtype traits：把「目标存储类型」与「是否复数」编译期固定下来。
 * 累加器一律为 fp32（复数为两路 fp32），这是精度策略的核心。
 */
template <typename T>
struct SpgemmValueTraits {
    static constexpr bool kIsComplex = false;
    static constexpr uint32_t kComponents = 1;
};

template <>
struct SpgemmValueTraits<float> {
    static constexpr bool kIsComplex = false;
    static constexpr uint32_t kComponents = 1;
};

// complex64 以「两个连续 float」表示一个元素（(re, im) 交错），与 PyTorch/cuSPARSE 一致。
struct SpgemmComplex64 {
    float re;
    float im;
};

template <>
struct SpgemmValueTraits<SpgemmComplex64> {
    static constexpr bool kIsComplex = true;
    static constexpr uint32_t kComponents = 2;
};

/**
 * 值操作集合：符号（融合路径）与数值两个 Kernel 共用。
 * 无状态，只有 alpha 作为成员；所有方法 inline。
 */
template <typename T>
struct SpgemmValueOps {
    static constexpr bool kIsComplex = SpgemmValueTraits<T>::kIsComplex;
    static constexpr uint32_t kAccComponents = kIsComplex ? 2 : 1;

    float alphaRe = 1.0f;
    float alphaIm = 0.0f;

    // half / bfloat16_t → float 的标量提升
    __aicore__ inline float ToFloat(half v) const { return static_cast<float>(v); }
    // bf16 → fp32：纯位扩展（bf16 即 fp32 的高 16 位），一次移位即可。
    // 与 AscendC::Cast(bfloat16_t) 逐位等价（含 Inf/NaN）。
    __aicore__ inline float ToFloat(bfloat16_t v) const
    {
        uint16_t bits = 0;
        *reinterpret_cast<bfloat16_t *>(&bits) = v;
        float f = 0.0f;
        *reinterpret_cast<uint32_t *>(&f) = static_cast<uint32_t>(bits) << 16;
        return f;
    }
    __aicore__ inline float ToFloat(float v) const { return v; }

    __aicore__ inline void LoadRaw(const AscendC::GlobalTensor<T> &gm, uint32_t idx,
                                   float &re, float &im) const
    {
        if constexpr (kIsComplex) {
            // complex64：(re, im) 交错存放，用 float 视图按 2×idx 读取。
            auto *p = reinterpret_cast<__gm__ float *>(gm.GetPhyAddr(idx));
            re = p[0];
            im = p[1];
        } else if constexpr (AscendC::IsSameType<T, float>::value) {
re = gm.GetValue(idx);
            im = 0.0f;
        } else {
            // fp16 / bf16：标量提升到 fp32，累加全程在 fp32 中进行。
            re = ToFloat(gm.GetValue(idx));
            im = 0.0f;
        }
    }

    /**
     * 复乘：(ar + ai·i)(br + bi·i) = (ar·br − ai·bi) + (ar·bi + ai·br)·i
     * 实数类型退化为单乘，编译期消除虚部运算。
     */
    __aicore__ inline void MulComplex(float ar, float ai, float br, float bi,
                                      float &outRe, float &outIm) const
    {
        if constexpr (kIsComplex) {
            outRe = ar * br - ai * bi;
            outIm = ar * bi + ai * br;
        } else {
            outRe = ar * br;
            outIm = 0.0f;
        }
    }

    __aicore__ inline void Accumulate(float aRe, float aIm, float bRe, float bIm,
                                      float &sumRe, float &sumIm) const
    {
        float pRe = 0.0f;
        float pIm = 0.0f;
        MulComplex(aRe, aIm, bRe, bIm, pRe, pIm);
        sumRe += pRe;
        if constexpr (kIsComplex) {
            sumIm += pIm;
        }
    }

    __aicore__ inline void ApplyAlpha(float &re, float &im) const
    {
        if constexpr (kIsComplex) {
            // alpha == (1, 0) 时直接早退：照算会让 0·Inf 产生 NaN 污染。
            // 全路径统一使用此早退，向量与标量结果一致。
            if (alphaRe == 1.0f && alphaIm == 0.0f) { return; }
        }
        float oRe = 0.0f;
        float oIm = 0.0f;
        MulComplex(alphaRe, alphaIm, re, im, oRe, oIm);
        re = oRe;
        im = oIm;
    }

    __aicore__ inline void AddSlot(const AscendC::LocalTensor<float> &acc, uint32_t slot,
                                   float re, float im) const
    {
        const uint32_t o = slot * kAccComponents;
        acc.SetValue(o, acc.GetValue(o) + re);
        if constexpr (kIsComplex) {
            acc.SetValue(o + 1, acc.GetValue(o + 1) + im);
        }
    }

    __aicore__ inline T FromFloat(float v) const
    {
        if constexpr (AscendC::IsSameType<T, half>::value) {
            return static_cast<half>(v);
        } else if constexpr (AscendC::IsSameType<T, bfloat16_t>::value) {
            // fp32 → bf16：取高 16 位 + round-half-to-even (RNE)。
            // 无分支实现，与带分支版本逐位相同。
            //   - NaN  → 0x7FFF
            //   - Inf  → 0x7F80 | sign
            //   - 其余 → u + 0x7FFF + ((u>>16)&1) 后取高 16 位 (RNE)
            uint32_t u = 0;
            *reinterpret_cast<float *>(&u) = v;
            const uint32_t exp = u & 0x7F800000u;
            const uint32_t man = u & 0x007FFFFFu;

            const uint32_t specM = 0u - ((((exp ^ 0x7F800000u) - 1u) >> 31) & 1u);
            const uint32_t nanM = 0u - (((0u - man) >> 31) & 1u);

            const uint32_t lsb = (u >> 16) & 1u;
            const uint32_t rne = (u + 0x7FFFu + lsb) >> 16;          // round-half-to-even
            const uint32_t inf = 0x7F80u | ((u >> 16) & 0x8000u);    // ±Inf 保号
            const uint32_t spec = (inf & ~nanM) | (0x7FFFu & nanM);  // NaN → 0x7FFF
            const uint32_t sel = (rne & ~specM) | (spec & specM);

            bfloat16_t r;
            *reinterpret_cast<uint16_t *>(&r) = static_cast<uint16_t>(sel);
            return r;
        } else {
            return static_cast<T>(v);
        }
    }

    // fp32 累加结果 → 目标存储类型
    __aicore__ inline void StoreVal(const AscendC::LocalTensor<T> &outVal, uint32_t slot,
                                    float re, float im) const
    {
        if constexpr (kIsComplex) {
            auto fv = outVal.template ReinterpretCast<float>();
            fv.SetValue(slot * 2, re);
            fv.SetValue(slot * 2 + 1, im);
        } else if constexpr (AscendC::IsSameType<T, float>::value) {
            outVal.SetValue(slot, re);
        } else {
            outVal.SetValue(slot, FromFloat(re));
        }
    }

    /**
     * 窄类型延迟转换写出：把 fp32 累加值原样存入暂存区，
     * 转换推迟到 flush 时用一条向量 Cast 批量完成。
     */
    __aicore__ inline void StoreValWide(const AscendC::LocalTensor<float> &wideVal,
                                        uint32_t slot, float re, float im) const
    {
        if constexpr (kIsComplex) {
            wideVal.SetValue(slot * 2, re);
            wideVal.SetValue(slot * 2 + 1, im);
        } else {
            wideVal.SetValue(slot, re);
        }
    }

    /**
     * 延迟转换对 bf16 与 fp16 都启用。
     * bf16：arch22 没有标量 cast 指令，走批量向量 Cast 显著更快。
     * fp16：static_cast<half> 虽是单条指令，但延迟到批量 Cast 同样更优。
     */
    static constexpr bool kDeferCast = AscendC::IsSameType<T, bfloat16_t>::value ||
                                       AscendC::IsSameType<T, half>::value;

    /**
     * 按类型分派写出：窄类型走 fp32 暂存（延迟到 flush 批量 Cast），
     * 其余类型仍写原类型缓冲。分派在编译期完成。
     */
    __aicore__ inline void StoreValDispatch(const AscendC::LocalTensor<T> &outVal,
                                            const AscendC::LocalTensor<float> &wideVal,
                                            uint32_t slot, float re, float im) const
    {
        if constexpr (kDeferCast) {
            StoreValWide(wideVal, slot, re, im);
        } else {
            StoreVal(outVal, slot, re, im);
        }
    }
};

}  // namespace

#endif  // SPGEMM_ARCH22_VALUE_H
