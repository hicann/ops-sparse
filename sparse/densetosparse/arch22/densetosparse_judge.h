/**
 * ----------------------------------------------------------------------------------------------------------
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software: you can redistribute it and/or modify it under the terms of conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ----------------------------------------------------------------------------------------------------------
 */

// Judge chains, bitmap builders and geometry/panel selection shared
// by the count and convert kernels (CSR/CSC/COO paths).
#ifndef DENSETOSPARSE_ARCH22_JUDGE_H_
#define DENSETOSPARSE_ARCH22_JUDGE_H_

#include "kernel_operator.h"
#include "densetosparse_kernel.h"
#include "densetosparse_kernel_shared.h"

namespace densetosparse {



// Compare/Select count-mode granularity: count * sizeof(T) must be a 256B
// multiple, so element counts are rounded up before those calls. Buffers are
// sized for the full tile, so the padded tail reads stale but in-bounds UB
// whose bits are never consumed (extraction bounds by the real length).
__aicore__ inline uint32_t CmpCount(uint32_t n, uint32_t elemBytes)
{
    const uint32_t per = 256U / (elemBytes != 0U ? elemBytes : 1U);
    return per != 0U ? (n + per - 1) / per * per : n;
}

// ReduceSum fp32 bits -> integer (|value| decode; sums of 2^k scaled bytes
// may look negative through an int8 view but the sign bit is masked off).
__aicore__ inline uint64_t DecodeFloatCount(const LocalTensor<float> &red)
{
    auto bits = red.template ReinterpretCast<uint32_t>();
    const uint32_t fbits = bits.GetValue(0);
    const uint32_t fexp = (fbits >> 23) & 0xFFU;
    if (fexp < 127U) {
        return 0;
    }
    const uint32_t sig = (fbits & 0x7FFFFFU) | 0x800000U;
    const int32_t shift = (int32_t)fexp - 127 - 23;
    return (uint64_t)(shift >= 0 ? (sig << shift) : (sig >> (-shift)));
}

// Bit offset of the target element inside its 32B window. The fixed
// coordinate is identical for every slot of a unit, so one window offset
// serves the whole gather; windows always stay inside the storage line.
__aicore__ inline uint32_t WindowOffset(uint64_t fixed, uint32_t slotElems,
    uint64_t limitDim)
{
    uint32_t k = fixed < (uint64_t)(slotElems - 1)
                     ? (uint32_t)fixed
                     : (uint32_t)(slotElems - 1);
    if (fixed + slotElems > limitDim) {
        k = (uint32_t)(fixed - (limitDim - slotElems));
    }
    return k;
}

// UB tensor bundle for the judge chains. One InitBuffer pass per kernel;
// dtype-conditional members keep the footprint lean.
class JudgeBufs {
public:
    template <typename T, bool kComplex>
    __aicore__ inline void Init(TPipe &pipe)
    {
        // indicator, zeros/ones, small constants, scratch, red, then the
        // dtype-conditional halves buffers. The order fixes every buffer's
        // UB address, and vector-op throughput is layout-sensitive.
        pipe.InitBuffer(mask1_, kD2sTileElems);
        if constexpr (kComplex) {
            pipe.InitBuffer(mask2_, kD2sTileElems);
        }
        pipe.InitBuffer(ind_, kD2sTileElems * sizeof(float));
        pipe.InitBuffer(zerosF_, kD2sTileElems * sizeof(float));
        pipe.InitBuffer(onesF_, kD2sTileElems * sizeof(float));
        pipe.InitBuffer(const1_, 512);
        pipe.InitBuffer(const3_, 512); // layout placeholder (unused)
        pipe.InitBuffer(tmpH_, 1024);
        pipe.InitBuffer(byteMask_, 1024);
        pipe.InitBuffer(red_, 32);
        if constexpr (kComplex || AscendC::IsSameType<T, int8_t>::value ||
                      AscendC::IsSameType<T, half>::value) {
            pipe.InitBuffer(zerosH_, kD2sTileElems * sizeof(half));
        }
        if constexpr (kComplex || AscendC::IsSameType<T, int8_t>::value) {
            pipe.InitBuffer(castH_, kD2sTileElems * sizeof(half));
        }
        auto zerosF = zerosF_.Get<float>();
        auto onesF = onesF_.Get<float>();
        for (uint32_t i = 0; i < kD2sTileElems; ++i) {
            zerosF.SetValue(i, 0.0f);
            onesF.SetValue(i, 1.0f);
        }
        auto c1 = const1_.Get<int8_t>();
        // Buffer is 512B and its only reader (the fp32 pad-form slot And)
        // covers <= maxSlots <= 512 bytes; writing past 512 would overrun
        // into const3_/tmpH_ (latent, masked by their write-before-read
        // usage, but still an out-of-bounds store).
        for (uint32_t i = 0; i < 512; ++i) {
            c1.SetValue(i, 1);
        }
        if constexpr (kComplex || AscendC::IsSameType<T, int8_t>::value ||
                      AscendC::IsSameType<T, half>::value) {
            auto zerosH = zerosH_.Get<half>();
            for (uint32_t i = 0; i < kD2sTileElems; ++i) {
                zerosH.SetValue(i, (half)0.0f);
            }
        }
        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline LocalTensor<uint8_t> Mask1()
    {
        return mask1_.Get<uint8_t>();
    }
    __aicore__ inline LocalTensor<uint8_t> Mask2()
    {
        return mask2_.Get<uint8_t>();
    }
    __aicore__ inline LocalTensor<float> Ind()
    {
        return ind_.Get<float>();
    }
    __aicore__ inline LocalTensor<float> ZerosF()
    {
        return zerosF_.Get<float>();
    }
    __aicore__ inline LocalTensor<float> OnesF()
    {
        return onesF_.Get<float>();
    }
    __aicore__ inline LocalTensor<int8_t> Const1()
    {
        return const1_.Get<int8_t>();
    }
    __aicore__ inline LocalTensor<int8_t> ByteMask()
    {
        return byteMask_.Get<int8_t>();
    }
    __aicore__ inline LocalTensor<half> TmpH()
    {
        return tmpH_.Get<half>();
    }
    __aicore__ inline LocalTensor<float> Red()
    {
        return red_.Get<float>();
    }
    __aicore__ inline LocalTensor<half> ZerosH()
    {
        return zerosH_.Get<half>();
    }
    __aicore__ inline LocalTensor<half> CastH()
    {
        return castH_.Get<half>();
    }

private:
    TBuf<TPosition::VECCALC> mask1_;
    TBuf<TPosition::VECCALC> mask2_;
    TBuf<TPosition::VECCALC> ind_;
    TBuf<TPosition::VECCALC> zerosF_;
    TBuf<TPosition::VECCALC> onesF_;
    TBuf<TPosition::VECCALC> const1_;
    TBuf<TPosition::VECCALC> byteMask_;
    TBuf<TPosition::VECCALC> tmpH_;
    TBuf<TPosition::VECCALC> red_;
    TBuf<TPosition::VECCALC> zerosH_;
    TBuf<TPosition::VECCALC> castH_;
    TBuf<TPosition::VECCALC> const3_;
};

// Byte-plane extraction mask for the strided window form: only the byte
// holding the target bit is non-zero, identical for every slot row.
__aicore__ inline void PrepareWindowMask(JudgeBufs &j, uint32_t slots,
    uint32_t strideBytes, uint32_t winOff)
{
    auto mv = j.ByteMask();
    const uint32_t byteOff = winOff >> 3;
    const uint32_t bitSh = winOff & 7;
    for (uint32_t i = 0; i < slots * strideBytes; ++i) {
        mv.SetValue(i, 0);
    }
    for (uint32_t i = 0; i < slots; ++i) {
        mv.SetValue(strideBytes * i + byteOff, (int8_t)(1 << bitSh));
    }
    PipeBarrier<PIPE_ALL>();
}

// ---------------------------------------------------------------------------
// Judge chains
// ---------------------------------------------------------------------------

// Count of nonzero elements in a contiguous tile of n logical elements.
template <typename T>
__aicore__ inline uint64_t CountTileComplex(JudgeBufs &j,
    const LocalTensor<T> &tile, uint32_t n)
{
    auto ind = j.Ind();
    auto zerosF = j.ZerosF();
    auto onesF = j.OnesF();
    auto red = j.Red();
    // T models complex64 as 2 interleaved fp32 (kGmStride=2).
    auto tileF = tile.template ReinterpretCast<float>();
    const uint32_t cmp = CmpCount(2U * n, sizeof(float));
    Compare(j.Mask1(), tileF, zerosF, CMPMODE::EQ, cmp);
    PipeBarrier<PIPE_V>();
    Select(ind, j.Mask1(), zerosF, onesF,
        SELMODE::VSEL_TENSOR_TENSOR_MODE, cmp);
    PipeBarrier<PIPE_V>();
    // Indicator halves pair up into one float per element; the pair is
    // zero iff both components are +-0.
    Cast<half, float>(j.CastH(), ind, RoundMode::CAST_NONE,
        CmpCount(2U * n, sizeof(half)));
    PipeBarrier<PIPE_V>();
    auto pairF = j.CastH().template ReinterpretCast<float>();
    const uint32_t cmpN = CmpCount(n, sizeof(float));
    Compare(j.Mask2(), pairF, zerosF, CMPMODE::EQ, cmpN);
    PipeBarrier<PIPE_V>();
    Select(ind, j.Mask2(), zerosF, onesF,
        SELMODE::VSEL_TENSOR_TENSOR_MODE, cmpN);
    PipeBarrier<PIPE_V>();
    ReduceSum<float>(red, ind, ind, (int32_t)n);
    PipeBarrier<PIPE_V>();
    return DecodeFloatCount(red);
}

template <typename T, bool kComplex>
__aicore__ inline uint64_t CountTile(JudgeBufs &j,
    const LocalTensor<T> &tile, uint32_t n)
{
    auto ind = j.Ind();
    auto zerosF = j.ZerosF();
    auto onesF = j.OnesF();
    auto red = j.Red();
    if constexpr (kComplex) {
        return CountTileComplex<T>(j, tile, n);
    } else if constexpr (AscendC::IsSameType<T, int8_t>::value) {
        const uint32_t cmp = CmpCount(n, sizeof(half));
        Cast<half, int8_t>(j.CastH(), tile, RoundMode::CAST_NONE, cmp);
        PipeBarrier<PIPE_V>();
        Compare(j.Mask1(), j.CastH(), j.ZerosH(), CMPMODE::EQ, cmp);
        PipeBarrier<PIPE_V>();
        Select(ind, j.Mask1(), zerosF, onesF,
            SELMODE::VSEL_TENSOR_TENSOR_MODE, cmp);
        PipeBarrier<PIPE_V>();
        ReduceSum<float>(red, ind, ind, (int32_t)n);
        PipeBarrier<PIPE_V>();
        return DecodeFloatCount(red);
    } else if constexpr (AscendC::IsSameType<T, half>::value) {
        const uint32_t cmp = CmpCount(n, sizeof(half));
        Compare(j.Mask1(), tile, j.ZerosH(), CMPMODE::EQ, cmp);
        PipeBarrier<PIPE_V>();
        Select(ind, j.Mask1(), zerosF, onesF,
            SELMODE::VSEL_TENSOR_TENSOR_MODE, cmp);
        PipeBarrier<PIPE_V>();
        ReduceSum<float>(red, ind, ind, (int32_t)n);
        PipeBarrier<PIPE_V>();
        return DecodeFloatCount(red);
    } else {
        const uint32_t cmp = CmpCount(n, sizeof(float));
        Compare(j.Mask1(), tile, zerosF, CMPMODE::EQ, cmp);
        PipeBarrier<PIPE_V>();
        Select(ind, j.Mask1(), zerosF, onesF,
            SELMODE::VSEL_TENSOR_TENSOR_MODE, cmp);
        PipeBarrier<PIPE_V>();
        ReduceSum<float>(red, ind, ind, (int32_t)n);
        PipeBarrier<PIPE_V>();
        return DecodeFloatCount(red);
    }
}

// Bitmap-only judge for the convert phase (no count): leaves the +-0 bitmap
// in Mask1() (Mask2() for complex64), one bit per logical element.
template <typename T, bool kComplex>
__aicore__ inline void BuildZeroBitmap(JudgeBufs &j,
    const LocalTensor<T> &tile, uint32_t n)
{
    auto zerosF = j.ZerosF();
    if constexpr (kComplex) {
        auto tileF = tile.template ReinterpretCast<float>();
        const uint32_t cmp = CmpCount(2U * n, sizeof(float));
        Compare(j.Mask1(), tileF, zerosF, CMPMODE::EQ, cmp);
        PipeBarrier<PIPE_V>();
        auto ind = j.Ind();
        Select(ind, j.Mask1(), zerosF, j.OnesF(),
            SELMODE::VSEL_TENSOR_TENSOR_MODE, cmp);
        PipeBarrier<PIPE_V>();
        Cast<half, float>(j.CastH(), ind, RoundMode::CAST_NONE,
            CmpCount(2U * n, sizeof(half)));
        PipeBarrier<PIPE_V>();
        auto pairF = j.CastH().template ReinterpretCast<float>();
        Compare(j.Mask2(), pairF, zerosF, CMPMODE::EQ,
            CmpCount(n, sizeof(float)));
        PipeBarrier<PIPE_V>();
    } else if constexpr (AscendC::IsSameType<T, int8_t>::value) {
        const uint32_t cmp = CmpCount(n, sizeof(half));
        Cast<half, int8_t>(j.CastH(), tile, RoundMode::CAST_NONE, cmp);
        PipeBarrier<PIPE_V>();
        Compare(j.Mask1(), j.CastH(), j.ZerosH(), CMPMODE::EQ, cmp);
        PipeBarrier<PIPE_V>();
    } else if constexpr (AscendC::IsSameType<T, half>::value) {
        Compare(j.Mask1(), tile, j.ZerosH(), CMPMODE::EQ,
            CmpCount(n, sizeof(half)));
        PipeBarrier<PIPE_V>();
    } else {
        Compare(j.Mask1(), tile, zerosF, CMPMODE::EQ,
            CmpCount(n, sizeof(float)));
        PipeBarrier<PIPE_V>();
    }
}

// complex64 slot judge: both components of the head element must be +-0
// for a zero slot (bits 0..1 of the slot byte).
template <typename T>
__aicore__ inline uint64_t CountSlotsComplex(JudgeBufs &j,
    const LocalTensor<T> &tile, uint32_t slots)
{
    constexpr uint32_t slotT = kSlotBytes / sizeof(T); // T elements per slot
    auto ind = j.Ind();
    auto zerosF = j.ZerosF();
    auto onesF = j.OnesF();
    auto red = j.Red();
    auto tileF = tile.template ReinterpretCast<float>();
    const uint32_t cmp = CmpCount(slots * slotT, sizeof(float));
    Compare(j.Mask1(), tileF, zerosF, CMPMODE::EQ, cmp);
    PipeBarrier<PIPE_V>();
    Select(ind, j.Mask1(), zerosF, onesF,
        SELMODE::VSEL_TENSOR_TENSOR_MODE, cmp);
    PipeBarrier<PIPE_V>();
    Cast<half, float>(j.CastH(), ind, RoundMode::CAST_NONE,
        CmpCount(slots * slotT, sizeof(half)));
    PipeBarrier<PIPE_V>();
    // Indicator halves: the real pair (halves 4i, 4i+1) forms float 2i;
    // pad pairs stay zero and never count.
    auto pairF = j.CastH().template ReinterpretCast<float>();
    const uint32_t pairs = slots * (slotT / 2);
    const uint32_t cmpN = CmpCount(pairs, sizeof(float));
    Compare(j.Mask2(), pairF, zerosF, CMPMODE::NE, cmpN);
    PipeBarrier<PIPE_V>();
    Select(ind, j.Mask2(), onesF, zerosF,
        SELMODE::VSEL_TENSOR_TENSOR_MODE, cmpN);
    PipeBarrier<PIPE_V>();
    ReduceSum<float>(red, ind, ind, (int32_t)pairs);
    PipeBarrier<PIPE_V>();
    return DecodeFloatCount(red);
}

// int8/half slot judge on the window form: a byte-plane AND isolates the
// target bit and the zero tally decodes from the 2^bit-scaled sum.
template <typename T>
__aicore__ inline uint64_t CountSlotsWindow(JudgeBufs &j,
    const LocalTensor<T> &tile, uint32_t slots, uint32_t winOff)
{
    constexpr uint32_t slotT = kSlotBytes / sizeof(T); // T elements per slot
    auto ind = j.Ind();
    auto zerosF = j.ZerosF();
    auto onesF = j.OnesF();
    auto red = j.Red();
    const uint32_t cmp = CmpCount(slots * slotT, sizeof(half));
    if constexpr (AscendC::IsSameType<T, int8_t>::value) {
        Cast<half, int8_t>(j.CastH(), tile, RoundMode::CAST_NONE, cmp);
        PipeBarrier<PIPE_V>();
        Compare(j.Mask1(), j.CastH(), j.ZerosH(), CMPMODE::EQ, cmp);
    } else {
        Compare(j.Mask1(), tile, j.ZerosH(), CMPMODE::EQ, cmp);
    }
    PipeBarrier<PIPE_V>();
    // Target bit sits at byte strideBytes*i + (winOff>>3) of the
    // bitmap; the in-place AND zeroes every other bit of the row.
    auto bm = j.Mask1().template ReinterpretCast<int8_t>();
    const uint32_t strideBytes = slotT / 8;
    const uint32_t bitSh = winOff & 7;
    const uint32_t bytes = slots * strideBytes;
    And<int8_t>(bm, bm, j.ByteMask(), bytes);
    PipeBarrier<PIPE_V>();
    Cast<half, int8_t>(j.TmpH(), bm, RoundMode::CAST_NONE, bytes);
    PipeBarrier<PIPE_V>();
    Cast<float, half>(ind, j.TmpH(), RoundMode::CAST_NONE, bytes);
    PipeBarrier<PIPE_V>();
    ReduceSum<float>(red, ind, ind, (int32_t)bytes);
    PipeBarrier<PIPE_V>();
    return slots - (DecodeFloatCount(red) >> bitSh);
}

// fp32 pad-form slot judge: element at the slot head, bit0 of each byte.
template <typename T>
__aicore__ inline uint64_t CountSlotsPad(JudgeBufs &j,
    const LocalTensor<T> &tile, uint32_t slots)
{
    constexpr uint32_t slotT = kSlotBytes / sizeof(T); // T elements per slot
    auto ind = j.Ind();
    auto zerosF = j.ZerosF();
    auto red = j.Red();
    const uint32_t cmp = CmpCount(slots * slotT, sizeof(float));
    Compare(j.Mask1(), tile, zerosF, CMPMODE::EQ, cmp);
    PipeBarrier<PIPE_V>();
    auto bm = j.Mask1().template ReinterpretCast<int8_t>();
    And<int8_t>(bm, bm, j.Const1(), slots);
    PipeBarrier<PIPE_V>();
    Cast<half, int8_t>(j.TmpH(), bm, RoundMode::CAST_NONE, slots);
    PipeBarrier<PIPE_V>();
    Cast<float, half>(ind, j.TmpH(), RoundMode::CAST_NONE, slots);
    PipeBarrier<PIPE_V>();
    ReduceSum<float>(red, ind, ind, (int32_t)slots);
    PipeBarrier<PIPE_V>();
    return slots - DecodeFloatCount(red);
}

// Count of nonzero slots in a strided slot tile (one element per 32B slot).
//   fp32/c64 pad form : element at slot head, EQ bit0 of each slot byte
//                       (c64: bits 0..1 = both components +-0)
//   b8/b16 window form: element at lane winOff; a byte-plane AND isolates
//                       the target bit and the zero tally is decoded from
//                       the 2^bit-scaled sum.
template <typename T, bool kComplex>
__aicore__ inline uint64_t CountSlots(JudgeBufs &j,
    const LocalTensor<T> &tile, uint32_t slots, uint32_t winOff)
{
    if constexpr (kComplex) {
        return CountSlotsComplex<T>(j, tile, slots);
    } else if constexpr (AscendC::IsSameType<T, int8_t>::value ||
                         AscendC::IsSameType<T, half>::value) {
        return CountSlotsWindow<T>(j, tile, slots, winOff);
    } else {
        return CountSlotsPad<T>(j, tile, slots);
    }
}

// b8/b16 slot walks need the window byte-plane mask staged before the
// gather; other dtypes skip this.
template <typename T, bool kComplex>
__aicore__ inline void PrepareSlotWalkMask(JudgeBufs &bufs, uint32_t maxSlots,
    uint32_t winOff)
{
    if constexpr (!kComplex &&
                  (AscendC::IsSameType<T, int8_t>::value ||
                   AscendC::IsSameType<T, half>::value)) {
        constexpr uint32_t strideBytes = (kSlotBytes / sizeof(T)) / 8;
        PrepareWindowMask(bufs, maxSlots, strideBytes, winOff);
    }
}

// Slot-level nonzero test against the bitmap left by CountSlots (used by
// convert extraction after a vectorized slot judge).
//   fp32/c64: slot byte i (c64: bits 0..1)
//   int8    : bitmap bit 32*i + winOff -> byte 4i + (winOff>>3)
//   fp16    : bitmap bit 16*i + winOff -> byte 2i + (winOff>>3)
// mask must be fetched once per tile and passed in: a TBuf::Get() per
// slot costs hundreds of nanoseconds and dominates extraction loops.
template <typename T, bool kComplex>
__aicore__ inline bool SlotNonzero(const LocalTensor<uint8_t> &mask,
    uint32_t i, uint32_t winOff)
{
    if constexpr (kComplex) {
        return (mask.GetValue(i) & 3) != 3;
    } else if constexpr (AscendC::IsSameType<T, int8_t>::value) {
        return ((mask.GetValue(4U * i + (winOff >> 3)) >> (winOff & 7)) & 1) == 0;
    } else if constexpr (AscendC::IsSameType<T, half>::value) {
        return ((mask.GetValue(2U * i + (winOff >> 3)) >> (winOff & 7)) & 1) == 0;
    } else {
        return (mask.GetValue(i) & 1) == 0;
    }
}

// Packed dense-window element test (element i at lane i; complex64 at
// Dense-row carve test (see kDenseCarveMin). int8 stays in the panels: its
// 1-byte packed blocks would serialize the MTE2 bursts.
__aicore__ inline bool DenseUnitP(uint64_t cnt, uint64_t minorSpan,
    uint32_t elemBytes)
{
    return elemBytes != 1 && cnt >= kDenseCarveMin &&
           cnt * 8 >= minorSpan;
}

// Raw slot element test straight off the tile (tail groups below the
// vectorized-slot threshold).
template <typename T, bool kComplex>
__aicore__ inline bool SlotElemNonzero(const LocalTensor<T> &tile, uint32_t i,
    uint32_t winOff)
{
    constexpr uint32_t slotT = kSlotBytes / sizeof(T);
    if constexpr (kComplex) {
        auto b = tile.template ReinterpretCast<uint32_t>();
        return (b.GetValue(slotT * i) & 0x7FFFFFFFU) != 0 ||
               (b.GetValue(slotT * i + 1) & 0x7FFFFFFFU) != 0;
    } else if constexpr (AscendC::IsSameType<T, int8_t>::value) {
        return tile.GetValue(slotT * i + winOff) != 0;
    } else if constexpr (AscendC::IsSameType<T, half>::value) {
        auto b = tile.template ReinterpretCast<uint16_t>();
        return (b.GetValue(slotT * i + winOff) & 0x7FFFU) != 0;
    } else {
        auto b = tile.template ReinterpretCast<uint32_t>();
        return (b.GetValue(slotT * i) & 0x7FFFFFFFU) != 0;
    }
}

// Raw element copy (values keep their exact bit pattern; complex64 moves the
// interleaved pair through one u64 access).
template <typename T, bool kComplex>
__aicore__ inline void CopyElem(LocalTensor<T> &dst, uint32_t dstIdx,
    const LocalTensor<T> &src, uint32_t srcIdx)
{
    if constexpr (kComplex) {
        dst.template ReinterpretCast<uint64_t>().SetValue(dstIdx,
            src.template ReinterpretCast<uint64_t>().GetValue(srcIdx));
    } else {
        dst.SetValue(dstIdx, src.GetValue(srcIdx));
    }
}

// Bitmap word scan state for contiguous extractors: bit=1 <=> +-0, words
// are consumed low-bit-first with all-ones words skipped wholesale.
struct BitmapScan {
    LocalTensor<uint32_t> words;
    uint32_t nWords;
    uint32_t word = 0;
    uint32_t pending = 0;
    uint32_t bitBase = 0;
};

__aicore__ inline void BitmapScanInit(BitmapScan &s,
    const LocalTensor<uint8_t> &mask, uint32_t len)
{
    s.words = mask.template ReinterpretCast<uint32_t>();
    s.nWords = (len + 31) >> 5;
    s.word = 0;
    s.pending = 0;
    s.bitBase = 0;
}

// Next nonzero element index in [0, len), or false when exhausted.
__aicore__ inline bool BitmapScanNext(BitmapScan &s, uint32_t len,
    uint32_t &idx)
{
    while (s.pending == 0U) {
        if (s.word >= s.nWords) {
            return false;
        }
        const uint32_t bits = s.words.GetValue(s.word);
        s.bitBase = s.word << 5;
        ++s.word;
        if (bits == 0xFFFFFFFFU) {
            continue; // all 32 elements +-0
        }
        s.pending = ~bits;
    }
    const uint32_t low = s.pending & (~(s.pending) + 1U);
    s.pending ^= low;
    const uint32_t i = s.bitBase + (uint32_t)__builtin_ctz((int)low);
    if (i >= len) {
        s.pending = 0U;
        return false;
    }
    idx = i;
    return true;
}

} // namespace densetosparse

#endif // DENSETOSPARSE_ARCH22_JUDGE_H_
