/**
 * ----------------------------------------------------------------------------------------------------------
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms of conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You should not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT OF MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ----------------------------------------------------------------------------------------------------------
 */

/*
 * Blocked-ELL convert kernel (dedicated translation unit). BELL performs no
 * structure discovery: tasks extract fixed blocks selected by the preset
 * block-column pattern, writing dtype-positive zeros for logically
 * out-of-range tail elements. Keeping this unit separate from the
 * CSR/CSC/COO translation unit isolates its transpose vector chains from
 * the code generation of the main paths (and vice versa).
 */

#include "kernel_operator.h"
#include "densetosparse_kernel.h"
#include "densetosparse_kernel_shared.h"

using namespace densetosparse;

namespace {

// Row-storage order alias (dense operand layout).
constexpr uint32_t kOrderRow = kD2sOrderRow;
constexpr uint32_t kGatherChunk = 1024; // u16 lanes per gather slice

__aicore__ inline void BuildIota(LocalTensor<int32_t> t, uint32_t n) {
    const uint32_t seed = (64 < n) ? 64 : n;
    for (uint32_t i = 0; i < seed; ++i) {
        t.SetValue(i, static_cast<int32_t>(i));
    }
    SetFlag<HardEvent::S_V>(EVENT_ID0);
    WaitFlag<HardEvent::S_V>(EVENT_ID0);
    uint32_t len = seed;
    while (len < n) {
        const uint32_t half = (n - len < len) ? (n - len) : len;
        Adds(t[len], t, 0, half);
        Adds(t[len], t[len], static_cast<int32_t>(len), half);
        len += half;
    }
}

// Derived staging geometry and table layout of one launch (everything
// the per-task helpers need; computed once per core).
struct BellGeo {
    uint64_t blockRows;
    uint64_t slots;
    uint32_t b;
    uint32_t e;
    uint32_t be; // b * b
    uint32_t blockBytes;
    uint32_t tBegin;
    uint32_t tEnd;
    uint32_t colBlocks;
    uint32_t colPitch;
    uint32_t stageW;
    uint32_t rowPitch;
    uint32_t stageSlot;
    uint32_t s; // staged u16 words per ROW slot
    uint32_t trSlot;
    uint32_t lb; // log2(b)
    uint32_t wordsOffA;
    uint32_t wordsOffA1;
    uint32_t tblNoAK;
    uint32_t packWords;
    uint32_t iotaNeed;
    uint32_t tblWords;
    uint32_t ringT; // task-ring depth
    int32_t base;
    bool active;
    bool rowLayout;
    bool needAlign;
    bool use2d;
    bool pow2;
    bool rowCompact;
    bool ringActive;
};

// Staging geometry (laws 29/30): stageW*E, (ld-stageW)*E and the src
// base must be 32B multiples. b*E%32!=0 only for (b=16,E=1).
__aicore__ inline void MakeBellStageLayout(BellGeo &g,
    const DenseToSparseTilingData &tiling) {
    g.needAlign = (g.b * g.e) % 32u != 0u;
    g.stageW = g.b;
    g.use2d = false;
    if (g.rowLayout && ((tiling.ld * g.e) & 31u) == 0) {
        // Guard: the rounded width must not exceed ld (small blocks on
        // narrow matrices: b=2 E=1 ld=20 -> stageW=32 > ld -> negative gap
        // -> u32 wrap -> MTE2 fault).
        const uint32_t rounded = (g.stageW * g.e + 31u) / 32u * 32u / g.e;
        if (rounded <= static_cast<uint32_t>(tiling.ld)) {
            g.stageW = rounded;
            g.use2d = true;
        }
    }
    const uint32_t rowStageBytes =
        g.use2d ? g.stageW * g.e : g.b * (g.e == 1 ? 2 : g.e);
    // E=8 ROW: pad the staged row pitch by 32B (512 -> 544B). The
    // gather reads walk ir*512B -- a power-of-two stride that aliases
    // UB banks; the odd pitch breaks the aliasing.
    g.rowPitch = (g.rowLayout && g.e == 8 && g.use2d) ? rowStageBytes + 32u
                                                      : rowStageBytes;
    g.stageSlot = g.rowLayout ? g.b * g.rowPitch : 0;
    g.s = g.stageSlot / 2; // staged u16 words per ROW slot
    g.trSlot = (!g.rowLayout && g.e == 1) ? g.be * 2 : g.blockBytes;
}

// tbl u32 word layout: offA (BE; E=8 ROW uses a column-half table,
// right half via (b/2)*E srcBase shift) | offA1 (E=1 needAlign) |
// pack1/pack2 (E=1, T*BE/2 each) | iota | scratch. Also fixes the
// task-ring depth T from the UB budget.
__aicore__ inline void MakeBellTableLayout(BellGeo &g) {
    g.wordsOffA = g.be;
    g.wordsOffA1 = g.needAlign ? g.be : 0;
    g.tblNoAK = g.wordsOffA + g.wordsOffA1 + 16;
    // Compact offset chains (ROW, E >= 2, b <= half a chunk): the iota
    // ramp only covers one kGatherChunk slice; per-chunk base constants
    // fold the chunk offset o (a multiple of 2^10, so ir and the E=8
    // half-parity depend only on c, and ic splits as ic_c + (o >> lb)).
    // E=8 needs three 1024-word temps, E=2/4 two.
    g.rowCompact = g.rowLayout && g.e >= 2 && g.b <= kGatherChunk / 2;
    uint32_t iotaNeedUB = g.rowCompact
        ? (g.be < kGatherChunk ? g.be : kGatherChunk)
        : g.be;
    iotaNeedUB = (iotaNeedUB + 7u) & ~7u; // 32B-aligned temp base
    const uint32_t tailWords =
        g.rowCompact ? (g.e == 8 ? 3 * 1024 : 2 * 1024) : 7 * 1024;

    const uint32_t perTask = (g.rowLayout ? g.stageSlot : 0) + g.trSlot;
    // E=1 per-task: w planes (2S) + scr1 (BE) + scrSh (BE) + pk2 (BE/2)
    // u16 scratch and 2 pack tables of BE/2 u32 words each.
    const uint32_t perTaskAll =
        perTask + (g.e == 1 ? (3 * g.s + 2 * g.be + g.be / 2) * 2 + g.be * 4 +
                                 (g.be / 2) * 4
                           : 0); // + iota growth (T*(BE/2) words)
    // E=1's iota scales with T and is charged per-task instead.
    const uint32_t fixedE1 = g.e == 1 ? (2 * g.s + g.be + 64) * 2 : 0;
    // Compact mode shrinks the static tables enough that the real
    // (~192KB) unified buffer allows a wider budget than the
    // conservative 128KB default (crash floor measured >= 206KB).
    const uint64_t ubBudget = g.e == 8 && g.rowLayout ? 184ull * 1024ull
                                                      : 128ull * 1024ull;
    const uint64_t fixedAll = fixedE1 + 1024 +
        (static_cast<uint64_t>(g.tblNoAK) + iotaNeedUB + tailWords) * 4;
    g.ringT = 1;
    while (perTaskAll * (g.ringT + 1) + fixedAll <= ubBudget && g.ringT < 128u) {
        ++g.ringT;
    }
    g.packWords = g.e == 1 ? g.ringT * (g.be / 2) : 0;
    g.iotaNeed = g.rowCompact ? iotaNeedUB : g.be;
    if (g.e == 1 && g.ringT * (g.be / 2) > g.iotaNeed) {
        g.iotaNeed = g.ringT * (g.be / 2); // in the T-dependent region
    }
    g.tblWords = g.tblNoAK + 2 * g.packWords + g.iotaNeed + tailWords;
}

__aicore__ inline BellGeo MakeBellGeo(const DenseToSparseTilingData &tiling,
    uint32_t core) {
    BellGeo g{};
    const uint64_t bU = tiling.ellBlockSize;
    // Ceil block grid: the tail block row/column covers fewer than b
    // elements; its out-of-range outputs stay dtype-positive zeros.
    g.blockRows = (tiling.rows + bU - 1) / bU;
    g.slots = (tiling.ellCols + bU - 1) / bU;
    const uint64_t tasks64 = g.blockRows * g.slots;
    // Host validation caps the task grid at the uint32 partition (see
    // ValidateBellTaskGrid); the bound here is a safety net against a
    // host-side regression silently truncating the partition.
    g.active = tasks64 != 0 && tasks64 <= 0xFFFFFFFFULL;
    if (!g.active) {
        return g;
    }
    g.b = static_cast<uint32_t>(bU);
    g.e = tiling.elementBytes;
    g.be = g.b * g.b;
    g.blockBytes = g.be * g.e;
    const uint32_t cores = tiling.numBlocks;
    const uint32_t tasks = static_cast<uint32_t>(tasks64);
    const uint32_t perCore = DivUp(tasks, cores);
    g.tBegin = core * perCore;
    g.tEnd = (g.tBegin + perCore) < tasks ? (g.tBegin + perCore) : tasks;
    g.active = g.tBegin < g.tEnd;
    if (!g.active) {
        return g;
    }
    g.base = static_cast<int32_t>(tiling.base);
    g.rowLayout = tiling.order == kOrderRow;
    g.colBlocks = static_cast<uint32_t>((tiling.cols + g.b - 1U) / g.b);
    g.colPitch = g.b * (g.e == 1 ? 2 : g.e);

    MakeBellStageLayout(g, tiling);
    MakeBellTableLayout(g);
    g.lb = 0;
    for (uint32_t v = g.b; v > 1u; v >>= 1u) {
        ++g.lb;
    }
    g.pow2 = (g.b & (g.b - 1u)) == 0u;
    g.ringActive = g.rowLayout && g.e != 1 && g.ringT >= 2u && g.use2d &&
                   g.b >= 64u && (g.tEnd - g.tBegin) <= 512u;
    return g;
}

// GM windows of one launch.
struct BellGm {
    GlobalTensor<uint8_t> denseGm;
    GlobalTensor<int32_t> patGm;
    GlobalTensor<uint8_t> patGm8; // byte view of the pattern buffer
    GlobalTensor<uint8_t> valGm;
};

__aicore__ inline BellGm MakeBellGm(const DenseToSparseTilingData &tiling,
    GM_ADDR dense, GM_ADDR ellColInd, GM_ADDR values, const BellGeo &g) {
    BellGm gm;
    gm.denseGm.SetGlobalBuffer((__gm__ uint8_t *)dense,
                                tiling.rows * tiling.ld * g.e);
    gm.patGm.SetGlobalBuffer((__gm__ int32_t *)ellColInd,
                             static_cast<uint64_t>(g.blockRows) * g.slots);
    gm.patGm8.SetGlobalBuffer((__gm__ uint8_t *)ellColInd,
                              static_cast<uint64_t>(g.blockRows) * g.slots *
                                  4);
    gm.valGm.SetGlobalBuffer(
        (__gm__ uint8_t *)values,
        g.blockRows * static_cast<uint64_t>(tiling.ellBlockSize) * g.slots *
            static_cast<uint64_t>(tiling.ellBlockSize) * g.e);
    return gm;
}

// Stage one task's b x b block into its ROW stage slot at byte offset
// slotOff: tail tasks pre-zero the slot and load only the valid
// sub-rectangle (right-padded to 32B); plain tasks take the single 2D
// DMA. Invalid (-1 / out-of-range) patterns leave the slot untouched.
__aicore__ inline void BellStageTask(const BellGeo &g,
    const DenseToSparseTilingData &t, const BellGm &gm,
    AscendC::TBuf<AscendC::TPosition::VECCALC> &stageBuf, int32_t enc,
    uint32_t rowBase, uint32_t slotOff) {
    if (enc < g.base ||
        static_cast<uint32_t>(enc - g.base) >= g.colBlocks) {
        return;
    }
    const uint32_t blockCol = static_cast<uint32_t>(enc - g.base);
    const uint32_t rb = static_cast<uint32_t>(
        t.rows - rowBase < g.b ? t.rows - rowBase : g.b);
    const uint32_t cb = static_cast<uint32_t>(
        t.cols - blockCol * g.b < g.b ? t.cols - blockCol * g.b : g.b);
    if (rb < g.b || cb < g.b) {
        for (uint32_t o = 0; o < g.stageSlot / 2; o += kVecChunk) {
            const uint32_t len = (g.stageSlot / 2 - o) < kVecChunk
                                     ? (g.stageSlot / 2 - o)
                                     : kVecChunk;
            Duplicate(stageBuf.Get<uint16_t>()[slotOff / 2 + o],
                      static_cast<uint16_t>(0), len);
        }
        PipeBarrier<PIPE_V>();
        // Sub-32B loads pad their UB writes to 32B with undefined bytes;
        // the right-pad form keeps the write an exact 32B multiple of
        // zeros instead.
        const uint32_t rowLoadElems = (cb * g.e + 31u) / 32u * 32u / g.e;
        const auto rowPad = static_cast<uint8_t>(rowLoadElems - cb);
        for (uint32_t r = 0; r < rb; ++r) {
            DataCopyPad(
                stageBuf.Get<uint8_t>()[slotOff + r * g.rowPitch],
                gm.denseGm[(static_cast<uint64_t>(rowBase + r) * t.ld +
                            blockCol * g.b) * g.e],
                {1, static_cast<uint16_t>(cb * g.e), 0, 0},
                {true, 0, rowPad, 0});
        }
        return;
    }
    uint32_t wcol = blockCol * g.b;
    if (g.use2d && g.needAlign) {
        wcol &= ~(g.stageW - 1u);
        if (wcol + g.stageW > t.ld) {
            wcol = static_cast<uint32_t>(t.ld) - g.stageW;
        }
    }
    DataCopyPad(
        stageBuf.Get<uint8_t>()[slotOff],
        gm.denseGm[(static_cast<uint64_t>(rowBase) * t.ld + wcol) * g.e],
        {static_cast<uint16_t>(g.b), g.stageW * g.e,
         static_cast<uint32_t>((t.ld - g.stageW) * g.e),
         (g.rowPitch - g.stageW * g.e) / 32u, 0},
        {false, 0, 0, 0});
}

// Transpose gather of one staged ROW task into its tr slot. E=8 uses the
// column-half table: pass L covers ic < b/2 (j < BE), pass R reuses it
// with a uniform (b/2)*E srcBase shift; dst words [BE, 2BE) are the
// ic >= b/2 half. Fewer, larger gather calls (repeatTime stays <= 255
// 32-bit reps = 16320 lanes): b=64 E=8 drops from 16 calls to 2 per
// task, trimming vector issue gaps.
__aicore__ inline void GatherTranspose(const BellGeo &g,
    AscendC::TBuf<AscendC::TPosition::VECCALC> &stageBuf,
    AscendC::TBuf<AscendC::TPosition::VECCALC> &trBuf,
    const LocalTensor<uint32_t> &offA, uint32_t sOff, uint32_t tOff) {
    auto stage8 = stageBuf.Get<uint8_t>()[sOff];
    auto tr16 = trBuf.Get<uint8_t>()[tOff].ReinterpretCast<uint16_t>();
    auto tr32 = trBuf.Get<uint8_t>()[tOff].ReinterpretCast<uint32_t>();
    const uint32_t gatherLen = g.be;
    const uint32_t gch = g.be < 8192u ? g.be : 8192u;
    for (uint32_t o = 0; o < gatherLen; o += gch) {
        const uint32_t len = (gatherLen - o) < gch ? (gatherLen - o) : gch;
        if (g.e == 8) {
            auto src32 = stage8.ReinterpretCast<uint32_t>();
            const uint32_t halfW = (g.b / 2) * g.e / 4;
            Gather(tr32[o], src32, offA[o], 0, len);
            Gather(tr32[g.be + o], src32[halfW], offA[o], 0, len);
        } else if (g.e == 2) {
            Gather(tr16[o], stage8.ReinterpretCast<uint16_t>(),
                   offA[o], 0, len);
        } else {
            Gather(tr32[o], stage8.ReinterpretCast<uint32_t>(),
                   offA[o], 0, len);
        }
    }
}

// ---- Offset tables. Vector build when b is a power of two (always:
// 16/32/64), scalar fallback otherwise. Offsets are BYTE offsets. ----

// E=8 ROW chunk: offA[j] = ((e&(b-1))*w + (e>>lb))*8 + 4*(j&1),
// e = j>>1. Compact ramp: iota holds only c = j - o for the chunk base
// o (multiple of 2^10); ir and j&1 depend only on c, and (j>>lb+1) =
// (c>>lb+1) + (o>>lb+1).
__aicore__ inline void BuildOffAChunkE8(const BellGeo &g,
    const LocalTensor<int32_t> &iota, const LocalTensor<int32_t> &offAI,
    const LocalTensor<int32_t> &tA, const LocalTensor<int32_t> &tB,
    const LocalTensor<int32_t> &tC, uint32_t o, uint32_t len) {
    ShiftRight(tA, iota, 1, len);
    ShiftRight(tB, tA, static_cast<int32_t>(g.lb), len);
    ShiftLeft(tB, tB, static_cast<int32_t>(g.lb), len);
    Sub(tA, tA, tB, len);                       // e&(b-1)
    Muls(tA, tA, static_cast<int32_t>(g.rowPitch), len);
    ShiftRight(tB, iota, static_cast<int32_t>(g.lb + 1), len);
    Adds(tB, tB, static_cast<int32_t>(o >> (g.lb + 1)), len);
    Muls(tB, tB, 8, len);                       // ic*8
    Add(offAI[o], tA, tB, len);
    ShiftRight(tC, iota, 1, len);
    ShiftLeft(tC, tC, 1, len);
    Sub(tC, iota, tC, len);                     // c&1 == j&1
    Muls(tC, tC, 4, len);
    Add(offAI[o], offAI[o], tC, len);
}

// COL E=1: offA[m] = (2m/b)*2b + (2m % b), m < BE/2.
__aicore__ inline void BuildOffAColE1(const BellGeo &g,
    const LocalTensor<int32_t> &iota,
    const LocalTensor<int32_t> &offAI) {
    auto tA = iota[g.iotaNeed];
    auto tB = iota[g.iotaNeed + 1024];
    auto tC = iota[g.iotaNeed + 2048];
    auto tD = iota[g.iotaNeed + 4096];
    const uint32_t CH = 1024;
    const uint32_t half = g.be / 2;
    for (uint32_t o = 0; o < half; o += CH) {
        const uint32_t len = (half - o) < CH ? (half - o) : CH;
        ShiftLeft(tA, iota[o], 1, len);             // 2m
        ShiftRight(tB, tA, static_cast<int32_t>(g.lb), len);
        Muls(tB, tB, static_cast<int32_t>(2 * g.b), len);
        ShiftRight(tC, tA, static_cast<int32_t>(g.lb), len);
        ShiftLeft(tC, tC, static_cast<int32_t>(g.lb), len);
        Sub(tD, tA, tC, len);                       // 2m % b
        Add(offAI[o], tB, tD, len);
    }
}

// needAlign (E=1, b=16): right-half table at a stageW/2 byte shift.
__aicore__ inline void BuildOffA1Align(const BellGeo &g,
    const LocalTensor<int32_t> &offAI,
    const LocalTensor<uint32_t> &offA1T) {
    const uint32_t CH = 1024;
    for (uint32_t o = 0; o < g.be / 2; o += CH) {
        const uint32_t len = (g.be / 2 - o) < CH ? (g.be / 2 - o) : CH;
        Adds(offA1T.ReinterpretCast<int32_t>()[o], offAI[o],
             static_cast<int32_t>(g.stageW / 2), len);
    }
}

// Chunked vector build (law 28: slice bases must be 32B aligned; all
// intermediates live in three fixed 1024-word aligned vectors).
__aicore__ inline void BuildOffAVec(const BellGeo &g,
    AscendC::TBuf<AscendC::TPosition::VECCALC> &tblBuf,
    const LocalTensor<int32_t> &iota) {
    auto offA = tblBuf.Get<uint32_t>();
    auto offAI = offA.ReinterpretCast<int32_t>();
    auto offA1T = tblBuf.Get<uint32_t>()[g.wordsOffA];
    auto tA = iota[g.iotaNeed];              // aligned 1024-word scratch
    auto tB = iota[g.iotaNeed + 1024];
    auto tC = iota[g.iotaNeed + 2048];
    auto tC2 = iota[g.iotaNeed + 3072];
    auto tD = iota[g.iotaNeed + 4096];
    auto tE = iota[g.iotaNeed + 5120];
    auto tF = iota[g.iotaNeed + 6144];
    const uint32_t CH = 1024;
    const uint32_t w = g.use2d ? g.stageW : g.b;
    const uint32_t gatherLen = g.be; // E=8 ROW: column-half table
    for (uint32_t o = 0; o < gatherLen; o += CH) {
        const uint32_t len = (gatherLen - o) < CH ? (gatherLen - o) : CH;
        if (g.e == 8 && g.rowLayout) {
            BuildOffAChunkE8(g, iota, offAI, tA, tB, tC, o, len);
        } else if (g.rowLayout && (g.e == 2 || g.e == 4)) {
            // ir = c&(b-1); ic = (o>>lb) + (c>>lb)
            ShiftRight(tA, iota, static_cast<int32_t>(g.lb), len);
            ShiftLeft(tA, tA, static_cast<int32_t>(g.lb), len);
            Sub(tA, iota, tA, len);                     // ir
            Muls(tA, tA, static_cast<int32_t>(w * g.e), len);
            ShiftRight(tB, iota, static_cast<int32_t>(g.lb), len);
            Adds(tB, tB, static_cast<int32_t>(o >> g.lb), len);
            Muls(tB, tB, static_cast<int32_t>(g.e), len);
            Add(offAI[o], tA, tB, len);
        } else if (g.rowLayout) {
            // E=1 planes, reordered pair index n = k*b + ir
            // (k = ic/2): offA[n] = ir*rs + 2k BYTES into the
            // task's plane-pool slot (plane chosen by gather src).
            ShiftRight(tD, iota[o], static_cast<int32_t>(g.lb), len);
            ShiftLeft(tD, tD, static_cast<int32_t>(g.lb), len);
            Sub(tE, iota[o], tD, len);                  // ir -> tE
            Muls(tE, tE, static_cast<int32_t>(g.rowPitch), len);
            ShiftRight(tD, iota[o], static_cast<int32_t>(g.lb), len);
            Muls(tD, tD, 2, len);                       // 2k
            Add(offAI[o], tE, tD, len);
        }
    }
    if (!g.rowLayout && g.e == 1) {
        BuildOffAColE1(g, iota, offAI);
    }
    if (g.needAlign) {
        BuildOffA1Align(g, offAI, offA1T);
    }
}

// m00 byte-pair mask pool (0x00FF) over the whole staged plane region.
__aicore__ inline void FillM00Pool(const BellGeo &g,
    AscendC::TBuf<AscendC::TPosition::VECCALC> &scrBuf) {
    auto m00 = scrBuf.Get<uint16_t>();
    const uint32_t mlen = g.ringT * g.s;
    for (uint32_t o = 0; o < mlen; o += 256) {
        const uint32_t n = (mlen - o) < 256 ? (mlen - o) : 256;
        Duplicate(m00[o], static_cast<uint16_t>(0x00FF), n);
    }
}

// Scalar fallback (non-pow2 b).
__aicore__ inline void BuildTablesScalar(const BellGeo &g,
    AscendC::TBuf<AscendC::TPosition::VECCALC> &tblBuf,
    AscendC::TBuf<AscendC::TPosition::VECCALC> &scrBuf) {
    auto offA = tblBuf.Get<uint32_t>();
    auto offA1T = tblBuf.Get<uint32_t>()[g.wordsOffA];
    const uint32_t w = g.use2d ? g.stageW : g.b;
    if (g.rowLayout) {
        for (uint32_t j = 0; j < g.be; ++j) {
            const uint32_t e = g.e == 8 ? (j >> 1) : j;
            const uint32_t ir = e % g.b;
            const uint32_t ic = e / g.b;
            if (g.e == 8) {
                // column-half table (ic < b/2 by construction)
                offA.SetValue(j, (ir * w + ic) * 8 + 4 * (j & 1u));
            } else if (g.e == 2 || g.e == 4) {
                offA.SetValue(j, (ir * w + ic) * g.e);
            } else {
                offA.SetValue(j, ir * g.rowPitch + (ic & ~1u) +
                                     2 * g.s * (ic & 1u));
            }
        }
        if (g.needAlign) {
            for (uint32_t j = 0; j < g.be; ++j) {
                offA1T.SetValue(j, offA.GetValue(j) + g.stageW / 2);
            }
        }
    } else if (g.e == 1) {
        for (uint32_t m = 0; m < g.be / 2; ++m) {
            offA.SetValue(m, (2 * m / g.b) * 2 * g.b + (2 * m % g.b));
        }
    }
    if (g.e == 1) {
        auto pack1 = tblBuf.Get<uint32_t>()[g.wordsOffA + g.wordsOffA1];
        auto pack2 =
            tblBuf.Get<uint32_t>()[g.wordsOffA + g.wordsOffA1 + g.packWords];
        for (uint32_t m = 0; m < g.ringT * (g.be / 2); ++m) {
            const uint32_t q = g.be / 2;
            pack1.SetValue(m, (m / q) * g.be + 2 * (m % q));
            pack2.SetValue(m, (m / q) * g.be + 2 * (m % q) + 2);
        }
        auto m00 = scrBuf.Get<uint16_t>();
        for (uint32_t i = 0; i < g.ringT * g.s; ++i) {
            m00.SetValue(i, 0x00FF);
        }
    }
}

__aicore__ inline void BuildPackPoolVec(const BellGeo &g,
    AscendC::TBuf<AscendC::TPosition::VECCALC> &tblBuf,
    AscendC::TBuf<AscendC::TPosition::VECCALC> &scrBuf,
    const LocalTensor<int32_t> &iota) {
    auto offA = tblBuf.Get<uint32_t>();
    auto offAI = offA.ReinterpretCast<int32_t>();
    auto tA = iota[g.iotaNeed];
    auto tB = iota[g.iotaNeed + 1024];
    auto tC = iota[g.iotaNeed + 2048];
    auto tC2 = iota[g.iotaNeed + 3072];
    auto tD = iota[g.iotaNeed + 4096];
    auto tE = iota[g.iotaNeed + 5120];
    auto tF = iota[g.iotaNeed + 6144];
    // q = b*b/2 = 2^(2lb-1) on the pow2 path, so log2(q) = 2*lb-1.
    const uint32_t lq = 2 * g.lb - 1;
    const uint32_t q = g.be / 2;
    const uint32_t pool = g.ringT * q;
    const uint32_t CH = 1024;
    for (uint32_t o = 0; o < pool; o += CH) {
        const uint32_t len = (pool - o) < CH ? (pool - o) : CH;
        // m = m' mod q
        ShiftRight(tA, iota[o], static_cast<int32_t>(lq), len);
        ShiftLeft(tA, tA, static_cast<int32_t>(lq), len);
        Sub(tD, iota[o], tA, len);                  // m -> tD
        ShiftLeft(tB, tD, 1, len);                  // 2m
        ShiftRight(tC, tB, static_cast<int32_t>(g.lb), len); // ic
        ShiftRight(tA, tB, static_cast<int32_t>(g.lb), len);
        ShiftLeft(tA, tA, static_cast<int32_t>(g.lb), len);
        Sub(tE, tB, tA, len);                       // ir -> tE
        ShiftRight(tB, tC, 1, len);
        ShiftLeft(tB, tB, static_cast<int32_t>(g.lb), len); // (ic/2)*b
        ShiftRight(tC2, tC, 1, len);
        ShiftLeft(tC2, tC2, 1, len);
        Sub(tF, tC, tC2, len);                      // ic&1 -> tF
        Muls(tF, tF, static_cast<int32_t>(g.be / 2), len);
        Add(tE, tE, tB, len);
        Add(tE, tE, tF, len);                       // r
        ShiftLeft(tE, tE, 1, len);                  // 2r bytes
        ShiftRight(tB, iota[o], static_cast<int32_t>(lq), len);
        Muls(tB, tB, static_cast<int32_t>(2 * g.be), len);
        Add(tblBuf.Get<uint32_t>()[g.wordsOffA + g.wordsOffA1]
                .ReinterpretCast<int32_t>()[o],
            tE, tB, len);
        Adds(tblBuf.Get<uint32_t>()
                 [g.wordsOffA + g.wordsOffA1 + g.packWords]
                     .ReinterpretCast<int32_t>()[o],
             tblBuf.Get<uint32_t>()[g.wordsOffA + g.wordsOffA1]
                 .ReinterpretCast<int32_t>()[o],
             2, len);
    }
    FillM00Pool(g, scrBuf);
}

__aicore__ inline void BuildOffsetTables(const BellGeo &g,
    AscendC::TBuf<AscendC::TPosition::VECCALC> &tblBuf,
    AscendC::TBuf<AscendC::TPosition::VECCALC> &scrBuf) {
    auto iota = tblBuf.Get<int32_t>()[g.wordsOffA + g.wordsOffA1 +
                                      2 * g.packWords];
    if (g.pow2) {
        BuildIota(iota, g.iotaNeed);
        BuildOffAVec(g, tblBuf, iota);
        if (g.e == 1) {
            BuildPackPoolVec(g, tblBuf, scrBuf, iota);
        }
    } else {
        BuildTablesScalar(g, tblBuf, scrBuf);
    }
    SetFlag<HardEvent::S_V>(EVENT_ID0);
    WaitFlag<HardEvent::S_V>(EVENT_ID0);
}

// E=1 pack-pool tables (vector build, pow2 b): pool offsets over T*q
// entries (q = BE/2): r(m) = (ic/2)*b + ir + (ic&1)*(BE/2), ic=2m/b,
// ir=2m%b; p1 = 2*r (bytes in the scr1 slot), p2 = p1+2, slot stride
// 2*BE bytes. m00 pool: chunked Duplicate fill (correct by
// construction; the previous doubling Copy({1,1,8,8}) from the first
// 256 entries left stride-unit-dependent holes in the tail of the pool:
// the byte-pair expansion mask read garbage/0xFF00 there, zeroing the
// LOW byte (even ic, ir >= half-tile rows) of extracted int8 pairs on
// larger block grids).
// ---- Small-block scalar path (b < 16): vector staging slots fall
// below the 32B granularity. Each task builds its b x b block in UB
// (zero-initialized, valid sub-rectangle loaded row by row) and flushes
// with a single MTE3. Handles tails and -1 slots uniformly. ----
__aicore__ inline void SmallBlockFillRows(const BellGeo &g,
    const DenseToSparseTilingData &t, const BellGm &gm,
    const LocalTensor<uint8_t> &rowStage, const LocalTensor<uint8_t> &scr,
    uint32_t rowBase, uint32_t blockCol, uint32_t rb, uint32_t cb) {
    for (uint32_t ir = 0; ir < rb; ++ir) {
        DataCopyPad(
            rowStage,
            gm.denseGm[(static_cast<uint64_t>(rowBase + ir) * t.ld +
                        blockCol * g.b) * g.e],
            {1, cb * g.e, 0, 0, 0}, {false, 0, 0, 0});
        SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
        for (uint32_t ic = 0; ic < cb; ++ic) {
            for (uint32_t e = 0; e < g.e; ++e) {
                scr.SetValue((ic * g.b + ir) * g.e + e,
                             rowStage.GetValue(ic * g.e + e));
            }
        }
    }
}

// COL storage: element (r, c) lives at c*ld + r; walk block columns and
// load each column's run of rows.
__aicore__ inline void SmallBlockFillCols(const BellGeo &g,
    const DenseToSparseTilingData &t, const BellGm &gm,
    const LocalTensor<uint8_t> &rowStage, const LocalTensor<uint8_t> &scr,
    uint32_t rowBase, uint32_t blockCol, uint32_t rb, uint32_t cb) {
    for (uint32_t ic = 0; ic < cb; ++ic) {
        DataCopyPad(
            rowStage,
            gm.denseGm[(static_cast<uint64_t>(blockCol * g.b + ic) * t.ld +
                        rowBase) * g.e],
            {1, rb * g.e, 0, 0, 0}, {false, 0, 0, 0});
        SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
        for (uint32_t ir = 0; ir < rb; ++ir) {
            for (uint32_t e = 0; e < g.e; ++e) {
                scr.SetValue((ic * g.b + ir) * g.e + e,
                             rowStage.GetValue(ir * g.e + e));
            }
        }
    }
}

__aicore__ inline void RunSmallBlockPath(const BellGeo &g,
    const DenseToSparseTilingData &t, const BellGm &gm,
    AscendC::TPipe &pipe) {
    const uint32_t blockBytesS = g.b * g.b * g.e;
    const uint32_t scrBytes = (blockBytesS + 31u) / 32u * 32u;
    AscendC::TBuf<AscendC::TPosition::VECCALC> scrB;
    pipe.InitBuffer(scrB, scrBytes + 64);
    AscendC::TBuf<AscendC::TPosition::VECCALC> rowB;
    pipe.InitBuffer(rowB, g.b * g.e + 16);
    auto scr = scrB.Get<uint8_t>();
    auto rowStage = rowB.Get<uint8_t>();
    for (uint32_t task = g.tBegin; task < g.tEnd; ++task) {
        for (uint32_t i = 0; i < blockBytesS; ++i) {
            scr.SetValue(i, 0);
        }
        DataCopyPad(rowStage.template ReinterpretCast<int32_t>(),
                    gm.patGm[task], {1, 4, 0, 0}, {false, 0, 0, 0});
        SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
        const int32_t enc = rowStage.template ReinterpretCast<int32_t>()
                                 .GetValue(0);
        if (enc >= g.base &&
            static_cast<uint32_t>(enc - g.base) < g.colBlocks) {
            const uint32_t blockCol = static_cast<uint32_t>(enc - g.base);
            const uint32_t rowBase =
                (task / static_cast<uint32_t>(g.slots)) * g.b;
            const uint32_t rb = static_cast<uint32_t>(
                t.rows - rowBase < g.b ? t.rows - rowBase : g.b);
            const uint32_t cb = static_cast<uint32_t>(
                t.cols - blockCol * g.b < g.b ? t.cols - blockCol * g.b
                                              : g.b);
            if (g.rowLayout) {
                SmallBlockFillRows(g, t, gm, rowStage, scr, rowBase,
                                   blockCol, rb, cb);
            } else {
                SmallBlockFillCols(g, t, gm, rowStage, scr, rowBase,
                                   blockCol, rb, cb);
            }
        }
        PipeBarrier<PIPE_ALL>();
        DataCopyPad(gm.valGm[static_cast<uint64_t>(task) * blockBytesS],
                    scr, {1, blockBytesS, 0, 0, 0});
        PipeBarrier<PIPE_MTE3>();
    }
}

// ---- Ring prologue: pattern load and the staging DMA of tasks 0/1
// BEFORE the static offset-chain build, so the MTE2 transfer overlaps
// the vector chain construction (the chain is per-core constant
// work). ----
__aicore__ inline void RingPrologue(const BellGeo &g,
    const DenseToSparseTilingData &t, const BellGm &gm,
    AscendC::TBuf<AscendC::TPosition::VECCALC> &patBuf,
    AscendC::TBuf<AscendC::TPosition::VECCALC> &stageBuf) {
    const uint32_t nTasksP = g.tEnd - g.tBegin;
    DataCopyPad(patBuf.Get<uint8_t>(),
                gm.patGm8[static_cast<uint64_t>(g.tBegin) * 4],
                {1, nTasksP * 4, 0, 0, 0}, {false, 0, 0, 0});
    SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
    WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
    const uint32_t slotsU = static_cast<uint32_t>(g.slots);
    BellStageTask(g, t, gm, stageBuf, patBuf.Get<int32_t>().GetValue(0),
                  (g.tBegin / slotsU) * g.b, 0);
    if (nTasksP > 1u) {
        BellStageTask(g, t, gm, stageBuf, patBuf.Get<int32_t>().GetValue(1),
                      ((g.tBegin + 1u) / slotsU) * g.b, g.stageSlot);
    }
}

#define D2S_SLOT_EV(op, s) \
    switch (s) { \
    case 0: op<HardEvent::MTE3_V>(EVENT_ID5); break; \
    case 1: op<HardEvent::MTE3_V>(EVENT_ID6); break; \
    default: op<HardEvent::MTE3_V>(EVENT_ID7); break; \
    }

// Ring turn drains: the vector-pipe drain of B(k-1) (load-bearing:
// without it ~50% of tasks read stale stage -- cross-pipe
// write-visibility latency), the tr-slot recycle wait, and the MTE2
// visibility barrier of this turn's staged block.
__aicore__ inline void RingPreflight(const BellGeo &g, uint32_t k,
    uint32_t sIdx, bool slotDrain, bool valid) {
    if (k >= 1u) {
        SetFlag<HardEvent::V_MTE2>(EVENT_ID4);
        WaitFlag<HardEvent::V_MTE2>(EVENT_ID4);
    }
    if (k >= g.ringT) {
        if (slotDrain) {
            D2S_SLOT_EV(WaitFlag, sIdx);
        } else {
            SetFlag<HardEvent::MTE3_V>(EVENT_ID5);
            WaitFlag<HardEvent::MTE3_V>(EVENT_ID5);
        }
    }
    if (valid) {
        SetFlag<HardEvent::MTE2_V>(EVENT_ID2);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID2);
    }
}

// Zero fill of an invalid task's tr slot.
__aicore__ inline void RingZeroFillTr(const BellGeo &g,
    AscendC::TBuf<AscendC::TPosition::VECCALC> &trBuf, uint32_t tOff) {
    for (uint32_t o = 0; o < g.trSlot / 2; o += kVecChunk) {
        const uint32_t len = (g.trSlot / 2 - o) < kVecChunk
                                 ? (g.trSlot / 2 - o)
                                 : kVecChunk;
        Duplicate(trBuf.Get<uint16_t>()[tOff / 2 + o],
                  static_cast<uint16_t>(0), len);
    }
}

// C(k): contiguous block store plus its per-slot (or every-turn) drain.
__aicore__ inline void RingStoreTask(const BellGeo &g, const BellGm &gm,
    AscendC::TBuf<AscendC::TPosition::VECCALC> &trBuf, uint32_t tOff,
    uint32_t k, uint32_t sIdx, bool slotDrain) {
    SetFlag<HardEvent::V_MTE3>(EVENT_ID3);
    WaitFlag<HardEvent::V_MTE3>(EVENT_ID3);
    DataCopyPad(
        gm.valGm[static_cast<uint64_t>(g.tBegin + k) * g.blockBytes],
        trBuf.Get<uint8_t>()[tOff],
        {1, g.blockBytes, 0, 0, 0});
    if (slotDrain) {
        D2S_SLOT_EV(SetFlag, sIdx);
    } else {
        SetFlag<HardEvent::MTE3_V>(EVENT_ID5);
        WaitFlag<HardEvent::MTE3_V>(EVENT_ID5);
    }
}

// ---- Task-ring pipeline (ROW, E >= 2, plain path) ----
// The batch loop below drains every engine once per batch, so staging /
// gather / store never overlap across batches. This ring keeps one task
// per T stage/tr slot pairs and issues the staging DMA of task k+1
// before draining the vector gather of task k, so MTE2 runs while V/MTE3
// finish task k. Engines are FIFO and slots recycle every T tasks, which
// the per-iteration drains cover (requires T >= 2: A(k+1) touches the
// stage slot last read by B(k+1-T), already drained).
// Per-slot MTE3_V drains: C(k) posts the flag of its own slot id right
// after the store; T iterations later B(k+T) waits that id before
// recycling the tr slot. The store thus gets a full ring period to land
// AND become visible (a plain every-T drain on one id read stale tr
// data on W31-class patterns), while short stores stay off the critical
// path.

__aicore__ inline void RingDrainTails(const BellGeo &g, uint32_t nTasks,
    bool slotDrain) {
    if (!slotDrain) {
        return;
    }
    for (uint32_t q = nTasks > g.ringT ? nTasks - g.ringT : 0u;
         q < nTasks; ++q) {
        D2S_SLOT_EV(WaitFlag, q % g.ringT);
    }
}

__aicore__ inline void RunTaskRing(const BellGeo &g,
    const DenseToSparseTilingData &t, const BellGm &gm,
    AscendC::TBuf<AscendC::TPosition::VECCALC> &stageBuf,
    AscendC::TBuf<AscendC::TPosition::VECCALC> &trBuf,
    AscendC::TBuf<AscendC::TPosition::VECCALC> &patBuf,
    AscendC::TBuf<AscendC::TPosition::VECCALC> &tblBuf) {
    const uint32_t nTasks = g.tEnd - g.tBegin;
    auto pat = patBuf.Get<int32_t>();
    auto offA = tblBuf.Get<uint32_t>();
    const bool slotDrain = g.ringT <= 3u;
    // Rolling bookkeeping (replaces per-iter % and /): sIdxR is the
    // current ring slot; rowBaseN is the row base of task k+1 (used by
    // the A(k+1) staging issue); rc tracks (tBegin+k+1)%slots so the
    // wrap at iter end advances rowBaseN exactly when task k+2 enters a
    // new block row. enc/encN carry the pattern of task k and k+1 across
    // iterations.
    const uint32_t slotsU = static_cast<uint32_t>(g.slots);
    uint32_t ringStaged = 2u; // tasks [0, 2) staged by the prologue
    uint32_t sIdxR = 0u;
    uint32_t rc = (g.tBegin + 1u) % slotsU;
    uint32_t rowBaseN = ((g.tBegin + 1u) / slotsU) * g.b;
    int32_t encR = pat.GetValue(0);
    int32_t encNR = nTasks > 1u ? pat.GetValue(1) : 0;
    for (uint32_t k = 0; k < nTasks; ++k) {
        const uint32_t sIdx = sIdxR;
        const uint32_t sOff = sIdx * g.stageSlot;
        const uint32_t tOff = sIdx * g.trSlot;
        const int32_t enc = encR;
        const bool valid = enc >= g.base &&
            static_cast<uint32_t>(enc - g.base) < g.colBlocks;
        RingPreflight(g, k, sIdx, slotDrain, valid);
        // B(k): transpose gather (or zero fill) into tr slot.
        if (!valid) {
            RingZeroFillTr(g, trBuf, tOff);
        } else {
            GatherTranspose(g, stageBuf, trBuf, offA, sOff, tOff);
        }
        if (k + 1u == ringStaged && k + 1u < nTasks) {
            ++ringStaged;
            // A(k+1): overlaps V(k); stage slot was last read by
            // B(k+1-T), already drained.
            const uint32_t nOff =
                (sIdx + 1u == g.ringT ? 0u : sIdx + 1u) * g.stageSlot;
            BellStageTask(g, t, gm, stageBuf, encNR, rowBaseN, nOff);
        }
        RingStoreTask(g, gm, trBuf, tOff, k, sIdx, slotDrain);
        sIdxR = sIdx + 1u == g.ringT ? 0u : sIdx + 1u;
        if (++rc == slotsU) {
            rc = 0u;
            rowBaseN += g.b;
        }
        encR = encNR;
        if (k + 2u < nTasks) {
            encNR = pat.GetValue(k + 2u);
        }
    }
    RingDrainTails(g, nTasks, slotDrain);
}

// Consume the trailing per-slot flags (at most min(T, nTasks) pending,
// one per distinct id).
// ---- Batch loop (T tasks per batch, engines drained once per batch) --

// COL-storage staging of one batch task: tail tasks pre-zero the tr
// slot and load only the valid sub-rectangle (right-padded to 32B);
// plain tasks load each column's run of rows.
__aicore__ inline void BatchStageColTask(const BellGeo &g,
    const DenseToSparseTilingData &t, const BellGm &gm,
    AscendC::TBuf<AscendC::TPosition::VECCALC> &trBuf, uint32_t rowBase,
    uint32_t blockCol, uint32_t i) {
    const uint32_t rb = static_cast<uint32_t>(
        t.rows - rowBase < g.b ? t.rows - rowBase : g.b);
    const uint32_t cb = static_cast<uint32_t>(
        t.cols - blockCol * g.b < g.b ? t.cols - blockCol * g.b : g.b);
    const bool tail = rb < g.b || cb < g.b;
    if (tail) {
        for (uint32_t o = 0; o < g.trSlot / 2; o += kVecChunk) {
            const uint32_t len = (g.trSlot / 2 - o) < kVecChunk
                                     ? (g.trSlot / 2 - o)
                                     : kVecChunk;
            Duplicate(trBuf.Get<uint16_t>()[i * g.trSlot / 2 + o],
                      static_cast<uint16_t>(0), len);
        }
        PipeBarrier<PIPE_V>();
        const uint32_t colLoadElems = (rb * g.e + 31u) / 32u * 32u / g.e;
        const auto colPad = static_cast<uint8_t>(colLoadElems - rb);
        for (uint32_t c = 0; c < cb; ++c) {
            DataCopyPad(
                trBuf.Get<uint8_t>()[i * g.trSlot + c * g.colPitch],
                gm.denseGm[(static_cast<uint64_t>(blockCol * g.b + c) *
                                t.ld + rowBase) * g.e],
                {1, static_cast<uint16_t>(rb * g.e), 0, 0},
                {true, 0, colPad, 0});
        }
        return;
    }
    for (uint32_t c = 0; c < g.b; ++c) {
        DataCopyPad(
            trBuf.Get<uint8_t>()[i * g.trSlot + c * g.colPitch],
            gm.denseGm[(static_cast<uint64_t>(blockCol * g.b + c) * t.ld +
                        rowBase) * g.e],
            {1, g.b * g.e, 0, 0, 0}, {false, 0, 0, 0});
    }
}

// Phase P0: pattern + dense staging of one batch.
__aicore__ inline void BatchStagePatterns(const BellGeo &g,
    const DenseToSparseTilingData &t, const BellGm &gm,
    AscendC::TBuf<AscendC::TPosition::VECCALC> &patBuf,
    AscendC::TBuf<AscendC::TPosition::VECCALC> &stageBuf,
    AscendC::TBuf<AscendC::TPosition::VECCALC> &trBuf, uint32_t t0,
    uint32_t nt) {
    DataCopyPad(patBuf.Get<uint8_t>(),
                gm.patGm8[static_cast<uint64_t>(t0) * 4],
                {1, nt * 4, 0, 0, 0}, {false, 0, 0, 0});
    SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
    WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
    for (uint32_t i = 0; i < nt; ++i) {
        const int32_t encoded = patBuf.Get<int32_t>().GetValue(i);
        if (encoded < g.base ||
            static_cast<uint32_t>(encoded - g.base) >= g.colBlocks) {
            continue;
        }
        const uint32_t blockCol = static_cast<uint32_t>(encoded - g.base);
        const uint32_t rowBase =
            ((t0 + i) / static_cast<uint32_t>(g.slots)) * g.b;
        if (g.rowLayout) {
            const uint32_t slot = i * g.stageSlot;
            const uint32_t rb = static_cast<uint32_t>(
                t.rows - rowBase < g.b ? t.rows - rowBase : g.b);
            const uint32_t cb = static_cast<uint32_t>(
                t.cols - blockCol * g.b < g.b ? t.cols - blockCol * g.b
                                              : g.b);
            if (g.use2d || rb < g.b || cb < g.b) {
                // Tail and plain 2D staging share the ring form.
                BellStageTask(g, t, gm, stageBuf, encoded, rowBase, slot);
            } else {
                for (uint32_t r = 0; r < g.b; ++r) {
                    DataCopyPad(
                        stageBuf.Get<uint8_t>()[slot + r * g.rowPitch],
                        gm.denseGm[(static_cast<uint64_t>(rowBase + r) *
                                        t.ld + blockCol * g.b) * g.e],
                        {1, g.b * g.e, 0, 0, 0}, {false, 0, 0, 0});
                }
            }
            continue;
        }
        BatchStageColTask(g, t, gm, trBuf, rowBase, blockCol, i);
    }
}

// Phase B, E=1 ROW prologue: batched planes over the whole staged pool.
__aicore__ inline void BatchPlanesE1(const BellGeo &g,
    AscendC::TBuf<AscendC::TPosition::VECCALC> &stageBuf,
    AscendC::TBuf<AscendC::TPosition::VECCALC> &scrBuf, uint32_t nt) {
    const uint32_t mlen = nt * g.s;
    auto m00 = scrBuf.Get<uint16_t>();
    auto wLo = scrBuf.Get<uint16_t>()[g.ringT * g.s];
    auto wHi = scrBuf.Get<uint16_t>()[2 * g.ringT * g.s];
    And(wLo, stageBuf.Get<uint16_t>(), m00, mlen);
    ShiftRight(wHi, stageBuf.Get<uint16_t>(),
               static_cast<uint16_t>(8), mlen);
    PipeBarrier<PIPE_V>();
}

// Phase B, one task: E=1 ROW parks the plane gathers in scr1; E=1 COL
// unpads the 2B/element staging through a per-word gather; E>=2 ROW
// takes the shared transpose gather.
__aicore__ inline void BatchUnpadColE1(const BellGeo &g,
    AscendC::TBuf<AscendC::TPosition::VECCALC> &trBuf,
    AscendC::TBuf<AscendC::TPosition::VECCALC> &scrBuf,
    const LocalTensor<uint32_t> &offA, uint32_t slot) {
    auto tr16 = trBuf.Get<uint16_t>()[slot / 2];
    auto colScr = scrBuf.Get<uint16_t>()[3 * g.ringT * g.s +
                                         2 * g.ringT * g.be +
                                         g.ringT * g.be / 2];
    for (uint32_t o = 0; o < g.be / 2; o += kGatherChunk) {
        const uint32_t len = (g.be / 2 - o) < kGatherChunk
                                 ? (g.be / 2 - o)
                                 : kGatherChunk;
        Gather(colScr[o], tr16, offA[o], 0, len);
    }
    for (uint32_t o = 0; o < g.be / 2; o += kVecChunk) {
        const uint32_t len = (g.be / 2 - o) < kVecChunk
                                 ? (g.be / 2 - o)
                                 : kVecChunk;
        Adds(tr16[o].ReinterpretCast<int16_t>(),
             colScr[o].ReinterpretCast<int16_t>(),
             static_cast<int16_t>(0), len);
    }
}

// Phase B, E=1 ROW epilogue: batched pair-pack over the pool.
__aicore__ inline void BatchTransposeOne(const BellGeo &g,
    AscendC::TBuf<AscendC::TPosition::VECCALC> &stageBuf,
    AscendC::TBuf<AscendC::TPosition::VECCALC> &trBuf,
    AscendC::TBuf<AscendC::TPosition::VECCALC> &scrBuf,
    const LocalTensor<uint32_t> &offA, const LocalTensor<uint32_t> &offA1T,
    int32_t encoded, uint32_t i) {
    const uint32_t slot = i * g.trSlot;
    if (!g.rowLayout) {
        if (g.e != 1) {
            return; // COL E>=2: staged in place by P0
        }
        BatchUnpadColE1(g, trBuf, scrBuf, offA, slot);
        return;
    }
    const uint32_t phaseBBlockCol =
        static_cast<uint32_t>(encoded - g.base);
    const uint32_t icoff =
        (g.use2d && g.needAlign &&
         ((phaseBBlockCol * g.b) & (g.stageW - 1u)) != 0)
            ? g.stageW / 2
            : 0;
    const LocalTensor<uint32_t> &offSel = icoff != 0 ? offA1T : offA;
    if (g.e == 1) {
        // scr1 per task: [even-k b*b/2 | odd-k b*b/2]; the shared offA
        // table serves both plane gathers.
        auto wLo = scrBuf.Get<uint16_t>()[g.ringT * g.s];
        auto wHi = scrBuf.Get<uint16_t>()[2 * g.ringT * g.s];
        auto scr1 = scrBuf.Get<uint16_t>()[3 * g.ringT * g.s];
        const uint32_t halfE = g.be / 2;
        Gather(scr1[i * g.be], wLo[i * g.s], offSel, 0, halfE);
        Gather(scr1[i * g.be + halfE], wHi[i * g.s], offSel, 0, halfE);
        return;
    }
    GatherTranspose(g, stageBuf, trBuf, offSel, i * g.stageSlot, slot);
}

// COL E=1 unpad of the 2B/element staging: per-word gather via
// scratch. Vector copy (add zero): the previous Copy(.., {1, 1, 8, 8})
// mis-copied for len >= 256 (b >= 32) -- same stride-unit defect class
// as the m00 pool fill. Adds has no uint16 overload; reinterpret as
// int16.
__aicore__ inline void BatchPairPackE1(const BellGeo &g,
    AscendC::TBuf<AscendC::TPosition::VECCALC> &trBuf,
    AscendC::TBuf<AscendC::TPosition::VECCALC> &tblBuf,
    AscendC::TBuf<AscendC::TPosition::VECCALC> &scrBuf, uint32_t nt) {
    const uint32_t pool = nt * (g.be / 2);
    auto pack1 = tblBuf.Get<uint32_t>()[g.wordsOffA + g.wordsOffA1];
    auto pack2 =
        tblBuf.Get<uint32_t>()[g.wordsOffA + g.wordsOffA1 + g.packWords];
    auto scr1 = scrBuf.Get<uint16_t>()[3 * g.ringT * g.s];
    auto scrSh = scrBuf.Get<uint16_t>()[3 * g.ringT * g.s + g.ringT * g.be];
    auto pk2 = scrBuf.Get<uint16_t>()[3 * g.ringT * g.s + 2 * g.ringT * g.be];
    ShiftLeft(scrSh, scr1, static_cast<uint16_t>(8), nt * g.be);
    auto trPool = trBuf.Get<uint16_t>();
    for (uint32_t o = 0; o < pool; o += kGatherChunk) {
        const uint32_t len = (pool - o) < kGatherChunk ? (pool - o)
                                                       : kGatherChunk;
        Gather(trPool[o], scr1, pack1[o], 0, len);
        Gather(pk2[o], scrSh, pack2[o], 0, len);
    }
    Or(trPool, trPool, pk2, pool);
}

// Phase C: contiguous output DMAs. Adjacent tasks are adjacent in the
// values buffer (task t at t*blockBytes) and their tr slots are
// contiguous in UB, so the whole batch is one DMA (saves nt-1 MTE3
// calls per batch). COL+E=1 stages 2B/element (trSlot = 2*blockBytes):
// keep the per-task form there.
__aicore__ inline void BatchStore(const BellGeo &g, const BellGm &gm,
    AscendC::TBuf<AscendC::TPosition::VECCALC> &trBuf, uint32_t t0,
    uint32_t nt) {
    if (g.trSlot == g.blockBytes) {
        DataCopyPad(gm.valGm[static_cast<uint64_t>(t0) * g.blockBytes],
                    trBuf.Get<uint8_t>(),
                    {1, nt * g.blockBytes, 0, 0, 0});
        return;
    }
    for (uint32_t i = 0; i < nt; ++i) {
        DataCopyPad(gm.valGm[static_cast<uint64_t>(t0 + i) * g.blockBytes],
                    trBuf.Get<uint8_t>()[i * g.trSlot],
                    {1, g.blockBytes, 0, 0, 0});
    }
}

// Zero fill of an invalid task's tr slot. E=1 ROW: the batched pair-pack
// rewrites every tr slot from scr1; zero the scr1 slot instead so the
// pack emits zeros into this tr slot.
__aicore__ inline void BatchZeroFill(const BellGeo &g,
    AscendC::TBuf<AscendC::TPosition::VECCALC> &trBuf,
    AscendC::TBuf<AscendC::TPosition::VECCALC> &scrBuf, uint32_t slot,
    uint32_t i) {
    if (g.e == 1 && g.rowLayout) {
        auto scr1 = scrBuf.Get<uint16_t>()[3 * g.ringT * g.s];
        for (uint32_t o = 0; o < g.be; o += kVecChunk) {
            const uint32_t len = (g.be - o) < kVecChunk ? (g.be - o)
                                                        : kVecChunk;
            Duplicate(scr1[i * g.be + o], static_cast<uint16_t>(0), len);
        }
        return;
    }
    for (uint32_t o = 0; o < DivUp(g.trSlot, 2); o += kVecChunk) {
        const uint32_t len = (DivUp(g.trSlot, 2) - o) < kVecChunk
                                 ? (DivUp(g.trSlot, 2) - o)
                                 : kVecChunk;
        Duplicate(trBuf.Get<uint16_t>()[slot / 2 + o],
                  static_cast<uint16_t>(0), len);
    }
}

// Phase B of one batch: E=1 ROW plane prologue, per-task transpose or
// zero fill, and the E=1 ROW pair-pack epilogue.
__aicore__ inline void BatchTransposeAll(const BellGeo &g,
    AscendC::TBuf<AscendC::TPosition::VECCALC> &stageBuf,
    AscendC::TBuf<AscendC::TPosition::VECCALC> &trBuf,
    AscendC::TBuf<AscendC::TPosition::VECCALC> &tblBuf,
    AscendC::TBuf<AscendC::TPosition::VECCALC> &scrBuf,
    AscendC::TBuf<AscendC::TPosition::VECCALC> &patBuf,
    const LocalTensor<uint32_t> &offA, const LocalTensor<uint32_t> &offA1T,
    uint32_t nt) {
    if (g.e == 1 && g.rowLayout) {
        BatchPlanesE1(g, stageBuf, scrBuf, nt);
    }
    for (uint32_t i = 0; i < nt; ++i) {
        const int32_t encoded = patBuf.Get<int32_t>().GetValue(i);
        const bool valid = encoded >= g.base &&
            static_cast<uint32_t>(encoded - g.base) < g.colBlocks;
        const uint32_t slot = i * g.trSlot;
        if (!valid) {
            BatchZeroFill(g, trBuf, scrBuf, slot, i);
            continue;
        }
        BatchTransposeOne(g, stageBuf, trBuf, scrBuf, offA,
                          offA1T, encoded, i);
    }
    if (g.e == 1 && g.rowLayout) {
        BatchPairPackE1(g, trBuf, tblBuf, scrBuf, nt);
    }
}

__aicore__ inline void RunBatchLoop(const BellGeo &g,
    const DenseToSparseTilingData &t, const BellGm &gm,
    AscendC::TBuf<AscendC::TPosition::VECCALC> &stageBuf,
    AscendC::TBuf<AscendC::TPosition::VECCALC> &trBuf,
    AscendC::TBuf<AscendC::TPosition::VECCALC> &tblBuf,
    AscendC::TBuf<AscendC::TPosition::VECCALC> &scrBuf,
    AscendC::TBuf<AscendC::TPosition::VECCALC> &patBuf) {
    auto offA = tblBuf.Get<uint32_t>();
    auto offA1T = tblBuf.Get<uint32_t>()[g.wordsOffA];
    for (uint32_t t0 = g.tBegin; t0 < g.tEnd; t0 += g.ringT) {
        const uint32_t nt = (t0 + g.ringT < g.tEnd) ? g.ringT
                                                    : (g.tEnd - t0);
        // ---- Phase P0: pattern staging ----
        BatchStagePatterns(g, t, gm, patBuf, stageBuf, trBuf, t0, nt);
        if (g.rowLayout || g.e == 1) {
            SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
        }

        // ---- Phase B: transposes / zero fills ----
        BatchTransposeAll(g, stageBuf, trBuf, tblBuf, scrBuf, patBuf,
                          offA, offA1T, nt);
        PipeBarrier<PIPE_V>();

        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        if (!g.rowLayout && g.e != 1) {
            SetFlag<HardEvent::MTE2_MTE3>(EVENT_ID6);
            WaitFlag<HardEvent::MTE2_MTE3>(EVENT_ID6);
        }
        // ---- Phase C: contiguous output DMAs ----
        BatchStore(g, gm, trBuf, t0, nt);
        if (t0 + g.ringT < g.tEnd) {
            // Drains before the next batch reuses the slot pools.
            SetFlag<HardEvent::MTE3_V>(EVENT_ID2);
            WaitFlag<HardEvent::MTE3_V>(EVENT_ID2);
            SetFlag<HardEvent::V_MTE2>(EVENT_ID3);
            WaitFlag<HardEvent::V_MTE2>(EVENT_ID3);
        }
    }
}

// ---------------------------------------------------------------------------
// Kernel entry
// ---------------------------------------------------------------------------

extern "C" __global__ __aicore__ void densetosparse_bell_kernel(
    GM_ADDR dense, GM_ADDR ellColInd, GM_ADDR values,
    DenseToSparseTilingData tiling) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    const uint32_t core = GetBlockIdx();
    const BellGeo g = MakeBellGeo(tiling, core);
    if (!g.active) {
        return;
    }
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECCALC> stageBuf;
    if (g.rowLayout) {
        pipe.InitBuffer(stageBuf, g.ringT * g.stageSlot + 64);
    }
    AscendC::TBuf<AscendC::TPosition::VECCALC> trBuf;
    pipe.InitBuffer(trBuf, g.ringT * g.trSlot + 64);
    AscendC::TBuf<AscendC::TPosition::VECCALC> tblBuf;
    pipe.InitBuffer(tblBuf, g.tblWords * 4 + 32);
    AscendC::TBuf<AscendC::TPosition::VECCALC> scrBuf;
    if (g.e == 1) {
        const uint32_t scrTotalU16 =
            g.ringT * (3 * g.s + 2 * g.be + g.be / 2) + 2 * g.s + g.be + 64;
        pipe.InitBuffer(scrBuf, scrTotalU16 * 2 + 32);
    }
    AscendC::TBuf<AscendC::TPosition::VECCALC> patBuf;
    {
        // Ring shapes need the whole per-core pattern staged at once;
        // everything else keeps the historical T*4+32 size so the UB
        // layout (and its bank alignment) stays identical for the
        // batch-loop paths.
        const bool patBig = g.ringActive;
        const uint32_t patCap =
            (g.tEnd - g.tBegin) < 512u ? (g.tEnd - g.tBegin) : 512u;
        const uint32_t patBytes =
            patBig ? (patCap * 4 > g.ringT * 4 ? patCap * 4 : g.ringT * 4)
                   : g.ringT * 4;
        pipe.InitBuffer(patBuf, (patBytes + 31u) / 32u * 32u + 64);
    }

    const BellGm gm = MakeBellGm(tiling, dense, ellColInd, values, g);
    // Ring prologue hoist: issue the pattern load and the staging DMA of
    // tasks 0/1 BEFORE the static offset-chain build, so the MTE2
    // transfer overlaps the vector chain construction.
    if (g.ringActive) {
        RingPrologue(g, tiling, gm, patBuf, stageBuf);
    }
    if (g.b < 16u) {
        RunSmallBlockPath(g, tiling, gm, pipe);
        return;
    }
    BuildOffsetTables(g, tblBuf, scrBuf);
    if (g.ringActive) {
        RunTaskRing(g, tiling, gm, stageBuf, trBuf, patBuf, tblBuf);
        return;
    }
    RunBatchLoop(g, tiling, gm, stageBuf, trBuf, tblBuf, scrBuf, patBuf);
}

// ---------------------------------------------------------------------------
// Host-side launchers.
// ---------------------------------------------------------------------------

} // namespace

void densetosparse_bell_kernel_do(GM_ADDR dense, GM_ADDR ellColInd,
    GM_ADDR values, uint32_t numBlocks, const DenseToSparseTilingData &tiling,
    void *stream)
{
    densetosparse_bell_kernel<<<numBlocks, nullptr, stream>>>(
        dense, ellColInd, values, tiling);
}
