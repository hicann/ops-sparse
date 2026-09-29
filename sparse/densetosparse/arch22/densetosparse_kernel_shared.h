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

#ifndef DENSETOSPARSE_ARCH22_KERNEL_SHARED_H_
#define DENSETOSPARSE_ARCH22_KERNEL_SHARED_H_

// Helpers shared by the CSR/CSC/COO and BELL kernel translation units.

#include "kernel_operator.h"
#include "densetosparse_tiling_data.h"

using namespace AscendC;

namespace densetosparse {

constexpr uint32_t kVecChunk = 256; // u16 words per vector transpose slice

// Callers guarantee b > 0 (validated on the host side); the clamp is a
// defensive fallback so a stray zero divisor degrades instead of faulting.
__aicore__ inline uint32_t DivUp(uint32_t a, uint32_t b)
{
    return b != 0U ? (a + b - 1) / b : a;
}

__aicore__ inline uint64_t DivUp64(uint64_t a, uint64_t b)
{
    return b != 0U ? (a + b - 1) / b : a;
}

__aicore__ inline uint64_t MinU64(uint64_t a, uint64_t b)
{
    return a < b ? a : b;
}

// Element-count alignment to a (possibly runtime-derived) granularity;
// the guard keeps a stray zero alignment a no-op instead of a fault.
__aicore__ inline uint32_t AlignUpU32(uint32_t v, uint32_t a)
{
    return a != 0U ? (v + a - 1U) / a * a : v;
}

// GM scalar write through UB staging: kernel-side GM stores must go through
// MTE3 (direct GlobalTensor::SetValue is not a legal AIV store path).
template <typename T>
__aicore__ inline void StoreGmScalar(GlobalTensor<T> gm, uint64_t index,
    T value, LocalTensor<T> stage)
{
    stage.SetValue(0, value);
    PipeBarrier<PIPE_ALL>();
    DataCopyPad(gm[index], stage, {1, (uint32_t)sizeof(T), 0, 0, 0});
    PipeBarrier<PIPE_ALL>();
}

// GM scalar reads likewise go through MTE2 into UB before consumption.
__aicore__ inline uint64_t LoadGmU64(GlobalTensor<uint64_t> gm, uint64_t index,
    LocalTensor<uint64_t> stage)
{
    DataCopyPad(stage, gm[index], {1, 8, 0, 0}, {false, 0, 0, 0});
    PipeBarrier<PIPE_MTE2>();
    return stage.GetValue(0);
}

__aicore__ inline int32_t LoadGmS32(GlobalTensor<int32_t> gm, uint64_t index,
    LocalTensor<int32_t> stage)
{
    DataCopyPad(stage, gm[index], {1, 4, 0, 0}, {false, 0, 0, 0});
    PipeBarrier<PIPE_MTE2>();
    return stage.GetValue(0);
}

// ---------------------------------------------------------------------------
// Shared kernel constants and geometry (count/convert kernels)
// ---------------------------------------------------------------------------

constexpr uint32_t kPanelCols = 32;     // CSC+ROW panel width (columns)
// Column-panel count width (CSR/COO+COL): 8-row groups run 4x fewer
// per-column tally chains than the 32-wide convert panels; the convert
// side stays 32 to keep its shared tile loads.
constexpr uint32_t kCountPanel = 8;
constexpr uint32_t kSlotBytes = 32;     // strided gather window width
constexpr uint32_t kL0Batch = 64;       // level0 entries per batched flush
constexpr uint32_t kStageElems = 4096;  // convert staging capacity (elements)
// COO+COL dense-row carve: a row chunk with at least this many nonzeros AND
// at least 1/8 of its minor span leaves the 32-row column panel for the
// interleaved direct path. Below that density the panel's shared tile load
// beats the per-row strided traffic.
constexpr uint64_t kDenseCarveMin = 256;
// Minimum contiguous nonzero run that pays for its own MTE2 staging load +
// barrier; shorter runs extract with the per-element scalar form.
constexpr uint32_t kRunVecMin = 64;
// Column-ramp slots kept in the COO prefix scratch buffer (2*1024*4 <= 8200).
constexpr uint32_t kRampElems = 1024;
constexpr uint32_t kPfxSpanElems = 1536; // per-core prefix span cap
constexpr uint32_t kOffSpanElems = 2048; // per-core offsets span cap
// Vectorized slot judging needs slotBytes*slots to be a 256B multiple of at
// least one DMA block row; smaller tails fall back to scalar slot reads.
constexpr uint64_t kSlotVecMinBytes = 2048;

struct UnitGeo {
    uint64_t minorDim;
    uint64_t chunks;
    bool contiguous;
};

__aicore__ inline UnitGeo MakeGeo(const DenseToSparseTilingData &t)
{
    UnitGeo g;
    if (t.format == kD2sFormatCoo && t.order == kD2sOrderRow) {
        // Flat row-major walk over row-major storage.
        g.minorDim = 0;
        g.chunks = 0;
        g.contiguous = false;
        return g;
    }
    // CSR majors are rows; CSC majors are columns. COO over column-major
    // storage adopts the CSR row x chunk decomposition so it can share
    // the CSR column-panel path (emission stays row-major).
    g.minorDim = (t.format == kD2sFormatCsc) ? t.rows : t.cols;
    g.chunks = DivUp64(g.minorDim, kD2sMinorChunk);
    // The equality form is CSR/CSC-only: for COO it would evaluate true
    // on the COL layout (false == false) and misroute the per-unit walk
    // to the contiguous segment judge.
    g.contiguous = t.format != kD2sFormatCoo &&
                   ((t.format == kD2sFormatCsr) ==
                    (t.order == kD2sOrderRow));
    return g;
}

// CSC+ROW panel activation: 32 adjacent columns share one row-panel DMA.
__aicore__ inline bool UsePanel(const DenseToSparseTilingData &t)
{
    return t.format == kD2sFormatCsc && t.order == kD2sOrderRow &&
           t.tileLen >= kPanelCols && t.ld >= kPanelCols &&
           (t.ld - kPanelCols) * t.elementBytes < 0xFFFFFFFFULL;
}

// CSR+COL / COO+COL column-panel activation (mirror of the CSC+ROW row
// panel): 32 adjacent rows share one column-panel DMA over the
// column-major storage, removing the 32B-per-element read amplification
// of the slot gather on this layout.
__aicore__ inline bool UseColPanel(const DenseToSparseTilingData &t)
{
    return (t.format == kD2sFormatCsr || t.format == kD2sFormatCoo) &&
           t.order == kD2sOrderCol &&
           t.tileLen >= kPanelCols && t.ld >= kPanelCols &&
           (t.ld - kPanelCols) * t.elementBytes < 0xFFFFFFFFULL;
}

// Per-core kernel prologue shared by the count and convert kernels:
// tiling view, geometry, dense GM window and the contiguous per-core
// unit range (empty on out-of-range cores).
template <typename T>
struct D2sCtx {
    const DenseToSparseTilingData *t = nullptr;
    UnitGeo geo{};
    GlobalTensor<T> denseGm{};
    uint64_t uBegin = 0;
    uint64_t uEnd = 0;
    uint32_t elemBytes = 0;
};

template <typename T>
__aicore__ inline D2sCtx<T> MakeD2sCtx(GM_ADDR dense,
    const DenseToSparseTilingData &tiling)
{
    D2sCtx<T> c;
    c.t = &tiling;
    c.elemBytes = tiling.elementBytes;
    c.geo = MakeGeo(tiling);
    c.denseGm.SetGlobalBuffer((__gm__ T *)dense);
    const uint64_t per = DivUp64(tiling.unitCount, tiling.numBlocks);
    const uint32_t core = GetBlockIdx();
    c.uBegin = MinU64((uint64_t)core * per, tiling.unitCount);
    c.uEnd = MinU64(c.uBegin + per, tiling.unitCount);
    return c;
}

// Common tiling/geometry state of the count and convert kernels; both
// Adopt the shared prologue context once (single definition of the
// field plumbing, kernels keep their private engine state below it).
template <typename T>
class D2sKernelBase {
protected:
    __aicore__ inline void AdoptCtx(const D2sCtx<T> &ctx)
    {
        t_ = ctx.t;
        elemBytes_ = ctx.elemBytes;
        geo_ = ctx.geo;
        denseGm_ = ctx.denseGm;
        uBegin_ = ctx.uBegin;
        uEnd_ = ctx.uEnd;
    }

    const DenseToSparseTilingData *t_ = nullptr;
    UnitGeo geo_;
    GlobalTensor<T> denseGm_;
    uint64_t uBegin_ = 0;
    uint64_t uEnd_ = 0;
    uint32_t elemBytes_ = 0;
};

// Slot-group staging shared by the count and convert paths: one 2D
// gather of `slots` 32B windows starting at minor t0 (complex/fp32 pad
// form, b8/b16 window form at winOff).
template <typename T, uint32_t kGmStride, bool kComplex, typename Q>
__aicore__ inline LocalTensor<T> LoadSlotGroup(Q &inQueue,
    GlobalTensor<T> &denseGm, const DenseToSparseTilingData &t,
    uint32_t elemBytes, uint64_t major, uint64_t t0, uint32_t slots,
    uint32_t winOff)
{
    auto tile = inQueue.template AllocTensor<T>();
    const uint32_t slotElems = kSlotBytes / elemBytes;
    if constexpr (kComplex || AscendC::IsSameType<T, float>::value) {
        // Pad form: one element at the slot head, right-padded to 32B.
        DataCopyPad(tile[0], denseGm[(t0 * t.ld + major) * kGmStride],
            {(uint16_t)slots, elemBytes,
             (uint32_t)((t.ld - 1) * elemBytes), 0, 0},
            {true, 0, (uint8_t)((kSlotBytes - elemBytes) / sizeof(T)), 0});
    } else {
        // Window form: 32B of the raw line per slot, element at winOff.
        DataCopyPad(tile[0],
            denseGm[(t0 * t.ld + major - winOff) * kGmStride],
            {(uint16_t)slots, kSlotBytes,
             (uint32_t)((t.ld - slotElems) * elemBytes), 0, 0},
            {false, 0, 0, 0});
    }
    inQueue.template EnQue(tile);
    return inQueue.template DeQue<T>();
}

// Flat COO tile [p0, p1): row range and per-row column bounds. A stray
// zero `cols` yields an empty range (the host never routes empty
// matrices here).
__aicore__ inline void CooTileRows(uint64_t p0, uint64_t p1, uint64_t cols,
    uint64_t &rBegin, uint64_t &rEnd)
{
    if (cols == 0) {
        rBegin = 1;
        rEnd = 0;
        return;
    }
    rBegin = p0 / cols;
    rEnd = (p1 - 1) / cols;
}

__aicore__ inline void CooRowCols(uint64_t p0, uint64_t p1, uint64_t cols,
    uint64_t r, uint64_t rBegin, uint64_t rEnd, uint64_t &cLo, uint64_t &cHi)
{
    cLo = (r == rBegin) ? p0 - r * cols : 0;
    cHi = (r == rEnd) ? (p1 - 1) - r * cols + 1 : cols;
}

// Flat COO tile [p0, p1) walked row span by row span: the handler's
// OnContig receives each row span's row/column bounds (ROW storage;
// the handler derives its GM/flat bases), OnStrided the same bounds
// (COL storage).
template <typename Handler>
__aicore__ inline void WalkCooTileRows(uint64_t p0, uint64_t p1,
    uint64_t cols, uint32_t order, Handler &h)
{
    uint64_t rBegin = 0;
    uint64_t rEnd = 0;
    CooTileRows(p0, p1, cols, rBegin, rEnd);
    for (uint64_t r = rBegin; r <= rEnd; ++r) {
        uint64_t cLo = 0;
        uint64_t cHi = 0;
        CooRowCols(p0, p1, cols, r, rBegin, rEnd, cLo, cHi);
        if (cHi <= cLo) {
            continue;
        }
        if (order == kD2sOrderRow) {
            h.OnContig(r, cLo, (uint32_t)(cHi - cLo));
        } else {
            h.OnStrided(r, cLo, (uint32_t)(cHi - cLo));
        }
    }
}

} // namespace densetosparse

#endif // DENSETOSPARSE_ARCH22_KERNEL_SHARED_H_
