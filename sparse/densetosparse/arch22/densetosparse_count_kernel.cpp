/**
 * ----------------------------------------------------------------------------------------------------------
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software: you can redistribute it and/or modify it under the terms of conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT OF MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ----------------------------------------------------------------------------------------------------------
 */

// Count/analysis kernel family: CountKernel, levelled uint64 scan
// chain, offsets/total epilogue, kernel entries and the host-side
// launch orchestration for CSR/CSC/COO.
#include "kernel_operator.h"
#include "densetosparse_kernel.h"
#include "densetosparse_kernel_shared.h"
#include "densetosparse_judge.h"

namespace densetosparse {
namespace {

// ---------------------------------------------------------------------------
// Count kernel
// ---------------------------------------------------------------------------

// Batched level0 staging: counts accumulate in UB and flush to GM every
// kL0Batch entries (unit-indexed stores only through the helpers below).
__aicore__ inline void FlushL0Entries(GlobalTensor<uint64_t> level0Gm,
    uint64_t base, const LocalTensor<uint64_t> &stage, uint32_t n)
{
    PipeBarrier<PIPE_ALL>();
    DataCopyPad(level0Gm[base], stage,
        {1, (uint32_t)(n * sizeof(uint64_t)), 0, 0, 0});
    PipeBarrier<PIPE_MTE3>();
}

__aicore__ inline void FlushL0Tail(GlobalTensor<uint64_t> level0Gm,
    uint64_t batchBase, LocalTensor<uint64_t> &stage, uint32_t batch)
{
    if (batch > 0) {
        FlushL0Entries(level0Gm, batchBase, stage, batch);
    }
}

__aicore__ inline void StageL0Entry(GlobalTensor<uint64_t> level0Gm,
    LocalTensor<uint64_t> &stage, uint32_t &batch, uint64_t &batchBase,
    uint64_t count)
{
    stage.SetValue(batch, count);
    ++batch;
    if (batch == kL0Batch) {
        FlushL0Entries(level0Gm, batchBase, stage, batch);
        batchBase += kL0Batch;
        batch = 0;
    }
}

template <typename T, uint32_t kGmStride, bool kComplex>
class CountKernel : public D2sKernelBase<T> {
public:
    using D2sKernelBase<T>::t_;
    using D2sKernelBase<T>::geo_;
    using D2sKernelBase<T>::denseGm_;
    using D2sKernelBase<T>::uBegin_;
    using D2sKernelBase<T>::uEnd_;
    using D2sKernelBase<T>::elemBytes_;
    __aicore__ inline void Init(GM_ADDR dense, GM_ADDR level0,
        const DenseToSparseTilingData &tiling)
    {
        if (GetBlockIdx() >= tiling.numBlocks) {
            return;
        }
        this->AdoptCtx(MakeD2sCtx<T>(dense, tiling));
        level0Gm_.SetGlobalBuffer((__gm__ uint64_t *)level0);
        bufs_.template Init<T, kComplex>(pipe_);
        panel_ = UsePanel(tiling);
        colPanel_ = UseColPanel(tiling);
        if (panel_ || colPanel_) {
            // Both panel orientations share the plane table: the bitmap
            // bit stride is 32 in either layout, so plane p selects byte
            // 4*i + (p>>3), bit (p&7) for the tally dimension i. The
            // column-panel (CSR/COO+COL) count side runs 8-row groups --
            // four times fewer per-column tally chains than the 32-wide
            // convert panels -- so it keeps its own width here while the
            // convert side stays 32-wide. Width 8 needs 8*E >= 32B dst
            // blocks (sub-32B 2D dst blocks do not advance on this MTE),
            // so half/int8 keep the 32-wide groups.
            cntPanelW_ = !panel_ && elemBytes_ * kCountPanel >= kSlotBytes
                             ? kCountPanel
                             : kPanelCols;
            const uint32_t rowsPerTile = tiling.tileLen / cntPanelW_;
            planeBytes_ = rowsPerTile * (cntPanelW_ / 8);
            pipe_.InitBuffer(planes_, cntPanelW_ * planeBytes_);
            pipe_.InitBuffer(acc_, cntPanelW_ * sizeof(uint32_t));
            InitPlanes(cntPanelW_, rowsPerTile);
        }
        pipe_.InitBuffer(inQueue_, 2, tiling.tileLen * tiling.elementBytes);
        pipe_.InitBuffer(l0Stage_, kL0Batch * sizeof(uint64_t));
    }

    __aicore__ inline void Process()
    {
        if (uBegin_ >= uEnd_) {
            return; // covers out-of-range cores and empty matrices
        }
        if (t_->format == kD2sFormatCoo && t_->order == kD2sOrderRow) {
            CountCoo();
        } else if (panel_) {
            CountPanels();
        } else if (colPanel_) {
            CountColPanels();
        } else {
            CountUnits();
        }
    }

private:
    __aicore__ inline void CountUnits()
    {
        auto stage = l0Stage_.Get<uint64_t>();
        uint32_t batch = 0;
        uint64_t batchBase = uBegin_;
        for (uint64_t u = uBegin_; u < uEnd_; ++u) {
            const uint64_t count = geo_.contiguous
                                       ? CountUnitContig(u)
                                       : CountUnitStrided(u);
            StageL0Entry(level0Gm_, stage, batch, batchBase, count);
        }
        FlushL0Tail(level0Gm_, batchBase, stage, batch);
    }

    // Contiguous minor direction: 1D tiles straight down the GM line.
    __aicore__ inline uint64_t CountUnitContig(uint64_t unit)
    {
        const uint64_t major = unit / geo_.chunks;
        const uint64_t begin = (unit - major * geo_.chunks) * kD2sMinorChunk;
        const uint64_t end = MinU64(begin + kD2sMinorChunk, geo_.minorDim);
        uint64_t count = 0;
        for (uint64_t t0 = begin; t0 < end; t0 += t_->tileLen) {
            const uint32_t len = (uint32_t)MinU64(t_->tileLen, end - t0);
            count += JudgeContig(major * t_->ld + t0, len);
        }
        return count;
    }

    __aicore__ inline uint64_t JudgeContig(uint64_t gmElem, uint32_t len)
    {
        auto tile = inQueue_.AllocTensor<T>();
        DataCopyPad(tile[0], denseGm_[gmElem * kGmStride],
            {1, (uint16_t)(len * elemBytes_), 0, 0}, {false, 0, 0, 0});
        inQueue_.EnQue(tile);
        auto tileIn = inQueue_.DeQue<T>();
        const uint64_t c = CountTile<T, kComplex>(bufs_, tileIn, len);
        inQueue_.FreeTensor(tileIn);
        return c;
    }

    // Strided minor direction: 32B windows gathered by one 2D DMA.
    __aicore__ inline uint64_t CountUnitStrided(uint64_t unit)
    {
        const uint64_t major = unit / geo_.chunks;
        const uint64_t begin = (unit - major * geo_.chunks) * kD2sMinorChunk;
        const uint64_t end = MinU64(begin + kD2sMinorChunk, geo_.minorDim);
        const uint32_t slotElems = kSlotBytes / elemBytes_;
        // The fixed coordinate bounds the window inside the storage line.
        const uint64_t fixedDim =
            t_->order == kD2sOrderRow ? t_->cols : t_->rows;
        if (fixedDim < slotElems ||
            (t_->ld - slotElems) * elemBytes_ >= 0xFFFFFFFFULL) {
            return CountUnitScalar(major, begin, end);
        }
        const uint32_t maxSlots = t_->tileLen * elemBytes_ / kSlotBytes;
        const uint32_t winOff = WindowOffset(major, slotElems, fixedDim);
        PrepareSlotWalkMask<T, kComplex>(bufs_, maxSlots, winOff);
        uint64_t count = 0;
        for (uint64_t t0 = begin; t0 < end;) {
            const uint32_t slots =
                (uint32_t)MinU64((uint64_t)maxSlots, end - t0);
            count += JudgeSlots(major, t0, slots, winOff);
            t0 += slots;
        }
        return count;
    }

    __aicore__ inline uint64_t JudgeSlots(uint64_t major, uint64_t t0,
        uint32_t slots, uint32_t winOff)
    {
        auto tileIn = LoadSlotGroup<T, kGmStride, kComplex>(
            inQueue_, denseGm_, *t_, elemBytes_, major, t0, slots, winOff);
        uint64_t count = 0;
        const uint64_t tileBytes = (uint64_t)slots * kSlotBytes;
        if (tileBytes % 256ULL == 0 && tileBytes >= kSlotVecMinBytes) {
            count = CountSlots<T, kComplex>(bufs_, tileIn, slots, winOff);
        } else {
            for (uint32_t i = 0; i < slots; ++i) {
                if (SlotElemNonzero<T, kComplex>(tileIn, i, winOff)) {
                    ++count;
                }
            }
        }
        inQueue_.FreeTensor(tileIn);
        return count;
    }

    // Extreme-ld / tiny-line defense: per-element contiguous judges.
    // CSR and COO majors are rows; the CSC major is a column.
    __aicore__ inline uint64_t CountUnitScalar(uint64_t major,
        uint64_t begin, uint64_t end)
    {
        const bool majorIsRow = t_->format != kD2sFormatCsc;
        uint64_t count = 0;
        for (uint64_t minor = begin; minor < end; ++minor) {
            const uint64_t gmElem = majorIsRow
                ? (t_->order == kD2sOrderRow ? major * t_->ld + minor
                                             : minor * t_->ld + major)
                : (t_->order == kD2sOrderRow ? minor * t_->ld + major
                                             : major * t_->ld + minor);
            count += JudgeContig(gmElem, 1);
        }
        return count;
    }

    // ---- CSC+ROW panel count ----
    __aicore__ inline void InitPlanes(uint32_t w, uint32_t rowsPerTile)
    {
        auto p = planes_.Get<int8_t>();
        for (uint32_t c = 0; c < w; ++c) {
            const uint32_t byteOff = c >> 3;
            const int8_t bitVal = (int8_t)(1U << (c & 7));
            const uint32_t base = c * planeBytes_;
            for (uint32_t i = 0; i < rowsPerTile; ++i) {
                for (uint32_t w2 = 0; w2 < w / 8; ++w2) {
                    p.SetValue(base + i * (w / 8) + w2,
                               w2 == byteOff ? bitVal : (int8_t)0);
                }
            }
        }
        PipeBarrier<PIPE_ALL>();
    }

    // Units not covered by a full panel group: counted individually.
    __aicore__ inline void CountTailBeyondPanels(uint64_t fullEnd,
        uint64_t chunks, LocalTensor<uint64_t> &stage)
    {
        if (chunks == 0) {
            return; // no minor chunks: every unit is beyond the panels
        }
        for (uint64_t u = uBegin_; u < uEnd_; ++u) {
            if (u / chunks < fullEnd) {
                continue; // covered by a full panel group above
            }
            StoreGmScalar<uint64_t>(level0Gm_, u, CountUnitStrided(u), stage);
        }
    }

    __aicore__ inline void CountPanels()
    {
        const uint64_t chunks = geo_.chunks;
        const uint64_t fullEnd = t_->cols - t_->cols % kPanelCols;
        auto stage = l0Stage_.Get<uint64_t>();
        auto acc = acc_.Get<uint32_t>();
        for (uint64_t c0 = (uBegin_ / chunks / kPanelCols) * kPanelCols;
             c0 + kPanelCols <= fullEnd; c0 += kPanelCols) {
            if (c0 * chunks >= uEnd_) {
                break;
            }
            for (uint64_t k = 0; k < chunks; ++k) {
                TallyPanelColumns(c0, k, acc);
                StorePanelUnits(c0, k, chunks, acc, stage);
            }
        }
        CountTailBeyondPanels(fullEnd, chunks, stage);
    }

    // Accumulate one panel group's per-column nonzero counts in UB.
    __aicore__ inline void TallyPanelColumns(uint64_t c0, uint64_t k,
        LocalTensor<uint32_t> &acc)
    {
        const uint32_t rowsPerTile = t_->tileLen / kPanelCols;
        for (uint32_t c = 0; c < kPanelCols; ++c) {
            acc.SetValue(c, 0U);
        }
        const uint64_t r1 = MinU64((k + 1) * kD2sMinorChunk, t_->rows);
        for (uint64_t t0 = k * kD2sMinorChunk; t0 < r1;
             t0 += rowsPerTile) {
            const uint32_t rows =
                (uint32_t)MinU64((uint64_t)rowsPerTile, r1 - t0);
            auto tile = inQueue_.AllocTensor<T>();
            DataCopyPad(tile[0],
                denseGm_[(t0 * t_->ld + c0) * kGmStride],
                {(uint16_t)rows, kPanelCols * elemBytes_,
                    (uint32_t)((t_->ld - kPanelCols) * elemBytes_),
                    0, 0},
                {false, 0, 0, 0});
            inQueue_.EnQue(tile);
            auto tileIn = inQueue_.DeQue<T>();
            BuildZeroBitmap<T, kComplex>(bufs_, tileIn,
                rows * kPanelCols);
            for (uint32_t c = 0; c < kPanelCols; ++c) {
                const uint32_t zeros = CountPanelColumnZeros(c, rows);
                acc.SetValue(c, acc.GetValue(c) + rows - zeros);
            }
            inQueue_.FreeTensor(tileIn);
        }
    }

    // Panel units are visited in (group, k) order, which is not ascending
    // unit order once chunks > 1: level0 must be written unit-indexed,
    // never batch-contiguous.
    __aicore__ inline void StorePanelUnits(uint64_t c0, uint64_t k,
        uint64_t chunks, LocalTensor<uint32_t> &acc,
        LocalTensor<uint64_t> &stage)
    {
        for (uint32_t c = 0; c < kPanelCols; ++c) {
            const uint64_t unit = (c0 + c) * chunks + k;
            if (unit < uBegin_ || unit >= uEnd_) {
                continue;
            }
            StoreGmScalar<uint64_t>(level0Gm_, unit,
                (uint64_t)acc.GetValue(c), stage);
        }
    }

    // ---- CSR+COL / COO+COL column-panel count (mirror of CountPanels) --
    // Panel group = 32 adjacent rows; tiles walk the minor (column)
    // direction with one 32-contiguous-row block per column.
    __aicore__ inline void CountColPanels()
    {
        const uint64_t chunks = geo_.chunks;
        const uint32_t w = cntPanelW_;
        if (w == 0) {
            return; // stray zero width: no panels to count
        }
        const uint64_t fullEnd = t_->rows - t_->rows % w;
        auto stage = l0Stage_.Get<uint64_t>();
        auto acc = acc_.Get<uint32_t>();
        for (uint64_t r0 = (uBegin_ / chunks / w) * w;
             r0 + w <= fullEnd; r0 += w) {
            if (r0 * chunks >= uEnd_) {
                break;
            }
            for (uint64_t k = 0; k < chunks; ++k) {
                TallyColPanelColumns(r0, k, w, acc);
                StoreColPanelUnits(r0, k, chunks, w, acc, stage);
            }
        }
        CountTailBeyondPanels(fullEnd, chunks, stage);
    }

    __aicore__ inline void TallyColPanelColumns(uint64_t r0, uint64_t k,
        uint32_t w, LocalTensor<uint32_t> &acc)
    {
        if (w == 0) {
            return; // stray zero width: no panels to tally
        }
        const uint32_t colsPerTile = t_->tileLen / w;
        for (uint32_t j = 0; j < w; ++j) {
            acc.SetValue(j, 0U);
        }
        const uint64_t c1 = MinU64((k + 1) * kD2sMinorChunk, t_->cols);
        for (uint64_t t0 = k * kD2sMinorChunk; t0 < c1;
             t0 += colsPerTile) {
            const uint32_t cols =
                (uint32_t)MinU64((uint64_t)colsPerTile, c1 - t0);
            auto tile = inQueue_.AllocTensor<T>();
            DataCopyPad(tile[0],
                denseGm_[(t0 * t_->ld + r0) * kGmStride],
                {(uint16_t)cols, w * elemBytes_,
                    (uint32_t)((t_->ld - w) * elemBytes_),
                    0, 0},
                {false, 0, 0, 0});
            inQueue_.EnQue(tile);
            auto tileIn = inQueue_.DeQue<T>();
            BuildZeroBitmap<T, kComplex>(bufs_, tileIn, cols * w);
            if (!PanelTileAllZero(cols, w)) {
                for (uint32_t j = 0; j < w; ++j) {
                    const uint32_t zeros =
                        CountPanelColumnZeros(j, cols);
                    acc.SetValue(j,
                        acc.GetValue(j) + cols - zeros);
                }
            }
            inQueue_.FreeTensor(tileIn);
        }
    }

    __aicore__ inline void StoreColPanelUnits(uint64_t r0, uint64_t k,
        uint64_t chunks, uint32_t w, LocalTensor<uint32_t> &acc,
        LocalTensor<uint64_t> &stage)
    {
        for (uint32_t j = 0; j < w; ++j) {
            const uint64_t unit = (r0 + j) * chunks + k;
            if (unit < uBegin_ || unit >= uEnd_) {
                continue;
            }
            StoreGmScalar<uint64_t>(level0Gm_, unit,
                (uint64_t)acc.GetValue(j), stage);
        }
    }

    // All-zero tile fast path: two aggregate chains replace the w
    // per-column tally chains. The bitmap is byte-per-8-rows; every
    // element is zero iff every byte is 0xFF == int8 -1. A magnitude sum
    // is NOT a valid test (mixed +ve/-ve bytes cancel), so bound instead:
    // max == min == -1.0h (bits 0xBC00).
    __aicore__ inline bool PanelTileAllZero(uint32_t cols, uint32_t w)
    {
        auto bmAgg = (kComplex ? bufs_.Mask2()
                               : bufs_.Mask1())
                         .template ReinterpretCast<int8_t>();
        const uint32_t bytesAgg = cols * (w / 8);
        Cast<half, int8_t>(bufs_.TmpH(), bmAgg,
            RoundMode::CAST_NONE, bytesAgg);
        PipeBarrier<PIPE_V>();
        auto redH = bufs_.Red().template ReinterpretCast<half>();
        auto redB = bufs_.Red().template ReinterpretCast<uint16_t>();
        bool allZero = false;
        ReduceMax<half>(redH, bufs_.TmpH(), bufs_.TmpH(),
            (int32_t)bytesAgg);
        PipeBarrier<PIPE_V>();
        if (redB.GetValue(0) == 0xBC00U) {
            ReduceMin<half>(redH, bufs_.TmpH(),
                bufs_.TmpH(), (int32_t)bytesAgg);
            PipeBarrier<PIPE_V>();
            allZero = redB.GetValue(0) == 0xBC00U;
        }
        return allZero;
    }

    // Per-column zero tally from the panel bitmap: AND with the column's
    // bit plane, cast, ReduceSum, decode by the plane's bit weight. The
    // byte scratch reuses ByteMask (the panel path never uses window masks).
    // complex64 parks its element bitmap in Mask2 (Mask1 holds the
    // intermediate per-component bitmap).
    __aicore__ inline uint32_t CountPanelColumnZeros(uint32_t c,
        uint32_t rows)
    {
        auto scratch8 = bufs_.ByteMask();
        auto bm = (kComplex ? bufs_.Mask2() : bufs_.Mask1())
                      .template ReinterpretCast<int8_t>();
        auto plane = planes_.Get<int8_t>()[c * planeBytes_];
        const uint32_t bytes = rows * (cntPanelW_ / 8);
        And<int8_t>(scratch8, bm, plane, bytes);
        PipeBarrier<PIPE_V>();
        Cast<half, int8_t>(bufs_.TmpH(), scratch8, RoundMode::CAST_NONE,
            bytes);
        PipeBarrier<PIPE_V>();
        Cast<float, half>(bufs_.Ind(), bufs_.TmpH(), RoundMode::CAST_NONE,
            bytes);
        PipeBarrier<PIPE_V>();
        ReduceSum<float>(bufs_.Red(), bufs_.Ind(), bufs_.Ind(),
            (int32_t)bytes);
        PipeBarrier<PIPE_V>();
        return (uint32_t)(DecodeFloatCount(bufs_.Red()) >> (c & 7));
    }

    // ---- COO count ----
    // Row-span handlers for the count side (see WalkCooTileRows).
    struct CooCountHandler {
        CountKernel *self;
        uint64_t *count;
        __aicore__ inline void OnContig(uint64_t r, uint64_t cLo, uint32_t len)
        {
            *count += self->JudgeContig(r * self->t_->ld + cLo, len);
        }
        __aicore__ inline void OnStrided(uint64_t r, uint64_t cLo,
            uint32_t len)
        {
            *count += self->JudgeCooRowStrided(r, cLo, len);
        }
    };

    __aicore__ inline void CountCoo()
    {
        const uint64_t logical = t_->rows * t_->cols;
        auto stage = l0Stage_.Get<uint64_t>();
        uint32_t batch = 0;
        uint64_t batchBase = uBegin_;
        for (uint64_t u = uBegin_; u < uEnd_; ++u) {
            const uint64_t p0 = u * kD2sCooTile;
            const uint64_t p1 = MinU64(p0 + kD2sCooTile, logical);
            uint64_t count = 0;
            CooCountHandler h{this, &count};
            WalkCooTileRows(p0, p1, t_->cols, t_->order, h);
            StageL0Entry(level0Gm_, stage, batch, batchBase, count);
        }
        FlushL0Tail(level0Gm_, batchBase, stage, batch);
    }

    // COO on COL layout: elements of one row sit one ld apart along the
    // storage lines; slot-walk them like the strided unit path.
    __aicore__ inline uint64_t JudgeCooRowStrided(uint64_t r, uint64_t cLo,
        uint32_t len)
    {
        const uint32_t slotElems = kSlotBytes / elemBytes_;
        if (t_->rows < slotElems ||
            (t_->ld - slotElems) * elemBytes_ >= 0xFFFFFFFFULL) {
            uint64_t cnt = 0;
            for (uint64_t c = cLo; c < cLo + len; ++c) {
                cnt += JudgeContig(c * t_->ld + r, 1);
            }
            return cnt;
        }
        const uint32_t maxSlots = t_->tileLen * elemBytes_ / kSlotBytes;
        const uint32_t winOff = WindowOffset(r, slotElems, t_->rows);
        PrepareSlotWalkMask<T, kComplex>(bufs_, maxSlots, winOff);
        uint64_t count = 0;
        for (uint32_t done = 0; done < len;) {
            const uint32_t slots =
                (uint32_t)MinU64((uint64_t)maxSlots, (uint64_t)(len - done));
            count += JudgeSlots(r, cLo + done, slots, winOff);
            done += slots;
        }
        return count;
    }

    TPipe pipe_;
    TQue<TPosition::VECIN, 2> inQueue_;
    TBuf<TPosition::VECCALC> l0Stage_;
    TBuf<TPosition::VECCALC> planes_;
    TBuf<TPosition::VECCALC> acc_;
    JudgeBufs bufs_;
    GlobalTensor<uint64_t> level0Gm_;
    uint32_t planeBytes_ = 0;
    uint32_t cntPanelW_ = kPanelCols;
    bool panel_ = false;
    bool colPanel_ = false;
};

} // namespace

namespace {

// ---------------------------------------------------------------------------
// Scan chain (uint64 workspace levels)
// ---------------------------------------------------------------------------

// Per-chunk inclusive scan with UB staging: MTE2 load, scalar prefix, MTE3
// store, optional chunk-sum handoff to the next level.
__aicore__ inline void ScanSegment(GlobalTensor<uint64_t> valGm,
    GlobalTensor<uint64_t> sumGm, bool hasSum, uint64_t count,
    uint64_t chunkCount, uint32_t numBlocks, LocalTensor<uint64_t> buf,
    LocalTensor<uint64_t> sumStage)
{
    const uint32_t core = GetBlockIdx();
    for (uint64_t chunk = core; chunk < chunkCount; chunk += numBlocks) {
        const uint64_t begin = chunk * kD2sScanBatch;
        const uint64_t end =
            begin + kD2sScanBatch < count ? begin + kD2sScanBatch : count;
        const uint32_t len = (uint32_t)(end - begin);
        if (len == 0) {
            continue;
        }
        DataCopyPad(buf, valGm[begin],
            {1, (uint16_t)(len * sizeof(uint64_t)), 0, 0}, {false, 0, 0, 0});
        PipeBarrier<PIPE_MTE2>();
        uint64_t sum = 0;
        for (uint32_t i = 0; i < len; ++i) {
            sum += buf.GetValue(i);
            buf.SetValue(i, sum);
        }
        PipeBarrier<PIPE_ALL>();
        DataCopyPad(valGm[begin], buf,
            {1, (uint32_t)(len * sizeof(uint64_t)), 0, 0, 0});
        if (hasSum) {
            StoreGmScalar<uint64_t>(sumGm, chunk, sum, sumStage);
        }
        PipeBarrier<PIPE_ALL>();
    }
}

// Chunk bases back-fill: every element of chunk c (>0) gains the prefix of
// chunk c-1.
__aicore__ inline void AddChunkBases(GlobalTensor<uint64_t> valGm,
    GlobalTensor<uint64_t> prefixGm, uint64_t count, uint32_t numBlocks,
    LocalTensor<uint64_t> buf, LocalTensor<uint64_t> stage)
{
    const uint32_t core = GetBlockIdx();
    const uint64_t chunkCount = DivUp64(count, kD2sScanBatch);
    for (uint64_t chunk = core; chunk < chunkCount; chunk += numBlocks) {
        if (chunk == 0) {
            continue;
        }
        const uint64_t base = LoadGmU64(prefixGm, chunk - 1, stage);
        const uint64_t begin = chunk * kD2sScanBatch;
        const uint64_t end =
            begin + kD2sScanBatch < count ? begin + kD2sScanBatch : count;
        const uint32_t len = (uint32_t)(end - begin);
        DataCopyPad(buf, valGm[begin],
            {1, (uint16_t)(len * sizeof(uint64_t)), 0, 0}, {false, 0, 0, 0});
        PipeBarrier<PIPE_MTE2>();
        for (uint32_t i = 0; i < len; ++i) {
            buf.SetValue(i, buf.GetValue(i) + base);
        }
        PipeBarrier<PIPE_ALL>();
        DataCopyPad(valGm[begin], buf,
            {1, (uint32_t)(len * sizeof(uint64_t)), 0, 0, 0});
        PipeBarrier<PIPE_ALL>();
    }
}

// ---------------------------------------------------------------------------
// Offsets / total / header / empty (Analysis epilogue)
// ---------------------------------------------------------------------------

// Prefix endpoints are strided by `chunks`; each core takes contiguous
// major groups so the covered prefix span and the offsets run stays
// contiguous and load/store in one pass each.
__aicore__ inline void WriteOffsets(GlobalTensor<uint64_t> prefixGm,
    GlobalTensor<int32_t> offsetsGm, GlobalTensor<int32_t> statusGm,
    uint64_t majorDim, uint64_t chunks, uint32_t base, uint32_t numBlocks,
    LocalTensor<uint64_t> spanBuf, LocalTensor<int32_t> outStage,
    LocalTensor<int32_t> statusStage, LocalTensor<int32_t> baseStage)
{
    const uint32_t core = GetBlockIdx();
    const uint64_t groupElems =
        chunks > 0 ? DivUp64(kD2sMinorChunk, chunks) : 1;
    for (uint64_t mStart = (uint64_t)core * groupElems; mStart < majorDim;
         mStart += (uint64_t)numBlocks * groupElems) {
        const uint64_t mEnd =
            mStart + groupElems < majorDim ? mStart + groupElems : majorDim;
        const uint64_t spanBegin = mStart * chunks;
        const uint64_t spanLen = mEnd * chunks - spanBegin;
        bool invalid = false;
        // outStage[i] corresponds to offsets[mStart+1+i]; the span is loaded
        // in <=4096-entry batches (blockLen stays inside uint16).
        for (uint64_t s0 = 0; s0 < spanLen; s0 += kD2sMinorChunk) {
            const uint64_t n = MinU64(spanLen - s0, kD2sMinorChunk);
            DataCopyPad(spanBuf, prefixGm[spanBegin + s0],
                {1, (uint16_t)(n * sizeof(uint64_t)), 0, 0}, {false, 0, 0, 0});
            PipeBarrier<PIPE_MTE2>();
            for (uint64_t m = mStart; m < mEnd; ++m) {
                const uint64_t idx = (m + 1) * chunks - 1 - spanBegin;
                if (idx < s0 || idx >= s0 + n) {
                    continue;
                }
                const uint64_t value =
                    spanBuf.GetValue((uint32_t)(idx - s0)) + base;
                if (value > 0x7FFFFFFFULL) {
                    invalid = true;
                } else {
                    outStage.SetValue((uint32_t)(m - mStart), (int32_t)value);
                }
            }
        }
        if (mStart == 0) {
            baseStage.SetValue(0, (int32_t)base);
            PipeBarrier<PIPE_ALL>();
            DataCopyPad(offsetsGm[0], baseStage, {1, 4, 0, 0, 0});
            PipeBarrier<PIPE_MTE3>();
        }
        if (!invalid) {
            PipeBarrier<PIPE_ALL>();
            DataCopyPad(offsetsGm[mStart + 1], outStage,
                {1, (uint32_t)((mEnd - mStart) * sizeof(int32_t)), 0, 0, 0});
            PipeBarrier<PIPE_MTE3>();
        }
    }
}

__aicore__ inline void MarkStatusInvalid(GlobalTensor<int32_t> statusGm,
    LocalTensor<int32_t> statusStage)
{
    statusStage.SetValue(0, kD2sDeviceStatusInvalid);
    PipeBarrier<PIPE_ALL>();
    DataCopyPad(statusGm[0], statusStage, {1, 4, 0, 0, 0});
    PipeBarrier<PIPE_MTE3>();
}

__aicore__ inline void WriteTotal(GlobalTensor<uint64_t> prefixGm,
    GlobalTensor<int32_t> statusGm, GlobalTensor<uint64_t> totalGm,
    uint64_t count, uint32_t base, LocalTensor<uint64_t> stage,
    LocalTensor<int32_t> statusStage)
{
    const uint64_t nnz = LoadGmU64(prefixGm, count - 1, stage);
    StoreGmScalar<uint64_t>(totalGm, 0, nnz, stage);
    if (nnz + base > 0x7FFFFFFFULL) {
        MarkStatusInvalid(statusGm, statusStage);
    }
}

} // namespace

template <typename T, uint32_t kGmStride, bool kComplex>
__aicore__ inline void RunCount(GM_ADDR dense, GM_ADDR level0,
    const DenseToSparseTilingData &tiling)
{
    CountKernel<T, kGmStride, kComplex> op;
    op.Init(dense, level0, tiling);
    op.Process();
}

} // namespace densetosparse

extern "C" __global__ __aicore__ void densetosparse_count_kernel(
    GM_ADDR dense, GM_ADDR level0, DenseToSparseTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if (tiling.elementBytes == 1) {
        densetosparse::RunCount<int8_t, 1, false>(dense, level0, tiling);
    } else if (tiling.elementBytes == 2) {
        densetosparse::RunCount<half, 1, false>(dense, level0, tiling);
    } else if (tiling.elementBytes == 4) {
        densetosparse::RunCount<float, 1, false>(dense, level0, tiling);
    } else {
        densetosparse::RunCount<float, 2, true>(dense, level0, tiling);
    }
}

extern "C" __global__ __aicore__ void densetosparse_header_kernel(
    GM_ADDR status, GM_ADDR total)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if (GetBlockIdx() != 0) {
        return;
    }
    GlobalTensor<int32_t> statusGm;
    GlobalTensor<uint64_t> totalGm;
    statusGm.SetGlobalBuffer((__gm__ int32_t *)status);
    totalGm.SetGlobalBuffer((__gm__ uint64_t *)total);
    TPipe pipe;
    TBuf<TPosition::VECCALC> bufS;
    TBuf<TPosition::VECCALC> bufT;
    pipe.InitBuffer(bufS, 16);
    pipe.InitBuffer(bufT, 16);
    densetosparse::StoreGmScalar<int32_t>(statusGm, 0,
        kD2sDeviceStatusSuccess, bufS.Get<int32_t>());
    densetosparse::StoreGmScalar<uint64_t>(totalGm, 0, 0,
        bufT.Get<uint64_t>());
}

extern "C" __global__ __aicore__ void densetosparse_scan_kernel(
    GM_ADDR values, GM_ADDR chunkSums, uint64_t count, uint64_t chunkCount,
    uint32_t numBlocks)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    GlobalTensor<uint64_t> valGm;
    GlobalTensor<uint64_t> sumGm;
    valGm.SetGlobalBuffer((__gm__ uint64_t *)values);
    sumGm.SetGlobalBuffer((__gm__ uint64_t *)chunkSums);
    TPipe pipe;
    TBuf<TPosition::VECCALC> buf;
    TBuf<TPosition::VECCALC> stageBuf;
    pipe.InitBuffer(buf, (uint32_t)(kD2sScanBatch * sizeof(uint64_t)));
    pipe.InitBuffer(stageBuf, 16);
    densetosparse::ScanSegment(valGm, sumGm, chunkSums != nullptr, count,
        chunkCount, numBlocks, buf.Get<uint64_t>(),
        stageBuf.Get<uint64_t>());
}

extern "C" __global__ __aicore__ void densetosparse_add_base_kernel(
    GM_ADDR values, GM_ADDR chunkPrefix, uint64_t count, uint32_t numBlocks)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    GlobalTensor<uint64_t> valGm;
    GlobalTensor<uint64_t> prefixGm;
    valGm.SetGlobalBuffer((__gm__ uint64_t *)values);
    prefixGm.SetGlobalBuffer((__gm__ uint64_t *)chunkPrefix);
    TPipe pipe;
    TBuf<TPosition::VECCALC> buf;
    TBuf<TPosition::VECCALC> stageBuf;
    pipe.InitBuffer(buf, (uint32_t)(kD2sScanBatch * sizeof(uint64_t)));
    pipe.InitBuffer(stageBuf, 16);
    densetosparse::AddChunkBases(valGm, prefixGm, count, numBlocks,
        buf.Get<uint64_t>(), stageBuf.Get<uint64_t>());
}

extern "C" __global__ __aicore__ void densetosparse_offsets_kernel(
    GM_ADDR prefix, GM_ADDR offsets, GM_ADDR status,
    DenseToSparseTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if (GetBlockIdx() >= tiling.numBlocks) {
        return;
    }
    const uint64_t majorDim =
        tiling.format == kD2sFormatCsr ? tiling.rows : tiling.cols;
    const uint64_t chunks = densetosparse::DivUp64(
        tiling.format == kD2sFormatCsr ? tiling.cols : tiling.rows,
        kD2sMinorChunk);
    GlobalTensor<uint64_t> prefixGm;
    GlobalTensor<int32_t> offsetsGm;
    GlobalTensor<int32_t> statusGm;
    prefixGm.SetGlobalBuffer((__gm__ uint64_t *)prefix);
    offsetsGm.SetGlobalBuffer((__gm__ int32_t *)offsets);
    statusGm.SetGlobalBuffer((__gm__ int32_t *)status);
    TPipe pipe;
    TBuf<TPosition::VECCALC> spanBuf;
    TBuf<TPosition::VECCALC> outBuf;
    TBuf<TPosition::VECCALC> statusBuf;
    pipe.InitBuffer(spanBuf, (uint32_t)(kD2sMinorChunk * sizeof(uint64_t)));
    pipe.InitBuffer(outBuf, (uint32_t)(kD2sMinorChunk * sizeof(int32_t)));
    pipe.InitBuffer(statusBuf, 16);
    densetosparse::WriteOffsets(prefixGm, offsetsGm, statusGm, majorDim,
        chunks, tiling.base, tiling.numBlocks, spanBuf.Get<uint64_t>(),
        outBuf.Get<int32_t>(), statusBuf.Get<int32_t>(),
        statusBuf.Get<int32_t>());
}

extern "C" __global__ __aicore__ void densetosparse_total_kernel(
    GM_ADDR prefix, GM_ADDR status, GM_ADDR total, uint64_t count,
    uint32_t base)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if (GetBlockIdx() != 0) {
        return;
    }
    GlobalTensor<uint64_t> prefixGm;
    GlobalTensor<int32_t> statusGm;
    GlobalTensor<uint64_t> totalGm;
    prefixGm.SetGlobalBuffer((__gm__ uint64_t *)prefix);
    statusGm.SetGlobalBuffer((__gm__ int32_t *)status);
    totalGm.SetGlobalBuffer((__gm__ uint64_t *)total);
    TPipe pipe;
    TBuf<TPosition::VECCALC> stageBuf;
    TBuf<TPosition::VECCALC> statusBuf;
    pipe.InitBuffer(stageBuf, 16);
    pipe.InitBuffer(statusBuf, 16);
    densetosparse::WriteTotal(prefixGm, statusGm, totalGm, count, base,
        stageBuf.Get<uint64_t>(), statusBuf.Get<int32_t>());
}

extern "C" __global__ __aicore__ void densetosparse_empty_offsets_kernel(
    GM_ADDR offsets, DenseToSparseTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if (GetBlockIdx() >= tiling.numBlocks) {
        return;
    }
    const uint64_t majorDim =
        (tiling.format == kD2sFormatCsr ? tiling.rows : tiling.cols) + 1;
    GlobalTensor<int32_t> offsetsGm;
    offsetsGm.SetGlobalBuffer((__gm__ int32_t *)offsets);
    TPipe pipe;
    TBuf<TPosition::VECCALC> stageBuf;
    pipe.InitBuffer(stageBuf, (uint32_t)(kD2sMinorChunk * sizeof(int32_t)));
    auto stage = stageBuf.Get<int32_t>();
    for (uint32_t i = 0; i < kD2sMinorChunk; ++i) {
        stage.SetValue(i, (int32_t)tiling.base);
    }
    PipeBarrier<PIPE_ALL>();
    const uint32_t core = GetBlockIdx();
    for (uint64_t start = (uint64_t)core * kD2sMinorChunk; start < majorDim;
         start += (uint64_t)tiling.numBlocks * kD2sMinorChunk) {
        const uint32_t len = (uint32_t)(start + kD2sMinorChunk < majorDim
                                             ? kD2sMinorChunk
                                             : majorDim - start);
        DataCopyPad(offsetsGm[start], stage,
            {1, (uint32_t)(len * sizeof(int32_t)), 0, 0, 0});
        PipeBarrier<PIPE_MTE3>();
    }
}

// ---------------------------------------------------------------------------
extern "C" __global__ __aicore__ void densetosparse_convert_kernel(
    GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR,
    DenseToSparseTilingData);

// Host-side launch orchestration
// ---------------------------------------------------------------------------

namespace {
constexpr uint32_t kMaxScanLevels = 16;

uint64_t AlignWs(uint64_t value)
{
    return (value + kD2sWorkspaceAlignment - 1) &
           ~(uint64_t)(kD2sWorkspaceAlignment - 1);
}

void LaunchCountAndScan(GM_ADDR dense, GM_ADDR workspace, uint32_t numBlocks,
    const DenseToSparseTilingData &tiling, void *stream)
{
    GM_ADDR status = workspace + tiling.statusOffset;
    GM_ADDR total = workspace + tiling.nnzOffset;
    densetosparse_header_kernel<<<1, nullptr, stream>>>(status, total);
    GM_ADDR level0 = workspace + tiling.level0Offset;
    densetosparse_count_kernel<<<numBlocks, nullptr, stream>>>(dense, level0,
        tiling);
    GM_ADDR levels[kMaxScanLevels] = {};
    uint64_t counts[kMaxScanLevels] = {};
    uint32_t levelCount = 1;
    levels[0] = level0;
    counts[0] = tiling.unitCount;
    uint64_t offset = tiling.level0Offset +
                      AlignWs(tiling.unitCount * sizeof(uint64_t));
    while (counts[levelCount - 1] > kD2sScanBatch &&
           levelCount < kMaxScanLevels) {
        counts[levelCount] =
            (counts[levelCount - 1] + kD2sScanBatch - 1) / kD2sScanBatch;
        levels[levelCount] = workspace + offset;
        offset += AlignWs(counts[levelCount] * sizeof(uint64_t));
        ++levelCount;
    }
    for (uint32_t level = 0; level < levelCount; ++level) {
        const uint64_t chunks =
            (counts[level] + kD2sScanBatch - 1) / kD2sScanBatch;
        GM_ADDR next = level + 1 < levelCount ? levels[level + 1] : nullptr;
        densetosparse_scan_kernel<<<numBlocks, nullptr, stream>>>(
            levels[level], next, counts[level], chunks, numBlocks);
    }
    for (uint32_t level = levelCount - 1; level > 0; --level) {
        densetosparse_add_base_kernel<<<numBlocks, nullptr, stream>>>(
            levels[level - 1], levels[level], counts[level - 1], numBlocks);
    }
}
} // namespace

void densetosparse_analysis_kernel_do(GM_ADDR dense, GM_ADDR offsets,
    GM_ADDR workspace, uint32_t numBlocks,
    const DenseToSparseTilingData &tiling, void *stream)
{
    if (tiling.unitCount == 0) {
        if (tiling.format != kD2sFormatCoo) {
            densetosparse_empty_offsets_kernel<<<numBlocks, nullptr, stream>>>(
                offsets, tiling);
        }
        return;
    }
    LaunchCountAndScan(dense, workspace, numBlocks, tiling, stream);
    GM_ADDR level0 = workspace + tiling.level0Offset;
    GM_ADDR status = workspace + tiling.statusOffset;
    GM_ADDR total = workspace + tiling.nnzOffset;
    if (tiling.format != kD2sFormatCoo) {
        densetosparse_offsets_kernel<<<numBlocks, nullptr, stream>>>(
            level0, offsets, status, tiling);
    }
    densetosparse_total_kernel<<<1, nullptr, stream>>>(
        level0, status, total, tiling.unitCount, tiling.base);
}

void densetosparse_convert_kernel_do(GM_ADDR dense, GM_ADDR workspace,
    GM_ADDR offsets, GM_ADDR indices, GM_ADDR rowIndices, GM_ADDR colIndices,
    GM_ADDR values, GM_ADDR ellColInd, uint32_t numBlocks,
    const DenseToSparseTilingData &tiling, void *stream)
{
    if (tiling.format == kD2sFormatBell) {
        densetosparse_bell_kernel_do(dense, ellColInd, values, numBlocks,
            tiling, stream);
        return;
    }
    if (tiling.unitCount == 0) {
        return;
    }
    // Convert trusts the Analysis prefix: the same-buffer protocol is
    // enforced on the host side; only the payload is written here.
    GM_ADDR level0 = workspace + tiling.level0Offset;
    GM_ADDR status = workspace + tiling.statusOffset;
    densetosparse_convert_kernel<<<numBlocks, nullptr, stream>>>(
        dense, level0, status, offsets, indices, rowIndices, colIndices,
        values, tiling);
}
