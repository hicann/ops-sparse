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

// Convert kernel: per-unit payload extraction for CSR/CSC/COO (panel
// fast paths, per-core position resolver), consuming the analysis
// prefix via the same-buffer protocol.
#include "kernel_operator.h"
#include "densetosparse_kernel.h"
#include "densetosparse_kernel_shared.h"
#include "densetosparse_judge.h"

namespace densetosparse {
namespace {

// Loads one dense tile through the input queue and builds its zero
// bitmap; shared prologue of the CSR/CSC and COO convert paths.
template <typename T, uint32_t kGmStride, bool kComplex, typename Q, typename B>
__aicore__ inline LocalTensor<T> LoadTileAndBuildBitmap(Q &inQueue,
    GlobalTensor<T> &denseGm, B &bufs, uint64_t gmElem, uint32_t len,
    uint32_t elemBytes)
{
    auto tile = inQueue.template AllocTensor<T>();
    DataCopyPad(tile[0], denseGm[gmElem * kGmStride],
        {1, (uint16_t)(len * elemBytes), 0, 0}, {false, 0, 0, 0});
    inQueue.template EnQue(tile);
    auto tileIn = inQueue.template DeQue<T>();
    BuildZeroBitmap<T, kComplex>(bufs, tileIn, len);
    return tileIn;
}


// ---------------------------------------------------------------------------
// Convert kernel
// ---------------------------------------------------------------------------

// Per-core output position resolver. CSR/CSC: outPos(unit) =
// (offsets[major]-base) + prefixBefore(unit) - prefixBefore(major start).
// The prefix/offsets entries are cached in contiguous UB spans; oversized
// core ranges fall back to staged per-entry GM loads.
class PositionResolver {
public:
    __aicore__ inline void Init(TPipe &pipe, GlobalTensor<uint64_t> prefixGm,
        GlobalTensor<int32_t> offsetsGm, uint64_t uBegin, uint64_t uEnd,
        uint64_t chunks, uint32_t format, uint32_t base)
    {
        prefixGm_ = prefixGm;
        offsetsGm_ = offsetsGm;
        uBegin_ = uBegin;
        uEnd_ = uEnd;
        chunks_ = chunks;
        format_ = format;
        base_ = base;
        pipe.InitBuffer(pfxSpan_, kPfxSpanElems * sizeof(uint64_t));
        pipe.InitBuffer(offSpan_, kOffSpanElems * sizeof(int32_t));
        pipe.InitBuffer(scratch_, 32);
        pfxValid_ = false;
        offValid_ = false;
    }

    // Rebind the unit range (per-unit fallback paths shrink it).
    __aicore__ inline void SetRange(uint64_t uBegin, uint64_t uEnd)
    {
        uBegin_ = uBegin;
        uEnd_ = uEnd;
        pfxValid_ = false;
        offValid_ = false;
    }

    __aicore__ inline void Prime()
    {
        if (format_ == kD2sFormatCoo) {
            LoadPfxSpan(uBegin_ > 0 ? uBegin_ - 1 : 0);
            return;
        }
        const uint64_t majorLo = uBegin_ / chunks_;
        const uint64_t majorHi = (uEnd_ - 1) / chunks_;
        // Lowest prefix index needed: min(uBegin-1, majorLo*chunks-1).
        uint64_t pfxBase = uBegin_;
        if (uBegin_ > 0) {
            pfxBase = uBegin_ - 1;
        }
        if (majorLo > 0 && majorLo * chunks_ - 1 < pfxBase) {
            pfxBase = majorLo * chunks_ - 1;
        }
        LoadPfxSpan(pfxBase);
        LoadOffSpan(majorLo, majorHi);
    }

    __aicore__ inline uint64_t PrefixAt(uint64_t idx)
    {
        if (pfxValid_ && idx >= pfxBase_ && idx < pfxTop_) {
            return pfxSpan_.Get<uint64_t>().GetValue((uint32_t)(idx - pfxBase_));
        }
        return LoadGmU64(prefixGm_, idx, scratch_.Get<uint64_t>());
    }

    __aicore__ inline int32_t OffsetAt(uint64_t major)
    {
        if (offValid_ && major >= offBase_ && major <= offTop_) {
            return offSpan_.Get<int32_t>().GetValue((uint32_t)(major - offBase_));
        }
        return LoadGmS32(offsetsGm_, major, scratch_.Get<int32_t>());
    }

    __aicore__ inline uint64_t UnitOutPos(uint64_t unit)
    {
        const uint64_t prefixBefore = unit > 0 ? PrefixAt(unit - 1) : 0;
        if (format_ == kD2sFormatCoo) {
            return prefixBefore;
        }
        const uint64_t major = unit / chunks_;
        const uint64_t majorPrefix =
            major > 0 ? PrefixAt(major * chunks_ - 1) : 0;
        return (uint64_t)((int64_t)OffsetAt(major) - (int64_t)base_) +
               prefixBefore - majorPrefix;
    }

private:
    __aicore__ inline void LoadPfxSpan(uint64_t baseIdx)
    {
        const uint64_t spanLen = uEnd_ - baseIdx;
        if (spanLen > kPfxSpanElems) {
            return; // fallback per-entry loads
        }
        DataCopyPad(pfxSpan_.Get<uint64_t>(), prefixGm_[baseIdx],
            {1, (uint16_t)(spanLen * sizeof(uint64_t)), 0, 0},
            {false, 0, 0, 0});
        PipeBarrier<PIPE_MTE2>();
        pfxBase_ = baseIdx;
        pfxTop_ = uEnd_;
        pfxValid_ = true;
    }

    __aicore__ inline void LoadOffSpan(uint64_t majorLo, uint64_t majorHi)
    {
        const uint64_t spanLen = majorHi - majorLo + 1;
        if (spanLen > kOffSpanElems) {
            return;
        }
        DataCopyPad(offSpan_.Get<int32_t>(), offsetsGm_[majorLo],
            {1, (uint16_t)(spanLen * sizeof(int32_t)), 0, 0}, {false, 0, 0, 0});
        PipeBarrier<PIPE_MTE2>();
        offBase_ = majorLo;
        offTop_ = majorHi;
        offValid_ = true;
    }

    TBuf<TPosition::VECCALC> pfxSpan_;
    TBuf<TPosition::VECCALC> offSpan_;
    TBuf<TPosition::VECCALC> scratch_;
    GlobalTensor<uint64_t> prefixGm_;
    GlobalTensor<int32_t> offsetsGm_;
    uint64_t uBegin_ = 0;
    uint64_t uEnd_ = 0;
    uint64_t chunks_ = 1;
    uint64_t pfxBase_ = 0;
    uint64_t pfxTop_ = 0;
    uint64_t offBase_ = 0;
    uint64_t offTop_ = 0;
    uint32_t format_ = 0;
    uint32_t base_ = 0;
    bool pfxValid_ = false;
    bool offValid_ = false;
};

template <typename T, uint32_t kGmStride, bool kComplex>
class ConvertKernel : public D2sKernelBase<T> {
public:
    using D2sKernelBase<T>::t_;
    using D2sKernelBase<T>::geo_;
    using D2sKernelBase<T>::denseGm_;
    using D2sKernelBase<T>::uBegin_;
    using D2sKernelBase<T>::uEnd_;
    using D2sKernelBase<T>::elemBytes_;
    __aicore__ inline void Init(GM_ADDR dense, GM_ADDR prefix, GM_ADDR status,
        GM_ADDR offsets, GM_ADDR indices, GM_ADDR rowIndices,
        GM_ADDR colIndices, GM_ADDR values,
        const DenseToSparseTilingData &tiling)
    {
        core_ = GetBlockIdx();
        if (core_ >= tiling.numBlocks) {
            return;
        }
        this->AdoptCtx(MakeD2sCtx<T>(dense, tiling));
        prefixGm_.SetGlobalBuffer((__gm__ uint64_t *)prefix);
        statusGm_.SetGlobalBuffer((__gm__ int32_t *)status);
        offsetsGm_.SetGlobalBuffer((__gm__ int32_t *)offsets);
        indicesGm_.SetGlobalBuffer((__gm__ int32_t *)indices);
        rowGm_.SetGlobalBuffer((__gm__ int32_t *)rowIndices);
        colGm_.SetGlobalBuffer((__gm__ int32_t *)colIndices);
        valuesGm_.SetGlobalBuffer((__gm__ T *)values);
        bufs_.template Init<T, kComplex>(pipe_);
        panel_ = UsePanel(tiling);
        colPanel_ = UseColPanel(tiling);
        // Both panel orientations stage the group metadata the same way;
        // an un-InitBuffer'd TBuf must never be Get()'d (instant fault).
        if (panel_ || colPanel_) {
            const uint64_t chunksInit = DivUp64(tiling.rows, kD2sMinorChunk);
            // Loaded prefix entries (<= PC*chunks + 2) plus two scratch
            // rows of PC slots each for outPos/cnt, parked past the load
            // domain so the loaded span is never overwritten. Grids whose
            // group span does not fit fall back to the per-unit path: a
            // clamped span would read stale prefix words and write its
            // scratch rows past the buffer.
            const uint64_t spanElems = kPanelCols * chunksInit + 2 +
                                       2 * kPanelCols;
            panelSpanOk_ = spanElems <= kPfxSpanElems;
            pipe_.InitBuffer(pfxSpan_, kPfxSpanElems * sizeof(uint64_t));
            pipe_.InitBuffer(offSpan_, kPanelCols * sizeof(int32_t));
        }
        pipe_.InitBuffer(inQueue_, 2, tiling.tileLen * tiling.elementBytes);
        // int8 staging doubles: a full 32-column panel group (~5.4k nnz at
        // task density) then fits without the half-group split, whose 16B
        // block length the 2D gather form does not accept for E=1.
        stageElems_ = tiling.elementBytes == 1 ? 2 * kStageElems : kStageElems;
        pipe_.InitBuffer(outValQ_, 1, stageElems_ * tiling.elementBytes);
        pipe_.InitBuffer(outIdxQ_, 1, stageElems_ * sizeof(int32_t));
        if (tiling.format == kD2sFormatCoo) {
            pipe_.InitBuffer(outColQ_, 1, stageElems_ * sizeof(int32_t));
            pipe_.InitBuffer(cooPfx_, 8200);
        }
        pipe_.InitBuffer(statusBuf_, 16);
        pipe_.InitBuffer(scratch_, 32);
        pos_.Init(pipe_, prefixGm_, offsetsGm_, 0, 0, geo_.chunks,
            tiling.format, tiling.base);
    }

    __aicore__ inline void Process()
    {
        if (uBegin_ < uEnd_) {
            const int32_t status =
                LoadGmS32(statusGm_, 0, statusBuf_.Get<int32_t>());
            if (status != kD2sDeviceStatusSuccess) {
                return;
            }
            if (t_->format == kD2sFormatCoo && t_->order == kD2sOrderRow) {
                ConvertCoo();
            } else if (panel_ && panelSpanOk_) {
                ConvertPanels();
            } else if (colPanel_ && panelSpanOk_) {
                ConvertColPanels();
            } else if (t_->format == kD2sFormatCoo) {
                // COO over column-major storage without the panel path
                // (narrow lines): per-unit row extraction.
                for (uint64_t u = uBegin_; u < uEnd_; ++u) {
                    pos_.SetRange(u, u + 1);
                    pos_.Prime();
                    ConvertCooColUnit(u);
                }
            } else {
                pos_.SetRange(uBegin_, uEnd_);
                pos_.Prime();
                for (uint64_t u = uBegin_; u < uEnd_; ++u) {
                    RunUnitWithStaging(u);
                }
            }
        }
        // Dense-row interleave runs even on cores whose contiguous range is
        // empty: their carved units would otherwise be orphaned.
        if (colPanel_ && t_->format == kD2sFormatCoo) {
            const int32_t status =
                LoadGmS32(statusGm_, 0, statusBuf_.Get<int32_t>());
            if (status != kD2sDeviceStatusSuccess) {
                return;
            }
            for (uint64_t u = core_; u < t_->unitCount; u += t_->numBlocks) {
                if (!DenseUnitOf(u)) {
                    continue;
                }
                pos_.SetRange(u, u + 1);
                pos_.Prime();
                ConvertCooDenseUnit(u);
            }
        }
    }

private:
    // One unit through the shared staging pattern: allocate the output
    // pair, convert, and release the staging when nothing was written.
    __aicore__ inline void RunUnitWithStaging(uint64_t unit)
    {
        auto outVal = outValQ_.AllocTensor<T>();
        auto outIdx = outIdxQ_.AllocTensor<int32_t>();
        if (ConvertUnit(unit, outVal, outIdx) == 0) {
            outValQ_.FreeTensor(outVal);
            outIdxQ_.FreeTensor(outIdx);
        }
    }

    __aicore__ inline void FlushUnit(uint64_t outPos,
        LocalTensor<T> &outVal, LocalTensor<int32_t> &outIdx, uint32_t k)
    {
        PipeBarrier<PIPE_ALL>();
        outValQ_.EnQue(outVal);
        outIdxQ_.EnQue(outIdx);
        auto vd = outValQ_.DeQue<T>();
        auto id = outIdxQ_.DeQue<int32_t>();
        DataCopyPad(valuesGm_[outPos * kGmStride], vd,
            {1, (uint32_t)((uint64_t)k * elemBytes_), 0, 0, 0});
        DataCopyPad(indicesGm_[outPos], id,
            {1, (uint32_t)((uint64_t)k * sizeof(int32_t)), 0, 0, 0});
        PipeBarrier<PIPE_MTE3>();
        outValQ_.FreeTensor(vd);
        outIdxQ_.FreeTensor(id);
    }

    __aicore__ inline uint32_t ConvertUnit(uint64_t unit,
        LocalTensor<T> &outVal, LocalTensor<int32_t> &outIdx)
    {
        const uint64_t major = unit / geo_.chunks;
        const uint64_t begin = (unit - major * geo_.chunks) * kD2sMinorChunk;
        const uint64_t end = MinU64(begin + kD2sMinorChunk, geo_.minorDim);
        const uint64_t outPos = pos_.UnitOutPos(unit);
        uint32_t k = 0;
        if (geo_.contiguous) {
            for (uint64_t t0 = begin; t0 < end; t0 += t_->tileLen) {
                const uint32_t len = (uint32_t)MinU64(t_->tileLen, end - t0);
                ExtractContig(major * t_->ld + t0, len, t0, outVal, outIdx, k);
            }
        } else {
            const uint32_t slotElems = kSlotBytes / elemBytes_;
            const uint64_t fixedDim =
                t_->order == kD2sOrderRow ? t_->cols : t_->rows;
            if (fixedDim < slotElems ||
                (t_->ld - slotElems) * elemBytes_ >= 0xFFFFFFFFULL) {
                ConvertUnitScalar(major, begin, end, outVal, outIdx, k);
            } else {
                const uint32_t maxSlots =
                    t_->tileLen * elemBytes_ / kSlotBytes;
                const uint32_t winOff =
                    WindowOffset(major, slotElems, fixedDim);
                PrepareSlotWalkMask<T, kComplex>(bufs_, maxSlots, winOff);
                for (uint64_t t0 = begin; t0 < end;) {
                    const uint32_t slots =
                        (uint32_t)MinU64((uint64_t)maxSlots, end - t0);
                    ExtractSlots(major, t0, slots, winOff, outVal, outIdx, k);
                    t0 += slots;
                }
            }
        }
        if (k > 0) {
            FlushUnit(outPos, outVal, outIdx, k);
            return k;
        }
        return 0;
    }

    // Extreme-ld / tiny-line defense: per-element contiguous extraction.
    __aicore__ inline void ConvertUnitScalar(uint64_t major, uint64_t begin,
        uint64_t end, LocalTensor<T> &outVal, LocalTensor<int32_t> &outIdx,
        uint32_t &k)
    {
        const bool majorIsRow = t_->format != kD2sFormatCsc;
        for (uint64_t minor = begin; minor < end; ++minor) {
            const uint64_t gmElem = majorIsRow
                ? (t_->order == kD2sOrderRow
                       ? major * t_->ld + minor
                       : minor * t_->ld + major)
                : (t_->order == kD2sOrderRow
                       ? minor * t_->ld + major
                       : major * t_->ld + minor);
            ExtractContig(gmElem, 1, minor, outVal, outIdx, k);
        }
    }

    // Contiguous tile: bitmap judge + word-scan extraction.
    __aicore__ inline void ExtractContig(uint64_t gmElem, uint32_t len,
        uint64_t minorBegin, LocalTensor<T> &outVal,
        LocalTensor<int32_t> &outIdx, uint32_t &k)
    {
        LocalTensor<T> tileIn = LoadTileAndBuildBitmap<T, kGmStride,
            kComplex>(inQueue_, denseGm_, bufs_, gmElem, len, elemBytes_);
        auto mask = kComplex ? bufs_.Mask2() : bufs_.Mask1();
        BitmapScan scan;
        BitmapScanInit(scan, mask, len);
        uint32_t i = 0;
        while (BitmapScanNext(scan, len, i)) {
            CopyElem<T, kComplex>(outVal, k, tileIn, i);
            outIdx.SetValue(k, (int32_t)(minorBegin + i) + (int32_t)t_->base);
            ++k;
        }
        inQueue_.FreeTensor(tileIn);
    }

    // Strided slot group: same gather forms as the count side; extraction
    // reads the slot bitmaps (or the tile directly for small groups).
    __aicore__ inline void ExtractSlots(uint64_t major, uint64_t t0,
        uint32_t slots, uint32_t winOff, LocalTensor<T> &outVal,
        LocalTensor<int32_t> &outIdx, uint32_t &k)
    {
        auto tileIn = LoadSlotGroup<T, kGmStride, kComplex>(
            inQueue_, denseGm_, *t_, elemBytes_, major, t0, slots, winOff);
        const uint64_t tileBytes = (uint64_t)slots * kSlotBytes;
        if (tileBytes % 256ULL == 0 && tileBytes >= kSlotVecMinBytes) {
            CountSlots<T, kComplex>(bufs_, tileIn, slots, winOff);
            auto mask = bufs_.Mask1();
            for (uint32_t i = 0; i < slots; ++i) {
                if (SlotNonzero<T, kComplex>(mask, i, winOff)) {
                    EmitSlotElem(tileIn, i, t0, winOff, outVal, outIdx, k);
                }
            }
        } else {
            for (uint32_t i = 0; i < slots; ++i) {
                if (SlotElemNonzero<T, kComplex>(tileIn, i, winOff)) {
                    EmitSlotElem(tileIn, i, t0, winOff, outVal, outIdx, k);
                }
            }
        }
        inQueue_.FreeTensor(tileIn);
    }

    // Slot element emission shared by the bitmap and raw-test forms.
    __aicore__ inline void EmitSlotElem(const LocalTensor<T> &tile,
        uint32_t i, uint64_t t0, uint32_t winOff, LocalTensor<T> &outVal,
        LocalTensor<int32_t> &outIdx, uint32_t &k)
    {
        CopySlotElem(tile, i, winOff, outVal, k);
        outIdx.SetValue(k, (int32_t)(t0 + i) + (int32_t)t_->base);
        ++k;
    }

    __aicore__ inline void CopySlotElem(const LocalTensor<T> &tile,
        uint32_t i, uint32_t winOff, LocalTensor<T> &dst, uint32_t dstIdx)
    {
        constexpr uint32_t slotT = kSlotBytes / sizeof(T);
        if constexpr (kComplex) {
            dst.template ReinterpretCast<uint64_t>().SetValue(dstIdx,
                tile.template ReinterpretCast<uint64_t>().GetValue(
                    (slotT / 2U) * i));
        } else if constexpr (AscendC::IsSameType<T, int8_t>::value ||
                             AscendC::IsSameType<T, half>::value) {
            dst.SetValue(dstIdx, tile.GetValue(slotT * i + winOff));
        } else {
            dst.SetValue(dstIdx, tile.GetValue(slotT * i));
        }
    }

    // ---- CSC+ROW panel convert ----
    // Region starts are element-aligned to 32B so each per-column flush
    // source is MTE3-aligned. The staging tensor is a T view (T=float for
    // complex64), so a logical element occupies viewScale T lanes and every
    // view offset scales accordingly.
    static constexpr uint32_t ViewScale()
    {
        return kComplex ? 2 : 1;
    }

    __aicore__ inline void ConvertPanels()
    {
        const uint64_t chunks = geo_.chunks;
        const uint64_t fullEnd = t_->cols - t_->cols % kPanelCols;
        const uint64_t colLo = uBegin_ / chunks;
        const uint64_t colHi =
            MinU64(t_->cols, DivUp64(uEnd_, chunks));
        for (uint64_t c0 = colLo / kPanelCols * kPanelCols;
             c0 + kPanelCols <= fullEnd && c0 < colHi; c0 += kPanelCols) {
            for (uint64_t k = 0; k < chunks; ++k) {
                ConvertPanelGroup(c0, k, chunks);
            }
        }
        for (uint64_t ccol = colLo < fullEnd ? fullEnd : colLo; ccol < colHi;
             ++ccol) {
            for (uint64_t k = 0; k < chunks; ++k) {
                const uint64_t unit = ccol * chunks + k;
                if (unit < uBegin_ || unit >= uEnd_) {
                    continue;
                }
                pos_.SetRange(unit, unit + 1);
                pos_.Prime();
                RunUnitWithStaging(unit);
            }
        }
    }

    __aicore__ inline void ConvertPanelGroup(uint64_t c0, uint64_t k,
        uint64_t chunks)
    {
        // Per-unit metadata from one contiguous prefix span plus one
        // offsets span (loaded domains are never written back into).
        auto span = pfxSpan_.Get<uint64_t>();
        auto offBuf = offSpan_.Get<int32_t>();
        const uint64_t loadBase = c0 > 0 ? c0 * chunks - 1 : 0;
        const uint64_t spanLen = MinU64(
            MinU64(kPanelCols * chunks + 2,
                   (uint64_t)(kPfxSpanElems - 2)),
            t_->cols * chunks - loadBase);
        uint32_t ownedMask = 0;
        LoadPanelGroupMeta(c0, k, chunks, span, offBuf, loadBase, spanLen,
            ownedMask);
        uint64_t total = 0;
        for (uint32_t c = 0; c < kPanelCols; ++c) {
            if ((ownedMask >> c) & 1U) {
                total += span.GetValue((uint32_t)(spanLen + kPanelCols + c));
            }
        }
        if (total == 0) {
            return;
        }
        // Region alignment (elements) keeps every flush source 32B aligned.
        // elemBytes_ is one of 1/2/4/8; clamp guards a stray zero.
        const uint32_t safeElem = elemBytes_ > 0 ? elemBytes_ : 1;
        const uint32_t alignE = kSlotBytes / safeElem > 8
                                    ? kSlotBytes / safeElem
                                    : 8;
        if (total + (uint64_t)kPanelCols * alignE <= stageElems_) {
            ExtractPanelCols(c0, kPanelCols, k, ownedMask, span, spanLen,
                alignE);
            return;
        }
        ConvertPanelOverflow(c0, k, chunks, ownedMask, span, spanLen, alignE);
    }

    // Load the per-unit outPos/cnt metadata of one row-panel group into
    // the span scratch (parked past the loaded domain) and mark the
    // core-owned columns in ownedMask.
    __aicore__ inline void LoadPanelGroupMeta(uint64_t c0, uint64_t k,
        uint64_t chunks, LocalTensor<uint64_t> &span,
        LocalTensor<int32_t> &offBuf, uint64_t loadBase, uint64_t spanLen,
        uint32_t &ownedMask)
    {
        DataCopyPad(span, prefixGm_[loadBase],
            {1, (uint16_t)(spanLen * sizeof(uint64_t)), 0, 0},
            {false, 0, 0, 0});
        DataCopyPad(offBuf, offsetsGm_[c0],
            {1, (uint16_t)(kPanelCols * sizeof(int32_t)), 0, 0},
            {false, 0, 0, 0});
        PipeBarrier<PIPE_MTE2>();
        for (uint32_t c = 0; c < kPanelCols; ++c) {
            const uint64_t unit = (c0 + c) * chunks + k;
            if (unit < uBegin_ || unit >= uEnd_) {
                continue;
            }
            ownedMask |= 1U << c;
            const uint64_t hiIdx = unit - loadBase;
            const uint64_t before =
                unit == 0 ? 0
                          : span.GetValue((uint32_t)(hiIdx - 1));
            const uint64_t majorBefore =
                (c0 == 0 && c == 0)
                    ? 0
                    : span.GetValue(
                          (uint32_t)((c0 + c) * chunks - 1 - loadBase));
            const uint64_t cnt =
                span.GetValue((uint32_t)hiIdx) - before;
            const uint64_t outPos =
                (uint64_t)((int64_t)offBuf.GetValue(c) -
                           (int64_t)t_->base) +
                before - majorBefore;
            // outPos/cnt parked past the loaded domain (scratch slots).
            span.SetValue((uint32_t)(spanLen + c), outPos);
            span.SetValue((uint32_t)(spanLen + kPanelCols + c), cnt);
        }
    }

    // Dense panels split into two half groups; int8 half panels have a
    // 16B block length, which the 2D form does not accept. Units that
    // still do not fit fall back to the per-unit path.
    __aicore__ inline void ConvertPanelOverflow(uint64_t c0, uint64_t k,
        uint64_t chunks, uint32_t ownedMask, LocalTensor<uint64_t> &span,
        uint64_t spanLen, uint32_t alignE)
    {
        const bool halfOk = ((kPanelCols / 2) * elemBytes_) % 32U == 0;
        bool ok = halfOk;
        for (uint32_t h = 0; h < 2 && ok; ++h) {
            uint64_t half = 0;
            for (uint32_t c = h * (kPanelCols / 2);
                 c < (h + 1) * (kPanelCols / 2); ++c) {
                if ((ownedMask >> c) & 1U) {
                    half += span.GetValue(
                        (uint32_t)(spanLen + kPanelCols + c));
                }
            }
            if (half + (uint64_t)(kPanelCols / 2) * alignE <= stageElems_) {
                // ownedMask stays group-absolute: ExtractPanelCols applies
                // its own (c0 % kPanelCols) shift for the half base.
                ExtractPanelCols(c0 + h * (kPanelCols / 2), kPanelCols / 2,
                    k, ownedMask, span, spanLen, alignE);
            } else {
                ok = false;
            }
        }
        if (!ok) {
            for (uint32_t c = 0; c < kPanelCols; ++c) {
                if ((ownedMask >> c) & 1U) {
                    const uint64_t unit = (c0 + c) * chunks + k;
                    pos_.SetRange(unit, unit + 1);
                    pos_.Prime();
                    RunUnitWithStaging(unit);
                }
            }
        }
    }

    __aicore__ inline void ExtractPanelCols(uint64_t c0, uint32_t nCols,
        uint64_t k, uint32_t ownedMask, LocalTensor<uint64_t> &span,
        uint64_t spanLen, uint32_t alignE)
    {
        const uint32_t rowsPerTile = t_->tileLen / kPanelCols;
        // Column-region starts in elements, each 32B aligned.
        uint32_t colStart[kPanelCols];
        uint32_t fill[kPanelCols];
        uint32_t run = 0;
        for (uint32_t c = 0; c < nCols; ++c) {
            const uint32_t ownedBit =
                (ownedMask >> (c0 % kPanelCols + c)) & 1U;
            const uint32_t cnt =
                ownedBit
                    ? (uint32_t)span.GetValue(
                          (uint32_t)(spanLen + kPanelCols + c0 % kPanelCols +
                                     c))
                    : 0U;
            colStart[c] = run;
            fill[c] = 0U;
            run += cnt;
            run = AlignUpU32(run, alignE);
        }
        auto outVal = outValQ_.AllocTensor<T>();
        auto outIdx = outIdxQ_.AllocTensor<int32_t>();
        const uint64_t r1 = MinU64((k + 1) * kD2sMinorChunk, t_->rows);
        for (uint64_t t0 = k * kD2sMinorChunk; t0 < r1; t0 += rowsPerTile) {
            const uint32_t rows =
                (uint32_t)MinU64((uint64_t)rowsPerTile, r1 - t0);
            EmitPanelTile(c0, nCols, t0, rows, ownedMask, outVal, outIdx,
                colStart, fill);
        }
        // Per-column flush. The staging tensor is a T view: a logical
        // element occupies viewScale lanes, so region starts scale.
        FlushPanelRegions(c0, nCols, span, spanLen, outVal, outIdx, colStart,
            fill);
    }

    // One staged tile of a row-panel group: bitmap word scan with the
    // per-column region scatter.
    // One staged 2D tile of a row-panel group through the input queue
    // with its zero bitmap built.
    __aicore__ inline LocalTensor<T> LoadPanelTile(uint64_t gmElem,
        uint32_t rows, uint32_t nCols)
    {
        auto tile = inQueue_.AllocTensor<T>();
        DataCopyPad(tile[0], denseGm_[gmElem * kGmStride],
            {(uint16_t)rows, nCols * elemBytes_,
                (uint32_t)((t_->ld - nCols) * elemBytes_), 0, 0},
            {false, 0, 0, 0});
        inQueue_.EnQue(tile);
        auto tileIn = inQueue_.DeQue<T>();
        BuildZeroBitmap<T, kComplex>(bufs_, tileIn, rows * nCols);
        return tileIn;
    }

    __aicore__ inline void EmitPanelTile(uint64_t c0, uint32_t nCols,
        uint64_t t0, uint32_t rows, uint32_t ownedMask,
        LocalTensor<T> &outVal, LocalTensor<int32_t> &outIdx,
        uint32_t *colStart, uint32_t *fill)
    {
        auto tileIn = LoadPanelTile(t0 * t_->ld + c0, rows, nCols);
        auto mask = kComplex ? bufs_.Mask2() : bufs_.Mask1();
        auto maskW = mask.template ReinterpretCast<uint32_t>();
        const uint32_t rowsPerWord = nCols != 0U ? 32U / nCols : 32U;
        const uint32_t nWords = (rows + rowsPerWord - 1) / rowsPerWord;
        const uint32_t ownedWindow =
            ownedMask >> (c0 % kPanelCols);
        for (uint32_t wI = 0; wI < nWords; ++wI) {
            const uint32_t bitsW = maskW.GetValue(wI);
            if (bitsW == 0xFFFFFFFFU) {
                continue;
            }
            for (uint32_t rr = 0; rr < rowsPerWord; ++rr) {
                const uint32_t i = wI * rowsPerWord + rr;
                if (i >= rows) {
                    break;
                }
                const uint32_t w = nCols == 32
                                       ? bitsW
                                       : (bitsW >> (rr * nCols)) &
                                             (0xFFFFU >> (16 - nCols));
                uint32_t nzBits = (~w) & ownedWindow;
                if (nCols < 32) {
                    nzBits &= 0xFFFFU >> (16 - nCols);
                }
                while (nzBits != 0U) {
                    const uint32_t low = nzBits & (~nzBits + 1U);
                    const uint32_t tB =
                        (uint32_t)__builtin_ctz((int)low);
                    const uint32_t dst = colStart[tB] + fill[tB];
                    ++fill[tB];
                    CopyElem<T, kComplex>(outVal, dst, tileIn,
                        i * nCols + tB);
                    outIdx.SetValue(dst,
                        (int32_t)(t0 + i) + (int32_t)t_->base);
                    nzBits ^= low;
                }
            }
        }
        inQueue_.FreeTensor(tileIn);
    }

    // Flush each column's staged region to its output run.
    __aicore__ inline void FlushPanelRegions(uint64_t c0, uint32_t nCols,
        LocalTensor<uint64_t> &span, uint64_t spanLen,
        LocalTensor<T> &outVal, LocalTensor<int32_t> &outIdx,
        uint32_t *colStart, uint32_t *fill)
    {
        const uint32_t viewScale = ViewScale();
        PipeBarrier<PIPE_ALL>();
        outValQ_.EnQue(outVal);
        outIdxQ_.EnQue(outIdx);
        auto outValD = outValQ_.DeQue<T>();
        auto outIdxD = outIdxQ_.DeQue<int32_t>();
        for (uint32_t c = 0; c < nCols; ++c) {
            const uint32_t f = fill[c];
            if (f == 0) {
                continue;
            }
            const uint32_t s = colStart[c];
            const uint64_t op = span.GetValue(
                (uint32_t)(spanLen + c0 % kPanelCols + c));
            DataCopyPad(valuesGm_[op * kGmStride], outValD[s * viewScale],
                {1, (uint32_t)((uint64_t)f * elemBytes_), 0, 0, 0});
            DataCopyPad(indicesGm_[op], outIdxD[s],
                {1, (uint32_t)((uint64_t)f * sizeof(int32_t)), 0, 0, 0});
        }
        PipeBarrier<PIPE_MTE3>();
        outValQ_.FreeTensor(outValD);
        outIdxQ_.FreeTensor(outIdxD);
    }

    // ---- CSR+COL / COO+COL column-panel convert (mirror of the CSC+ROW
    // row-panel family). Panel group = 32 adjacent rows; a group's tiles
    // walk the minor (column) direction with one 32-row contiguous block
    // per column. CSR writes per-row regions of col indices; COO writes
    // the flat row-major (row, col) stream.
    __aicore__ inline void ConvertColPanels()
    {
        const uint64_t chunks = geo_.chunks;
        const uint64_t fullEnd = t_->rows - t_->rows % kPanelCols;
        const uint64_t rowLo = uBegin_ / chunks;
        const uint64_t rowHi =
            MinU64(t_->rows, DivUp64(uEnd_, chunks));
        for (uint64_t r0 = rowLo / kPanelCols * kPanelCols;
             r0 + kPanelCols <= fullEnd && r0 < rowHi; r0 += kPanelCols) {
            for (uint64_t k = 0; k < chunks; ++k) {
                ConvertColPanelGroup(r0, k, chunks);
            }
        }
        for (uint64_t rrow = rowLo < fullEnd ? fullEnd : rowLo; rrow < rowHi;
             ++rrow) {
            for (uint64_t k = 0; k < chunks; ++k) {
                const uint64_t unit = rrow * chunks + k;
                if (unit < uBegin_ || unit >= uEnd_) {
                    continue;
                }
                if (t_->format == kD2sFormatCoo && DenseUnitOf(unit)) {
                    continue; // owned by the interleaved pass in Process
                }
                pos_.SetRange(unit, unit + 1);
                pos_.Prime();
                if (t_->format == kD2sFormatCsr) {
                    RunUnitWithStaging(unit);
                } else {
                    ConvertCooColUnit(unit);
                }
            }
        }
    }

    __aicore__ inline void ConvertColPanelGroup(uint64_t r0, uint64_t k,
        uint64_t chunks)
    {
        auto span = pfxSpan_.Get<uint64_t>();
        auto offBuf = offSpan_.Get<int32_t>();
        const uint64_t loadBase = r0 > 0 ? r0 * chunks - 1 : 0;
        const uint64_t spanLen = MinU64(
            MinU64(kPanelCols * chunks + 2,
                   (uint64_t)(kPfxSpanElems - 2)),
            t_->rows * chunks - loadBase);
        uint32_t ownedMask = 0;
        LoadColPanelGroupMeta(r0, k, chunks, span, offBuf, loadBase, spanLen,
            ownedMask);
        uint64_t total = 0;
        for (uint32_t j = 0; j < kPanelCols; ++j) {
            if ((ownedMask >> j) & 1U) {
                total += span.GetValue((uint32_t)(spanLen + kPanelCols + j));
            }
        }
        if (total == 0) {
            return;
        }
        // elemBytes_ is one of 1/2/4/8; clamp guards a stray zero.
        const uint32_t safeElem = elemBytes_ > 0 ? elemBytes_ : 1;
        const uint32_t alignE = kSlotBytes / safeElem > 8
                                    ? kSlotBytes / safeElem
                                    : 8;
        // Incremental flush keeps every group inside staging regardless of
        // density: no half-split, no per-unit fallback.
        ExtractColPanelRows(r0, kPanelCols, k, ownedMask, span, spanLen,
            alignE);
    }

    // Carve test for dense COO row chunks leaving the shared column
    // panel (must mirror DenseUnitOf exactly or units drop).
    __aicore__ inline bool CooDenseCarveUnit(uint64_t cnt, uint64_t k)
    {
        return t_->format == kD2sFormatCoo && !kComplex &&
               (t_->ld - 1) * elemBytes_ < 0xFFFFFFFFULL &&
               t_->rows >= kSlotBytes / elemBytes_ &&
               DenseUnitP(cnt,
                   MinU64(kD2sMinorChunk,
                          geo_.minorDim - k * kD2sMinorChunk),
                   elemBytes_);
    }

    // Load the per-unit outPos/cnt metadata of one column-panel group;
    // dense COO rows stay out of ownedMask (they run on the interleaved
    // direct path and must mirror DenseUnitOf exactly or units drop).
    __aicore__ inline void LoadColPanelGroupMeta(uint64_t r0, uint64_t k,
        uint64_t chunks, LocalTensor<uint64_t> &span,
        LocalTensor<int32_t> &offBuf, uint64_t loadBase, uint64_t spanLen,
        uint32_t &ownedMask)
    {
        DataCopyPad(span, prefixGm_[loadBase],
            {1, (uint16_t)(spanLen * sizeof(uint64_t)), 0, 0},
            {false, 0, 0, 0});
        if (t_->format == kD2sFormatCsr) {
            DataCopyPad(offBuf, offsetsGm_[r0],
                {1, (uint16_t)(kPanelCols * sizeof(int32_t)), 0, 0},
                {false, 0, 0, 0});
        }
        PipeBarrier<PIPE_MTE2>();
        for (uint32_t j = 0; j < kPanelCols; ++j) {
            const uint64_t unit = (r0 + j) * chunks + k;
            if (unit < uBegin_ || unit >= uEnd_) {
                continue;
            }
            const uint64_t hiIdx = unit - loadBase;
            const uint64_t before =
                unit == 0 ? 0
                          : span.GetValue((uint32_t)(hiIdx - 1));
            const uint64_t cnt =
                span.GetValue((uint32_t)hiIdx) - before;
            if (CooDenseCarveUnit(cnt, k)) {
                // Dense rows leave the shared panel for the
                // interleaved direct path (rebalance + run vectors).
                continue;
            }
            ownedMask |= 1U << j;
            uint64_t outPos;
            if (t_->format == kD2sFormatCsr) {
                const uint64_t majorBefore =
                    (r0 == 0 && j == 0)
                        ? 0
                        : span.GetValue(
                              (uint32_t)((r0 + j) * chunks - 1 -
                                         loadBase));
                outPos = (uint64_t)((int64_t)offBuf.GetValue(j) -
                                    (int64_t)t_->base) +
                         before - majorBefore;
            } else {
                // COO: unit streams row (r0+j)'s chunk in ascending
                // column order; the flat row-major position is the
                // prefix before the unit.
                outPos = before;
            }
            span.SetValue((uint32_t)(spanLen + j), outPos);
            span.SetValue((uint32_t)(spanLen + kPanelCols + j), cnt);
        }
    }

    __aicore__ inline void ExtractColPanelRows(uint64_t r0, uint32_t nRows,
        uint64_t k, uint32_t ownedMask, LocalTensor<uint64_t> &span,
        uint64_t spanLen, uint32_t alignE)
    {
        const uint32_t colsPerTile = t_->tileLen / kPanelCols;
        const uint32_t viewScale = ViewScale();
        const bool isCoo = t_->format == kD2sFormatCoo;
        // Fixed-capacity per-row regions with incremental flush: one tile
        // emits at most colsPerTile elements per row, so a region of that
        // size (alignment-padded) never overflows between flushes and the
        // total staging stays nRows * rowCap <= tileLen elements for every
        // dtype -- dense groups stream out instead of falling back.
        const uint32_t rowCap = AlignUpU32(colsPerTile, alignE);
        int32_t op[kPanelCols];
        uint32_t fill[kPanelCols];
        uint32_t flushed[kPanelCols];
        for (uint32_t j = 0; j < nRows; ++j) {
            const uint32_t ownedBit =
                (ownedMask >> (r0 % kPanelCols + j)) & 1U;
            op[j] = ownedBit
                        ? (int32_t)span.GetValue(
                              (uint32_t)(spanLen + r0 % kPanelCols + j))
                        : -1;
            fill[j] = 0U;
            flushed[j] = 0U;
        }
        auto outVal = outValQ_.AllocTensor<T>();
        auto outCol = outIdxQ_.AllocTensor<int32_t>();
        LocalTensor<int32_t> outRow;
        if (isCoo) {
            outRow = outColQ_.AllocTensor<int32_t>();
        }
        const uint32_t flushMark = stageElems_ / 2;
        uint64_t staged = 0;
        const uint64_t c1 = MinU64((k + 1) * kD2sMinorChunk, t_->cols);
        for (uint64_t t0 = k * kD2sMinorChunk; t0 < c1; t0 += colsPerTile) {
            const uint32_t cols =
                (uint32_t)MinU64((uint64_t)colsPerTile, c1 - t0);
            staged += EmitColPanelTile(r0, nRows, t0, cols, ownedMask,
                outVal, outCol, outRow, rowCap, fill);
            if (ColPanelNeedsFlush(fill, nRows, colsPerTile, rowCap,
                                   staged, flushMark)) {
                FlushColPanelRows(outVal, outCol, outRow, op, fill,
                    flushed, rowCap, nRows, isCoo, viewScale);
                staged = 0;
            }
        }
        FlushColPanelRows(outVal, outCol, outRow, op, fill, flushed,
            rowCap, nRows, isCoo, viewScale);
        outValQ_.FreeTensor(outVal);
        outIdxQ_.FreeTensor(outCol);
        if (isCoo) {
            outColQ_.FreeTensor(outRow);
        }
    }

    // Flush decision: total volume OR any single row could overflow its
    // region on the next tile (one tile adds at most colsPerTile
    // elements to a row).
    __aicore__ inline bool ColPanelNeedsFlush(const uint32_t *fill,
        uint32_t nRows, uint32_t colsPerTile, uint32_t rowCap,
        uint64_t staged, uint32_t flushMark)
    {
        if (staged >= flushMark) {
            return true;
        }
        for (uint32_t j = 0; j < nRows; ++j) {
            if (fill[j] + colsPerTile > rowCap) {
                return true;
            }
        }
        return false;
    }

    // One staged tile of a column-panel group; returns the element count
    // staged into the per-row regions. Tile element (column m, row j)
    // sits at index m*nRows + j: one 32-bit word covers 32/nRows columns.
    // Column-panel tile emission context shared by the word-scan helper.
    struct ColPanelEmit {
        LocalTensor<T> outVal;
        LocalTensor<int32_t> outCol;
        LocalTensor<int32_t> outRow;
        LocalTensor<T> tileIn;
        uint32_t *fill;
        uint64_t staged;
        uint32_t rowCap;
        uint32_t nRows;
        uint32_t ownedWindow;
        uint64_t t0;
        uint64_t r0;
        bool isCoo;
    };

    // One 32-bit bitmap word: emit the nonzero, owned elements of its
    // minorsPerWord columns into the per-row regions.
    __aicore__ inline void EmitColPanelWord(ColPanelEmit &e, uint32_t bitsW,
        uint32_t wI, uint32_t minorsPerWord, uint32_t cols)
    {
        for (uint32_t q = 0; q < minorsPerWord; ++q) {
            const uint32_t m = wI * minorsPerWord + q;
            if (m >= cols) {
                break;
            }
            const uint32_t w =
                e.nRows == 32
                    ? bitsW
                    : (bitsW >> (q * e.nRows)) &
                          (0xFFFFU >> (16 - e.nRows));
            uint32_t nzBits = (~w) & e.ownedWindow;
            if (e.nRows < 32) {
                nzBits &= 0xFFFFU >> (16 - e.nRows);
            }
            while (nzBits != 0U) {
                const uint32_t low = nzBits & (~nzBits + 1U);
                const uint32_t tB =
                    (uint32_t)__builtin_ctz((int)low);
                const uint32_t dst = tB * e.rowCap + e.fill[tB];
                ++e.fill[tB];
                ++e.staged;
                CopyElem<T, kComplex>(e.outVal, dst, e.tileIn,
                    m * e.nRows + tB);
                e.outCol.SetValue(dst,
                    (int32_t)(e.t0 + m) + (int32_t)t_->base);
                if (e.isCoo) {
                    e.outRow.SetValue(dst,
                        (int32_t)(e.r0 + tB) + (int32_t)t_->base);
                }
                nzBits ^= low;
            }
        }
    }

    __aicore__ inline uint64_t EmitColPanelTile(uint64_t r0, uint32_t nRows,
        uint64_t t0, uint32_t cols, uint32_t ownedMask,
        LocalTensor<T> &outVal, LocalTensor<int32_t> &outCol,
        LocalTensor<int32_t> &outRow, uint32_t rowCap, uint32_t *fill)
    {
        ColPanelEmit e;
        e.outVal = outVal;
        e.outCol = outCol;
        e.outRow = outRow;
        e.fill = fill;
        e.staged = 0;
        e.rowCap = rowCap;
        e.nRows = nRows;
        e.ownedWindow = ownedMask >> (r0 % kPanelCols);
        e.t0 = t0;
        e.r0 = r0;
        e.isCoo = t_->format == kD2sFormatCoo;
        auto tile = inQueue_.AllocTensor<T>();
        DataCopyPad(tile[0], denseGm_[(t0 * t_->ld + r0) * kGmStride],
            {(uint16_t)cols, nRows * elemBytes_,
                (uint32_t)((t_->ld - nRows) * elemBytes_), 0, 0},
            {false, 0, 0, 0});
        inQueue_.EnQue(tile);
        e.tileIn = inQueue_.DeQue<T>();
        BuildZeroBitmap<T, kComplex>(bufs_, e.tileIn, cols * nRows);
        auto mask = kComplex ? bufs_.Mask2() : bufs_.Mask1();
        auto maskW = mask.template ReinterpretCast<uint32_t>();
        const uint32_t minorsPerWord = nRows != 0U ? 32U / nRows : 32U;
        const uint32_t nWords =
            (cols + minorsPerWord - 1) / minorsPerWord;
        for (uint32_t wI = 0; wI < nWords; ++wI) {
            const uint32_t bitsW = maskW.GetValue(wI);
            if (bitsW != 0xFFFFFFFFU) {
                EmitColPanelWord(e, bitsW, wI, minorsPerWord, cols);
            }
        }
        inQueue_.FreeTensor(e.tileIn);
        return e.staged;
    }

    // (tail free: the final FlushColPanelRows call may have nothing staged;
    // the tensors it re-allocated above are released here.)

    // Flush every row's staged prefix to its output run and reset the
    // region; row j's stream continues at op[j] + flushed[j]. The staging
    // tensors complete their EnQue/DeQue/FreeTensor lifecycle here and
    // fresh ones are allocated for the next batch (a single-buffer VECOUT
    // queue must never see two EnQue cycles on one handle).
    __aicore__ inline void FlushColPanelRows(LocalTensor<T> &outVal,
        LocalTensor<int32_t> &outCol, LocalTensor<int32_t> &outRow,
        int32_t *op, uint32_t *fill, uint32_t *flushed, uint32_t rowCap,
        uint32_t nRows, bool isCoo, uint32_t viewScale)
    {
        bool any = false;
        for (uint32_t j = 0; j < nRows; ++j) {
            any = any || (fill[j] > 0 && op[j] >= 0);
        }
        if (!any) {
            return;
        }
        PipeBarrier<PIPE_ALL>();
        outValQ_.EnQue(outVal);
        outIdxQ_.EnQue(outCol);
        auto outValD = outValQ_.DeQue<T>();
        auto outColD = outIdxQ_.DeQue<int32_t>();
        LocalTensor<int32_t> outRowD;
        if (isCoo) {
            outColQ_.EnQue(outRow);
            outRowD = outColQ_.DeQue<int32_t>();
        }
        for (uint32_t j = 0; j < nRows; ++j) {
            StoreColPanelRow(outValD, outColD, outRowD, op, fill, flushed,
                rowCap, j, isCoo, viewScale);
        }
        PipeBarrier<PIPE_MTE3>();
        outValQ_.FreeTensor(outValD);
        outIdxQ_.FreeTensor(outColD);
        if (isCoo) {
            outColQ_.FreeTensor(outRowD);
        }
        outVal = outValQ_.AllocTensor<T>();
        outCol = outIdxQ_.AllocTensor<int32_t>();
        if (isCoo) {
            outRow = outColQ_.AllocTensor<int32_t>();
        }
    }

    // One row's staged prefix to its output run; the row's stream
    // continues at op[j] + flushed[j].
    __aicore__ inline void StoreColPanelRow(LocalTensor<T> &outValD,
        LocalTensor<int32_t> &outColD, LocalTensor<int32_t> &outRowD,
        int32_t *op, uint32_t *fill, uint32_t *flushed, uint32_t rowCap,
        uint32_t j, bool isCoo, uint32_t viewScale)
    {
        const uint32_t f = fill[j];
        if (f == 0 || op[j] < 0) {
            fill[j] = 0;
            return;
        }
        const uint64_t outPos = (uint64_t)op[j] + flushed[j];
        DataCopyPad(valuesGm_[outPos * kGmStride],
            outValD[j * rowCap * viewScale],
            {1, (uint32_t)((uint64_t)f * elemBytes_), 0, 0, 0});
        if (!isCoo) {
            DataCopyPad(indicesGm_[outPos], outColD[j * rowCap],
                {1, (uint32_t)((uint64_t)f * sizeof(int32_t)), 0, 0,
                 0});
        } else {
            DataCopyPad(rowGm_[outPos], outRowD[j * rowCap],
                {1, (uint32_t)((uint64_t)f * sizeof(int32_t)), 0, 0,
                 0});
            DataCopyPad(colGm_[outPos], outColD[j * rowCap],
                {1, (uint32_t)((uint64_t)f * sizeof(int32_t)), 0, 0,
                 0});
        }
        flushed[j] += f;
        fill[j] = 0;
    }

    // COO+COL per-unit fallback (tail rows / panel overflow): strided
    // slot extraction writing the flat (row, col) stream; the row index
    // is constant across the unit.
    __aicore__ inline void ConvertCooColUnit(uint64_t unit)
    {
        const uint64_t major = unit / geo_.chunks;
        const uint64_t begin = (unit - major * geo_.chunks) * kD2sMinorChunk;
        const uint64_t end = MinU64(begin + kD2sMinorChunk, geo_.minorDim);
        const uint64_t outPos = pos_.UnitOutPos(unit);
        auto outVal = outValQ_.AllocTensor<T>();
        auto outRow = outIdxQ_.AllocTensor<int32_t>();
        auto outCol = outColQ_.AllocTensor<int32_t>();
        uint32_t k = 0;
        const uint32_t slotElems = kSlotBytes / elemBytes_;
        const uint64_t fixedDim = t_->rows;
        if (fixedDim < slotElems ||
            (t_->ld - slotElems) * elemBytes_ >= 0xFFFFFFFFULL) {
            for (uint64_t minor = begin; minor < end; ++minor) {
                ExtractContig(minor * t_->ld + major, 1, minor, outVal,
                              outCol, k);
            }
        } else {
            const uint32_t maxSlots = t_->tileLen * elemBytes_ / kSlotBytes;
            const uint32_t winOff = WindowOffset(major, slotElems, fixedDim);
            PrepareSlotWalkMask<T, kComplex>(bufs_, maxSlots, winOff);
            for (uint64_t t0 = begin; t0 < end;) {
                const uint32_t slots =
                    (uint32_t)MinU64((uint64_t)maxSlots, end - t0);
                ExtractSlots(major, t0, slots, winOff, outVal, outCol, k);
                t0 += slots;
            }
        }
        if (k > 0) {
            for (uint32_t i = 0; i < k; ++i) {
                outRow.SetValue(i, (int32_t)major + (int32_t)t_->base);
            }
            FlushCoo(outVal, outRow, outCol, outPos, k);
        } else {
            ReleaseCooStaging(outVal, outRow, outCol);
        }
    }

    // Carve test through the GM prefix (tail rows and the interleaved pass
    // have no panel span handy).
    __aicore__ inline bool DenseUnitOf(uint64_t unit)
    {
        // complex64 keeps the shared panel path: its paired lanes defeat
        // the vector run form (odd lane never 32B-aligned) and the scalar
        // direct path showed vector-core instability at dense scale.
        if constexpr (kComplex) {
            return false;
        }
        if (t_->format != kD2sFormatCoo || elemBytes_ == 1 ||
            (t_->ld - 1) * elemBytes_ >= 0xFFFFFFFFULL ||
            t_->rows < kSlotBytes / elemBytes_) {
            return false;
        }
        const uint64_t major = unit / geo_.chunks;
        const uint64_t mBegin = (unit - major * geo_.chunks) * kD2sMinorChunk;
        const uint64_t cnt = pos_.PrefixAt(unit) -
            (unit > 0 ? pos_.PrefixAt(unit - 1) : 0ULL);
        return DenseUnitP(cnt,
            MinU64(kD2sMinorChunk, geo_.minorDim - mBegin), elemBytes_);
    }

    // Column ramp ramp[i] = i and gather byte ramp stride[i] = i*32 (the
    // vgather offset table and base offset are byte-granular; element i of
    // the slot tile sits at byte 32*i for every dtype). Built once in the
    // COO prefix scratch buffer (the ROW-order COO path that owns it never
    // runs in the same launch). Two 1024-entry ramps fit the 8200 bytes.
    __aicore__ inline void BuildRamp()
    {
        if (rampBuilt_) {
            return;
        }
        auto base = cooPfx_.Get<int32_t>();
        auto ramp = base[0];
        auto strideI = base[kRampElems];
        for (uint32_t i = 0; i < 8; ++i) {
            ramp.SetValue(i, (int32_t)i);
            strideI.SetValue(i, (int32_t)(i * kSlotBytes));
        }
        for (uint32_t len = 8; len < kRampElems; len <<= 1) {
            const int32_t m = (int32_t)MinU64(len, kRampElems - len);
            Adds(ramp[len], ramp, (int32_t)len, m);
            Adds(strideI[len], strideI,
                (int32_t)(len * kSlotBytes), m);
        }
        PipeBarrier<PIPE_V>();
        rampBuilt_ = true;
    }

    __aicore__ inline void AppendRampCols(LocalTensor<int32_t> &outCol,
        uint32_t k, int32_t firstCol, uint32_t len)
    {
        auto ramp = cooPfx_.Get<int32_t>()[0];
        for (uint32_t c = 0; c < len; c += kRampElems) {
            const uint32_t m = (uint32_t)MinU64(kRampElems, len - c);
            Adds(outCol[k + c], ramp,
                (int32_t)(firstCol + (int64_t)c), (int32_t)m);
        }
    }

    // ---- COO+COL dense-row direct path ----
    // Windows load with the proven slot-gather forms (fp32 padded lane-0
    // slots, half windows at winOff); a contiguous nonzero run then stages
    // its values with one per-lane Gather (bit-exact by construction),
    // its column indices with one ramp Adds, and short runs keep the
    // per-element scalar form. Runs close at their terminating zero, so
    // emission stays in ascending column order.
    __aicore__ inline void ConvertCooDenseUnit(uint64_t unit)
    {
        const uint64_t major = unit / geo_.chunks;
        const uint64_t begin = (unit - major * geo_.chunks) * kD2sMinorChunk;
        const uint64_t end = MinU64(begin + kD2sMinorChunk, geo_.minorDim);
        const uint64_t outPos = pos_.UnitOutPos(unit);
        constexpr uint32_t slotElems = kSlotBytes / sizeof(T);
        const uint32_t maxSlots = t_->tileLen * elemBytes_ / kSlotBytes;
        const uint64_t fixedDim = t_->rows;
        const uint32_t winOff = WindowOffset(major, slotElems, fixedDim);
        BuildRamp();
        auto stride = cooPfx_.Get<int32_t>()[kRampElems]
                          .template ReinterpretCast<uint32_t>();
        auto outVal = outValQ_.AllocTensor<T>();
        auto outRow = outIdxQ_.AllocTensor<int32_t>();
        auto outCol = outColQ_.AllocTensor<int32_t>();
        uint32_t k = 0;
        for (uint64_t t0 = begin; t0 < end;) {
            const uint32_t slots =
                (uint32_t)MinU64((uint64_t)maxSlots, end - t0);
            auto tileIn = LoadSlotGroup<T, kGmStride, kComplex>(
                inQueue_, denseGm_, *t_, elemBytes_, major, t0, slots,
                winOff);
            uint32_t runStart = 0xFFFFFFFFU;
            for (uint32_t i = 0; i <= slots; ++i) {
                if (i < slots &&
                    SlotElemNonzero<T, kComplex>(tileIn, i, winOff)) {
                    if (runStart == 0xFFFFFFFFU) {
                        runStart = i;
                    }
                    continue;
                }
                if (runStart == 0xFFFFFFFFU) {
                    continue;
                }
                // Runs close at their terminating zero, so emission
                // stays in ascending column order.
                EmitDenseRun(tileIn, runStart, i - runStart, t0, winOff,
                    stride, outVal, outCol, k);
                runStart = 0xFFFFFFFFU;
            }
            inQueue_.FreeTensor(tileIn);
            t0 += slots;
        }
        if (k > 0) {
            Duplicate(outRow, (int32_t)(major + t_->base), (int32_t)k);
            PipeBarrier<PIPE_V>();
            FlushCoo(outVal, outRow, outCol, outPos, k);
        } else {
            ReleaseCooStaging(outVal, outRow, outCol);
        }
    }

    // Release an unused COO staging triple.
    __aicore__ inline void ReleaseCooStaging(LocalTensor<T> &outVal,
        LocalTensor<int32_t> &outRow, LocalTensor<int32_t> &outCol)
    {
        outValQ_.FreeTensor(outVal);
        outIdxQ_.FreeTensor(outRow);
        outColQ_.FreeTensor(outCol);
    }

    // Scalar per-element emission of a run slice.
    __aicore__ inline void EmitDenseRunScalar(const LocalTensor<T> &tile,
        uint32_t runStart, uint32_t len, uint64_t t0, uint32_t winOff,
        LocalTensor<T> &outVal, LocalTensor<int32_t> &outCol, uint32_t &k)
    {
        for (uint32_t e = runStart; e < runStart + len; ++e) {
            CopySlotElem(tile, e, winOff, outVal, k);
            outCol.SetValue(k,
                (int32_t)(t0 + e) + (int32_t)t_->base);
            ++k;
        }
    }

    // One contiguous nonzero run of a dense row: long runs stage values
    // with one per-lane Gather (bit-exact by construction) and column
    // indices with one ramp Adds; short runs keep the scalar form.
    // complex64's paired lanes can never both land on 32B boundaries, so
    // c64 keeps the scalar form (the carve + interleave balance still
    // applies). Vector dst bases must stay 32B-aligned: a preceding
    // scalar run can leave k off the boundary, so absorb the
    // misalignment with a scalar head before the Gather.
    __aicore__ inline void EmitDenseRun(const LocalTensor<T> &tile,
        uint32_t runStart, uint32_t len, uint64_t t0, uint32_t winOff,
        const LocalTensor<uint32_t> &stride, LocalTensor<T> &outVal,
        LocalTensor<int32_t> &outCol, uint32_t &k)
    {
        if (len < kRunVecMin || kComplex ||
            AscendC::IsSameType<T, int8_t>::value) {
            EmitDenseRunScalar(tile, runStart, len, t0, winOff, outVal,
                outCol, k);
            return;
        }
        const uint32_t alignE = 32U / elemBytes_;
        const uint32_t pad = (alignE - k % alignE) % alignE;
        const uint32_t head = pad < len ? pad : len;
        EmitDenseRunScalar(tile, runStart, head, t0, winOff, outVal, outCol,
            k);
        runStart += head;
        const uint32_t len2 = len - head;
        if (len2 >= kRunVecMin) {
            AppendRampCols(outCol, k,
                (int32_t)(t0 + runStart + t_->base), len2);
            // Byte-granular gather: element e lives at byte
            // 32*(slot index e) of the tile (fp32 lane 0, half at
            // +2*winOff).
            const uint32_t byteBase = runStart * kSlotBytes +
                (AscendC::IsSameType<T, half>::value
                     ? 2U * winOff
                     : 0U);
            for (uint32_t c = 0; c < len2; c += kRampElems) {
                const uint32_t m = (uint32_t)MinU64(
                    kRampElems, len2 - c);
                Gather(outVal[k + c], tile, stride,
                    byteBase + c * kSlotBytes, m);
            }
            PipeBarrier<PIPE_V>();
            k += len2;
        } else {
            EmitDenseRunScalar(tile, runStart, len2, t0, winOff, outVal,
                outCol, k);
        }
    }

    // ---- COO convert ----
    __aicore__ inline void ConvertCoo()
    {
        const uint64_t logical = t_->rows * t_->cols;
        const uint64_t uEnd = MinU64(uEnd_, DivUp64(logical, kD2sCooTile));
        if (uBegin_ >= uEnd) {
            return;
        }
        auto meta = scratch_.Get<uint64_t>();
        auto pfxSpan = cooPfx_.Get<uint64_t>();
        uint64_t spanBase = 0;
        uint64_t spanLen = 0;
        bool haveSpan = false;
        uint64_t prevPrefix = 0;
        if (uBegin_ > 0) {
            prevPrefix = LoadGmU64(prefixGm_, uBegin_ - 1, meta);
        }
        bool have = false;
        uint32_t k = 0;
        uint64_t outPosBase = 0;
        LocalTensor<T> outVal;
        LocalTensor<int32_t> outRow;
        LocalTensor<int32_t> outCol;
        for (uint64_t unit = uBegin_; unit < uEnd; ++unit) {
            if (!haveSpan || unit >= spanBase + spanLen) {
                RefillCooSpan(pfxSpan, unit, uEnd, spanBase, spanLen,
                    haveSpan);
            }
            const uint64_t pfxHi =
                pfxSpan.GetValue((uint32_t)(unit - spanBase));
            const uint64_t outPos = prevPrefix;
            const uint64_t cnt = pfxHi - prevPrefix;
            prevPrefix = pfxHi;
            RotateCooStaging(k, cnt, outPos, have, outPosBase, outVal,
                             outRow, outCol);
            if (cnt > 0) {
                ConvertCooUnitInto(unit, outVal, outRow, outCol, k);
            }
        }
        if (k > 0) {
            FlushCoo(outVal, outRow, outCol, outPosBase, k);
        } else if (have) {
            outValQ_.FreeTensor(outVal);
            outIdxQ_.FreeTensor(outRow);
            outColQ_.FreeTensor(outCol);
        }
    }

    // Flush-and-realloc handshake for the COO staging tensors: rotate
    // them out when the next unit would overflow, allocate on first use.
    __aicore__ inline void RotateCooStaging(uint32_t &k, uint64_t cnt,
        uint64_t outPos, bool &have, uint64_t &outPosBase,
        LocalTensor<T> &outVal, LocalTensor<int32_t> &outRow,
        LocalTensor<int32_t> &outCol)
    {
        if (k > 0 && k + cnt > stageElems_) {
            FlushCoo(outVal, outRow, outCol, outPosBase, k);
            k = 0;
            have = false;
        }
        if (!have) {
            outVal = outValQ_.AllocTensor<T>();
            outRow = outIdxQ_.AllocTensor<int32_t>();
            outCol = outColQ_.AllocTensor<int32_t>();
            outPosBase = outPos;
            have = true;
        }
    }

    // Windowed prefix loads: each 1024-entry span serves the units that
    // fall inside it.
    __aicore__ inline void RefillCooSpan(LocalTensor<uint64_t> &pfxSpan,
        uint64_t unit, uint64_t uEnd, uint64_t &spanBase, uint64_t &spanLen,
        bool &haveSpan)
    {
        spanBase = unit;
        spanLen = MinU64((uint64_t)1024, uEnd - unit);
        DataCopyPad(pfxSpan, prefixGm_[spanBase],
            {1, (uint16_t)(spanLen * sizeof(uint64_t)), 0, 0},
            {false, 0, 0, 0});
        PipeBarrier<PIPE_MTE2>();
        haveSpan = true;
    }

    __aicore__ inline void FlushCoo(LocalTensor<T> &outVal,
        LocalTensor<int32_t> &outRow, LocalTensor<int32_t> &outCol,
        uint64_t outPos, uint32_t k)
    {
        PipeBarrier<PIPE_ALL>();
        outValQ_.EnQue(outVal);
        outIdxQ_.EnQue(outRow);
        outColQ_.EnQue(outCol);
        auto vd = outValQ_.DeQue<T>();
        auto rd = outIdxQ_.DeQue<int32_t>();
        auto cd = outColQ_.DeQue<int32_t>();
        DataCopyPad(valuesGm_[outPos * kGmStride], vd,
            {1, (uint32_t)((uint64_t)k * elemBytes_), 0, 0, 0});
        DataCopyPad(rowGm_[outPos], rd,
            {1, (uint32_t)((uint64_t)k * sizeof(int32_t)), 0, 0, 0});
        DataCopyPad(colGm_[outPos], cd,
            {1, (uint32_t)((uint64_t)k * sizeof(int32_t)), 0, 0, 0});
        PipeBarrier<PIPE_MTE3>();
        outValQ_.FreeTensor(vd);
        outIdxQ_.FreeTensor(rd);
        outColQ_.FreeTensor(cd);
    }

    // Row-span handlers for the convert side (see WalkCooTileRows).
    struct CooSegmentHandler {
        ConvertKernel *self;
        LocalTensor<T> *outVal;
        LocalTensor<int32_t> *outRow;
        LocalTensor<int32_t> *outCol;
        uint32_t *k;
        __aicore__ inline void OnContig(uint64_t r, uint64_t cLo, uint32_t len)
        {
            self->CooSegmentContig(r * self->t_->ld + cLo, len,
                r * self->t_->cols + cLo, *outVal, *outRow, *outCol, *k);
        }
        __aicore__ inline void OnStrided(uint64_t r, uint64_t cLo,
            uint32_t len)
        {
            self->CooSegmentStrided(r, cLo, len, *outVal, *outRow, *outCol,
                *k);
        }
    };

    __aicore__ inline void ConvertCooUnitInto(uint64_t unit,
        LocalTensor<T> &outVal, LocalTensor<int32_t> &outRow,
        LocalTensor<int32_t> &outCol, uint32_t &k)
    {
        const uint64_t logical = t_->rows * t_->cols;
        const uint64_t p0 = unit * kD2sCooTile;
        const uint64_t p1 = MinU64(p0 + kD2sCooTile, logical);
        CooSegmentHandler h{this, &outVal, &outRow, &outCol, &k};
        WalkCooTileRows(p0, p1, t_->cols, t_->order, h);
    }

    __aicore__ inline void CooSegmentContig(uint64_t gmElem, uint32_t len,
        uint64_t flatBase, LocalTensor<T> &outVal,
        LocalTensor<int32_t> &outRow, LocalTensor<int32_t> &outCol,
        uint32_t &k)
    {
        LocalTensor<T> tileIn = LoadTileAndBuildBitmap<T, kGmStride,
            kComplex>(inQueue_, denseGm_, bufs_, gmElem, len, elemBytes_);
        auto mask = kComplex ? bufs_.Mask2() : bufs_.Mask1();
        BitmapScan scan;
        BitmapScanInit(scan, mask, len);
        uint32_t i = 0;
        while (BitmapScanNext(scan, len, i)) {
            CopyElem<T, kComplex>(outVal, k, tileIn, i);
            const uint64_t flat = flatBase + i;
            const uint64_t row = flat / t_->cols;
            const uint64_t col = flat - row * t_->cols;
            outRow.SetValue(k, (int32_t)row + (int32_t)t_->base);
            outCol.SetValue(k, (int32_t)col + (int32_t)t_->base);
            ++k;
        }
        inQueue_.FreeTensor(tileIn);
    }

    __aicore__ inline void CooSegmentStrided(uint64_t r, uint64_t cLo,
        uint32_t len, LocalTensor<T> &outVal, LocalTensor<int32_t> &outRow,
        LocalTensor<int32_t> &outCol, uint32_t &k)
    {
        const uint32_t slotElems = kSlotBytes / elemBytes_;
        if (t_->rows < slotElems ||
            (t_->ld - slotElems) * elemBytes_ >= 0xFFFFFFFFULL) {
            for (uint64_t c = cLo; c < cLo + len; ++c) {
                CooSegmentContig(c * t_->ld + r, 1, r * t_->cols + c, outVal,
                    outRow, outCol, k);
            }
            return;
        }
        const uint32_t maxSlots = t_->tileLen * elemBytes_ / kSlotBytes;
        const uint32_t winOff = WindowOffset(r, slotElems, t_->rows);
        PrepareSlotWalkMask<T, kComplex>(bufs_, maxSlots, winOff);
        for (uint32_t done = 0; done < len;) {
            const uint32_t slots =
                (uint32_t)MinU64((uint64_t)maxSlots, (uint64_t)(len - done));
            const uint64_t c0 = cLo + done;
            auto tileIn = LoadSlotGroup<T, kGmStride, kComplex>(
                inQueue_, denseGm_, *t_, elemBytes_, r, c0, slots, winOff);
            const uint64_t tileBytes = (uint64_t)slots * kSlotBytes;
            if (tileBytes % 256ULL == 0 && tileBytes >= kSlotVecMinBytes) {
                CountSlots<T, kComplex>(bufs_, tileIn, slots, winOff);
                auto mask = bufs_.Mask1();
                for (uint32_t i = 0; i < slots; ++i) {
                    if (SlotNonzero<T, kComplex>(mask, i, winOff)) {
                        EmitCooSlotElem(tileIn, i, r, c0, winOff, outVal,
                            outRow, outCol, k);
                    }
                }
            } else {
                for (uint32_t i = 0; i < slots; ++i) {
                    if (SlotElemNonzero<T, kComplex>(tileIn, i, winOff)) {
                        EmitCooSlotElem(tileIn, i, r, c0, winOff, outVal,
                            outRow, outCol, k);
                    }
                }
            }
            inQueue_.FreeTensor(tileIn);
            done += slots;
        }
    }

    // COO slot element emission shared by the bitmap and raw-test forms.
    __aicore__ inline void EmitCooSlotElem(const LocalTensor<T> &tile,
        uint32_t i, uint64_t r, uint64_t c0, uint32_t winOff,
        LocalTensor<T> &outVal, LocalTensor<int32_t> &outRow,
        LocalTensor<int32_t> &outCol, uint32_t &k)
    {
        CopySlotElem(tile, i, winOff, outVal, k);
        outRow.SetValue(k, (int32_t)r + (int32_t)t_->base);
        outCol.SetValue(k, (int32_t)(c0 + i) + (int32_t)t_->base);
        ++k;
    }

    TPipe pipe_;
    TQue<TPosition::VECIN, 2> inQueue_;
    TQue<TPosition::VECOUT, 1> outValQ_;
    TQue<TPosition::VECOUT, 1> outIdxQ_;
    TQue<TPosition::VECOUT, 1> outColQ_;
    TBuf<TPosition::VECCALC> pfxSpan_;
    TBuf<TPosition::VECCALC> offSpan_;
    TBuf<TPosition::VECCALC> cooPfx_;
    TBuf<TPosition::VECCALC> statusBuf_;
    TBuf<TPosition::VECCALC> scratch_;
    JudgeBufs bufs_;
    PositionResolver pos_;
    GlobalTensor<uint64_t> prefixGm_;
    GlobalTensor<int32_t> statusGm_;
    GlobalTensor<int32_t> offsetsGm_;
    GlobalTensor<int32_t> indicesGm_;
    GlobalTensor<int32_t> rowGm_;
    GlobalTensor<int32_t> colGm_;
    GlobalTensor<T> valuesGm_;
    uint32_t core_ = 0;
    bool panel_ = false;
    bool colPanel_ = false;
    bool panelSpanOk_ = false;
    uint32_t stageElems_ = kStageElems;
    bool rampBuilt_ = false;
};

} // namespace

template <typename T, uint32_t kGmStride, bool kComplex>
__aicore__ inline void RunConvert(GM_ADDR dense, GM_ADDR prefix,
    GM_ADDR status, GM_ADDR offsets, GM_ADDR indices, GM_ADDR rowIndices,
    GM_ADDR colIndices, GM_ADDR values, const DenseToSparseTilingData &tiling)
{
    ConvertKernel<T, kGmStride, kComplex> op;
    op.Init(dense, prefix, status, offsets, indices, rowIndices, colIndices,
        values, tiling);
    op.Process();
}

} // namespace densetosparse

extern "C" __global__ __aicore__ void densetosparse_convert_kernel(
    GM_ADDR dense, GM_ADDR prefix, GM_ADDR status, GM_ADDR offsets,
    GM_ADDR indices, GM_ADDR rowIndices, GM_ADDR colIndices, GM_ADDR values,
    DenseToSparseTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if (tiling.elementBytes == 1) {
        densetosparse::RunConvert<int8_t, 1, false>(dense, prefix, status,
            offsets, indices, rowIndices, colIndices, values, tiling);
    } else if (tiling.elementBytes == 2) {
        densetosparse::RunConvert<half, 1, false>(dense, prefix, status,
            offsets, indices, rowIndices, colIndices, values, tiling);
    } else if (tiling.elementBytes == 4) {
        densetosparse::RunConvert<float, 1, false>(dense, prefix, status,
            offsets, indices, rowIndices, colIndices, values, tiling);
    } else {
        densetosparse::RunConvert<float, 2, true>(dense, prefix, status,
            offsets, indices, rowIndices, colIndices, values, tiling);
    }
}
