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
 * \file sddmm_simd_kernel.h
 * \brief sddmm arch22 kernel 实现（SIMD：TPipe/TBuf/LocalTensor + 低阶 AscendC API）。
 */

#ifndef SDDMM_ARCH22_SIMD_KERNEL_H_
#define SDDMM_ARCH22_SIMD_KERNEL_H_

#include "kernel_operator.h"
#include "sddmm.h"

namespace {
// UB block granularity for fp32: 32 bytes / 4 bytes = 8 elements.
constexpr uint32_t kAlign = 8;
// Elements a single vector repeat can cover for fp32 (256B / 4B).
constexpr uint32_t kElemPerRep = 64;
// Repeat-stride fields are 8-bit, so a staged row pitch may span at most 255 blocks.
constexpr uint32_t kMaxPitchBlocks = 255;
// Staging capacity for the CSR value / column-index / output segments.
constexpr uint32_t kOutStageCap = 4096;
// Nonzeros processed per batched segment. Each needs one kElemPerRep-wide product
// row in mulBuf and one staged B column in bColBuf, so this bounds both buffers.
constexpr uint32_t kJTileCap = 128;
// Scanning for consecutive column runs only pays off above ~1/kRunDensityNum
// density; sparser patterns almost never yield runs longer than one.
constexpr uint32_t kRunDensityNum = 4;
constexpr uint32_t kHugeKCap = 8192;
constexpr int kACacheNone = 0;
constexpr int kACacheSingle = 1;
constexpr int kACacheFull = 2;
constexpr uint32_t kSmallDmaMinK = 32;
constexpr uint32_t kSmallDmaMinRows = 256;
// DMA alignment requirement: DataCopy requires lenBytes to be 32-byte aligned.
constexpr uint32_t kDmaAlignBytes = 32;
constexpr uint32_t kDmaAlignMask = kDmaAlignBytes - 1;
// Strided GM->UB gather: DataCopyPad lands each element on its own 32B block, so
// the staging buffer holds kStrideChunk elements at kStrideElemBytes pitch.
constexpr uint32_t kStrideElemBytes = 32;
constexpr uint32_t kStrideChunk = 64;

__aicore__ inline uint32_t AlignUp(uint32_t x, uint32_t a) {
    if (a == 0) {
        return x;
    }
    return (x + a - 1) / a * a;
}

__aicore__ inline uint32_t MinU32(uint32_t a, uint32_t b) {
    return (a < b) ? a : b;
}
} // namespace

class KernelSparseSddmm {
public:
    __aicore__ inline KernelSparseSddmm() {}

    __aicore__ inline void Init(GM_ADDR matA, GM_ADDR matB, GM_ADDR csrRowOffsets,
                                GM_ADDR csrColIndices, GM_ADDR csrValues,
                                const SddmmTilingData &td, AscendC::TPipe *pipeIn) {
        pipe = pipeIn;
        CopyTiling(td);

        matAGm.SetGlobalBuffer((__gm__ float *)matA,
                               static_cast<uint64_t>(matARows) * ldx);
        const uint64_t bGmSize = static_cast<uint64_t>(matBRows) *
                                 ((ldy > matBCols) ? ldy : matBCols);
        matBGm.SetGlobalBuffer((__gm__ float *)matB, bGmSize);
        rowOffsetsGm.SetGlobalBuffer((__gm__ int32_t *)csrRowOffsets,
                                     static_cast<uint64_t>(rowCount + 1));
        colIndicesGm.SetGlobalBuffer((__gm__ int32_t *)csrColIndices, nnz);
        csrValuesGm.SetGlobalBuffer((__gm__ float *)csrValues, nnz);

        const uint32_t kbuf = AlignUp(kBlockLen, kAlign);
        kbufAligned = kbuf;
        pipe->InitBuffer(aRowBuf, kbuf * sizeof(float));
        // Doubles as the double-buffered scalar B column pair and as the batched B tile.
        pipe->InitBuffer(bColBuf, 2 * kbuf * sizeof(float));
        hugeKPath = (matARows > kBlockLen) || (matACols > kBlockLen);
        if (hugeKPath) {
            pipe->InitBuffer(aFullBuf, AlignUp(kHugeKCap, kAlign) * sizeof(float));
        }
        pipe->InitBuffer(inQueueCsr, 1, kOutStageCap * sizeof(float));
        pipe->InitBuffer(outQueueY, 1, kOutStageCap * sizeof(float));
        pipe->InitBuffer(colIndBuf, kOutStageCap * sizeof(int32_t));
        // mulBuf holds either one K-length product vector (scalar path) or
        // kJTileCap product rows of kElemPerRep each (batched path).
        const uint32_t mulElems = (kJTileCap * kElemPerRep > kbuf) ? (kJTileCap * kElemPerRep) : kbuf;
        pipe->InitBuffer(mulBuf, mulElems * sizeof(float));
        pipe->InitBuffer(reduceTmpBuf, kbuf * sizeof(float));
        pipe->InitBuffer(reduceDstBuf, kAlign * sizeof(float));
        // Dot-product accumulator, kept separate from reduceTmpBuf because
        // ReduceSum overwrites its sharedTmpBuffer argument.
        pipe->InitBuffer(accBuf, AlignUp(kJTileCap, kAlign) * sizeof(float));
        pipe->InitBuffer(partialBuf, AlignUp(kJTileCap, kAlign) * sizeof(float));
        // The strided gather is only reachable when a row of X or a column of Y is
        // non-contiguous in GM; skip its staging buffers otherwise so contiguous
        // shapes (notably the huge-K path) keep the UB headroom.
        const bool aCtg = (orderXCol == 0 && opAValue == 0) || (orderXCol == 1 && opAValue == 1);
        const bool bCtg = (orderYCol == 0 && opBValue == 1) || (orderYCol == 1 && opBValue == 0);
        needStrideGather = (!aCtg || !bCtg);
        if (needStrideGather) {
            pipe->InitBuffer(stridePadBuf, kStrideChunk * kStrideElemBytes);
            pipe->InitBuffer(strideOffsetBuf, kStrideChunk * sizeof(int32_t));
        }
    }


    __aicore__ inline void Process() {
        if (!SetupLayout()) {
            return;
        }
        SetupBatchPlan();
        if (nnzPartition != 0) {
            RunNnzPartition();
        } else {
            RunRowPartition();
        }
    }

private:

    // Split the reduction dimension K and the contiguity flags out of the tiling
    // data. Returns false when there is nothing to do or the shapes disagree.
    __aicore__ inline bool SetupLayout() {
        if (nnz == 0 || rowCount == 0) {
            return false;
        }
        const uint32_t innerA = (opAValue == 0) ? matACols : matARows;
        const uint32_t innerB = (opBValue == 0) ? matBRows : matBCols;
        if (innerA != innerB) {
            return false;
        }
        K = innerA;
        colCount = (opBValue == 0) ? matBCols : matBRows;
        // A row (resp. B column) is contiguous in GM when the memory order and the
        // requested transpose agree.
        aContig = (orderXCol == 0 && opAValue == 0) || (orderXCol == 1 && opAValue == 1);
        bContig = (orderYCol == 0 && opBValue == 1) || (orderYCol == 1 && opBValue == 0);
        // The strided gather (and hence its offset table) is only reachable when a
        // row of X or a column of Y is non-contiguous in GM.
        if (needStrideGather) {
            InitStrideOffsets();
        }
        return SetupDmaFlags();
    }

    // Each core takes a contiguous slice of nonzeros; used for huge-K shapes where
    // a single row would not fill the cores.
    __aicore__ inline void RunNnzPartition() {
        const uint32_t coreIdx = AscendC::GetBlockIdx();
        const uint32_t extraN = (coreIdx < remainderNnz) ? 1u : 0u;
        const uint32_t myNnz = nnzPerCore + extraN;
        const uint32_t myBase = coreIdx * nnzPerCore +
                                ((coreIdx < remainderNnz) ? coreIdx : remainderNnz);
        for (uint32_t j = 0; j < myNnz; ++j) {
            if (myBase + j < nnz) {
                ProcessNnz(myBase + j);
            }
        }
    }

    // Each core takes a contiguous slice of rows (the common case).
    __aicore__ inline void RunRowPartition() {
        const uint32_t totalRows = rowCount;
        const uint32_t coreIdx = AscendC::GetBlockIdx();
        const uint32_t extra = (coreIdx < remainderRows) ? 1u : 0u;
        const uint32_t myCount = rowsPerCore + extra;
        const uint32_t myBase = coreIdx * rowsPerCore +
                                ((coreIdx < remainderRows) ? coreIdx : remainderRows);
        for (uint32_t g = myBase; g < myBase + myCount && g < totalRows; ++g) {
            ProcessRow(g);
        }
    }

    // Contiguous small-K loads are only worth a DMA above a size threshold.
    __aicore__ inline bool SetupDmaFlags() {
        const uint32_t totalRows = rowCount;
        const bool kInRange = (K <= kBlockLen) && (K >= kSmallDmaMinK);
        smallKContigADma = aContig && kInRange && (totalRows >= kSmallDmaMinRows);
        smallKContigBDma = bContig && kInRange;
        return true;
    }
    // Copy the scalar tiling fields into members.
    __aicore__ inline void CopyTiling(const SddmmTilingData &td) {
        nnz = static_cast<uint64_t>(td.nnz);
        rowCount = td.rowCount;
        matARows = td.matARows;
        matACols = td.matACols;
        matBRows = td.matBRows;
        matBCols = td.matBCols;
        rowsPerCore = td.rowsPerCore;
        remainderRows = td.remainderRows;
        kBlockLen = td.kBlockLen;
        nnzPartition = td.nnzPartition;
        nnzPerCore = td.nnzPerCore;
        remainderNnz = td.remainderNnz;
        alphaValue = td.alphaHost;
        betaValue = td.betaHost;
        opAValue = static_cast<int64_t>(td.opX);
        opBValue = static_cast<int64_t>(td.opY);
        ldx = td.ldx;
        ldy = td.ldy;
        orderXCol = td.orderX;
        orderYCol = td.orderY;
    }
    // Decide whether the batched (vector-reduction) path is usable, and size its tile.
    //
    // The batched path stages jTile B columns contiguously in UB as a jTile x kPitch
    // matrix, multiplies them against the cached A row using a repeat-stride-0
    // broadcast, then collapses each staged row to a scalar with a single
    // WholeReduceSum. That removes the per-nonzero ReduceSum + GetValue vector-to-scalar
    // stall that dominates the scalar path.
    __aicore__ inline void SetupBatchPlan() {
        batchEnabled = false;
        if (!bContig || K == 0 || K > kBlockLen) {
            return;
        }
        kPitch = AlignUp(K, kAlign);
        if (kPitch == 0) {
            return;
        }
        // src1RepStride / dstRepStride are 8-bit block counts.
        if ((kPitch / kAlign) > kMaxPitchBlocks) {
            return;
        }
        // B tile lives in bColBuf (2 * kbufAligned floats).
        const uint32_t byTile = (2u * kbufAligned) / kPitch;
        const uint32_t cap = MinU32(byTile, kJTileCap);
        if (cap == 0) {
            return;
        }
        jTile = cap;
        batchEnabled = true;
        // Multi-block staging needs contiguous destination slots (dstStride == 0),
        // which only holds when K is already block-aligned, and needs the source
        // gap to be non-negative. Only worth the index scan when C is dense enough
        // that consecutive column runs actually occur.
        runBatchable = (kPitch == K) && (ldy >= K) &&
                       (static_cast<uint64_t>(nnz) * kRunDensityNum >
                        static_cast<uint64_t>(rowCount) * colCount);
    }

    // Build the byte-offset table 0, 32, 64, ... used by the strided gather below.
    // Computed once per kernel invocation.
    __aicore__ inline void InitStrideOffsets() {
        AscendC::LocalTensor<int32_t> offs = strideOffsetBuf.Get<int32_t>();
        AscendC::CreateVecIndex<int32_t>(offs, 0, kStrideChunk);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Muls(offs, offs, static_cast<int32_t>(kStrideElemBytes),
                      static_cast<int32_t>(kStrideChunk));
        AscendC::PipeBarrier<PIPE_V>();
    }

    // Strided GM->UB load: DataCopyPad pulls `len` single-element blocks (each
    // padded to 32B by the blockCount form), then one Gather compacts them into
    // ub[0..len). Vectorised throughout — no per-element scalar access.
    __aicore__ inline void GatherChunk(AscendC::LocalTensor<float> &ub,
                                       AscendC::GlobalTensor<float> &gm,
                                       uint64_t off, uint32_t stride, uint32_t len) {
        if (stride == 1u) {
            CopyContig(ub, gm, off, len);
            return;
        }
        AscendC::LocalTensor<float> pad32 = stridePadBuf.Get<float>();
        AscendC::LocalTensor<int32_t> offs = strideOffsetBuf.Get<int32_t>();
        AscendC::LocalTensor<uint32_t> offsU = offs.ReinterpretCast<uint32_t>();
        const uint32_t srcGapBytes = (stride - 1u) * (uint32_t)sizeof(float);
        for (uint32_t base = 0; base < len; base += kStrideChunk) {
            const uint32_t cnt = MinU32(len - base, kStrideChunk);
            AscendC::DataCopyExtParams cp{static_cast<uint16_t>(cnt), (uint32_t)sizeof(float),
                                          srcGapBytes, 0, 0};
            AscendC::DataCopyPadExtParams<float> pd{false, 0, 0, 0.0f};
            AscendC::DataCopyPad(pad32, gm[off + static_cast<uint64_t>(base) * stride], cp, pd);
            SyncMte2ToV();
            AscendC::Gather<float>(ub[base], pad32, offsU, 0, cnt);
            AscendC::PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline void CopyContig(AscendC::LocalTensor<float> &ub,
                                      AscendC::GlobalTensor<float> &gm,
                                      uint64_t off, uint32_t len) {
        const uint32_t lenBytes = len * (uint32_t)sizeof(float);
        if ((lenBytes & kDmaAlignMask) == 0 && lenBytes > 0) {
            AscendC::DataCopy(ub, gm[off], len);
        } else {
            AscendC::DataCopyExtParams p{1, lenBytes, 0, 0, 0};
            AscendC::DataCopyPadExtParams<float> pad{false, 0, 0, 0.0f};
            AscendC::DataCopyPad(ub, gm[off], p, pad);
        }
    }

    // Stage `run` consecutive B columns (GM row pitch ldy) into `run` adjacent UB
    // slots of kPitch elements each, as a single multi-block DMA.
    //
    // Only used when kPitch == K, so the destination blocks are back-to-back and
    // dstStride is 0. srcStride is the inter-block gap in bytes, matching the
    // convention used by the merged arch35 path.
    __aicore__ inline void CopyBRun(AscendC::LocalTensor<float> dst, uint64_t gmOff, uint32_t run) {
        if (run == 1) {
            CopyContig(dst, matBGm, gmOff, K);
            return;
        }
        AscendC::DataCopyExtParams p{static_cast<uint16_t>(run), K * (uint32_t)sizeof(float),
                                     (ldy - K) * (uint32_t)sizeof(float), 0, 0};
        AscendC::DataCopyPadExtParams<float> pad{false, 0, 0, 0.0f};
        AscendC::DataCopyPad(dst, matBGm[gmOff], p, pad);
    }

    __aicore__ inline void SyncMte2ToV() {
        event_t ev = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_V));
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(ev);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(ev);
    }

    __aicore__ inline void SyncMte2ToS() {
        event_t ev = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_S));
        AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(ev);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(ev);
    }

    __aicore__ inline void SyncVToS() {
        event_t ev = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_S));
        AscendC::SetFlag<AscendC::HardEvent::V_S>(ev);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(ev);
    }

    __aicore__ inline float DotProduct(uint64_t aOff, uint32_t aStride, uint64_t bOff,
                                       uint32_t bStride, int aCache) {
        AscendC::LocalTensor<float> aC = aRowBuf.Get<float>();
        AscendC::LocalTensor<float> bC = bColBuf.Get<float>();
        AscendC::LocalTensor<float> ml = mulBuf.Get<float>();
        AscendC::LocalTensor<float> rt = reduceTmpBuf.Get<float>();
        AscendC::LocalTensor<float> rd = reduceDstBuf.Get<float>();
        float acc = 0.0f;
        const bool bUseDma = (bStride == 1u && (aCache == kACacheFull || smallKContigBDma));
        for (uint32_t base = 0; base < K; base += kBlockLen) {
            const uint32_t len = (K - base < kBlockLen) ? (K - base) : kBlockLen;
            AscendC::LocalTensor<float> aOp = aC;
            if (aCache == kACacheNone) {
                GatherChunk(aC, matAGm, aOff + static_cast<uint64_t>(base) * aStride, aStride, len);
            } else if (aCache == kACacheFull) {
                aOp = aFullBuf.Get<float>()[base];
            }
            if (bUseDma) {
                CopyContig(bC, matBGm, bOff + static_cast<uint64_t>(base) * bStride, len);
                AscendC::PipeBarrier<PIPE_MTE2>();
            } else {
                GatherChunk(bC, matBGm, bOff + static_cast<uint64_t>(base) * bStride, bStride, len);
            }
            AscendC::Mul(ml, aOp, bC, len);
            AscendC::ReduceSum<float, true>(rd, ml, rt, static_cast<int32_t>(len));
            acc += rd.GetValue(0);
        }
        return acc;
    }

    __aicore__ inline void ProcessNnz(uint32_t gnnz) {
        uint32_t lo = 0;
        uint32_t hi = rowCount;
        while (lo < hi) {
            uint32_t mid = (lo + hi) / 2;
            if (rowOffsetsGm.GetValue(mid + 1) <= static_cast<int32_t>(gnnz)) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        const uint32_t row = lo;
        const uint64_t aOff = aContig ? static_cast<uint64_t>(row) * ldx : static_cast<uint64_t>(row);
        const uint32_t aStride = aContig ? 1u : ldx;
        int aCache = kACacheNone;
        if (alphaValue != 0.0f) {
            if (K <= kBlockLen) {
                aCache = kACacheSingle;
                AscendC::LocalTensor<float> a = aRowBuf.Get<float>();
                GatherChunk(a, matAGm, aOff, aStride, K);
            } else if (hugeKPath && K <= kHugeKCap) {
                aCache = kACacheFull;
                AscendC::LocalTensor<float> a = aFullBuf.Get<float>();
                if (aStride == 1u) {
                    CopyContig(a, matAGm, aOff, K);
                } else {
                    GatherChunk(a, matAGm, aOff, aStride, K);
                }
            }
        }
        const int32_t col = colIndicesGm.GetValue(gnnz);
        float dot = 0.0f;
        if (alphaValue != 0.0f && col >= 0 && static_cast<uint32_t>(col) < colCount) {
            const uint64_t bOff = bContig ? static_cast<uint64_t>(col) * ldy
                                          : static_cast<uint64_t>(col);
            const uint32_t bStr = bContig ? 1u : ldy;
            dot = DotProduct(aOff, aStride, bOff, bStr, aCache);
        }
        const float cVal = csrValuesGm.GetValue(gnnz);
        AscendC::LocalTensor<float> y = outQueueY.AllocTensor<float>();
        y.SetValue(0, alphaValue * dot + betaValue * cVal);
        outQueueY.EnQue(y);
        AscendC::LocalTensor<float> yo = outQueueY.DeQue<float>();
        AscendC::DataCopyExtParams op{1, (uint32_t)sizeof(float), 0, 0, 0};
        AscendC::DataCopyPad(csrValuesGm[gnnz], yo, op);
        outQueueY.FreeTensor(yo);
    }

    // Load the A row (or A column when opA/order make it strided) into UB once per row.
    __aicore__ inline int LoadARow(uint64_t aOff, uint32_t aStride) {
        if (alphaValue == 0.0f) {
            return kACacheNone;
        }
        if (K <= kBlockLen) {
            AscendC::LocalTensor<float> a = aRowBuf.Get<float>();
            if (smallKContigADma || (batchEnabled && aStride == 1u)) {
                CopyContig(a, matAGm, aOff, K);
            } else {
                GatherChunk(a, matAGm, aOff, aStride, K);
            }
            return kACacheSingle;
        }
        if (hugeKPath && K <= kHugeKCap) {
            AscendC::LocalTensor<float> a = aFullBuf.Get<float>();
            if (aStride == 1u) {
                CopyContig(a, matAGm, aOff, K);
            } else {
                GatherChunk(a, matAGm, aOff, aStride, K);
            }
            return kACacheFull;
        }
        return kACacheNone;
    }

    // Batched segment: compute `segNnz` dot products with vector instructions only.
    //
    // Layout in UB:
    //   bTile[j * kPitch .. + K)   staged B column for nonzero j
    //   prod[j * kElemPerRep ..)   per-chunk products, one repeat row per nonzero
    //   acc[j]                     running dot product
    __aicore__ inline void ComputeSegmentBatched(AscendC::LocalTensor<int32_t> &colInd,
                                                 uint32_t segBase, uint32_t segNnz,
                                                 uint64_t bBB,
                                                 AscendC::LocalTensor<float> &aOp,
                                                 AscendC::LocalTensor<float> &acc) {
        AscendC::LocalTensor<float> bTile = bColBuf.Get<float>();
        StageBTile(colInd, segBase, segNnz, bBB, bTile);
        SyncMte2ToV();
        ReduceBTile(bTile, aOp, acc, segNnz);
    }

    // Stage this segment's B columns into `bTile` as segNnz rows of kPitch each.
    // CSR column indices ascend, so runs of consecutive columns are common; each
    // run is issued as one multi-block DMA instead of one DMA per nonzero.
    // Out-of-range columns contribute zero.
    __aicore__ inline void StageBTile(AscendC::LocalTensor<int32_t> &colInd,
                                      uint32_t segBase, uint32_t segNnz, uint64_t bBB,
                                      AscendC::LocalTensor<float> &bTile) {
        uint32_t j = 0;
        while (j < segNnz) {
            const int32_t col = colInd.GetValue(segBase + j);
            if (col < 0 || static_cast<uint32_t>(col) >= colCount) {
                AscendC::Duplicate(bTile[j * kPitch], 0.0f, static_cast<int32_t>(kPitch));
                ++j;
                continue;
            }
            uint32_t run = 1;
            if (runBatchable) {
                while (j + run < segNnz &&
                       colInd.GetValue(segBase + j + run) == col + static_cast<int32_t>(run)) {
                    ++run;
                }
            }
            CopyBRun(bTile[j * kPitch], bBB + static_cast<uint64_t>(col) * ldy, run);
            j += run;
        }
    }

    // Multiply the staged tile by the cached A row and collapse each staged row to
    // one dot product. acc[0..segNnz) receives the results.
    __aicore__ inline void ReduceBTile(AscendC::LocalTensor<float> &bTile,
                                       AscendC::LocalTensor<float> &aOp,
                                       AscendC::LocalTensor<float> &acc, uint32_t segNnz) {
        AscendC::LocalTensor<float> prod = mulBuf.Get<float>();
        const uint8_t rep = static_cast<uint8_t>(segNnz);
        const int32_t repStride = static_cast<int32_t>(kElemPerRep / kAlign);
        AscendC::BinaryRepeatParams mp;
        mp.dstBlkStride = 1;
        mp.src0BlkStride = 1;
        mp.src1BlkStride = 1;
        mp.dstRepStride = static_cast<uint8_t>(repStride);
        // src0RepStride == 0 broadcasts the single cached A row across all repeats.
        mp.src0RepStride = 0;
        mp.src1RepStride = static_cast<uint8_t>(kPitch / kAlign);

        if (K <= kElemPerRep) {
            // Single chunk: reduce straight into acc, no accumulator pass needed.
            AscendC::Mul(prod, aOp, bTile, static_cast<uint64_t>(K), rep, mp);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::WholeReduceSum<float>(acc, prod, static_cast<int32_t>(K),
                                           static_cast<int32_t>(segNnz), 1, 1, repStride);
            AscendC::PipeBarrier<PIPE_V>();
            return;
        }

        AscendC::LocalTensor<float> partial = partialBuf.Get<float>();
        AscendC::Duplicate(acc, 0.0f, static_cast<int32_t>(AlignUp(segNnz, kAlign)));
        AscendC::PipeBarrier<PIPE_V>();
        for (uint32_t base = 0; base < K; base += kElemPerRep) {
            const uint32_t len = MinU32(K - base, kElemPerRep);
            // prod[j][0..len) = aOp[base..base+len) * bTile[j][base..base+len)
            AscendC::Mul(prod, aOp[base], bTile[base], static_cast<uint64_t>(len), rep, mp);
            AscendC::PipeBarrier<PIPE_V>();
            // One vcadd collapses each staged repeat row to a single dot-product term.
            AscendC::WholeReduceSum<float>(partial, prod, static_cast<int32_t>(len),
                                           static_cast<int32_t>(segNnz), 1, 1, repStride);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Add(acc, acc, partial, static_cast<int32_t>(segNnz));
            AscendC::PipeBarrier<PIPE_V>();
        }
    }

    // Scalar fallback for layouts the batched path cannot serve (strided B, huge K).
    __aicore__ inline void ComputeSegmentScalar(AscendC::LocalTensor<int32_t> &colInd,
                                                 uint32_t segBase, uint32_t segNnz,
                                                 uint64_t bBB, uint64_t aOff, uint32_t aStride,
                                                 int aCache, AscendC::LocalTensor<float> &acc) {
        for (uint32_t j = 0; j < segNnz; ++j) {
            const int32_t col = colInd.GetValue(segBase + j);
            float dot = 0.0f;
            if (alphaValue != 0.0f && col >= 0 && static_cast<uint32_t>(col) < colCount) {
                const uint64_t bOff = bContig ? (bBB + static_cast<uint64_t>(col) * ldy)
                                              : (bBB + static_cast<uint64_t>(col));
                const uint32_t bStr = bContig ? 1u : ldy;
                dot = DotProduct(aOff, aStride, bOff, bStr, aCache);
            }
            acc.SetValue(j, dot);
        }
    }

    // alpha == 0 case: y = beta * C over the row, in segments of kOutStageCap.
    __aicore__ inline void ScaleRowOnly(uint64_t rowOff, uint32_t rowNnz) {
        for (uint32_t segBase = 0; segBase < rowNnz; segBase += kOutStageCap) {
            const uint32_t segNnz = MinU32(rowNnz - segBase, kOutStageCap);
            const uint64_t segOff = rowOff + segBase;
            const uint32_t segBytes = segNnz * (uint32_t)sizeof(float);
            AscendC::LocalTensor<float> yStage = outQueueY.AllocTensor<float>();
            if (betaValue == 0.0f) {
                AscendC::Duplicate(yStage, 0.0f, static_cast<int32_t>(AlignUp(segNnz, kAlign)));
                AscendC::PipeBarrier<PIPE_V>();
            } else {
                AscendC::LocalTensor<float> csrIn = inQueueCsr.AllocTensor<float>();
                AscendC::DataCopyExtParams p{1, segBytes, 0, 0, 0};
                AscendC::DataCopyPadExtParams<float> pad{false, 0, 0, 0.0f};
                AscendC::DataCopyPad(csrIn, csrValuesGm[segOff], p, pad);
                inQueueCsr.EnQue(csrIn);
                AscendC::LocalTensor<float> csrStage = inQueueCsr.DeQue<float>();
                AscendC::Muls(yStage, csrStage, betaValue, static_cast<int32_t>(segNnz));
                AscendC::PipeBarrier<PIPE_V>();
                inQueueCsr.FreeTensor(csrStage);
            }
            outQueueY.EnQue(yStage);
            AscendC::LocalTensor<float> yOut = outQueueY.DeQue<float>();
            AscendC::DataCopyExtParams outP{1, segBytes, 0, 0, 0};
            AscendC::DataCopyPad(csrValuesGm[segOff], yOut, outP);
            outQueueY.FreeTensor(yOut);
        }
    }

    __aicore__ inline void ProcessRow(uint32_t row) {
        const int32_t start = rowOffsetsGm.GetValue(row);
        const int32_t end = rowOffsetsGm.GetValue(row + 1);
        const uint32_t rowNnz = static_cast<uint32_t>(end - start);
        if (rowNnz == 0) {
            return;
        }

        const uint64_t aOff = aContig ? (static_cast<uint64_t>(row) * ldx) : static_cast<uint64_t>(row);
        const uint32_t aStride = aContig ? 1u : ldx;

        if (alphaValue == 0.0f) {
            ScaleRowOnly(static_cast<uint64_t>(start), rowNnz);
            return;
        }

        const uint32_t segCap = batchEnabled ? jTile : kOutStageCap;
        const int aCache = LoadARow(aOff, aStride);
        for (uint32_t segBase = 0; segBase < rowNnz; segBase += segCap) {
            const uint32_t segNnz = MinU32(rowNnz - segBase, segCap);
            ProcessSegment(static_cast<uint64_t>(start) + segBase, segNnz,
                           0ULL, aOff, aStride, aCache);
        }
    }

    // Load one segment's column indices (and C values when beta != 0), compute the
    // dot products, then write back y = alpha * dot + beta * C.
    __aicore__ inline void ProcessSegment(uint64_t segOff, uint32_t segNnz, uint64_t bBB,
                                          uint64_t aOff, uint32_t aStride, int aCache) {
        AscendC::LocalTensor<int32_t> colInd = colIndBuf.Get<int32_t>();
        AscendC::DataCopyExtParams cp{1, segNnz * (uint32_t)sizeof(int32_t), 0, 0, 0};
        AscendC::DataCopyPadExtParams<int32_t> cpd{false, 0, 0, 0};
        AscendC::DataCopyPad(colInd, colIndicesGm[segOff], cp, cpd);

        AscendC::LocalTensor<float> csrIn = inQueueCsr.AllocTensor<float>();
        if (betaValue != 0.0f) {
            AscendC::DataCopyExtParams p{1, segNnz * (uint32_t)sizeof(float), 0, 0, 0};
            AscendC::DataCopyPadExtParams<float> pad{false, 0, 0, 0.0f};
            AscendC::DataCopyPad(csrIn, csrValuesGm[segOff], p, pad);
        }
        if (batchEnabled) {
            // colInd is read by the scalar unit while staging B; csrStage is
            // consumed by the vector unit. Wait for both DMAs once.
            SyncMte2ToS();
        } else {
            AscendC::PipeBarrier<PIPE_MTE2>();
        }
        inQueueCsr.EnQue(csrIn);
        AscendC::LocalTensor<float> csrStage = inQueueCsr.DeQue<float>();

        AscendC::LocalTensor<float> yStage = outQueueY.AllocTensor<float>();
        AscendC::LocalTensor<float> acc = accBuf.Get<float>();
        if (batchEnabled) {
            AscendC::LocalTensor<float> aOp = (aCache == kACacheFull) ? aFullBuf.Get<float>()
                                                                     : aRowBuf.Get<float>();
            ComputeSegmentBatched(colInd, 0, segNnz, bBB, aOp, acc);
            CombineBatched(yStage, acc, csrStage, segNnz);
        } else {
            ComputeSegmentScalar(colInd, 0, segNnz, bBB, aOff, aStride, aCache, acc);
            for (uint32_t j = 0; j < segNnz; ++j) {
                const float cv = (betaValue != 0.0f) ? csrStage.GetValue(j) : 0.0f;
                yStage.SetValue(j, alphaValue * acc.GetValue(j) + betaValue * cv);
            }
        }
        inQueueCsr.FreeTensor(csrStage);

        outQueueY.EnQue(yStage);
        AscendC::LocalTensor<float> yOut = outQueueY.DeQue<float>();
        AscendC::DataCopyExtParams outP{1, segNnz * (uint32_t)sizeof(float), 0, 0, 0};
        AscendC::DataCopyPad(csrValuesGm[segOff], yOut, outP);
        outQueueY.FreeTensor(yOut);
    }

    // y = alpha * acc + beta * csr, entirely in the vector pipe.
    __aicore__ inline void CombineBatched(AscendC::LocalTensor<float> &yStage,
                                          AscendC::LocalTensor<float> &acc,
                                          AscendC::LocalTensor<float> &csrStage,
                                          uint32_t segNnz) {
        AscendC::Muls(yStage, acc, alphaValue, static_cast<int32_t>(segNnz));
        AscendC::PipeBarrier<PIPE_V>();
        if (betaValue != 0.0f) {
            AscendC::Axpy<float, float>(yStage, csrStage, betaValue,
                                        static_cast<int32_t>(segNnz));
            AscendC::PipeBarrier<PIPE_V>();
        }
    }

    AscendC::TPipe *pipe = nullptr;
    AscendC::TBuf<AscendC::TPosition::VECCALC> aRowBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> bColBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> aFullBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> mulBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> reduceTmpBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> reduceDstBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> colIndBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> accBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> partialBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> stridePadBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> strideOffsetBuf;
    bool needStrideGather = false;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueueCsr;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> outQueueY;
    AscendC::GlobalTensor<float> matAGm;
    AscendC::GlobalTensor<float> matBGm;
    AscendC::GlobalTensor<int32_t> rowOffsetsGm;
    AscendC::GlobalTensor<int32_t> colIndicesGm;
    AscendC::GlobalTensor<float> csrValuesGm;
    uint64_t nnz = 0;
    uint32_t rowCount = 0;
    uint32_t matARows = 0;
    uint32_t matACols = 0;
    uint32_t matBRows = 0;
    uint32_t matBCols = 0;
    uint32_t rowsPerCore = 0;
    uint32_t remainderRows = 0;
    uint32_t kBlockLen = 0;
    uint32_t K = 0;
    uint32_t colCount = 0;
    int64_t opAValue = 0;
    int64_t opBValue = 0;
    float alphaValue = 0.0f;
    float betaValue = 0.0f;
    bool hugeKPath = false;
    bool smallKContigADma = false;
    bool smallKContigBDma = false;
    bool aContig = true;
    bool bContig = true;
    bool batchEnabled = false;
    bool runBatchable = false;
    uint32_t nnzPartition = 0;
    uint32_t nnzPerCore = 0;
    uint32_t remainderNnz = 0;
    uint32_t ldx = 0;
    uint32_t ldy = 0;
    uint32_t orderXCol = 0;
    uint32_t orderYCol = 0;
    uint32_t kbufAligned = 0;
    uint32_t kPitch = 0;
    uint32_t jTile = 0;
};

#include "sddmm_fp16_kernel.h"

#endif // SDDMM_ARCH22_SIMD_KERNEL_H_
