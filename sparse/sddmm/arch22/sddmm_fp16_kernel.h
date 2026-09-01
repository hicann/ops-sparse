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

#ifndef SDDMM_ARCH22_FP16_KERNEL_H_
#define SDDMM_ARCH22_FP16_KERNEL_H_

class KernelSparseSddmmFp16 {
public:
    __aicore__ inline KernelSparseSddmmFp16() {}

    __aicore__ inline void Init(GM_ADDR matA, GM_ADDR matB, GM_ADDR csrRowOffsets,
                                GM_ADDR csrColIndices, GM_ADDR csrValues,
                                const SddmmTilingData &td, AscendC::TPipe *pipeIn) {
        pipe = pipeIn;
        CopyTiling(td);

        matAGm.SetGlobalBuffer((__gm__ half *)matA,
                               static_cast<uint64_t>(matARows) * ldx);
        const uint64_t bGmSize = static_cast<uint64_t>(matBRows) *
                                   ((ldy > matBCols) ? ldy : matBCols);
        matBGm.SetGlobalBuffer((__gm__ half *)matB, bGmSize);
        rowOffsetsGm.SetGlobalBuffer((__gm__ int32_t *)csrRowOffsets,
                                     static_cast<uint64_t>(rowCount + 1));
        colIndicesGm.SetGlobalBuffer((__gm__ int32_t *)csrColIndices, nnz);
        csrValuesGm.SetGlobalBuffer((__gm__ half *)csrValues, nnz);

        const uint32_t kbuf = AlignUp(kBlockLen, kAlign);
        pipe->InitBuffer(aRowBuf, kbuf * sizeof(float));
        pipe->InitBuffer(bColBuf, kbuf * sizeof(float));
        hugeKPath = (matARows > kBlockLen) || (matACols > kBlockLen);
        pipe->InitBuffer(inQueueCsr, 1, kOutStageCap * sizeof(float));
        pipe->InitBuffer(outQueueY, 1, kOutStageCap * sizeof(float));
        pipe->InitBuffer(colIndBuf, kOutStageCap * sizeof(int32_t));
        pipe->InitBuffer(mulBuf, kbuf * sizeof(float));
        pipe->InitBuffer(reduceTmpBuf, kbuf * sizeof(float));
        pipe->InitBuffer(reduceDstBuf, kAlign * sizeof(float));
        pipe->InitBuffer(halfBuf, AlignUp(kBlockLen, static_cast<uint32_t>(16)) * sizeof(half));
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

    // The FP16 path has no small-K DMA specialisation; nothing extra to set up.
    __aicore__ inline bool SetupDmaFlags() {
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
    // Byte-offset table for the strided gather. Gather indexes in element units
    // of the gathered type, so for half the 32B block pitch is 16 elements.
    __aicore__ inline void InitStrideOffsets() {
        AscendC::LocalTensor<int32_t> offs = strideOffsetBuf.Get<int32_t>();
        AscendC::CreateVecIndex<int32_t>(offs, 0, kStrideChunk);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Muls(offs, offs, static_cast<int32_t>(kStrideElemBytes),
                      static_cast<int32_t>(kStrideChunk));
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void SyncMte2ToV() {
        event_t ev = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_V));
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(ev);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(ev);
    }

    __aicore__ inline void Gather16(AscendC::LocalTensor<float> &ub,
                                    AscendC::GlobalTensor<half> &gm,
                                    uint64_t off, uint32_t stride, uint32_t len) {
        if (stride == 1u && len > 0) {
            AscendC::LocalTensor<half> hBuf = halfBuf.Get<half>();
            AscendC::DataCopyExtParams copyParams{1, static_cast<uint32_t>(len * sizeof(half)),
                                                  0, 0, 0};
            AscendC::DataCopyPadExtParams<half> padParams{false, 0, 0, static_cast<half>(0.0f)};
            AscendC::DataCopyPad(hBuf, gm[off], copyParams, padParams);
            AscendC::PipeBarrier<PIPE_MTE2>();
            AscendC::Cast<float, half>(ub, hBuf, AscendC::RoundMode::CAST_NONE, len);
            AscendC::PipeBarrier<PIPE_V>();
            return;
        }
        // Strided: DataCopyPad lands each half on its own 32B block, one Gather
        // compacts them, then a single Cast converts the run to fp32.
        AscendC::LocalTensor<half> pad32 = stridePadBuf.Get<half>();
        AscendC::LocalTensor<half> hBuf = halfBuf.Get<half>();
        AscendC::LocalTensor<int32_t> offs = strideOffsetBuf.Get<int32_t>();
        AscendC::LocalTensor<uint32_t> offsU = offs.ReinterpretCast<uint32_t>();
        const uint32_t srcGapBytes = (stride - 1u) * (uint32_t)sizeof(half);
        for (uint32_t base = 0; base < len; base += kStrideChunk) {
            const uint32_t cnt = MinU32(len - base, kStrideChunk);
            AscendC::DataCopyExtParams cp{static_cast<uint16_t>(cnt), (uint32_t)sizeof(half),
                                          srcGapBytes, 0, 0};
            AscendC::DataCopyPadExtParams<half> pd{false, 0, 0, static_cast<half>(0.0f)};
            AscendC::DataCopyPad(pad32, gm[off + static_cast<uint64_t>(base) * stride], cp, pd);
            SyncMte2ToV();
            AscendC::Gather<half>(hBuf[base], pad32, offsU, 0, cnt);
            AscendC::PipeBarrier<PIPE_V>();
        }
        AscendC::Cast<float, half>(ub, hBuf, AscendC::RoundMode::CAST_NONE, len);
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline float DotProduct(uint64_t aOff, uint32_t aStride, uint64_t bOff,
                                       uint32_t bStride, int aCache) {
        if (K <= kBlockLen) {
            return DotProductSingle(aOff, aStride, bOff, bStride, aCache);
        }
        return DotProductScalar(aOff, aStride, bOff, bStride);
    }

    // Vectorized single-chunk dot product (K <= kBlockLen, proven correct).
    __aicore__ inline float DotProductSingle(uint64_t aOff, uint32_t aStride, uint64_t bOff,
                                             uint32_t bStride, int aCache) {
        AscendC::LocalTensor<float> aC = aRowBuf.Get<float>();
        AscendC::LocalTensor<float> bC = bColBuf.Get<float>();
        AscendC::LocalTensor<float> ml = mulBuf.Get<float>();
        AscendC::LocalTensor<float> rt = reduceTmpBuf.Get<float>();
        AscendC::LocalTensor<float> rd = reduceDstBuf.Get<float>();
        AscendC::LocalTensor<float> aOp = aC;
        if (aCache == kACacheSingle) {
            // A already loaded into aRowBuf
        } else if (aCache == kACacheNone) {
            Gather16(aC, matAGm, aOff, aStride, K);
        }
        Gather16(bC, matBGm, bOff, bStride, K);
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::Mul(ml, aOp, bC, K);
        AscendC::ReduceSum<float, true>(rd, ml, rt, static_cast<int32_t>(K));
        return rd.GetValue(0);
    }

    // Scalar dot product for K > kBlockLen (hugeK FP16 path).
    // Loads A and B elements via chunk-based Gather16 but accumulates the dot
    // product chunk-by-chunk using separate A and B buffers.
    __aicore__ inline float DotProductScalar(uint64_t aOff, uint32_t aStride,
                                             uint64_t bOff, uint32_t bStride) {
        AscendC::LocalTensor<float> aC = aRowBuf.Get<float>();
        AscendC::LocalTensor<float> bC = bColBuf.Get<float>();
        float acc = 0.0f;
        for (uint32_t base = 0; base < K; base += kBlockLen) {
            const uint32_t len = (K - base < kBlockLen) ? (K - base) : kBlockLen;
            Gather16(aC, matAGm, aOff + static_cast<uint64_t>(base) * aStride, aStride, len);
            AscendC::PipeBarrier<PIPE_ALL>();
            Gather16(bC, matBGm, bOff + static_cast<uint64_t>(base) * bStride, bStride, len);
            AscendC::PipeBarrier<PIPE_ALL>();
            for (uint32_t i = 0; i < len; ++i) {
                acc += aC.GetValue(i) * bC.GetValue(i);
            }
        }
        return acc;
    }

    static constexpr float FP16_MAX_VALUE = 65504.0f;

    __aicore__ inline float Saturate(float v) {
        if (v > FP16_MAX_VALUE) {
            return FP16_MAX_VALUE;
        }
        if (v < -FP16_MAX_VALUE) {
            return -FP16_MAX_VALUE;
        }
        return v;
    }

    // Load a full-length vector (up to kHugeKCap) into dst by chunking through
    // halfBuf in kBlockLen-sized pieces, avoiding halfBuf overflow.
    __aicore__ inline void GatherFull16(AscendC::LocalTensor<float> &dst,
                                        AscendC::GlobalTensor<half> &gm,
                                        uint64_t off, uint32_t stride, uint32_t len) {
        for (uint32_t base = 0; base < len; base += kBlockLen) {
            const uint32_t chunk = MinU32(len - base, kBlockLen);
            AscendC::LocalTensor<float> slice = dst[base];
            Gather16(slice, gm, off + static_cast<uint64_t>(base) * stride, stride, chunk);
        }
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
        if (alphaValue != 0.0f && K <= kBlockLen) {
            aCache = kACacheSingle;
            AscendC::LocalTensor<float> a = aRowBuf.Get<float>();
            Gather16(a, matAGm, aOff, aStride, K);
        }
        const int32_t col = colIndicesGm.GetValue(gnnz);
        float dot = 0.0f;
        if (alphaValue != 0.0f && col >= 0 && static_cast<uint32_t>(col) < colCount) {
            const uint64_t bOff = bContig ? static_cast<uint64_t>(col) * ldy
                                          : static_cast<uint64_t>(col);
            dot = DotProduct(aOff, aStride, bOff, bContig ? 1u : ldy, aCache);
        }
        const float cVal = static_cast<float>(csrValuesGm.GetValue(gnnz));
        csrValuesGm.SetValue(gnnz, static_cast<half>(Saturate(alphaValue * dot + betaValue * cVal)));
    }

    // Stage the row's C values (fp16 -> fp32) and column indices into UB.
    __aicore__ inline void LoadRowStage(uint64_t rowOff, uint32_t rowNnz,
                                        AscendC::LocalTensor<float> &csrIn,
                                        AscendC::LocalTensor<int32_t> &colInd) {
        AscendC::LocalTensor<half> hBuf = halfBuf.Get<half>();
        AscendC::DataCopyExtParams p{1, static_cast<uint32_t>(rowNnz * sizeof(half)), 0, 0, 0};
        AscendC::DataCopyPadExtParams<half> pad{false, 0, 0, static_cast<half>(0.0f)};
        AscendC::DataCopyPad(hBuf, csrValuesGm[rowOff], p, pad);
        AscendC::PipeBarrier<PIPE_MTE2>();
        AscendC::Cast<float, half>(csrIn, hBuf, AscendC::RoundMode::CAST_NONE, rowNnz);
        AscendC::PipeBarrier<PIPE_V>();

        AscendC::DataCopyExtParams cp{1, rowNnz * (uint32_t)sizeof(int32_t), 0, 0, 0};
        AscendC::DataCopyPadExtParams<int32_t> cpd{false, 0, 0, 0};
        AscendC::DataCopyPad(colInd, colIndicesGm[rowOff], cp, cpd);
        AscendC::PipeBarrier<PIPE_MTE2>();
    }

    // Load the A row (or column) into UB once; returns the cache mode used.
    __aicore__ inline int LoadARow16(uint64_t aOff, uint32_t aStride) {
        if (alphaValue == 0.0f) {
            return kACacheNone;
        }
        if (K <= kBlockLen) {
            AscendC::LocalTensor<float> a = aRowBuf.Get<float>();
            Gather16(a, matAGm, aOff, aStride, K);
            return kACacheSingle;
        }
        return kACacheNone;
    }

    // Cast the fp32 results back to fp16 (with saturation) and write them out.
    __aicore__ inline void StoreRow16(uint64_t rowOff, uint32_t rowNnz,
                                      AscendC::LocalTensor<float> &yOut) {
        AscendC::LocalTensor<half> hBuf = halfBuf.Get<half>();
        AscendC::Cast<half, float>(hBuf, yOut, AscendC::RoundMode::CAST_RINT, rowNnz);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::DataCopyExtParams p{1, static_cast<uint32_t>(rowNnz * sizeof(half)), 0, 0, 0};
        AscendC::DataCopyPad(csrValuesGm[rowOff], hBuf, p);
        AscendC::PipeBarrier<PIPE_MTE3>();
    }

    __aicore__ inline void ProcessRow(uint32_t row) {
        const int32_t start = rowOffsetsGm.GetValue(row);
        const int32_t end = rowOffsetsGm.GetValue(row + 1);
        const uint32_t rowNnz = static_cast<uint32_t>(end - start);
        if (rowNnz == 0) {
            return;
        }
        const uint64_t rowOff = static_cast<uint64_t>(start);
        const uint64_t aOff = aContig ? (static_cast<uint64_t>(row) * ldx) : static_cast<uint64_t>(row);
        const uint32_t aStride = aContig ? 1u : ldx;

        const int aCache = LoadARow16(aOff, aStride);
        for (uint32_t segBase = 0; segBase < rowNnz; segBase += kOutStageCap) {
            const uint32_t segNnz = MinU32(rowNnz - segBase, kOutStageCap);
            ProcessRowSegment(rowOff + segBase, segNnz, aOff, aStride, aCache);
        }
    }

    // Process one segment (up to kOutStageCap nonzeros) of a row.
    __aicore__ inline void ProcessRowSegment(uint64_t segOff, uint32_t segNnz,
                                              uint64_t aOff, uint32_t aStride, int aCache) {
        AscendC::LocalTensor<float> csrIn = inQueueCsr.AllocTensor<float>();
        AscendC::LocalTensor<int32_t> colInd = colIndBuf.Get<int32_t>();
        LoadRowStage(segOff, segNnz, csrIn, colInd);
        inQueueCsr.EnQue(csrIn);
        AscendC::LocalTensor<float> csrStage = inQueueCsr.DeQue<float>();

        AscendC::LocalTensor<float> yStage = outQueueY.AllocTensor<float>();
        for (uint32_t i = 0; i < segNnz; ++i) {
            const int32_t col = colInd.GetValue(i);
            float dot = 0.0f;
            if (alphaValue != 0.0f && col >= 0 && static_cast<uint32_t>(col) < colCount) {
                const uint64_t bOff = bContig ? (static_cast<uint64_t>(col) * ldy)
                                              : static_cast<uint64_t>(col);
                dot = DotProduct(aOff, aStride, bOff, bContig ? 1u : ldy, aCache);
            }
            yStage.SetValue(i, Saturate(alphaValue * dot + betaValue * csrStage.GetValue(i)));
        }
        inQueueCsr.FreeTensor(csrStage);

        outQueueY.EnQue(yStage);
        AscendC::LocalTensor<float> yOut = outQueueY.DeQue<float>();
        StoreRow16(segOff, segNnz, yOut);
        outQueueY.FreeTensor(yOut);
    }

    AscendC::TPipe *pipe = nullptr;
    AscendC::TBuf<AscendC::TPosition::VECCALC> aRowBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> bColBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> aFullBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> mulBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> reduceTmpBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> reduceDstBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> halfBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> colIndBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> stridePadBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> strideOffsetBuf;
    bool needStrideGather = false;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueueCsr;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> outQueueY;
    AscendC::GlobalTensor<half> matAGm;
    AscendC::GlobalTensor<half> matBGm;
    AscendC::GlobalTensor<int32_t> rowOffsetsGm;
    AscendC::GlobalTensor<int32_t> colIndicesGm;
    AscendC::GlobalTensor<half> csrValuesGm;
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
    bool aContig = true;
    bool bContig = true;
    uint32_t nnzPartition = 0;
    uint32_t nnzPerCore = 0;
    uint32_t remainderNnz = 0;
    uint32_t ldx = 0;
    uint32_t ldy = 0;
    uint32_t orderXCol = 0;
    uint32_t orderYCol = 0;
};

#endif // SDDMM_ARCH22_FP16_KERNEL_H_
