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
 * \file spmm_op_kernel.cpp
 * \brief spmm_op SIMD kernel 实现（仅 arch35/DAV-3510 可用）。
 *
 * 两个 kernel：
 *   Kernel 1 (Compute):   C = alpha * op(A) * op(B) + beta * C
 *   Kernel 2 (BetaC):     nnz==0 时，C = beta * C
 *
 * 模板实例化矩阵（24 特化，由 dispatcher 按 dtype × highPrecision ×
 * opB × orderPair 选择）：
 *   FP32: HighPrecision(true/false) × OpBTranspose × BRowMajor × CRowMajor = 16
 *   FP16: HighPrecision(false only) × OpBTranspose × BRowMajor × CRowMajor = 8
 *   总计：24
 *
 * UseReorder 和 RowOffT 不再是模板参数，改为运行时判断。
 */

#include <stdint.h>
#include <type_traits>
#include "kernel_operator.h"
#include "spmm_op_kernel.h"

using namespace AscendC;

constexpr int32_t STRIDED_CHUNK = 128;
constexpr int32_t STRIDED_ELEM_BYTES = 32;
constexpr float FP16_MAX = 65504.0f;

constexpr int32_t MAX_CSR_BATCH = SPMM_OP_MAX_CSR_BATCH;
constexpr int32_t MAX_ROW_OFF_BATCH = SPMM_OP_MAX_ROW_OFF_BATCH;

template <typename ValT>
__aicore__ inline void StridedLoadGather(
    LocalTensor<ValT> &dst, int32_t dstOffset,
    const GlobalTensor<ValT> &gm, uint64_t gmBase,
    int32_t elemStride, int32_t count,
    LocalTensor<ValT> &stridedBuf, const LocalTensor<int32_t> &offsetInt)
{
    auto offsetUInt = offsetInt.ReinterpretCast<uint32_t>();
    int64_t srcStrideBytes = static_cast<int64_t>(elemStride - 1) * static_cast<int64_t>(sizeof(ValT));
    for (int32_t cBase = 0; cBase < count; cBase += STRIDED_CHUNK) {
        const int32_t cEnd = (cBase + STRIDED_CHUNK < count) ? (cBase + STRIDED_CHUNK) : count;
        const int32_t cSize = cEnd - cBase;
        uint64_t chunkGmBase = gmBase + static_cast<uint64_t>(cBase) * static_cast<uint64_t>(elemStride);
        DataCopyExtParams cp{static_cast<uint16_t>(cSize), static_cast<uint32_t>(sizeof(ValT)),
                              srcStrideBytes, 0, 0};
        DataCopyPadExtParams<ValT> pad{false, 0, 0, 0};
        DataCopyPad(stridedBuf, gm[chunkGmBase], cp, pad);
        PipeBarrier<PIPE_MTE2>();
        Gather(dst[dstOffset + cBase], stridedBuf, offsetUInt, 0, cSize);
        PipeBarrier<PIPE_V>();
    }
}

__aicore__ inline void SpmmOpAlg1RowRange(
    uint32_t rowsPerBlock, int32_t m,
    int32_t &rowStart, int32_t &rowEnd)
{
    uint32_t blockId = GetBlockIdx();
    uint32_t rowStartU = blockId * rowsPerBlock;
    uint32_t rowEndU = rowStartU + rowsPerBlock;
    if (rowEndU > static_cast<uint32_t>(m)) {
        rowEndU = static_cast<uint32_t>(m);
    }
    rowStart = static_cast<int32_t>(rowStartU);
    rowEnd = static_cast<int32_t>(rowEndU);
}

template <typename CT>
__aicore__ inline void SpmmOpWriteBackContiguous(
    GlobalTensor<CT> &matCGm, int32_t row, int32_t colFirst,
    int32_t actualNTile, int32_t ldc,
    LocalTensor<float> &outBuf,
    TBuf<TPosition::VECCALC> *castOutBuf)
{
    static constexpr bool IS_FP16 = std::is_same_v<CT, half>;
    uint64_t cOffset = static_cast<uint64_t>(row) * static_cast<uint64_t>(ldc) +
                       static_cast<uint64_t>(colFirst);
    if constexpr (IS_FP16) {
        Mins<float>(outBuf, outBuf, FP16_MAX, actualNTile);
        PipeBarrier<PIPE_V>();
        Maxs<float>(outBuf, outBuf, -FP16_MAX, actualNTile);
        PipeBarrier<PIPE_V>();
        auto castOut = castOutBuf->Get<CT>();
        Cast<CT, float>(castOut, outBuf, RoundMode::CAST_ROUND, actualNTile);
        PipeBarrier<PIPE_V>();
        DataCopyExtParams cp{1, static_cast<uint32_t>(actualNTile * sizeof(CT)), 0, 0, 0};
        DataCopyPad(matCGm[cOffset], castOut, cp);
        PipeBarrier<PIPE_MTE3>();
    } else {
        DataCopyExtParams cp{1, static_cast<uint32_t>(actualNTile * sizeof(float)), 0, 0, 0};
        DataCopyPad(matCGm[cOffset], outBuf, cp);
        PipeBarrier<PIPE_MTE3>();
    }
}

template <typename CT>
__aicore__ inline void SpmmOpWriteBackStrided(
    GlobalTensor<CT> &matCGm, int32_t row, int32_t colFirst,
    int32_t actualNTile, int32_t ldc,
    LocalTensor<float> &outBuf,
    TBuf<TPosition::VECCALC> *castOutBuf,
    LocalTensor<CT> &stridedBuf, const LocalTensor<uint32_t> &offsetUInt)
{
    static constexpr bool IS_FP16 = std::is_same_v<CT, half>;
    uint64_t cBase = static_cast<uint64_t>(colFirst) * static_cast<uint64_t>(ldc) +
                     static_cast<uint64_t>(row);
    int64_t dstStride = static_cast<int64_t>(ldc - 1) * static_cast<int64_t>(sizeof(CT));
    if constexpr (IS_FP16) {
        Mins<float>(outBuf, outBuf, FP16_MAX, actualNTile);
        PipeBarrier<PIPE_V>();
        Maxs<float>(outBuf, outBuf, -FP16_MAX, actualNTile);
        PipeBarrier<PIPE_V>();
        auto castOut = castOutBuf->Get<CT>();
        Cast<CT, float>(castOut, outBuf, RoundMode::CAST_ROUND, actualNTile);
        PipeBarrier<PIPE_V>();
    }
    // Chunk Scatter+DataCopyPad into blocks of STRIDED_CHUNK to handle nTile > 128.
    // The offset array has STRIDED_CHUNK entries and stridedBuf has STRIDED_CHUNK * STRIDED_ELEM_BYTES bytes.
    for (int32_t cBaseOff = 0; cBaseOff < actualNTile; cBaseOff += STRIDED_CHUNK) {
        const int32_t cEnd = (cBaseOff + STRIDED_CHUNK < actualNTile) ? (cBaseOff + STRIDED_CHUNK) : actualNTile;
        const int32_t cSize = cEnd - cBaseOff;
        if constexpr (IS_FP16) {
            auto castOut = castOutBuf->Get<CT>();
            Scatter(stridedBuf, castOut[cBaseOff], offsetUInt, 0, static_cast<uint32_t>(cSize));
        } else {
            Scatter(stridedBuf, outBuf[cBaseOff], offsetUInt, 0, static_cast<uint32_t>(cSize));
        }
        PipeBarrier<PIPE_V>();
        DataCopyExtParams cp{static_cast<uint16_t>(cSize),
                              static_cast<uint32_t>(sizeof(CT)), 0, dstStride, 0};
        DataCopyPad(matCGm[cBase + static_cast<uint64_t>(cBaseOff) * static_cast<uint64_t>(ldc)], stridedBuf, cp);
        PipeBarrier<PIPE_MTE3>();
    }
}

template <typename ValT, bool HighPrecision, bool OpBTranspose, bool BRowMajor, bool CRowMajor>
class KernelSpmmOpSimd {
    static constexpr bool IS_FP16 = std::is_same_v<ValT, half>;
    static constexpr bool CONTIG_B = (BRowMajor != OpBTranspose);

public:
    __aicore__ inline KernelSpmmOpSimd() {}

    __aicore__ inline void Init(
        GM_ADDR gmRowOffsets, GM_ADDR gmColInd, GM_ADDR gmValues,
        GM_ADDR gmMatB, GM_ADDR gmMatC,
        GM_ADDR gmReorder, GM_ADDR gmBinEdge,
        const SpmmOpTilingData *tiling, TPipe *pipe)
    {
        pipe_ = pipe;

        m_ = tiling->m;
        n_ = tiling->n;
        k_ = tiling->k;
        indexBase_ = tiling->indexBase;
        rowOffsetType_ = tiling->rowOffsetType;
        ldb_ = tiling->ldb;
        ldc_ = tiling->ldc;
        nTile_ = tiling->nTile;
        if (nTile_ <= 0) {
            nTile_ = SPMM_OP_N_TILE;
        }
        rowsPerBlock_ = tiling->rowsPerBlock;

        // alpha/beta：DEVICE mode 从 GM 读，HOST mode 从 tiling 读
        if (tiling->alphaPtr != 0ULL && tiling->betaPtr != 0ULL) {
            GlobalTensor<float> alphaGm;
            alphaGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(tiling->alphaPtr), 1);
            alpha_ = alphaGm.GetValue(0);
            GlobalTensor<float> betaGm;
            betaGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(tiling->betaPtr), 1);
            beta_ = betaGm.GetValue(0);
        } else {
            alpha_ = tiling->alpha;
            beta_ = tiling->beta;
        }
        betaZero_ = (beta_ == 0.0f);

        InitGlobalTensors(gmRowOffsets, gmColInd, gmValues,
                          gmMatB, gmMatC, gmReorder, gmBinEdge);
        InitUbBuffers();
    }

private:
    __aicore__ inline void InitGlobalTensors(
        GM_ADDR gmRowOffsets, GM_ADDR gmColInd, GM_ADDR gmValues,
        GM_ADDR gmMatB, GM_ADDR gmMatC,
        GM_ADDR gmReorder, GM_ADDR gmBinEdge)
    {
        gmRowOffsets_ = gmRowOffsets;
        gmReorder_ = gmReorder;
        gmBinEdge_ = gmBinEdge;
        if (gmReorder != nullptr) {
            reorderGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(gmReorder),
                                        static_cast<uint64_t>(m_));
        }
        if (gmBinEdge != nullptr) {
            binEdgeGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(gmBinEdge),
                                       static_cast<uint64_t>(m_) + 1ULL);
        }
        colIndGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(gmColInd),
                                   static_cast<uint64_t>(m_) * static_cast<uint64_t>(k_ > 0 ? k_ : 1));
        valuesGm_.SetGlobalBuffer(reinterpret_cast<__gm__ ValT *>(gmValues),
                                   static_cast<uint64_t>(m_) * static_cast<uint64_t>(k_ > 0 ? k_ : 1));

        uint64_t bElemCount = CONTIG_B ?
            static_cast<uint64_t>(k_) * static_cast<uint64_t>(ldb_) :
            static_cast<uint64_t>(n_) * static_cast<uint64_t>(ldb_);
        if (bElemCount == 0) { bElemCount = 1; }
        matBGm_.SetGlobalBuffer(reinterpret_cast<__gm__ ValT *>(gmMatB), bElemCount);

        uint64_t cElemCount = CRowMajor ?
            static_cast<uint64_t>(m_) * static_cast<uint64_t>(ldc_) :
            static_cast<uint64_t>(n_) * static_cast<uint64_t>(ldc_);
        if (cElemCount == 0) { cElemCount = 1; }
        matCGm_.SetGlobalBuffer(reinterpret_cast<__gm__ ValT *>(gmMatC), cElemCount);
    }

public:
    __aicore__ inline void Process()
    {
        int32_t rowStart{};
        int32_t rowEnd{};
        ComputeRowRange(rowStart, rowEnd);
        BatchLoadRowOffsets();

        for (int32_t logicalRow = rowStart; logicalRow < rowEnd; ++logicalRow) {
            int32_t origRow = logicalRow;
            if (gmReorder_ != nullptr) {
                origRow = reorderGm_.GetValue(logicalRow);
            }
            int32_t rStart = 0;
            int32_t rEnd = 0;
            ReadRowOffsets(origRow, rStart, rEnd);

            bool useCsrBatch = BatchLoadCsrData(rStart, rEnd);

            int32_t colFirst = 0;
            while (colFirst < n_) {
                int32_t actualNTile = (n_ - colFirst < nTile_) ? (n_ - colFirst) : nTile_;
                ProcessTile(origRow, rStart, rEnd, colFirst, actualNTile, useCsrBatch);
                colFirst += actualNTile;
            }
        }
    }

private:
    __aicore__ inline void ComputeRowRange(int32_t &rowStart, int32_t &rowEnd)
    {
        // ALG2: 使用 binEdge 覆盖行范围
        if (gmBinEdge_ != nullptr) {
            uint32_t blockId = GetBlockIdx();
            rowStart = binEdgeGm_.GetValue(blockId);
            rowEnd = binEdgeGm_.GetValue(blockId + 1);
            return;
        }
        // ALG1: 均匀切分
        SpmmOpAlg1RowRange(rowsPerBlock_, m_, rowStart, rowEnd);
    }

    __aicore__ inline void ReadRowOffsets(int32_t origRow, int32_t &rStart, int32_t &rEnd)
    {
        if (rowOffInUb_) {
            if (rowOffsetType_ == SPMM_OP_IDX_RT_I64) {
                auto rowOffUb = rowOffBuf_.Get<int64_t>();
                int64_t rStart64 = rowOffUb.GetValue(origRow) - static_cast<int64_t>(indexBase_);
                int64_t rEnd64 = rowOffUb.GetValue(origRow + 1) - static_cast<int64_t>(indexBase_);
                rStart = static_cast<int32_t>(rStart64);
                rEnd = static_cast<int32_t>(rEnd64);
            } else {
                auto rowOffUb = rowOffBuf_.Get<int32_t>();
                rStart = rowOffUb.GetValue(origRow) - indexBase_;
                rEnd = rowOffUb.GetValue(origRow + 1) - indexBase_;
            }
        } else {
            // 回退路径：从 GM 读取
            if (rowOffsetType_ == SPMM_OP_IDX_RT_I64) {
                GlobalTensor<int64_t> rowOffGm;
                rowOffGm.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(gmRowOffsets_),
                                           static_cast<uint64_t>(m_) + 1ULL);
                int64_t rStart64 = rowOffGm.GetValue(origRow) - static_cast<int64_t>(indexBase_);
                int64_t rEnd64 = rowOffGm.GetValue(origRow + 1) - static_cast<int64_t>(indexBase_);
                rStart = static_cast<int32_t>(rStart64);
                rEnd = static_cast<int32_t>(rEnd64);
            } else {
                GlobalTensor<int32_t> rowOffGm;
                rowOffGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(gmRowOffsets_),
                                           static_cast<uint64_t>(m_) + 1ULL);
                rStart = rowOffGm.GetValue(origRow) - indexBase_;
                rEnd = rowOffGm.GetValue(origRow + 1) - indexBase_;
            }
        }
    }

    __aicore__ inline void InitUbBuffers()
    {
        const uint32_t nTileBytes = static_cast<uint32_t>(nTile_) * sizeof(ValT);
        const uint32_t nTileFp32Bytes = static_cast<uint32_t>(nTile_) * sizeof(float);

        // B 加载用 TQue（深度 1，提供 MTE2→V 自动同步）
        pipe_->InitBuffer(bQue_, 1, nTileBytes);

        pipe_->InitBuffer(accBuf_, nTileFp32Bytes);
        pipe_->InitBuffer(mulBuf_, nTileFp32Bytes);
        pipe_->InitBuffer(outBuf_, nTileFp32Bytes);
        pipe_->InitBuffer(cOldBuf_, nTileBytes);

        if constexpr (HighPrecision) {
            pipe_->InitBuffer(compBuf_, nTileFp32Bytes);
            pipe_->InitBuffer(yBuf_, nTileFp32Bytes);
            pipe_->InitBuffer(tBuf_, nTileFp32Bytes);
            pipe_->InitBuffer(tmpBuf_, nTileFp32Bytes);
        }

        if constexpr (IS_FP16) {
            pipe_->InitBuffer(bFp32Buf_, nTileFp32Bytes);
            pipe_->InitBuffer(cOldFp32Buf_, nTileFp32Bytes);
            pipe_->InitBuffer(castOutBuf_, nTileBytes);
        }

        pipe_->InitBuffer(stridedBuf_, static_cast<uint32_t>(STRIDED_CHUNK * STRIDED_ELEM_BYTES));
        pipe_->InitBuffer(offsetBuf_, static_cast<uint32_t>(STRIDED_CHUNK * sizeof(int32_t)));

        // CSR 批量加载 buffer（API-1/PERF-1 修复）
        const int32_t maxCsrBatch = (k_ < MAX_CSR_BATCH) ? k_ : MAX_CSR_BATCH;
        pipe_->InitBuffer(colIndBuf_, static_cast<uint32_t>(maxCsrBatch) * sizeof(int32_t));
        pipe_->InitBuffer(valuesBuf_, static_cast<uint32_t>(maxCsrBatch) * sizeof(ValT));

        // rowOffsets 批量加载 buffer（PERF-6 修复）
        const int32_t maxRowOff = (m_ < MAX_ROW_OFF_BATCH) ? (m_ + 1) : 0;
        if (maxRowOff > 0) {
            pipe_->InitBuffer(rowOffBuf_, static_cast<uint32_t>(maxRowOff) * sizeof(int64_t));
        }

        InitOffsetTable();
    }

    __aicore__ inline void InitOffsetTable()
    {
        auto offsetInt = offsetBuf_.Get<int32_t>();
        ArithProgression<int32_t>(offsetInt, 0, STRIDED_ELEM_BYTES, STRIDED_CHUNK);
        PipeBarrier<PIPE_V>();
    }

    template <typename ColIndTensor, typename ValuesTensor>
    __aicore__ inline void ProcessCsrElements(
        int32_t pStart, int32_t pEnd,
        int32_t colFirst, int32_t actualNTile,
        ColIndTensor &colInd, ValuesTensor &values,
        LocalTensor<float> &accBuf,
        LocalTensor<ValT> &stridedBuf, const LocalTensor<int32_t> &offsetInt)
    {
        for (int32_t p = pStart; p < pEnd; ++p) {
            int32_t c = colInd.GetValue(p) - indexBase_;
            if (c < 0 || c >= k_) {
                continue;
            }
            float v = static_cast<float>(values.GetValue(p));

            auto bBuf = bQue_.AllocTensor<ValT>();
            LoadBFragment(bBuf, c, colFirst, actualNTile, stridedBuf, offsetInt);
            bQue_.EnQue(bBuf);
            auto bIn = bQue_.DeQue<ValT>();

            auto mulBuf = mulBuf_.Get<float>();
            if constexpr (IS_FP16) {
                auto bFp32Buf = bFp32Buf_.Get<float>();
                Cast<float, half>(bFp32Buf, bIn, RoundMode::CAST_NONE, actualNTile);
                PipeBarrier<PIPE_V>();
                Muls<float>(mulBuf, bFp32Buf, v, actualNTile);
            } else {
                Muls<float>(mulBuf, bIn, v, actualNTile);
            }
            PipeBarrier<PIPE_V>();

            if constexpr (HighPrecision) {
                KahanAccumulate(accBuf, mulBuf, actualNTile);
            } else {
                Add<float>(accBuf, accBuf, mulBuf, actualNTile);
                PipeBarrier<PIPE_V>();
            }

            bQue_.FreeTensor(bIn);
        }
    }

    __aicore__ inline void BatchLoadRowOffsets()
    {
        if (m_ >= MAX_ROW_OFF_BATCH) { return; }
        rowOffInUb_ = true;
        int32_t offCount = m_ + 1;
        if (rowOffsetType_ == SPMM_OP_IDX_RT_I64) {
            auto rowOffUb = rowOffBuf_.Get<int64_t>();
            DataCopyExtParams cp{1, static_cast<uint32_t>(offCount * sizeof(int64_t)), 0, 0, 0};
            DataCopyPadExtParams<int64_t> pad{false, 0, 0, 0};
            GlobalTensor<int64_t> rowOffGm;
            rowOffGm.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(gmRowOffsets_),
                                     static_cast<uint64_t>(offCount));
            DataCopyPad(rowOffUb, rowOffGm, cp, pad);
            PipeBarrier<PIPE_MTE2>();
        } else {
            auto rowOffUb = rowOffBuf_.Get<int32_t>();
            DataCopyExtParams cp{1, static_cast<uint32_t>(offCount * sizeof(int32_t)), 0, 0, 0};
            DataCopyPadExtParams<int32_t> pad{false, 0, 0, 0};
            GlobalTensor<int32_t> rowOffGm;
            rowOffGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(gmRowOffsets_),
                                     static_cast<uint64_t>(offCount));
            DataCopyPad(rowOffUb, rowOffGm, cp, pad);
            PipeBarrier<PIPE_MTE2>();
        }
    }

    __aicore__ inline bool BatchLoadCsrData(int32_t rStart, int32_t rEnd)
    {
        int32_t rowNnz = rEnd - rStart;
        if (rowNnz <= 0 || rowNnz > MAX_CSR_BATCH) { return false; }
        auto colIndUb = colIndBuf_.Get<int32_t>();
        auto valuesUb = valuesBuf_.Get<ValT>();
        DataCopyExtParams cpCI{1, static_cast<uint32_t>(rowNnz * sizeof(int32_t)), 0, 0, 0};
        DataCopyPadExtParams<int32_t> padCI{false, 0, 0, 0};
        DataCopyPad(colIndUb, colIndGm_[rStart], cpCI, padCI);
        DataCopyExtParams cpV{1, static_cast<uint32_t>(rowNnz * sizeof(ValT)), 0, 0, 0};
        DataCopyPadExtParams<ValT> padV{false, 0, 0, 0};
        DataCopyPad(valuesUb, valuesGm_[rStart], cpV, padV);
        PipeBarrier<PIPE_MTE2>();
        return true;
    }

    __aicore__ inline void ProcessTile(
        int32_t origRow, int32_t rStart, int32_t rEnd,
        int32_t colFirst, int32_t actualNTile, bool useCsrBatch)
    {
        auto accBuf = accBuf_.Get<float>();
        auto stridedBuf = stridedBuf_.Get<ValT>();
        auto offsetInt = offsetBuf_.Get<int32_t>();

        Duplicate<float>(accBuf, 0.0f, nTile_);
        PipeBarrier<PIPE_V>();

        if constexpr (HighPrecision) {
            auto compBuf = compBuf_.Get<float>();
            Duplicate<float>(compBuf, 0.0f, nTile_);
            PipeBarrier<PIPE_V>();
        }

        if (useCsrBatch) {
            auto colIndUb = colIndBuf_.Get<int32_t>();
            auto valuesUb = valuesBuf_.Get<ValT>();
            ProcessCsrElements(0, rEnd - rStart, colFirst, actualNTile,
                               colIndUb, valuesUb, accBuf, stridedBuf, offsetInt);
        } else {
            ProcessCsrElements(rStart, rEnd, colFirst, actualNTile,
                               colIndGm_, valuesGm_, accBuf, stridedBuf, offsetInt);
        }

        WriteBackC(origRow, colFirst, actualNTile, accBuf, stridedBuf, offsetInt);
    }

    __aicore__ inline void LoadBFragment(
        LocalTensor<ValT> &bBuf, int32_t c, int32_t colFirst, int32_t actualNTile,
        LocalTensor<ValT> &stridedBuf, const LocalTensor<int32_t> &offsetInt)
    {
        if constexpr (CONTIG_B) {
            uint64_t offset = static_cast<uint64_t>(c) * static_cast<uint64_t>(ldb_) +
                              static_cast<uint64_t>(colFirst);
            DataCopyExtParams cp{1, static_cast<uint32_t>(actualNTile * sizeof(ValT)), 0, 0, 0};
            DataCopyPadExtParams<ValT> pad{false, 0, 0, 0};
            DataCopyPad(bBuf, matBGm_[offset], cp, pad);
        } else {
            uint64_t gmBase = static_cast<uint64_t>(colFirst) * static_cast<uint64_t>(ldb_) +
                              static_cast<uint64_t>(c);
            StridedLoadGather(bBuf, 0, matBGm_, gmBase, ldb_, actualNTile,
                              stridedBuf, offsetInt);
        }
    }

    __aicore__ inline void KahanAccumulate(
        LocalTensor<float> &accBuf, const LocalTensor<float> &mulBuf, int32_t actualNTile)
    {
        auto compBuf = compBuf_.Get<float>();
        auto yBuf = yBuf_.Get<float>();
        auto tBuf = tBuf_.Get<float>();
        auto tmpBuf = tmpBuf_.Get<float>();

        Sub<float>(yBuf, mulBuf, compBuf, actualNTile);
        PipeBarrier<PIPE_V>();
        Add<float>(tBuf, accBuf, yBuf, actualNTile);
        PipeBarrier<PIPE_V>();
        Sub<float>(tmpBuf, tBuf, accBuf, actualNTile);
        PipeBarrier<PIPE_V>();
        Sub<float>(compBuf, tmpBuf, yBuf, actualNTile);
        PipeBarrier<PIPE_V>();
        Adds<float>(accBuf, tBuf, 0.0f, actualNTile);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void ApplyBetaC(
        int32_t origRow, int32_t colFirst, int32_t actualNTile,
        LocalTensor<float> &outBuf,
        LocalTensor<ValT> &stridedBuf, const LocalTensor<int32_t> &offsetInt)
    {
        auto cOldBuf = cOldBuf_.Get<ValT>();
        if constexpr (CRowMajor) {
            uint64_t cOffset = static_cast<uint64_t>(origRow) * static_cast<uint64_t>(ldc_) +
                               static_cast<uint64_t>(colFirst);
            DataCopyExtParams cp{1, static_cast<uint32_t>(actualNTile * sizeof(ValT)), 0, 0, 0};
            DataCopyPadExtParams<ValT> pad{false, 0, 0, 0};
            DataCopyPad(cOldBuf, matCGm_[cOffset], cp, pad);
            PipeBarrier<PIPE_MTE2>();
        } else {
            uint64_t gmBase = static_cast<uint64_t>(colFirst) * static_cast<uint64_t>(ldc_) +
                               static_cast<uint64_t>(origRow);
            StridedLoadGather<ValT>(cOldBuf, 0, matCGm_, gmBase, ldc_, actualNTile,
                                    stridedBuf, offsetInt);
        }

        if constexpr (IS_FP16) {
            auto cOldFp32Buf = cOldFp32Buf_.Get<float>();
            Cast<float, half>(cOldFp32Buf, cOldBuf, RoundMode::CAST_NONE, actualNTile);
            PipeBarrier<PIPE_V>();
            Muls<float>(cOldFp32Buf, cOldFp32Buf, beta_, actualNTile);
            PipeBarrier<PIPE_V>();
            Add<float>(outBuf, outBuf, cOldFp32Buf, actualNTile);
        } else {
            Muls<float>(cOldBuf, cOldBuf, beta_, actualNTile);
            PipeBarrier<PIPE_V>();
            Add<float>(outBuf, outBuf, cOldBuf, actualNTile);
        }
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void WriteBackC(
        int32_t origRow, int32_t colFirst, int32_t actualNTile,
        const LocalTensor<float> &accBuf,
        LocalTensor<ValT> &stridedBuf, const LocalTensor<int32_t> &offsetInt)
    {
        auto outBuf = outBuf_.Get<float>();

        Muls<float>(outBuf, accBuf, alpha_, actualNTile);
        PipeBarrier<PIPE_V>();

        if (!betaZero_) {
            ApplyBetaC(origRow, colFirst, actualNTile, outBuf, stridedBuf, offsetInt);
        }

        if constexpr (CRowMajor) {
            if constexpr (IS_FP16) {
                SpmmOpWriteBackContiguous<ValT>(matCGm_, origRow, colFirst, actualNTile, ldc_,
                                                outBuf, &castOutBuf_);
            } else {
                SpmmOpWriteBackContiguous<ValT>(matCGm_, origRow, colFirst, actualNTile, ldc_,
                                                outBuf, nullptr);
            }
        } else {
            auto offsetUInt = offsetInt.ReinterpretCast<uint32_t>();
            if constexpr (IS_FP16) {
                SpmmOpWriteBackStrided<ValT>(matCGm_, origRow, colFirst, actualNTile, ldc_,
                                             outBuf, &castOutBuf_, stridedBuf, offsetUInt);
            } else {
                SpmmOpWriteBackStrided<ValT>(matCGm_, origRow, colFirst, actualNTile, ldc_,
                                             outBuf, nullptr, stridedBuf, offsetUInt);
            }
        }
    }

private:
    TPipe *pipe_{nullptr};

    GlobalTensor<ValT> matBGm_;
    GlobalTensor<ValT> matCGm_;

    GM_ADDR gmRowOffsets_{nullptr};
    GM_ADDR gmReorder_{nullptr};
    GM_ADDR gmBinEdge_{nullptr};
    GlobalTensor<int32_t> reorderGm_;
    GlobalTensor<int32_t> binEdgeGm_;
    GlobalTensor<int32_t> colIndGm_;
    GlobalTensor<ValT> valuesGm_;

    TQue<TPosition::VECIN, 1> bQue_;
    TBuf<TPosition::VECCALC> accBuf_;
    TBuf<TPosition::VECCALC> mulBuf_;
    TBuf<TPosition::VECCALC> outBuf_;
    TBuf<TPosition::VECCALC> cOldBuf_;

    TBuf<TPosition::VECCALC> compBuf_;
    TBuf<TPosition::VECCALC> yBuf_;
    TBuf<TPosition::VECCALC> tBuf_;
    TBuf<TPosition::VECCALC> tmpBuf_;

    TBuf<TPosition::VECCALC> bFp32Buf_;
    TBuf<TPosition::VECCALC> cOldFp32Buf_;
    TBuf<TPosition::VECCALC> castOutBuf_;

    TBuf<TPosition::VECCALC> stridedBuf_;
    TBuf<TPosition::VECCALC> offsetBuf_;

    TBuf<TPosition::VECCALC> colIndBuf_;
    TBuf<TPosition::VECCALC> valuesBuf_;
    TBuf<TPosition::VECCALC> rowOffBuf_;
    bool rowOffInUb_{false};

    int32_t m_{0};
    int32_t n_{0};
    int32_t k_{0};
    int32_t indexBase_{0};
    int32_t rowOffsetType_{SPMM_OP_IDX_RT_I32};
    int32_t ldb_{0};
    int32_t ldc_{0};
    int32_t nTile_{SPMM_OP_N_TILE};
    uint32_t rowsPerBlock_{0};
    float alpha_{1.0f};
    float beta_{0.0f};
    bool betaZero_{true};
};

template <typename CT>
class KernelSpmmOpBetaCSimd {
    static constexpr bool IS_FP16 = std::is_same_v<CT, half>;

public:
    __aicore__ inline KernelSpmmOpBetaCSimd() {}

    __aicore__ inline void Init(
        GM_ADDR gmMatC,
        const SpmmOpTilingData *tiling, TPipe *pipe)
    {
        pipe_ = pipe;
        m_ = tiling->m;
        n_ = tiling->n;
        ldc_ = tiling->ldc;
        orderPair_ = tiling->orderPair;
        nTile_ = tiling->nTile;
        if (nTile_ <= 0) {
            nTile_ = SPMM_OP_N_TILE;
        }
        rowsPerBlock_ = tiling->rowsPerBlock;

        // beta：DEVICE mode 从 GM 读，HOST mode 从 tiling 读
        if (tiling->alphaPtr != 0ULL && tiling->betaPtr != 0ULL) {
            GlobalTensor<float> betaGm;
            betaGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(tiling->betaPtr), 1);
            beta_ = betaGm.GetValue(0);
        } else {
            beta_ = tiling->beta;
        }
        betaZero_ = (beta_ == 0.0f);

        const bool cRowMajor = (orderPair_ == SPMM_OP_ORDER_RR ||
                                orderPair_ == SPMM_OP_ORDER_CR);
        uint64_t cElemCount = cRowMajor ?
            static_cast<uint64_t>(m_) * static_cast<uint64_t>(ldc_) :
            static_cast<uint64_t>(n_) * static_cast<uint64_t>(ldc_);
        if (cElemCount == 0) { cElemCount = 1; }
        matCGm_.SetGlobalBuffer(reinterpret_cast<__gm__ CT *>(gmMatC), cElemCount);

        const uint32_t nTileBytes = static_cast<uint32_t>(nTile_) * sizeof(CT);
        const uint32_t nTileFp32Bytes = static_cast<uint32_t>(nTile_) * sizeof(float);
        pipe_->InitBuffer(outBuf_, nTileFp32Bytes);
        pipe_->InitBuffer(cBuf_, nTileBytes);
        if constexpr (IS_FP16) {
            pipe_->InitBuffer(castOutBuf_, nTileBytes);
        }
        pipe_->InitBuffer(stridedBuf_, static_cast<uint32_t>(STRIDED_CHUNK * STRIDED_ELEM_BYTES));
        pipe_->InitBuffer(offsetBuf_, static_cast<uint32_t>(STRIDED_CHUNK * sizeof(int32_t)));
        auto offsetInt = offsetBuf_.Get<int32_t>();
        ArithProgression<int32_t>(offsetInt, 0, STRIDED_ELEM_BYTES, STRIDED_CHUNK);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void Process()
    {
        const bool cRowMajor = (orderPair_ == SPMM_OP_ORDER_RR ||
                                orderPair_ == SPMM_OP_ORDER_CR);

        int32_t rowStart{};
        int32_t rowEnd{};
        SpmmOpAlg1RowRange(rowsPerBlock_, m_, rowStart, rowEnd);

        for (int32_t row = rowStart; row < rowEnd; ++row) {
            int32_t colFirst = 0;
            while (colFirst < n_) {
                int32_t actualNTile = (n_ - colFirst < nTile_) ? (n_ - colFirst) : nTile_;
                ProcessTile(row, colFirst, actualNTile, cRowMajor);
                colFirst += actualNTile;
            }
        }
    }

private:
    __aicore__ inline void ProcessTile(
        int32_t row, int32_t colFirst, int32_t actualNTile, bool cRowMajor)
    {
        auto outBuf = outBuf_.Get<float>();
        auto stridedBuf = stridedBuf_.Get<CT>();
        auto offsetInt = offsetBuf_.Get<int32_t>();
        auto offsetUInt = offsetInt.ReinterpretCast<uint32_t>();

        if (betaZero_) {
            Duplicate<float>(outBuf, 0.0f, actualNTile);
            PipeBarrier<PIPE_V>();
        } else {
            auto cBuf = cBuf_.Get<CT>();
            if (cRowMajor) {
                uint64_t cOffset = static_cast<uint64_t>(row) * static_cast<uint64_t>(ldc_) +
                                   static_cast<uint64_t>(colFirst);
                DataCopyExtParams cp{1, static_cast<uint32_t>(actualNTile * sizeof(CT)), 0, 0, 0};
                DataCopyPadExtParams<CT> pad{false, 0, 0, 0};
                DataCopyPad(cBuf, matCGm_[cOffset], cp, pad);
                PipeBarrier<PIPE_MTE2>();
            } else {
                uint64_t gmBase = static_cast<uint64_t>(colFirst) * static_cast<uint64_t>(ldc_) +
                                   static_cast<uint64_t>(row);
                StridedLoadGather<CT>(cBuf, 0, matCGm_, gmBase, ldc_, actualNTile,
                                      stridedBuf, offsetInt);
            }

            if constexpr (IS_FP16) {
                Cast<float, half>(outBuf, cBuf, RoundMode::CAST_NONE, actualNTile);
                PipeBarrier<PIPE_V>();
                Muls<float>(outBuf, outBuf, beta_, actualNTile);
            } else {
                Muls<float>(outBuf, cBuf, beta_, actualNTile);
            }
            PipeBarrier<PIPE_V>();
        }

        if (cRowMajor) {
            if constexpr (IS_FP16) {
                SpmmOpWriteBackContiguous<CT>(matCGm_, row, colFirst, actualNTile, ldc_,
                                              outBuf, &castOutBuf_);
            } else {
                SpmmOpWriteBackContiguous<CT>(matCGm_, row, colFirst, actualNTile, ldc_,
                                              outBuf, nullptr);
            }
        } else {
            if constexpr (IS_FP16) {
                SpmmOpWriteBackStrided<CT>(matCGm_, row, colFirst, actualNTile, ldc_,
                                           outBuf, &castOutBuf_, stridedBuf, offsetUInt);
            } else {
                SpmmOpWriteBackStrided<CT>(matCGm_, row, colFirst, actualNTile, ldc_,
                                           outBuf, nullptr, stridedBuf, offsetUInt);
            }
        }
    }

private:
    TPipe *pipe_{nullptr};
    GlobalTensor<CT> matCGm_;

    TBuf<TPosition::VECCALC> outBuf_;
    TBuf<TPosition::VECCALC> cBuf_;
    TBuf<TPosition::VECCALC> castOutBuf_;
    TBuf<TPosition::VECCALC> stridedBuf_;
    TBuf<TPosition::VECCALC> offsetBuf_;

    int32_t m_{0};
    int32_t n_{0};
    int32_t ldc_{0};
    int32_t orderPair_{0};
    int32_t nTile_{SPMM_OP_N_TILE};
    uint32_t rowsPerBlock_{0};
    float beta_{0.0f};
    bool betaZero_{true};
};

class SpmmOpSimdDispatcher {
public:
    __aicore__ inline void Init(
        GM_ADDR gmRowOffsets, GM_ADDR gmColInd, GM_ADDR gmValues,
        GM_ADDR gmMatB, GM_ADDR gmMatC,
        GM_ADDR gmReorder, GM_ADDR gmBinEdge,
        const SpmmOpTilingData *tiling, TPipe *pipe)
    {
        pipe_ = pipe;
        gmRowOffsets_ = gmRowOffsets;
        gmColInd_ = gmColInd;
        gmValues_ = gmValues;
        gmMatB_ = gmMatB;
        gmMatC_ = gmMatC;
        gmReorder_ = gmReorder;
        gmBinEdge_ = gmBinEdge;
        tiling_ = tiling;

        m_ = tiling->m;
        n_ = tiling->n;
        dtype_ = tiling->dtype;
        highPrecision_ = tiling->highPrecision;
        opB_ = tiling->opB;
        orderPair_ = tiling->orderPair;
        rowsPerBlock_ = tiling->rowsPerBlock;
    }

    __aicore__ inline void Process()
    {
        int32_t rowStart{};
        int32_t rowEnd{};
        ComputeRowRange(rowStart, rowEnd);
        if (rowStart >= rowEnd) {
            return;
        }
        DispatchCompute(rowStart, rowEnd);
    }

private:
    __aicore__ inline void ComputeRowRange(int32_t &rowStart, int32_t &rowEnd)
    {
        // ALG2: 使用 binEdge 覆盖行范围
        if (gmBinEdge_ != nullptr) {
            GlobalTensor<int32_t> binEdgeGm;
            binEdgeGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(gmBinEdge_),
                                       static_cast<uint64_t>(m_) + 1ULL);
            uint32_t blockId = GetBlockIdx();
            rowStart = binEdgeGm.GetValue(blockId);
            rowEnd = binEdgeGm.GetValue(blockId + 1);
            return;
        }
        // ALG1: 均匀切分
        SpmmOpAlg1RowRange(rowsPerBlock_, m_, rowStart, rowEnd);
    }

    __aicore__ inline void DispatchCompute(int32_t rowStart, int32_t rowEnd)
    {
        const bool opBT = (opB_ == SPMM_OP_OPB_TRANSPOSE);
        const bool bRowMajor = (orderPair_ == SPMM_OP_ORDER_RR ||
                                orderPair_ == SPMM_OP_ORDER_RC);
        const bool cRowMajor = (orderPair_ == SPMM_OP_ORDER_RR ||
                                orderPair_ == SPMM_OP_ORDER_CR);

        if (dtype_ == SPMM_OP_DTYPE_FP32) {
            if (highPrecision_ != 0) {
                DispatchTyped<float, true>(rowStart, rowEnd, opBT, bRowMajor, cRowMajor);
            } else {
                DispatchTyped<float, false>(rowStart, rowEnd, opBT, bRowMajor, cRowMajor);
            }
        } else {
            // FP16: HighPrecision 始终 false
            DispatchTyped<half, false>(rowStart, rowEnd, opBT, bRowMajor, cRowMajor);
        }
    }

    template <typename ValT, bool HighPrecision>
    __aicore__ inline void DispatchTyped(
        int32_t rowStart, int32_t rowEnd,
        bool opBT, bool bRowMajor, bool cRowMajor)
    {
        if (opBT) {
            if (bRowMajor) {
                if (cRowMajor) {
                    RunKernel<ValT, HighPrecision, true, true, true>(rowStart, rowEnd);
                } else {
                    RunKernel<ValT, HighPrecision, true, true, false>(rowStart, rowEnd);
                }
            } else {
                if (cRowMajor) {
                    RunKernel<ValT, HighPrecision, true, false, true>(rowStart, rowEnd);
                } else {
                    RunKernel<ValT, HighPrecision, true, false, false>(rowStart, rowEnd);
                }
            }
        } else {
            if (bRowMajor) {
                if (cRowMajor) {
                    RunKernel<ValT, HighPrecision, false, true, true>(rowStart, rowEnd);
                } else {
                    RunKernel<ValT, HighPrecision, false, true, false>(rowStart, rowEnd);
                }
            } else {
                if (cRowMajor) {
                    RunKernel<ValT, HighPrecision, false, false, true>(rowStart, rowEnd);
                } else {
                    RunKernel<ValT, HighPrecision, false, false, false>(rowStart, rowEnd);
                }
            }
        }
    }

    template <typename ValT, bool HighPrecision, bool OpBT, bool BRM, bool CRM>
    __aicore__ inline void RunKernel(int32_t rowStart, int32_t rowEnd)
    {
        KernelSpmmOpSimd<ValT, HighPrecision, OpBT, BRM, CRM> op;
        op.Init(gmRowOffsets_, gmColInd_, gmValues_,
                gmMatB_, gmMatC_,
                gmReorder_, gmBinEdge_,
                tiling_, pipe_);
        op.Process();
    }

private:
    TPipe *pipe_{nullptr};
    const SpmmOpTilingData *tiling_{nullptr};
    GM_ADDR gmRowOffsets_{nullptr};
    GM_ADDR gmColInd_{nullptr};
    GM_ADDR gmValues_{nullptr};
    GM_ADDR gmMatB_{nullptr};
    GM_ADDR gmMatC_{nullptr};
    GM_ADDR gmReorder_{nullptr};
    GM_ADDR gmBinEdge_{nullptr};

    int32_t m_{0};
    int32_t n_{0};
    int32_t dtype_{SPMM_OP_DTYPE_FP32};
    int32_t highPrecision_{0};
    int32_t opB_{0};
    int32_t orderPair_{0};
    uint32_t rowsPerBlock_{0};
};

class SpmmOpBetaCSimdDispatcher {
public:
    __aicore__ inline void Init(
        GM_ADDR gmMatC,
        const SpmmOpTilingData *tiling, TPipe *pipe)
    {
        pipe_ = pipe;
        gmMatC_ = gmMatC;
        tiling_ = tiling;
        dtype_ = tiling->dtype;
    }

    __aicore__ inline void Process()
    {
        if (dtype_ == SPMM_OP_DTYPE_FP32) {
            KernelSpmmOpBetaCSimd<float> op;
            op.Init(gmMatC_, tiling_, pipe_);
            op.Process();
        } else {
            KernelSpmmOpBetaCSimd<half> op;
            op.Init(gmMatC_, tiling_, pipe_);
            op.Process();
        }
    }

private:
    TPipe *pipe_{nullptr};
    const SpmmOpTilingData *tiling_{nullptr};
    GM_ADDR gmMatC_{nullptr};
    int32_t dtype_{SPMM_OP_DTYPE_FP32};
};

extern "C" __global__ __aicore__ void spmm_op_kernel(
    GM_ADDR gmRowOffsets, GM_ADDR gmColInd, GM_ADDR gmValues,
    GM_ADDR gmMatB, GM_ADDR gmMatC_out,
    GM_ADDR gmReorder, GM_ADDR gmBinEdge,
    const SpmmOpTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe;
    SpmmOpSimdDispatcher dispatcher;
    dispatcher.Init(gmRowOffsets, gmColInd, gmValues,
                    gmMatB, gmMatC_out,
                    gmReorder, gmBinEdge, &tiling, &pipe);
    dispatcher.Process();
}

// kernel_do 启动器（主计算）
extern "C" void spmm_op_kernel_do(
    GM_ADDR csrRowOffsets, GM_ADDR csrColInd, GM_ADDR csrValues,
    GM_ADDR matB, GM_ADDR matCOut,
    GM_ADDR reorder, GM_ADDR binEdge,
    const SpmmOpTilingData &tiling,
    uint32_t numBlocks, void *stream)
{
    spmm_op_kernel<<<numBlocks, nullptr, stream>>>(
        csrRowOffsets, csrColInd, csrValues, matB, matCOut,
        reorder, binEdge, tiling);
}

// __global__ 调度器（BetaC）
extern "C" __global__ __aicore__ void spmm_op_beta_c_kernel(
    GM_ADDR gmMatC_out,
    const SpmmOpTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe;
    SpmmOpBetaCSimdDispatcher dispatcher;
    dispatcher.Init(gmMatC_out, &tiling, &pipe);
    dispatcher.Process();
}

// kernel_do 启动器（BetaC：nnz==0 快捷路径）
extern "C" void spmm_op_beta_c_kernel_do(
    GM_ADDR matCOut,
    const SpmmOpTilingData &tiling,
    uint32_t numBlocks,
    void *stream)
{
    spmm_op_beta_c_kernel<<<numBlocks, nullptr, stream>>>(matCOut, tiling);
}
