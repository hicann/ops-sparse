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
#pragma once

#include "kernel_operator.h"
#include "../spmv_tiling_data.h"

constexpr uint32_t BUFFER_NUM = 1;

/**
 * @brief CSR 格式稀疏矩阵-稠密向量乘法（SpMV）AscendC 算子
 *
 * 采用行并行策略：每个 AI Core 处理若干整行，
 * 每行内通过 CopyIn → Compute → CopyOut 三级流水完成计算。
 *
 * @tparam T 数据类型（支持 float / int32_t）
 */
template <typename CompT, typename ValT = CompT, typename OutT = CompT>
class SpmvKernel {
public:
    __aicore__ inline SpmvKernel(){};

    __aicore__ inline void Init(
        GM_ADDR csrRowPtr, GM_ADDR csrColInd, GM_ADDR csrVal, GM_ADDR xVec, GM_ADDR yVec,
        uint32_t totalRowsNum, uint32_t totalColsNum,
        CompT alpha, CompT beta);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int32_t currentRow, int32_t validNum);
    __aicore__ inline void CopyOut(int32_t currentRow);
    __aicore__ inline void Compute(int32_t currentRow, int32_t validNum);

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueColIdx, inQueueVals;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueY;
    AscendC::TQue<AscendC::TPosition::VECCALC, BUFFER_NUM> workQueueX, workQueueReduce, floatQueueReduce;
    AscendC::GlobalTensor<int32_t> csrRowPtrGm;
    AscendC::GlobalTensor<int32_t> csrColIndGm;
    AscendC::GlobalTensor<ValT> csrValGm;
    AscendC::GlobalTensor<ValT> xVecGm;
    AscendC::GlobalTensor<OutT> yVecGm;

    uint32_t totalRowsNum;
    uint32_t totalColsNum;
    uint32_t startRow;
    uint32_t blockRowNum;
    uint32_t blockLength;
    uint32_t tileLength;
    uint32_t startValIdx;
    uint32_t endValIdx;
    CompT alpha;
    CompT beta;
};

template <typename CompT, typename ValT, typename OutT>
__aicore__ inline void SpmvKernel<CompT, ValT, OutT>::Init(
    GM_ADDR csrRowPtr, GM_ADDR csrColInd, GM_ADDR csrVal, GM_ADDR xVec, GM_ADDR yVec,
    uint32_t totalRowsNum, uint32_t totalColsNum,
    CompT alpha, CompT beta) {
    this->totalRowsNum = totalRowsNum;
    this->totalColsNum = totalColsNum;
    this->alpha = alpha;
    this->beta = beta;

    uint32_t blockNum = AscendC::GetBlockNum();
    if (blockNum == 0) {
        this->blockRowNum = 0;
        return;
    }

    // 均分行：前 tempRowNum 个核各多拿一行
    uint32_t tempRowNum = totalRowsNum % blockNum;
    if (AscendC::GetBlockIdx() < tempRowNum) {
        this->blockRowNum = this->totalRowsNum / AscendC::GetBlockNum() + 1;
        this->startRow = this->blockRowNum * AscendC::GetBlockIdx();
    }
    else {
        this->blockRowNum = this->totalRowsNum / AscendC::GetBlockNum();
        this->startRow = this->blockRowNum * AscendC::GetBlockIdx() + tempRowNum;
    }

    if (this->blockRowNum == 0) {
        return;
    }

    // 每核只切出自己负责的 rowPtr 段 + vals/colIdx 段
    csrRowPtrGm.SetGlobalBuffer((__gm__ int32_t *)csrRowPtr + this->startRow, this->blockRowNum + 1);

    this->startValIdx = csrRowPtrGm(0);
    this->endValIdx = csrRowPtrGm(this->blockRowNum);
    if (endValIdx >= startValIdx) {
        this->blockLength = endValIdx - startValIdx;
    } else {
        this->blockLength = 0;
    }

    csrColIndGm.SetGlobalBuffer((__gm__ int32_t *)csrColInd + startValIdx, this->blockLength);
    csrValGm.SetGlobalBuffer((__gm__ ValT *)csrVal + startValIdx, this->blockLength);
    xVecGm.SetGlobalBuffer((__gm__ ValT *)xVec, this->totalColsNum);
    yVecGm.SetGlobalBuffer((__gm__ OutT *)yVec + this->startRow, this->blockRowNum);

    // 计算当前核处理行中的最大行长（tile 大小依据）
    this->tileLength = 0;
    for (int i = 0; i < this->blockRowNum; i++) {
        uint32_t rowLength = csrRowPtrGm(i + 1) - csrRowPtrGm(i);
        if (rowLength > this->tileLength)
            this->tileLength = rowLength;
    }
    if (this->tileLength == 0) {
        this->tileLength = 8; // 全空行也给最小对齐量，避免 InitBuffer(0)
    }

    // 通过共享 UB 的精度进行计算优化
    pipe.InitBuffer(inQueueColIdx, BUFFER_NUM, this->tileLength * sizeof(int32_t));
    pipe.InitBuffer(inQueueVals, BUFFER_NUM, this->tileLength * sizeof(CompT));
    pipe.InitBuffer(workQueueX, BUFFER_NUM, this->tileLength * sizeof(CompT));
    pipe.InitBuffer(workQueueReduce, BUFFER_NUM, this->tileLength * sizeof(float));
    pipe.InitBuffer(floatQueueReduce, BUFFER_NUM, this->tileLength * sizeof(float));
    pipe.InitBuffer(outQueueY, BUFFER_NUM, sizeof(CompT));
}

template <typename CompT, typename ValT, typename OutT>
__aicore__ inline void SpmvKernel<CompT, ValT, OutT>::CopyIn(int32_t currentRow, int32_t validNum) {
    AscendC::LocalTensor<int32_t> colIdxLocal = inQueueColIdx.AllocTensor<int32_t>();
    AscendC::LocalTensor<CompT> valsLocal = inQueueVals.AllocTensor<CompT>();
    if (validNum > 0) {
        AscendC::DataCopyExtParams copyParamsColIdx{1, (uint32_t)(validNum * sizeof(int32_t)), 0, 0, 0};
        AscendC::DataCopyPadExtParams<int32_t> padParamsColIdx{false, 0, 0, 0};

        uint32_t offset = csrRowPtrGm(currentRow) - this->startValIdx;
        AscendC::DataCopyPad(colIdxLocal, csrColIndGm[offset], copyParamsColIdx, padParamsColIdx);

        if constexpr (std::is_same_v<ValT, CompT>) {
            AscendC::DataCopyExtParams cpV{1, (uint32_t)(validNum * sizeof(ValT)), 0, 0, 0};
            AscendC::DataCopyPadExtParams<ValT> padV{false, 0, 0, 0};
            AscendC::DataCopyPad(valsLocal, csrValGm[offset], cpV, padV);
        }
        else {
            // Cast 矢量指令（新 CANN 版本验证）
            AscendC::LocalTensor<ValT> valsTmp = workQueueX.AllocTensor<ValT>();
            AscendC::DataCopyExtParams cpV{1, (uint32_t)(validNum * sizeof(ValT)), 0, 0, 0};
            AscendC::DataCopyPadExtParams<ValT> padV{false, 0, 0, 0};
            AscendC::DataCopyPad(valsTmp, csrValGm[offset], cpV, padV);
            AscendC::Cast<CompT, ValT>(valsLocal, valsTmp,
                                       AscendC::RoundMode::CAST_NONE, validNum);
            AscendC::PipeBarrier<PIPE_V>();
            workQueueX.FreeTensor(valsTmp);
        }
    }
    inQueueColIdx.EnQue(colIdxLocal);
    inQueueVals.EnQue(valsLocal);

    // Read y from GM for beta*y computation in Compute
    AscendC::LocalTensor<CompT> yLocal = outQueueY.AllocTensor<CompT>();
    if constexpr (std::is_same_v<OutT, CompT>) {
        AscendC::DataCopyExtParams cp{1, (uint32_t)sizeof(CompT), 0, 0, 0};
        AscendC::DataCopyPadExtParams<CompT> padParams{false, 0, 0, 0};
        AscendC::DataCopyPad(yLocal, yVecGm[currentRow], cp, padParams);
    }
    else {
        AscendC::LocalTensor<OutT> yTmp = workQueueX.AllocTensor<OutT>();
        AscendC::DataCopyExtParams cp{1, (uint32_t)sizeof(OutT), 0, 0, 0};
        AscendC::DataCopyPadExtParams<OutT> padParams{false, 0, 0, 0};
        AscendC::DataCopyPad(yTmp, yVecGm[currentRow], cp, padParams);
        AscendC::Cast<CompT, OutT>(yLocal, yTmp,
                                   AscendC::RoundMode::CAST_NONE, 1);
        AscendC::PipeBarrier<PIPE_V>();
        workQueueX.FreeTensor(yTmp);
    }
    outQueueY.EnQue(yLocal);
}

template <typename CompT, typename ValT, typename OutT>
__aicore__ inline void SpmvKernel<CompT, ValT, OutT>::CopyOut(int32_t currentRow) {
    AscendC::LocalTensor<CompT> yLocal = outQueueY.DeQue<CompT>();
    if constexpr (std::is_same_v<CompT, OutT>) {
        AscendC::DataCopyExtParams copyParams{1, (uint32_t)sizeof(OutT), 0, 0, 0};
        AscendC::DataCopyPad(yVecGm[currentRow], yLocal, copyParams);
    }
    else {
        AscendC::LocalTensor<OutT> yOut = workQueueX.AllocTensor<OutT>();
        AscendC::Cast<OutT, CompT>(yOut, yLocal,
                                   AscendC::RoundMode::CAST_ROUND, 1);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::DataCopyExtParams copyParams{1, (uint32_t)sizeof(OutT), 0, 0, 0};
        AscendC::DataCopyPad(yVecGm[currentRow], yOut, copyParams);
        workQueueX.FreeTensor(yOut);
    }
    outQueueY.FreeTensor(yLocal);
}

template <typename CompT, typename ValT, typename OutT>
__aicore__ inline void SpmvKernel<CompT, ValT, OutT>::Compute(int32_t currentRow, int32_t validNum) {
    AscendC::LocalTensor<int32_t> colIdxLocal = inQueueColIdx.DeQue<int32_t>();
    AscendC::LocalTensor<CompT> valsLocal = inQueueVals.DeQue<CompT>();
    AscendC::LocalTensor<CompT> xLocal = workQueueX.AllocTensor<CompT>();
    AscendC::LocalTensor<float> sharedTmpBuffer;
    AscendC::LocalTensor<float> floatTmpBuffer;

    if (validNum > 0) {
        if constexpr (std::is_same_v<ValT, CompT>) {
            for (int i = 0; i < validNum; i++) {
                ValT v = xVecGm(colIdxLocal(i));
                xLocal.SetValue(i, v);
            }
        }
        else {
            // ValT≠CompT: Cast 转换，从 floatQueueReduce 分配 xTmp 避免 workQueueX 冲突
            AscendC::LocalTensor<ValT> xTmp = floatQueueReduce.AllocTensor<ValT>();
            for (int i = 0; i < validNum; i++) {
                ValT v = xVecGm(colIdxLocal(i));
                xTmp.SetValue(i, v);
            }
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Cast<CompT, ValT>(xLocal, xTmp,
                                       AscendC::RoundMode::CAST_NONE, validNum);
            floatQueueReduce.FreeTensor(xTmp);
        }

        // 逐元素乘：xLocal *= valsLocal
        AscendC::Mul(xLocal, xLocal, valsLocal, validNum);
        AscendC::PipeBarrier<PIPE_V>();

        sharedTmpBuffer = workQueueReduce.AllocTensor<float>();
        floatTmpBuffer = floatQueueReduce.AllocTensor<float>();

        // ReduceSum 在 float 做，其余计算以 CompT 进行
        if constexpr (std::is_same_v<CompT, float>) {
            // float 路径：xLocal 就是 float，不需要 Cast，直接 ReduceSum
            AscendC::ReduceSum<float>(floatTmpBuffer, xLocal,
                                      sharedTmpBuffer, validNum);
            AscendC::PipeBarrier<PIPE_V>();

            AscendC::LocalTensor<CompT> yLocal = outQueueY.DeQue<CompT>();

            AscendC::Muls(floatTmpBuffer, floatTmpBuffer, this->alpha, 1);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Muls(yLocal, yLocal, this->beta, 1);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Add(yLocal, yLocal, floatTmpBuffer, 1);
            AscendC::PipeBarrier<PIPE_V>();

            outQueueY.EnQue<CompT>(yLocal);
        }
        else {
            // 非 float 路径：Cast→float→ReduceSum→Cast 回 CompT
            AscendC::Cast<float, CompT>(floatTmpBuffer, xLocal,
                                        AscendC::RoundMode::CAST_ROUND, validNum);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::ReduceSum<float>(floatTmpBuffer, floatTmpBuffer,
                                      sharedTmpBuffer, validNum);
            AscendC::PipeBarrier<PIPE_V>();

            AscendC::LocalTensor<CompT> yLocal = outQueueY.DeQue<CompT>();

            // Cast sum from float back to CompT（复用 xLocal）
            AscendC::Cast<CompT, float>(xLocal, floatTmpBuffer,
                                        AscendC::RoundMode::CAST_ROUND, 1);
            AscendC::PipeBarrier<PIPE_V>();

            // z = alpha * sum + beta * y   (all CompT)
            AscendC::Muls(xLocal, xLocal, this->alpha, 1);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Muls(yLocal, yLocal, this->beta, 1);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Add(yLocal, yLocal, xLocal, 1);
            AscendC::PipeBarrier<PIPE_V>();

            outQueueY.EnQue<CompT>(yLocal);
        }

        workQueueReduce.FreeTensor(sharedTmpBuffer);
        floatQueueReduce.FreeTensor(floatTmpBuffer);
    }
    else {
        // 空行：z = beta * y  (CompT)
        AscendC::LocalTensor<CompT> yLocal = outQueueY.DeQue<CompT>();
        AscendC::Muls(yLocal, yLocal, this->beta, 1);
        AscendC::PipeBarrier<PIPE_V>();
        outQueueY.EnQue<CompT>(yLocal);
    }

    inQueueColIdx.FreeTensor(colIdxLocal);
    inQueueVals.FreeTensor(valsLocal);
    workQueueX.FreeTensor(xLocal);
}

template <typename CompT, typename ValT, typename OutT>
__aicore__ inline void SpmvKernel<CompT, ValT, OutT>::Process() {
    if (this->blockRowNum == 0)
        return;

    for (int32_t m = 0; m < this->blockRowNum; m++) {
        int32_t validNum = csrRowPtrGm(m + 1) - csrRowPtrGm(m);
        CopyIn(m, validNum);
        Compute(m, validNum);
        CopyOut(m);
    }
}

/**
 * @brief CSR 格式稀疏矩阵转置-稠密向量乘法（SpMV Transpose）AscendC 算子
 *
 * 采用与 SpmvKernel 一致的行并行策略：每个 AI Core 处理若干整行，
 * 但结果 y 不再是一行一个标量，而是一行中每个非零元对 y 的某个列产生贡献。
 * 多行对同一 y[j] 的累加通过 atomic add 避免核间踩踏。
 *
 * @tparam T 数据类型（支持 float / int32_t）
 */
template <typename CompT, typename ValT = CompT, typename OutT = CompT>
class SpmvKernelTrans {
public:
    __aicore__ inline SpmvKernelTrans(){};

    __aicore__ inline void Init(
        GM_ADDR csrRowPtr, GM_ADDR csrColInd, GM_ADDR csrVal, GM_ADDR xVec, GM_ADDR yVec,
        uint32_t totalRowsNum, uint32_t totalColsNum,
        CompT alpha, CompT beta);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ScaleBeta();
    __aicore__ inline void CopyIn(int32_t currentRow, int32_t validNum);
    __aicore__ inline void Compute(int32_t currentRow, int32_t validNum);
    __aicore__ inline void CopyOut(int32_t currentRow, int32_t validNum);

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueColIdx, inQueueVals;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueY;
    AscendC::TQue<AscendC::TPosition::VECCALC, BUFFER_NUM> workQueueX, workQueueReduce, floatQueueReduce;
    AscendC::GlobalTensor<int32_t> csrRowPtrGm;
    AscendC::GlobalTensor<int32_t> csrColIndGm;
    AscendC::GlobalTensor<ValT> csrValGm;
    AscendC::GlobalTensor<ValT> xVecGm;
    AscendC::GlobalTensor<OutT> yVecGm;

    uint32_t totalRowsNum;
    uint32_t totalColsNum;
    uint32_t startRow;
    uint32_t blockRowNum;
    uint32_t blockLength;
    uint32_t tileLength;
    uint32_t startValIdx;
    uint32_t endValIdx;
    CompT alpha;
    CompT beta;

    // beta 缩放阶段：当前核负责的 y 段（读）
    uint32_t yBlockStart;
    uint32_t yBlockCount;
};

template <typename CompT, typename ValT, typename OutT>
__aicore__ inline void SpmvKernelTrans<CompT, ValT, OutT>::Init(
    GM_ADDR csrRowPtr, GM_ADDR csrColInd, GM_ADDR csrVal, GM_ADDR xVec, GM_ADDR yVec,
    uint32_t totalRowsNum, uint32_t totalColsNum,
    CompT alpha, CompT beta) {
    this->totalRowsNum = totalRowsNum;
    this->totalColsNum = totalColsNum;
    this->alpha = alpha;
    this->beta = beta;

    uint32_t blockNum = AscendC::GetBlockNum();
    if (blockNum > 0) {
        // ===== 行分布（与 SpmvKernel 一致） =====
        uint32_t tempRowNum = totalRowsNum % blockNum;
        if (AscendC::GetBlockIdx() < tempRowNum) {
            this->blockRowNum = this->totalRowsNum / blockNum + 1;
            this->startRow = this->blockRowNum * AscendC::GetBlockIdx();
        }
        else {
            this->blockRowNum = this->totalRowsNum / blockNum;
            this->startRow = this->blockRowNum * AscendC::GetBlockIdx() + tempRowNum;
        }

        // ===== y 列分布（beta 缩放阶段：均分 totalColsNum 个 y 元素到各核） =====
        uint32_t tempColNum = totalColsNum % blockNum;
        if (AscendC::GetBlockIdx() < tempColNum) {
            this->yBlockCount = this->totalColsNum / blockNum + 1;
            this->yBlockStart = this->yBlockCount * AscendC::GetBlockIdx();
        }
        else {
            this->yBlockCount = this->totalColsNum / blockNum;
            this->yBlockStart = this->yBlockCount * AscendC::GetBlockIdx() + tempColNum;
        }
    }
    else {
        this->blockRowNum = 0;
        this->yBlockCount = 0;
    }

    // xVecGm/yVecGm 对所有核均需初始化（ScaleBeta 在所有核上运行）
    xVecGm.SetGlobalBuffer((__gm__ ValT *)xVec, this->totalRowsNum);
    yVecGm.SetGlobalBuffer((__gm__ OutT *)yVec, this->totalColsNum);

    if (this->blockRowNum > 0) {
        csrRowPtrGm.SetGlobalBuffer((__gm__ int32_t *)csrRowPtr + this->startRow, this->blockRowNum + 1);

        this->startValIdx = csrRowPtrGm(0);
        this->endValIdx = csrRowPtrGm(this->blockRowNum);
        if (endValIdx >= startValIdx) {
            this->blockLength = endValIdx - startValIdx;
        } else {
            this->blockLength = 0;
        }

        csrColIndGm.SetGlobalBuffer((__gm__ int32_t *)csrColInd + startValIdx, this->blockLength);
        csrValGm.SetGlobalBuffer((__gm__ ValT *)csrVal + startValIdx, this->blockLength);

        this->tileLength = 0;
        for (int i = 0; i < this->blockRowNum; i++) {
            uint32_t rowLength = csrRowPtrGm(i + 1) - csrRowPtrGm(i);
            if (rowLength > this->tileLength)
                this->tileLength = rowLength;
        }
        if (this->tileLength == 0) {
            this->tileLength = 8;
        }
    }
    else {
        this->tileLength = 8;
    }

    // UB 缓冲区初始化（所有核均需初始化 pipe）
    pipe.InitBuffer(inQueueColIdx, BUFFER_NUM, this->tileLength * sizeof(int32_t));
    pipe.InitBuffer(inQueueVals, BUFFER_NUM, this->tileLength * sizeof(CompT));
    pipe.InitBuffer(workQueueX, BUFFER_NUM, this->tileLength * sizeof(CompT));
    pipe.InitBuffer(workQueueReduce, BUFFER_NUM, this->tileLength * sizeof(float));
    pipe.InitBuffer(floatQueueReduce, BUFFER_NUM, this->tileLength * sizeof(float));
    pipe.InitBuffer(outQueueY, BUFFER_NUM, sizeof(CompT));
}

/**
 * @brief beta 缩放阶段：每个核将自己负责的 y 段乘以 beta
 *
 * 由于多核间 y 段不重叠，无需原子操作。
 * 当 beta == 0 时 host 端已做 y 全零初始化，本函数直接返回。
 * 当 beta == 1 时无需缩放，直接返回。
 */
template <typename CompT, typename ValT, typename OutT>
__aicore__ inline void SpmvKernelTrans<CompT, ValT, OutT>::ScaleBeta() {
    if (this->yBlockCount == 0)
        return;

    // beta == 1：无需缩放
    if (this->beta == static_cast<CompT>(1))
        return;

    // 以 tileLength 为块大小逐块处理 y[ yBlockStart .. yBlockStart+yBlockCount )
    for (uint32_t offset = 0; offset < this->yBlockCount; offset += this->tileLength) {
        uint32_t chunkSize = this->tileLength;
        if (offset + chunkSize > this->yBlockCount) {
            chunkSize = this->yBlockCount - offset;
        }

        if constexpr (std::is_same_v<CompT, OutT>) {
            AscendC::LocalTensor<CompT> yChunkLocal = workQueueX.AllocTensor<CompT>();

            AscendC::DataCopyExtParams cp{1, (uint32_t)(chunkSize * sizeof(CompT)), 0, 0, 0};
            AscendC::DataCopyPadExtParams<CompT> padParams{false, 0, 0, 0};
            AscendC::DataCopyPad(yChunkLocal,
                                 yVecGm[this->yBlockStart + offset],
                                 cp, padParams);
            AscendC::PipeBarrier<PIPE_V>();

            AscendC::Muls(yChunkLocal, yChunkLocal, this->beta, chunkSize);
            AscendC::PipeBarrier<PIPE_V>();

            AscendC::DataCopyExtParams cp2{1, (uint32_t)(chunkSize * sizeof(CompT)), 0, 0, 0};
            AscendC::DataCopyPad(yVecGm[this->yBlockStart + offset], yChunkLocal, cp2);
            AscendC::PipeBarrier<PIPE_V>();

            workQueueX.FreeTensor(yChunkLocal);
        }
        else {
            // OutT ≠ CompT (e.g. half output with float compute)
            // Read y as OutT → cast to CompT → *beta → cast back → write y
            AscendC::LocalTensor<OutT> yChunkOut = workQueueX.AllocTensor<OutT>();
            AscendC::LocalTensor<CompT> yChunkComp =
                floatQueueReduce.AllocTensor<CompT>();

            AscendC::DataCopyExtParams cp{1, (uint32_t)(chunkSize * sizeof(OutT)), 0, 0, 0};
            AscendC::DataCopyPadExtParams<OutT> padParams{false, 0, 0, 0};
            AscendC::DataCopyPad(yChunkOut,
                                 yVecGm[this->yBlockStart + offset],
                                 cp, padParams);
            AscendC::PipeBarrier<PIPE_V>();

            AscendC::Cast<CompT, OutT>(yChunkComp, yChunkOut,
                                       AscendC::RoundMode::CAST_ROUND, chunkSize);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Muls(yChunkComp, yChunkComp, this->beta, chunkSize);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Cast<OutT, CompT>(yChunkOut, yChunkComp,
                                       AscendC::RoundMode::CAST_ROUND, chunkSize);
            AscendC::PipeBarrier<PIPE_V>();

            AscendC::DataCopyExtParams cp2{1, (uint32_t)(chunkSize * sizeof(OutT)), 0, 0, 0};
            AscendC::DataCopyPad(yVecGm[this->yBlockStart + offset], yChunkOut, cp2);
            AscendC::PipeBarrier<PIPE_V>();

            floatQueueReduce.FreeTensor(yChunkComp);
            workQueueX.FreeTensor(yChunkOut);
        }
    }
}

template <typename CompT, typename ValT, typename OutT>
__aicore__ inline void SpmvKernelTrans<CompT, ValT, OutT>::CopyIn(int32_t currentRow, int32_t validNum) {
    AscendC::LocalTensor<int32_t> colIdxLocal = inQueueColIdx.AllocTensor<int32_t>();
    AscendC::LocalTensor<CompT> valsLocal = inQueueVals.AllocTensor<CompT>();
    if (validNum > 0) {
        AscendC::DataCopyExtParams copyParamsColIdx{1, (uint32_t)(validNum * sizeof(int32_t)), 0, 0, 0};
        AscendC::DataCopyPadExtParams<int32_t> padParamsColIdx{false, 0, 0, 0};

        uint32_t offset = csrRowPtrGm(currentRow) - this->startValIdx;
        AscendC::DataCopyPad(colIdxLocal, csrColIndGm[offset], copyParamsColIdx, padParamsColIdx);

        if constexpr (std::is_same_v<ValT, CompT>) {
            AscendC::DataCopyExtParams cpV{1, (uint32_t)(validNum * sizeof(ValT)), 0, 0, 0};
            AscendC::DataCopyPadExtParams<ValT> padV{false, 0, 0, 0};
            AscendC::DataCopyPad(valsLocal, csrValGm[offset], cpV, padV);
        }
        else {
            AscendC::LocalTensor<ValT> valsTmp = workQueueX.AllocTensor<ValT>();
            AscendC::DataCopyExtParams cpV{1, (uint32_t)(validNum * sizeof(ValT)), 0, 0, 0};
            AscendC::DataCopyPadExtParams<ValT> padV{false, 0, 0, 0};
            AscendC::DataCopyPad(valsTmp, csrValGm[offset], cpV, padV);
            AscendC::Cast<CompT, ValT>(valsLocal, valsTmp,
                                       AscendC::RoundMode::CAST_NONE, validNum);
            AscendC::PipeBarrier<PIPE_V>();
            workQueueX.FreeTensor(valsTmp);
        }
    }
    inQueueColIdx.EnQue(colIdxLocal);
    inQueueVals.EnQue(valsLocal);
}

template <typename CompT, typename ValT, typename OutT>
__aicore__ inline void SpmvKernelTrans<CompT, ValT, OutT>::Compute(int32_t currentRow, int32_t validNum) {
    AscendC::LocalTensor<CompT> valsLocal = inQueueVals.DeQue<CompT>();
    AscendC::LocalTensor<CompT> contribLocal = workQueueX.AllocTensor<CompT>();

    if (validNum > 0) {
        CompT xVal;
        if constexpr (std::is_same_v<ValT, CompT>) {
            xVal = xVecGm(this->startRow + currentRow);
        }
        else {
            AscendC::LocalTensor<ValT> xTmp = outQueueY.AllocTensor<ValT>();
            AscendC::LocalTensor<CompT> cTmp = floatQueueReduce.AllocTensor<CompT>();
            xTmp.SetValue(0, xVecGm(this->startRow + currentRow));
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Cast<CompT, ValT>(cTmp, xTmp,
                                       AscendC::RoundMode::CAST_NONE, 1);
            AscendC::PipeBarrier<PIPE_V>();
            xVal = cTmp.GetValue(0);
            outQueueY.FreeTensor(xTmp);
            floatQueueReduce.FreeTensor(cTmp);
        }

        AscendC::Muls(contribLocal, valsLocal, xVal, validNum);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Muls(contribLocal, contribLocal,
                      this->alpha, validNum);
        AscendC::PipeBarrier<PIPE_V>();
    }
    workQueueX.EnQue(contribLocal);
    inQueueVals.FreeTensor(valsLocal);
}

template <typename CompT, typename ValT, typename OutT>
__aicore__ inline void SpmvKernelTrans<CompT, ValT, OutT>::CopyOut(int32_t currentRow, int32_t validNum) {
    AscendC::LocalTensor<int32_t> colIdxLocal = inQueueColIdx.DeQue<int32_t>();
    AscendC::LocalTensor<CompT> contribLocal = workQueueX.DeQue<CompT>();

    if (validNum > 0) {
        if constexpr (std::is_same_v<CompT, OutT>) {
            AscendC::LocalTensor<CompT> atomicElem = outQueueY.AllocTensor<CompT>();
            for (int j = 0; j < validNum; j++) {
                atomicElem.SetValue(0, contribLocal(j));
                AscendC::DataCopyExtParams cp{1, (uint32_t)sizeof(CompT), 0, 0, 0};
                AscendC::SetAtomicAdd<CompT>();
                AscendC::DataCopyPad(yVecGm[colIdxLocal(j)], atomicElem, cp);
                AscendC::SetAtomicNone();
            }
            outQueueY.FreeTensor(atomicElem);
        }
        else {
            // CompT≠OutT: cast contrib→OutT，用 OutT 做原子累加直接写入 yVecGm
            AscendC::LocalTensor<OutT> oCast = floatQueueReduce.AllocTensor<OutT>();

            AscendC::Cast<OutT, CompT>(oCast, contribLocal,
                                       AscendC::RoundMode::CAST_ROUND, validNum);
            AscendC::PipeBarrier<PIPE_V>();

            AscendC::LocalTensor<OutT> atomicElem = outQueueY.AllocTensor<OutT>();
            for (int j = 0; j < validNum; j++) {
                atomicElem.SetValue(0, oCast(j));
                AscendC::DataCopyExtParams cp{1, (uint32_t)sizeof(OutT), 0, 0, 0};
                AscendC::SetAtomicAdd<OutT>();
                AscendC::DataCopyPad(yVecGm[colIdxLocal(j)], atomicElem, cp);
                AscendC::SetAtomicNone();
            }
            outQueueY.FreeTensor(atomicElem);
            floatQueueReduce.FreeTensor(oCast);
        }
    }

    inQueueColIdx.FreeTensor(colIdxLocal);
    workQueueX.FreeTensor(contribLocal);
}

template <typename CompT, typename ValT, typename OutT>
__aicore__ inline void SpmvKernelTrans<CompT, ValT, OutT>::Process() {
    // 阶段 1：beta 缩放（多核分工无重叠，无需原子）
    ScaleBeta();
    AscendC::SyncAll();

    // 阶段 2：逐行处理，每行贡献原子累加到 y
    for (int32_t m = 0; m < this->blockRowNum; m++) {
        int32_t validNum = csrRowPtrGm(m + 1) - csrRowPtrGm(m);
        CopyIn(m, validNum);
        Compute(m, validNum);
        CopyOut(m, validNum);
    }
    AscendC::SyncAll();
}

// -------------------------------------------------------------------
// 具体 Kernel 入口 + Host 端启动封装
// -------------------------------------------------------------------

// ---- per-type __global__ kernel 定义（各自 TU 展开一个实例） ----
#define DEFINE_SPMV_KERNEL(CompT, ValT, OutT)                                                           \
    __global__ __aicore__ void spmv_kernel_##CompT##_##ValT##_##OutT(                                   \
        GM_ADDR csrRowPtr, GM_ADDR csrColInd, GM_ADDR csrVal,                                           \
        GM_ADDR xVec, GM_ADDR yVec,                                                                      \
        uint32_t totalRowsNum, uint32_t totalColNum,                                                     \
        CompT alpha, CompT beta, bool trans)                                                            \
    {                                                                                                   \
        KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);                                                 \
        if (trans)                                                                                      \
        {                                                                                               \
            SpmvKernelTrans<CompT, ValT, OutT> op;                                                      \
            op.Init(csrRowPtr, csrColInd, csrVal, xVec, yVec, totalRowsNum, totalColNum, alpha, beta);  \
            op.Process();                                                                               \
        }                                                                                               \
        else                                                                                            \
        {                                                                                               \
            SpmvKernel<CompT, ValT, OutT> op;                                                           \
            op.Init(csrRowPtr, csrColInd, csrVal, xVec, yVec, totalRowsNum, totalColNum, alpha, beta);  \
            op.Process();                                                                               \
        }                                                                                               \
    }

// ---- extern 声明（供 spmv_kernel_do 引用各 TU 中的 __global__ kernel） ----
#define SPMV_EXTERN(CompT, ValT, OutT) \
    extern __global__ __aicore__ void spmv_kernel_##CompT##_##ValT##_##OutT( \
        GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, uint32_t, uint32_t, CompT, CompT, bool);
SPMV_EXTERN(float, float, float)
SPMV_EXTERN(float, half, half)
SPMV_EXTERN(float, half, float)
SPMV_EXTERN(float, bfloat16_t, bfloat16_t)
SPMV_EXTERN(float, bfloat16_t, float)
SPMV_EXTERN(float, int32_t, float)
SPMV_EXTERN(int32_t, int32_t, int32_t)
#undef SPMV_EXTERN
