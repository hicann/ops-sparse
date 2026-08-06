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

/*
 * scatter Kernel — Optimized v1 (Template D: no in-kernel sort + DoubleBuffer)
 */

#include "kernel_operator.h"
#include "scatter.h"
#include "scatter_kernel.h"

using namespace AscendC;

#define BUFFER_NUM 2
constexpr uint32_t COPY_MASK_MAX = 64;

class KernelScatter {
public:
    __aicore__ inline KernelScatter()
    {}

    __aicore__ inline void Init(GM_ADDR valGmAddr, GM_ADDR idxGmAddr,
                                GM_ADDR yGmAddr, const ScatterTilingData &tiling)
    {
        uint32_t blockIdx = static_cast<uint32_t>(GetBlockIdx());
        if (blockIdx >= tiling.blockNum) {
            return;
        }
        coreNnz_   = tiling.coreNnzCount[blockIdx];
        coreOffset_ = tiling.coreNnzOffset[blockIdx];

        valGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(valGmAddr), tiling.nnz);
        idxGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(idxGmAddr), tiling.nnz);
        yGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(yGmAddr), tiling.yLen / sizeof(float));

        if (coreNnz_ == 0) {
            return;
        }

        uint32_t tileNn    = tiling.tileNn;
        uint32_t aligned8  = tiling.tileNnAligned8;
        uint32_t aligned4  = tiling.tileNnAligned4;

        pipe.InitBuffer(inQueueVal, BUFFER_NUM, aligned8 * sizeof(float));
        pipe.InitBuffer(inQueueIdx, BUFFER_NUM, aligned4 * sizeof(int32_t));
        pipe.InitBuffer(outQueue, 1, aligned8 * sizeof(float));

        tileNn_ = tileNn;
    }

    __aicore__ inline void Process()
    {
        if (coreNnz_ == 0) {
            return;
        }

        uint32_t tileNum = (coreNnz_ + tileNn_ - 1) / tileNn_;

        {
            uint32_t curTileNn = (coreNnz_ < tileNn_) ? coreNnz_ : tileNn_;
            uint32_t aligned4  = ((curTileNn + 3) / 4) * 4;
            CopyInAsync(coreOffset_, curTileNn, aligned4);
        }

        for (uint32_t tile = 0; tile < tileNum; tile++) {
            uint32_t localOffset = tile * tileNn_;
            if (localOffset >= coreNnz_) {
                break;
            }
            uint32_t curTileNn = coreNnz_ - localOffset;
            if (curTileNn > tileNn_) {
                curTileNn = tileNn_;
            }

            WaitCopyIn();

            if (tile + 1 < tileNum) {
                uint32_t nextOffset = coreOffset_ + (tile + 1) * tileNn_;
                uint32_t nextNn     = coreNnz_ - (tile + 1) * tileNn_;
                if (nextNn > tileNn_) {
                    nextNn = tileNn_;
                }
                uint32_t nextAligned4 = ((nextNn + 3) / 4) * 4;
                CopyInAsync(nextOffset, nextNn, nextAligned4);
            }

            Compute(curTileNn);
        }
    }

private:
    __aicore__ inline void CopyInAsync(uint32_t gmOffset, uint32_t curTileNn, uint32_t curTileNnAligned)
    {
        auto valLocal = inQueueVal.AllocTensor<float>();
        DataCopyPad(valLocal, valGm[gmOffset],
            {1, static_cast<uint16_t>(curTileNn * sizeof(float)), 0, 0},
            {false, 0, 0, 0});
        inQueueVal.EnQue(valLocal);

        auto idxLocal = inQueueIdx.AllocTensor<int32_t>();
        DataCopyPad(idxLocal, idxGm[gmOffset],
            {1, static_cast<uint16_t>(curTileNn * sizeof(int32_t)), 0, 0},
            {false, 0, 0, 0});
        inQueueIdx.EnQue(idxLocal);
    }

    __aicore__ inline void WaitCopyIn()
    {
        valBuf_   = inQueueVal.DeQue<float>();
        idxBuf32_ = inQueueIdx.DeQue<int32_t>();
    }

    __aicore__ inline void Compute(uint32_t curTileNn)
    {
        uint32_t runStart = 0;
        int32_t prevIdx = idxBuf32_.GetValue(0);

        for (uint32_t i = 1; i <= curTileNn; i++) {
            bool isEnd = (i == curTileNn);
            bool isDisconnected = false;
            int32_t curIdx = 0;

            if (!isEnd) {
                curIdx = idxBuf32_.GetValue(i);
                isDisconnected = (curIdx != prevIdx + 1);
            }

            if (isEnd || isDisconnected) {
                uint32_t runLen = i - runStart;
                int32_t targetIdx = idxBuf32_.GetValue(runStart);

                WriteRun(targetIdx, runStart, runLen);

                runStart = i;
                if (!isEnd) {
                    prevIdx = curIdx;
                }
            } else {
                prevIdx = curIdx;
            }
        }

        inQueueVal.FreeTensor(valBuf_);
        inQueueIdx.FreeTensor(idxBuf32_);
    }

    __aicore__ inline void WriteRun(int32_t targetIdx, uint32_t srcOffset, uint32_t runLen)
    {
        auto outLocal = outQueue.AllocTensor<float>();

        if (runLen == 1) {
            outLocal.SetValue(0, valBuf_.GetValue(srcOffset));
            DataCopyPad(yGm[targetIdx], outLocal,
                {1, static_cast<uint32_t>(sizeof(float)), 0, 0, 0});
        } else {
            if ((srcOffset & 7) == 0) {
                uint32_t remaining = runLen;
                uint32_t srcOff = srcOffset;
                uint32_t dstOff = 0;
                while (remaining > 0) {
                    uint32_t chunkSize = (remaining > COPY_MASK_MAX) ? COPY_MASK_MAX : remaining;
                    Copy<float>(outLocal[dstOff], valBuf_[srcOff],
                        static_cast<uint64_t>(chunkSize), 1, {1, 1, 8, 8});
                    remaining -= chunkSize;
                    srcOff += chunkSize;
                    dstOff += chunkSize;
                }
                PipeBarrier<PIPE_V>();
            } else {
                for (uint32_t j = 0; j < runLen; j++) {
                    outLocal.SetValue(j, valBuf_.GetValue(srcOffset + j));
                }
            }

            DataCopyPad(yGm[targetIdx], outLocal,
                {1, static_cast<uint32_t>(runLen * sizeof(float)), 0, 0, 0});
        }

        PipeBarrier<PIPE_MTE3>();
        outQueue.FreeTensor(outLocal);
    }

private:
    TPipe pipe;
    TQue<TPosition::VECIN, BUFFER_NUM> inQueueVal;
    TQue<TPosition::VECIN, BUFFER_NUM> inQueueIdx;
    TQue<TPosition::VECOUT, 1> outQueue;
    GlobalTensor<float> valGm;
    GlobalTensor<int32_t> idxGm;
    GlobalTensor<float> yGm;
    LocalTensor<float> valBuf_;
    LocalTensor<int32_t> idxBuf32_;
    uint32_t coreNnz_;
    uint32_t coreOffset_;
    uint32_t tileNn_;
};

__global__ __vector__ void scatter_custom(
    GM_ADDR valGm, GM_ADDR idxGm, GM_ADDR yGm, const ScatterTilingData tiling)
{
    KernelScatter op;
    op.Init(valGm, idxGm, yGm, tiling);
    op.Process();
}

void scatter_kernel_do(
    void *valDev, void *idxDev, void *yDev,
    const ScatterTilingData &tiling, uint32_t blockNum, void *stream)
{
    scatter_custom<<<blockNum, nullptr, stream>>>(
        (GM_ADDR)valDev, (GM_ADDR)idxDev, (GM_ADDR)yDev, tiling);
}
