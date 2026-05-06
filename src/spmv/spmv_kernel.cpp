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

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "spmv.h"
using namespace AscendC;

#define BUFFER_NUM 2
constexpr uint64_t SYNC_AIV_TO_AIC = 3;
constexpr uint64_t SYNC_AIC_TO_AIV = 5;

#define CUBE_BLOCK_SIZE (MAX_SUB_ROW_SIZE * MAX_SUB_ROW_SIZE)

class KernelSpmv {
public:
    __aicore__ inline KernelSpmv()
    {}
    __aicore__ inline void Init(
        GM_ADDR sync, GM_ADDR buffer, GM_ADDR x, GM_ADDR y, uint64_t rows, uint64_t cols, uint64_t nnz)
    {
        info = reinterpret_cast<__gm__ SpmvCsrInfo *>(buffer);
        blockNum = GetBlockNum();
        nParts = info->num;
        id = GetBlockIdx();
        this->rows = rows;
        this->cols = cols;
        this->nnz = nnz;

        yGm.SetGlobalBuffer(reinterpret_cast<__gm__ float32_t *>(y), ROUNDUP(rows, 32));
        xSrcGm.SetGlobalBuffer(reinterpret_cast<__gm__ float32_t *>(x), ROUNDUP(cols, 32));
        syncGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(sync), GM_SYNC_SIZE / sizeof(int32_t));
        GM_ADDR tmpSwap = (info->swap) + id * info->maxBlockSize * sizeof(float) * 4;
        swapAivToAicGm.SetGlobalBuffer(reinterpret_cast<__gm__ float32_t *>(tmpSwap), info->maxBlockSize);
        tmpSwap = tmpSwap + info->maxBlockSize * sizeof(float) * 2;
        swapAicToAivGm.SetGlobalBuffer(reinterpret_cast<__gm__ float32_t *>(tmpSwap), info->maxBlockSize);
        elemNumPerBlock = info->maxBlockSize / sizeof(float32_t);
        pipe.InitBuffer(this->queueXSrcIn, 1, MAX_SUB_COL_SIZE * sizeof(float32_t));
        pipe.InitBuffer(this->queueXIdxIn, BUFFER_NUM, info->maxBlockSize * sizeof(uint32_t));
        pipe.InitBuffer(this->queueXOut, BUFFER_NUM, info->maxBlockSize * sizeof(float32_t));
        pipe.InitBuffer(this->queueXB1, BUFFER_NUM, info->maxBlockSize * sizeof(float32_t));
        pipe.InitBuffer(this->queueXB2, BUFFER_NUM, info->maxBlockSize * sizeof(float32_t));
        pipe.InitBuffer(this->queueMatA1, BUFFER_NUM, info->maxBlockSize * sizeof(float32_t));
        pipe.InitBuffer(this->queueMatA2, BUFFER_NUM, ROUNDUP(info->maxBlockSize * sizeof(float32_t), 512));
        pipe.InitBuffer(this->queueYCO1, BUFFER_NUM, MAX_SUB_ROW_SIZE * MAX_SUB_ROW_SIZE * sizeof(float32_t));
        pipe.InitBuffer(this->queueYOut, BUFFER_NUM, MAX_SUB_ROW_SIZE * sizeof(float32_t));
        pipe.InitBuffer(this->queueYOutIdx, BUFFER_NUM, MAX_SUB_ROW_SIZE * sizeof(float32_t));
        pipe.InitBuffer(this->queueYIn, BUFFER_NUM, MAX_SUB_ROW_SIZE * MAX_SUB_ROW_SIZE * sizeof(float32_t));

        uint32_t len = info->maxBlockSize > GM_SYNC_SIZE ? info->maxBlockSize : GM_SYNC_SIZE;
        pipe.InitBuffer(this->syncBuf, len);
        syncGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(sync), GM_SYNC_SIZE / sizeof(int32_t));
        auto lc = syncBuf.Get<int32_t>();
        Duplicate<int32_t>(lc, 0, GM_SYNC_SIZE / sizeof(int32_t));
        DataCopy(syncGm, lc, GM_SYNC_SIZE / sizeof(int32_t));
    }

    __aicore__ inline void SetYZero()
    {
        auto lc = syncBuf.Get<float32_t>();
        Duplicate<float32_t>(lc, 0.0, elemNumPerBlock);
        uint64_t upRows = ROUNDUP(rows, 32);
        for (int i = id * elemNumPerBlock; i < upRows; i += blockNum * elemNumPerBlock) {
            uint64_t len = i + elemNumPerBlock <= upRows ? elemNumPerBlock : upRows - i;
            DataCopy(yGm[i], lc, len);
        }
    }

    __aicore__ inline void DoSync()
    {
        LocalTensor lcTensor = syncBuf.Get<int32_t>();
        SyncAll(syncGm, lcTensor, GetBlockNum());
    }
    
    __aicore__ inline void DoSpmv()
    {
        if ASCEND_IS_AIV {
            yOutIdxLm = queueYOutIdx.AllocTensor<uint32_t>();
            uint32_t tmp = 0;
            for (uint32_t i = 0; i < MAX_SUB_ROW_SIZE; i++) {
                yOutIdxLm.SetValue(i, tmp);
                tmp += (MAX_SUB_ROW_SIZE + 1) * sizeof(float32_t);
            }
        }
        for (uint64_t i = 0; i < nParts; i++) {
            __gm__ SpmvCsrSubMatInfo *infos = &info->infos[i];
            matGm.SetGlobalBuffer(infos->values, infos->valueLen / sizeof(float32_t));
            GlobalTensor<uint32_t> tXIdxGm;
            tXIdxGm.SetGlobalBuffer(infos->colIdx, infos->colIdxLen / sizeof(float32_t));
            LocalTensor<float32_t> xin;
            if ASCEND_IS_AIV {
                xin = queueXSrcIn.AllocTensor<float32_t>();
                uint64_t colNum = ROUNDUP(infos->colNum, 32);
                DataCopy(xin, xSrcGm[infos->startCol], colNum);
            }
            for (uint64_t j = id; j < infos->blockNum; j += blockNum) {
                uint32_t blockId = (j << 1);
                uint32_t start = infos->blockPtr[blockId];
                uint32_t end = infos->blockPtr[blockId + 2];
                uint32_t rawId = infos->blockPtr[blockId + 1];
                uint32_t len = end - start;
                VectorComputor(j, infos, start, len, tXIdxGm, xin);
                CubeComputor(j, infos, start, len, tXIdxGm);
                VectorPostComputor(j, infos, rawId, len, tXIdxGm);
            }
            if ASCEND_IS_AIV {
                queueXSrcIn.FreeTensor(xin);
            }
        }
        if ASCEND_IS_AIV {
            queueYOutIdx.FreeTensor(yOutIdxLm);
        }
    }
    __aicore__ inline void Process()
    {
        SetYZero();
        DoSync();
        DoSpmv();
    }

private:
    __aicore__ inline void VectorComputor(
        int32_t j, __gm__ SpmvCsrSubMatInfo *infos, uint32_t start, uint32_t len, GlobalTensor<uint32_t> &xId, LocalTensor<float32_t> &xSrc)
    {
        if ASCEND_IS_AIC {
            return;
        }
        XIdxCopyIn(j, infos, start, len, xId);
        XIdxGather(j, infos, xSrc, len);
        XIdxCopyOut(j, infos, len);
        AscendC::CrossCoreSetFlag<2, PIPE_MTE2>(SYNC_AIV_TO_AIC);  // 2: mode为2, group内同步
    }

    __aicore__ inline void CubeComputor(
        int32_t j, __gm__ SpmvCsrSubMatInfo *infos, uint32_t start, uint32_t len, GlobalTensor<uint32_t> &xId)
    {
        if ASCEND_IS_AIV {
            return;
        }
        AscendC::CrossCoreWaitFlag(SYNC_AIV_TO_AIC);
        CopyInB1(j, infos, start, len);
        CopyInB2(j, infos, start, len);
        CopyInA1(j, infos, start, len);
        CopyInA2(j, infos, start, len);
        Compute(j, infos, start, len);
        CopyOut(j, 0);
        AscendC::CrossCoreSetFlag<2, PIPE_FIX>(SYNC_AIC_TO_AIV);  // 2: mode为2, group内同步
    }

    __aicore__ inline void VectorPostComputor(
        int32_t j, __gm__ SpmvCsrSubMatInfo *infos, uint32_t start, uint32_t len, GlobalTensor<uint32_t> &xId)
    {
        if ASCEND_IS_AIC {
            return;
        }
        AscendC::CrossCoreWaitFlag(SYNC_AIC_TO_AIV);
        auto yin = queueYIn.AllocTensor<float32_t>();
        auto yout = queueYOut.AllocTensor<float32_t>();
        DataCopy(yin, swapAicToAivGm, CUBE_BLOCK_SIZE);

        Gather(yout, yin, yOutIdxLm, 0, MAX_SUB_ROW_SIZE);
        
        AscendC::SetAtomicAdd<float>();
        DataCopy(yGm[start], yout, MAX_SUB_ROW_SIZE);

        AscendC::SetAtomicNone();
        queueYOut.FreeTensor(yout);
        queueYIn.FreeTensor(yin);
    }

    __aicore__ inline void XIdxCopyIn(
        int32_t j, __gm__ SpmvCsrSubMatInfo *infos, uint32_t start, uint32_t len, GlobalTensor<uint32_t> &xId)
    {
        auto xidx = queueXIdxIn.AllocTensor<uint32_t>();
        DataCopy(xidx, xId[start], len);
        queueXIdxIn.EnQue(xidx);
    }

    __aicore__ inline void XIdxGather(
        int32_t j, __gm__ SpmvCsrSubMatInfo *infos, LocalTensor<float32_t> &xin, uint32_t len)
    {
        auto xidx = queueXIdxIn.DeQue<uint32_t>();
        for (uint32_t i = 0; i < len; i++) {
            xidx.SetValue(i, xidx.GetValue(i) * sizeof(float32_t));
        }
        auto xcalc = queueXOut.AllocTensor<float32_t>();
        Gather(xcalc, xin, xidx, 0, len);
        queueXOut.EnQue(xcalc);
        queueXIdxIn.FreeTensor(xidx);
    }

    __aicore__ inline void XIdxCopyOut(int32_t j, __gm__ SpmvCsrSubMatInfo *infos, uint32_t len)
    {
        auto x = queueXOut.DeQue<float32_t>();
        DataCopy(swapAivToAicGm, x, len);
        queueXOut.FreeTensor(x);
    }

    __aicore__ inline void CopyInA1(int32_t j, __gm__ SpmvCsrSubMatInfo *infos, uint32_t start, uint32_t len)
    {
        auto x = queueMatA1.AllocTensor<float32_t>();
        DataCopy(x, matGm[start], len);
        queueMatA1.EnQue(x);
    }

    __aicore__ inline void CopyInB1(int32_t j, __gm__ SpmvCsrSubMatInfo *infos, uint32_t start, uint32_t len)
    {
        auto x = queueXB1.AllocTensor<float32_t>();
        DataCopy(x, swapAivToAicGm, len);
        queueXB1.EnQue(x);
    }

    __aicore__ inline void CopyInA2(int32_t j, __gm__ SpmvCsrSubMatInfo *infos, uint32_t start, uint32_t len)
    {
        auto a2 = queueMatA2.AllocTensor<float32_t>();
        auto x = queueMatA1.DeQue<float32_t>();
        AscendC::LoadData2DParams loadDataParams;
        loadDataParams.repeatTimes = len / (512 / sizeof(float32_t));  // 512: 分型大小
        loadDataParams.srcStride = 1;
        loadDataParams.dstGap = 0;
        loadDataParams.ifTranspose = false;
        AscendC::LoadData(a2, x, loadDataParams);
        queueMatA2.EnQue(a2);
        queueMatA1.FreeTensor(x);
    }

    __aicore__ inline void CopyInB2(int32_t j, __gm__ SpmvCsrSubMatInfo *infos, uint32_t start, uint32_t len)
    {
        auto b2 = queueXB2.AllocTensor<float32_t>();
        auto x = queueXB1.DeQue<float32_t>();
        AscendC::LoadData2DParams loadDataParams;
        loadDataParams.repeatTimes = len / (512 / sizeof(float32_t));  // 512: 分型大小
        loadDataParams.srcStride = 1;
        loadDataParams.dstGap = 0;
        loadDataParams.ifTranspose = false;
        AscendC::LoadData(b2, x, loadDataParams);
        queueXB2.EnQue(b2);
        queueXB1.FreeTensor(x);
    }

    __aicore__ inline void Compute(int32_t j, __gm__ SpmvCsrSubMatInfo *infos, uint32_t start, uint32_t len)
    {
        auto mat = queueMatA2.DeQue<float32_t>();
        auto x = queueXB2.DeQue<float32_t>();
        auto c1 = queueYCO1.AllocTensor<float32_t>();

        AscendC::MmadParams mmadParams;
        mmadParams.m = MAX_SUB_ROW_SIZE;
        mmadParams.n = MAX_SUB_ROW_SIZE;
        mmadParams.k = len / MAX_SUB_ROW_SIZE;
        AscendC::Mmad(c1, mat, x, mmadParams);

        queueYCO1.EnQue(c1);
        queueMatA2.FreeTensor(mat);
        queueXB2.FreeTensor(x);
    }

    __aicore__ inline void CopyOut(int32_t progress, int32_t num)
    {
        auto c1 = queueYCO1.DeQue<float32_t>();
        AscendC::FixpipeParamsV220 fixpipeParams;
        fixpipeParams.nSize = 16;
        fixpipeParams.mSize = 16;
        fixpipeParams.srcStride = 16;
        fixpipeParams.dstStride = 16;

        fixpipeParams.ndNum = 1;
        fixpipeParams.srcNdStride = 0;
        fixpipeParams.dstNdStride = 0;
        AscendC::Fixpipe(swapAicToAivGm, c1, fixpipeParams);
        queueYCO1.FreeTensor(c1);
    }

private:
    TPipe pipe;
    TBuf<TPosition::VECCALC> syncBuf;
    uint32_t blockNum;
    uint32_t id;
    uint64_t nParts;
    uint64_t rows;
    uint64_t cols;
    uint64_t nnz;
    uint64_t elemNumPerBlock;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> queueXSrcIn;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> queueXIdxIn;
    TQue<TPosition::VECOUT, BUFFER_NUM> queueXOut;
    TQue<TPosition::B1, BUFFER_NUM> queueXB1;
    TQue<TPosition::B2, BUFFER_NUM> queueXB2;
    TQue<TPosition::CO1, BUFFER_NUM> queueYCO1;
    TQueBind<TPosition::CO2, TPosition::VECIN, BUFFER_NUM> queueYIn;
    AscendC::TQue<AscendC::TPosition::VECCALC, BUFFER_NUM> queueYOutIdx;
    TQueBind<TPosition::VECIN, TPosition::VECOUT, BUFFER_NUM> queueYOut;
    TQue<TPosition::A1, BUFFER_NUM> queueMatA1;
    TQue<TPosition::A2, BUFFER_NUM> queueMatA2;
    AscendC::GlobalTensor<float> xSrcGm;
    AscendC::GlobalTensor<uint32_t> xIdxGm;
    AscendC::GlobalTensor<float> yGm;
    AscendC::GlobalTensor<float> matGm;
    AscendC::GlobalTensor<int32_t> syncGm;
    AscendC::GlobalTensor<float32_t> swapAivToAicGm;
    AscendC::GlobalTensor<float32_t> swapAicToAivGm;
    AscendC::LocalTensor<uint32_t> yOutIdxLm;
    __gm__ SpmvCsrInfo *info;
};

__global__ __aicore__ void spmv_custom(
    GM_ADDR buffer, GM_ADDR x, GM_ADDR y, uint64_t rows, uint64_t cols, uint32_t nnz)
{   
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_1);
    KernelSpmv op;
    op.Init(buffer, buffer + GM_SYNC_SIZE, x, y, rows, cols, nnz);
    op.Process();
}

void spmv_kernel_do(void* buffer, void* x, void* y, uint64_t rows, uint64_t cols, uint32_t nnz, uint32_t numBlocks, void *stream)
{
    spmv_custom<<<numBlocks, nullptr, stream>>>((GM_ADDR)buffer, (GM_ADDR)x, (GM_ADDR)y, rows, cols, nnz);
}