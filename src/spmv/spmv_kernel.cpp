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
#define MAX_TILE_SIZE (64)

class KernelSpmv {
public:
    __aicore__ inline KernelSpmv()
    {}
    __aicore__ inline void Init(
        GM_ADDR sync, GM_ADDR buffer, GM_ADDR x, GM_ADDR y, uint64_t rows, uint64_t cols, uint64_t nnz)
    {
        info = reinterpret_cast<__gm__ SpmvCsrInfo *>(buffer);
        blockNum = GetBlockNum();
        id = GetBlockIdx();
        this->x = (__gm__ float *)x;
        this->y = (__gm__ float *)y;
        this->rows = rows;
        this->cols = cols;
        this->nnz = nnz;
    }
    __aicore__ inline void Process()
    {
        for (uint64_t i = MAX_TILE_SIZE * id; i < rows; i += MAX_TILE_SIZE * blockNum) {
            uint64_t end = ((i + MAX_TILE_SIZE) > rows) ? rows : (i + MAX_TILE_SIZE);
            GlobalTensor<float> gm;
            gm.SetGlobalBuffer(y + i, (end - i + 15) / 15 * 15);
            for (uint64_t j = i; j < end; j++) {
                uint32_t rstart = info->ptrs[j];
                uint32_t rend = info->ptrs[j + 1];
                float res = 0.0;
                for (uint32_t l = rstart; l < rend; l++) {
                    res += info->values[l] * x[info->idxs[l]];
                }

                y[j] = res;
            }
            DataCacheCleanAndInvalid<float, AscendC::CacheLine::ENTIRE_DATA_CACHE, AscendC::DcciDst::CACHELINE_OUT>(gm);
        }
    }

private:
    uint32_t blockNum;
    uint32_t id;
    uint64_t rows;
    uint64_t cols;
    uint64_t nnz;
    __gm__ float *x;
    __gm__ float *y;
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