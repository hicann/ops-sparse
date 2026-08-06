/**
 * ----------------------------------------------------------------------------------------------------------
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under
 * the terms and conditions of CANN Open Software License Agreement Version 2.0
 * (the "License"). Please refer to the License for details. You may not use
 * this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
 * AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
 * FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
 * for the full text of the License.
 * ----------------------------------------------------------------------------------------------------------
 */

/*!
 * \file sparse2dense_kernel.cpp
 * \brief SparseToDense SIMT kernel 实现（仅 arch35/DAV-3510 可用）。
 *
 * 三层结构：
 *   Layer 1: __simt_vf__ 计算函数（线程级并行，grid-stride loop）
 *   Layer 2: __global__ 调度器（读 tiling → asc_vf_call 分发）
 *   Layer 3: kernel_do 启动器（<<<>>> 异步 launch）
 *
 * 支持三种稀疏格式：
 *   CSR: ptrs=rowOffsets(m+1), idxs=colInd(nnz) — 按行切分，行并行
 *   CSC: ptrs=colOffsets(n+1), idxs=rowInd(nnz) — 按列切分，列并行
 *   COO: ptrs=cooRowInd(nnz),  idxs=cooColInd(nnz) — 按 nnz grid-stride
 *
 * CSR/CSC 安全性：不同行/列由不同线程处理，无跨线程写冲突。
 * COO 安全性：重复 (row,col) 走 last-write-wins（与 cuSPARSE 一致）。
 *
 * 模板实例化矩阵：
 *   [f32] [f16] [bf16] [i32] [i8]
 * 由 dispatcher 按 tiling.valueType 选择。
 */

#include <cstdint>
#include "kernel_operator.h"
#include "simt_api/asc_simt.h"
#include "sparse2dense_kernel.h"

// ===========================================================================
// Dispatcher 基类：提取公共逻辑（block→范围映射、线程数计算）
// ===========================================================================

struct Sparse2DenseDispatcherBase {
    int32_t  m_{0};
    int32_t  n_{0};
    int32_t  indexBase_{0};
    int32_t  valueType_{SPARSE2DENSE_VAL_F32};
    int32_t  isColMajor_{0};
    int32_t  ld_{0};
    int32_t  format_{SPARSE2DENSE_FMT_CSR};
    uint32_t perBlock_{0};
    uint64_t nnz_{0};

    __aicore__ inline static uint32_t AlignToWarp(uint32_t simtThreadNum)
    {
        return (simtThreadNum + kSparse2DenseWarpSize - 1u) & ~(kSparse2DenseWarpSize - 1u);
    }

    // CSR/CSC: 根据 GetBlockIdx() 计算本 core 行/列范围，并对齐 SIMT 线程数到 warp 整数倍
    // 使用 int64_t 中间变量避免 blockId * perBlock_ 在 INT32_MAX 附近溢出
    // 尾核可能 rangeStart >= totalDim，此时 coreRangeLen <= 0，simtThreadNum 置 0
    __aicore__ inline void ComputeRangeAndThreads(
        int32_t &rangeStart, int32_t &rangeEnd, uint32_t &simtThreadNum)
    {
        uint32_t blockId = AscendC::GetBlockIdx();
        int64_t rangeStart64 = static_cast<int64_t>(blockId) * static_cast<int64_t>(perBlock_);
        int64_t totalDim64 = static_cast<int64_t>(format_ == SPARSE2DENSE_FMT_CSC ? n_ : m_);
        if (rangeStart64 >= totalDim64) {
            rangeStart = static_cast<int32_t>(totalDim64);
            rangeEnd = rangeStart;
            simtThreadNum = 0;
            return;
        }
        int64_t rangeEnd64 = rangeStart64 + static_cast<int64_t>(perBlock_);
        if (rangeEnd64 > totalDim64) {
            rangeEnd64 = totalDim64;
        }
        rangeStart = static_cast<int32_t>(rangeStart64);
        rangeEnd = static_cast<int32_t>(rangeEnd64);
        int32_t coreRangeLen = rangeEnd - rangeStart;
        if (coreRangeLen <= 0) {
            simtThreadNum = 0;
            return;
        }
        simtThreadNum = (coreRangeLen < static_cast<int32_t>(kSparse2DenseMaxThreadsPerBlock))
            ? static_cast<uint32_t>(coreRangeLen)
            : kSparse2DenseMaxThreadsPerBlock;
        simtThreadNum = AlignToWarp(simtThreadNum);
    }

    // COO: 每核处理的 nnz 范围
    // 尾核可能 nnzStart >= nnz_，此时 simtThreadNum 置 0 避免 uint64 下溢
    __aicore__ inline void ComputeNnzRangeAndThreads(
        uint64_t &nnzStart, uint64_t &nnzEnd, uint32_t &simtThreadNum)
    {
        uint32_t blockId = AscendC::GetBlockIdx();
        nnzStart = static_cast<uint64_t>(blockId) * static_cast<uint64_t>(perBlock_);
        if (nnzStart >= nnz_) {
            nnzEnd = nnz_;
            simtThreadNum = 0;
            return;
        }
        nnzEnd = nnzStart + static_cast<uint64_t>(perBlock_);
        if (nnzEnd > nnz_) {
            nnzEnd = nnz_;
        }
        uint64_t coreNnz = nnzEnd - nnzStart;
        simtThreadNum = (coreNnz < kSparse2DenseMaxThreadsPerBlock)
            ? static_cast<uint32_t>(coreNnz)
            : kSparse2DenseMaxThreadsPerBlock;
        simtThreadNum = AlignToWarp(simtThreadNum);
    }
};

// ===========================================================================
// Layer 1a: CSR/CSC SIMT VF 计算函数
//   CSR: 逐行遍历非零元，scatter 到 dnMat[r, col]
//   CSC: 逐列遍历非零元，scatter 到 dnMat[row, c]
// ===========================================================================
template <typename ValT>
__simt_vf__ __aicore__ __launch_bounds__(kSparse2DenseMaxThreadsPerBlock) inline void
Sparse2DenseSimtComputeCsrCsc(
    __gm__ const int32_t *offsets,   // CSR: rowOffsets; CSC: colOffsets
    __gm__ const int32_t *indices,   // CSR: colInd;    CSC: rowInd
    __gm__ const ValT    *values,
    __gm__ ValT          *dnMat,
    int32_t indexBase,
    int32_t isColMajor,
    int32_t ld,
    int32_t rangeStart, int32_t rangeEnd,
    int32_t threadsPerCore,
    int32_t format)
{
    const int32_t rangeLen = rangeEnd - rangeStart;
    for (int32_t idx = static_cast<int32_t>(threadIdx.x);
         idx < rangeLen;
         idx += threadsPerCore) {
        const int32_t major = rangeStart + idx;  // CSR: row; CSC: col
        const int32_t oStart = offsets[major]     - indexBase;
        const int32_t oEnd   = offsets[major + 1] - indexBase;

        for (int32_t p = oStart; p < oEnd; ++p) {
            const int32_t minor = indices[p] - indexBase;  // CSR: col; CSC: row
            const ValT val = values[p];

            int64_t denseIdx;
            if (format == SPARSE2DENSE_FMT_CSC) {
                // CSC: major=col, minor=row
                if (isColMajor) {
                    denseIdx = static_cast<int64_t>(major) * ld + minor;
                } else {
                    denseIdx = static_cast<int64_t>(minor) * ld + major;
                }
            } else {
                // CSR: major=row, minor=col
                if (isColMajor) {
                    denseIdx = static_cast<int64_t>(minor) * ld + major;
                } else {
                    denseIdx = static_cast<int64_t>(major) * ld + minor;
                }
            }
            dnMat[denseIdx] = val;
        }
    }
}

// ===========================================================================
// Layer 1b: COO SIMT VF 计算函数
//   按 nnz grid-stride 遍历，每个线程处理若干非零元
// ===========================================================================
template <typename ValT>
__simt_vf__ __aicore__ __launch_bounds__(kSparse2DenseMaxThreadsPerBlock) inline void
Sparse2DenseSimtComputeCoo(
    __gm__ const int32_t *cooRowInd,
    __gm__ const int32_t *cooColInd,
    __gm__ const ValT    *values,
    __gm__ ValT          *dnMat,
    int32_t indexBase,
    int32_t isColMajor,
    int32_t ld,
    uint64_t nnzStart, uint64_t nnzEnd,
    int32_t threadsPerCore)
{
    const uint64_t init = static_cast<uint64_t>(threadIdx.x);
    for (uint64_t p = nnzStart + init; p < nnzEnd; p += static_cast<uint64_t>(threadsPerCore)) {
        const int32_t row = cooRowInd[p] - indexBase;
        const int32_t col = cooColInd[p] - indexBase;
        const ValT val = values[p];

        int64_t denseIdx;
        if (isColMajor) {
            denseIdx = static_cast<int64_t>(col) * ld + row;
        } else {
            denseIdx = static_cast<int64_t>(row) * ld + col;
        }
        dnMat[denseIdx] = val;
    }
}

// ===========================================================================
// 主转换 Dispatcher
// ===========================================================================
class Sparse2DenseDispatcher : public Sparse2DenseDispatcherBase {
public:
    __aicore__ inline void Init(
        GM_ADDR gmOffsets, GM_ADDR gmIndices, GM_ADDR gmValues,
        GM_ADDR gmDnMat,
        const Sparse2DenseTilingData *tiling)
    {
        offsets_  = (__gm__ const int32_t *)gmOffsets;
        indices_  = (__gm__ const int32_t *)gmIndices;
        values_   = (__gm__ const uint8_t *)gmValues;
        dnMat_    = (__gm__ uint8_t *)gmDnMat;
        m_          = tiling->m;
        n_          = tiling->n;
        indexBase_  = tiling->indexBase;
        valueType_  = tiling->valueType;
        isColMajor_ = tiling->isColMajor;
        ld_         = tiling->ld;
        perBlock_   = tiling->perBlock;
        format_     = tiling->format;
        nnz_        = tiling->nnz;
    }

    __aicore__ inline void Process()
    {
        if (format_ == SPARSE2DENSE_FMT_COO) {
            uint64_t nnzStart{}, nnzEnd{};
            uint32_t simtThreadNum{};
            ComputeNnzRangeAndThreads(nnzStart, nnzEnd, simtThreadNum);
            if (simtThreadNum == 0) return;
            DispatchCoo(simtThreadNum, nnzStart, nnzEnd, static_cast<int32_t>(simtThreadNum));
        } else {
            int32_t rangeStart{}, rangeEnd{};
            uint32_t simtThreadNum{};
            ComputeRangeAndThreads(rangeStart, rangeEnd, simtThreadNum);
            if (simtThreadNum == 0) return;
            DispatchCsrCsc(simtThreadNum, rangeStart, rangeEnd, static_cast<int32_t>(simtThreadNum));
        }
    }

private:
    __aicore__ inline void DispatchCsrCsc(
        uint32_t simtThreadNum,
        int32_t rangeStart, int32_t rangeEnd, int32_t threadsPerCore)
    {
#define S2D_DISPATCH_CSR(KernelFunc, ValT) \
        asc_vf_call<KernelFunc<ValT>>(dim3{simtThreadNum}, offsets_, indices_, \
            reinterpret_cast<__gm__ const ValT *>(values_), \
            reinterpret_cast<__gm__ ValT *>(dnMat_), \
            indexBase_, isColMajor_, ld_, rangeStart, rangeEnd, threadsPerCore, format_)
        if (valueType_ == SPARSE2DENSE_VAL_F16)      { S2D_DISPATCH_CSR(Sparse2DenseSimtComputeCsrCsc, half); }
        else if (valueType_ == SPARSE2DENSE_VAL_BF16) { S2D_DISPATCH_CSR(Sparse2DenseSimtComputeCsrCsc, bfloat16_t); }
        else if (valueType_ == SPARSE2DENSE_VAL_I32)  { S2D_DISPATCH_CSR(Sparse2DenseSimtComputeCsrCsc, int32_t); }
        else if (valueType_ == SPARSE2DENSE_VAL_I8)   { S2D_DISPATCH_CSR(Sparse2DenseSimtComputeCsrCsc, int8_t); }
        else                                           { S2D_DISPATCH_CSR(Sparse2DenseSimtComputeCsrCsc, float); }
#undef S2D_DISPATCH_CSR
    }

    __aicore__ inline void DispatchCoo(
        uint32_t simtThreadNum,
        uint64_t nnzStart, uint64_t nnzEnd, int32_t threadsPerCore)
    {
#define S2D_DISPATCH_COO(KernelFunc, ValT) \
        asc_vf_call<KernelFunc<ValT>>(dim3{simtThreadNum}, offsets_, indices_, \
            reinterpret_cast<__gm__ const ValT *>(values_), \
            reinterpret_cast<__gm__ ValT *>(dnMat_), \
            indexBase_, isColMajor_, ld_, nnzStart, nnzEnd, threadsPerCore)
        if (valueType_ == SPARSE2DENSE_VAL_F16)      { S2D_DISPATCH_COO(Sparse2DenseSimtComputeCoo, half); }
        else if (valueType_ == SPARSE2DENSE_VAL_BF16) { S2D_DISPATCH_COO(Sparse2DenseSimtComputeCoo, bfloat16_t); }
        else if (valueType_ == SPARSE2DENSE_VAL_I32)  { S2D_DISPATCH_COO(Sparse2DenseSimtComputeCoo, int32_t); }
        else if (valueType_ == SPARSE2DENSE_VAL_I8)   { S2D_DISPATCH_COO(Sparse2DenseSimtComputeCoo, int8_t); }
        else                                           { S2D_DISPATCH_COO(Sparse2DenseSimtComputeCoo, float); }
#undef S2D_DISPATCH_COO
    }

    __gm__ const int32_t *offsets_{nullptr};
    __gm__ const int32_t *indices_{nullptr};
    __gm__ const uint8_t *values_{nullptr};
    __gm__ uint8_t       *dnMat_{nullptr};
};

// ===========================================================================
// __global__ 调度器
// ===========================================================================
extern "C" __global__ __aicore__ void sparse2dense_kernel(
    GM_ADDR gmOffsets, GM_ADDR gmIndices, GM_ADDR gmValues,
    GM_ADDR gmDnMat,
    const Sparse2DenseTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    Sparse2DenseDispatcher dispatcher;
    dispatcher.Init(gmOffsets, gmIndices, gmValues, gmDnMat, &tiling);
    dispatcher.Process();
}

// kernel_do 启动器
extern "C" void sparse2dense_kernel_do(
    GM_ADDR sparseOffsets, GM_ADDR sparseIndices, GM_ADDR sparseValues,
    GM_ADDR dense,
    const Sparse2DenseTilingData &tiling,
    uint32_t numBlocks,
    void *stream)
{
    sparse2dense_kernel<<<numBlocks, nullptr, stream>>>(
        sparseOffsets, sparseIndices, sparseValues, dense, tiling);
}
