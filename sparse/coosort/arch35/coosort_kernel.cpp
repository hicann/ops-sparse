/**
 * ----------------------------------------------------------------------------------------------------------
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms of conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ----------------------------------------------------------------------------------------------------------
 */

/*!
 * \file coosort_kernel.cpp
 * \brief aclsparseXcoosort kernel 实现（标准 AscendC Vector Op，单核，arch35）。
 *
 * 算法（两趟 LSB radix 稳定排序，int32 signed）：
 *   全程 TQue 搬运 GM↔UB：TQue<VECIN> 的 AllocTensor/DataCopy/EnQue/DeQue 自动插入
 *   MTE2→V HardEvent 同步；TQue<VECOUT> 的 AllocTensor/Compute/EnQue/DeQue/DataCopy
 *   自动插入 V→MTE3 同步。不再使用标量 SetValue 读取 MTE2 加载的数据（前一实现的
 *   MTE2→S / MTE2→V 同步缺失正是 bug 根因）。
 *
 *   1. CopyIn：TQue<VECIN> DataCopy row/col(int32) → DeQue 得 rowTensor/colTensor。
 *   2. Pass1（次键 Sort）：对次键稳定排序（ByRow 次键=col，ByColumn 次键=row）。
 *      4 参 Sort<int32>(dstKey1, dstIdx1, secondaryTensor, nnz)（srcIndex 默认
 *      identity），dstIdx1 = 次键稳定排序的排列。
 *   3. Gather：按 dstIdx1 重排主键数组，得到 primaryGathered[i]=primary[dstIdx1[i]]。
 *   4. Pass2（主键 Sort）：对 primaryGathered 稳定排序 → dstIdx2。
 *   5. 最终 P = dstIdx1[dstIdx2[i]]；outRow[i]=rowTensor[P[i]]，
 *      outCol[i]=colTensor[P[i]]（向量化 Gather）。
 *   6. CopyOut：TQue<VECOUT> DataCopy outRow/outCol/P → GM。
 *
 * 两趟 LSB radix 排序天然稳定：相等的 (主键,次键) 保持原序。全程 int32 signed 升序
 * 排序，Sort 原生支持负数，无需符号位归一（区别于复合 uint64 键方案）。两趟间用
 * PipeBarrier<PIPE_V> 保证 V→V 依赖。
 */

#include <cstdint>
#include "kernel_operator.h"
#include "coosort_kernel.h"
#include "coosort_multi_core_kernel.h"

using namespace AscendC;

namespace {
constexpr uint32_t kOneBlkSize = 32;  // AscendC ONE_BLK_SIZE

/// 将元素数向上对齐到 32B 对齐所需的元素数（按 sizeof(T)）。
template <typename T>
__aicore__ inline uint32_t AlignTo32B(uint32_t count)
{
    constexpr uint32_t elemPer32B = kOneBlkSize / sizeof(T);
    return (count + elemPer32B - 1U) & ~(elemPer32B - 1U);
}

/// 用标量 SetValue 把 [nnz, alignNnz) 填为常量哨兵（升序最大），并做 S→V 同步。
/// 写常量不依赖 MTE2 加载的 GM 数据，S→V HardEvent 同步即可（参见文件头注释）。
/// 用于 Sort 输入尾部填充，避免 LSB radix 读取填充区零值污染 [0,nnz) 输出。
__aicore__ inline void PadTailConst(LocalTensor<int32_t> &tensor, uint32_t nnz, uint32_t alignNnz, int32_t value)
{
    for (uint32_t i = nnz; i < alignNnz; i++) {
        tensor.SetValue(i, value);
    }
    auto evt = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
    SetFlag<HardEvent::S_V>(evt);
    WaitFlag<HardEvent::S_V>(evt);
}
}  // namespace

/// 单核 coosort：CopyIn / 两趟 Sort / Gather / CopyOut 全程 TQue。
class CoosortSort {
public:
    __aicore__ inline CoosortSort()
    {}

    __aicore__ inline void Init(GM_ADDR gmRow, GM_ADDR gmCol, GM_ADDR gmP, const CoosortTilingData &tiling)
    {
        nnz_ = tiling.nnz;
        sortByRow_ = tiling.sortByRow != 0U;

        // 32B 对齐元素数（int32 / uint32 一致）。
        alignI32_ = AlignTo32B<int32_t>(nnz_);

        rowGM_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(gmRow));
        colGM_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(gmCol));
        pGM_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(gmP));

        // GM→UB 输入队列（自动 MTE2→V 同步）。
        pipe_.InitBuffer(rowInQue_, 1, alignI32_ * sizeof(int32_t));
        pipe_.InitBuffer(colInQue_, 1, alignI32_ * sizeof(int32_t));

        // UB→GM 输出队列（自动 V→MTE3 同步）。
        pipe_.InitBuffer(outRowQue_, 1, alignI32_ * sizeof(int32_t));
        pipe_.InitBuffer(outColQue_, 1, alignI32_ * sizeof(int32_t));
        pipe_.InitBuffer(outPQue_, 1, alignI32_ * sizeof(int32_t));

        // V-pipe 工作缓冲（全程向量化，不依赖 MTE2 加载）。
        pipe_.InitBuffer(primaryGathered_, alignI32_ * sizeof(int32_t));
        pipe_.InitBuffer(dstKey1Buf_, alignI32_ * sizeof(int32_t));
        pipe_.InitBuffer(dstIdx1Buf_, alignI32_ * sizeof(uint32_t));
        pipe_.InitBuffer(dstKey2Buf_, alignI32_ * sizeof(int32_t));
        pipe_.InitBuffer(dstIdx2Buf_, alignI32_ * sizeof(uint32_t));
        pipe_.InitBuffer(pU32Buf_, alignI32_ * sizeof(uint32_t));
        // Gather 的 srcOffset 按「字节偏移」解释（dav-3510 实现内右移 2 位转元素索引），
        // 故需把 Sort 输出的元素索引 ×sizeof(int32) 得到字节偏移后再传入 Gather。
        pipe_.InitBuffer(idx1ByteBuf_, alignI32_ * sizeof(uint32_t));
        pipe_.InitBuffer(idx2ByteBuf_, alignI32_ * sizeof(uint32_t));
        pipe_.InitBuffer(pByteBuf_, alignI32_ * sizeof(uint32_t));

        // Sort 共享临时（host 侧 GetSortMaxMinTmpSize 按 int32/uint32 取 max；两趟 Sort<int32_t> 串行复用同一缓冲）。
        pipe_.InitBuffer(sharedTmpBuf_, tiling.sortTmpSize);
    }

    __aicore__ inline void Process()
    {
        if (nnz_ == 0) {
            return;
        }
        CopyIn();
        Compute();
        CopyOut();
    }

private:
    __aicore__ inline void CopyIn()
    {
        // row
        LocalTensor<int32_t> rowTensor = rowInQue_.template AllocTensor<int32_t>();
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(nnz_ * sizeof(int32_t)), 0, 0, 0};
        DataCopyPadExtParams<int32_t> padParams{false, 0, 0, 0};
        DataCopyPad(rowTensor, rowGM_, copyParams, padParams);
        rowInQue_.EnQue(rowTensor);

        // col
        LocalTensor<int32_t> colTensor = colInQue_.template AllocTensor<int32_t>();
        DataCopyPad(colTensor, colGM_, copyParams, padParams);
        colInQue_.EnQue(colTensor);
    }

    __aicore__ inline void Compute()
    {
        // DeQue 后 MTE2→V 已同步，rowTensor/colTensor 可安全被 V-pipe 读取。
        LocalTensor<int32_t> rowTensor = rowInQue_.template DeQue<int32_t>();
        LocalTensor<int32_t> colTensor = colInQue_.template DeQue<int32_t>();

        LocalTensor<int32_t> primaryGathered = primaryGathered_.Get<int32_t>();
        LocalTensor<int32_t> dstKey1 = dstKey1Buf_.Get<int32_t>();
        LocalTensor<uint32_t> dstIdx1 = dstIdx1Buf_.Get<uint32_t>();
        LocalTensor<int32_t> dstKey2 = dstKey2Buf_.Get<int32_t>();
        LocalTensor<uint32_t> dstIdx2 = dstIdx2Buf_.Get<uint32_t>();
        LocalTensor<uint8_t> sharedTmp = sharedTmpBuf_.Get<uint8_t>();

        // 主/次键选择：ByRow 主=row 次=col；ByColumn 主=col 次=row。
        LocalTensor<int32_t> primaryTensor = sortByRow_ ? rowTensor : colTensor;
        LocalTensor<int32_t> secondarySrc = sortByRow_ ? colTensor : rowTensor;

        // Sort 的 LSB radix 读取整条 256B 寄存器（含 nnz 之后的对齐填充区）。
        // 默认 srcIndex=[0,1,...,alignNnz-1]，若填充区为 0（升序最小）会排到前面，
        // 污染 [0,nnz) 的输出。将 [nnz, alignNnz) 填 INT32_MAX（升序最大）使其排到末尾，
        // 落在 calCount 之外，保证 [0,nnz) 输出均为 [0,nnz) 内的有效索引。
        // 用标量 SetValue 写常量（不依赖 MTE2 加载的 GM 数据，S→V 同步可用——
        // 前一实现 bug 根因是 SetValue 读 GM 的 MTE2→S 缺失，写常量无此问题）。
        constexpr int32_t kPadSentinel = 0x7FFFFFFF;  // INT32_MAX
        PadTailConst(secondarySrc, nnz_, alignI32_, kPadSentinel);

        // --- Pass 1：对次键稳定排序（升序）---
        // secondaryTensor 直接复用 col/row 的 DeQue 结果（V-pipe 可读），无需拷贝；
        // 但 Sort 禁止输出与输入地址重叠，secondaryTensor 作为 src 与 dstKey1/dstIdx1
        // 不同缓冲，安全。
        // kernel 侧 SortConfig 指定排序类型和方向；Host 查询临时空间时还会
        // 显式设置 hasSrcIndex/hasDstIndex，使查询配置与实际调用方式一致。
        static constexpr SortConfig sortConfig{SortType::RADIX_SORT, false};
        Sort<int32_t, false, sortConfig>(dstKey1, dstIdx1, secondarySrc, sharedTmp, nnz_);
        // 两趟 Sort 间需 V→V 同步。
        PipeBarrier<PIPE_V>();

        // --- Gather：按 dstIdx1 重排主键 → primaryGathered ---
        // Gather 的 srcOffset 按「字节偏移」解释（dav-3510 VfGatherApi0B32 内右移 2 位
        // 转元素索引），故先用 ShiftLeft 把元素索引 dstIdx1 ×4 得到字节偏移 idx1Byte。
        LocalTensor<uint32_t> idx1Byte = idx1ByteBuf_.Get<uint32_t>();
        ShiftLeft<uint32_t>(idx1Byte, dstIdx1, 2U, static_cast<int32_t>(nnz_));
        PipeBarrier<PIPE_V>();
        // dst[i]=src[(srcBaseOffset+srcOffset[i])/sizeof(T)]
        Gather<int32_t>(primaryGathered, primaryTensor, idx1Byte, 0U, nnz_);
        PadTailConst(primaryGathered, nnz_, alignI32_, kPadSentinel);
        PipeBarrier<PIPE_V>();

        // --- Pass 2：对 primaryGathered 稳定排序（升序）→ dstIdx2 ---
        Sort<int32_t, false, sortConfig>(dstKey2, dstIdx2, primaryGathered, sharedTmp, nnz_);
        PipeBarrier<PIPE_V>();

        // --- 最终 P[i] = dstIdx1[dstIdx2[i]] ---
        // 先用 dstIdx2 gather dstIdx1 得到 P（uint32），再 DataCopy 到 int32 outP 写出。
        LocalTensor<int32_t> outP = outPQue_.template AllocTensor<int32_t>();
        LocalTensor<uint32_t> pU32 = pU32Buf_.Get<uint32_t>();
        LocalTensor<uint32_t> idx2Byte = idx2ByteBuf_.Get<uint32_t>();
        ShiftLeft<uint32_t>(idx2Byte, dstIdx2, 2U, static_cast<int32_t>(nnz_));
        PipeBarrier<PIPE_V>();
        Gather<uint32_t>(pU32, dstIdx1, idx2Byte, 0U, nnz_);
        PipeBarrier<PIPE_V>();
        // uint32 permutation 位等价 reinterpret 为 int32 输出(避免 Cast<int32_t,uint32_t>
        // 在 dav-3510 损坏数据, 见 dav3510-ascendc-pitfalls)。
        DataCopy(outP, pU32Buf_.Get<int32_t>(), alignI32_);
        outPQue_.EnQue(outP);

        // --- outRow[i]=rowTensor[P[i]], outCol[i]=colTensor[P[i]] ---
        LocalTensor<int32_t> outRow = outRowQue_.template AllocTensor<int32_t>();
        LocalTensor<int32_t> outCol = outColQue_.template AllocTensor<int32_t>();
        LocalTensor<uint32_t> pByte = pByteBuf_.Get<uint32_t>();
        ShiftLeft<uint32_t>(pByte, pU32, 2U, static_cast<int32_t>(nnz_));
        PipeBarrier<PIPE_V>();
        Gather<int32_t>(outRow, rowTensor, pByte, 0U, nnz_);
        Gather<int32_t>(outCol, colTensor, pByte, 0U, nnz_);
        outRowQue_.EnQue(outRow);
        outColQue_.EnQue(outCol);

        rowInQue_.FreeTensor(rowTensor);
        colInQue_.FreeTensor(colTensor);
    }

    __aicore__ inline void CopyOut()
    {
        // DeQue 后 V→MTE3 已同步。
        LocalTensor<int32_t> outRow = outRowQue_.template DeQue<int32_t>();
        LocalTensor<int32_t> outCol = outColQue_.template DeQue<int32_t>();
        LocalTensor<int32_t> outP = outPQue_.template DeQue<int32_t>();
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(nnz_ * sizeof(int32_t)), 0, 0, 0};
        DataCopyPad(rowGM_, outRow, copyParams);
        DataCopyPad(colGM_, outCol, copyParams);
        DataCopyPad(pGM_, outP, copyParams);
        outRowQue_.FreeTensor(outRow);
        outColQue_.FreeTensor(outCol);
        outPQue_.FreeTensor(outP);
    }

private:
    TPipe pipe_;

    GlobalTensor<int32_t> rowGM_;
    GlobalTensor<int32_t> colGM_;
    GlobalTensor<int32_t> pGM_;

    TQue<TPosition::VECIN, 1> rowInQue_;
    TQue<TPosition::VECIN, 1> colInQue_;
    TQue<TPosition::VECOUT, 1> outRowQue_;
    TQue<TPosition::VECOUT, 1> outColQue_;
    TQue<TPosition::VECOUT, 1> outPQue_;

    TBuf<TPosition::VECCALC> primaryGathered_;
    TBuf<TPosition::VECCALC> dstKey1Buf_;
    TBuf<TPosition::VECCALC> dstIdx1Buf_;
    TBuf<TPosition::VECCALC> dstKey2Buf_;
    TBuf<TPosition::VECCALC> dstIdx2Buf_;
    TBuf<TPosition::VECCALC> pU32Buf_;
    TBuf<TPosition::VECCALC> idx1ByteBuf_;
    TBuf<TPosition::VECCALC> idx2ByteBuf_;
    TBuf<TPosition::VECCALC> pByteBuf_;
    TBuf<TPosition::VECCALC> sharedTmpBuf_;

    uint32_t nnz_{0};
    uint32_t alignI32_{0};
    bool sortByRow_{true};
};

extern "C" __global__ __aicore__ void coosort_kernel(
    GM_ADDR row, GM_ADDR col, GM_ADDR pOut, const CoosortTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    CoosortSort op;
    op.Init(row, col, pOut, tiling);
    op.Process();
}

extern "C" void coosort_kernel_do(
    GM_ADDR row, GM_ADDR col, GM_ADDR pOut, const CoosortTilingData &tiling, uint32_t numBlocks, void *stream)
{
    coosort_kernel<<<numBlocks, nullptr, stream>>>(row, col, pOut, tiling);
}
