/**
 * ----------------------------------------------------------------------------------------------------------
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ----------------------------------------------------------------------------------------------------------
 */

/*!
 * \file cscsort_kernel.cpp
 * \brief aclsparseXcscsort vector kernel（arch22 / Ascend 910B/910C）。
 *
 * arch22 限制：AscendC Sort/MrgSort 仅支持 half/float 键且为纯降序（int32
 * RADIX_SORT 仅 950 起），无 SIMT。热路径采用【唯一 float 合成键 + 硬件
 * Sort 降序全排】构造稳定升序：
 *   key = (segMaxRow-row) * 2^posBits + (len-1-pos)
 * segMaxRow 为段内实测最大行号（kernel 内 ReduceMax 自适应，不依赖 host 的
 * m，torch hook 等传占位 m 的调用方亦正确）。key 为 < 2^24 的非负精确整数
 * （float32 尾数精确上限），段内唯一 ⇒ 排序结果完全确定，稳定性不依赖硬件
 * tie 行为；降序排 key ⇒ 行号升序、同行按原始位置升序（稳定）。pad 槽位键
 * 为 -1.0f，小于一切真实键，降序沉底。Sort 的 index 通道预填 iota，输出每
 * 秩源下标，Extract 解交织后按字节偏移 Gather 同步重排 rowInd/P。键宽预算
 * segRowBits+posBits ≤ 24，超预算的段回退自实现标量两路稳定归并
 * （MergeSortUb，正确性兜底）。
 *
 *   - 短列（len <= runSize）：多列聚合成批，每列独占一个 32 元素对齐的 UB
 *     槽位，逐列 DataCopyPad 搬入、整批一次硬件 Sort（isFullSort=false 按
 *     32 组独立排序，repeat=段数，单段固定开销按批摊薄）、逐列 DataCopyPad
 *     写回。批内实测最大行号（整批一次 ReduceMax，不依赖 host 的 m，行号
 *     超出 m 也保持按值排序语义）统一全批 baseTerm，唯一键语义与逐段全排
 *     等价；键宽超预算 / 混入 >32 槽位段时回退逐段路径（短段标量归并、
 *     长段矢量全排）。len<=1 的列天然有序不进批。批量化摊薄 pipe Reset / event
 *     屏障开销。注意：DataCopyPad 的 UB 侧地址必须 32B 对齐（否则触发
 *     vector core exception 507035）；排序后的 UB 数据只能由 MTE3 读出，
 *     故写回不能用标量循环。
 *   - 长列（len > runSize）：按 runSize 分块在 UB 排序写入 GM workspace
 *     形成有序 run，再多趟两路稳定归并。归并的 GM 写一律经 UB 暂存 +
 *     DataCopyPad（MTE3 带字节使能），不用标量 GM 写——实测标量 GM 写与
 *     相邻列所属核的 MTE3 写在共享 32B 扇区上会互相踩回旧数据（列边界
 *     非 32B 对齐时）。归并输入为标量 GM 只读，无此问题。
 *
 * 以累计 nnz 为权重把完整列区间分配给各核；列边界通过 cscColPtr 二分定位，
 * 不拆分单列。各核只访问自己列区间对应的 rowInd/P/workspace，无需核间同步。
 * 每列 segment [begin, end) 内对 cscRowInd 做单键稳定升序排序，P 同步重排。
 */

#include <cstdint>
#include "kernel_operator.h"
#include "cscsort_kernel.h"

using namespace AscendC;

class CscsortKernel {
public:
    __aicore__ inline CscsortKernel() {}

    __aicore__ inline void Init(GM_ADDR gmColPtr, GM_ADDR gmRowInd, GM_ADDR gmP, GM_ADDR gmWs,
                                const CscsortTilingData &td, TPipe *pipe)
    {
        n_ = td.n;
        nnz_ = td.nnz;
        indexBase_ = td.indexBase;
        runSize_ = td.runSize;
        coreNum_ = td.coreNum;
        sortTmpBytes_ = td.sortTmpBytes;
        id_ = GetBlockIdx();
        pipe_ = pipe;

        colPtr_ = reinterpret_cast<__gm__ int32_t *>(gmColPtr);
        rowInd_ = reinterpret_cast<__gm__ int32_t *>(gmRowInd);
        p_ = reinterpret_cast<__gm__ int32_t *>(gmP);
        scratchRowInd_ = reinterpret_cast<__gm__ int32_t *>(gmWs);
        scratchP_ = scratchRowInd_ + td.nnz;
        colPtrGM_.SetGlobalBuffer(colPtr_);
        rowIndGM_.SetGlobalBuffer(rowInd_);
        pGM_.SetGlobalBuffer(p_);
        scratchRowIndGM_.SetGlobalBuffer(scratchRowInd_);
        scratchPGM_.SetGlobalBuffer(scratchP_);
    }

    __aicore__ inline void Process()
    {
        if (coreNum_ == 0U || id_ >= coreNum_) {
            return;
        }
        uint64_t elemBegin = static_cast<uint64_t>(nnz_) * id_ / coreNum_;
        uint64_t elemEnd = static_cast<uint64_t>(nnz_) * (id_ + 1U) / coreNum_;
        uint32_t colBegin = FindColBoundary(elemBegin);
        uint32_t colEnd = FindColBoundary(elemEnd);
        InitRunBuffers();
        segCount_ = 0U;
        batchSlots_ = 0U;
        int32_t begin =
            (colBegin < colEnd)
                ? colPtrGM_.GetValue(colBegin) - static_cast<int32_t>(indexBase_) : 0;
        for (uint32_t col = colBegin; col < colEnd; col++) {
            int32_t end =
                colPtrGM_.GetValue(col + 1U) - static_cast<int32_t>(indexBase_);
            uint32_t len = static_cast<uint32_t>(end - begin);
            if (len > runSize_) {
                FlushBatch();
                SortLongCol(begin, len);
            } else if (len > 1U) {
                // UB 槽位按 32B（8 个 int32）对齐，保证 DataCopyPad 的 UB 侧地址对齐
                uint32_t slot = Align32Elems(len);
                if (slot != kSlotElems) {
                    // 非 32 槽位段不能与 32 槽位段同批：批量化 isFullSort=false 的
                    // 硬件 Sort 按 32 组独立排序，仅对单 32 组段（len<=32）全局有
                    // 序。先清空既有纯 32 批，再让该段独占一批，FlushBatch 内部
                    // 因批内含非 32 槽位段自动回退逐段路径。
                    FlushBatch();
                    segBegin_[0] = begin;
                    segLen_[0] = len;
                    segCount_ = 1U;
                    batchSlots_ = slot;
                    FlushBatch();
                } else {
                    if (segCount_ > 0U &&
                        (batchSlots_ + slot > Align32Elems(runSize_) || segCount_ >= kMaxBatchSegs)) {
                        FlushBatch();
                    }
                    segBegin_[segCount_] = begin;
                    segLen_[segCount_] = len;
                    segCount_++;
                    batchSlots_ += slot;
                }
            }
            begin = end;
        }
        FlushBatch();
    }

private:
    static constexpr uint32_t kKeyMaxBits = 24U;    // float32 精确整数位宽红线
    static constexpr uint32_t kVecSortMinLen = 64U;  // 矢量 Sort 最短段长（交叉点~32，留裕量）
    static constexpr uint32_t kSlotElems = 32U;      // 短列批处理槽位宽（=硬件 Sort 单组宽）
    static constexpr uint32_t kSlotPosBits = 5U;     // 槽内位置字段位宽 log2(32)

    __aicore__ inline uint32_t Align32Elems(uint32_t count)
    {
        return (count + 31U) & ~31U;
    }

    // ceil(log2(x+1))；posBits（位置字段）与 rowBits（行号字段）位宽用
    __aicore__ inline uint32_t BitWidth(uint64_t v)
    {
        uint32_t bits = 0U;
        while ((static_cast<uint64_t>(1U) << bits) <= v) {
            ++bits;
        }
        return bits;
    }

    /// 计算合成键行域基准 baseTerm（= 行域上界 * 2^posBits）。始终 ReduceMax
    /// 实测段内最大行号（V_S 屏障读回），不依赖 host 传入的 m：行号超出 m 时
    /// 仍按整数值正确排序（对齐 cuSPARSE 按值排序语义），占位 m（INT_MAX 等）
    /// 调用方同样正确。
    /// 返回 false 表示键宽超 24 位预算，调用方需回退标量归并。
    ///（独立函数以控制 NBNC ≤ 50，codecheck 要求）
    __aicore__ inline bool ProbeSegBaseTerm(uint32_t off, uint32_t len, uint32_t posBits,
                                            LocalTensor<float> &keyF, LocalTensor<float> &tmp,
                                            int32_t &baseTermI)
    {
        // 段内实测最大行号自适应。maxT 借用 dstIdx 槽位（后续 Extract 会覆盖，
        // 无冲突）。
        LocalTensor<float> maxT = dstIdxBuf_.Get<float>();
        ReduceMax<float>(maxT, keyF[off], tmp, static_cast<int32_t>(Align32Elems(len)));
        auto vS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(vS);
        WaitFlag<HardEvent::V_S>(vS);
        int32_t segMaxRow = static_cast<int32_t>(maxT.GetValue(0));
        // float 舍入只会把 >=2^24 的值舍到 >=2^24（BitWidth>=25 必回退），
        // <2^24 精确；行域位宽 + 位置位宽超 24 位则回退标量归并（绝不静默出错）。
        if (BitWidth(static_cast<uint64_t>(segMaxRow)) + posBits > kKeyMaxBits) {
            return false;
        }
        baseTermI = segMaxRow * static_cast<int32_t>(static_cast<uint64_t>(1U) << posBits);
        return true;
    }

    /// 矢量热路径：对 UB 槽位 [off, off+len) 执行硬件 Sort 全排。返回 false 表示
    /// 键宽超预算（行域位宽 + 位置位宽 > 24），调用方需回退标量归并。
    ///
    /// 合成键 key = (segMaxRow-row) * 2^posBits + (len-1-pos)：segMaxRow 为段内
    /// 实测最大行号（ReduceMax），不依赖 host 传入的 m（torch hook 等调用方可能
    /// 传占位值）；key 为 < 2^24 的非负精确整数（float32 尾数精确上限），段内唯一
    /// ⇒ 降序全排 = 行号升序、同行按原始位置升序（稳定），结果不依赖硬件 tie 行为。
    /// pad 槽位键为 -1.0f，降序沉底；ReduceMax 计入 pad 不影响（行号恒非负）。
    /// Sort 的 index 通道预填 iota（int32 版 CreateVecIndex，uint32 版内部 Adds 不
    /// 被 A2 支持，非负值位型等价），Extract 解交织后按字节偏移 Gather 重排到
    /// keyOut/pOut 同槽位。调用方负责 MTE2_V 前置与 V_MTE3 后置屏障。
    __aicore__ inline bool VecSortSlot(uint32_t off, uint32_t len)
    {
        uint32_t padded = Align32Elems(len);
        uint32_t posBits = BitWidth(len - 1U);

        LocalTensor<int32_t> keyIn = keyInBuf_.Get<int32_t>();
        LocalTensor<int32_t> pIn = pInBuf_.Get<int32_t>();
        LocalTensor<float> keyF = keyFBuf_.Get<float>();
        LocalTensor<int32_t> iota = iotaBuf_.Get<int32_t>();
        LocalTensor<int32_t> posTerm = posTermBuf_.Get<int32_t>();
        LocalTensor<float> dstKey = dstKeyBuf_.Get<float>();
        LocalTensor<uint32_t> dstIdx = dstIdxBuf_.Get<uint32_t>();
        LocalTensor<float> tmp = sortTmpBuf_.Get<float>();

        Duplicate(keyF[off], -1.0f, padded);
        Cast(keyF[off], keyIn[off], RoundMode::CAST_NONE, static_cast<int32_t>(len));
        int32_t baseTermI = 0;
        if (!ProbeSegBaseTerm(off, len, posBits, keyF, tmp, baseTermI)) {
            return false;
        }
        int32_t posMulI = static_cast<int32_t>(static_cast<uint64_t>(1U) << posBits);

        CreateVecIndex(iota[off], 0, padded);
        Muls(posTerm[off], iota[off], -1, static_cast<int32_t>(len));
        Adds(posTerm[off], posTerm[off], static_cast<int32_t>(len - 1U),
             static_cast<int32_t>(len));
        Cast(posTerm[off].ReinterpretCast<float>(), posTerm[off], RoundMode::CAST_NONE,
             static_cast<int32_t>(len));

        Muls(keyF[off], keyF[off], -static_cast<float>(posMulI), static_cast<int32_t>(len));
        Adds(keyF[off], keyF[off], static_cast<float>(baseTermI), static_cast<int32_t>(len));
        Add(keyF[off], keyF[off], posTerm[off].ReinterpretCast<float>(),
            static_cast<int32_t>(len));

        uint32_t repeat = padded / 32U;
        AscendC::Sort<float, true>(dstKey[off * 2U], keyF[off],
                                   iota[off].ReinterpretCast<uint32_t>(), tmp,
                                   static_cast<int32_t>(repeat));
        Extract(keyF[off], dstIdx[off], dstKey[off * 2U], static_cast<int32_t>(repeat));

        // 字节偏移后 Gather：pad 秩指向 pad 槽位，写回窗口只取前 len 秩
        ShiftLeft<uint32_t>(dstIdx[off], dstIdx[off], 2U, static_cast<int32_t>(padded));
        LocalTensor<int32_t> keyOut = keyOutBuf_.Get<int32_t>();
        LocalTensor<int32_t> pOut = pOutBuf_.Get<int32_t>();
        Gather<int32_t>(keyOut[off], keyIn[off], dstIdx[off], 0U, padded);
        Gather<int32_t>(pOut[off], pIn[off], dstIdx[off], 0U, padded);
        return true;
    }

    /// 在 cscColPtr 中二分定位：返回包含 elementOffset（按 nnz 累计、不含
    /// indexBase）的列号，即首个满足 colPtr[col] >= elementOffset+indexBase 的列。
    __aicore__ inline uint32_t FindColBoundary(uint64_t elementOffset)
    {
        if (elementOffset == 0U) {
            return 0U;
        }
        if (elementOffset >= nnz_) {
            return n_;
        }
        int64_t target = static_cast<int64_t>(elementOffset) + static_cast<int64_t>(indexBase_);
        uint32_t low = 0U;
        uint32_t high = n_;
        while (low < high) {
            uint32_t mid = low + (high - low) / 2U;
            if (static_cast<int64_t>(colPtrGM_.GetValue(mid)) < target) {
                low = mid + 1U;
            } else {
                high = mid;
            }
        }
        return low;
    }

    /// 归并单对相邻 run：[pairBase, pairBase+aLen) 与 [bBase, bBase+bLen)，
    /// 相等键优先取左 run 保证稳定。归并头缓存在标量寄存器：每元素仅 1 次
    /// UB 读（前进侧）+ 2 次 UB 写。（独立函数以控制嵌套深度 ≤5，codecheck 要求）
    __aicore__ inline void MergePairUb(__ubuf__ int32_t *srcKey, __ubuf__ int32_t *srcP,
                                       __ubuf__ int32_t *dstKey, __ubuf__ int32_t *dstP,
                                       uint32_t pairBase, uint32_t aLen,
                                       uint32_t bBase, uint32_t bLen)
    {
        uint32_t i = 0U;
        uint32_t j = 0U;
        int32_t aKey = srcKey[pairBase];
        int32_t aP = srcP[pairBase];
        int32_t bKey = 0;
        int32_t bP = 0;
        if (bLen > 0U) {
            bKey = srcKey[bBase];
            bP = srcP[bBase];
        }
        for (uint32_t k = 0U; k < aLen + bLen; k++) {
            bool takeA = (j >= bLen) || (i < aLen && aKey <= bKey);
            if (takeA) {
                dstKey[pairBase + k] = aKey;
                dstP[pairBase + k] = aP;
                i++;
                if (i < aLen) {
                    aKey = srcKey[pairBase + i];
                    aP = srcP[pairBase + i];
                }
            } else {
                dstKey[pairBase + k] = bKey;
                dstP[pairBase + k] = bP;
                j++;
                if (j < bLen) {
                    bKey = srcKey[bBase + j];
                    bP = srcP[bBase + j];
                }
            }
        }
    }

    /// UB 内自底向上稳定归并：srcKey/srcP ->（ping-pong）-> 结果位于 src 侧返回。
    /// 返回最终有序数据所在的 buffer 侧（false=in，true=out）。
    /// 用 __ubuf__ 裸指针访问 UB（比 GetValue/SetValue 开销低）。
    __aicore__ inline bool MergeSortUb(__ubuf__ int32_t *srcKey, __ubuf__ int32_t *srcP,
                                       __ubuf__ int32_t *dstKey, __ubuf__ int32_t *dstP,
                                       uint32_t len)
    {
        bool inOut = false;
        for (uint32_t width = 1U; width < len; width *= 2U) {
            uint32_t pairSpan = width * 2U;
            for (uint32_t pairBase = 0U; pairBase < len; pairBase += pairSpan) {
                uint32_t aLen = (pairBase + width < len) ? width : len - pairBase;
                uint32_t bBase = pairBase + aLen;
                uint32_t bLen =
                    (bBase < len) ? ((bBase + width < len) ? width : len - bBase) : 0U;
                MergePairUb(srcKey, srcP, dstKey, dstP, pairBase, aLen, bBase, bLen);
            }
            __ubuf__ int32_t *swapKey = srcKey;
            __ubuf__ int32_t *swapP = srcP;
            srcKey = dstKey;
            srcP = dstP;
            dstKey = swapKey;
            dstP = swapP;
            inOut = !inOut;
        }
        return inOut;
    }

    /// 每个核一次性初始化 UB run buffer（原实现逐列 Reset+InitBuffer，
    /// 是短列场景的主要固定开销之一）。
    __aicore__ inline void InitRunBuffers()
    {
        pipe_->Reset();
        uint32_t bufferBytes = Align32Elems(runSize_) * sizeof(int32_t);
        pipe_->InitBuffer(keyInBuf_, bufferBytes);
        pipe_->InitBuffer(pInBuf_, bufferBytes);
        pipe_->InitBuffer(keyOutBuf_, bufferBytes);
        pipe_->InitBuffer(pOutBuf_, bufferBytes);
        pipe_->InitBuffer(keyFBuf_, bufferBytes);
        pipe_->InitBuffer(iotaBuf_, bufferBytes);
        pipe_->InitBuffer(posTermBuf_, bufferBytes);
        pipe_->InitBuffer(dstIdxBuf_, bufferBytes);
        pipe_->InitBuffer(dstKeyBuf_, bufferBytes * 2U);  // 全排输出 2N (score,index) 交织
        pipe_->InitBuffer(sortTmpBuf_, sortTmpBytes_);
    }

    /// 短列批处理：批内每列独占一个 32B 对齐的 UB 槽位（DataCopyPad 的 UB 侧
    /// 地址必须 32B 对齐，否则小 blockLen 拷贝会触发 vector core exception
    /// 507035；标量读 UB 写在归并之后也拿不到新数据，故写回只能用 MTE3）。
    /// len<=1 的列天然有序不进批。屏障与 pipe 初始化按批摊薄。
    /// FlushBatch 步骤 1：逐段搬入 32 元素对齐槽位（GM 侧不对齐由 DataCopyPad
    /// 处理；矢量 Sort 以 repeat×32 整体处理槽内 padded 区）。
    ///（独立函数以控制 NBNC ≤ 50，codecheck 要求）
    __aicore__ inline void BatchCopyIn(LocalTensor<int32_t> &keyIn, LocalTensor<int32_t> &pIn)
    {
        DataCopyPadExtParams<int32_t> pad{false, 0, 0, 0};
        uint32_t off = 0U;
        for (uint32_t s = 0U; s < segCount_; s++) {
            DataCopyExtParams cp{1, static_cast<uint32_t>(segLen_[s] * sizeof(int32_t)), 0, 0, 0};
            DataCopyPad(keyIn[off], rowIndGM_[segBegin_[s]], cp, pad);
            DataCopyPad(pIn[off], pGM_[segBegin_[s]], cp, pad);
            off += Align32Elems(segLen_[s]);
        }
    }

    /// FlushBatch 步骤 3：按各列结果侧逐段写回 GM（UB 源 32B 对齐；GM 侧不对齐
    /// 由 DataCopyPad 处理）。
    __aicore__ inline void BatchWriteBack(LocalTensor<int32_t> &keyIn, LocalTensor<int32_t> &pIn,
                                          LocalTensor<int32_t> &keyOut, LocalTensor<int32_t> &pOut)
    {
        uint32_t off = 0U;
        for (uint32_t s = 0U; s < segCount_; s++) {
            DataCopyExtParams cp{1, static_cast<uint32_t>(segLen_[s] * sizeof(int32_t)), 0, 0, 0};
            LocalTensor<int32_t> resKey = segInOut_[s] ? keyOut[off] : keyIn[off];
            LocalTensor<int32_t> resP = segInOut_[s] ? pOut[off] : pIn[off];
            DataCopyPad(rowIndGM_[segBegin_[s]], resKey, cp);
            DataCopyPad(pGM_[segBegin_[s]], resP, cp);
            off += Align32Elems(segLen_[s]);
        }
    }

    /// 批内单段装载原始行号到 float 键缓冲（槽位统一 32 宽）：整槽先置
    /// -1.0f（pad），再 Cast 有效区；行号恒非负，pad 不影响后续 ReduceMax。
    __aicore__ inline void BatchVecSegLoad(uint32_t off, uint32_t len,
                                           LocalTensor<float> &keyF,
                                           LocalTensor<int32_t> &keyIn)
    {
        Duplicate(keyF[off], -1.0f, kSlotElems);
        Cast(keyF[off], keyIn[off], RoundMode::CAST_NONE, static_cast<int32_t>(len));
    }

    /// 批内单段合成键（槽位统一 32 宽）：key = (batchMaxRow-row)*2^5 + (31-pos)
    /// = bias - globalLane - 32*row，其中 bias = baseTerm+31+off（+off 与
    /// negIotaF 中的 -globalLane 抵消，等价槽内 31-pos）。pad 槽位在 -1.0f
    /// 基础上加 -globalLane 后为深负键，降序沉底且不与真实键（恒 ≥0）冲突。
    __aicore__ inline void BatchVecSegKeys(uint32_t off, uint32_t len, float biasF,
                                           LocalTensor<float> &keyF,
                                           LocalTensor<float> &negIotaF)
    {
        Muls(keyF[off], keyF[off],
             -static_cast<float>(static_cast<int32_t>(kSlotElems)), static_cast<int32_t>(len));
        Adds(keyF[off], keyF[off], biasF, static_cast<int32_t>(len));
        Add(keyF[off], keyF[off], negIotaF[off], kSlotElems);
    }

    /// 短列批量化矢量排序：批内每列独占一个 32 元素槽位、槽间天然独立，整批
    /// 合成一次硬件 Sort（isFullSort=false 按 32 组独立排序，repeat=段数），
    /// 把单段 Sort 的固定指令/屏障开销按批摊薄（P-01 量级 18 元素/列的关键
    /// 提速：标量归并 ~14μs/列 → 批摊后约 1-2μs/列）。槽位统一 32 宽 ⇒
    /// posBits=5；行域上界取批内实测最大行号（整批一次 ReduceMax，不依赖
    /// host 的 m，行号超出 m 也保持按值排序语义），baseTerm 全批统一，唯一键
    /// 语义与逐段全排等价：降序 = 行号升序、同行按原始位置升序（稳定）。
    /// 返回 false 表示批内存在非 32 槽位段或键宽超预算，调用方回退逐段路径。
    /// 结果恒在 out 侧。
    __aicore__ inline bool TryBatchVecSort(LocalTensor<int32_t> &keyIn, LocalTensor<int32_t> &pIn,
                                           LocalTensor<int32_t> &keyOut, LocalTensor<int32_t> &pOut)
    {
        for (uint32_t s = 0U; s < segCount_; s++) {
            if (Align32Elems(segLen_[s]) != kSlotElems) {
                return false;
            }
        }
        uint32_t slots = batchSlots_;  // 每段恰占 32 槽，slots = segCount_*32
        LocalTensor<int32_t> iota = iotaBuf_.Get<int32_t>();
        LocalTensor<float> negIotaF = posTermBuf_.Get<float>();
        LocalTensor<float> keyF = keyFBuf_.Get<float>();
        LocalTensor<float> dstKey = dstKeyBuf_.Get<float>();
        LocalTensor<uint32_t> dstIdx = dstIdxBuf_.Get<uint32_t>();
        LocalTensor<float> tmp = sortTmpBuf_.Get<float>();
        uint32_t off = 0U;
        for (uint32_t s = 0U; s < segCount_; s++) {
            BatchVecSegLoad(off, segLen_[s], keyF, keyIn);
            off += kSlotElems;
        }
        // 批内实测最大行号（V_S 屏障读回）；maxT 借用 dstIdx 槽位（后续
        // Extract 会覆盖，无冲突）。行号 < 2^24 时 float 精确；≥2^24 必触发
        // 下方键宽回退（BitWidth>=25），绝不静默出错。
        LocalTensor<float> maxT = dstIdxBuf_.Get<float>();
        ReduceMax<float>(maxT, keyF[0], tmp, static_cast<int32_t>(slots));
        auto vS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(vS);
        WaitFlag<HardEvent::V_S>(vS);
        int32_t batchMaxRow = static_cast<int32_t>(maxT.GetValue(0));
        if (BitWidth(static_cast<uint64_t>(batchMaxRow)) + kSlotPosBits > kKeyMaxBits) {
            return false;
        }
        int32_t baseTermI = batchMaxRow << kSlotPosBits;
        CreateVecIndex(iota[0], 0, slots);
        Cast(negIotaF[0], iota[0], RoundMode::CAST_NONE, static_cast<int32_t>(slots));
        Muls(negIotaF[0], negIotaF[0], -1.0f, static_cast<int32_t>(slots));
        off = 0U;
        for (uint32_t s = 0U; s < segCount_; s++) {
            float biasF = static_cast<float>(
                baseTermI + static_cast<int32_t>(kSlotElems - 1U) + static_cast<int32_t>(off));
            BatchVecSegKeys(off, segLen_[s], biasF, keyF, negIotaF);
            off += kSlotElems;
        }
        // 整批一次硬件排序（槽间独立），index 通道携带全批 lane 下标
        AscendC::Sort<float, false>(dstKey[0], keyF[0], iota[0].ReinterpretCast<uint32_t>(),
                                    tmp, static_cast<int32_t>(segCount_));
        Extract(keyF[0], dstIdx[0], dstKey[0], static_cast<int32_t>(segCount_));
        // 字节偏移后 Gather：pad 秩指向 pad 槽位，写回窗口只取各列前 len 秩
        ShiftLeft<uint32_t>(dstIdx[0], dstIdx[0], 2U, static_cast<int32_t>(slots));
        Gather<int32_t>(keyOut[0], keyIn[0], dstIdx[0], 0U, slots);
        Gather<int32_t>(pOut[0], pIn[0], dstIdx[0], 0U, slots);
        return true;
    }

    __aicore__ inline void FlushBatch()
    {
        if (segCount_ == 0U) {
            return;
        }
        LocalTensor<int32_t> keyIn = keyInBuf_.Get<int32_t>();
        LocalTensor<int32_t> pIn = pInBuf_.Get<int32_t>();
        LocalTensor<int32_t> keyOut = keyOutBuf_.Get<int32_t>();
        LocalTensor<int32_t> pOut = pOutBuf_.Get<int32_t>();

        // 1) 逐段搬入对齐槽位
        BatchCopyIn(keyIn, pIn);
        // MTE2（GM->UB 搬运）→ V/S（矢量排序与标量归并读 UB）屏障
        auto mte2v = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(mte2v);
        WaitFlag<HardEvent::MTE2_V>(mte2v);
        auto mte2s = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
        SetFlag<HardEvent::MTE2_S>(mte2s);
        WaitFlag<HardEvent::MTE2_S>(mte2s);

        // 2) 批内排序：优先整批一次硬件 Sort（槽间独立，isFullSort=false，
        //    单段固定开销按批摊薄）；不可用（键宽超预算 / 含 >32 槽位段）
        //    回退逐段路径。
        if (TryBatchVecSort(keyIn, pIn, keyOut, pOut)) {
            for (uint32_t s = 0U; s < segCount_; s++) {
                // 批量化矢量热路径：硬件 Sort 全组独立排序，结果恒在 out 侧
                segInOut_[s] = true;
            }
        } else {
            __ubuf__ int32_t *keyInPhy = reinterpret_cast<__ubuf__ int32_t *>(keyIn.GetPhyAddr());
            __ubuf__ int32_t *pInPhy = reinterpret_cast<__ubuf__ int32_t *>(pIn.GetPhyAddr());
            __ubuf__ int32_t *keyOutPhy = reinterpret_cast<__ubuf__ int32_t *>(keyOut.GetPhyAddr());
            __ubuf__ int32_t *pOutPhy = reinterpret_cast<__ubuf__ int32_t *>(pOut.GetPhyAddr());
            uint32_t off = 0U;
            for (uint32_t s = 0U; s < segCount_; s++) {
                // 矢量 Sort 单段固定开销大（十余条矢量指令 + Sort 内 PipeBarrier），
                // 短列（P-01 量级 ~18/列）标量归并反而更快；kVecSortMinLen 以上且
                // 键宽够才走矢量。实测交叉点约 32 元素，取 64 留裕量。
                if (segLen_[s] >= kVecSortMinLen && VecSortSlot(off, segLen_[s])) {
                    // 矢量热路径：硬件 Sort 全排，结果恒在 out 侧
                    segInOut_[s] = true;
                } else {
                    segInOut_[s] = MergeSortUb(keyInPhy + off, pInPhy + off,
                                               keyOutPhy + off, pOutPhy + off, segLen_[s]);
                }
                off += Align32Elems(segLen_[s]);
            }
        }
        // V/S（排序写 UB）→ MTE3（DataCopyPad 读 UB）屏障
        auto vMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(vMte3);
        WaitFlag<HardEvent::V_MTE3>(vMte3);
        auto sMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
        SetFlag<HardEvent::S_MTE3>(sMte3);
        WaitFlag<HardEvent::S_MTE3>(sMte3);

        // 3) 逐段写回
        BatchWriteBack(keyIn, pIn, keyOut, pOut);
        // MTE3（写回读 UB）→ MTE2（下一批 copy-in 复写 UB）屏障
        auto mte3Mte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
        SetFlag<HardEvent::MTE3_MTE2>(mte3Mte2);
        WaitFlag<HardEvent::MTE3_MTE2>(mte3Mte2);
        segCount_ = 0U;
        batchSlots_ = 0U;
    }

    /// 排序单个 run（len <= runSize_）：从原始 rowInd/P 读入 UB，稳定归并排序后
    /// 写到 dstKeyGM/dstPGM（原地或 scratch）。buffer 已由 InitRunBuffers 初始化。
    __aicore__ inline void SortSingleRun(int64_t begin, uint32_t len,
                                         GlobalTensor<int32_t> &dstKeyGM,
                                         GlobalTensor<int32_t> &dstPGM)
    {
        DataCopyExtParams cp{1, static_cast<uint32_t>(len * sizeof(int32_t)), 0, 0, 0};
        DataCopyPadExtParams<int32_t> pad{false, 0, 0, 0};

        LocalTensor<int32_t> keyIn = keyInBuf_.Get<int32_t>();
        LocalTensor<int32_t> pIn = pInBuf_.Get<int32_t>();
        DataCopyPad(keyIn, rowIndGM_[begin], cp, pad);
        DataCopyPad(pIn, pGM_[begin], cp, pad);
        // MTE2（GM->UB 搬运）→ V/S（矢量排序与标量归并读 UB）屏障
        auto mte2v = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(mte2v);
        WaitFlag<HardEvent::MTE2_V>(mte2v);
        auto mte2s = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
        SetFlag<HardEvent::MTE2_S>(mte2s);
        WaitFlag<HardEvent::MTE2_S>(mte2s);

        LocalTensor<int32_t> keyOut = keyOutBuf_.Get<int32_t>();
        LocalTensor<int32_t> pOut = pOutBuf_.Get<int32_t>();
        bool inOut = true;
        if (!VecSortSlot(0U, len)) {
            inOut = MergeSortUb(
                reinterpret_cast<__ubuf__ int32_t *>(keyIn.GetPhyAddr()),
                reinterpret_cast<__ubuf__ int32_t *>(pIn.GetPhyAddr()),
                reinterpret_cast<__ubuf__ int32_t *>(keyOut.GetPhyAddr()),
                reinterpret_cast<__ubuf__ int32_t *>(pOut.GetPhyAddr()), len);
        }
        LocalTensor<int32_t> resKey = inOut ? keyOut : keyIn;
        LocalTensor<int32_t> resP = inOut ? pOut : pIn;

        // V/S（排序写 UB）→ MTE3（UB->GM 搬运）屏障
        auto vMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(vMte3);
        WaitFlag<HardEvent::V_MTE3>(vMte3);
        auto sMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
        SetFlag<HardEvent::S_MTE3>(sMte3);
        WaitFlag<HardEvent::S_MTE3>(sMte3);
        DataCopyPad(dstKeyGM[begin], resKey, cp);
        DataCopyPad(dstPGM[begin], resP, cp);
        auto mte3Mte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
        SetFlag<HardEvent::MTE3_MTE2>(mte3Mte2);
        WaitFlag<HardEvent::MTE3_MTE2>(mte3Mte2);
    }

    /// 长列（len > runSize_）：分块 UB 排序成 run 写入 scratch，随后 GM 上
    /// 多趟两路稳定归并。归并读为标量 GM 读（只读无并发写冲突），写一律经
    /// UB 暂存 + DataCopyPad（MTE3 字节使能下刷），避免标量 GM 写与相邻列
    /// 所属核的 MTE3 写在共享 32B/128B 扇区上互相覆盖（实测非对齐列边界会
    /// 踩回旧数据，RunSizeBoundaries 回归）。
    __aicore__ inline void SortLongCol(int64_t begin, uint32_t len)
    {
        for (uint32_t off = 0U; off < len; off += runSize_) {
            uint32_t runLen =
                (off + runSize_ < len) ? runSize_ : (len - off);
            SortSingleRun(begin + static_cast<int64_t>(off), runLen,
                          scratchRowIndGM_, scratchPGM_);
        }
        // MTE3（DataCopyPad 写 scratch）→ S（标量归并读 GM）屏障
        auto mte3s = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_S));
        SetFlag<HardEvent::MTE3_S>(mte3s);
        WaitFlag<HardEvent::MTE3_S>(mte3s);

        bool srcIsScratch = true;
        for (uint32_t width = runSize_; width < len; width *= 2U) {
            if (srcIsScratch) {
                MergeRoundUb(scratchRowInd_, scratchP_, rowIndGM_, pGM_, begin, len, width);
            } else {
                MergeRoundUb(rowInd_, p_, scratchRowIndGM_, scratchPGM_, begin, len, width);
            }
            srcIsScratch = !srcIsScratch;
        }
        if (srcIsScratch) {
            // 奇数趟后结果位于 scratch，回拷到 rowInd/P（UB 中转，无标量 GM 写）
            CopyColUb(scratchRowIndGM_, scratchPGM_, rowIndGM_, pGM_, begin, len);
        }
    }

    /// 单个归并输出的 UB 分块填充：从两 run 的当前头取较小者（相等取左保稳定），
    /// 写满 chunk 个或 pair 耗尽为止，返回实际产出数。输入为标量 GM 只读。
    ///（独立函数以控制嵌套深度 ≤5，codecheck 要求）
    __aicore__ inline uint32_t MergeChunkUb(__gm__ int32_t *srcKey, __gm__ int32_t *srcP,
                                            __ubuf__ int32_t *ubKey, __ubuf__ int32_t *ubP,
                                            int64_t aOff, uint32_t aLen, int64_t bOff, uint32_t bLen,
                                            uint32_t &i, uint32_t &j, int32_t &aKey, int32_t &aP,
                                            int32_t &bKey, int32_t &bP, uint32_t chunk)
    {
        uint32_t c = 0U;
        while (c < chunk && (i < aLen || j < bLen)) {
            bool takeA = (j >= bLen) || (i < aLen && aKey <= bKey);
            if (takeA) {
                ubKey[c] = aKey;
                ubP[c] = aP;
                i++;
                if (i < aLen) {
                    aKey = srcKey[aOff + i];
                    aP = srcP[aOff + i];
                }
            } else {
                ubKey[c] = bKey;
                ubP[c] = bP;
                j++;
                if (j < bLen) {
                    bKey = srcKey[bOff + j];
                    bP = srcP[bOff + j];
                }
            }
            c++;
        }
        return c;
    }

    /// 单个相邻 run 对的流式归并：输出按 chunkCap（Align32(runSize_)）分块经
    /// UB 暂存 + DataCopyPad 下刷，杜绝标量 GM 写（独立函数以控制 NBNC ≤ 50，
    /// codecheck 要求）。
    __aicore__ inline void MergePairStream(__gm__ int32_t *srcKey, __gm__ int32_t *srcP,
                                           GlobalTensor<int32_t> &dstKeyGM,
                                           GlobalTensor<int32_t> &dstPGM,
                                           int64_t begin, uint32_t pairBase, uint32_t aLen,
                                           uint32_t bBase, uint32_t bLen)
    {
        LocalTensor<int32_t> ubKey = keyOutBuf_.Get<int32_t>();
        LocalTensor<int32_t> ubP = pOutBuf_.Get<int32_t>();
        __ubuf__ int32_t *ubKeyPhy = reinterpret_cast<__ubuf__ int32_t *>(ubKey.GetPhyAddr());
        __ubuf__ int32_t *ubPPhy = reinterpret_cast<__ubuf__ int32_t *>(ubP.GetPhyAddr());
        uint32_t chunkCap = Align32Elems(runSize_);
        int64_t aOff = begin + static_cast<int64_t>(pairBase);
        int64_t bOff = begin + static_cast<int64_t>(bBase);
        uint32_t i = 0U;
        uint32_t j = 0U;
        uint32_t produced = 0U;
        uint32_t pairLen = aLen + bLen;
        int32_t aKey = srcKey[aOff];
        int32_t aP = srcP[aOff];
        int32_t bKey = 0;
        int32_t bP = 0;
        if (bLen > 0U) {
            bKey = srcKey[bOff];
            bP = srcP[bOff];
        }
        while (produced < pairLen) {
            uint32_t chunk = pairLen - produced;
            if (chunk > chunkCap) {
                chunk = chunkCap;
            }
            uint32_t filled = MergeChunkUb(srcKey, srcP, ubKeyPhy, ubPPhy, aOff, aLen,
                                           bOff, bLen, i, j, aKey, aP, bKey, bP, chunk);
            if (filled == 0U) {
                break;
            }
            // S（标量写 UB）→ MTE3（DataCopyPad 读 UB）屏障
            auto sMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
            SetFlag<HardEvent::S_MTE3>(sMte3);
            WaitFlag<HardEvent::S_MTE3>(sMte3);
            DataCopyExtParams cp{1, static_cast<uint32_t>(filled * sizeof(int32_t)), 0, 0, 0};
            int64_t dstOff = begin + static_cast<int64_t>(pairBase + produced);
            DataCopyPad(dstKeyGM[dstOff], ubKey, cp);
            DataCopyPad(dstPGM[dstOff], ubP, cp);
            // MTE3（读 UB 下刷）→ S（下一块覆写 UB）屏障
            auto mte3s = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_S));
            SetFlag<HardEvent::MTE3_S>(mte3s);
            WaitFlag<HardEvent::MTE3_S>(mte3s);
            produced += filled;
        }
    }

    /// 单趟两路稳定归并（GM 读 + UB 暂存 + DataCopyPad 写）：把 [begin, begin+len)
    /// 内相邻的 width 长 run 两两合并。
    __aicore__ inline void MergeRoundUb(__gm__ int32_t *srcKey, __gm__ int32_t *srcP,
                                        GlobalTensor<int32_t> &dstKeyGM,
                                        GlobalTensor<int32_t> &dstPGM,
                                        int64_t begin, uint32_t len, uint32_t width)
    {
        uint32_t pairSpan = width * 2U;
        for (uint32_t pairBase = 0U; pairBase < len; pairBase += pairSpan) {
            uint32_t aLen = (pairBase + width < len) ? width : len - pairBase;
            uint32_t bBase = pairBase + aLen;
            uint32_t bLen =
                (bBase < len) ? ((bBase + width < len) ? width : len - bBase) : 0U;
            MergePairStream(srcKey, srcP, dstKeyGM, dstPGM, begin, pairBase, aLen, bBase, bLen);
        }
    }

    /// 整列回拷（scratch -> rowInd/P）：UB 分块中转 DataCopyPad，无标量 GM 写。
    __aicore__ inline void CopyColUb(GlobalTensor<int32_t> &srcKeyGM,
                                     GlobalTensor<int32_t> &srcPGM,
                                     GlobalTensor<int32_t> &dstKeyGM,
                                     GlobalTensor<int32_t> &dstPGM,
                                     int64_t begin, uint32_t len)
    {
        LocalTensor<int32_t> ubKey = keyOutBuf_.Get<int32_t>();
        LocalTensor<int32_t> ubP = pOutBuf_.Get<int32_t>();
        uint32_t chunkCap = Align32Elems(runSize_);
        DataCopyPadExtParams<int32_t> pad{false, 0, 0, 0};
        // 前序归并趟的 MTE3 写 → 本次 MTE2 读 屏障
        auto mte3mte2Pre = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
        SetFlag<HardEvent::MTE3_MTE2>(mte3mte2Pre);
        WaitFlag<HardEvent::MTE3_MTE2>(mte3mte2Pre);
        for (uint32_t off = 0U; off < len; off += chunkCap) {
            uint32_t chunk = (off + chunkCap < len) ? chunkCap : (len - off);
            DataCopyExtParams cp{1, static_cast<uint32_t>(chunk * sizeof(int32_t)), 0, 0, 0};
            int64_t pos = begin + static_cast<int64_t>(off);
            DataCopyPad(ubKey, srcKeyGM[pos], cp, pad);
            DataCopyPad(ubP, srcPGM[pos], cp, pad);
            auto mte2mte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_MTE3));
            SetFlag<HardEvent::MTE2_MTE3>(mte2mte3);
            WaitFlag<HardEvent::MTE2_MTE3>(mte2mte3);
            DataCopyPad(dstKeyGM[pos], ubKey, cp);
            DataCopyPad(dstPGM[pos], ubP, cp);
            auto mte3mte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
            SetFlag<HardEvent::MTE3_MTE2>(mte3mte2);
            WaitFlag<HardEvent::MTE3_MTE2>(mte3mte2);
        }
    }

private:
    TPipe *pipe_{nullptr};
    uint32_t id_{0U};
    uint32_t coreNum_{1U};
    uint32_t sortTmpBytes_{0U};
    uint32_t n_{0U};
    uint32_t nnz_{0U};
    uint32_t indexBase_{0U};
    uint32_t runSize_{0U};
    __gm__ int32_t *colPtr_{nullptr};
    __gm__ int32_t *rowInd_{nullptr};
    __gm__ int32_t *p_{nullptr};
    __gm__ int32_t *scratchRowInd_{nullptr};
    __gm__ int32_t *scratchP_{nullptr};
    GlobalTensor<int32_t> colPtrGM_;
    GlobalTensor<int32_t> rowIndGM_;
    GlobalTensor<int32_t> pGM_;
    GlobalTensor<int32_t> scratchRowIndGM_;
    GlobalTensor<int32_t> scratchPGM_;
    TBuf<TPosition::VECCALC> keyInBuf_;
    TBuf<TPosition::VECCALC> pInBuf_;
    TBuf<TPosition::VECCALC> keyOutBuf_;
    TBuf<TPosition::VECCALC> pOutBuf_;
    TBuf<TPosition::VECCALC> keyFBuf_;
    TBuf<TPosition::VECCALC> iotaBuf_;
    TBuf<TPosition::VECCALC> posTermBuf_;
    TBuf<TPosition::VECCALC> dstIdxBuf_;
    TBuf<TPosition::VECCALC> dstKeyBuf_;
    TBuf<TPosition::VECCALC> sortTmpBuf_;
    // 短列批处理状态：一批短列共享一次屏障/pipe 初始化，列间槽位 32B 对齐
    static constexpr uint32_t kMaxBatchSegs = 256U;
    int32_t segBegin_[kMaxBatchSegs];
    uint32_t segLen_[kMaxBatchSegs];
    bool segInOut_[kMaxBatchSegs];
    uint32_t segCount_{0U};
    uint32_t batchSlots_{0U};
};

extern "C" __global__ __aicore__ void cscsort_kernel(
    GM_ADDR cscColPtr, GM_ADDR cscRowInd, GM_ADDR P, GM_ADDR workspace, const CscsortTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    CscsortKernel op;
    TPipe pipe;
    op.Init(cscColPtr, cscRowInd, P, workspace, tiling, &pipe);
    op.Process();
}

void cscsort_kernel_do(
    void *cscColPtr, void *cscRowInd, void *P, void *workspace,
    const CscsortTilingData &tiling, uint32_t numBlocks, void *stream)
{
    // void * → GM_ADDR 需越过地址空间限定，用 C 风格转换（AscendC 惯例写法）
    cscsort_kernel<<<numBlocks, nullptr, stream>>>(
        (GM_ADDR)cscColPtr, (GM_ADDR)cscRowInd, (GM_ADDR)P, (GM_ADDR)workspace, tiling);
}
