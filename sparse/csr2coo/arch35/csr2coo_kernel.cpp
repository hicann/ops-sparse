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
 * \file csr2coo_kernel.cpp
 * \brief aclsparseXcsr2coo kernel implementation (SIMT + SIMD hybrid, Ascend950 / arch35).
 *
 * Converts CSR rowPtr (length m+1) to COO cooRowInd (length nnz).
 * Algorithm: per-row Duplicate fill + chunked DataCopyPad output.
 * Only supports int32_t index type (Legacy API).
 */

#include <cstdint>

#include "kernel_operator.h"
#include "simt_api/asc_simt.h"
#include "csr2coo_kernel.h"

namespace {

// ---------------------------------------------------------------------------
// SIMT VF: 两层分解——调度器按 simtRowsPerBlock 预算核间行范围 [myFirstRow,
// myLastRow)，VF 内核内 grid-stride（stride=threadNum）。
// 偏离 ascendc-simt-best-practices 标准单层 grid-stride（stride=threadNum*blockNum），
// 原因：csr2coo 的 m 通常远小于 numBlocks*threadNum（如 m=4096 vs 56*1024=57344），
// 标准模式下大量 block 完全空转（仅前 ceil(m/threadNum) 个 block 有工作），实测
// medium/large 场景退化 14%~22%。两层分解确保每个 block 都有连续行块处理。
// 直读 __gm__ rowPtr[row]/rowPtr[row+1]，逐元素写 __gm__ cooRowIndGm[j]=rowVal。
// 模板 <IDX_BASE> 使 idxBase 模板化（0/1 两个特化），rowVal 计算为编译期常量加法。
// 异常行用 continue 跳过（SIMT 线程独立执行，return 会影响整个 warp，故不用 return）。
// 越界校验保留：Host 无法读 Device rowPtr，校验必须在 kernel 侧（与 SIMD 路径一致）。
// ---------------------------------------------------------------------------
template <int32_t IDX_BASE>
__simt_vf__ __aicore__ __launch_bounds__(csr2coo::kCsr2CooSimtThreads) inline void Csr2CooSimtCompute(
    int32_t myFirstRow, int32_t myLastRow, int64_t nnz,
    __gm__ int32_t *rowPtrGm, __gm__ int32_t *cooRowIndGm)
{
    auto threadIdx = static_cast<int32_t>(AscendC::Simt::GetThreadIdx());
    auto threadNum = static_cast<int32_t>(AscendC::Simt::GetThreadNum());
    for (int64_t row = static_cast<int64_t>(myFirstRow) + threadIdx;
         row < static_cast<int64_t>(myLastRow); row += threadNum) {
        int32_t rowStart = rowPtrGm[row] - IDX_BASE;
        int32_t rowEnd = rowPtrGm[row + 1] - IDX_BASE;
        int32_t rowNnz = rowEnd - rowStart;
        if (rowNnz <= 0) {
            continue;
        }
        if (rowStart < 0 ||
            static_cast<int64_t>(rowStart) + static_cast<int64_t>(rowNnz) > nnz) {
            continue;
        }
        int32_t rowVal = static_cast<int32_t>(row) + IDX_BASE;
        for (int32_t j = rowStart; j < rowEnd; j++) {
            cooRowIndGm[j] = rowVal;
        }
    }
}

// ---------------------------------------------------------------------------
// Kernel class (int32_t only) — SIMD path
// ---------------------------------------------------------------------------
class Csr2CooAiv {
public:
    __aicore__ inline Csr2CooAiv() = default;

    __aicore__ inline void Init(
        GM_ADDR gmRowPtr, GM_ADDR gmCooRowInd,
        const csr2coo::Csr2CooTilingData &tiling,
        AscendC::TPipe *pipe);
    __aicore__ inline void Process();

private:
    // 签名接收局部值参数（cooChunkSize / cooRowIndBase），避免内部访问成员变量，
    // 消除 hot loop 内成员变量反复 Load（别名分析压制）。
    __aicore__ inline void ProcessRowChunked(int32_t rowNnz, int32_t rowVal,
                                             int32_t &writeOffset,
                                             uint32_t cooChunkSize,
                                             __gm__ int32_t *cooRowIndBase);

    // 单行累积处理。小行（rowNnz <= cooChunkSize）累积到 accumBuf，
    // 大行（rowNnz > cooChunkSize）flush 后走 ProcessRowChunked 分块路径。
    // accumBuf/cooGlobal/flushParams 按值传递（轻量视图对象，底层地址不变），
    // accumElems/accumWriteOffset 按引用传递（累积状态跨行维护）。
    __aicore__ inline void AccumulateRow(
        int32_t rowNnz, int32_t rowStart, int32_t rowVal,
        uint32_t cooChunkSize, __gm__ int32_t *cooRowIndBase,
        AscendC::LocalTensor<int32_t> accumBuf,
        AscendC::GlobalTensor<int32_t> cooGlobal,
        AscendC::DataCopyExtParams &flushParams,
        uint32_t &accumElems, int32_t &accumWriteOffset);

    // 公共 flush 函数：提取 AccumulateRow / Process 末尾共 3 处重复的 flush 代码。
    // accumBuf 由 SetValue(S)+Duplicate(V) 写入，DataCopyPad(MTE3) 读取，需 S→MTE3
    // 和 V→MTE3 同步。isFinal=true（Process 末尾）时无需 MTE3→S/V 反向同步（accumBuf
    // 不再被写入）；isFinal=false（AccumulateRow 行间）时需反向同步防止 RAW 竞态。
    // accumElems/accumWriteOffset 的重置由调用方负责（各调用点重置时机不同）。
    __aicore__ inline void FlushAccumBuf(
        AscendC::LocalTensor<int32_t> accumBuf,
        AscendC::GlobalTensor<int32_t> cooGlobal,
        AscendC::DataCopyExtParams &flushParams,
        uint32_t accumElems, int32_t accumWriteOffset, bool isFinal);

    // Process 拆分子函数（R7 NBNC ≤ 50）。各子函数保持 HardEvent flag 同步位置
    // 不变（跨 pipe 同步不能移动）。rpLocal/nnzLocal/startLocal 是 LocalTensor
    // 视图（轻量），按值传递；accumElems/accumWriteOffset 按引用传递（累积状态跨行维护）。
    __aicore__ inline AscendC::LocalTensor<int32_t> LoadRowPtr(
        int64_t myFirstRow, uint32_t actualElemCount, uint32_t rowPtrCountAligned);
    __aicore__ inline void BuildOffsetTable(uint32_t actualRows, uint32_t rowPtrCountAligned);
    __aicore__ inline void ComputeNnzAndStart(
        uint32_t actualRows, int32_t idxBase, uint32_t rowPtrCountAligned,
        AscendC::LocalTensor<int32_t> rpLocal,
        AscendC::LocalTensor<int32_t> &nnzLocal,
        AscendC::LocalTensor<int32_t> &startLocal);
    // idxBase 模板化（与 SIMT 路径 Csr2CooSimtCompute<IDX_BASE> 一致），
    // IDX_BASE 为编译期常量，保持常量传播优化。
    template <int32_t IDX_BASE>
    __aicore__ inline void ProcessRowsImpl(
        int64_t myFirstRow, int64_t myLastRow, int64_t nnzVal,
        uint32_t cooChunkSize, __gm__ int32_t *cooRowIndBase,
        AscendC::LocalTensor<int32_t> nnzLocal,
        AscendC::LocalTensor<int32_t> startLocal,
        AscendC::LocalTensor<int32_t> accumBuf,
        AscendC::GlobalTensor<int32_t> cooGlobal,
        AscendC::DataCopyExtParams &flushParams,
        uint32_t &accumElems, int32_t &accumWriteOffset);

    __gm__ int32_t *rowPtrGm_{nullptr};
    __gm__ int32_t *cooRowIndGm_{nullptr};

    AscendC::TPipe *pipe_{nullptr};
    AscendC::TBuf<AscendC::TPosition::VECIN> rowPtrBuf_;
    // shiftBuf 通过 Gather 从 rpLocal 提取 shift，无需独立 buffer。
    AscendC::TBuf<AscendC::TPosition::VECCALC> nnzBuf_;      // per-row nnz (vectorized precompute)
    AscendC::TBuf<AscendC::TPosition::VECCALC> startBuf_;    // per-row start offset; also reused for Gather offset table
    AscendC::TBuf<AscendC::TPosition::VECCALC> accumBuf_;    // multi-row output accumulation
    AscendC::TQue<AscendC::TPosition::VECOUT, 2> cooQueue_;  // num=2 (V/MTE3 double-buffer)

    int64_t m_{0};
    int64_t nnz_{0};
    int32_t idxBase_{0};
    int64_t blockSize_{0};
    uint32_t cooChunkSize_{0};
    uint32_t rowPtrBytes_{0};
    int64_t myFirstRow_{0};
    int64_t myLastRow_{0};
};

__aicore__ inline void Csr2CooAiv::Init(
    GM_ADDR gmRowPtr, GM_ADDR gmCooRowInd,
    const csr2coo::Csr2CooTilingData &tiling,
    AscendC::TPipe *pipe)
{
    rowPtrGm_ = reinterpret_cast<__gm__ int32_t *>(gmRowPtr);
    cooRowIndGm_ = reinterpret_cast<__gm__ int32_t *>(gmCooRowInd);

    pipe_ = pipe;
    m_ = tiling.m;
    nnz_ = tiling.nnz;
    idxBase_ = tiling.idxBase;
    blockSize_ = tiling.blockSize;
    cooChunkSize_ = tiling.cooChunkSize;
    rowPtrBytes_ = tiling.rowPtrBytes;

    // 余数分配：前 remainder 个核多处理 1 行（对齐仓内 gtsv2_strided_batch 先例）
    uint32_t bid = AscendC::GetBlockIdx();
    uint32_t baseRows = static_cast<uint32_t>(tiling.blockSize);
    uint32_t rem = tiling.remainder;
    myFirstRow_ = static_cast<int64_t>(bid) * baseRows +
                  static_cast<int64_t>(bid < rem ? bid : rem);
    myLastRow_ = myFirstRow_ + static_cast<int64_t>(baseRows) +
                 (bid < rem ? 1 : 0);
    if (myFirstRow_ >= myLastRow_) {
        return;
    }

    // Allocate UB for rowPtr slice (already 32B aligned by Host)
    pipe_->InitBuffer(rowPtrBuf_, rowPtrBytes_);

    // nnzBuf_/startBuf_ each occupy rowPtrBytes (no separate shiftBuf; shift via
    // Gather from rpLocal). startBuf_ is reused for Gather offset table (temporary)
    // before Subs overwrites it.
    pipe_->InitBuffer(nnzBuf_, rowPtrBytes_);
    pipe_->InitBuffer(startBuf_, rowPtrBytes_);

    // cooQueue_ num=2: ultrarow 大行路径 (ProcessRowChunked) 的 Duplicate(V) 与
    // DataCopyPad(MTE3) 双缓冲重叠。累积模式用 accumBuf_（TBuf），不依赖 cooQueue_
    // 的 depth，两者独立。
    // coo 相关 UB = accumBuf(1) + cooQueue[2](2) = 3 × cooBytes（Host 预算 /3）。
    uint32_t cooBytes = static_cast<uint32_t>(
        static_cast<int64_t>(cooChunkSize_) * sizeof(int32_t));
    pipe_->InitBuffer(accumBuf_, cooBytes);
    pipe_->InitBuffer(cooQueue_, 2, cooBytes);
}

// ---------------------------------------------------------------------------
// Process 拆分子函数（R7 NBNC ≤ 50）：拆分为 LoadRowPtr / BuildOffsetTable /
// ComputeNnzAndStart / ProcessRowsImpl + Process 编排。HardEvent flag 同步位置
// 保持不变（跨 pipe 同步不能移动）。
// ---------------------------------------------------------------------------

// 一次 DataCopyPad 搬入 rpLocal（actualRows+1 个元素）。shift 通过 Gather
// 从 rpLocal 偏移 1 元素提取，无需第二次 DataCopyPad。
__aicore__ inline AscendC::LocalTensor<int32_t> Csr2CooAiv::LoadRowPtr(
    int64_t myFirstRow, uint32_t actualElemCount, uint32_t rowPtrCountAligned)
{
    AscendC::GlobalTensor<int32_t> rowPtrGlobal;
    rowPtrGlobal.SetGlobalBuffer(rowPtrGm_ + myFirstRow);

    AscendC::DataCopyExtParams rowPtrCopyParams{
        1U,
        actualElemCount * static_cast<uint32_t>(sizeof(int32_t)),
        0, 0, 0};
    AscendC::DataCopyPadExtParams<int32_t> rowPtrPadParams{false, 0, 0, 0};

    auto rpLocal = rowPtrBuf_.template Get<int32_t>(rowPtrCountAligned);
    AscendC::DataCopyPad(rpLocal, rowPtrGlobal, rowPtrCopyParams, rowPtrPadParams);

    // rowPtrBuf_ 是 TBuf（非 TQue），DataCopyPad(MTE2) 写后 Gather/Subs(V) 读，
    // 需手动跨 pipe 同步。用 HardEvent::MTE2_V flag 替代 PipeBarrier（同仓 coosort
    // 已在 arch35 验证 HardEvent 同步模式），仅同步 MTE2→V 而非全流水线停顿。
    auto evtMte2V = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_V));
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(evtMte2V);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(evtMte2V);

    return rpLocal;
}

// 生成 Gather 偏移表（复用 startBuf，后续被 Subs 覆盖）。
// offsetTable[i] = (i+1) * sizeof(int32_t) = rpLocal 第 i+1 个元素的字节偏移。
// 用 SetValue 生成基础模式 {4,8,...,32}（8 个），再用 Adds 向量扩展后续组。
// 注意：Adds 不支持 uint32_t，故偏移表用 int32_t 生成（正值，与 uint32_t 位等价），
// Gather 读取时通过 Get<uint32_t> 重解释同一内存。
__aicore__ inline void Csr2CooAiv::BuildOffsetTable(
    uint32_t actualRows, uint32_t rowPtrCountAligned)
{
    auto offsetTable = startBuf_.template Get<int32_t>(rowPtrCountAligned);

    constexpr uint32_t kGroupElems = 8U;

    // SetValue 基础模式（S pipe 写常量）
    uint32_t baseCount = (actualRows < kGroupElems) ? actualRows : kGroupElems;
    for (uint32_t i = 0U; i < baseCount; i++) {
        offsetTable.SetValue(i, static_cast<int32_t>((i + 1U) * sizeof(int32_t)));
    }
    // S→V 同步：SetValue(S) 写偏移表 → Adds(V) 扩展 / Gather(V) 读。
    // 用 HardEvent::S_V flag 替代 PipeBarrier<PIPE_ALL>（同仓 coosort arch35 已验证），
    // 仅同步 S→V 而非全流水线停顿。
    auto evtSV = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::S_V));
    AscendC::SetFlag<AscendC::HardEvent::S_V>(evtSV);
    AscendC::WaitFlag<AscendC::HardEvent::S_V>(evtSV);

    // Adds 向量扩展（每组 8 个元素，组间差值恒定 32 字节）
    constexpr int32_t kGroupStride = static_cast<int32_t>(kGroupElems * sizeof(int32_t));  // 32
    uint32_t fullGroups = actualRows / kGroupElems;
    uint32_t remainElems = actualRows % kGroupElems;
    for (uint32_t g = 1U; g < fullGroups; g++) {
        AscendC::Adds(offsetTable[g * kGroupElems], offsetTable[0U],
                      static_cast<int32_t>(g) * kGroupStride, static_cast<int32_t>(kGroupElems));
    }
    if (fullGroups > 0U && remainElems > 0U) {
        AscendC::Adds(offsetTable[fullGroups * kGroupElems], offsetTable[0U],
                      static_cast<int32_t>(fullGroups) * kGroupStride,
                      static_cast<int32_t>(remainElems));
    }
}

// Gather 提取 rpLocal[1:] 到 nnzLocal（无需 shiftBuf 和第二次 DataCopyPad）。
// nnzLocal[i] = rpLocal[(0 + offsetTable[i]) / sizeof(int32_t)] = rpLocal[i+1]
// Gather 的 srcOffset 按 uint32_t 字节偏移解释，故用 Get<uint32_t> 重解释偏移表。
//
// Subs 计算 startLocal（覆盖 startBuf 中的偏移表——Gather 已消费完毕）。
// startLocal[i] = rpLocal[i] - idxBase（读局部变量 idxBase，不再读成员 idxBase_）
// Sub 计算 nnzLocal（in-place: nnzLocal = nnzLocal - rpLocal）
// nnzLocal[i] = rpLocal[i+1] - rpLocal[i] = rowPtr[firstRow+i+1] - rowPtr[firstRow+i]
// Gather 写 nnzLocal → Sub 读 nnzLocal：同 V pipe 内按序执行，无需额外同步。
__aicore__ inline void Csr2CooAiv::ComputeNnzAndStart(
    uint32_t actualRows, int32_t idxBase, uint32_t rowPtrCountAligned,
    AscendC::LocalTensor<int32_t> rpLocal,
    AscendC::LocalTensor<int32_t> &nnzLocal,
    AscendC::LocalTensor<int32_t> &startLocal)
{
    auto offsetTableU32 = startBuf_.template Get<uint32_t>(rowPtrCountAligned);
    AscendC::Gather<int32_t>(nnzLocal, rpLocal, offsetTableU32, 0U, actualRows);

    AscendC::Subs(startLocal, rpLocal, idxBase, static_cast<int32_t>(actualRows));
    AscendC::Sub(nnzLocal, nnzLocal, rpLocal, static_cast<int32_t>(actualRows));

    // V pipeline 写 nnzLocal/startLocal → S pipeline GetValue 读，需跨 pipe
    // 同步。用 HardEvent::V_S flag 替代 PipeBarrier<PIPE_ALL>（CANN softmax 实现已
    // 验证 V_S 方向），仅同步 V→S 而非全流水线停顿。非 hot loop，每核仅 1 次。
    auto evtVS = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_S));
    AscendC::SetFlag<AscendC::HardEvent::V_S>(evtVS);
    AscendC::WaitFlag<AscendC::HardEvent::V_S>(evtVS);
}

// idxBase 模板化（与 SIMT 路径 Csr2CooSimtCompute<IDX_BASE> 模式一致）。
// IDX_BASE 为编译期常量，保持常量传播优化：IDX_BASE==0 时编译器消除加法，
// IDX_BASE==1 时用 add-immediate。
//
// 异常行（rowNnz<0 || rowStart<0 || rowStart+rowNnz>nnz）用 continue 跳过，
// 与 golden/SIMT 路径统一。异常行被跳过（不调用 AccumulateRow），累积状态保持
// 不变，末尾 flush 仍正常执行。
//
// 越界校验读局部变量 nnzVal（不读成员 nnz_）；rowNnz<0||rowStart<0 已合并为单判断。
// Host 无法读 Device rowPtr，故校验保留在 kernel 侧。
template <int32_t IDX_BASE>
__aicore__ inline void Csr2CooAiv::ProcessRowsImpl(
    int64_t myFirstRow, int64_t myLastRow, int64_t nnzVal,
    uint32_t cooChunkSize, __gm__ int32_t *cooRowIndBase,
    AscendC::LocalTensor<int32_t> nnzLocal,
    AscendC::LocalTensor<int32_t> startLocal,
    AscendC::LocalTensor<int32_t> accumBuf,
    AscendC::GlobalTensor<int32_t> cooGlobal,
    AscendC::DataCopyExtParams &flushParams,
    uint32_t &accumElems, int32_t &accumWriteOffset)
{
    for (int64_t row = myFirstRow; row < myLastRow; row++) {
        uint32_t idx = static_cast<uint32_t>(row - myFirstRow);
        int32_t rowNnz = nnzLocal.GetValue(idx);
        int32_t rowStart = startLocal.GetValue(idx);

        if (rowNnz <= 0) {
            continue;
        }
        if (rowStart < 0 ||
            static_cast<int64_t>(rowStart) + static_cast<int64_t>(rowNnz) > nnzVal) {
            continue;  // 异常行跳过（与 golden/SIMT 统一），不停止整核
        }

        int32_t rowVal = static_cast<int32_t>(row) + IDX_BASE;
        AccumulateRow(rowNnz, rowStart, rowVal, cooChunkSize, cooRowIndBase,
                      accumBuf, cooGlobal, flushParams, accumElems, accumWriteOffset);
    }
}

__aicore__ inline void Csr2CooAiv::Process()
{
    if (myFirstRow_ >= myLastRow_) {
        return;
    }

    // 防御性校验：Host 保证 cooChunkSize 是 8 的倍数（32B 对齐），kernel 侧复核。
    // 若 Host bug 导致非对齐，提前 return 避免静默错误（正常情况不触发）。
    constexpr uint32_t kAlignElems = 8U;
    if (cooChunkSize_ == 0U || cooChunkSize_ % kAlignElems != 0U) {
        return;
    }

    // hot loop 循环不变量外提为 const 局部变量。成员变量经 AccumulateRow
    // 内联展开后，编译器无法证明 AllocTensor/EnQue/DeQue/FreeTensor 不会修改它们
    // （别名分析失败），导致每轮迭代从内存重新 Load；未取地址的局部变量可全程驻留
    // 寄存器。参考 scalar_story 实证：FIA 成员→局部变量 Scalar 占比降 15%。
    const int64_t myFirstRow = myFirstRow_;
    const int64_t myLastRow = myLastRow_;
    const int64_t nnzVal = nnz_;
    const int32_t idxBase = idxBase_;
    const uint32_t cooChunkSize = cooChunkSize_;
    __gm__ int32_t *cooRowIndBase = cooRowIndGm_;

    const uint32_t actualRows = static_cast<uint32_t>(myLastRow - myFirstRow);
    const uint32_t actualElemCount = actualRows + 1U;
    const uint32_t rowPtrCountAligned = rowPtrBytes_ / static_cast<uint32_t>(sizeof(int32_t));

    auto rpLocal = LoadRowPtr(myFirstRow, actualElemCount, rowPtrCountAligned);
    BuildOffsetTable(actualRows, rowPtrCountAligned);

    auto nnzLocal = nnzBuf_.template Get<int32_t>(rowPtrCountAligned);
    auto startLocal = startBuf_.template Get<int32_t>(rowPtrCountAligned);
    ComputeNnzAndStart(actualRows, idxBase, rowPtrCountAligned, rpLocal, nnzLocal, startLocal);

    // 多行输出 UB 累积后批量 DataCopyPad 写出。
    // 小行（rowNnz <= cooChunkSize）累积到 accumBuf，满 cooChunkSize 或不连续时 flush；
    // 大行（rowNnz > cooChunkSize）flush 后走 ProcessRowChunked 分块路径。
    // 收益：even 分布（每行 1 nnz）从 m 次 4B DataCopyPad 降为 m/cooChunkSize 次批量写出，
    // 消除 DMA 启动开销累积（参考 moe_init_routing MTE3 21us→2us）。
    auto accumBuf = accumBuf_.template Get<int32_t>(cooChunkSize);
    AscendC::GlobalTensor<int32_t> cooGlobal;
    cooGlobal.SetGlobalBuffer(cooRowIndBase);
    AscendC::DataCopyExtParams flushParams{1U, 0, 0, 0, 0};

    uint32_t accumElems = 0U;
    int32_t accumWriteOffset = 0;

    // idxBase 模板分发（两循环体合并为 ProcessRowsImpl 模板）。
    if (idxBase == 0) {
        ProcessRowsImpl<0>(myFirstRow, myLastRow, nnzVal, cooChunkSize,
                           cooRowIndBase, nnzLocal, startLocal,
                           accumBuf, cooGlobal, flushParams,
                           accumElems, accumWriteOffset);
    } else {
        ProcessRowsImpl<1>(myFirstRow, myLastRow, nnzVal, cooChunkSize,
                           cooRowIndBase, nnzLocal, startLocal,
                           accumBuf, cooGlobal, flushParams,
                           accumElems, accumWriteOffset);
    }

    // flush 尾段累积（S→MTE3 + V→MTE3 sync）。
    // accumBuf 由 SetValue(S) 和 Duplicate(V) 写入，DataCopyPad(MTE3) 读取，
    // 需等 S 和 V 都完成。isFinal=true：Process 返回后 accumBuf 不再被写入，
    // 无需 MTE3→S/V 反向同步。
    if (accumElems > 0U) {
        FlushAccumBuf(accumBuf, cooGlobal, flushParams, accumElems, accumWriteOffset, true);
    }
}

// 公共 flush 函数。accumBuf 由 SetValue(S)+Duplicate(V) 写入，DataCopyPad(MTE3)
// 读取，需 S→MTE3 和 V→MTE3 同步。isFinal=false 时额外加 MTE3→S/V 反向同步，防止
// 后续 SetValue/Duplicate 写 accumBuf 与 MTE3 异步读形成 RAW 竞态（同仓 coosort
// arch35 已验证 MTE3→V 反向同步模式）。isFinal=true（Process 末尾）时 accumBuf 不再
// 被写入，无需反向同步。accumElems/accumWriteOffset 重置由调用方负责。
__aicore__ inline void Csr2CooAiv::FlushAccumBuf(
    AscendC::LocalTensor<int32_t> accumBuf,
    AscendC::GlobalTensor<int32_t> cooGlobal,
    AscendC::DataCopyExtParams &flushParams,
    uint32_t accumElems, int32_t accumWriteOffset, bool isFinal)
{
    // S→MTE3 + V→MTE3 sync：accumBuf 由 SetValue(S)+Duplicate(V) 写入。
    auto evtSMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::S_MTE3));
    AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(evtSMte3);
    auto evtVMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_MTE3));
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(evtVMte3);
    AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(evtSMte3);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(evtVMte3);
    flushParams.blockLen = accumElems * static_cast<uint32_t>(sizeof(int32_t));
    AscendC::DataCopyPad(
        cooGlobal[static_cast<uint64_t>(accumWriteOffset)], accumBuf, flushParams);
    if (!isFinal) {
        // MTE3→S/V 反向同步。DataCopyPad(MTE3) 异步读 accumBuf，返回后 MTE3 仍在读
        // accumBuf[0..oldN)。后续 SetValue(S)/Duplicate(V) 写 accumBuf[0..] 需等 MTE3
        // 读完，防止 RAW 竞态。
        auto evtMte3S = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE3_S));
        AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(evtMte3S);
        // MTE3_V 在 Ascend950 上的支持证据链：
        // 1. enum: HardEvent::MTE3_V 存在于 AscendC HardEvent 枚举（kernel_operator.h）。
        // 2. 文档: 核内同步能力概述 AIV 合法同步组合表（表3）PIPE_MTE3→PIPE_V 行列出
        //    SetFlag<MTE3_V>/WaitFlag<MTE3_V>，950 继承该同步组合。
        // 3. 对称性: 反向 V_MTE3 已在本函数上方（S→MTE3 + V→MTE3 sync 段）使用，
        //    MTE3_V 为其逆向同步，同一 pipe 对双向均支持。
        // 4. 同仓实证: csrsort arch35（csrsort_kernel.cpp:166-168）已在同进程
        //    TPipe+VF 混合场景中验证 MTE3_V，语义等价（MTE3 完成后通知 V pipe）。
        auto evtMte3V = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE3_V));
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(evtMte3V);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(evtMte3S);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(evtMte3V);
    }
}

__aicore__ inline void Csr2CooAiv::AccumulateRow(
    int32_t rowNnz, int32_t rowStart, int32_t rowVal,
    uint32_t cooChunkSize, __gm__ int32_t *cooRowIndBase,
    AscendC::LocalTensor<int32_t> accumBuf,
    AscendC::GlobalTensor<int32_t> cooGlobal,
    AscendC::DataCopyExtParams &flushParams,
    uint32_t &accumElems, int32_t &accumWriteOffset)
{
    if (rowNnz > static_cast<int32_t>(cooChunkSize)) {
        // 大行：先 flush 当前累积，再走分块路径
        if (accumElems > 0U) {
            // isFinal=false：后续行会写 accumBuf，需 MTE3→S/V 反向同步防止 RAW。
            FlushAccumBuf(accumBuf, cooGlobal, flushParams, accumElems, accumWriteOffset, false);
            accumElems = 0U;
        }
        int32_t writeOffset = rowStart;
        ProcessRowChunked(rowNnz, rowVal, writeOffset, cooChunkSize, cooRowIndBase);
        // 更新 accumWriteOffset：消除后续非单调输入下连续性判断错误通过的边界风险
        accumWriteOffset = rowStart + rowNnz;
    } else {
        // 小行：累积到 accumBuf
        // 连续性判断：rowStart == accumWriteOffset + accumElems（rowPtr 单调递增保证）
        bool continuous = (accumElems > 0U &&
            rowStart == accumWriteOffset + static_cast<int32_t>(accumElems));
        if (!continuous || accumElems + static_cast<uint32_t>(rowNnz) > cooChunkSize) {
            // 不连续或超出容量：flush 旧段，开始新段
            if (accumElems > 0U) {
                // isFinal=false：本行会写 accumBuf，需 MTE3→S/V 反向同步防止 RAW。
                FlushAccumBuf(accumBuf, cooGlobal, flushParams, accumElems, accumWriteOffset, false);
            }
            accumWriteOffset = rowStart;
            accumElems = 0U;
        }

        // 填充 accumBuf[accumElems..accumElems+rowNnz-1] with rowVal。
        // 对齐分段优化：SetValue(S pipe) 填非对齐头部(≤7) → Duplicate(V pipe) 填对齐主体
        // → SetValue(S pipe) 填非对齐尾部(≤7)。将 S pipe 写次数从 rowNnz 降为 ≤14，
        // 主体段由 V pipe 向量化执行。
        // Duplicate 要求 32B 对齐起始地址（int32 × 8 = 32B），accumElems 非 8 对齐时
        // 先用 SetValue 填至对齐边界，再 Duplicate 主体，尾段再用 SetValue。
        constexpr uint32_t kAlignElems = 8U;  // 32B / sizeof(int32_t)
        uint32_t writeStart = accumElems;
        uint32_t writeEnd = accumElems + static_cast<uint32_t>(rowNnz);
        uint32_t alignedStart = (writeStart + kAlignElems - 1U) & ~(kAlignElems - 1U);
        uint32_t alignedEnd = writeEnd & ~(kAlignElems - 1U);

        // 头部：非对齐段（0~7 个 SetValue）
        for (uint32_t i = writeStart; i < alignedStart && i < writeEnd; i++) {
            accumBuf.SetValue(i, rowVal);
        }
        // 主体：对齐段（Duplicate 向量化填充）
        if (alignedEnd > alignedStart) {
            AscendC::Duplicate(accumBuf[alignedStart], rowVal,
                               static_cast<int32_t>(alignedEnd - alignedStart));
        }
        // 尾部：非对齐段（0~7 个 SetValue）
        // tail 起点钳位 max(alignedEnd, writeStart)，防止 alignedEnd < writeStart
        // 时越界写覆盖前序行 accumBuf 数据（连续小行 + 非对齐累积场景）。
        uint32_t tailStart = (alignedEnd > writeStart) ? alignedEnd : writeStart;
        for (uint32_t i = tailStart; i < writeEnd; i++) {
            accumBuf.SetValue(i, rowVal);
        }
        // SetValue(S pipe) 填头部/尾部、Duplicate(V pipe) 填主体，写 accumBuf 的不同
        // 地址段。正常情况下 head/body/tail 互不重叠，无数据依赖故无需行内 S↔V 同步。
        // 边界场景（alignedEnd < writeStart，即 rowNnz < 8 且 accumElems 非 8 对齐）下
        // head 与 tail 重叠写同值（均为 rowVal），S pipe 内按序执行无风险。
        // S 与 V 不是同 pipe（S=Scalar, V=Vector），行间 flush 后需 MTE3→S/V 反向同步
        // 防止 RAW，此处行内不涉及。accumElems 仅在行间更新，行内无竞争。
        accumElems = writeEnd;
    }
}

__aicore__ inline void Csr2CooAiv::ProcessRowChunked(int32_t rowNnz, int32_t rowVal,
                                                     int32_t &writeOffset,
                                                     uint32_t cooChunkSize,
                                                     __gm__ int32_t *cooRowIndBase)
{
    // GlobalTensor 构造 + SetGlobalBuffer 外提到循环外（循环内用 cooGlobal[writeOffset]
    // 偏移，避免每 chunk 重新构造对象 + 取地址语义触发 Spill）。
    AscendC::GlobalTensor<int32_t> cooGlobal;
    cooGlobal.SetGlobalBuffer(cooRowIndBase);

    // DataCopyExtParams 在循环外构造一次，循环内只更新 blockLen 字段，避免每轮
    // 重新构造整个结构体（Hot Loop 内不构造对象、不取地址）。
    AscendC::DataCopyExtParams copyParams{1U, 0, 0, 0, 0};

    int32_t remaining = rowNnz;

    // while 主尾块分离（三段式）。主块恒为满 chunk，分支预测器对"大多数满块、最后
    // 1 轮尾块"模式预测效果好；尾块单独处理，消除每轮三元判断 + chunkLen==0 判断。
    // 主块：满 chunk
    while (remaining > static_cast<int32_t>(cooChunkSize)) {
        AscendC::LocalTensor<int32_t> cooLocal = cooQueue_.AllocTensor<int32_t>();
        AscendC::Duplicate(cooLocal, rowVal, static_cast<int32_t>(cooChunkSize));
        cooQueue_.EnQue(cooLocal);
        AscendC::LocalTensor<int32_t> outLocal = cooQueue_.DeQue<int32_t>();

        copyParams.blockLen = cooChunkSize * static_cast<uint32_t>(sizeof(int32_t));
        AscendC::DataCopyPad(cooGlobal[static_cast<uint64_t>(writeOffset)], outLocal, copyParams);
        cooQueue_.FreeTensor(outLocal);

        remaining -= static_cast<int32_t>(cooChunkSize);
        writeOffset += static_cast<int32_t>(cooChunkSize);
    }

    // 尾块（remaining <= cooChunkSize，含 remaining==0 时不进入）
    if (remaining > 0) {
        AscendC::LocalTensor<int32_t> cooLocal = cooQueue_.AllocTensor<int32_t>();
        AscendC::Duplicate(cooLocal, rowVal, remaining);
        cooQueue_.EnQue(cooLocal);
        AscendC::LocalTensor<int32_t> outLocal = cooQueue_.DeQue<int32_t>();

        copyParams.blockLen = static_cast<uint32_t>(remaining) * static_cast<uint32_t>(sizeof(int32_t));
        AscendC::DataCopyPad(cooGlobal[static_cast<uint64_t>(writeOffset)], outLocal, copyParams);
        cooQueue_.FreeTensor(outLocal);

        writeOffset += remaining;
    }
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// Kernel entry point
// ---------------------------------------------------------------------------

extern "C" __global__ __aicore__ void csr2coo_kernel(
    GM_ADDR gmRowPtr, GM_ADDR gmCooRowInd,
    csr2coo::Csr2CooTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

    if (tiling.idxBase != 0 && tiling.idxBase != 1) {
        return;
    }

    // 混合路径分发。Host 按矩阵特征(行数/平均nnz)选择 useSimt。
    if (tiling.useSimt != 0U) {
        __gm__ int32_t *rowPtrGm = reinterpret_cast<__gm__ int32_t *>(gmRowPtr);
        __gm__ int32_t *cooRowIndGm = reinterpret_cast<__gm__ int32_t *>(gmCooRowInd);
        int64_t firstRow64 = static_cast<int64_t>(AscendC::GetBlockIdx()) *
                             static_cast<int64_t>(tiling.simtRowsPerBlock);
        int64_t lastRow64 = firstRow64 + static_cast<int64_t>(tiling.simtRowsPerBlock);
        if (lastRow64 > tiling.m) {
            lastRow64 = tiling.m;
        }
        if (firstRow64 >= lastRow64) {
            return;
        }
        int32_t myFirstRow = static_cast<int32_t>(firstRow64);
        int32_t myLastRow = static_cast<int32_t>(lastRow64);
        if (tiling.idxBase == 0) {
            asc_vf_call<Csr2CooSimtCompute<0>>(
                dim3{csr2coo::kCsr2CooSimtThreads, 1, 1},
                myFirstRow, myLastRow, tiling.nnz, rowPtrGm, cooRowIndGm);
        } else {
            asc_vf_call<Csr2CooSimtCompute<1>>(
                dim3{csr2coo::kCsr2CooSimtThreads, 1, 1},
                myFirstRow, myLastRow, tiling.nnz, rowPtrGm, cooRowIndGm);
        }
    } else {
        // SIMD 路径: 大行场景(ultrarow, 单行nnz极大)。保留
        // Duplicate+DataCopyPad, 一次填 cooChunkSize 元素高效写出。
        AscendC::TPipe pipe;
        Csr2CooAiv op;
        op.Init(gmRowPtr, gmCooRowInd, tiling, &pipe);
        op.Process();
    }
}

// ---------------------------------------------------------------------------
// Kernel launcher (called from Host)
// ---------------------------------------------------------------------------

extern "C" void csr2coo_kernel_do(
    GM_ADDR gmRowPtr, GM_ADDR gmCooRowInd,
    const csr2coo::Csr2CooTilingData &tiling, uint32_t numBlocks, void *stream)
{
    csr2coo_kernel<<<numBlocks, nullptr, stream>>>(gmRowPtr, gmCooRowInd, tiling);
}
