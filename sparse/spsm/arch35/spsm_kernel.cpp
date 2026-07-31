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

// SPSM v2 kernel for Ascend 950 (dav-3510).
//
// 编程模型:
//   - SpsmSolveAIV: MemBase vector API (TPipe + TQue + LocalTensor + DataCopyPad), 仅走 ROW 路径
//   - SpsmTransposeAIV: vector SIMD (TPipe + TBuf + LocalTensor + DataCopyPad), COL<->ROW 转置
//
// Kernel 组成:
//   1) spsm_transpose_kernel — vector transpose (COL<->ROW), Solve 前后做数据转置
//   2) SpsmSolveAIV — 多 pass 逐 level 求解 (多核, blockDim=numCores), 仅 ROW-order 搬运
// Analysis 在 host CPU 计算 level scheduling, 不再有 Analysis kernel。
// TilingData 由 host by-value 传入, 不从 GM 读取。
// COL-order 散列搬运改用 vector transpose kernel (DataCopyPad 跨步搬运),
//   Solve kernel 内 LoadDnMatSlice/StoreDnMatSlice 仅保留 ROW 连续搬运路径。
//
// 数学公式:
//   NON_UNIT: X[i,:] = (α·B[i,:] − Σ A[i,j]·X[j,:]) / A[i,i]
//   UNIT:     X[i,:] =  α·B[i,:] − Σ A[i,j]·X[j,:]

#include <cstdint>
#include "kernel_operator.h"
#include "spsm.h"
#include "spsm_kernel.h"

// 注: log/log.h 不在此 include。host dispatcher 函数 (spsm_solve_kernel_do)
// 未使用 dlog 宏; 且 log/log.h 会重定义 unlikely/likely 宏, 与
// kernel_utils_macros.h 冲突产生 -Wmacro-redefined.

using namespace AscendC;

namespace {

// kBufferNum=2: 三个 TQue 统一双缓冲。
// colIndQue_/valsQue_ 每行只加载一次, 双缓冲无直接性能收益但不影响正确性, 保持统一。
// inQue_ reduction 循环采用预取模式: 加载 dep[k+1] (MTE2) 与计算 dep[k] (Muls+Sub, V) 重叠。
// UB 占用: colIndQue_/valsQue_ 各 2 份 maxRowLenAlign, inQue_ 2 份 kChunkAlign,
// accBuf_/tmpBuf_ 各 1 份 kChunkAlign, 合计 kChunk 部分 4 份。
constexpr uint32_t kBufferNum = 2;
// UB 最小槽位字节数 (diagBuf 固定占用 1 个 float, 按 32B 槽位对齐)
constexpr uint32_t kUbSlotBytes = 32;

// ============================================================================
// SpsmSolveAIV: 多 pass 逐 level 求解 (多核)
//
// 单 kernel 多 pass, 按 level 0→L-1 顺序:
//   - level 内行跨核均分
//   - 每行 ProcessRow: acc = α·B − Σ A·X, NON_UNIT 时 acc /= A[i,i]
//   - level 间 SyncAll
// ============================================================================
class SpsmSolveAIV {
public:
    __aicore__ inline SpsmSolveAIV() {}

    // TPipe 由 kernel 入口栈上创建并传入: 避免类成员持有 TPipe 导致的栈空间膨胀
    // 与 GetTPipePtr() 失效问题. 生命周期覆盖整个 op.Process()。
    // tiling 由 host by-value 传入, 不从 GM 读取。
    __aicore__ inline void Init(GM_ADDR csrRowOffsets, GM_ADDR csrColInd,
                                GM_ADDR csrValues, GM_ADDR matB, GM_ADDR matC,
                                GM_ADDR workspaceGM, const SpsmTilingData& tiling,
                                TPipe *pipe)
    {
        pipePtr_ = pipe;
        td_ = tiling;
        m_ = td_.m;
        n_ = td_.n;
        maxRowLen_ = td_.maxRowLen;
        if (maxRowLen_ < 1) { maxRowLen_ = 1; }
        kChunkSize_ = td_.kChunkSize;
        if (kChunkSize_ < 1) { kChunkSize_ = 1; }
        isNonUnit_ = (td_.diagType == SPSM_DIAG_NON_UNIT);
        blockIdx_ = static_cast<int32_t>(GetBlockIdx());
        numCores_ = static_cast<int32_t>(GetBlockNum());
        if (numCores_ < 1) { numCores_ = 1; }

        __gm__ uint8_t *wsBase = reinterpret_cast<__gm__ uint8_t *>(workspaceGM);

        // CSR 源选择
        if (td_.needTranspose != 0) {
            rowOffGm_ = reinterpret_cast<__gm__ int32_t *>(wsBase + td_.transRowOffOff);
            colIndGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(wsBase + td_.transColIndOff));
            valsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(wsBase + td_.transValOff));
        } else {
            rowOffGm_ = reinterpret_cast<__gm__ int32_t *>(csrRowOffsets);
            colIndGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(csrColInd));
            valsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(csrValues));
        }

        bGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(matB));
        cGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(matC));

        // COL-order 转置缓冲区: orderB=COL || orderC=COL 时由 SIMT
        // transpose kernel 写入/读取, Solve kernel 通过 orderB/orderC 选择数据源。
        // 仅 orderB==COL || orderC==COL 时绑定 (workspace 条件分配, ROW&&ROW 路径不分配)。
        // 未绑定时 (orderB==ROW && orderC==ROW) Solve kernel 不访问 denseBufGm_, 安全。
        if (td_.orderB == SPSM_ORDER_COL || td_.orderC == SPSM_ORDER_COL) {
            denseBufGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(wsBase + td_.denseBufOff));
        }

        levelRowPtrGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(wsBase + td_.levelRowPtrOff));
        levelRowIdxGm_ = reinterpret_cast<__gm__ int32_t *>(wsBase + td_.levelRowIdxOff);
        if (isNonUnit_) {
            diagValGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(wsBase + td_.diagValOff));
        }

        // UB buffer 初始化
        int32_t maxRowLenAlign = (maxRowLen_ + 7) / 8 * 8;
        int32_t kChunkAlign = (kChunkSize_ + 7) / 8 * 8;

        pipePtr_->InitBuffer(colIndQue_, kBufferNum, maxRowLenAlign * sizeof(int32_t));
        // UNIT 也需要 vals (A[i,j] 值), 用于 Muls; NON_UNIT 同样需要, 无条件初始化
        pipePtr_->InitBuffer(valsQue_, kBufferNum, maxRowLenAlign * sizeof(float));
        pipePtr_->InitBuffer(inQue_, kBufferNum, kChunkAlign * sizeof(float));
        pipePtr_->InitBuffer(accBuf_, kChunkAlign * sizeof(float));
        pipePtr_->InitBuffer(tmpBuf_, kChunkAlign * sizeof(float));
        pipePtr_->InitBuffer(diagBuf_, kUbSlotBytes);

        // levelRowPtr 预加载到 UB (避免 Process 中裸 __gm__ 标量读取)
        InitLevelRowPtrBuf();
    }

    // levelRowPtr 预加载到 UB (避免 Process 中裸 __gm__ 标量读取)
    __aicore__ inline void InitLevelRowPtrBuf()
    {
        int32_t levelRowPtrCount = td_.L + 1;
        int32_t levelRowPtrBytes = (levelRowPtrCount * static_cast<int32_t>(sizeof(int32_t)) + 31) / 32 * 32;
        if (levelRowPtrBytes < static_cast<int32_t>(kUbSlotBytes)) { levelRowPtrBytes = kUbSlotBytes; }
        pipePtr_->InitBuffer(levelRowPtrBuf_, static_cast<uint32_t>(levelRowPtrBytes));
        if (td_.L > 0) {
            LocalTensor<int32_t> levelRowPtrLocal = levelRowPtrBuf_.Get<int32_t>();
            DataCopyExtParams cpLrp{1, static_cast<uint32_t>(levelRowPtrCount * sizeof(int32_t)), 0, 0, 0};
            DataCopyPadExtParams<int32_t> padLrp{false, 0, 0, 0};
            DataCopyPad(levelRowPtrLocal, levelRowPtrGm_, cpLrp, padLrp);
            event_t evtLRP = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
            SetFlag<HardEvent::MTE2_S>(evtLRP);
            WaitFlag<HardEvent::MTE2_S>(evtLRP);
            GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_S>(evtLRP);
        }
    }

    __aicore__ inline void Process()
    {
        bool isEmpty = (m_ <= 0 || n_ <= 0);

        // COL→ROW / ROW→COL 转置已外移至 SIMT transpose kernel
        // (host 在 Solve 前后 launch spsm_transpose_kernel), 本 kernel 仅做 Solve。

        // Solve pass 循环
        if (!isEmpty) {
            LocalTensor<int32_t> levelRowPtrLocal = levelRowPtrBuf_.Get<int32_t>();
            for (int32_t lv = 0; lv < td_.L; lv++) {
                int32_t lvStart = levelRowPtrLocal.GetValue(lv);
                int32_t lvEnd = levelRowPtrLocal.GetValue(lv + 1);
                int32_t lvRows = lvEnd - lvStart;

                if (lvRows > 0) {
                    int32_t rowsPerCore = (lvRows + numCores_ - 1) / numCores_;
                    int32_t myStart = lvStart + blockIdx_ * rowsPerCore;
                    int32_t myEnd = myStart + rowsPerCore;
                    if (myEnd > lvEnd) { myEnd = lvEnd; }

                    // levelRowIdx 保持 GM 标量读取 (m 较大时预加载 UB 占用过多)
                    for (int32_t r = myStart; r < myEnd; r++) {
                        int32_t i = levelRowIdxGm_[r];
                        ProcessRow(i);
                    }
                }

                // 跨核 MTE3→MTE2 同步 (level 间数据依赖保证):
                //
                // 数据流: 本 level 各核通过 StoreDnMatSlice 的 DataCopyPad (UB→GM, MTE3)
                // 写回 X[i,:]; 下一 level 的核通过 LoadDnMatSlice 的 DataCopyPad (GM→UB,
                // MTE2) 读取 X[j,:] 作为依赖行。需保证跨核 GM 写→读可见性。
                //
                // 同步序列与语义 (dav-3510):
                //   1) PipeBarrier<PIPE_MTE3> — 排空本核 MTE3 流水线, 确保此前所有
                //      UB→GM 写已提交到 GM 一致性域。这是 intra-core 屏障, 不阻塞其他核。
                //   2) SyncAll — 全核栅栏, 等待所有核到达。此时所有核的 MTE3 写均已
                //      提交 (因步骤 1 在前), 故下一 level 任何核的 MTE2 读都能看到
                //      最新 X[j,:]。
                //
                // 顺序约束: PipeBarrier 必须在 SyncAll 之前。若颠倒, 核 A 可能在 MTE3
                // 写未提交时就到达 SyncAll, 核 B 随后读到过期数据 → 数据竞争。
                //
                // 无死锁风险: PipeBarrier 是 intra-core 非阻塞, SyncAll 是唯一的跨核
                // 等待点; 所有核 (含空核, 见下方 else 分支) 均执行 SyncAll, 满足
                // dav-3510 SyncAll 全核握手要求。
                PipeBarrier<PIPE_MTE3>();

                // level 间核间同步
                SyncAll();
            }
        } else {
            // 空核仍参与 SyncAll (dav-3510 握手要求)
            for (int32_t lv = 0; lv < td_.L; lv++) {
                SyncAll();
            }
        }
    }

private:
    // 加载行 CSR 数据 (colInd + values) 到 UB. len<=0 时直接返回 (不 Alloc).
    __aicore__ inline void LoadRowCsrData(int32_t i, int32_t len,
                                          LocalTensor<int32_t>& colIndLocal,
                                          LocalTensor<float>& valsLocal)
    {
        if (len <= 0) {
            return;
        }
        int32_t s = rowOffGm_[i];

        // 加载 colInd[s..e-1] 和 values[s..e-1] (跨 kChunk 复用)
        // len 上界已由 host 侧 ValidateMaxRowLenUbCapacity 保证 (远小于 2^30),
        // static_cast<uint32_t>(len * sizeof(...)) 不会截断.
        colIndLocal = colIndQue_.template AllocTensor<int32_t>();
        DataCopyExtParams cpCol{1, static_cast<uint32_t>(len * sizeof(int32_t)), 0, 0, 0};
        DataCopyPadExtParams<int32_t> padCol{false, 0, 0, 0};
        DataCopyPad(colIndLocal, colIndGm_[s], cpCol, padCol);
        colIndQue_.EnQue(colIndLocal);
        colIndLocal = colIndQue_.template DeQue<int32_t>();

        valsLocal = valsQue_.template AllocTensor<float>();
        DataCopyExtParams cpVal{1, static_cast<uint32_t>(len * sizeof(float)), 0, 0, 0};
        DataCopyPadExtParams<float> padVal{false, 0, 0, 0.0f};
        DataCopyPad(valsLocal, valsGm_[s], cpVal, padVal);
        valsQue_.EnQue(valsLocal);
        valsLocal = valsQue_.template DeQue<float>();

        // MTE2→S 同步: DeQue 保证 MTE2→V, 但后续 GetValue 是 S 操作.
        event_t evt = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
        SetFlag<HardEvent::MTE2_S>(evt);
        WaitFlag<HardEvent::MTE2_S>(evt);
        GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_S>(evt);
    }

    // NON_UNIT 时加载 diagVal[i] 并计算倒数, UNIT 返回 1.0f.
    __aicore__ inline float PrepareInvDiag(int32_t i)
    {
        float invDiag = 1.0f;
        if (isNonUnit_) {
            LocalTensor<float> diagLocal = diagBuf_.Get<float>();
            DataCopyExtParams cpDiag{1, static_cast<uint32_t>(sizeof(float)), 0, 0, 0};
            DataCopyPadExtParams<float> padDiag{false, 0, 0, 0.0f};
            DataCopyPad(diagLocal, diagValGm_[i], cpDiag, padDiag);
            event_t evt = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
            SetFlag<HardEvent::MTE2_S>(evt);
            WaitFlag<HardEvent::MTE2_S>(evt);
            GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_S>(evt);
            float dv = diagLocal.GetValue(0);
            if (dv == 0.0f) {
                invDiag = 0.0f;  // 奇异矩阵防御: Analysis 已检测, 此处兜底避免除零产生 Inf 污染
            } else {
                invDiag = 1.0f / dv;
            }
        }
        return invDiag;
    }

    // ---- reduction: acc -= Σ A[i,j] * X[j, kStart:kEnd] ----
    // 双缓冲预取: 加载 dep[k+1] 的 X[j,:] (MTE2) 与计算 dep[k] 的 Muls+Sub (V) 重叠。
    // TQue num=2: AllocTensor 自动交替使用两份物理 buffer, EnQue 提交 MTE2 异步搬运,
    // DeQue 等待 MTE2 完成。预取模式使 MTE2 与 V 流水线并行。
    // indexBase 归一化: kernel 读取 colInd 时统一减去 td_.indexBase
    // (opA=T: host Csr2Csc 已归一化, td_.indexBase=0; opA=N: host 检测后设置)
    // 注: 采用 Muls+Sub 紧邻序列, dav-3510 硬件会自动融合为 FMA (单次舍入).
    // 经实测, Kahan 补偿 / Axpy / 分离累加 / 逆序遍历 均打破融合或改变舍入模式,
    // 导致精度下降. 此为该平台最优累加序列。Muls 与 Sub 之间不插屏障 (V 流水
    // 同 pipe 依赖指令本身有序), 以保证紧邻触发 FMA 融合。
    __aicore__ inline void ReduceDeps(int32_t i, int32_t len, int32_t kStart, int32_t kLen,
                                      LocalTensor<int32_t>& colIndLocal,
                                      LocalTensor<float>& valsLocal,
                                      LocalTensor<float>& accBuf,
                                      LocalTensor<float>& tmpBuf,
                                      GlobalTensor<float>& cSrc,
                                      int32_t cLd)
    {
        bool hasDep = false;
        float aValCurr = 0.0f;

        for (int32_t k = 0; k < len; k++) {
            int32_t j = colIndLocal.GetValue(k) - td_.indexBase;
            if (j < 0 || j >= m_) { continue; }
            if (j == i) { continue; }
            bool isDep = (td_.effectiveFillMode == SPSM_FILL_LOWER) ? (j < i) : (j > i);
            if (!isDep) { continue; }

            float aVal = valsLocal.GetValue(k);

            if (!hasDep) {
                // 首个依赖: 仅加载 (MTE2), 不计算
                LocalTensor<float> xDepLocal = inQue_.template AllocTensor<float>();
                LoadDnMatSlice(xDepLocal, cSrc, j, kStart, kLen, cLd);
                inQue_.EnQue(xDepLocal);
                aValCurr = aVal;
                hasDep = true;
                continue;
            }

            // 预取当前 dep (MTE2, 与上一轮 V 计算重叠)
            LocalTensor<float> xDepNext = inQue_.template AllocTensor<float>();
            LoadDnMatSlice(xDepNext, cSrc, j, kStart, kLen, cLd);
            inQue_.EnQue(xDepNext);

            // 计算上一个 dep (V, 与上述 MTE2 重叠)
            LocalTensor<float> xDepCurr = inQue_.template DeQue<float>();
            Muls(tmpBuf, xDepCurr, aValCurr, kLen);

            Sub(accBuf, accBuf, tmpBuf, kLen);
            PipeBarrier<PIPE_V>();

            inQue_.FreeTensor(xDepCurr);

            aValCurr = aVal;
        }

        // 尾部: 计算最后一个已预取的 dep (无下一 dep 可重叠)
        if (hasDep) {
            LocalTensor<float> xDepCurr = inQue_.template DeQue<float>();
            Muls(tmpBuf, xDepCurr, aValCurr, kLen);

            Sub(accBuf, accBuf, tmpBuf, kLen);
            PipeBarrier<PIPE_V>();

            inQue_.FreeTensor(xDepCurr);
        }
    }

    // n 维 kChunk 列块循环 (init + reduction + divide + write back).
    // bOrder/cOrder 恒为 ROW (COL 转置已外移至 SIMT transpose kernel),
    // LoadDnMatSlice/StoreDnMatSlice 仅走 ROW 连续搬运路径。
    __aicore__ inline void SolveKChunkLoop(int32_t i, int32_t len,
                                           LocalTensor<int32_t>& colIndLocal,
                                           LocalTensor<float>& valsLocal,
                                           GlobalTensor<float>& bSrc,
                                           GlobalTensor<float>& cSrc,
                                           int32_t bLd, int32_t cLd,
                                           float invDiag)
    {
        LocalTensor<float> accBuf = accBuf_.Get<float>();
        LocalTensor<float> tmpBuf = tmpBuf_.Get<float>();

        // n 维 kChunk 列块循环
        for (int32_t kStart = 0; kStart < n_; kStart += kChunkSize_) {
            int32_t kEnd = kStart + kChunkSize_;
            if (kEnd > n_) { kEnd = n_; }
            int32_t kLen = kEnd - kStart;

            // ---- init: acc = alpha * B[i, kStart:kEnd] ----
            LocalTensor<float> bLocal = inQue_.template AllocTensor<float>();
            LoadDnMatSlice(bLocal, bSrc, i, kStart, kLen, bLd);
            inQue_.EnQue(bLocal);
            bLocal = inQue_.template DeQue<float>();

            Muls(accBuf, bLocal, td_.alpha_host, kLen);
            PipeBarrier<PIPE_V>();

            inQue_.FreeTensor(bLocal);

            // ---- reduction: acc -= Σ A[i,j] * X[j, kStart:kEnd] ----
            ReduceDeps(i, len, kStart, kLen, colIndLocal, valsLocal, accBuf, tmpBuf, cSrc, cLd);

            // ---- divide (NON_UNIT): acc *= invDiag ----
            if (isNonUnit_) {
                Muls(accBuf, accBuf, invDiag, kLen);
            }

            // V→MTE3 同步: 确保 Muls/Sub (V 操作) 完成后, StoreDnMatSlice 的
            // DataCopyPad (MTE3 操作, UB→GM) 才读 accBuf. accBuf 由 V 写、由 MTE3 读,
            // 跨流水依赖需排空所有流水线 (PIPE_ALL) 保证 MTE3 能看到 V 的结果.
            PipeBarrier<PIPE_ALL>();

            // ---- write back: X[i, kStart:kEnd] = acc ----
            StoreDnMatSlice(cSrc, accBuf, i, kStart, kLen, cLd);
            // accBuf 是 TBuf (无 EnQue/DeQue 自动同步): StoreDnMatSlice (MTE3) 读 accBuf,
            // 下一轮迭代首条 Muls(accBuf, ...) (V) 写同一 buffer, 需 MTE3→V 跨流水同步.
            PipeBarrier<PIPE_ALL>();
        }
    }

    // ProcessRow 仅做协调, 逻辑等价于原实现.
    __aicore__ inline void ProcessRow(int32_t i)
    {
        // rowOff 保持 GM 标量读取 (m 较大时预加载 UB 占用过多)
        int32_t s = rowOffGm_[i];
        int32_t e = rowOffGm_[i + 1];
        int32_t len = e - s;
        if (len < 0) { len = 0; }

        // 数据源选择基于 orderB/orderC。
        // orderB=COL 时 B 已被 SIMT transpose kernel 转成 denseBuf(ROW); orderB=ROW 直接用 bGm(ROW)。
        // orderC=COL 时 C 由 SIMT transpose kernel 从 denseBuf(ROW) 转出; Solve 写 denseBuf(ROW)。
        // bOrder/cOrder 恒为 ROW, bLd/cLd 随数据源切换 (denseBuf 行主序 ld=n, 原 GM 用 ldb/ldc)。
        const bool bFromDense = (td_.orderB == SPSM_ORDER_COL);
        const bool cFromDense = (td_.orderC == SPSM_ORDER_COL);
        GlobalTensor<float>& bSrc = bFromDense ? denseBufGm_ : bGm_;
        GlobalTensor<float>& cSrc = cFromDense ? denseBufGm_ : cGm_;
        int32_t bLd = bFromDense ? n_ : td_.ldb;
        int32_t cLd = cFromDense ? n_ : td_.ldc;

        LocalTensor<int32_t> colIndLocal;
        LocalTensor<float> valsLocal;
        LoadRowCsrData(i, len, colIndLocal, valsLocal);

        float invDiag = PrepareInvDiag(i);

        SolveKChunkLoop(i, len, colIndLocal, valsLocal, bSrc, cSrc, bLd, cLd, invDiag);

        if (len > 0) {
            colIndQue_.FreeTensor(colIndLocal);
            valsQue_.FreeTensor(valsLocal);
        }
    }

    // 加载稠密矩阵切片 B[i, kStart:kEnd] 到 UB (仅 ROW-order 连续搬运)
    // COL-order 散列搬运已外移至 SIMT transpose kernel,
    // Solve kernel 仅处理 ROW-order (bGm/cGm 或 denseBuf 均 ROW-order)。
    __aicore__ inline void LoadDnMatSlice(LocalTensor<float>& dst, GlobalTensor<float>& srcGm,
                                          int32_t i, int32_t kStart, int32_t kLen,
                                          int32_t ld)
    {
        DataCopyExtParams cp{1, static_cast<uint32_t>(kLen * sizeof(float)), 0, 0, 0};
        DataCopyPadExtParams<float> pad{false, 0, 0, 0.0f};
        DataCopyPad(dst, srcGm[static_cast<uint64_t>(i) * ld + kStart], cp, pad);
    }

    // 写回稠密矩阵切片 X[i, kStart:kEnd] 到 GM (仅 ROW-order 连续搬运)
    __aicore__ inline void StoreDnMatSlice(GlobalTensor<float>& dstGm, LocalTensor<float>& src,
                                           int32_t i, int32_t kStart, int32_t kLen,
                                           int32_t ld)
    {
        DataCopyExtParams cp{1, static_cast<uint32_t>(kLen * sizeof(float)), 0, 0, 0};
        DataCopyPad(dstGm[static_cast<uint64_t>(i) * ld + kStart], src, cp);
    }

private:
    TPipe *pipePtr_{nullptr};
    SpsmTilingData td_{};
    int32_t m_{0};
    int32_t n_{0};
    int32_t maxRowLen_{0};
    int32_t kChunkSize_{0};
    int32_t blockIdx_{0};
    int32_t numCores_{0};
    bool isNonUnit_{false};

    __gm__ int32_t *rowOffGm_{nullptr};
    GlobalTensor<int32_t> colIndGm_;
    GlobalTensor<float> valsGm_;
    GlobalTensor<float> bGm_;
    GlobalTensor<float> cGm_;
    GlobalTensor<float> denseBufGm_;  // COL-order 转置缓冲区 (SIMT transpose kernel 写入/读取)
    GlobalTensor<int32_t> levelRowPtrGm_;
    __gm__ int32_t *levelRowIdxGm_{nullptr};
    GlobalTensor<float> diagValGm_;

    TQue<TPosition::VECIN, kBufferNum> colIndQue_;
    TQue<TPosition::VECIN, kBufferNum> valsQue_;
    TQue<TPosition::VECIN, kBufferNum> inQue_;
    TBuf<TPosition::VECCALC> accBuf_;
    TBuf<TPosition::VECCALC> tmpBuf_;
    TBuf<TPosition::VECCALC> diagBuf_;
    TBuf<TPosition::VECCALC> levelRowPtrBuf_;
};

// ============================================================================
// SpsmTransposeAIV: COL<->ROW 转置 (vector SIMD, DataCopyPad)
//
// 替代原 SIMT 标量 GM 读写方式. 使用 TPipe + TBuf + LocalTensor
// + DataCopyPad, 多核按行切分, 每核处理若干行。
//
// direction=0: COL->ROW  src[k*ld+i] -> dst[i*n+k]  (B COL → denseBuf ROW)
//   源列主序 (ld 为跨步), 目标行主序 (n 为跨步):
//   对每行 i, 跨步读 src[k*ld+i] (步长 ld) 到 UB, 连续写 dst[i*n+k].
// direction=1: ROW->COL  src[i*n+k] -> dst[k*ld+i]  (denseBuf ROW → C COL)
//   源行主序 (n 为跨步), 目标列主序 (ld 为跨步):
//   对每行 i, 连续读 src[i*n+k] 到 UB, 跨步写 dst[k*ld+i] (步长 ld).
//
// DataCopyPad 跨步搬运说明 (dav-3510):
//   - GM 侧 stride (srcStride/dstStride) 单位为字节, 表示相邻数据块间隔 (gap),
//     pitch = stride + blockLen. GM 侧无 32B 对齐要求, 任意 ld 均可。
//   - UB 侧 stride 单位为 32 字节; Normal 模式下 1-float 数据块在 UB 中按 32B 槽位
//     排布 (pitch=32B). 因此跨步读后 UB 数据呈 32B 间隔散列, 连续写时以 srcStride=0
//     (UB pitch=32B) 从散列槽位 gather 回连续 GM。
//   - 分段处理 (kTransSegMax): 同时约束 blockCount (<=4095) 与 UB 占用 (32B/元素).
// ============================================================================
class SpsmTransposeAIV {
public:
    __aicore__ inline SpsmTransposeAIV() {}

    __aicore__ inline void Init(GM_ADDR srcGM, GM_ADDR dstGM,
                                int32_t m, int32_t n, int32_t ld, int32_t direction, TPipe *pipe)
    {
        pipePtr_ = pipe;
        srcGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(srcGM));
        dstGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(dstGM));
        m_ = m;
        n_ = n;
        ld_ = ld;
        direction_ = direction;
        blockIdx_ = static_cast<int32_t>(GetBlockIdx());
        numCores_ = static_cast<int32_t>(GetBlockNum());
        if (numCores_ < 1) { numCores_ = 1; }

        // UB 行缓冲: 跨步 DataCopyPad 以 1-float 块搬运, Normal 模式下每元素占 32B 槽位,
        // 故 buffer 大小 = 32B * kTransSegMax。
        pipePtr_->InitBuffer(rowBuf_, kTransSegBytes);
    }

    __aicore__ inline void Process()
    {
        if (m_ <= 0 || n_ <= 0) { return; }

        // 多核按行均分 (空核直接返回, AIV_ONLY 无 cube/vector 握手要求)
        int32_t rowsPerCore = (m_ + numCores_ - 1) / numCores_;
        int32_t myStart = blockIdx_ * rowsPerCore;
        int32_t myEnd = myStart + rowsPerCore;
        if (myEnd > m_) { myEnd = m_; }
        if (myStart >= myEnd) { return; }

        for (int32_t i = myStart; i < myEnd; i++) {
            ProcessRow(i);
        }
    }

private:
    __aicore__ inline void ProcessRow(int32_t i)
    {
        // 分段处理 n 个元素, 每段 kTransSegMax 个 (受 blockCount / UB 限制)
        for (int32_t kStart = 0; kStart < n_; kStart += kTransSegMax) {
            int32_t kEnd = kStart + kTransSegMax;
            if (kEnd > n_) { kEnd = n_; }
            int32_t segLen = kEnd - kStart;
            if (direction_ == 0) {
                TransposeColToRow(i, kStart, segLen);
            } else {
                TransposeRowToCol(i, kStart, segLen);
            }
        }
    }

    // direction=0 (COL→ROW): 跨步读 src[k*ld+i] -> 连续写 dst[i*n+k]
    __aicore__ inline void TransposeColToRow(int32_t i, int32_t kStart, int32_t segLen)
    {
        LocalTensor<float> rowBuf = rowBuf_.Get<float>();
        // 跨步读: blockCount=segLen, blockLen=4B(1 float), srcStride=(ld-1)*4B (GM gap),
        // dstStride=0 (UB 32B 槽位散列). 读 src[k*ld+i], k=kStart..kStart+segLen-1.
        // dav-3510 下 DataCopyExtParams.srcStride/dstStride 为 int64_t, 与此处 int64_t
        // 计算匹配, 大 ld (stride>4GB) 不会截断。
        const int64_t srcStrideBytes = static_cast<int64_t>(ld_ - 1) * static_cast<int64_t>(sizeof(float));
        DataCopyExtParams cpIn{static_cast<uint16_t>(segLen), static_cast<uint32_t>(sizeof(float)),
                               srcStrideBytes, 0, 0};
        DataCopyPadExtParams<float> padIn{false, 0, 0, 0.0f};
        DataCopyPad(rowBuf, srcGm_[static_cast<uint64_t>(i) + static_cast<uint64_t>(kStart) * static_cast<uint64_t>(ld_)], cpIn, padIn);
        // MTE2→MTE3 同步: 确保 GM→UB 加载完成后再 UB→GM 读取
        PipeBarrier<PIPE_MTE2>();
        // 连续写: blockCount=segLen, blockLen=4B, srcStride=0 (UB 32B 槽位 gather),
        // dstStride=0 (GM 连续). 写 dst[i*n+kStart+k].
        DataCopyExtParams cpOut{static_cast<uint16_t>(segLen), static_cast<uint32_t>(sizeof(float)),
                                0, 0, 0};
        DataCopyPad(dstGm_[static_cast<uint64_t>(i) * static_cast<uint64_t>(n_) + kStart], rowBuf, cpOut);
        // MTE3→MTE2 同步: 确保 UB→GM 写出完成后再复用 rowBuf
        PipeBarrier<PIPE_MTE3>();
    }

    // direction=1 (ROW→COL): 连续读 src[i*n+k] -> 跨步写 dst[k*ld+i]
    __aicore__ inline void TransposeRowToCol(int32_t i, int32_t kStart, int32_t segLen)
    {
        LocalTensor<float> rowBuf = rowBuf_.Get<float>();
        // 连续读: blockCount=segLen, blockLen=4B, srcStride=0 (GM 连续),
        // dstStride=0 (UB 32B 槽位散列). 读 src[i*n+kStart+k].
        DataCopyExtParams cpIn{static_cast<uint16_t>(segLen), static_cast<uint32_t>(sizeof(float)),
                               0, 0, 0};
        DataCopyPadExtParams<float> padIn{false, 0, 0, 0.0f};
        DataCopyPad(rowBuf, srcGm_[static_cast<uint64_t>(i) * static_cast<uint64_t>(n_) + kStart], cpIn, padIn);
        // MTE2→MTE3 同步
        PipeBarrier<PIPE_MTE2>();
        // 跨步写: blockCount=segLen, blockLen=4B, srcStride=0 (UB 32B 槽位 gather),
        // dstStride=(ld-1)*4B (GM gap). 写 dst[(kStart+k)*ld+i].
        const int64_t dstStrideBytes = static_cast<int64_t>(ld_ - 1) * static_cast<int64_t>(sizeof(float));
        DataCopyExtParams cpOut{static_cast<uint16_t>(segLen), static_cast<uint32_t>(sizeof(float)),
                                0, dstStrideBytes, 0};
        DataCopyPad(dstGm_[static_cast<uint64_t>(i) + static_cast<uint64_t>(kStart) * static_cast<uint64_t>(ld_)],
                    rowBuf, cpOut);
        // MTE3→MTE2 同步
        PipeBarrier<PIPE_MTE3>();
    }

private:
    // 分段大小: 约束 blockCount<=4095 且 UB 占用 (32B/元素) 可控
    static constexpr int32_t kTransSegMax = 2048;
    static constexpr uint32_t kTransSegBytes = static_cast<uint32_t>(kTransSegMax) * 32u;

    TPipe *pipePtr_{nullptr};
    GlobalTensor<float> srcGm_;
    GlobalTensor<float> dstGm_;
    int32_t m_{0};
    int32_t n_{0};
    int32_t ld_{0};
    int32_t direction_{0};
    int32_t blockIdx_{0};
    int32_t numCores_{0};
    TBuf<TPosition::VECCALC> rowBuf_;
};

} // namespace

// ============================================================================
// Kernel entry points
// ============================================================================

// Solve kernel: 多核 (blockDim=numCores)
// tiling 由 host by-value 传入 (const 引用 → kernel by value 接收)
extern "C" __global__ __aicore__ void spsm_solve_kernel(
    GM_ADDR csrRowOffsets, GM_ADDR csrColInd, GM_ADDR csrValues,
    GM_ADDR matB, GM_ADDR matC,
    GM_ADDR workspaceGM, const SpsmTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe;
    SpsmSolveAIV op;
    op.Init(csrRowOffsets, csrColInd, csrValues, matB, matC, workspaceGM, tiling, &pipe);
    op.Process();
}

// ============================================================================
// Host-side launch dispatchers
// ============================================================================
extern "C" void spsm_solve_kernel_do(
    GM_ADDR csrRowOffsets, GM_ADDR csrColInd, GM_ADDR csrValues,
    GM_ADDR matB, GM_ADDR matC,
    GM_ADDR workspaceGM, const SpsmTilingData& tiling,
    uint32_t blockDim, void *stream)
{
    if (blockDim == 0) { blockDim = 1; }
    spsm_solve_kernel<<<blockDim, nullptr, stream>>>(
        csrRowOffsets, csrColInd, csrValues, matB, matC,
        workspaceGM, tiling);
}

// ============================================================================
// Vector transpose kernel (COL<->ROW 转置)
//
// 多核 (blockDim=GetSpsmBlockDim), 按 GetBlockIdx/GetBlockNum 切分行, 每核处理若干行。
// AIV_ONLY 任务, 无 cube/vector 握手要求, 空核直接返回。
// 转置逻辑由 SpsmTransposeAIV 用 DataCopyPad 跨步搬运实现 (无标量 GM 读写)。
// ============================================================================
extern "C" __global__ __aicore__ void spsm_transpose_kernel(
    GM_ADDR src, GM_ADDR dst, int32_t m, int32_t n, int32_t ld, int32_t direction)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe;
    SpsmTransposeAIV op;
    op.Init(src, dst, m, n, ld, direction, &pipe);
    op.Process();
}

extern "C" void spsm_transpose_kernel_do(
    GM_ADDR src, GM_ADDR dst, int32_t m, int32_t n, int32_t ld,
    int32_t direction, uint32_t blockDim, void *stream)
{
    if (blockDim == 0) { blockDim = 1; }
    spsm_transpose_kernel<<<blockDim, nullptr, stream>>>(src, dst, m, n, ld, direction);
}
