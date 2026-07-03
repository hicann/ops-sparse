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

// TEMPLATE: RegBase Kernel 实现（Sparse 算子，仅 arch35 可用）
// 标准三层结构：Kernel 类（Init + CopyIn + ComputeWithRegBase + CopyOut + Process）+ kernel entry + launcher
//
// 与 SIMD 的区别：
//   - 使用 TBuf<QuePosition::VECCALC> + LocalTensor 替代 TQue 队列
//   - 计算在 __VEC_SCOPE__ 块内使用 RegTensor（寄存器级 API）
//   - host 预计算 UB buffer 大小并放入 tiling

#include <cstdint>
#include "kernel_operator.h"
#include "{{op}}_kernel.h"

using namespace AscendC;

template <typename T>
class {{Op}}AIV {
public:
    __aicore__ inline {{Op}}AIV() {}
    __aicore__ inline void Init(/* GM_ADDR 各输入/输出, */ const {{Op}}TilingData& tiling);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyTilingData(const {{Op}}TilingData& src);
    __aicore__ inline void InitUbuf();
    __aicore__ inline void ProcessRowRegs(uint32_t rowIdx);

    TPipe pipe;

    // 稀疏矩阵 GM 指针
    // GlobalTensor<int32_t> rowPtrGM;
    // GlobalTensor<int32_t> colIndGM;
    // GlobalTensor<float>   valuesGM;
    // GlobalTensor<float>   xVecGM;
    // GlobalTensor<float>   yVecGM;

    // RegBase 使用 TBuf（非 TQue）
    TBuf<QuePosition::VECCALC> colIndBuf;
    TBuf<QuePosition::VECCALC> valuesBuf;
    TBuf<QuePosition::VECCALC> accBuf;

    LocalTensor<int32_t> colIndLocal;
    LocalTensor<float>   valuesLocal;
    LocalTensor<float>   accLocal;

    {{Op}}TilingData tiling_;
    uint32_t blockIdx_;
};

template <typename T>
__aicore__ inline void {{Op}}AIV<T>::Init(
    /* GM_ADDR 各输入/输出, */ const {{Op}}TilingData& tiling)
{
    blockIdx_ = GetBlockIdx();
    CopyTilingData(tiling);
    InitUbuf();
}

template <typename T>
__aicore__ inline void {{Op}}AIV<T>::CopyTilingData(const {{Op}}TilingData& src)
{
    // tiling 已通过 launch 参数（const 引用）拷贝至本地，直接按字段赋值
}

template <typename T>
__aicore__ inline void {{Op}}AIV<T>::InitUbuf()
{
    // TEMPLATE: 使用 host 预计算的 buffer 大小初始化（256B 对齐）
    // pipe.InitBuffer(colIndBuf, tiling_.bufColInd);
    // pipe.InitBuffer(valuesBuf, tiling_.bufValues);
    // pipe.InitBuffer(accBuf,    tiling_.bufAcc);
    // colIndLocal = colIndBuf.Get<int32_t>();
    // valuesLocal = valuesBuf.Get<float>();
    // accLocal    = accBuf.Get<float>();
}

template <typename T>
__aicore__ inline void {{Op}}AIV<T>::ProcessRowRegs(uint32_t rowIdx)
{
    // 稀疏算子 RegBase 处理流程（以 SpMV CSR 为例）：

    // 1. 读取 rowPtr[rowIdx:rowIdx+1]，得到该行的 nnz 范围和数量
    // int32_t rowStart = rowPtrGM[rowIdx].GetValue();
    // int32_t rowEnd   = rowPtrGM[rowIdx + 1].GetValue();
    // uint32_t rowNnz  = rowEnd - rowStart;

    // 2. 批量搬入 colInd 和 values 到 UB（DataCopyPad）
    // DataCopyExtParams copyParams{1, rowNnz * sizeof(int32_t), 0, 0, 0};
    // DataCopyPad(colIndLocal, colIndGM[rowStart], copyParams);

    // 3. 同步 MTE2->V
    // SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
    // WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);

    // 4. RegBase 计算：在 __VEC_SCOPE__ 块内使用寄存器级 API
    __VEC_SCOPE__
    {
        // 声明 RegTensor
        // AscendC::MicroAPI::RegTensor<float, AscendC::MicroAPI::RegTraitNumOne> vregAcc;
        // AscendC::MicroAPI::Duplicate<float>(vregAcc, 0.0f, maskAll);

        // 分块处理（每块 VL=64 个 float）
        // for (uint32_t i = 0; i < rowNnz; i += VL) { ... }

        // 寄存器级 gather + multiply + accumulate
        // AscendC::MicroAPI::DataCopy<float, ...>(vregValues, ...);
        // AscendC::MicroAPI::Mul<float, ...>(vregProd, vregValues, vregX, mask);
        // AscendC::MicroAPI::Add<float, ...>(vregAcc, vregAcc, vregProd, maskAll);

        // 归约求和
        // AscendC::MicroAPI::ReduceSum(vregAcc, vregAcc, maskAll);

        // 将标量结果写回 UB
        // AscendC::MicroAPI::DataCopy<float, ...>(accAddr, vregAcc, 1, maskAll);
    }

    // 5. 写回 yVec[rowIdx] = alpha * acc + beta * yVec[rowIdx]
}

template <typename T>
__aicore__ inline void {{Op}}AIV<T>::Process()
{
    // uint32_t startRow = blockIdx_ * tiling_.rowsPerCore + min(blockIdx_, tiling_.remainderRows);
    // uint32_t endRow   = startRow + tiling_.rowsPerCore + (blockIdx_ < tiling_.remainderRows ? 1 : 0);
    // for (uint32_t row = startRow; row < endRow; row++) {
    //     ProcessRowRegs(row);
    // }
}

// Kernel 入口函数
extern "C" __global__ __aicore__ void {{op}}_kernel(
    /* GM_ADDR 各数据指针, */ const {{Op}}TilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

    // TEMPLATE: 根据 dtype 实例化（RegBase 通常需要按 dtype 分支）
    // if (tiling.valType == ACL_SPARSE_R_32F) {
    //     {{Op}}AIV<float> op;
    //     op.Init(/* GM 指针 */, tiling);
    //     op.Process();
    // } else if (tiling.valType == ACL_SPARSE_R_16F) {
    //     {{Op}}AIV<half> op;
    //     op.Init(/* GM 指针 */, tiling);
    //     op.Process();
    // }
}

// Kernel 启动器（host 侧调用）
void {{op}}_kernel_do(
    /* GM_ADDR 各数据指针, */ uint32_t numBlocks,
    const {{Op}}TilingData& tiling, void* stream)
{
    {{op}}_kernel<<<numBlocks, nullptr, stream>>>(/* ..., */ tiling);
}
