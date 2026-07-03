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

// TEMPLATE: SIMD Kernel 实现（Sparse 算子）
// 标准骨架：Kernel 类（Init + Process）+ kernel entry + launcher
// 稀疏算子特点：间接索引（通过 rowPtr/colInd 访问数据），不规则内存访问模式

#include <cstdint>
#include "kernel_operator.h"
#include "{{op}}_kernel.h"

using namespace AscendC;

// TEMPLATE: BUFFER_NUM=1 纯搬运算子；BUFFER_NUM=2 有 Vector 计算需 CopyIn/Compute 重叠
constexpr uint32_t BUFFER_NUM = 1;

class {{Op}}AIV {
public:
    __aicore__ inline {{Op}}AIV() {}
    __aicore__ inline void Init(/* GM_ADDR 各输入/输出, */ const {{Op}}TilingData& tiling, TPipe* pipe);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyTilingData(const {{Op}}TilingData& src);
    __aicore__ inline void ProcessRow(uint32_t rowIdx);

    TPipe* pipe_;
    // 稀疏矩阵 GM 指针
    // GlobalTensor<int32_t> rowPtrGM_;
    // GlobalTensor<int32_t> colIndGM_;
    // GlobalTensor<float>   valuesGM_;
    // GlobalTensor<float>   xVecGM_;
    // GlobalTensor<float>   yVecGM_;
    // TQue<VECIN, BUFFER_NUM> / TQue<VECOUT, BUFFER_NUM>
    {{Op}}TilingData tiling_;
    uint32_t blockIdx_;
};

__aicore__ inline void {{Op}}AIV::Init(
    /* GM_ADDR 各输入/输出, */ const {{Op}}TilingData& tiling, TPipe* pipe)
{
    pipe_ = pipe;
    blockIdx_ = GetBlockIdx();
    CopyTilingData(tiling);

    // 多核切分 — 计算当前核负责的行范围
    // uint32_t startRow = blockIdx_ * tiling_.rowsPerCore + min(blockIdx_, tiling_.remainderRows);
    // uint32_t endRow   = startRow + tiling_.rowsPerCore + (blockIdx_ < tiling_.remainderRows ? 1 : 0);

    // SetGlobalBuffer — 按算子的 GM 指针逐个绑定
    // InitBuffer — 按队列数量和 tile 大小初始化
}

__aicore__ inline void {{Op}}AIV::CopyTilingData(const {{Op}}TilingData& src)
{
    // tiling 已通过 launch 参数拷贝至本地，直接按字段赋值即可
}

__aicore__ inline void {{Op}}AIV::ProcessRow(uint32_t rowIdx)
{
    // 稀疏算子典型处理流程（以 SpMV CSR 为例）：
    //
    // 1. 读取 rowPtr[rowIdx] 和 rowPtr[rowIdx+1]，得到该行的 nnz 范围和数量
    // 2. 批量搬入 colInd 和 values 到 UB
    // 3. 根据 colInd 间接读取 xVec 到 UB（Gather 操作）
    // 4. 计算 dot product: sum += values[j] * xVec[colInd[j]]
    // 5. 写回 yVec[rowIdx] = alpha * sum + beta * yVec[rowIdx]
    //
    // 注意：间接索引导致内存访问不规则，需要合理设计 UB buffer 复用策略
}

__aicore__ inline void {{Op}}AIV::Process()
{
    // uint32_t startRow = /* 当前核起始行 */;
    // uint32_t endRow   = /* 当前核结束行 */;
    // for (uint32_t row = startRow; row < endRow; row++) {
    //     ProcessRow(row);
    // }
}

// Kernel 入口函数
extern "C" __global__ __aicore__ void {{op}}_kernel(
    /* GM_ADDR 各数据指针, */ const {{Op}}TilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe;
    {{Op}}AIV op;
    op.Init(/* ..., */ tiling, &pipe);
    op.Process();
}

// Kernel 启动器（host 侧调用）
void {{op}}_kernel_do(
    /* GM_ADDR 各数据指针, */ uint32_t numBlocks,
    const {{Op}}TilingData& tiling, void* stream)
{
    {{op}}_kernel<<<numBlocks, nullptr, stream>>>(/* ..., */ tiling);
}
