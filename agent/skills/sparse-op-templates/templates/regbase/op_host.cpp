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

// TEMPLATE: RegBase Host 侧实现（Sparse 算子，仅 arch35 可用）
// 强制结构：Validate{Op}Params + Launch{Op}Kernel 拆分；dlog 集成（log/log.h）
// Tiling 传递方式：host 侧以 const 引用传入 kernel_do（无 GM 设备内存分配，无同步）
//
// 与 SIMD 的区别：
//   - 使用 TBuf + LocalTensor 替代 TQue 队列
//   - __VEC_SCOPE__ 块内使用寄存器级 API（RegTensor）
//   - host 侧需预计算 UB buffer 大小（256B 对齐）放入 tiling

#include <cstdint>
#include "log/log.h"
#include "cann_ops_sparse.h"
#include "{{op}}_kernel.h"
#include "aclsparse_handle_internal.h"
#include "aclsparse_descr_internal.h"
#include "host_utils.h"

// GetAivCoreCount：由 host_utils.h 提供的公共版本
// **禁止**在算子 host 文件中重复定义此函数

// TEMPLATE: UB 可用大小（理论最大值 248KB，实际需预留栈空间）
constexpr uint32_t UB_SIZE = 190 * 1024;
constexpr uint32_t ALIGN_BYTES = 256;

static uint32_t AlignUp(uint32_t size, uint32_t align)
{
    return (size + align - 1) & ~(align - 1);
}

// TEMPLATE: 计算 UB buffer 布局
struct BufferLayout {
    uint32_t bufRowPtr;   // rowPtr 读取 buffer
    uint32_t bufColInd;   // colInd 读取 buffer
    uint32_t bufValues;   // values 读取 buffer
    uint32_t bufAcc;      // 累加器 buffer
    uint32_t total;
};

static BufferLayout ComputeBufferLayout(uint32_t maxRowNnz)
{
    BufferLayout layout = {};
    // TEMPLATE: 按算子需求计算各 buffer 大小（256B 对齐）
    // layout.bufRowPtr  = AlignUp(sizeof(int32_t), ALIGN_BYTES);
    // layout.bufColInd  = AlignUp(maxRowNnz * sizeof(int32_t), ALIGN_BYTES);
    // layout.bufValues  = AlignUp(maxRowNnz * sizeof(float), ALIGN_BYTES);
    // layout.bufAcc     = AlignUp(sizeof(float), ALIGN_BYTES);
    layout.total = layout.bufRowPtr + layout.bufColInd + layout.bufValues + layout.bufAcc;
    return layout;
}

// TEMPLATE: Validate{Op}Params — 参数校验
static aclsparseStatus_t Validate{{Op}}Params(
    aclsparseHandle_t handle /* , 算子的参数列表 */)
{
    // 描述符 nullptr 校验
    // if (matA == nullptr) { OP_LOGE("aclsparse{{Op}}", "matA is nullptr"); return ACL_SPARSE_STATUS_INVALID_VALUE; }
    // 维度合法性校验
    // 格式支持校验（仅 CSR 等）
    return ACL_SPARSE_STATUS_SUCCESS;
}

// TEMPLATE: Cal{Op}TilingData — 在 host 侧计算多核切分参数
// RegBase 特有：需将 UB buffer 大小放入 tiling
static {{Op}}TilingData Cal{{Op}}TilingData(
    /* 稀疏矩阵维度参数, */ uint32_t coreNum, uint32_t usedCoreNum, uint32_t rowsPerCore, uint32_t remainderRows)
{
    {{Op}}TilingData tiling{};
    tiling.totalRows = /* rows */;
    tiling.rowsPerCore = rowsPerCore;
    tiling.remainderRows = remainderRows;
    tiling.usedCoreNum = usedCoreNum;

    // TEMPLATE: 填充 UB buffer 大小
    // auto layout = ComputeBufferLayout(/* maxRowNnz */);
    // tiling.bufColInd = layout.bufColInd;
    // tiling.bufValues = layout.bufValues;
    // tiling.bufAcc    = layout.bufAcc;
    return tiling;
}

// TEMPLATE: Launch{Op}Kernel — 负责描述符解析 + tiling 计算 + kernel launch
// - 异步执行：launch kernel 后直接返回，不调用 aclrtSynchronizeStream
static aclsparseStatus_t Launch{{Op}}Kernel(aclsparseHandle_t handle /* , 算子参数 */)
{
    auto* h = ToInternalHandle(handle);
    aclrtStream useStream = h->stream;

    uint32_t coreNum = GetAivCoreCount();
    if (coreNum == 0) {
        OP_LOGE("aclsparse{{Op}}", "GetAivCoreCount failed");
        return ACL_SPARSE_STATUS_INTERNAL_ERROR;
    }

    // 描述符解析 — 从 SpMatDescr 提取关键信息
    // auto* matInner = ToMatInner(matA);
    // uint64_t rows = matInner->rows;
    // uint64_t cols = matInner->cols;
    // uint64_t nnz  = matInner->nnz;
    // void* csrRowPtr = matInner->ptrs;
    // void* csrColInd = matInner->idxs;
    // void* csrValues = matInner->values;

    // RegBase 按行数分配 core（与 SIMD 类似）
    uint32_t rowsPerCore = /* (rows + coreNum - 1) / coreNum */;
    uint32_t usedCoreNum = /* (rows + rowsPerCore - 1) / rowsPerCore */;
    uint32_t remainderRows = /* rows - (usedCoreNum - 1) * rowsPerCore */;

    {{Op}}TilingData tiling = Cal{{Op}}TilingData(/* 维度参数, */ coreNum, usedCoreNum, rowsPerCore, remainderRows);

    OP_LOGD("aclsparse{{Op}}", "tiling: usedCoreNum=%u rowsPerCore=%u", usedCoreNum, rowsPerCore);
    OP_LOGI("aclsparse{{Op}}", "launching kernel: usedCoreNum=%u", usedCoreNum);

    // Tiling 直接以 const 引用传入（无 H2D 拷贝）
    {{op}}_kernel_do(/* GM_ADDR 各数据指针, */ usedCoreNum, tiling, useStream);

    return ACL_SPARSE_STATUS_SUCCESS;
}

// TEMPLATE: 公共 API 入口
aclsparseStatus_t aclsparse{{Op}}(aclsparseHandle_t handle /* , 算子参数 */)
{
    // 边界快速返回
    // if (rows == 0 || cols == 0) return ACL_SPARSE_STATUS_SUCCESS;

    if (handle == nullptr) {
        OP_LOGE("aclsparse{{Op}}", "handle is nullptr");
        return ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR;
    }

    aclsparseStatus_t st = Validate{{Op}}Params(handle /* , 参数 */);
    if (st != ACL_SPARSE_STATUS_SUCCESS) return st;
    return Launch{{Op}}Kernel(handle /* , 参数 */);
}
