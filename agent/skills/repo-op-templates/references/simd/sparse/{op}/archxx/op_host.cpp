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

// TEMPLATE: SIMD Host 侧实现（Sparse 算子）
// 强制结构：Validate{Op}Params + Launch{Op}Kernel 拆分；dlog 集成（log/log.h）
// Tiling 传递方式：host 侧以 const 引用传入 kernel_do（无 GM 设备内存分配，无同步）
// 描述符解析：从 SpMatDescr/DnVecDescr/DnMatDescr 提取 format/rows/cols/nnz/valueType

#include <cstdint>
#include "log/log.h"
#include "cann_ops_sparse.h"
#include "{{op}}_kernel.h"
#include "aclsparse_handle_internal.h"
#include "aclsparse_descr_internal.h"
#include "host_utils.h"

// TEMPLATE: Validate{Op}Params — 参数校验
static aclsparseStatus_t Validate{{Op}}Params(
    aclsparseHandle_t handle /* , 算子的参数列表 */)
{
    // 描述符 nullptr 校验
    // if (matA == nullptr) { OP_LOGE("aclsparse{{Op}}", "matA is nullptr"); return ACL_SPARSE_STATUS_INVALID_VALUE; }
    // 维度合法性校验（rows >= 0, cols >= 0, nnz >= 0）
    // 格式支持校验（format == ACL_SPARSE_FORMAT_CSR 等）
    // 数据类型支持校验
    return ACL_SPARSE_STATUS_SUCCESS;
}

// TEMPLATE: Cal{Op}TilingData — 在 host 侧计算多核切分参数
static {{Op}}TilingData Cal{{Op}}TilingData(/* 稀疏矩阵维度参数, */ uint32_t coreNum)
{
    {{Op}}TilingData tiling{};
    // TEMPLATE: 按算子的切分策略填充 tiling 各字段
    // 稀疏算子典型切分：按行均分（每核处理 rows/coreNum 行）
    // tiling.totalRows = rows;
    // tiling.rowsPerCore = rows / coreNum;
    // tiling.remainderRows = rows % coreNum;
    return tiling;
}

// TEMPLATE: Launch{Op}Kernel — 负责描述符解析 + tiling 计算 + kernel launch
// - 异步执行：launch kernel 后直接返回，不调用 aclrtSynchronizeStream
static aclsparseStatus_t Launch{{Op}}Kernel(
    aclsparseHandle_t handle /* , 算子参数 */)
{
    auto* h = ToInternalHandle(handle);
    aclrtStream useStream = h->stream;

    // 描述符解析 — 从 SpMatDescr/DnVecDescr 提取关键信息
    // auto* matInner = ToMatInner(matA);
    // uint64_t rows = matInner->rows;
    // uint64_t cols = matInner->cols;
    // uint64_t nnz  = matInner->nnz;
    // void* csrRowPtr = matInner->ptrs;
    // void* csrColInd = matInner->idxs;
    // void* csrValues = matInner->values;

    uint32_t aivCoreNum = GetAivCoreCount();
    if (aivCoreNum == 0) {
        OP_LOGE("aclsparse{{Op}}", "GetAivCoreCount failed");
        return ACL_SPARSE_STATUS_INTERNAL_ERROR;
    }

    uint32_t numBlocks = /* min(totalRows, aivCoreNum) */;

    {{Op}}TilingData tiling = Cal{{Op}}TilingData(/* 维度参数, */ numBlocks);

    OP_LOGD("aclsparse{{Op}}", "tiling: ...");
    OP_LOGI("aclsparse{{Op}}", "launching kernel: blocks=%u", numBlocks);

    // Tiling 直接以 const 引用传入（无 H2D 拷贝）
    {{op}}_kernel_do(/* GM_ADDR 各数据指针, */ numBlocks, tiling, useStream);

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
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        return st;
    }

    return Launch{{Op}}Kernel(handle /* , 参数 */);
}
