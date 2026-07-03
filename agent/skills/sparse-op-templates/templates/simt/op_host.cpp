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

// TEMPLATE: SIMT Host 侧实现（Sparse 算子，仅 arch35 可用）
// 强制结构：Validate{Op}Params + Launch{Op}Kernel 拆分；dlog 集成（log/log.h）
// Tiling 传递方式：host 侧以 const 引用传入 kernel_do（无 GM 设备内存分配，无同步）
//
// 与 SIMD/RegBase 的区别：
//   - 需额外计算 nthreads（每 block 线程数）
//   - block 数按 CeilDiv(totalWork, SIMT_MIN_THREAD_NUM) 与 coreNum 取小
//   - tiling 中包含标量参数（alpha/beta 等）直接传给 kernel
//   - kernel 侧无 TPipe/TQue，使用线程级并行 + grid-stride loop

#include <cstdint>
#include "log/log.h"
#include "cann_ops_sparse.h"
#include "{{op}}_kernel.h"
#include "aclsparse_handle_internal.h"
#include "aclsparse_descr_internal.h"
#include "host_utils.h"

// GetAivCoreCount：由 host_utils.h 提供的公共版本
// **禁止**在算子 host 文件中重复定义此函数

// SIMT 常量（来自 kernel_constant.h）
// SIMT_MIN_THREAD_NUM = 128, SIMT_MAX_THREAD_NUM = 2048

// TEMPLATE: Validate{Op}Params — 参数校验
static aclsparseStatus_t Validate{{Op}}Params(
    aclsparseHandle_t handle /* , 算子的参数列表 */)
{
    // 描述符 nullptr 校验
    // if (matA == nullptr) { OP_LOGE("aclsparse{{Op}}", "matA is nullptr"); return ACL_SPARSE_STATUS_INVALID_VALUE; }
    // 维度合法性校验（rows >= 0, cols >= 0, nnz >= 0）
    // 格式支持校验
    // 数据类型支持校验
    return ACL_SPARSE_STATUS_SUCCESS;
}

// TEMPLATE: Cal{Op}TilingData — 在 host 侧计算多核切分参数
// SIMT 特有：计算 nthreads
static {{Op}}TilingData Cal{{Op}}TilingData(uint32_t useNumBlocks /* , 算子参数 */)
{
    {{Op}}TilingData tiling{};
    // TEMPLATE: SIMT 计算 nthreads
    // tiling.nthreads = std::min(
    //     CeilAlign<uint32_t>(CeilDiv<uint32_t>(totalWork, useNumBlocks), SIMT_MIN_THREAD_NUM),
    //     SIMT_MAX_THREAD_NUM);
    // TEMPLATE: 填充稀疏矩阵维度 + 标量参数（alpha/beta 直接放入 tiling，避免 kernel 再读 GM）
    return tiling;
}

// TEMPLATE: Launch{Op}Kernel — 负责描述符解析 + tiling 计算 + kernel launch
// - 异步执行：launch kernel 后直接返回，不调用 aclrtSynchronizeStream
static aclsparseStatus_t Launch{{Op}}Kernel(aclsparseHandle_t handle /* , 算子参数 */)
{
    auto* h = ToInternalHandle(handle);
    aclrtStream useStream = h->stream;

    // 描述符解析 — 从 SpMatDescr 提取关键信息
    // auto* matInner = ToMatInner(matA);
    // uint64_t rows   = matInner->rows;
    // uint64_t cols   = matInner->cols;
    // uint64_t nnz    = matInner->nnz;

    uint32_t aivCoreNum = GetAivCoreCount();
    if (aivCoreNum == 0) {
        OP_LOGE("aclsparse{{Op}}", "GetAivCoreCount failed");
        return ACL_SPARSE_STATUS_INTERNAL_ERROR;
    }

    // TEMPLATE: SIMT block 数计算（与 SIMD/RegBase 不同）
    // useNumBlocks = min(CeilDiv(totalWork/*rows or nnz*/, SIMT_MIN_THREAD_NUM), aivCoreNum)
    uint32_t useNumBlocks = aivCoreNum;  // TEMPLATE: 按算子实际需求计算

    {{Op}}TilingData tiling = Cal{{Op}}TilingData(useNumBlocks /* , 算子参数 */);

    OP_LOGD("aclsparse{{Op}}", "tiling: nthreads=%u numBlocks=%u", tiling.nthreads, useNumBlocks);
    OP_LOGI("aclsparse{{Op}}", "launching kernel: blocks=%u", useNumBlocks);

    // Tiling 直接以 const 引用传入（无 H2D 拷贝）
    {{op}}_kernel_do(/* GM_ADDR 各数据指针, */ useNumBlocks, tiling, useStream);

    return ACL_SPARSE_STATUS_SUCCESS;
}

// TEMPLATE: 公共 API 入口
aclsparseStatus_t aclsparse{{Op}}(aclsparseHandle_t handle /* , 算子参数 */)
{
    // 边界快速返回
    // if (rows == 0 || cols == 0 || nnz == 0) return ACL_SPARSE_STATUS_SUCCESS;

    if (handle == nullptr) {
        OP_LOGE("aclsparse{{Op}}", "handle is nullptr");
        return ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR;
    }

    aclsparseStatus_t st = Validate{{Op}}Params(handle /* , 参数 */);
    if (st != ACL_SPARSE_STATUS_SUCCESS) return st;
    return Launch{{Op}}Kernel(handle /* , 参数 */);
}
