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
 * \file spmv_op.h
 * \brief spmv_op Host 侧辅助函数：描述符转换、workspace 布局计算。
 *
 * Workspace 布局统一由 SpmvOpWorkspaceLayout 计算，避免多处重复实现：
 *   [+0]            header (kSpmvOpReorderOffset 字节)
 *   [+reorder]      reorder[m]
 *   [+tmpReorder]   tmpReorder[m]       - 归并排序临时缓冲
 *   [+scratch]      scratch[m]          - 每行 nnz
 *   [+tmpScratch]   tmpScratch[m]       - 归并排序临时缓冲
 *   [+bin_edge]     bin_edge[numBlocks+1]
 */

#ifndef SPMV_OP_H_
#define SPMV_OP_H_

#include <cstdint>
#include <cstddef>

#include "cann_ops_sparse.h"
#include "aclsparse_handle_internal.h"
#include "aclsparse_descr_internal.h"
#include "spmv_op_tiling_data.h"

/// 将 opaque handle 转为内部结构体指针
inline struct aclsparseContext *SpmvOpToInternalHandle(aclsparseHandle_t handle)
{
    return reinterpret_cast<struct aclsparseContext *>(handle);
}

/// 将 const SpMat 描述符转为内部结构体指针（解除 const）
inline struct aclsparseSpMatDescr *SpmvOpToMatInner(aclsparseConstSpMatDescr_t desc)
{
    return const_cast<struct aclsparseSpMatDescr *>(
        reinterpret_cast<const struct aclsparseSpMatDescr *>(desc));
}

/// 将 const DnVec 描述符转为内部结构体指针（解除 const）
inline struct aclsparseDnVecDescr *SpmvOpToVecInner(aclsparseConstDnVecDescr_t desc)
{
    return const_cast<struct aclsparseDnVecDescr *>(
        reinterpret_cast<const struct aclsparseDnVecDescr *>(desc));
}

/// ALG2 workspace 布局：统一计算所有偏移和总量。
/// 布局（按 64 字节对齐）：
///   [+0]            header (kSpmvOpReorderOffset 字节)
///   [+reorder]      reorder[m]
///   [+tmpReorder]   tmpReorder[m]       - 归并排序临时缓冲
///   [+scratch]      scratch[m]          - 每行 nnz
///   [+tmpScratch]   tmpScratch[m]       - 归并排序临时缓冲
///   [+bin_edge]     bin_edge[numBlocks+1]
struct SpmvOpWorkspaceLayout {
    size_t reorderOffset{0};
    size_t tmpReorderOffset{0};
    size_t scratchOffset{0};
    size_t tmpScratchOffset{0};
    size_t binEdgeOffset{0};
    size_t totalBytes{0};

    /// 计算完整布局，所有偏移均为 64 字节对齐。
    inline void Compute(int32_t m, uint32_t numBlocks) {
        constexpr size_t kMask = ~size_t(kSpmvOpWsAlignment - 1u);
        const size_t rb = static_cast<size_t>(m) * sizeof(int32_t);  // reorder / scratch / tmp 每块字节数
        const size_t binEdgeBytes = static_cast<size_t>(numBlocks + 1) * sizeof(int32_t);

        reorderOffset = kSpmvOpReorderOffset;
        tmpReorderOffset = ((reorderOffset + rb) + kSpmvOpWsAlignment - 1u) & kMask;
        scratchOffset    = ((tmpReorderOffset + rb) + kSpmvOpWsAlignment - 1u) & kMask;
        tmpScratchOffset = ((scratchOffset + rb) + kSpmvOpWsAlignment - 1u) & kMask;
        binEdgeOffset    = ((tmpScratchOffset + rb) + kSpmvOpWsAlignment - 1u) & kMask;
        totalBytes       = ((binEdgeOffset + binEdgeBytes) + kSpmvOpWsAlignment - 1u) & kMask;
    }
};

#endif  // SPMV_OP_H_
