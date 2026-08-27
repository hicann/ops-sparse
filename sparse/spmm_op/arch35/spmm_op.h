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
 * \file spmm_op.h
 * \brief spmm_op Host 侧辅助函数：描述符转换、workspace 布局计算。
 *
 * Workspace 布局统一由 SpmmOpWorkspaceLayout 计算，避免多处重复实现：
 *   [+0]            header (SPMM_OP_REORDER_OFFSET 字节)
 *   [+reorder]      reorder[m]
 *   [+bin_edge]     bin_edge[numBlocks+1]
 *
 * 注：ALG2 preprocess 在 host 侧同步完成（std::stable_sort + aclrtMemcpy），
 *     排序临时缓冲位于 host 侧 std::vector，不占用 device workspace。
 */

#ifndef SPMM_OP_H_
#define SPMM_OP_H_

#include <cstdint>
#include <cstddef>

#include "cann_ops_sparse.h"
#include "aclsparse_handle_internal.h"
#include "aclsparse_descr_internal.h"
#include "spmm_op_tiling_data.h"

/// 将 opaque handle 转为内部结构体指针
inline struct aclsparseContext *SpmmOpToInternalHandle(aclsparseHandle_t handle)
{
    return handle;
}

/// 将 const SpMat 描述符转为内部结构体指针
inline const struct aclsparseSpMatDescr *SpmmOpToMatInner(aclsparseConstSpMatDescr_t desc)
{
    return desc;
}

/// 将 const DnMat 描述符转为内部结构体指针
inline const struct aclsparseDnMatDescr *SpmmOpToDnMatInner(aclsparseConstDnMatDescr_t desc)
{
    return desc;
}

/// 将 DnMat 描述符转为内部结构体指针（非 const 输出描述符）
inline struct aclsparseDnMatDescr *SpmmOpToDnMatInner(aclsparseDnMatDescr_t desc)
{
    return desc;
}

/// ALG2 workspace 布局：统一计算所有偏移和总量。
/// 布局（按 64 字节对齐）：
///   [+0]            header (SPMM_OP_REORDER_OFFSET 字节)
///   [+reorder]      reorder[m]
///   [+bin_edge]     bin_edge[numBlocks+1]
///
/// host-side preprocess（std::stable_sort）的临时缓冲位于 host 侧 std::vector，
/// 不在此 device workspace 中分配。
struct SpmmOpWorkspaceLayout {
    size_t reorderOffset{0};
    size_t binEdgeOffset{0};
    size_t totalBytes{0};

    /// 计算完整布局，所有偏移均为 64 字节对齐。
    inline void Compute(int32_t m, uint32_t numBlocks)
    {
        constexpr size_t mask = ~size_t(SPMM_OP_WS_ALIGNMENT - 1u);
        const size_t rb = static_cast<size_t>(m) * sizeof(int32_t);  // reorder 字节数
        const size_t binEdgeBytes = static_cast<size_t>(numBlocks + 1) * sizeof(int32_t);

        reorderOffset = SPMM_OP_REORDER_OFFSET;
        binEdgeOffset = ((reorderOffset + rb) + SPMM_OP_WS_ALIGNMENT - 1u) & mask;
        totalBytes    = ((binEdgeOffset + binEdgeBytes) + SPMM_OP_WS_ALIGNMENT - 1u) & mask;
    }
};

#endif  // SPMM_OP_H_
