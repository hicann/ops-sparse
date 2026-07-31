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

#ifndef SPSM_H_
#define SPSM_H_

#include <cstdint>
#include "cann_ops_sparse.h"
#include "spsm_tiling_data.h"

// ============================================================================
// workspace 布局
//
//   [ 0             , levelRowPtrOff )  : 64B header
//   [ levelRowPtrOff, levelRowIdxOff )  : int32 levelRowPtr[L+1]
//   [ levelRowIdxOff, diagValOff     )  : int32 levelRowIdx[m]
//   [ diagValOff    , transRowOffOff )  : float diagVal[m]  (NON_UNIT)
//   [ transRowOffOff, transColIndOff )  : int32 transRowOff[m+1]  (opA=T)
//   [ transColIndOff, transValOff    )  : int32 transColInd[nnz]  (opA=T)
//   [ transValOff   , denseBufOff    )  : float transValues[nnz]  (opA=T)
//   [ denseBufOff   , endOff         )  : float denseBuf[m*n]  (COL-order 转置缓冲, ROW-order; 仅 orderB==COL || orderC==COL 时分配)
//
// tiling by-value 传递, 不落盘 GM。
// BufferSize 统一按 opA=T + NON_UNIT 最大值预留 (不含 denseBuf, denseBuf 按 order 条件分配)。
// ============================================================================

#define SPSM_WS_HEADER_BYTES     64
#define SPSM_WS_ALIGN            64

// order 编码
#define SPSM_ORDER_ROW 0
#define SPSM_ORDER_COL 1

// fillMode 编码
#define SPSM_FILL_LOWER 0
#define SPSM_FILL_UPPER 1

// diagType 编码
#define SPSM_DIAG_NON_UNIT 0
#define SPSM_DIAG_UNIT     1

// solveDir 编码
#define SPSM_SOLVE_FORWARD  0
#define SPSM_SOLVE_BACKWARD 1

static inline int64_t spsm_align_up(int64_t v, int64_t a) {
    if (a == 0) {
        return v;
    }
    if (v > INT64_MAX - a) {
        return v;
    }
    return ((v + a - 1) / a) * a;
}

// SpSM 描述符内部结构：跨 BufferSize / Analysis / Solve 三阶段共享。
// 注: 公共框架暂未覆盖 SpSM 描述符管理, 故本算子自行实现创建/销毁
// (aclsparseSpSMCreateDescr / aclsparseSpSMDestroyDescr)。
struct aclsparseSpSMDescr {
    bool analyzed = false;
    void *buffer = nullptr;
    aclsparseOperation_t opA = ACL_SPARSE_OP_NON_TRANSPOSE;
    aclsparseOperation_t opB = ACL_SPARSE_OP_NON_TRANSPOSE;
    int32_t needTranspose = 0;
    int32_t L = 0;
    // 缓存 BufferSize 阶段算出的 workspace 总大小, Analysis 阶段校验一致性
    int64_t cachedBufferSize = 0;
    // 缓存完整 TilingData (Analysis 时构造, Solve 时刷新 alpha 后 by-value 传入)
    SpsmTilingData cachedTiling{};
    // 缓存 Analysis 阶段 matA 的 (rows, cols, nnz, format), 跨阶段一致性校验用
    uint64_t matARows = 0;
    uint64_t matACols = 0;
    uint64_t matANnz = 0;
    aclsparseFormat_t matAFormat = ACL_SPARSE_FORMAT_CSR;
    // 缓存 Analysis 阶段 B/C 的 (rows, cols, order, ld), 跨阶段一致性校验用
    int64_t matBRows = 0;
    int64_t matBCols = 0;
    int64_t matBLd = 0;
    aclsparseOrder_t matBOrder = ACL_SPARSE_ORDER_ROW;
    int64_t matCRows = 0;
    int64_t matCCols = 0;
    int64_t matCLd = 0;
    aclsparseOrder_t matCOrder = ACL_SPARSE_ORDER_ROW;
};

// ============================================================================
// 描述符转换统一函数 (const_cast 收敛于此, 禁止在业务代码中直接 const_cast)。
// ============================================================================
inline struct aclsparseSpMatDescr *SpsmToMatInner(aclsparseConstSpMatDescr_t desc)
{
    return const_cast<struct aclsparseSpMatDescr *>(
        reinterpret_cast<const struct aclsparseSpMatDescr *>(desc));
}

inline struct aclsparseDnMatDescr *SpsmToDnMatInner(aclsparseConstDnMatDescr_t desc)
{
    return const_cast<struct aclsparseDnMatDescr *>(
        reinterpret_cast<const struct aclsparseDnMatDescr *>(desc));
}

// 公共 getter: 读取 SpSM 描述符缓存的 TilingData (供测试侧采集性能元数据)。
// 避免外部经 offsetof 直读 opaque 描述符内部布局。
inline const SpsmTilingData& GetSpsmCachedTiling(const aclsparseSpSMDescr *d)
{
    return d->cachedTiling;
}

#endif // SPSM_H_
