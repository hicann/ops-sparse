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
 * \file gebsr2gebsc_tiling_data.h
 * \brief gebsr2gebsc TilingData 结构体定义（Host / Kernel 共用）。
 *
 * gebsr2gebsc 将每个 block 视为标量时等价于 csr2csc_ex2，
 * 包含六个 kernel：
 *   - Kernel 1 (CountCols):     遍历 bsrColIndA，原子加统计每 stripe 列直方图
 *   - Kernel 1.5 (SumStripeHist): 对 stripeHist 按列求和重建 colCount
 *   - Kernel 2 (PrefixSum):     单 warp 并行 exclusive prefix sum -> bscColPtr
 *   - Kernel 3 (StripeBase):    按列对 stripe 直方图做前缀和 -> 每 stripe 写游标基址
 *   - Kernel 4 (Scatter):       每 block 单线程顺序 scatter 写 bscRowInd / bscVal
 *                              （含块值拷贝：direct copy 或 block transpose）
 * 各使用独立的 TilingData 结构体。
 */

#ifndef GEBSR2GEBSC_TILING_DATA_H_
#define GEBSR2GEBSC_TILING_DATA_H_

#include <cstddef>
#include <cstdint>

/// 每 block 最大线程数（SIMT VF launch_bounds）
constexpr uint32_t kGebsr2GebscThreadsPerBlock = 256;

/// Warp 大小（向上取整对齐用）
constexpr uint32_t kGebsr2GebscWarpSize = 32;

/// Workspace 固定段数：段 0 = colCount（(nb+1) 个 int32）
/// 段 1 起为 stripe 直方图/游标区（stripeCount 段，每段 (nb+1) 个 int32）
constexpr uint32_t kGebsr2GebscWorkspaceSegments = 1;

/// Stripe 直方图区 workspace 上限（字节）。当 nb 较大时，每 stripe 需要 (nb+1)×sizeof(int32_t)
/// 空间，stripeCount 过大会导致：(1) workspace 过大；(2) StripeBase kernel 复杂度
/// O(nb × stripeCount) 过高。本常量限制 stripeHist 区总大小，使大 nb 场景自动退化为较少
/// stripe，避免 StripeBase 复杂度过高。
/// stripeCount = min(ceil(nnzb/256), aivCoreNum)，每 block 单线程处理一个 stripe。
constexpr size_t kGebsr2GebscMaxStripeWorkspaceBytes = 16 * 1024 * 1024;

/// copyValues 取值（TilingData 内部表示，与 aclsparseAction_t 对应）。
/// 使用 int32_t 而非 enum：TilingData 作为 host/kernel 间 ABI 载体，
/// int32_t 保证枚举底层类型确定性与序列化稳定性（enum 底层类型由编译器决定，
/// 跨编译器/编译选项可能不一致）。
constexpr int32_t kGebsr2GebscSymbolic = 0; // 对应 ACL_SPARSE_ACTION_SYMBOLIC
constexpr int32_t kGebsr2GebscNumeric = 1;  // 对应 ACL_SPARSE_ACTION_NUMERIC

/// 块值拷贝模式（由 rowBlockDimC/colBlockDimC vs rowBlockDimA/colBlockDimA 决定）。
constexpr int32_t kGebsr2GebscBlockDirectCopy = 0;  ///< rC==rA && cC==cA
constexpr int32_t kGebsr2GebscBlockTranspose = 1;   ///< rC==cA && cC==rA

/// 块内内存布局（与 aclsparseDirection_t 对应）。
constexpr int32_t kGebsr2GebscDirRow = 0;            ///< ACL_SPARSE_DIRECTION_ROW
constexpr int32_t kGebsr2GebscDirColumn = 1;         ///< ACL_SPARSE_DIRECTION_COLUMN

/// Kernel 1 (CountCols) TilingData
struct Gebsr2GebscCountTilingData {
    int32_t nnzb;         ///< 非零块总数
    int32_t nb;           ///< 块列数
    int32_t idxBase;      ///< 索引基值 (0 或 1)
    int32_t stripeCount;  ///< Scatter 分段数
    int32_t stripeSize;   ///< 每 stripe 覆盖的 nnzb 元素数
};

/// Kernel 1.5 (SumStripeHist) TilingData
struct Gebsr2GebscSumStripeHistTilingData {
    int32_t nb;           ///< 块列数
    int32_t stripeCount;  ///< Scatter 分段数
};

/// Kernel 2 (PrefixSum) TilingData
struct Gebsr2GebscPrefixSumTilingData {
    int32_t nb;       ///< 块列数
    int32_t idxBase;  ///< 索引基值 (0 或 1)
};

/// Kernel 3 (StripeBase) TilingData
struct Gebsr2GebscStripeBaseTilingData {
    int32_t nb;           ///< 块列数
    int32_t idxBase;      ///< 索引基值 (0 或 1)
    int32_t stripeCount;  ///< Scatter 分段数
};

/// Kernel 4 (Scatter) TilingData
struct Gebsr2GebscScatterTilingData {
    int32_t mb;            ///< 块行数
    int32_t nb;            ///< 块列数
    int32_t nnzb;          ///< 非零块总数
    int32_t idxBase;       ///< 索引基值 (0 或 1)
    int32_t copyValues;    ///< kGebsr2GebscSymbolic / kGebsr2GebscNumeric
    int32_t stripeSize;    ///< 每 stripe 覆盖的 nnzb 元素数
    uint32_t valSize;      ///< 值类型字节数 (4/8/16)
    int32_t rowBlockDimA;  ///< 输入块行维
    int32_t colBlockDimA;  ///< 输入块列维
    int32_t rowBlockDimC;  ///< 输出块行维
    int32_t colBlockDimC;  ///< 输出块列维
    int32_t copyMode;      ///< kGebsr2GebscBlockDirectCopy / kGebsr2GebscBlockTranspose
    int32_t dirA;          ///< kGebsr2GebscDirRow / kGebsr2GebscDirColumn
    int32_t reserved = 0;  ///< 保留字段，补齐到 56 字节（8 字节对齐）
};

#endif  // GEBSR2GEBSC_TILING_DATA_H_
