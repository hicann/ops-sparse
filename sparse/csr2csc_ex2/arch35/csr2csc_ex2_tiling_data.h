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
 * \file csr2csc_ex2_tiling_data.h
 * \brief csr2csc_ex2 TilingData 结构体定义（Host / Kernel 共用）。
 *
 * csr2csc_ex2 包含五个 kernel：
 *   - Kernel 1 (CountCols):     遍历 csrColInd，原子加统计每 stripe 列直方图
 *   - Kernel 1.5 (SumStripeHist): 对 stripeHist 按列求和重建 colCount
 *   - Kernel 2 (PrefixSum):     单 warp 并行 exclusive prefix sum -> cscColPtr
 *   - Kernel 3 (StripeBase):    按列对 stripe 直方图做前缀和 -> 每 stripe 写游标基址
 *   - Kernel 4 (Scatter):       每 block 单线程顺序 scatter 写 cscRowInd / cscVal
 * 各使用独立的 TilingData 结构体。
 */

#ifndef CSR2CSC_EX2_TILING_DATA_H_
#define CSR2CSC_EX2_TILING_DATA_H_

#include <cstddef>
#include <cstdint>

/// 每 block 最大线程数（SIMT VF launch_bounds）
constexpr uint32_t kCsr2CscThreadsPerBlock = 256;

/// Warp 大小（向上取整对齐用）
constexpr uint32_t kCsr2CscWarpSize = 32;

/// Workspace 固定段数：段 0 = colCount（(n+1) 个 int32）
/// 段 1 起为 stripe 直方图/游标区（stripeCount 段，每段 (n+1) 个 int32）
constexpr uint32_t kCsr2CscWorkspaceSegments = 1;

/// Stripe 直方图区 workspace 上限（字节）。当 n 较大时，每 stripe 需要 (n+1)×sizeof(int32_t)
/// 空间，stripeCount 过大会导致：(1) workspace 过大；(2) StripeBase kernel 复杂度
/// O(n × stripeCount) 过高。本常量限制 stripeHist 区总大小，使大 n 场景自动退化为较少
/// stripe，避免 StripeBase 复杂度过高。
/// stripeCount = min(ceil(nnz/256), aivCoreNum)，每 block 单线程处理一个 stripe。
constexpr size_t kCsr2CscMaxStripeWorkspaceBytes = 16 * 1024 * 1024;

/// copyValues 取值（TilingData 内部表示，与 aclsparseAction_t 对应）。
/// 使用 int32_t 而非 enum：TilingData 作为 host/kernel 间 ABI 载体，
/// int32_t 保证枚举底层类型确定性与序列化稳定性（enum 底层类型由编译器决定，
/// 跨编译器/编译选项可能不一致）。
constexpr int32_t kCsr2CscSymbolic = 0; // 对应 ACL_SPARSE_ACTION_SYMBOLIC
constexpr int32_t kCsr2CscNumeric = 1;  // 对应 ACL_SPARSE_ACTION_NUMERIC

/// Kernel 1 (CountCols) TilingData
///
/// 注意：CountCols 只写 stripeHist，不再写 colCount（K1 性能优化）。
/// colCount 由 Kernel 1.5 (SumStripeHist) 从 stripeHist 按列求和重建。
struct Csr2CscCountTilingData {
    int32_t nnz;         ///< 非零元素总数
    int32_t n;           ///< CSR 列数 / CSC 行数
    int32_t idxBase;     ///< 索引基值 (0 或 1)
    int32_t stripeCount; ///< Scatter 分段数（stripe 直方图段数）
    int32_t stripeSize;  ///< 每 stripe 覆盖的 nnz 元素数
};

/// Kernel 1.5 (SumStripeHist) TilingData
///
/// 对 stripeHist 按列求和重建 colCount：colCount[j] = sum_t stripeHist[t*(n+1)+j]。
/// 多线程并行，每线程处理若干列。正确性依据：原 CountCols 中 colCount[j] 与
/// stripeHist[t][j] 对同一 (col==j, k in stripe t) 元素计数，故
/// sum_t stripeHist[t][j] == colCount[j]（atomic_add 满足交换律）。
struct Csr2CscSumStripeHistTilingData {
    int32_t n;           ///< CSR 列数 / CSC 行数
    int32_t stripeCount; ///< Scatter 分段数
};

/// Kernel 2 (PrefixSum) TilingData
struct Csr2CscPrefixSumTilingData {
    int32_t n;       ///< CSR 列数 / CSC 行数
    int32_t idxBase; ///< 索引基值 (0 或 1)
};

/// Kernel 3 (StripeBase) TilingData
struct Csr2CscStripeBaseTilingData {
    int32_t n;           ///< CSR 列数 / CSC 行数
    int32_t idxBase;     ///< 索引基值 (0 或 1)
    int32_t stripeCount; ///< Scatter 分段数
};

/// Kernel 4 (Scatter) TilingData
struct Csr2CscScatterTilingData {
    int32_t m;          ///< CSR 行数
    int32_t n;          ///< CSR 列数 / CSC 行数
    int32_t nnz;        ///< 非零元素总数
    int32_t idxBase;    ///< 索引基值 (0 或 1)
    int32_t copyValues; ///< kCsr2CscSymbolic=仅结构, kCsr2CscNumeric=结构+值
    int32_t stripeSize; ///< 每 stripe 覆盖的 nnz 元素数
    uint32_t valSize;   ///< 值类型字节数 (1/2/4)
    int32_t reserved = 0; ///< 保留字段，补齐到 32 字节（8 字节对齐）
};

#endif  // CSR2CSC_EX2_TILING_DATA_H_
