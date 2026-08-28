/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once

#include <cstdint>
#include <iostream>
#include <random>
#include <vector>

#include "acl/acl.h"
#include "cann_ops_sparse.h"
#include "descriptor_manager.h"

// ============================================================================
// NPU Workflow Result
// ============================================================================
struct CooGetNpuResult {
    aclsparseStatus_t createRet;
    aclsparseStatus_t getRet;

    // CooGet 读回的标量字段（Host 元数据，非张量计算）
    int64_t rows{0};
    int64_t cols{0};
    int64_t nnz{0};
    aclsparseIndexType_t idxType{};
    aclsparseIndexBase_t idxBase{};
    aclDataType valueType{};

    // CooGet 读回的设备指针（应与 Create 传入的设备指针一致）
    void *devRowInd{nullptr};
    void *devColInd{nullptr};
    void *devValues{nullptr};
    bool devPtrMatch{false};  // 读回指针是否与 Create 传入的设备指针逐字一致
};

// ============================================================================
// NPU wrapper: CreateCoo -> CooGet accessor 验证
// 全程在 NPU 侧建设备缓冲存指针；Host 仅做元数据生成与 accessor 断言。
// ============================================================================
inline CooGetNpuResult CooGetNpu(aclrtStream stream,
                                 const std::vector<int32_t> &hostRowInd,
                                 const std::vector<int32_t> &hostColInd,
                                 int rows, int cols, int nnz, int idx_base,
                                 bool isFp16)
{
    using namespace sparse_test;
    (void)stream;
    CooGetNpuResult r{};

    if (nnz < 0 || static_cast<int>(hostRowInd.size()) != nnz ||
        static_cast<int>(hostColInd.size()) != nnz) {
        r.createRet = ACL_SPARSE_STATUS_INVALID_VALUE;
        return r;
    }

    // 输入张量迁移到 NPU 设备
    auto dRow = DeviceBuffer::copyFrom(hostRowInd.data(), nnz * sizeof(int32_t));
    auto dCol = DeviceBuffer::copyFrom(hostColInd.data(), nnz * sizeof(int32_t));

    aclDataType valueType = isFp16 ? ACL_FLOAT16 : ACL_FLOAT;
    aclsparseIndexBase_t idxBase =
        (idx_base == 1) ? ACL_SPARSE_INDEX_BASE_ONE : ACL_SPARSE_INDEX_BASE_ZERO;

    size_t valBytes = isFp16 ? (static_cast<size_t>(nnz) * sizeof(uint16_t))
                             : (static_cast<size_t>(nnz) * sizeof(float));
    auto dVal = DeviceBuffer::alloc(valBytes);  // accessor 不读数据，设备缓冲仅占位

    // aclsparseCreateCoo
    aclsparseSpMatDescr_t descr = nullptr;
    r.createRet = aclsparseCreateCoo(&descr, rows, cols, nnz,
        dRow.get(), dCol.get(), dVal.get(), ACL_SPARSE_INDEX_32I, idxBase, valueType);
    if (r.createRet != ACL_SPARSE_STATUS_SUCCESS || descr == nullptr) {
        std::cerr << "[NPU] aclsparseCreateCoo failed: " << r.createRet << std::endl;
        return r;
    }

    // aclsparseCooGet
    r.getRet = aclsparseCooGet(descr, &r.rows, &r.cols, &r.nnz,
        &r.devRowInd, &r.devColInd, &r.devValues, &r.idxType, &r.idxBase, &r.valueType);

    if (r.getRet != ACL_SPARSE_STATUS_SUCCESS) {
        aclsparseDestroySpMat(descr);
        std::cerr << "[NPU] aclsparseCooGet failed: " << r.getRet << std::endl;
        return r;
    }

    // accessor 语义断言：Get 读回的设备指针应与 Create 传入的逐字一致
    r.devPtrMatch = (r.devRowInd == dRow.get() && r.devColInd == dCol.get() &&
                     r.devValues == dVal.get());

    aclsparseDestroySpMat(descr);
    return r;
}

// ============================================================================
// 生成 COO 测试数据（host 侧仅生成输入，不在 CPU 做算子计算）
// ============================================================================
inline void GenerateCooData(int rows, int cols, int nnz, int idx_base,
                            uint32_t seed,
                            std::vector<int32_t> &rowInd, std::vector<int32_t> &colInd)
{
    rowInd.clear();
    colInd.clear();
    if (rows <= 0 || cols <= 0 || nnz <= 0) {
        return;
    }
    rowInd.reserve(nnz);
    colInd.reserve(nnz);
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int32_t> rowDist(0, static_cast<int32_t>(rows) - 1);
    std::uniform_int_distribution<int32_t> colDist(0, static_cast<int32_t>(cols) - 1);
    int32_t base = (idx_base == 1) ? 1 : 0;
    for (int i = 0; i < nnz; i++) {
        rowInd.push_back(rowDist(rng) + base);
        colInd.push_back(colDist(rng) + base);
    }
}
