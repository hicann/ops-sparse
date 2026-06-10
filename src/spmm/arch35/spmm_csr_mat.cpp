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

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <algorithm>
#include <vector>
#include "acl/acl.h"
#include "spmm.h"
#include "spmm_csr_mat.h"
#include "cann_ops_sparse_common.h"

namespace {

// Greedy bin packing: sort rows by nnz descending, then place each row into the
// bin with the smallest accumulated load. This balances workload across cores.
void GreedyRowBinPack(const std::vector<int32_t> &rowNnz,
                      int32_t binNum,
                      std::vector<int32_t> *reorder,
                      std::vector<int32_t> *binEdges)
{
    const int32_t m = static_cast<int32_t>(rowNnz.size());

    std::vector<int32_t> order(m);
    for (int32_t i = 0; i < m; ++i) {
        order[i] = i;
    }
    std::sort(order.begin(), order.end(), [&](int32_t a, int32_t b) {
        return rowNnz[a] > rowNnz[b];
    });

    std::vector<std::vector<int32_t>> bins(binNum);
    std::vector<int64_t> binLoad(binNum, 0);
    for (int32_t idx : order) {
        int32_t pick = 0;
        for (int32_t b = 1; b < binNum; ++b) {
            if (binLoad[b] < binLoad[pick]) {
                pick = b;
            }
        }
        bins[pick].push_back(idx);
        binLoad[pick] += rowNnz[idx];
    }

    reorder->resize(m);
    binEdges->resize(static_cast<size_t>(binNum) + 1);
    int32_t cursor = 0;
    (*binEdges)[0] = 0;
    for (int32_t b = 0; b < binNum; ++b) {
        for (int32_t r : bins[b]) {
            (*reorder)[cursor++] = r;
        }
        (*binEdges)[b + 1] = cursor;
    }
}

void BuildIdentityRowReorder(int64_t m,
    int32_t blockDim,
    std::vector<int32_t> *reorder,
    std::vector<int32_t> *binEdges)
{
    reorder->resize(static_cast<size_t>(m));
    for (int64_t i = 0; i < m; ++i) {
        (*reorder)[i] = static_cast<int32_t>(i);
    }
    binEdges->resize(static_cast<size_t>(blockDim) + 1);
    const int64_t rowsPerBin = (m + blockDim - 1) / blockDim;
    for (int32_t b = 0; b <= blockDim; ++b) {
        const int64_t edge = static_cast<int64_t>(b) * rowsPerBin;
        (*binEdges)[b] = static_cast<int32_t>(edge < m ? edge : m);
    }
}

uint32_t BuildGreedyRowReorderFromCsr(const aclsparseSpMatDescr *matDesc,
    int64_t m,
    int32_t blockDim,
    std::vector<int32_t> *reorder,
    std::vector<int32_t> *binEdges)
{
    std::vector<int32_t> rowOff(static_cast<size_t>(m) + 1);
    aclError aclRet = aclrtMemcpy(rowOff.data(),
                                  sizeof(int32_t) * (m + 1),
                                  matDesc->ptrs,
                                  sizeof(int32_t) * (m + 1),
                                  ACL_MEMCPY_DEVICE_TO_HOST);
    if (aclRet != ACL_ERROR_NONE) {
        return ACL_SPARSE_STATUS_EXECUTION_FAILED;
    }
    std::vector<int32_t> rowNnz(static_cast<size_t>(m));
    for (int64_t i = 0; i < m; ++i) {
        rowNnz[i] = rowOff[i + 1] - rowOff[i];
    }
    GreedyRowBinPack(rowNnz, blockDim, reorder, binEdges);
    return ACL_SPARSE_STATUS_SUCCESS;
}

uint32_t WriteReorderToWorkspace(uint8_t *dWorkspaceBase,
    int64_t reorderOffset,
    int64_t binEdgeOffset,
    int64_t m,
    int32_t blockDim,
    const std::vector<int32_t> &reorder,
    const std::vector<int32_t> &binEdges)
{
    aclError aclRet = aclrtMemcpy(dWorkspaceBase + reorderOffset,
                                  sizeof(int32_t) * m,
                                  reorder.data(),
                                  sizeof(int32_t) * m,
                                  ACL_MEMCPY_HOST_TO_DEVICE);
    if (aclRet != ACL_ERROR_NONE) {
        return ACL_SPARSE_STATUS_EXECUTION_FAILED;
    }
    aclRet = aclrtMemcpy(dWorkspaceBase + binEdgeOffset,
                         sizeof(int32_t) * (blockDim + 1),
                         binEdges.data(),
                         sizeof(int32_t) * (blockDim + 1),
                         ACL_MEMCPY_HOST_TO_DEVICE);
    if (aclRet != ACL_ERROR_NONE) {
        return ACL_SPARSE_STATUS_EXECUTION_FAILED;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

} // namespace

uint32_t SpmmCsrMat::DoPreProcess(uint8_t *dWorkspaceBase,
                                  int64_t reorderOffset,
                                  int64_t binEdgeOffset,
                                  bool identity)
{
    if (dWorkspaceBase == nullptr) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (m <= 0 || blockDim <= 0) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }

    std::vector<int32_t> reorder;
    std::vector<int32_t> binEdges;

    if (identity || matDesc == nullptr || matDesc->ptrs == nullptr) {
        BuildIdentityRowReorder(m, blockDim, &reorder, &binEdges);
    } else {
        uint32_t st = BuildGreedyRowReorderFromCsr(matDesc, m, blockDim, &reorder, &binEdges);
        if (st != ACL_SPARSE_STATUS_SUCCESS) {
            return st;
        }
    }

    return WriteReorderToWorkspace(dWorkspaceBase, reorderOffset, binEdgeOffset,
                                   m, blockDim, reorder, binEdges);
}
