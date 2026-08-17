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

#include "spmm_csr_mat.h"
#include "spmm.h"
#include "acl/acl.h"
#include <algorithm>
#include <string.h>
#include <vector>
#include <numeric>

void SpmmArch22CsrMat::GreedyRowBinPack(const uint32_t *rowNnz, int32_t *reorder, int32_t *binEdges)
{
    int32_t m = static_cast<int32_t>(rows);
    std::vector<int32_t> sortedIdx(m);
    std::iota(sortedIdx.begin(), sortedIdx.end(), 0);
    std::sort(sortedIdx.begin(), sortedIdx.end(), [&](int32_t a, int32_t b) {
        return rowNnz[a] > rowNnz[b];
    });

    std::vector<uint64_t> binLoad(blockDim, 0);
    std::vector<std::vector<int32_t>> bins(blockDim);

    for (int32_t i = 0; i < m; i++) {
        int32_t minBin = 0;
        for (int32_t b = 1; b < blockDim; b++) {
            if (binLoad[b] < binLoad[minBin]) { minBin = b; }
        }
        bins[minBin].push_back(sortedIdx[i]);
        binLoad[minBin] += rowNnz[sortedIdx[i]];
    }

    int32_t pos = 0;
    binEdges[0] = 0;
    for (int32_t b = 0; b < blockDim; b++) {
        for (int32_t r : bins[b]) {
            reorder[pos++] = r;
        }
        binEdges[b + 1] = pos;
    }
}

void SpmmArch22CsrMat::BuildIdentityReorder(int32_t *reorder, int32_t *binEdges)
{
    int32_t m = static_cast<int32_t>(rows);
    for (int32_t i = 0; i < m; i++) { reorder[i] = i; }

    int32_t rowsPerBin = (m + blockDim - 1) / blockDim;
    binEdges[0] = 0;
    for (int32_t b = 0; b < blockDim; b++) {
        int32_t end = (b + 1) * rowsPerBin;
        if (end > m)
        {
            end = m;
        }
        binEdges[b + 1] = end;
    }
}

aclsparseStatus_t SpmmArch22CsrMat::DoPreProcess(uint8_t *dWorkspace, int64_t reorderOff, int64_t binEdgeOff)
{
    int32_t m = static_cast<int32_t>(rows);
    std::vector<int32_t> reorder(m);
    std::vector<int32_t> binEdges(blockDim + 1);

    if (matDesc->ptrs == nullptr || m <= 0) {
        BuildIdentityReorder(reorder.data(), binEdges.data());
    } else {
        std::vector<uint32_t> rowOff(m + 1);
        aclError aclRet = aclrtMemcpy(rowOff.data(), (m + 1) * sizeof(uint32_t),
                                      matDesc->ptrs, (m + 1) * sizeof(uint32_t),
                                      ACL_MEMCPY_DEVICE_TO_HOST);
        if (aclRet != ACL_ERROR_NONE) {
            return ACL_SPARSE_STATUS_EXECUTION_FAILED;
        }

        std::vector<uint32_t> rowNnz(m);
        for (int32_t i = 0; i < m; i++) {
            rowNnz[i] = rowOff[i + 1] - rowOff[i];
        }
        GreedyRowBinPack(rowNnz.data(), reorder.data(), binEdges.data());
    }

    aclError aclRet = aclrtMemcpy(dWorkspace + reorderOff, m * sizeof(int32_t),
                                  reorder.data(), m * sizeof(int32_t),
                                  ACL_MEMCPY_HOST_TO_DEVICE);
    if (aclRet != ACL_ERROR_NONE) {
        return ACL_SPARSE_STATUS_EXECUTION_FAILED;
    }
    aclRet = aclrtMemcpy(dWorkspace + binEdgeOff, (blockDim + 1) * sizeof(int32_t),
                         binEdges.data(), (blockDim + 1) * sizeof(int32_t),
                         ACL_MEMCPY_HOST_TO_DEVICE);
    if (aclRet != ACL_ERROR_NONE) {
        return ACL_SPARSE_STATUS_EXECUTION_FAILED;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}
