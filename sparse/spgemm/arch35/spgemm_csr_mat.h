/**
 * ----------------------------------------------------------------------------------------------------------
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ----------------------------------------------------------------------------------------------------------
 */

#ifndef SPGEMM_CSR_MAT_H
#define SPGEMM_CSR_MAT_H

#include <stdint.h>
#include <stdlib.h>
#include <vector>
#include "spgemm.h"
#include "cann_ops_sparse.h"
#include "aclsparse_host_utils.h"
#include "aclsparse_descr_internal.h"

/* Result of greedy bin-packing: row reorder + bin edges */
struct SpgemmRowBinPackResult {
    std::vector<int32_t> reorder;
    std::vector<int32_t> binEdges;
};

/* 贪心装箱：按负载降序排列，放入负载最小的核。 */
void SpgemmGreedyRowBinPack(
    const std::vector<int32_t> &rowWeight,
    int32_t binNum,
    SpgemmRowBinPackResult &result);

/* Compute row weights as "product pairs" = sum of nnz(B.row(k)) for each k in A.row(i).
 * This is the true work metric for SpGEMM, not just nnz(A.row). */
aclsparseStatus_t SpgemmComputeRowWeights(
    const std::vector<int32_t> &aRowPtr,
    const std::vector<int32_t> &aColInd,
    const std::vector<int32_t> &bRowPtr,
    std::vector<int32_t> &rowWeights);

/* Compute total numProds = sum over all rows of product pairs */
int64_t SpgemmComputeNumProds(
    const std::vector<int32_t> &aRowPtr,
    const std::vector<int32_t> &aColInd,
    const std::vector<int32_t> &bRowPtr);

/* Write reorder + binEdges to device workspace */
aclsparseStatus_t SpgemmWriteReorderToWorkspace(
    const SpgemmRowBinPackResult &binPack,
    void *dWorkspace,
    const SpgemmWsOffsets &off,
    aclrtStream stream);

/* Copy B rowPtr to workspace (both symbolic+numeric kernels use workspace copy) */
aclsparseStatus_t SpgemmCopyBRowPtrToWorkspace(
    aclsparseSpMatDescr *matB,
    void *dWorkspace,
    const SpgemmWsOffsets &off,
    aclrtStream stream);

/* 校验 CSR 列索引是否按升序排列。
 * 输入 A/B 列索引须 sorted。
 * Returns ACL_SPARSE_STATUS_NOT_SUPPORTED if any row has unsorted colInd. */
aclsparseStatus_t SpgemmValidateCsrSorted(
    aclsparseSpMatDescr *matDesc,
    const char *label);

#endif /* SPGEMM_CSR_MAT_H */
