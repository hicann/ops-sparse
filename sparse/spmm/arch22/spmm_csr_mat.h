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

#ifndef SPMM_ARCH22_CSR_MAT_H
#define SPMM_ARCH22_CSR_MAT_H

#include "spmm.h"
#include "aclsparse_descr_internal.h"
#include <stdio.h>
#include <stdlib.h>

class SpmmArch22CsrMat {
public:
    SpmmArch22CsrMat(aclsparseSpMatDescr *inner, int32_t blockDim)
        : matDesc(inner),
          blockDim(blockDim)
    {
        if (inner == nullptr) {
            printf("[SpmmArch22CsrMat] inner descriptor is nullptr\n");
            abort();
        }
        rows = inner->rows;
        cols = inner->cols;
        nnz = inner->nnz;
    }

    aclsparseStatus_t DoPreProcess(uint8_t *dWorkspace, int64_t reorderOff, int64_t binEdgeOff);

private:
    void GreedyRowBinPack(const uint32_t *rowNnz, int32_t *reorder, int32_t *binEdges);
    void BuildIdentityReorder(int32_t *reorder, int32_t *binEdges);

    aclsparseSpMatDescr *matDesc;
    int64_t rows;
    int64_t cols;
    int64_t nnz;
    int32_t blockDim;
};

#endif
