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

#ifndef SPMV_CSR_MAT_H
#define SPMV_CSR_MAT_H
#include "spmv.h"
#include "cann_ops_sparse.h"
#include "cann_ops_sparse_common.h"
#include "aclsparse_descr_internal.h"

class SpmvCsrMat {
public:
    SpmvCsrMat(aclsparseSpMatDescr *inner) : rows(inner->rows), cols(inner->cols), nnz(inner->nnz), matDesc(inner)
    {
        // 支持 CSC 并且 transpose的格式
        if (inner->format == ACL_SPARSE_FORMAT_CSC) {
            rows = inner->cols;
            cols = inner->rows;
        }
    }
    ~SpmvCsrMat()
    {}
    uint32_t DoPreProcess(uint8_t *workspace);

private:
    uint32_t spmvCheckSubMatStats(SpmvCsrInfo *info);
    uint32_t MallocMatInfoAndCopyFromDev();
    void FreeMatInfo();
    uint32_t spmvDivideSubMat(SpmvCsrInfo *infos);
    uint32_t spmvDoPost(SpmvCsrInfo *info, uint8_t *gm);
    void GetRowElem(
        uint64_t startRow, uint64_t rowNum, uint32_t *idx, float *vals, uint64_t max, uint64_t curId);
    void ShowElem(uint32_t *idx, float *vals, uint64_t max, uint64_t rows);
    void GetRowElemNum(uint64_t startRow, uint64_t rowNum, uint32_t *nums, uint64_t curId);
    void ShowRowElem(uint64_t startRow, uint64_t rowNum, uint64_t curId);
    uint64_t GetMaxNum(uint32_t *nums, uint64_t num);
    uint64_t rows;
    uint64_t cols;
    uint64_t nnz;
    float *values;
    uint32_t *idxs;
    uint32_t *ptrs;
    aclsparseSpMatDescr *matDesc;
};

#endif