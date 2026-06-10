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

#include "cann_ops_sparse.h"
#include <iostream>
#include <stdlib.h>
#include <new>
#include <string.h>
#include "unistd.h"
#include "spmv.h"
#include "cann_ops_sparse_common.h"
#include "aclsparse_descr_internal.h"
#include "spmv_csr_mat.h"

extern void spmv_kernel_do(void* buffer, void* x, void* y, uint64_t rows, uint64_t cols, uint32_t nnz, uint32_t numBlocks, void *stream);
const uint32_t blockNum = 24; // 24为A2默认blockNum

aclsparseStatus_t aclSparseCreateDnVec(aclsparseDnVecDescr_t *dnVecDescr, int64_t size, void *values, aclDataType valueType)
{
    auto *inner = new (std::nothrow) aclsparseDnVecDescr();
    if (inner == nullptr) {
        return ACL_SPARSE_STATUS_ALLOC_FAILED;
    }

    inner->nums = size;
    inner->values = values;
    inner->valueType = valueType;
    *dnVecDescr = (aclsparseDnVecDescr_t)inner;
    return ACL_SPARSE_STATUS_SUCCESS;
}

aclsparseStatus_t aclSparseDestroyDnVec(aclsparseConstDnVecDescr_t dnVecDescr)
{
    if (dnVecDescr == nullptr) {
        return ACL_SPARSE_STATUS_SUCCESS;
    }
    delete const_cast<aclsparseDnVecDescr *>(dnVecDescr);
    return ACL_SPARSE_STATUS_SUCCESS;
}

aclsparseStatus_t aclsparseCreateCsr(aclsparseSpMatDescr_t *spMatDescr, int64_t rows, int64_t cols, int64_t nnz,
    void *csrRowOffsets, void *csrColInd, void *csrValues, aclsparseIndexType_t csrRowOffsetsType,
    aclsparseIndexType_t csrColIndType, aclsparseIndexBase_t idxBase, aclDataType valueType)
{
    auto *inner = new (std::nothrow) aclsparseSpMatDescr();
    if (inner == nullptr) {
        return ACL_SPARSE_STATUS_ALLOC_FAILED;
    }

    inner->activeBuffer = nullptr;
    inner->format = ACL_SPARSE_FORMAT_CSR;
    inner->rows = rows;
    inner->cols = cols;
    inner->nnz = nnz;
    inner->ptrs = csrRowOffsets;
    inner->idxs = csrColInd;
    inner->values = csrValues;
    inner->baseType = idxBase;
    inner->ptrType = csrRowOffsetsType;
    inner->IdxType = csrColIndType;
    inner->valueType = valueType;
    *spMatDescr = (aclsparseSpMatDescr_t)inner;
    return ACL_SPARSE_STATUS_SUCCESS;
}

aclsparseStatus_t aclsparseCreateCsc(aclsparseSpMatDescr_t *spMatDescr, int64_t rows, int64_t cols, int64_t nnz,
    void *cscColOffsets, void *cscRowInd, void *cscValues, aclsparseIndexType_t cscColOffsetsType,
    aclsparseIndexType_t cscRowIndType, aclsparseIndexBase_t idxBase, aclDataType valueType)
{
    auto *inner = new (std::nothrow) aclsparseSpMatDescr();
    if (inner == nullptr) {
        return ACL_SPARSE_STATUS_ALLOC_FAILED;
    }

    inner->activeBuffer = nullptr;
    inner->format = ACL_SPARSE_FORMAT_CSC;
    inner->rows = rows;
    inner->cols = cols;
    inner->nnz = nnz;
    inner->ptrs = cscColOffsets;
    inner->idxs = cscRowInd;
    inner->values = cscValues;
    inner->baseType = idxBase;
    inner->ptrType = cscColOffsetsType;
    inner->IdxType = cscRowIndType;
    inner->valueType = valueType;
    *spMatDescr = (aclsparseSpMatDescr_t)inner;
    return ACL_SPARSE_STATUS_SUCCESS;
}

aclsparseStatus_t aclsparseDestroySpMat(aclsparseConstSpMatDescr_t spMatDescr)
{
    if (spMatDescr == nullptr) {
        return ACL_SPARSE_STATUS_SUCCESS;
    }

    delete const_cast<aclsparseSpMatDescr *>(spMatDescr);
    return ACL_SPARSE_STATUS_SUCCESS;
}

// ---- 只读(const)构造接口：内部委托非 const 版本，
// ----   const 数据指针经 const_cast 存入（const 类型本身已禁止把它传给会写入的接口）。
aclsparseStatus_t aclsparseCreateConstDnVec(aclsparseConstDnVecDescr_t *dnVecDescr, int64_t size,
    const void *values, aclDataType valueType)
{
    aclsparseDnVecDescr_t tmp = nullptr;
    aclsparseStatus_t st = aclSparseCreateDnVec(&tmp, size, const_cast<void *>(values), valueType);
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        return st;
    }
    *dnVecDescr = tmp;
    return ACL_SPARSE_STATUS_SUCCESS;
}

aclsparseStatus_t aclsparseCreateConstCsr(aclsparseConstSpMatDescr_t *spMatDescr, int64_t rows, int64_t cols, int64_t nnz,
    const void *csrRowOffsets, const void *csrColInd, const void *csrValues, aclsparseIndexType_t csrRowOffsetsType,
    aclsparseIndexType_t csrColIndType, aclsparseIndexBase_t idxBase, aclDataType valueType)
{
    aclsparseSpMatDescr_t tmp = nullptr;
    aclsparseStatus_t st = aclsparseCreateCsr(&tmp, rows, cols, nnz,
        const_cast<void *>(csrRowOffsets), const_cast<void *>(csrColInd), const_cast<void *>(csrValues),
        csrRowOffsetsType, csrColIndType, idxBase, valueType);
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        return st;
    }
    *spMatDescr = tmp;
    return ACL_SPARSE_STATUS_SUCCESS;
}

aclsparseStatus_t aclsparseCreateConstCsc(aclsparseConstSpMatDescr_t *spMatDescr, int64_t rows, int64_t cols, int64_t nnz,
    const void *cscColOffsets, const void *cscRowInd, const void *cscValues, aclsparseIndexType_t cscColOffsetsType,
    aclsparseIndexType_t cscRowIndType, aclsparseIndexBase_t idxBase, aclDataType valueType)
{
    aclsparseSpMatDescr_t tmp = nullptr;
    aclsparseStatus_t st = aclsparseCreateCsc(&tmp, rows, cols, nnz,
        const_cast<void *>(cscColOffsets), const_cast<void *>(cscRowInd), const_cast<void *>(cscValues),
        cscColOffsetsType, cscRowIndType, idxBase, valueType);
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        return st;
    }
    *spMatDescr = tmp;
    return ACL_SPARSE_STATUS_SUCCESS;
}

aclsparseStatus_t aclSparseSpmvGetBufferSize(aclsparseHandle_t handle, aclsparseOperation_t op, const void *alpha,
    aclsparseConstSpMatDescr_t mat, aclsparseConstDnVecDescr_t x, const void *beta, aclsparseDnVecDescr_t y, aclDataType computeType,
    aclsparseSpMVAlg_t alg, size_t *size)
{
    aclsparseSpMatDescr *matInner = (aclsparseSpMatDescr *)mat;
    *size = GET_SPMV_WOKR_SIZE(matInner->nnz);
    return ACL_SPARSE_STATUS_SUCCESS;
}

aclsparseStatus_t aclSparseSpmvPreprocess(aclsparseHandle_t handle, aclsparseOperation_t op, const void *alpha,
    aclsparseConstSpMatDescr_t mat, aclsparseConstDnVecDescr_t x, const void *beta, aclsparseDnVecDescr_t y, aclDataType computeType,
    aclsparseSpMVAlg_t alg, void *buffer)
{
    aclsparseSpMatDescr *matInner = (aclsparseSpMatDescr *)mat;


    SpmvCsrMat spmvmat(matInner);
    spmvmat.DoPreProcess((uint8_t *)buffer);
    matInner->activeBuffer = buffer;
    return ACL_SPARSE_STATUS_SUCCESS;
}

/* 当前 不支持 alpha 和 beta */
aclsparseStatus_t aclSparseSpmv(aclsparseHandle_t handle, aclsparseOperation_t op, const void *alpha, aclsparseConstSpMatDescr_t mat,
    aclsparseConstDnVecDescr_t x, const void *beta, aclsparseDnVecDescr_t y, aclDataType computeType, aclsparseSpMVAlg_t alg,
    void *buffer)
{
    aclrtStream stream = nullptr;
    aclsparseGetStream(handle, &stream);
    aclsparseSpMatDescr *matInner = (aclsparseSpMatDescr *)mat;
    if (matInner->activeBuffer != buffer) {
        // inactive buffer 或未调用 Preprocess：本次就地构建到该 buffer，但不修改 mat（不设 active）。
        SpmvCsrMat spmvmat(matInner);
        spmvmat.DoPreProcess((uint8_t *)buffer);
    }
    aclsparseDnVecDescr *xInner = (aclsparseDnVecDescr *)x;
    aclsparseDnVecDescr *yInner = (aclsparseDnVecDescr *)y;
    if (matInner->format == ACL_SPARSE_FORMAT_CSC) {
        // CSC and transpose 格式 行列数需要交换
        spmv_kernel_do(buffer, xInner->values, yInner->values, matInner->cols, matInner->rows, matInner->nnz, blockNum, stream);
    } else {
        spmv_kernel_do(buffer, xInner->values, yInner->values, matInner->rows, matInner->cols, matInner->nnz, blockNum, stream);
    }
    CHECK_ACL(aclrtSynchronizeStream(stream));

    return ACL_SPARSE_STATUS_SUCCESS;
}

aclsparseStatus_t aclSparseSpmvShowWorkSpace(aclsparseHandle_t handle, void *buffer)
{
    uint32_t len = 60 * 1024 * 1024;
    uint8_t *ptr = (uint8_t *)malloc(len);
    SpmvCsrInfo *info = (SpmvCsrInfo *)(ptr + GM_SYNC_SIZE);
    CHECK_ACL(aclrtMemcpy(ptr, len, buffer, len, ACL_MEMCPY_DEVICE_TO_HOST));
    uint64_t tlen = GM_SYNC_SIZE + SPMV_CSR_INFO_LEN(info->num);
    printf("############### %s start ##################\r\nnum:    %lu\r\nmbs:    %lu\r\nid\tbnum\tscol\tcnum\tnnz\tmbs\trbl\tcil\tvll\twsl\r\n", __func__, info->num, info->maxBlockSize);
    
    for (uint64_t i = 0; i < info->num; i++) {
        SpmvCsrSubMatInfo *subInfo = info->infos + i;
        printf("%lu\t%lu\t%lu\t%lu\t%lu\t%lu\t%lu\t%lu\t%lu\t%lu\t%p\t%p\t%p\r\n", 
            i, subInfo->blockNum, subInfo->startCol, subInfo->colNum, subInfo->nnz, subInfo->maxBlockSize, 
            subInfo->rowBlockLen, subInfo->colIdxLen, subInfo->valueLen, subInfo->workspaceLen, 
            subInfo->blockPtr, subInfo->colIdx, subInfo->values);
        subInfo->blockPtr = (uint32_t *)(ptr + tlen);
        subInfo->colIdx = (uint32_t *)(ptr + tlen + subInfo->rowBlockLen);
        subInfo->values = (float *)(ptr + tlen + subInfo->rowBlockLen + subInfo->colIdxLen);
        
        for (uint32_t j = 0; j < subInfo->blockNum; j++) {
            uint32_t offset = (j << 1), start = subInfo->blockPtr[offset], rowId = subInfo->blockPtr[offset + 1], end = subInfo->blockPtr[offset + 2];
            if (end - start > subInfo->maxBlockSize || start % (MAX_SUB_ROW_SIZE * 8) != 0)
                printf("i = %lu, j = %d, row = %d start = %d, len = %d\r\n", i, j, rowId, start, end - start);
        }
        for (uint32_t j = 0; j < subInfo->nnz; j++)
            if (subInfo->colIdx[j] > subInfo->colNum)
                printf("i = %lu, j = %d, colIdx = %d\r\n", i, j, subInfo->colIdx[j]);
        tlen += subInfo->workspaceLen;
    }
    
    printf("total Len: %lu\r\n", tlen);
    for (uint64_t i = 0; i < info->num; i++) {
        SpmvCsrSubMatInfo *subInfo = info->infos + i;
        printf("blockptr: \r\n");
        for (uint32_t j = 0; j < 10; j++) {
            uint32_t offset = (j << 1);
            printf("[%d, %d] ", subInfo->blockPtr[offset], subInfo->blockPtr[offset + 1]);
        }
        printf("\r\nvalue:\r\n");
        for (uint32_t j = 0; j < 20; j++) printf("[%f] ", subInfo->values[j]);
        printf("\r\nidx\r\n");
        for (uint32_t j = 0; j < 20; j++) printf("[%d] ", subInfo->colIdx[j]);
        printf("\r\n");
    }
    printf("############### %s end ##################\r\n", __func__);
    free(ptr);
    return ACL_SPARSE_STATUS_SUCCESS;
}