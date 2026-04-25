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
#include <string.h>
#include "unistd.h"
#include "spmv.h"
#include "cann_ops_sparse_common.h"
#include "spmv_csr_mat.h"

extern void spmv_kernel_do(void* buffer, void* x, void* y, uint64_t rows, uint64_t cols, uint32_t nnz, uint32_t numBlocks, void *stream);

AclSparseStatus aclSparseCreate(AclSparseHandler *handle)
{
    AclSparseHandlerInner *inner = (AclSparseHandlerInner *)malloc(sizeof(AclSparseHandlerInner));
    if (inner == nullptr) {
        return ACL_SPARSE_STATUS_ALLOC_FAILED;
    }

    aclrtCreateStream(&inner->stream);

    *handle = (AclSparseHandler)inner;
    return ACL_SPARSE_STATUS_SUCCESS;
}

AclSparseStatus aclSparseDestroy(AclSparseHandler handle)
{
    if (handle != nullptr) {
        return ACL_SPARSE_STATUS_SUCCESS;
    }

    AclSparseHandlerInner *inner = (AclSparseHandlerInner *)handle;
    aclrtDestroyStream(inner->stream);

    free(handle);
    return ACL_SPARSE_STATUS_SUCCESS;
}

AclSparseStatus aclSparseCreateDnVec(AclSparseDnVecDesc *dnVecDescr, int64_t size, void *values, aclDataType valueType)
{
    AclSparseDnVecDescInner *inner = (AclSparseDnVecDescInner *)malloc(sizeof(AclSparseDnVecDescInner));
    if (inner == nullptr) {
        return ACL_SPARSE_STATUS_ALLOC_FAILED;
    }

    inner->nums = size;
    inner->values = values;
    inner->valueType = valueType;
    *dnVecDescr = (AclSparseDnVecDesc)inner;
    return ACL_SPARSE_STATUS_SUCCESS;
}

AclSparseStatus aclSparseDestroyDnVec(AclSparseDnVecDesc dnVecDescr)
{
    if (dnVecDescr == nullptr) {
        return ACL_SPARSE_STATUS_SUCCESS;
    }
    free(dnVecDescr);
    return ACL_SPARSE_STATUS_SUCCESS;
}

AclSparseStatus aclSparseCreateCsr(AclSparseSpMatDesc *spMatDescr, int64_t rows, int64_t cols, int64_t nnz,
    void *csrRowOffsets, void *csrColInd, void *csrValues, AclSparseIndexType csrRowOffsetsType,
    AclSparseIndexType csrColIndType, AclSparseIndexBase idxBase, aclDataType valueType)
{
    AclSparseSpMatDescInner *inner = (AclSparseSpMatDescInner *)malloc(sizeof(AclSparseSpMatDescInner));
    if (inner == nullptr) {
        return ACL_SPARSE_STATUS_ALLOC_FAILED;
    }

    inner->isDoPreProgress = false;
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
    *spMatDescr = (AclSparseSpMatDesc)inner;
    return ACL_SPARSE_STATUS_SUCCESS;
}

AclSparseStatus aclSparseCreateCsc(AclSparseSpMatDesc *spMatDescr, int64_t rows, int64_t cols, int64_t nnz,
    void *cscColOffsets, void *cscRowInd, void *cscValues, AclSparseIndexType cscColOffsetsType,
    AclSparseIndexType cscRowIndType, AclSparseIndexBase idxBase, aclDataType valueType)
{
    AclSparseSpMatDescInner *inner = (AclSparseSpMatDescInner *)malloc(sizeof(AclSparseSpMatDescInner));
    if (inner == nullptr) {
        return ACL_SPARSE_STATUS_ALLOC_FAILED;
    }

    inner->isDoPreProgress = false;
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
    *spMatDescr = (AclSparseSpMatDesc)inner;
    return ACL_SPARSE_STATUS_SUCCESS;
}

AclSparseStatus aclSparseDestroySpMat(AclSparseSpMatDesc spMatDescr)
{
    if (spMatDescr == nullptr) {
        return ACL_SPARSE_STATUS_SUCCESS;
    }

    free(spMatDescr);
    return ACL_SPARSE_STATUS_SUCCESS;
}

AclSparseStatus aclSparseSpmvGetBufferSize(AclSparseHandler handle, AclSparseOp op, const void *alpha,
    AclSparseSpMatDesc mat, AclSparseDnVecDesc x, const void *beta, AclSparseDnVecDesc y, aclDataType computeType,
    AclSparseSpmvAlg alg, size_t *size)
{
    AclSparseSpMatDescInner *matInner = (AclSparseSpMatDescInner *)mat;
    *size = GET_SPMV_WOKR_SIZE(matInner->nnz);
    return ACL_SPARSE_STATUS_SUCCESS;
}

AclSparseStatus aclSparseSpmvPreprocess(AclSparseHandler handle, AclSparseOp op, const void *alpha,
    AclSparseSpMatDesc mat, AclSparseDnVecDesc x, const void *beta, AclSparseDnVecDesc y, aclDataType computeType,
    AclSparseSpmvAlg alg, void *buffer)
{
    AclSparseSpMatDescInner *matInner = (AclSparseSpMatDescInner *)mat;


    SpmvCsrMat spmvmat(matInner);
    spmvmat.DoPreProcess((uint8_t *)buffer);
/*
    AclSparseHandlerInner *handleInner = (AclSparseHandlerInner *)handle;
    printf("11 dA_values = %p\r\n", matInner->values);
    ACLRT_LAUNCH_KERNEL(spmv_pre_custom)(20, handleInner->stream, matInner->ptrs, matInner->idxs, matInner->values, buffer,
                        matInner->rows, matInner->cols, matInner->nnz);
    CHECK_ACL(aclrtSynchronizeStream(handleInner->stream));
    */
    matInner->isDoPreProgress = true;
    return ACL_SPARSE_STATUS_SUCCESS;
}

/* 当前 不支持 alpha 和 beta */
AclSparseStatus aclSparseSpmv(AclSparseHandler handle, AclSparseOp op, const void *alpha, AclSparseSpMatDesc mat,
    AclSparseDnVecDesc x, const void *beta, AclSparseDnVecDesc y, aclDataType computeType, AclSparseSpmvAlg alg,
    void *buffer)
{
    AclSparseHandlerInner *handleInner = (AclSparseHandlerInner *)handle;
    AclSparseSpMatDescInner *matInner = (AclSparseSpMatDescInner *)mat;
    if (matInner->isDoPreProgress == false) {
        AclSparseStatus ret = aclSparseSpmvPreprocess(handle, op, alpha, mat, x, beta, y, computeType, alg, buffer);
        if (ret != ACL_SPARSE_STATUS_SUCCESS) {
            return ret;
        }
    }
    AclSparseDnVecDescInner *xInner = (AclSparseDnVecDescInner *)x;
    AclSparseDnVecDescInner *yInner = (AclSparseDnVecDescInner *)y;
    if (matInner->format == ACL_SPARSE_FORMAT_CSC) {
        // CSC and transpose 格式 行列数需要交换
        spmv_kernel_do(buffer, xInner->values, yInner->values, matInner->cols, matInner->rows, matInner->nnz, 20, handleInner->stream);
    } else {
        spmv_kernel_do(buffer, xInner->values, yInner->values, matInner->rows, matInner->cols, matInner->nnz, 20, handleInner->stream);
    }
    CHECK_ACL(aclrtSynchronizeStream(handleInner->stream));

    return ACL_SPARSE_STATUS_SUCCESS;
}

AclSparseStatus aclSparseSpmvShowWorkSpace(AclSparseHandler handle, void *buffer)
{
    uint32_t len = 60 * 1024 * 1024;
    uint8_t *ptr = (uint8_t *)malloc(len);
    SpmvCsrInfo *info = (SpmvCsrInfo *)(ptr + GM_SYNC_SIZE);
    CHECK_ACL(aclrtMemcpy(ptr, len, buffer, len, ACL_MEMCPY_DEVICE_TO_HOST));
    uint64_t tlen = GM_SYNC_SIZE + SPMV_CSR_INFO_LEN(info->num);
    printf("############### %s start ##################\r\n", __func__);
    printf("num:    %lu\r\n", info->num);
    printf("mbs:    %lu\r\n", info->maxBlockSize);
    printf("id\tbnum\tscol\tcnum\tnnz\tmbs\trbl\tcil\tvll\twsl\r\n");
    for (uint64_t i = 0; i < info->num; i++) {
        SpmvCsrSubMatInfo *subInfo = info->infos + i;
        printf("%lu\t%lu\t%lu\t%lu\t%lu\t%lu\t%lu\t%lu\t%lu\t%lu\t%p\t%p\t%p\r\n", 
            i, 
            subInfo->blockNum, 
            subInfo->startCol, 
            subInfo->colNum,
            subInfo->nnz,
            subInfo->maxBlockSize,
            subInfo->rowBlockLen,
            subInfo->colIdxLen,
            subInfo->valueLen,
            subInfo->workspaceLen,
            subInfo->blockPtr,
            subInfo->colIdx,
            subInfo->values);
            subInfo->blockPtr = (uint32_t *)(ptr + tlen);
            subInfo->colIdx = (uint32_t *)(ptr + tlen + subInfo->rowBlockLen);
            subInfo->values = (float *)(ptr + tlen + subInfo->rowBlockLen + subInfo->colIdxLen);
            for (uint32_t j = 0; j < subInfo->blockNum; j++) {
                uint32_t offset = (j << 1);
                uint32_t start = subInfo->blockPtr[offset];
                uint32_t rowId = subInfo->blockPtr[offset + 1];
                uint32_t end = subInfo->blockPtr[offset + 2];
                if (end - start > subInfo->maxBlockSize) {
                    printf("i = %lu, j = %d, row = %d start = %d, len = %d\r\n", i, j, rowId, start, end - start);
                }
                if (start % (MAX_SUB_ROW_SIZE * 8) != 0) {
                    printf("i = %lu, j = %d, row = %d start = %d, len = %d\r\n", i, j, rowId, start, end - start);
                }
            }
            for (uint32_t j = 0; j < subInfo->nnz; j++) {
                if (subInfo->colIdx[j] > subInfo->colNum) {
                    printf("i = %lu, j = %d, colIdx = %d\r\n", i, j, subInfo->colIdx[j]);
                }
            }
          
        tlen += subInfo->workspaceLen;
    }
    printf("total Len: %lu\r\n", tlen);
    for (uint64_t i = 0; i < info->num; i++) {
        SpmvCsrSubMatInfo *subInfo = info->infos + i;
        printf("blockptr: \r\n");
        for (uint32_t j = 0; j < 10; j++) {
                uint32_t offset = (j << 1);
                uint32_t start = subInfo->blockPtr[offset];
                uint32_t rowId = subInfo->blockPtr[offset + 1];
                printf("[%d, %d] ", start, rowId);
        }
        printf("\r\n");
        printf("value:\r\n");
        for (uint32_t j = 0; j < 20; j++) {
                printf("[%f] ", subInfo->values[j]);
        }
        printf("\r\n");
        printf("idx\r\n");
        for (uint32_t j = 0; j < 20; j++) {
                printf("[%d] ", subInfo->colIdx[j]);
        }
        printf("\r\n");
    }

    printf("############### %s end ##################\r\n", __func__);
    free(ptr);
    return ACL_SPARSE_STATUS_SUCCESS;
}