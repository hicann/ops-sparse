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
#include <stdio.h>
#include <string.h>
#include "spmv.h"
#include "acl/acl.h"
#include "sparse_common.h"
#include "spmv_csr_mat.h"

#define ROUNDUP(a, b) (((a) + (b) - 1) / (b) * (b))
void TranValNdToZZ(float *subValues, float *subBlock, uint64_t rowNum, uint64_t max)
{
    for (uint64_t i = 0; i < rowNum; i++) {
        for (uint64_t j = 0; j < max; j++) {
            subBlock[i * max + j] = subValues[i * max + j];
        }
    }

    uint64_t k = 0;
    for (uint64_t j = 0; j < max; j += TRANSPOSE_BLOCK_SIZE) {
        for (uint64_t i = 0; i < MAX_SUB_ROW_SIZE; i ++) {
            for (uint64_t l = 0; l < TRANSPOSE_BLOCK_SIZE; l++) {
                subValues[k++] = subBlock[i * max + j + l];
            }
        }
    }
}

/* 右矩阵已经转置过，并且行数为16， 所以和左矩阵处理一样 */
void TranIdxNdToZN(uint32_t *subValues, uint32_t *subBlock, uint64_t rowNum, uint64_t max)
{    
    for (uint64_t i = 0; i < rowNum; i++) {
        for (uint64_t j = 0; j < max; j++) {
            subBlock[i * max + j] = subValues[i * max + j];
        }
    }

    uint64_t k = 0;
    for (uint64_t j = 0; j < max; j += TRANSPOSE_BLOCK_SIZE) {
        for (uint64_t i = 0; i < MAX_SUB_ROW_SIZE; i ++) {
            for (uint64_t l = 0; l < TRANSPOSE_BLOCK_SIZE; l++) {
                subValues[k++] = subBlock[i * max + j + l];
            }
        }
    }
}

uint32_t SpmvCsrMat::MallocMatInfoAndCopyFromDev()
{
    ptrs = (uint32_t *)malloc(sizeof(uint32_t) * (rows + 1));
    idxs = (uint32_t *)malloc(sizeof(uint32_t) * (nnz));
    values = (float *)malloc(sizeof(float) * (nnz));
    if (!ptrs || !idxs || !values) {
        FreeMatInfo();
        return ACL_SPARSE_STATUS_ALLOC_FAILED;
    }
    CHECK_ACL(aclrtMemcpy(ptrs,
        sizeof(uint32_t) * (rows + 1),
        matDesc->ptrs,
        sizeof(uint32_t) * (rows + 1),
        ACL_MEMCPY_DEVICE_TO_HOST));
    CHECK_ACL(aclrtMemcpy(idxs,
        sizeof(uint32_t) * (nnz),
        matDesc->idxs,
        sizeof(uint32_t) * (nnz),
        ACL_MEMCPY_DEVICE_TO_HOST));
    CHECK_ACL(aclrtMemcpy(values,
        sizeof(float) * (nnz),
        matDesc->values,
        sizeof(float) * (nnz),
        ACL_MEMCPY_DEVICE_TO_HOST));
    return ACL_SPARSE_STATUS_SUCCESS;
}

void SpmvCsrMat::FreeMatInfo()
{
    if (ptrs) free(ptrs);
    if (idxs) free(idxs);
    if (values) free(values);
}

uint32_t SpmvCsrMat::spmvDivideSubMat(SpmvCsrInfo *infos)
{
    for (uint64_t curId = 0; curId < infos->num; curId ++) {
        SpmvCsrSubMatInfo *subInfo = &infos->infos[curId];
        uint64_t offset = SPMV_CSR_INFO_LEN(infos->num);
        for (uint64_t i = 0; i < curId; i++) {
            offset += infos->infos[i].workspaceLen;
        }
        uint8_t *ptrs = (uint8_t *)infos + offset;
        uint8_t *sptrs = ptrs;
        subInfo->blockPtr = (uint32_t *)(ptrs);
        uint32_t *subBlockPtr = subInfo->blockPtr;
        ptrs += subInfo->rowBlockLen;
        subInfo->colIdx = (uint32_t *)(ptrs);
        uint32_t *subColIdx = subInfo->colIdx;
        ptrs += subInfo->colIdxLen;
        subInfo->values = (float *)(ptrs);
        float *subValues = subInfo->values;
        ptrs += subInfo->valueLen;
        uint8_t *subBlock = (uint8_t *)(ptrs);
        uint64_t subNnzOffset = 0;
        for (uint64_t i = 0; i < rows; i += MAX_SUB_ROW_SIZE) {
            uint64_t start = i;
            uint64_t end = (start + MAX_SUB_ROW_SIZE) > rows ? rows : start + MAX_SUB_ROW_SIZE;

            uint32_t nums[MAX_SUB_ROW_SIZE] = {0};
            GetRowElemNum(start, end - start, nums, curId);
            uint32_t max = GetMaxNum(nums, end - start);
            max = (max + 7) / 8 * 8;
            if (max == 0) {
                continue;
            }
            uint64_t blockLen = max * MAX_SUB_ROW_SIZE;
            subBlockPtr[0] = subNnzOffset;
            subBlockPtr[1] = start;
            GetRowElem(start, end - start, subColIdx, subValues, max, curId);
            TranValNdToZZ(subValues, (float *)subBlock, MAX_SUB_ROW_SIZE, max);
            TranIdxNdToZN(subColIdx, (uint32_t *)subBlock, MAX_SUB_ROW_SIZE, max);
            subBlockPtr += 2;
            subColIdx += blockLen;
            subValues += blockLen;
            subNnzOffset += blockLen;
                if (subBlockPtr > subInfo->blockPtr + (subInfo->rowBlockLen / sizeof(uint32_t))) {
                    printf("blockPtr error\r\n");
                }
                if (subColIdx > subInfo->colIdx + (subInfo->colIdxLen / sizeof(uint32_t))) {
                    printf("colIdxLen error\r\n");
                }
                if (subValues > subInfo->values + (subInfo->valueLen / sizeof(float))) {
                    printf("values error\r\n");
                }
                if (subBlock + blockLen > sptrs + subInfo->workspaceLen) {
                    printf("subBlock error\r\n");
                }
        }
        subBlockPtr[0] = subNnzOffset;
        subBlockPtr[1] = rows;
        if (subNnzOffset != subInfo->nnz) {
            printf("curId = %lu, nnz[%lu %lu]\r\n", curId, subNnzOffset, subInfo->nnz);
        }
        for (uint64_t i = 0; i < subInfo->nnz; i++) {
            if (subInfo->colIdx[i] >= subInfo->colIdxLen) {
                printf("curId = %lu, idx[%lu] = %d\r\n", curId, i, subInfo->colIdx[i]);
            }
        }
    }
    return 0;
}
uint64_t SpmvCsrMat::GetMaxNum(uint32_t *nums, uint64_t num)
{
    uint64_t max = 0;
    for (uint64_t i = 0; i < num; i++) {
        if (nums[i] > max) {
            max = nums[i];
        }
    }

    return max;
}

uint32_t SpmvCsrMat::DoPreProcess(uint8_t *gmworkspace)
{
    uint32_t ret = MallocMatInfoAndCopyFromDev();
    if (ret != ACL_SPARSE_STATUS_SUCCESS) {
        return ret;
    }
    uint64_t size = GET_SPMV_WOKR_SIZE(nnz);
    uint8_t *workspace = (uint8_t *)malloc(size);
    SpmvCsrInfo *info = (SpmvCsrInfo *)(workspace + GM_SYNC_SIZE);
    info->num = ROUNDUP(cols, MAX_SUB_COL_SIZE) / MAX_SUB_COL_SIZE;
    spmvCheckSubMatStats(info);  // 统计子矩阵NNZ等信息，对workspace进行子矩阵内存划分
    spmvDivideSubMat(info);      // 每个子矩阵生成各自的CSR，存入workspace
    spmvDoPost(info, gmworkspace);

    aclrtMemcpy(gmworkspace, size, workspace, size, ACL_MEMCPY_HOST_TO_DEVICE);
    FreeMatInfo();
    free(workspace);

    return 0;
}

uint32_t SpmvCsrMat::spmvDoPost(SpmvCsrInfo *info, uint8_t *gm)
{
    uint8_t *ptr = gm;
    uint64_t len = GM_SYNC_SIZE + SPMV_CSR_INFO_LEN(info->num);
    uint32_t maxBlockSize = 0;
    for (uint64_t id = 0; id < info->num; id++) {
        SpmvCsrSubMatInfo *subInfo = info->infos + id;
        subInfo->blockPtr = (uint32_t *)(ptr + len);
        subInfo->colIdx = (uint32_t *)(ptr + len + subInfo->rowBlockLen);
        subInfo->values = (float *)(ptr + len + subInfo->rowBlockLen + subInfo->colIdxLen);

        len += subInfo->workspaceLen;
        maxBlockSize = (subInfo->maxBlockSize > maxBlockSize) ? subInfo->maxBlockSize : maxBlockSize;
    }
    info->ptrs = (uint32_t *)matDesc->ptrs;
    info->idxs = (uint32_t *)matDesc->idxs;
    info->values = (float *)matDesc->values;
    info->maxBlockSize = maxBlockSize;
    info->swap = ptr + len;
    return 0;
}

uint32_t SpmvCsrMat::spmvCheckSubMatStats(SpmvCsrInfo *info)
{
    for (uint64_t curId = 0; curId < info->num; curId++) {
        uint64_t startColNum = curId * MAX_SUB_COL_SIZE;
        SpmvCsrSubMatInfo *subInfo = &info->infos[curId];
        uint64_t maxBlockLen = 0;
        uint64_t subNnz = 0;
        uint64_t blockNum = 0;
        for (uint64_t i = 0; i < rows; i += MAX_SUB_ROW_SIZE) {
            uint64_t start = i;
            uint64_t end = (start + MAX_SUB_ROW_SIZE) > rows ? rows : start + MAX_SUB_ROW_SIZE;

            uint32_t nums[MAX_SUB_ROW_SIZE] = {0};
            GetRowElemNum(start, end - start, nums, curId);
            uint32_t max = GetMaxNum(nums, end - start);
            max = ROUNDUP(max, 8);  // 方便aic和aiv 处理，进行32B对齐
            if (max > 0) {          // 排除全0子矩阵
                blockNum += 1;
                uint64_t blockLen = max * MAX_SUB_ROW_SIZE;
                subNnz += blockLen;
                maxBlockLen = blockLen > maxBlockLen ? blockLen : maxBlockLen;
            }
        }
        uint64_t rowBlockLen = ROUNDUP((blockNum + 1) * 2 * sizeof(uint32_t), 64);
        uint64_t colIdxLen = ROUNDUP(subNnz * sizeof(uint32_t), 64);
        uint64_t valueLen = ROUNDUP(subNnz * sizeof(float), 64);
        subInfo->blockNum = blockNum;
        subInfo->startCol = startColNum;
        subInfo->colNum = (curId == info->num - 1) ? cols - startColNum : MAX_SUB_COL_SIZE;
        subInfo->maxBlockSize = maxBlockLen;
        subInfo->nnz = subNnz;
        subInfo->workspaceLen = rowBlockLen + colIdxLen + valueLen + maxBlockLen * sizeof(uint32_t) * 2 + 128;
        subInfo->rowBlockLen = rowBlockLen;
        subInfo->colIdxLen = colIdxLen;
        subInfo->valueLen = valueLen;
    }

    return 0;
}

void SpmvCsrMat::GetRowElem(uint64_t startRow, uint64_t rowNum, uint32_t *idx, float *vals, uint64_t max, uint64_t curId)
{
    uint64_t i = 0;
    for (i = 0; i < rowNum; i++) {
        uint64_t start = ptrs[startRow + i];
        uint64_t end = ptrs[startRow + i + 1];
        uint64_t j = 0;
        uint64_t k = 0;
        for (j = start; j < end; j++) {
            if (idxs[j] / MAX_SUB_COL_SIZE != curId) {
                continue;
            }
            idx[i * max + k] = idxs[j] % MAX_SUB_COL_SIZE;
            vals[i * max + k] = this->values[j];
            k++;
        }

        for (; k < max; k++) {
            idx[i * max + k] = 0;
            vals[i * max + k] = 0;
        }
    }

    for (; i < MAX_SUB_ROW_SIZE; i++) {
        for (uint64_t j = 0; j < max; j++) {
            idx[i * max + j] = 0;
            vals[i * max + j] = 0;
        }
    }
}

void SpmvCsrMat::GetRowElemNum(uint64_t startRow, uint64_t rowNum, uint32_t *nums, uint64_t curId)
{
    for (uint64_t i = 0; i < rowNum; i++) {
        uint64_t start = ptrs[startRow + i];
        uint64_t end = ptrs[startRow + i + 1];
        nums[i] = 0;
        for (uint64_t j = start; j < end; j++) {
            if (idxs[j] / MAX_SUB_COL_SIZE != curId) {
                continue;
            }
            nums[i]++;
        }
    }
}