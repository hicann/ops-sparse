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
 
#ifndef SPMV_H
#define SPMV_H

#include <stdint.h>
#include <unistd.h>

#ifndef __gm__
#define __gm__
#endif

typedef struct {
     uint64_t blockNum;
     uint64_t startCol;
     uint64_t colNum;
     uint64_t nnz;
     uint64_t maxBlockSize;
     uint64_t rowBlockLen;
     uint64_t colIdxLen;
     uint64_t valueLen;
     uint64_t workspaceLen;
     uint64_t pading[4];
     __gm__ uint32_t *blockPtr;
     __gm__ uint32_t *colIdx;
     __gm__ float *values;
} SpmvCsrSubMatInfo;
typedef struct {
     uint64_t num;
     uint64_t maxBlockSize;
     __gm__ uint8_t *swap;
     __gm__ uint32_t *ptrs;
     __gm__ uint32_t *idxs;
     __gm__ float *values;
     uint64_t pading[2];
     SpmvCsrSubMatInfo infos[0];
} SpmvCsrInfo;

#define ROUNDUP(a, b) (((a) + (b) - 1) / (b) * (b))

#define GM_SYNC_SIZE (48*32)

#define SPMV_CSR_INFO_LEN(n) (sizeof(SpmvCsrInfo) + (n) * sizeof(SpmvCsrSubMatInfo))

#define MAX_SUB_COL_SIZE (16 * 1024)
#define MAX_SUB_ROW_SIZE (16)
#define TRANSPOSE_BLOCK_SIZE (8)

#define GET_SPMV_WOKR_SIZE(nnz)  (2 * (nnz) * sizeof(float) * 4)

#endif