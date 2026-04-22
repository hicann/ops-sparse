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

#ifndef SPARSE_COMMON_H
#define SPARSE_COMMON_H
#include <iostream>
#include <acl/acl.h>
#define CHECK_ACL(x)                                                                        \
    do {                                                                                    \
        aclError __ret = x;                                                                 \
        if (__ret != ACL_ERROR_NONE) {                                                      \
            std::cerr << __FILE__ << ":" << __LINE__ << " aclError:" << __ret << std::endl; \
        }                                                                                   \
    } while (0);

#ifdef __cplusplus
extern "C" {
#endif

#include "spmv_host.h"

#define BLOCK_DIM 20

typedef struct AclSparseSpMatDescInner {
    uint64_t isDoPreProgress;
    AclSparseFormat format;
    uint64_t rows;
    uint64_t cols;
    uint64_t nnz;
    void *ptrs;
    void *idxs;
    void *values;
    AclSparseIndexBase baseType;
    AclSparseIndexType ptrType;
    AclSparseIndexType IdxType;
    AclDataType valueType;
} AclSparseSpMatDescInner;

typedef struct AclSparseDnVecDescInner {
    uint64_t nums;
    void *values;
    AclDataType valueType;
} AclSparseDnVecDescInner;

typedef struct AclSparseHandlerInner {
    int id;
    aclrtStream stream;
} AclSparseHandlerInner;

#ifdef __cplusplus
}
#endif

#endif