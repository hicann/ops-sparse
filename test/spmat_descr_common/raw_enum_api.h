/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_SPARSE_TEST_RAW_ENUM_API_H
#define OPS_SPARSE_TEST_RAW_ENUM_API_H

#include "cann_ops_sparse.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * These wrappers are compiled as C on purpose.  C permits an enum variable to
 * hold any value of its compatible integer type, so passing an out-of-range
 * int through a C translation unit exercises the public C ABI without the
 * undefined behavior of casting an out-of-range value to an enum in C++.
 */
aclsparseStatus_t AclsparseCreateCooRawEnums(aclsparseSpMatDescr_t* spMatDescr, int64_t rows, int64_t cols, int64_t nnz,
                                             void* cooRowInd, void* cooColInd, void* cooValues,
                                             aclsparseIndexType_t cooIdxType, int rawIdxBase, int rawValueType);

aclsparseStatus_t AclsparseCreateCscRawEnums(aclsparseSpMatDescr_t* spMatDescr, int64_t rows, int64_t cols, int64_t nnz,
                                             void* cscColOffsets, void* cscRowInd, void* cscValues,
                                             aclsparseIndexType_t cscColOffsetsType, aclsparseIndexType_t cscRowIndType,
                                             int rawIdxBase, int rawValueType);

aclsparseStatus_t AclsparseCreateSlicedEllRawEnums(aclsparseSpMatDescr_t* spMatDescr, int64_t rows, int64_t cols,
                                                   int64_t nnz, int64_t sliceNnz, int64_t numSlices, void* sellSlicePtr,
                                                   void* sellColInd, void* sellValues, aclsparseIndexType_t sellIdxType,
                                                   int rawIdxBase, int rawValueType);

#ifdef __cplusplus
}
#endif

#endif /* OPS_SPARSE_TEST_RAW_ENUM_API_H */
