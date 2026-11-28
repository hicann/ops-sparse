/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*
 * C-side wrappers for the sparse descriptor create APIs.  Keep this file
 * compiled as C: passing an out-of-range enum value through a C translation
 * unit is well-defined by the C ABI, while the equivalent C++ cast is not.
 */
#include "raw_enum_api.h"

aclsparseStatus_t AclsparseCreateCooRawEnums(aclsparseSpMatDescr_t* spMatDescr, int64_t rows, int64_t cols, int64_t nnz,
                                             void* cooRowInd, void* cooColInd, void* cooValues,
                                             aclsparseIndexType_t cooIdxType, int rawIdxBase, int rawValueType)
{
    return aclsparseCreateCoo(spMatDescr, rows, cols, nnz, cooRowInd, cooColInd, cooValues, cooIdxType,
                              (aclsparseIndexBase_t)rawIdxBase, (aclDataType)rawValueType);
}

aclsparseStatus_t AclsparseCreateCscRawEnums(aclsparseSpMatDescr_t* spMatDescr, int64_t rows, int64_t cols, int64_t nnz,
                                             void* cscColOffsets, void* cscRowInd, void* cscValues,
                                             aclsparseIndexType_t cscColOffsetsType, aclsparseIndexType_t cscRowIndType,
                                             int rawIdxBase, int rawValueType)
{
    return aclsparseCreateCsc(spMatDescr, rows, cols, nnz, cscColOffsets, cscRowInd, cscValues, cscColOffsetsType,
                              cscRowIndType, (aclsparseIndexBase_t)rawIdxBase, (aclDataType)rawValueType);
}

aclsparseStatus_t AclsparseCreateSlicedEllRawEnums(aclsparseSpMatDescr_t* spMatDescr, int64_t rows, int64_t cols,
                                                   int64_t nnz, int64_t sliceNnz, int64_t numSlices, void* sellSlicePtr,
                                                   void* sellColInd, void* sellValues, aclsparseIndexType_t sellIdxType,
                                                   int rawIdxBase, int rawValueType)
{
    return aclsparseCreateSlicedEll(spMatDescr, rows, cols, nnz, sliceNnz, numSlices, sellSlicePtr, sellColInd,
                                    sellValues, sellIdxType, (aclsparseIndexBase_t)rawIdxBase,
                                    (aclDataType)rawValueType);
}
