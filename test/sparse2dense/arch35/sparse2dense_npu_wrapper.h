/**
 * ----------------------------------------------------------------------------------------------------------
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under
 * the terms and conditions of CANN Open Software License Agreement Version 2.0
 * (the "License"). Please refer to the License for details. You may not use
 * this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
 * AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
 * FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
 * for the full text of the License.
 * ----------------------------------------------------------------------------------------------------------
 */

#ifndef TEST_SPARSE2DENSE_ARCH35_NPU_WRAPPER_H_
#define TEST_SPARSE2DENSE_ARCH35_NPU_WRAPPER_H_

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "descriptor_manager.h"
#include "sparse2dense_golden.h"
#include "sparse2dense_param.h"

namespace sparse_test {

inline aclDataType Sparse2DenseValueType(const std::string &name) {
    if (name == "INT8")    return ACL_INT8;
    if (name == "FP16")    return ACL_FLOAT16;
    if (name == "BF16")    return ACL_BF16;
    if (name == "INT32")   return ACL_INT32;
    if (name == "FP32")    return ACL_FLOAT;
    if (name == "FP64")    return ACL_DOUBLE;
    return static_cast<aclDataType>(-1);
}

inline aclsparseIndexBase_t Sparse2DenseBase(const std::string &name) {
    return name == "ONE" ? ACL_SPARSE_INDEX_BASE_ONE : ACL_SPARSE_INDEX_BASE_ZERO;
}

inline aclsparseOrder_t Sparse2DenseOrder(const std::string &name) {
    return name == "COL" ? ACL_SPARSE_ORDER_COL : ACL_SPARSE_ORDER_ROW;
}

struct Sparse2DenseHostInput {
    std::vector<uint8_t> sparseOffsets;
    std::vector<uint8_t> sparseIndices;
    std::vector<uint8_t> sparseValues;
    int64_t nnz = 0;
    size_t offsetCount = 0;
    size_t indexCount = 0;
};

inline Sparse2DenseHostInput
MakeSparse2DenseInput(const Sparse2DenseParam &p) {
    const aclDataType type = Sparse2DenseValueType(p.value_type);
    const size_t width = Sparse2DenseElemWidth(type);
    if (width == 0 || p.m < 0 || p.n < 0)
        throw std::invalid_argument("invalid sparse2dense parameter");

    const int64_t base = p.base == "ONE" ? 1 : 0;
    const bool rowMajor = p.order == "ROW";
    const CsrData csr = GenerateCsrGolden(p.m, p.n, p.ld, rowMajor, base,
                                           type, p.distribution, p.seed);
    Sparse2DenseHostInput input;
    input.nnz = csr.nnz;

    if (p.format == "COO") {
        const CooData coo = CsrToCooGolden(csr, p.m, base);
        input.offsetCount = static_cast<size_t>(coo.nnz);
        input.indexCount = static_cast<size_t>(coo.nnz);
        input.sparseOffsets.resize(coo.nnz * sizeof(int32_t));
        input.sparseIndices.resize(coo.nnz * sizeof(int32_t));
        std::copy_n(reinterpret_cast<const uint8_t *>(coo.rowInd.data()),
                    coo.nnz * sizeof(int32_t), input.sparseOffsets.begin());
        std::copy_n(reinterpret_cast<const uint8_t *>(coo.colInd.data()),
                    coo.nnz * sizeof(int32_t), input.sparseIndices.begin());
        input.sparseValues = coo.values;
    } else if (p.format == "CSC") {
        const CscData csc = CsrToCscGolden(csr, p.m, p.n, base, width);
        input.offsetCount = static_cast<size_t>(p.n + 1);
        input.indexCount = static_cast<size_t>(csc.nnz);
        input.sparseOffsets.resize((p.n + 1) * sizeof(int32_t));
        input.sparseIndices.resize(csc.nnz * sizeof(int32_t));
        std::copy_n(reinterpret_cast<const uint8_t *>(csc.colOff.data()),
                    (p.n + 1) * sizeof(int32_t), input.sparseOffsets.begin());
        std::copy_n(reinterpret_cast<const uint8_t *>(csc.rowInd.data()),
                    csc.nnz * sizeof(int32_t), input.sparseIndices.begin());
        input.sparseValues = csc.values;
    } else {
        input.offsetCount = static_cast<size_t>(p.m + 1);
        input.indexCount = static_cast<size_t>(csr.nnz);
        input.sparseOffsets.resize((p.m + 1) * sizeof(int32_t));
        input.sparseIndices.resize(csr.nnz * sizeof(int32_t));
        std::copy_n(reinterpret_cast<const uint8_t *>(csr.rowOff.data()),
                    (p.m + 1) * sizeof(int32_t), input.sparseOffsets.begin());
        std::copy_n(reinterpret_cast<const uint8_t *>(csr.colInd.data()),
                    csr.nnz * sizeof(int32_t), input.sparseIndices.begin());
        input.sparseValues = csr.values;
    }
    return input;
}

struct Sparse2DenseRunResult {
    aclsparseStatus_t descriptorStatus = ACL_SPARSE_STATUS_SUCCESS;
    aclsparseStatus_t queryStatus = ACL_SPARSE_STATUS_SUCCESS;
    aclsparseStatus_t executeStatus = ACL_SPARSE_STATUS_SUCCESS;
    aclError syncStatus = ACL_SUCCESS;
    size_t workspaceSize = 0;
    std::vector<uint8_t> dense;
};

} // namespace sparse_test

#include "sparse2dense_npu_execution.h"
#endif
