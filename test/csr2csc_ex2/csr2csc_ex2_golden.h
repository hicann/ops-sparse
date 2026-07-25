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

#ifndef TEST_CSR2CSC_EX2_CSR2CSC_EX2_GOLDEN_H_
#define TEST_CSR2CSC_EX2_CSR2CSC_EX2_GOLDEN_H_

#include <cstdint>
#include <vector>

#include "fill.h"

namespace sparse_test {

// ============================================================================
// Golden result structure for csr2csc_ex2: CSR -> CSC format conversion
// ============================================================================

struct Csr2CscGoldenResult {
    std::vector<int32_t> cscColPtr;   // size n+1, in idxBase
    std::vector<int32_t> cscRowInd;   // size nnz, in idxBase
    std::vector<uint8_t> cscVal;      // size nnz * valSize bytes (raw)
    int32_t              nnz;         // total nonzero count
};

// ============================================================================
// CPU golden reference: CSR -> CSC serial conversion
//
// Algorithm:
//   1. Column count: iterate csrColInd, count nonzeros per column
//   2. Column offset: exclusive prefix sum -> cscColPtr
//   3. Scatter: for each nonzero, write to cscRowInd and cscVal by column
//
// This is a pure data-rearrangement operation (no floating-point arithmetic).
// Values are copied as raw bytes, preserving bit-exact representation for
// all dtypes (INT8, FP16, BF16, FP32).
//
// idxBase support: input and output index arrays are offset by idxBase.
// ============================================================================

template <typename T>
inline Csr2CscGoldenResult Csr2CscGolden(
    const CsrMatrix& csr,
    const std::vector<T>& values,
    int idxBase)
{
    int m = static_cast<int>(csr.rows);
    int n = static_cast<int>(csr.cols);
    int nnz = static_cast<int>(csr.nnz);

    Csr2CscGoldenResult result;
    result.nnz = nnz;

    // cscColPtr: size n+1, initialized to idxBase
    result.cscColPtr.assign(n + 1, idxBase);

    if (nnz == 0) {
        return result;
    }

    result.cscRowInd.resize(nnz);
    result.cscVal.resize(static_cast<size_t>(nnz) * sizeof(T));

    // Step 1: Count nonzeros per column
    std::vector<int32_t> colCount(n, 0);
    for (int k = 0; k < nnz; k++) {
        int col = csr.colIndices[k] - idxBase;  // convert to 0-based
        colCount[col]++;
    }

    // Step 2: Exclusive prefix sum -> cscColPtr
    result.cscColPtr[0] = idxBase;
    for (int j = 0; j < n; j++) {
        result.cscColPtr[j + 1] = result.cscColPtr[j] + colCount[j];
    }

    // Step 3: Scatter — place each nonzero into its CSC position
    // colOffset[j] tracks the next write position for column j
    std::vector<int32_t> colOffset(n);
    for (int j = 0; j < n; j++) {
        colOffset[j] = result.cscColPtr[j] - idxBase;  // 0-based offset
    }

    for (int i = 0; i < m; i++) {
        int rowStart = csr.rowOffsets[i] - idxBase;
        int rowEnd = csr.rowOffsets[i + 1] - idxBase;
        for (int k = rowStart; k < rowEnd; k++) {
            int col = csr.colIndices[k] - idxBase;  // 0-based column
            int dest = colOffset[col];
            result.cscRowInd[dest] = i + idxBase;   // write with idxBase
            // Copy value as raw bytes (bit-exact), 按字节显式拷贝（避免 memcpy 类危险函数）
            const uint8_t* srcBytes = reinterpret_cast<const uint8_t*>(&values[k]);
            uint8_t* dstBytes = &result.cscVal[static_cast<size_t>(dest) * sizeof(T)];
            for (size_t b = 0; b < sizeof(T); b++) {
                dstBytes[b] = srcBytes[b];
            }
            colOffset[col]++;
        }
    }

    return result;
}

}  // namespace sparse_test

#endif  // TEST_CSR2CSC_EX2_CSR2CSC_EX2_GOLDEN_H_
