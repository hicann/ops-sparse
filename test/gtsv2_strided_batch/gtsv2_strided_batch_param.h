/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TEST_GTSV2_STRIDED_BATCH_GTSV2_STRIDED_BATCH_PARAM_H_
#define TEST_GTSV2_STRIDED_BATCH_GTSV2_STRIDED_BATCH_PARAM_H_

#include "csv_loader.h"
#include <cstdint>
#include <string>

// ============================================================================
// Test parameter structure for aclsparseSgtsv2StridedBatch
//
// Solves a batch of m x m tridiagonal systems A^{(i)} y^{(i)} = x^{(i)}
// in-place (x is overwritten by y). Each batch starts at offset b * batchStride.
//
// CSV columns (in order):
//   case_name, m, batchCount, batchStride, pattern, seed,
//   mere_threshold, mare_multiplier, abs_threshold, expect_result
//
// expect_result semantics:
//   - "SUCCESS"            : normal precision comparison (MERE_MARE)
//   - "SINGULAR"           : singular matrix (d contains 0); skip precision
//                            comparison, only verify SUCCESS return + Inf/NaN
//   - "SUCCESS_NO_OUTPUT"  : batchCount=0 degenerate case; skip precision
//                            comparison, only verify SUCCESS return
// ============================================================================

namespace sparse_test {

struct Gtsv2StridedBatchTestParam : public SparseTestParamBase {
    // Case identification
    std::string case_name;

    // System size per batch (3 <= m <= 2^30; m <= 2048 pure UB path,
    // m > 2048 outer GM blocked-CR path; m > 2^30 -> NOT_SUPPORTED)
    int m = 0;

    // Batch count (>= 0; 0 means early-return SUCCESS without launch)
    int batchCount = 0;

    // Stride between batches in elements (>= ceil(m_pad/8) * 8, DataCopy 32B align)
    int batchStride = 0;

    // Matrix condition pattern:
    //   well_cond / diag_dom / const_diag / mixed_sign / extreme_val / singular
    std::string pattern;

    // Random seed for reproducible dl/d/du/x generation
    uint32_t seed = 0;

    // Precision verification thresholds (MERE_MARE mode)
    double mere_threshold = 0.0001220703125;  // 2^-13, FP32 community default
    double mare_multiplier = 10.0;             // MARE outlier limit = multiplier * threshold
    double abs_threshold = 1e-5;               // reserved column (unused in MERE_MARE mode)

    // Expected result: "SUCCESS" / "SINGULAR" / "SUCCESS_NO_OUTPUT"
    std::string expect_result = "SUCCESS";

    void fillCustom(const csv_map& row) override {
        case_name       = parseString(row, "case_name");
        m               = parseInt(row, "m");
        batchCount      = parseInt(row, "batchCount");
        batchStride     = parseInt(row, "batchStride");
        pattern         = parseString(row, "pattern");
        seed            = static_cast<uint32_t>(parseInt(row, "seed"));
        mere_threshold  = parseDouble(row, "mere_threshold");
        mare_multiplier = parseDouble(row, "mare_multiplier");
        abs_threshold   = parseDouble(row, "abs_threshold");
        expect_result   = parseString(row, "expect_result");
    }

    std::string caseId() const override { return case_name; }
};

}  // namespace sparse_test

#endif  // TEST_GTSV2_STRIDED_BATCH_GTSV2_STRIDED_BATCH_PARAM_H_