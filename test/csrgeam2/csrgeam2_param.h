/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TEST_CSRGEAM2_CSRGEAM2_PARAM_H_
#define TEST_CSRGEAM2_CSRGEAM2_PARAM_H_

#include "csv_loader.h"
#include <string>

namespace sparse_test {

struct CsrGeam2TestParam : public SparseTestParamBase {
    std::string case_name;

    int m = 0;
    int n = 0;

    double sparsity_a = 0.0;
    double sparsity_b = 0.0;

    uint32_t seed_a = 0;
    uint32_t seed_b = 0;

    double empty_row_prob_a = 0.0;
    double empty_row_prob_b = 0.0;

    float alpha = 0.0f;
    float beta = 0.0f;

    std::string pointer_mode;  // "HOST" or "DEVICE"

    int index_base_a = 0;
    int index_base_b = 0;
    int index_base_c = 0;

    std::string pattern;       // "random" or "diag"

    double mere_threshold = 0.0;
    double mare_multiplier = 0.0;
    double abs_threshold = 0.0;

    std::string expect_result;

    void fillCustom(const csv_map& row) override {
        case_name          = parseString(row, "case_name");
        m                  = parseInt(row, "m");
        n                  = parseInt(row, "n");
        sparsity_a         = parseDouble(row, "sparsity_a");
        sparsity_b         = parseDouble(row, "sparsity_b");
        seed_a             = static_cast<uint32_t>(parseInt(row, "seed_a"));
        seed_b             = static_cast<uint32_t>(parseInt(row, "seed_b"));
        empty_row_prob_a   = parseDouble(row, "empty_row_prob_a");
        empty_row_prob_b   = parseDouble(row, "empty_row_prob_b");
        alpha              = parseFloat(row, "alpha");
        beta               = parseFloat(row, "beta");
        pointer_mode       = parseString(row, "pointer_mode");
        index_base_a       = parseInt(row, "index_base_a");
        index_base_b       = parseInt(row, "index_base_b");
        index_base_c       = parseInt(row, "index_base_c");
        pattern            = parseString(row, "pattern");
        mere_threshold     = parseDouble(row, "mere_threshold");
        mare_multiplier    = parseDouble(row, "mare_multiplier");
        abs_threshold      = parseDouble(row, "abs_threshold");
        expect_result      = parseString(row, "expect_result");
    }

    std::string caseId() const override { return case_name; }
};

inline void PrintTo(const CsrGeam2TestParam& p, std::ostream* os) {
    *os << p.case_name;
}

}  // namespace sparse_test

#endif
