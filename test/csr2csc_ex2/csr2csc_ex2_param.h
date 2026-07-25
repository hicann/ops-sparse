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

#ifndef TEST_CSR2CSC_EX2_CSR2CSC_EX2_PARAM_H_
#define TEST_CSR2CSC_EX2_CSR2CSC_EX2_PARAM_H_

#include "csv_loader.h"
#include <string>

namespace sparse_test {

struct Csr2CscEx2Param : public SparseTestParamBase {
    std::string case_name;

    int m = 0;
    int n = 0;

    double sparsity = 0.0;
    double empty_row_prob = 0.0;

    uint32_t seed = 0;

    std::string val_type;      // "INT8" / "FP16" / "BF16" / "FP32"
    std::string copy_values;   // "SYMBOLIC" / "NUMERIC"
    int idx_base = 0;          // 0 or 1

    std::string pattern;       // "random" / "diag"

    double mere_threshold = 0.0;
    double mare_multiplier = 0.0;
    double abs_threshold = 0.0;

    std::string expect_result;

    void fillCustom(const csv_map& row) override {
        case_name         = parseString(row, "case_name");
        m                 = parseInt(row, "m");
        n                 = parseInt(row, "n");
        sparsity          = parseDouble(row, "sparsity");
        empty_row_prob    = parseDouble(row, "empty_row_prob");
        seed              = static_cast<uint32_t>(parseInt(row, "seed"));
        val_type          = parseString(row, "val_type");
        copy_values       = parseString(row, "copy_values");
        idx_base          = parseInt(row, "idx_base");
        pattern           = parseString(row, "pattern");
        mere_threshold    = parseDouble(row, "mere_threshold");
        mare_multiplier   = parseDouble(row, "mare_multiplier");
        abs_threshold     = parseDouble(row, "abs_threshold");
        expect_result     = parseString(row, "expect_result");
    }

    std::string caseId() const override { return case_name; }
};

inline void PrintTo(const Csr2CscEx2Param& p, std::ostream* os) {
    *os << p.case_name;
}

}  // namespace sparse_test

#endif
