/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TEST_CSR2COO_CSR2COO_PARAM_H_
#define TEST_CSR2COO_CSR2COO_PARAM_H_

#include "csv_loader.h"
#include <string>

namespace sparse_test {

struct Csr2CooParam : public SparseTestParamBase {
    std::string case_name;

    int64_t m = 0;               // 矩阵行数（或 CSC 列数）；传入 API 的 m 参数
    int64_t n = 0;               // 矩阵列数（或 CSC 行数）；仅用于数据生成

    double sparsity = 0.0;
    double empty_row_prob = 0.0;

    uint32_t seed = 0;

    int         idx_base = 0;    // 0 or 1
    std::string pattern;         // "random" / "diag" / "csc"

    double mere_threshold = 0.0;    // unused (INTEGER mode)
    double mare_multiplier = 0.0;   // unused (INTEGER mode)
    double abs_threshold = 0.0;     // unused (INTEGER mode)

    std::string expect_result;   // "SUCCESS" or error code string

    void fillCustom(const csv_map &row) override {
        case_name       = parseString(row, "case_name");
        m               = parseInt(row, "m");
        n               = parseInt(row, "n");
        sparsity        = parseDouble(row, "sparsity");
        empty_row_prob  = parseDouble(row, "empty_row_prob");
        seed            = static_cast<uint32_t>(parseInt(row, "seed"));
        idx_base        = parseInt(row, "idx_base");
        pattern         = parseString(row, "pattern");
        mere_threshold  = parseDouble(row, "mere_threshold");
        mare_multiplier = parseDouble(row, "mare_multiplier");
        abs_threshold   = parseDouble(row, "abs_threshold");
        expect_result   = parseString(row, "expect_result");
    }

    std::string caseId() const override { return case_name; }
};

inline void PrintTo(const Csr2CooParam &p, std::ostream *os) {
    *os << p.case_name;
}

}  // namespace sparse_test

#endif  // TEST_CSR2COO_CSR2COO_PARAM_H_
