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

#ifndef TEST_GEBSR2GEBSC_GEBSR2GEBSC_PARAM_H_
#define TEST_GEBSR2GEBSC_GEBSR2GEBSC_PARAM_H_

#include "csv_loader.h"
#include <string>

namespace sparse_test {

struct Gebsr2GebscParam : public SparseTestParamBase {
    std::string case_name;

    int mb = 0;
    int nb = 0;

    double sparsity = 0.0;
    double empty_row_prob = 0.0;

    uint32_t seed = 0;

    std::string val_type;      // "S" / "D" / "C" / "Z"
    std::string copy_values;   // "SYMBOLIC" / "NUMERIC"
    int idx_base = 0;          // 0 or 1

    int row_block_dim_a = 1;
    int col_block_dim_a = 1;
    int row_block_dim_c = 1;
    int col_block_dim_c = 1;
    std::string dir_a;         // "ROW" / "COLUMN"

    std::string pattern;       // "random" / "diag"

    std::string expect_result;

    void fillCustom(const csv_map& row) override {
        case_name         = parseString(row, "case_name");
        mb                = parseInt(row, "mb");
        nb                = parseInt(row, "nb");
        sparsity          = parseDouble(row, "sparsity");
        empty_row_prob    = parseDouble(row, "empty_row_prob");
        seed              = static_cast<uint32_t>(parseInt(row, "seed"));
        val_type          = parseString(row, "val_type");
        copy_values       = parseString(row, "copy_values");
        idx_base          = parseInt(row, "idx_base");
        row_block_dim_a   = parseInt(row, "row_block_dim_a");
        col_block_dim_a   = parseInt(row, "col_block_dim_a");
        row_block_dim_c   = parseInt(row, "row_block_dim_c");
        col_block_dim_c   = parseInt(row, "col_block_dim_c");
        dir_a             = parseString(row, "dir_a");
        pattern           = parseString(row, "pattern");
        expect_result     = parseString(row, "expect_result");
    }

    std::string caseId() const override { return case_name; }
};

inline void PrintTo(const Gebsr2GebscParam& p, std::ostream* os) {
    *os << p.case_name;
}

}  // namespace sparse_test

#endif
