/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TEST_CSR2GEBSR_CSR2GEBSR_PARAM_H_
#define TEST_CSR2GEBSR_CSR2GEBSR_PARAM_H_

#include "csv_loader.h"
#include <cstdint>
#include <string>

namespace sparse_test {

struct Csr2GebsrTestParam : public SparseTestParamBase {
    std::string case_name;

    int m = 0;
    int n = 0;

    double sparsity = 0.0;
    double empty_row_prob = 0.0;

    uint32_t seed = 0;

    std::string dtype;        // FP32 / FP16 / BF16 / INT32
    int row_block_dim = 1;
    int col_block_dim = 1;
    std::string dir;          // ROW / COLUMN
    int index_base_a = 0;
    int index_base_c = 0;

    void fillCustom(const csv_map& row) override {
        case_name        = parseString(row, "case_name");
        m                = parseInt(row, "m");
        n                = parseInt(row, "n");
        sparsity         = parseDouble(row, "sparsity");
        empty_row_prob   = parseDouble(row, "empty_row_prob");
        seed             = static_cast<uint32_t>(parseInt(row, "seed"));
        dtype            = parseString(row, "dtype");
        row_block_dim    = parseInt(row, "row_block_dim");
        col_block_dim    = parseInt(row, "col_block_dim");
        dir              = parseString(row, "dir");
        index_base_a     = parseInt(row, "index_base_a");
        index_base_c     = parseInt(row, "index_base_c");
    }

    std::string caseId() const override { return case_name; }
};

inline void PrintTo(const Csr2GebsrTestParam& p, std::ostream* os) {
    *os << p.case_name;
}

}  // namespace sparse_test

#endif
