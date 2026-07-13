/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TEST_NNZ_SNNZ_SNNZ_PARAM_H_
#define TEST_NNZ_SNNZ_SNNZ_PARAM_H_

#include "csv_loader.h"
#include <string>

namespace sparse_test {

struct NnzTestParam : public SparseTestParamBase {
    std::string case_name;
    int m = 0;
    int n = 0;
    std::string dir;          // "ROW" or "COLUMN"
    double density = 0.0;
    std::string distribution; // "mixed", "allzero", "allnonzero", "diag", "extreme"
    uint32_t seed = 0;
    std::string pointer_mode; // "DEVICE" or "HOST"

    void fillCustom(const csv_map& row) override {
        case_name    = parseString(row, "case_name");
        m            = parseInt(row, "m");
        n            = parseInt(row, "n");
        dir          = parseString(row, "dir");
        density      = parseDouble(row, "density");
        distribution = parseString(row, "distribution");
        seed         = static_cast<uint32_t>(parseInt(row, "seed"));
        pointer_mode = parseString(row, "pointer_mode");
    }

    std::string caseId() const override { return case_name; }
};

inline void PrintTo(const NnzTestParam& p, std::ostream* os) {
    *os << p.case_name;
}

}  // namespace sparse_test

#endif
