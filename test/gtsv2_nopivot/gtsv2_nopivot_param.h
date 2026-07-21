/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TEST_GTSV2_NOPIVOT_GTSV2_NOPIVOT_PARAM_H_
#define TEST_GTSV2_NOPIVOT_GTSV2_NOPIVOT_PARAM_H_

#include "csv_loader.h"
#include <string>

namespace sparse_test {

struct Gtsv2NopivotParam : public SparseTestParamBase {
    std::string case_name;

    int m = 0;
    int n = 0;
    int ldb = 0;

    int is_diag_dominant = 0;
    double value_lo = 0.0;
    double value_hi = 0.0;

    uint32_t seed = 0;

    std::string expect_result;

    double mere_threshold = 0.0;
    double mare_multiplier = 0.0;

    void fillCustom(const csv_map& row) override {
        case_name        = parseString(row, "case_name");
        m                = parseInt(row, "m");
        n                = parseInt(row, "n");
        ldb              = parseInt(row, "ldb");
        is_diag_dominant = parseInt(row, "is_diag_dominant");
        value_lo         = parseDouble(row, "value_lo");
        value_hi         = parseDouble(row, "value_hi");
        seed             = static_cast<uint32_t>(parseInt(row, "seed"));
        expect_result    = parseString(row, "expect_result");
        mere_threshold   = parseDouble(row, "mere_threshold");
        mare_multiplier  = parseDouble(row, "mare_multiplier");
    }

    std::string caseId() const override { return case_name; }
};

}  // namespace sparse_test

#endif