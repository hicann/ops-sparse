/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TEST_GTSV_INTERLEAVED_BATCH_PARAM_H_
#define TEST_GTSV_INTERLEAVED_BATCH_PARAM_H_

#include "csv_loader.h"
#include <sstream>
#include <string>

namespace sparse_test {

struct GtsvInterleavedBatchParam : public SparseTestParamBase {
    std::string case_name;
    int m = 0;
    int batch_count = 0;
    int algo = 0;
    int is_diag_dominant = 0;
    uint32_t seed = 0;
    double value_lo = 0.0;
    double value_hi = 0.0;
    std::string expect_result;
    double mere_threshold = 0.0;
    double mare_multiplier = 0.0;

    void fillCustom(const csv_map& row) override {
        case_name        = parseString(row, "case_name");
        m                = parseInt(row, "m");
        batch_count      = parseInt(row, "batch_count");
        algo             = parseInt(row, "algo");
        is_diag_dominant = parseInt(row, "is_diag_dominant");
        seed             = static_cast<uint32_t>(parseInt(row, "seed"));
        value_lo         = parseDouble(row, "value_lo");
        value_hi         = parseDouble(row, "value_hi");
        expect_result    = parseString(row, "expect_result");
        mere_threshold   = parseDouble(row, "mere_threshold");
        mare_multiplier  = parseDouble(row, "mare_multiplier");
    }

    std::string caseId() const override { return case_name; }
};

// Print human-readable case info for GTest output
inline std::string PrintCaseInfo(const GtsvInterleavedBatchParam& p) {
    std::ostringstream ss;
    ss << p.case_name
       << "_m" << p.m
       << "_b" << p.batch_count
       << "_algo" << p.algo;
    return ss.str();
}

// Required by GTest for parameterized test output
inline void PrintTo(const GtsvInterleavedBatchParam& p, std::ostream* os) {
    *os << PrintCaseInfo(p);
}

}  // namespace sparse_test

#endif
