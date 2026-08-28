/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TEST_COO_GET_COO_GET_PARAM_H_
#define TEST_COO_GET_COO_GET_PARAM_H_

#include "csv_loader.h"
#include <string>

namespace sparse_test {

/// CooGet 测试参数（由 coo_get_test.csv 加载）。
/// 字段对齐 cases.yaml：rows/cols/nnz/dtype/idx_base/value_range/seed 等。
struct CooGetParam : public SparseTestParamBase {
    std::string case_name;

    int rows = 0;
    int cols = 0;
    int nnz = 0;
    int idx_base = 0;        // 0=ZERO, 1=ONE
    std::string dtype;       // "float32" / "float16"
    double value_lo = -1.0;
    double value_hi = 1.0;
    uint32_t seed = 0;

    std::string expect_result;

    bool isFp16() const { return dtype == "float16" || dtype == "fp16"; }

    void fillCustom(const csv_map &row) override {
        case_name       = parseString(row, "case_name");
        rows            = parseInt(row, "rows");
        cols            = parseInt(row, "cols");
        nnz             = parseInt(row, "nnz");
        idx_base        = parseInt(row, "idx_base");
        dtype           = parseString(row, "dtype");
        value_lo        = parseDouble(row, "value_lo");
        value_hi        = parseDouble(row, "value_hi");
        seed            = static_cast<uint32_t>(parseInt(row, "seed"));
        expect_result   = parseString(row, "expect_result");
    }

    std::string caseId() const override { return case_name; }
};

}  // namespace sparse_test

#endif  // TEST_COO_GET_COO_GET_PARAM_H_
