/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * SpMVOp test parameter structure (inherits SparseTestParamBase).
 */

#ifndef TEST_SPMV_OP_PARAM_H_
#define TEST_SPMV_OP_PARAM_H_

#include <string>
#include "csv_loader.h"
#include "types.h"

namespace sparse_test {

struct SpMVOpParam : public SparseTestParamBase {
    std::string case_name;
    int64_t m = 0;
    int64_t k = 0;
    double sparsity = 0.5;
    double empty_row_prob = 0.0;
    uint32_t seed = 42;
    float alpha = 1.0f;
    float beta = 0.0f;
    std::string pointer_mode = "DEVICE";
    std::string row_offset_type = "i32";
    int alias_z_y = 0;
    int unsorted = 0;
    std::string alg = "DEFAULT";
    std::string pattern = "random";
    double mere_threshold = 1e-6;
    double mare_multiplier = 5.0;
    double abs_threshold = 1e-6;
    std::string expect_result = "SUCCESS";

    void fillCustom(const csv_map& row) override {
        case_name = parseString(row, "case_name");
        m = static_cast<int64_t>(parseInt(row, "m"));
        k = static_cast<int64_t>(parseInt(row, "k"));
        sparsity = parseDouble(row, "sparsity");
        empty_row_prob = parseDouble(row, "empty_row_prob");
        seed = static_cast<uint32_t>(parseInt(row, "seed"));
        alpha = parseFloat(row, "alpha");
        beta = parseFloat(row, "beta");
        pointer_mode = parseString(row, "pointer_mode");
        row_offset_type = parseString(row, "row_offset_type");
        alias_z_y = parseInt(row, "alias_z_y");
        unsorted = parseInt(row, "unsorted");
        alg = parseString(row, "alg");
        pattern = parseString(row, "pattern");
        mere_threshold = parseDouble(row, "mere_threshold");
        mare_multiplier = parseDouble(row, "mare_multiplier");
        abs_threshold = parseDouble(row, "abs_threshold");
        expect_result = parseString(row, "expect_result");
    }

    std::string caseId() const override {
        return case_name;
    }

    bool expectSuccess() const {
        return expect_result == "SUCCESS";
    }
};

}  // namespace sparse_test

#endif
