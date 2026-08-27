/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * SpGEMM test parameter structure (inherits SparseTestParamBase).
 */

#ifndef TEST_SPGEMM_PARAM_H_
#define TEST_SPGEMM_PARAM_H_

#include <string>
#include "csv_loader.h"
#include "types.h"

namespace sparse_test {

struct SpGEMMParam : public SparseTestParamBase {
    std::string case_name;
    int64_t m = 0;
    int64_t k = 0;
    int64_t n = 0;
    double sparsity_a = 0.5;
    double sparsity_b = 0.5;
    uint32_t seed = 42;
    float alpha = 1.0f;
    float beta = 0.0f;
    std::string dtype = "fp32";
    std::string alg = "DEFAULT";
    double mere_threshold = 1e-4;
    double mare_multiplier = 10.0;
    std::string expect_result = "SUCCESS";

    void fillCustom(const csv_map& row) override {
        case_name = parseString(row, "case_name");
        m = static_cast<int64_t>(parseInt(row, "m"));
        k = static_cast<int64_t>(parseInt(row, "k"));
        n = static_cast<int64_t>(parseInt(row, "n"));
        sparsity_a = parseDouble(row, "sparsity_a");
        sparsity_b = parseDouble(row, "sparsity_b");
        seed = static_cast<uint32_t>(parseInt(row, "seed"));
        alpha = parseFloat(row, "alpha");
        beta = parseFloat(row, "beta");
        dtype = parseString(row, "dtype");
        alg = parseString(row, "alg");
        mere_threshold = parseDouble(row, "mere_threshold");
        mare_multiplier = parseDouble(row, "mare_multiplier");
        expect_result = parseString(row, "expect_result");
    }

    std::string caseId() const override {
        return case_name;
    }

    bool expectSuccess() const {
        return expect_result == "SUCCESS";
    }

    aclDataType aclType() const {
        if (dtype == "fp16")  return ACL_FLOAT16;
        if (dtype == "bf16")  return ACL_BF16;
        return ACL_FLOAT;
    }
};

}  // namespace sparse_test

#endif
