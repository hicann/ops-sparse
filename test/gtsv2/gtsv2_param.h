/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TEST_GTSV2_GTSV2_PARAM_H_
#define TEST_GTSV2_GTSV2_PARAM_H_

#include "csv_loader.h"
#include <string>
#include <unordered_map>

namespace sparse_test {

// Tridiagonal matrix generation strategy for gtsv2 tests.
//   - DIAG_DOMINANT:    diagonally dominant (no pivoting expected in golden)
//   - RANDOM:           random entries in [value_lo, value_hi] (pivoting likely)
//   - CONSTANT:         all diagonal entries equal to value_lo (off-diagonals 0)
//   - IDENTITY:         identity matrix (off-diagonals 0, diagonal = 1)
//   - SINGULAR:         a zero on the main diagonal (zero-pivot -> Inf/NaN, no protection)
//   --- White-box crafted pivot patterns (deterministic branch coverage) ---
//   - PIVOT_ALL_SWAP:   |dl[i]| > |d'[i-1]| at every forward step -> swap every step
//   - PIVOT_FIRST_ONLY: swap only at step i=1, no swap for i=2..m-1
//   - PIVOT_LAST_ONLY:  no swap for i=1..m-2, swap only at step i=m-1
//   - PIVOT_EXTREME_DL: |dl[i]| >> |d[i]| (1e6:1 ratio) -> extreme swap
//   - PIVOT_EXTREME_D:  |d[i]| >> |dl[i]| (1e6:1 ratio) -> extreme no-swap
enum class Gtsv2MatrixType {
    DIAG_DOMINANT = 0,
    RANDOM = 1,
    CONSTANT = 2,
    IDENTITY = 3,
    SINGULAR = 4,
    PIVOT_ALL_SWAP = 5,
    PIVOT_FIRST_ONLY = 6,
    PIVOT_LAST_ONLY = 7,
    PIVOT_EXTREME_DL = 8,
    PIVOT_EXTREME_D = 9
};

inline Gtsv2MatrixType ParseGtsv2MatrixType(const std::string& s) {
    static const std::unordered_map<std::string, Gtsv2MatrixType> kLookup = {
        {"diag_dominant",    Gtsv2MatrixType::DIAG_DOMINANT},
        {"0",                Gtsv2MatrixType::DIAG_DOMINANT},
        {"random",           Gtsv2MatrixType::RANDOM},
        {"1",                Gtsv2MatrixType::RANDOM},
        {"constant",         Gtsv2MatrixType::CONSTANT},
        {"2",                Gtsv2MatrixType::CONSTANT},
        {"identity",         Gtsv2MatrixType::IDENTITY},
        {"3",                Gtsv2MatrixType::IDENTITY},
        {"singular",         Gtsv2MatrixType::SINGULAR},
        {"4",                Gtsv2MatrixType::SINGULAR},
        {"pivot_all_swap",   Gtsv2MatrixType::PIVOT_ALL_SWAP},
        {"5",                Gtsv2MatrixType::PIVOT_ALL_SWAP},
        {"pivot_first_only", Gtsv2MatrixType::PIVOT_FIRST_ONLY},
        {"6",                Gtsv2MatrixType::PIVOT_FIRST_ONLY},
        {"pivot_last_only",  Gtsv2MatrixType::PIVOT_LAST_ONLY},
        {"7",                Gtsv2MatrixType::PIVOT_LAST_ONLY},
        {"pivot_extreme_dl", Gtsv2MatrixType::PIVOT_EXTREME_DL},
        {"8",                Gtsv2MatrixType::PIVOT_EXTREME_DL},
        {"pivot_extreme_d",  Gtsv2MatrixType::PIVOT_EXTREME_D},
        {"9",                Gtsv2MatrixType::PIVOT_EXTREME_D},
    };
    auto it = kLookup.find(s);
    return (it != kLookup.end()) ? it->second : Gtsv2MatrixType::DIAG_DOMINANT;
}

struct Gtsv2Param : public SparseTestParamBase {
    std::string case_name;

    int m = 0;
    int n = 0;
    int ldb = 0;

    Gtsv2MatrixType matrixType = Gtsv2MatrixType::DIAG_DOMINANT;

    // Value range for random / constant matrix generation
    double value_lo = 0.0;
    double value_hi = 0.0;

    uint32_t seed = 0;

    // Precision tolerances for golden-vs-NPU comparison
    double rtol = 0.0;
    double atol = 0.0;
    double mareMultiplier = 0.0;

    std::string description;
    std::string expect_result;

    void fillCustom(const csv_map& row) override {
        case_name      = parseString(row, "case_name");
        m              = parseInt(row, "m");
        n              = parseInt(row, "n");
        ldb            = parseInt(row, "ldb");
        matrixType     = ParseGtsv2MatrixType(parseString(row, "matrixType"));
        value_lo       = parseDouble(row, "value_lo");
        value_hi       = parseDouble(row, "value_hi");
        seed           = static_cast<uint32_t>(parseInt(row, "seed"));
        rtol           = parseDouble(row, "rtol");
        atol           = parseDouble(row, "atol");
        mareMultiplier = parseDouble(row, "mareMultiplier");
        description    = parseString(row, "description");
        expect_result  = parseString(row, "expect_result");
    }

    std::string caseId() const override { return case_name; }
};

}  // namespace sparse_test

#endif  // TEST_GTSV2_GTSV2_PARAM_H_
