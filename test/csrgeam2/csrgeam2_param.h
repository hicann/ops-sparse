/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TEST_CSRGEAM2_CSRGEAM2_PARAM_H_
#define TEST_CSRGEAM2_CSRGEAM2_PARAM_H_

#include <cstdint>
#include <exception>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

// ============================================================================
// Test parameter structure for aclsparseScsrgeam2
// C = alpha * A + beta * B  (CSR format, FP32)
// ============================================================================

struct CsrGeam2TestParam {
    // Case identification
    std::string case_name;

    // Matrix dimensions
    int m;
    int n;

    // Sparsity (0.0 = dense, 1.0 = empty/nnz=0)
    double sparsity_a;
    double sparsity_b;

    // Random seeds for reproducible data generation
    uint32_t seed_a;
    uint32_t seed_b;

    // Empty row probability for sparse pattern generation
    double empty_row_prob_a;
    double empty_row_prob_b;

    // Scalar coefficients: C = alpha * A + beta * B
    float alpha;
    float beta;

    // Pointer mode: "HOST" or "DEVICE"
    std::string pointer_mode;

    // Index base per matrix (0 or 1)
    int index_base_a;
    int index_base_b;
    int index_base_c;

    // Pattern: "random" (default) or "diag" (diagonal matrix via makeDiagCsr)
    std::string pattern;

    // Precision verification thresholds
    double mere_threshold;    // MERE threshold (FP32 default: 2^-13 ~ 0.000122)
    double mare_multiplier;  // MARE outlier limit = multiplier * MERE
    double abs_threshold;    // Absolute error tolerance for boundary cases

    // Expected result: "SUCCESS" or an error status string
    std::string expect_result;
};

// ============================================================================
// CSV path configuration
// ============================================================================

static std::string GetCsrGeam2CsvPath() {
#ifdef CSRGEAM2_TEST_CSV_PATH
    return CSRGEAM2_TEST_CSV_PATH;
#else
    return "csrgeam2_test.csv";
#endif
}

// ============================================================================
// CSV parsing utilities
// ============================================================================

static std::string TrimStr(const std::string &s) {
    size_t start = s.find_first_not_of(" \t\r\n");
    if (start == std::string::npos) return "";
    size_t end = s.find_last_not_of(" \t\r\n");
    return s.substr(start, end - start + 1);
}

static std::vector<std::string> SplitCsvFields(const std::string &line) {
    std::vector<std::string> fields;
    std::stringstream ss(line);
    std::string field;
    while (std::getline(ss, field, ',')) {
        fields.push_back(field);
    }
    return fields;
}

// ============================================================================
// CSV Loader
// Columns (in order):
//   case_name, m, n, sparsity_a, sparsity_b, seed_a, seed_b,
//   empty_row_prob_a, empty_row_prob_b, alpha, beta, pointer_mode,
//   index_base_a, index_base_b, index_base_c, pattern,
//   mere_threshold, mare_multiplier, abs_threshold, expect_result
// ============================================================================

static std::vector<CsrGeam2TestParam> LoadCsrGeam2CasesFromCsv() {
    std::vector<CsrGeam2TestParam> cases;
    std::string csvPath = GetCsrGeam2CsvPath();
    std::ifstream ifs(csvPath);
    if (!ifs.is_open()) {
        std::cerr << "[CSV] Failed to open: " << csvPath << std::endl;
        return cases;
    }
    std::string line;
    bool isHeader = true;
    while (std::getline(ifs, line)) {
        line = TrimStr(line);
        if (line.empty() || line[0] == '#') continue;
        if (isHeader) { isHeader = false; continue; }
        auto fields = SplitCsvFields(line);
        if (fields.size() < 20) {
            std::cerr << "[CSV] Skipping malformed line (" << fields.size()
                      << " fields, expected 20): " << line << std::endl;
            continue;
        }
        CsrGeam2TestParam p;
        p.case_name = TrimStr(fields[0]);
        try {
            p.m = std::stoi(TrimStr(fields[1])); p.n = std::stoi(TrimStr(fields[2]));
            p.sparsity_a = std::stod(TrimStr(fields[3])); p.sparsity_b = std::stod(TrimStr(fields[4]));
            p.seed_a = static_cast<uint32_t>(std::stoul(TrimStr(fields[5])));
            p.seed_b = static_cast<uint32_t>(std::stoul(TrimStr(fields[6])));
            p.empty_row_prob_a = std::stod(TrimStr(fields[7])); p.empty_row_prob_b = std::stod(TrimStr(fields[8]));
            p.alpha = std::stof(TrimStr(fields[9])); p.beta = std::stof(TrimStr(fields[10]));
            p.pointer_mode     = TrimStr(fields[11]);
            p.index_base_a = std::stoi(TrimStr(fields[12])); p.index_base_b = std::stoi(TrimStr(fields[13]));
            p.index_base_c     = std::stoi(TrimStr(fields[14]));
            p.pattern          = TrimStr(fields[15]);
            p.mere_threshold = std::stod(TrimStr(fields[16])); p.mare_multiplier = std::stod(TrimStr(fields[17]));
            p.abs_threshold    = std::stod(TrimStr(fields[18]));
            p.expect_result    = TrimStr(fields[19]);
        } catch (const std::exception &e) {
            std::cerr << "[CSV] Skipping case '" << p.case_name
                      << "': parse error: " << e.what() << std::endl;
            continue;
        }
        cases.push_back(p);
    }
    std::cout << "[CSV] Loaded " << cases.size() << " test cases from " << csvPath << std::endl;
    return cases;
}

#endif  // TEST_CSRGEAM2_CSRGEAM2_PARAM_H_
