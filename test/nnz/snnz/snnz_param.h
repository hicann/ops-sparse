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

#include <cstdint>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

// ============================================================================
// Test parameter structure
// ============================================================================

struct NnzTestParam {
    std::string case_name;
    int m;
    int n;
    std::string dir;          // "ROW" or "COLUMN"
    double density;
    std::string distribution; // "mixed", "allzero", "allnonzero", "diag", "extreme"
    uint32_t seed;
    std::string pointer_mode; // "DEVICE" or "HOST"
};

// ============================================================================
// CSV parsing utilities
// ============================================================================

static std::string GetCsvPath() {
#ifdef SNNZ_TEST_CSV_PATH
    return SNNZ_TEST_CSV_PATH;
#else
    return "snnz_test.csv";
#endif
}

static std::vector<std::string> SplitCsvLine(const std::string &line) {
    std::vector<std::string> fields;
    std::stringstream ss(line);
    std::string field;
    while (std::getline(ss, field, ',')) {
        fields.push_back(field);
    }
    return fields;
}

static std::string TrimWhitespace(const std::string &s) {
    size_t start = s.find_first_not_of(" \t\r\n");
    if (start == std::string::npos) return "";
    size_t end = s.find_last_not_of(" \t\r\n");
    return s.substr(start, end - start + 1);
}

static std::vector<NnzTestParam> LoadTestCasesFromCsv() {
    std::vector<NnzTestParam> cases;
    std::string csvPath = GetCsvPath();
    std::ifstream ifs(csvPath);
    if (!ifs.is_open()) {
        std::cerr << "[CSV] Failed to open: " << csvPath << std::endl;
        return cases;
    }

    std::string line;
    bool isHeader = true;
    while (std::getline(ifs, line)) {
        line = TrimWhitespace(line);
        if (line.empty() || line[0] == '#') continue;
        if (isHeader) {
            isHeader = false;
            continue;
        }
        auto fields = SplitCsvLine(line);
        if (fields.size() < 7) {
            std::cerr << "[CSV] Skipping malformed line: " << line << std::endl;
            continue;
        }
        NnzTestParam p;
        p.case_name = TrimWhitespace(fields[0]);
        p.m = std::stoi(TrimWhitespace(fields[1]));
        p.n = std::stoi(TrimWhitespace(fields[2]));
        p.dir = TrimWhitespace(fields[3]);
        p.density = std::stod(TrimWhitespace(fields[4]));
        p.distribution = TrimWhitespace(fields[5]);
        p.seed = static_cast<uint32_t>(std::stoul(TrimWhitespace(fields[6])));
        p.pointer_mode = (fields.size() > 7) ? TrimWhitespace(fields[7]) : "DEVICE";
        cases.push_back(p);
    }

    std::cout << "[CSV] Loaded " << cases.size() << " test cases from " << csvPath << std::endl;
    return cases;
}

#endif  // TEST_NNZ_SNNZ_SNNZ_PARAM_H_
