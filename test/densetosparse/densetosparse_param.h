/**
 * ----------------------------------------------------------------------------------------------------------
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under
 * the terms and conditions of CANN Open Software License Agreement Version 2.0
 * (the "License"). Please refer to the License for details. You may not use
 * this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
 * AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
 * FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
 * for the full text of the License.
 * ----------------------------------------------------------------------------------------------------------
 */

#ifndef TEST_DENSETOSPARSE_PARAM_H_
#define TEST_DENSETOSPARSE_PARAM_H_

#include <cstdint>
#include <string>

#include "csv_loader.h"

namespace sparse_test {

struct DenseToSparseParam : public SparseTestParamBase {
  std::string case_name;
  std::string level;
  std::string mode;
  std::string format;
  std::string offset_type;
  std::string index_type;
  std::string base;
  std::string value_type;
  std::string order;
  std::string distribution;
  int64_t m = 0;
  int64_t n = 0;
  int64_t ld = 0;
  int64_t block_size = 1;
  int64_t ell_cols = 0;
  uint32_t seed = 1;
  int repeat = 1;
  bool non_default_stream = false;

  void fillCustom(const csv_map &row) override {
    case_name = parseString(row, "case_name");
    level = parseString(row, "level");
    mode = parseString(row, "mode");
    format = parseString(row, "format");
    offset_type = parseString(row, "offset_type");
    index_type = parseString(row, "index_type");
    base = parseString(row, "base");
    value_type = parseString(row, "value_type");
    order = parseString(row, "order");
    distribution = parseString(row, "distribution");
    m = std::stoll(parseString(row, "m"));
    n = std::stoll(parseString(row, "n"));
    ld = std::stoll(parseString(row, "ld"));
    block_size = std::stoll(parseString(row, "block_size"));
    ell_cols = std::stoll(parseString(row, "ell_cols"));
    seed = static_cast<uint32_t>(std::stoul(parseString(row, "seed")));
    repeat = parseInt(row, "repeat");
    non_default_stream = parseBool(row, "non_default_stream");
  }

  std::string caseId() const override { return case_name; }
  bool descriptorOnly() const { return mode == "descriptor_only"; }
  bool bufferSizeOnly() const { return mode == "buffer_size_only"; }
};

} // namespace sparse_test
#endif
