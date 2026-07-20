/**
 * ----------------------------------------------------------------------------------------------------------
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ----------------------------------------------------------------------------------------------------------
 */

// TEMPLATE: 参数结构体模板（与芯片无关）
// 继承 SparseTestParamBase，实现 fillCustom() + caseId()
// 字段按 API 参数顺序排列，数组参数使用 fill.h 中的生成函数

#pragma once

#include <string>
#include "acl/acl.h"
#include "cann_ops_sparse.h"
#include "csv_loader.h"

struct {{Op}}Param : public sparse_test::SparseTestParamBase {
    // TEMPLATE: 按 API 参数顺序声明字段
    // 示例（以 SpMV 为例）：
    int64_t m = 0;
    int64_t n = 0;
    double sparsity = 0.0;
    float alpha = 1.0f;
    float beta = 0.0f;
    bool transpose = false;
    aclDataType computeType = ACL_FLOAT;

    // TEMPLATE: 精度阈值字段（若使用 MERE_MARE 模式）
    float mereThreshold = 1e-5f;
    float mareMultiplier = 10.0f;

    {{Op}}Param(const sparse_test::csv_map& m) : SparseTestParamBase(m) {
        // TEMPLATE: 从 CSV 读取字段值
        // m_    = parseInt(ReadMap(m, "m", "0"));
        // n_    = parseInt(ReadMap(m, "n", "0"));
        // sparsity_ = parseFloat(ReadMap(m, "sparsity", "0.9"));
        // alpha_ = parseFloat(ReadMap(m, "alpha", "1.0"));
        // beta_  = parseFloat(ReadMap(m, "beta", "0.0"));
        // transpose_ = parseBool(ReadMap(m, "transpose", "false"));
        // mereThreshold_ = parseFloat(ReadMap(m, "mere_threshold", "1e-5"));
        // mareMultiplier_ = parseFloat(ReadMap(m, "mare_multiplier", "10.0"));
    }

    // TEMPLATE: caseId — 返回用例的唯一标识（用于 GTest 显示名）
    std::string caseId() const override {
        return caseName;
    }
};
