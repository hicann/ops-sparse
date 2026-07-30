/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once

#include "acl/acl_base_rt.h"
#include "csv_loader.h"
#include "cann_ops_sparse.h"
#include <map>
#include <string>

namespace sparse_test {

struct GatherTestParam : public SparseTestParamBase {
    std::string case_name;
    int vec_size = 0;
    int nnz = 0;
    aclDataType value_type;
    aclsparseIndexBase_t idx_base;
    aclsparseIndexType_t idx_type;
    int seed = 0;

    void fillCustom(const csv_map& row) override
    {
        case_name = parseString(row, "case_name");
        vec_size = parseInt(row, "vec_size");
        nnz = parseInt(row, "nnz");
        value_type = std::map<std::string, aclDataType>{
            {"FLOAT", ACL_FLOAT},
            {"FLOAT16", ACL_FLOAT16},
            {"BF16", ACL_BF16},
            {"DOUBLE", ACL_DOUBLE},
        }[parseString(row, "value_type")];
        idx_base = std::map<std::string, aclsparseIndexBase_t>{
            {"ZERO", ACL_SPARSE_INDEX_BASE_ZERO}, {"ONE", ACL_SPARSE_INDEX_BASE_ONE}}[parseString(row, "idx_base")];
        idx_type = std::map<std::string, aclsparseIndexType_t>{
            {"32I", ACL_SPARSE_INDEX_32I}, {"64I", ACL_SPARSE_INDEX_64I}}[parseString(row, "idx_type")];
        seed = parseInt(row, "seed");
    }

    std::string caseId() const override { return case_name; }
};

} // namespace sparse_test
