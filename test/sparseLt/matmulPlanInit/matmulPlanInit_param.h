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

/**
 * @file matmulPlanInit_param.h
 * @brief CSV 行 -> 参数结构体，Plan/AlgSelection 测试专用。
 *
 * 自包含设计：不依赖 matmulDescriptorInit 的测试头文件。
 * 枚举映射从 cann_ops_sparseLt.h 公共 API 推导，非从 descriptor init 测试代码继承。
 */

#ifndef TEST_SPARSELT_MATMULPLANINIT_PARAM_H_
#define TEST_SPARSELT_MATMULPLANINIT_PARAM_H_

#include "cann_ops_sparseLt.h"

#include "csv_loader.h"

#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <string>

namespace sparse_test
{

// ============================================================================
// int64_t 解析
// ============================================================================
inline int64_t PlanParseInt64(const csv_map& row, const std::string& key)
{
    auto it = row.find(key);
    if (it == row.end() || it->second.empty())
        return 0;
    const std::string& s = it->second;
    errno = 0;
    char* endptr = nullptr;
    long long val = std::strtoll(s.c_str(), &endptr, 10);
    if (endptr == s.c_str() || *endptr != '\0' || errno == ERANGE)
    {
        return INT64_MIN;
    }
    return static_cast<int64_t>(val);
}

// ============================================================================
// 枚举字符串映射
// ============================================================================

constexpr int32_t PLAN_INVALID_ENUM_SENTINEL = 999;

inline aclDataType PlanParseDtype(const std::string& s)
{
    if (s == "FP32")
        return ACL_FLOAT;
    if (s == "FP16")
        return ACL_FLOAT16;
    if (s == "BF16")
        return ACL_BF16;
    if (s == "INT8")
        return ACL_INT8;
    if (s == "FP8_E4M3")
        return ACL_FLOAT8_E4M3FN;
    if (s == "FP8_E5M2")
        return ACL_FLOAT8_E5M2;
    if (s == "FP4_E2M1")
        return ACL_FLOAT4_E2M1;
    if (s.empty())
        return ACL_DT_UNDEFINED;
    return static_cast<aclDataType>(PLAN_INVALID_ENUM_SENTINEL);
}

inline aclsparseOrder_t PlanParseOrder(const std::string& s)
{
    if (s == "COL")
        return ACL_SPARSE_ORDER_COL;
    if (s == "ROW")
        return ACL_SPARSE_ORDER_ROW;
    if (s.empty())
        return ACL_SPARSE_ORDER_ROW;
    return static_cast<aclsparseOrder_t>(PLAN_INVALID_ENUM_SENTINEL);
}

inline aclsparseOperation_t PlanParseOp(const std::string& s)
{
    if (s == "N")
        return ACL_SPARSE_OP_NON_TRANSPOSE;
    if (s == "T")
        return ACL_SPARSE_OP_TRANSPOSE;
    if (s == "CONJ")
        return ACL_SPARSE_OP_CONJUGATE_TRANSPOSE;
    if (s.empty())
        return ACL_SPARSE_OP_NON_TRANSPOSE;
    return static_cast<aclsparseOperation_t>(PLAN_INVALID_ENUM_SENTINEL);
}

inline aclsparseLtSparsity_t PlanParseSparsity(const std::string& s)
{
    if (s == "50_PERCENT")
        return ACL_SPARSE_LT_SPARSITY_50_PERCENT;
    if (s.empty())
        return ACL_SPARSE_LT_SPARSITY_50_PERCENT;
    return static_cast<aclsparseLtSparsity_t>(PLAN_INVALID_ENUM_SENTINEL);
}

inline aclsparseComputeType_t PlanParseComputeType(const std::string& s)
{
    if (s == "16F")
        return ACL_SPARSE_COMPUTE_16F;
    if (s == "32F")
        return ACL_SPARSE_COMPUTE_32F;
    if (s == "32I")
        return ACL_SPARSE_COMPUTE_32I;
    if (s.empty())
        return ACL_SPARSE_COMPUTE_16F;
    return static_cast<aclsparseComputeType_t>(PLAN_INVALID_ENUM_SENTINEL);
}

inline aclsparseLtMatmulAlg_t PlanParseAlg(const std::string& s)
{
    if (s == "DEFAULT")
        return ACL_SPARSE_LT_MATMUL_ALG_DEFAULT;
    if (s.empty())
        return ACL_SPARSE_LT_MATMUL_ALG_DEFAULT;
    return static_cast<aclsparseLtMatmulAlg_t>(PLAN_INVALID_ENUM_SENTINEL);
}

constexpr int64_t PLAN_STATUS_SKIP_SENTINEL = -911;

inline int64_t PlanParseStatus(const std::string& s)
{
    if (s == "SUCCESS")
        return static_cast<int64_t>(ACL_SPARSE_STATUS_SUCCESS);
    if (s == "INVALID_VALUE")
        return static_cast<int64_t>(ACL_SPARSE_STATUS_INVALID_VALUE);
    if (s == "NOT_SUPPORTED")
        return static_cast<int64_t>(ACL_SPARSE_STATUS_NOT_SUPPORTED);
    if (s == "HANDLE_IS_NULLPTR")
        return static_cast<int64_t>(ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR);
    if (s == "ALLOC_FAILED")
        return static_cast<int64_t>(ACL_SPARSE_STATUS_ALLOC_FAILED);
    if (s == "SKIP")
        return PLAN_STATUS_SKIP_SENTINEL;
    return static_cast<int64_t>(ACL_SPARSE_STATUS_INTERNAL_ERROR);
}

inline aclsparseStatus_t PlanExpectedStatus(int64_t v)
{
    return static_cast<aclsparseStatus_t>(v);
}

// ============================================================================
// 单个矩阵描述符参数
// ============================================================================

struct PlanMatParam
{
    int64_t rows = 0;
    int64_t cols = 0;
    int64_t ld = 0;
    uint32_t align = 0;
    aclDataType dtype = ACL_DT_UNDEFINED;
    aclsparseOrder_t order = ACL_SPARSE_ORDER_ROW;
    aclsparseLtSparsity_t sparsity = ACL_SPARSE_LT_SPARSITY_50_PERCENT;
    bool isStructured = false;
    bool valid = false;

    void fill(const csv_map& row, const std::string& prefix)
    {
        rows = PlanParseInt64(row, prefix + "rows");
        cols = PlanParseInt64(row, prefix + "cols");
        ld = PlanParseInt64(row, prefix + "ld");
        align = static_cast<uint32_t>(parseInt(row, prefix + "align"));
        dtype = PlanParseDtype(parseString(row, prefix + "dtype"));
        order = PlanParseOrder(parseString(row, prefix + "order"));
        std::string sp = parseString(row, prefix + "sparse");
        if (!sp.empty())
        {
            isStructured = true;
            sparsity = PlanParseSparsity(sp);
        }
        valid = !parseString(row, prefix + "dtype").empty();
    }
};

// ============================================================================
// 测试参数结构体
// ============================================================================

struct MatmulPlanInitParam : public SparseTestParamBase
{
    std::string case_name;
    std::string level;        // L0/L1/L2
    std::string target;       // algSelection / plan
    int64_t expectStatus = 0;
    int verifyFields = 0;

    aclsparseLtMatmulAlg_t alg = ACL_SPARSE_LT_MATMUL_ALG_DEFAULT;

    PlanMatParam a;
    PlanMatParam b;
    PlanMatParam c;
    PlanMatParam d;

    aclsparseOperation_t opA = ACL_SPARSE_OP_NON_TRANSPOSE;
    aclsparseOperation_t opB = ACL_SPARSE_OP_NON_TRANSPOSE;
    aclsparseComputeType_t computeType = ACL_SPARSE_COMPUTE_16F;

    void fillCustom(const csv_map& row) override
    {
        case_name = parseString(row, "case_name");
        level = parseString(row, "level");
        target = parseString(row, "target");
        expectStatus = PlanParseStatus(parseString(row, "expect_status"));
        verifyFields = parseInt(row, "verify_fields");
        alg = PlanParseAlg(parseString(row, "alg"));

        a.fill(row, "a_");
        b.fill(row, "b_");
        c.fill(row, "c_");
        d.fill(row, "d_");

        opA = PlanParseOp(parseString(row, "op_a"));
        opB = PlanParseOp(parseString(row, "op_b"));
        computeType = PlanParseComputeType(parseString(row, "compute_type"));
    }

    std::string caseId() const override { return case_name; }
};

inline void PrintTo(const MatmulPlanInitParam& p, std::ostream* os)
{
    *os << p.case_name;
}

} // namespace sparse_test

#endif // TEST_SPARSELT_MATMULPLANINIT_PARAM_H_
