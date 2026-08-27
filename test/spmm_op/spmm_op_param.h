/**
 * ----------------------------------------------------------------------------------------------------------
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms of the
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR
 * PURPOSE.
 * ----------------------------------------------------------------------------------------------------------
 */

#ifndef TEST_SPMM_OP_SPMM_OP_PARAM_H_
#define TEST_SPMM_OP_SPMM_OP_PARAM_H_

#include "csv_loader.h"

#include <cstdint>
#include <string>

namespace sparse_test {

// SpMMOp test parameter loaded from CSV.
// CSV columns (per test plan §5.0):
//   case_name,description,m,k,n,sparsity_ratio,alpha,beta,dtype,compute_type,
//   op_b,order_b,order_c,alg,pointer_mode,value_lo,value_hi,expect_result,random_seed,
//   skip_precision,index_base,row_offset_type,unsorted
// dtype / compute_type / op_b / order_b / order_c / alg are stored as raw
// enum-name strings and converted to ACL enum values in the test body / NPU
// wrapper (keeps this header free of cann_ops_sparse.h dependencies).
struct SpmmOpTestParam : public SparseTestParamBase {
    std::string case_name;
    std::string description;

    int64_t m = 0;  // rows of A / C
    int64_t k = 0;  // reduction dim (cols of A)
    int64_t n = 0;  // cols of C

    double sparsity_ratio = 0.0;
    double alpha = 0.0;
    double beta = 0.0;

    std::string dtype;         // "ACL_FLOAT" / "ACL_FLOAT16"
    std::string compute_type;  // "ACL_FLOAT" (fixed)

    std::string op_b;          // "ACL_SPARSE_OP_NON_TRANSPOSE" / "ACL_SPARSE_OP_TRANSPOSE"

    // Dense matrix memory layout for B / C.
    std::string order_b;       // "ACL_SPARSE_ORDER_ROW" / "ACL_SPARSE_ORDER_COL"
    std::string order_c;       // "ACL_SPARSE_ORDER_ROW" / "ACL_SPARSE_ORDER_COL"

    std::string alg;           // "ACL_SPARSE_SPMMOP_ALG_DEFAULT" / "_ALG1" / "_ALG2" / "_ALG1_HIGH_PRECISION"

    std::string pointer_mode;  // "HOST" / "DEVICE"

    double value_lo = -1.0;
    double value_hi = 1.0;

    std::string expect_result;  // "ACL_SPARSE_STATUS_SUCCESS"
    uint32_t random_seed = 0;

    // WB/L1 boundary flag: skip numerical precision verification (API return
    // code still asserted). Used for cases where FP32 accumulation under
    // extreme alpha makes value comparison meaningless (expected behavior,
    // not a code defect — mirrors sddmm nnz==0 API-only verification pattern).
    bool skip_precision = false;

    // Extended fields: index base / row offset type / unsorted CSR.
    // Backward-compatible: missing CSV columns default to ZERO / 32I / false.
    std::string index_base;       // "ACL_SPARSE_INDEX_BASE_ZERO" / "ACL_SPARSE_INDEX_BASE_ONE"
    std::string row_offset_type;  // "ACL_SPARSE_INDEX_32I" / "ACL_SPARSE_INDEX_64I"
    bool unsorted = false;        // shuffle colInd/values within each row

    void fillCustom(const csv_map& row) override
    {
        case_name        = parseString(row, "case_name");
        description      = parseString(row, "description");
        m                = static_cast<int64_t>(parseInt(row, "m"));
        k                = static_cast<int64_t>(parseInt(row, "k"));
        n                = static_cast<int64_t>(parseInt(row, "n"));
        sparsity_ratio   = parseDouble(row, "sparsity_ratio");
        alpha            = parseDouble(row, "alpha");
        beta             = parseDouble(row, "beta");
        dtype            = parseString(row, "dtype");
        compute_type     = parseString(row, "compute_type");
        op_b             = parseString(row, "op_b");
        order_b          = parseString(row, "order_b");
        if (order_b.empty()) {
            order_b = "ACL_SPARSE_ORDER_ROW";
        }
        order_c          = parseString(row, "order_c");
        if (order_c.empty()) {
            order_c = "ACL_SPARSE_ORDER_ROW";
        }
        alg              = parseString(row, "alg");
        pointer_mode     = parseString(row, "pointer_mode");
        if (pointer_mode.empty()) {
            pointer_mode = "DEVICE";
        }
        value_lo         = parseDouble(row, "value_lo");
        value_hi         = parseDouble(row, "value_hi");
        expect_result    = parseString(row, "expect_result");
        random_seed      = static_cast<uint32_t>(parseInt(row, "random_seed"));
        skip_precision   = parseBool(row, "skip_precision");
        index_base       = parseString(row, "index_base");
        if (index_base.empty()) {
            index_base = "ACL_SPARSE_INDEX_BASE_ZERO";
        }
        row_offset_type  = parseString(row, "row_offset_type");
        if (row_offset_type.empty()) {
            row_offset_type = "ACL_SPARSE_INDEX_32I";
        }
        unsorted         = parseBool(row, "unsorted");
    }

    std::string caseId() const override { return case_name; }
};

inline void PrintTo(const SpmmOpTestParam& p, std::ostream* os)
{
    *os << p.case_name;
}

}  // namespace sparse_test

#endif  // TEST_SPMM_OP_SPMM_OP_PARAM_H_
