/**
 * ----------------------------------------------------------------------------------------------------------
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ----------------------------------------------------------------------------------------------------------
 */

#ifndef TEST_MATMUL_PARAM_H_
#define TEST_MATMUL_PARAM_H_

#include <sstream>
#include <string>

#include "csv_loader.h"

namespace sparse_test {

// =============================================================================
// CSV-driven test parameter for aclsparseLtMatmul (v2: 4 dtype × 2 path).
//
// CSV columns (16 columns, v2):
//   case_id, m, k, n, dtype, output_dtype, matrix_type,
//   alpha, beta, transA, transB, alg_config_id, split_k, split_k_mode, prune_alg,
//   alpha_vector_scaling, beta_vector_scaling
//
// dtype:         FP32 / FP16 / BF16 / INT8
// output_dtype:  INT8 / INT32 / -   (INT8场景区分输出类型; 非INT8填-)
// matrix_type:   sparse×dense / dense×dense
// prune_alg:     STRIP / TILE / -   (dense×dense填-)
// alpha_vector_scaling: 0=scalar, 1=per-row device pointer
// beta_vector_scaling:  0=scalar, 1=per-row device pointer
// =============================================================================

struct MatmulParam : public SparseTestParamBase {
    int32_t case_id = 0;
    int32_t m = 0;
    int32_t k = 0;
    int32_t n = 0;
    std::string dtype;          // FP32 / FP16 / BF16 / INT8
    std::string output_dtype;   // INT8 / INT32 / - (INT8场景)
    std::string matrix_type;    // sparse×dense / dense×dense

    float alpha = 1.0f;
    float beta = 0.0f;
    std::string transA;         // "true" / "false"
    std::string transB;         // "true" / "false"
    int32_t alg_config_id = 0;
    int32_t split_k = 1;
    int32_t split_k_mode = 0;   // 0=ONE_KERNEL, 1=TWO_KERNELS
    std::string prune_alg;      // STRIP / TILE / -
    int32_t alpha_vector_scaling = 0;
    int32_t beta_vector_scaling = 0;

    float range_low = -1.0f;
    float range_high = 1.0f;
    std::string level;
    std::string note;
    std::string order;
    std::string sparse_side;    // A / B (default A)

    // Convenience accessors
    bool isFp32() const { return dtype == "FP32"; }
    bool isFp16() const { return dtype == "FP16"; }
    bool isBf16() const { return dtype == "BF16"; }
    bool isInt8() const { return dtype == "INT8"; }
    bool isInt32Output() const { return output_dtype == "INT32"; }
    bool isInt8Output() const { return output_dtype == "INT8"; }
    bool isDensePath() const { return matrix_type == "dense×dense"; }
    bool isSparsePath() const { return matrix_type == "sparse×dense" || matrix_type.empty(); }
    bool isTransA() const { return transA == "true"; }
    bool isTransB() const { return transB == "true"; }
    bool isColOrder() const { return order == "col"; }
    bool isSparseA() const { return sparse_side != "B"; }
    bool isTilePrune() const { return prune_alg == "TILE"; }
    bool isStripPrune() const { return prune_alg == "STRIP" || prune_alg.empty(); }
    bool isTwoKernels() const { return split_k_mode == 1; }

    void fillCustom(const csv_map& row) override {
        case_id         = parseInt(row, "case_id");
        m               = parseInt(row, "m");
        k               = parseInt(row, "k");
        n               = parseInt(row, "n");
        dtype           = parseString(row, "dtype");
        output_dtype    = parseString(row, "output_dtype");
        matrix_type     = parseString(row, "matrix_type");
        alpha           = parseFloat(row, "alpha");
        beta            = parseFloat(row, "beta");
        transA          = parseString(row, "transA");
        transB          = parseString(row, "transB");
        alg_config_id   = parseInt(row, "alg_config_id");
        split_k         = parseInt(row, "split_k");
        split_k_mode    = parseInt(row, "split_k_mode");
        prune_alg       = parseString(row, "prune_alg");
        alpha_vector_scaling = parseInt(row, "alpha_vector_scaling");
        beta_vector_scaling  = parseInt(row, "beta_vector_scaling");
        range_low       = parseFloat(row, "range_low");
        range_high      = parseFloat(row, "range_high");
        level           = parseString(row, "level");
        note            = parseString(row, "note");
        order           = parseString(row, "order");
        sparse_side     = parseString(row, "sparse_side");
        // Defaults
        if (matrix_type.empty()) { matrix_type = "sparse×dense"; }
        if (transA.empty()) { transA = "false"; }
        if (transB.empty()) { transB = "false"; }
        if (order.empty()) { order = "row"; }
        if (sparse_side.empty()) { sparse_side = "A"; }
        if (prune_alg.empty() && isSparsePath()) { prune_alg = "STRIP"; }
        if (prune_alg.empty()) { prune_alg = "-"; }
        if (output_dtype.empty()) { output_dtype = "-"; }
        if (range_low == 0.0f && range_high == 0.0f) {
            range_low = -1.0f;
            range_high = 1.0f;
        }
    }

    std::string caseId() const override {
        std::ostringstream oss;
        oss << "case" << case_id << "_" << level;
        return oss.str();
    }
};

inline void PrintTo(const MatmulParam& p, std::ostream* os) {
    *os << "case" << p.case_id << "(" << p.dtype << " " << p.m << "x" << p.k
        << "x" << p.n << " " << p.matrix_type
        << " out=" << p.output_dtype
        << " a=" << p.alpha << " b=" << p.beta
        << " splitk" << p.split_k << " mode" << p.split_k_mode
        << " " << (p.isTilePrune() ? "TILE" : "STRIP")
        << " tA=" << (p.isTransA() ? "T" : "N")
        << " tB=" << (p.isTransB() ? "T" : "N")
        << " side=" << (p.isSparseA() ? "A" : "B")
        << " alphaVec=" << p.alpha_vector_scaling
        << " betaVec=" << p.beta_vector_scaling << ")";
}

}  // namespace sparse_test

#endif  // TEST_MATMUL_PARAM_H_

