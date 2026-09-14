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

/*!
 * \file spgemm_ref.h
 * \brief SpGEMM CPU 参考实现（两趟法），供 UT 做逐元素对拍。
 */

#ifndef SPGEMM_TEST_REF_H
#define SPGEMM_TEST_REF_H

#include <cstdint>
#include <algorithm>
#include <map>
#include <utility>
#include <vector>

/**
 * CSR 矩阵的 host 侧表示。
 * complex64 时 valRe/valIm 均为长度 nnz，实数类型 valIm 全 0。
 */
struct SpgemmRefCsr {
    int32_t rows = 0;
    int32_t cols = 0;
    std::vector<int32_t> rowPtr;   // 长度 rows+1
    std::vector<int32_t> colIdx;   // 长度 nnz
    std::vector<double> valRe;     // 长度 nnz
    std::vector<double> valIm;     // 长度 nnz（实数类型全 0）

    int32_t Nnz() const { return rowPtr.empty() ? 0 : rowPtr.back(); }
};

/**
 * CPU 两趟参考实现：C = alpha * A * B + beta * C_in。
 * 用 std::map 逐行累加，有序遍历保证升序列索引，插入即建立条目保证显式零保留。
 */
// 行内累加器：key = 列索引，value = (re, im) 的 double 累加对。
using RefAccMap = std::map<int32_t, std::pair<double, double>>;

// beta * C_in 的第 i 行入账。
inline void RefAccumulateBetaC(RefAccMap &acc, const SpgemmRefCsr &cIn, int32_t i,
                               double betaRe, double betaIm, bool isComplex)
{
    for (int32_t p = cIn.rowPtr[i]; p < cIn.rowPtr[i + 1]; p++) {
        const int32_t col = cIn.colIdx[p];
        const double cr = cIn.valRe[p];
        const double ci = isComplex ? cIn.valIm[p] : 0.0;
        double pr = betaRe * cr;
        double pi = 0.0;
        if (isComplex) {
            pr = betaRe * cr - betaIm * ci;
            pi = betaRe * ci + betaIm * cr;
        }
        auto &slot = acc[col];
        slot.first += pr;
        slot.second += pi;
    }
}

// alpha * A * B 的第 i 行入账。
inline void RefAccumulateAlphaAb(RefAccMap &acc, const SpgemmRefCsr &a,
                                 const SpgemmRefCsr &b, int32_t i, double alphaRe,
                                 double alphaIm, bool isComplex)
{
    for (int32_t ap = a.rowPtr[i]; ap < a.rowPtr[i + 1]; ap++) {
        const int32_t k = a.colIdx[ap];
        if (k < 0 || k >= b.rows) {
            continue;
        }
        const double ar = a.valRe[ap];
        const double ai = isComplex ? a.valIm[ap] : 0.0;
        // alpha 提前乘入 A 值
        double sar = alphaRe * ar;
        double sai = 0.0;
        if (isComplex) {
            sar = alphaRe * ar - alphaIm * ai;
            sai = alphaRe * ai + alphaIm * ar;
        }

        for (int32_t bp = b.rowPtr[k]; bp < b.rowPtr[k + 1]; bp++) {
            const int32_t col = b.colIdx[bp];
            const double br = b.valRe[bp];
            const double bi = isComplex ? b.valIm[bp] : 0.0;
            double pr = sar * br;
            double pi = 0.0;
            if (isComplex) {
                pr = sar * br - sai * bi;
                pi = sar * bi + sai * br;
            }
            auto &slot = acc[col];
            slot.first += pr;
            slot.second += pi;
        }
    }
}

inline SpgemmRefCsr SpgemmRefCompute(
    const SpgemmRefCsr &a, const SpgemmRefCsr &b,
    double alphaRe, double alphaIm, double betaRe, double betaIm,
    const SpgemmRefCsr *cIn, bool isComplex)
{
    SpgemmRefCsr c;
    c.rows = a.rows;
    c.cols = b.cols;
    c.rowPtr.assign(static_cast<size_t>(a.rows) + 1, 0);

    const bool useBeta = (betaRe != 0.0 || betaIm != 0.0) && (cIn != nullptr);

    for (int32_t i = 0; i < a.rows; i++) {
        // key = 列索引；value = (re, im) 的 double 累加器
        RefAccMap acc;

        // 入账顺序：先 beta * C_in，再 alpha * A * B。
        if (useBeta) {
            RefAccumulateBetaC(acc, *cIn, i, betaRe, betaIm, isComplex);
        }
        RefAccumulateAlphaAb(acc, a, b, i, alphaRe, alphaIm, isComplex);

        // map 有序遍历 → 行内列索引严格升序，无重复列
        for (const auto &kv : acc) {
            c.colIdx.push_back(kv.first);
            c.valRe.push_back(kv.second.first);
            c.valIm.push_back(isComplex ? kv.second.second : 0.0);
        }
        c.rowPtr[static_cast<size_t>(i) + 1] = static_cast<int32_t>(c.colIdx.size());
    }
    return c;
}

/**
 * 构造一个每行 degree 个非零的 CSR，列索引 = (i*stride + j) mod cols。
 * @param emptyEvery > 0 时每 emptyEvery 行造一个空行。
 */
inline SpgemmRefCsr SpgemmRefBuildCsr(int32_t rows, int32_t cols, int32_t degree,
                                      int32_t stride, int32_t emptyEvery,
                                      bool isComplex, uint32_t seed)
{
    SpgemmRefCsr m;
    m.rows = rows;
    m.cols = cols;
    m.rowPtr.assign(static_cast<size_t>(rows) + 1, 0);
    if (rows == 0 || cols == 0 || degree == 0) {
        return m;
    }
    if (degree > cols) {
        degree = cols;
    }
    if (stride < 1) {
        stride = 1;
    }

    uint32_t rng = seed * 2654435761U + 1U;
    auto nextVal = [&rng]() -> double {
        rng = rng * 1664525U + 1013904223U;
        // 映射到 [-1, 1)，避免极端值掩盖结构错误
        return static_cast<double>(static_cast<int32_t>(rng >> 8) % 2000 - 1000) / 1000.0;
    };

    for (int32_t i = 0; i < rows; i++) {
        int32_t rowLen = degree;
        if (emptyEvery > 0 && (i % emptyEvery) == 0) {
            rowLen = 0;
        }
        // 去重后排序：保证行内列索引严格升序（CSR 规范化输入前提）
        std::vector<int32_t> cs;
        for (int32_t j = 0; j < rowLen; j++) {
            const int32_t col = (i + j * stride) % cols;
            bool dup = false;
            for (int32_t prev : cs) {
                if (prev == col) {
                    dup = true;
                    break;
                }
            }
            if (!dup) {
                cs.push_back(col);
            }
        }
        std::sort(cs.begin(), cs.end());
        for (int32_t col : cs) {
            m.colIdx.push_back(col);
            m.valRe.push_back(nextVal());
            m.valIm.push_back(isComplex ? nextVal() * 0.25 : 0.0);
        }
        m.rowPtr[static_cast<size_t>(i) + 1] = static_cast<int32_t>(m.colIdx.size());
    }
    return m;
}

#endif  // SPGEMM_TEST_REF_H
