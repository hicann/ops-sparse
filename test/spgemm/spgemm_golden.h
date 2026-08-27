/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * SpGEMM CPU golden reference (pure C++ FP64).
 * Formula: C = alpha * A * B + beta * C_in
 */

#ifndef TEST_SPGEMM_GOLDEN_H_
#define TEST_SPGEMM_GOLDEN_H_

#include <algorithm>
#include <cmath>
#include <set>
#include <vector>
#include "../frame/fill.h"

namespace sparse_test {

// Collect (col, val) product pairs for row i of A * B
inline void SpgemmCollectRowPairs(
    const CsrMatrix& a, const CsrMatrix& b, int i,
    std::vector<std::pair<int, double>>& pairs)
{
    for (int idxA = a.rowOffsets[i]; idxA < a.rowOffsets[i + 1]; idxA++) {
        int colA = a.colIndices[idxA];
        double valA = static_cast<double>(a.values[idxA]);
        for (int idxB = b.rowOffsets[colA]; idxB < b.rowOffsets[colA + 1]; idxB++) {
            pairs.emplace_back(b.colIndices[idxB],
                               valA * static_cast<double>(b.values[idxB]));
        }
    }
}

// Sort pairs by column and merge duplicates (sum into first entry)
inline void SpgemmMergePairs(std::vector<std::pair<int, double>>& pairs)
{
    std::sort(pairs.begin(), pairs.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });
    std::vector<std::pair<int, double>> merged;
    for (const auto& [col, val] : pairs) {
        if (!merged.empty() && merged.back().first == col) {
            merged.back().second += val;
        } else {
            merged.emplace_back(col, val);
        }
    }
    pairs = std::move(merged);
}

// Merge beta * C_in[i,:] into the already-sorted merged pairs
inline void SpgemmMergeCInRow(
    std::vector<std::pair<int, double>>& merged,
    const CsrMatrix& cIn, int i, double beta)
{
    std::vector<std::pair<int, double>> combined;
    size_t j = 0;
    for (int idxC = cIn.rowOffsets[i]; idxC < cIn.rowOffsets[i + 1]; idxC++) {
        int colC = cIn.colIndices[idxC];
        double valC = static_cast<double>(cIn.values[idxC]) * beta;
        while (j < merged.size() && merged[j].first < colC) {
            combined.push_back(merged[j++]);
        }
        if (j < merged.size() && merged[j].first == colC) {
            combined.emplace_back(colC, merged[j++].second + valC);
        } else {
            combined.emplace_back(colC, valC);
        }
    }
    while (j < merged.size()) combined.push_back(merged[j++]);
    merged = std::move(combined);
}

// SpGEMM golden: C = alpha * A * B (+ beta * C_in when beta != 0)
// A: CSR matrix (rows = m, cols = k)
// B: CSR matrix (rows = k, cols = n)
// cIn: CSR matrix (rows = m, cols = n), only used when beta != 0
// Returns: C as CSR matrix (FP32 values, FP64 accumulation)
inline CsrMatrix SpGEMMGolden(
    const CsrMatrix& a, const CsrMatrix& b,
    float alpha, float beta, const CsrMatrix* cIn = nullptr)
{
    const int m = static_cast<int>(a.rows);
    const int n = static_cast<int>(b.cols);

    CsrMatrix out;
    out.rows = m;
    out.cols = n;
    if (m <= 0 || n <= 0) {
        out.nnz = 0;
        out.rowOffsets.assign(m + 1, 0);
        return out;
    }
    out.rowOffsets.resize(m + 1, 0);

    bool useCIn = (beta != 0.0f && cIn != nullptr && cIn->nnz > 0);
    double alphaD = static_cast<double>(alpha);
    double betaD  = static_cast<double>(beta);

    for (int i = 0; i < m; i++) {
        std::vector<std::pair<int, double>> pairs;
        SpgemmCollectRowPairs(a, b, i, pairs);
        SpgemmMergePairs(pairs);
        for (auto& [col, val] : pairs) val *= alphaD;
        if (useCIn) SpgemmMergeCInRow(pairs, *cIn, i, betaD);
        for (const auto& [col, val] : pairs) {
            out.colIndices.push_back(static_cast<int32_t>(col));
            out.values.push_back(static_cast<float>(val));
        }
        out.rowOffsets[i + 1] = static_cast<int32_t>(out.colIndices.size());
    }
    out.nnz = static_cast<int64_t>(out.colIndices.size());
    return out;
}

}  // namespace sparse_test

#endif
