/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TEST_FRAME_FILL_SPARSE_H_
#define TEST_FRAME_FILL_SPARSE_H_

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

#include "fill.h"

namespace sparse_test {

struct SlicedEllMatrix {
    std::vector<int32_t> sliceOffsets;
    std::vector<int32_t> colIndices;
    std::vector<float> values;
    int64_t rows = 0;
    int64_t cols = 0;
    int64_t nnz = 0;
    int32_t sliceWidth = 1;
};

inline void sortCsrRowIndices(CsrMatrix& csr) {
    for (int64_t i = 0; i < csr.rows; i++) {
        int32_t begin = csr.rowOffsets[i];
        int32_t end = csr.rowOffsets[i + 1];
        for (int32_t a = begin; a < end - 1; a++) {
            for (int32_t b = a + 1; b < end; b++) {
                if (csr.colIndices[a] > csr.colIndices[b]) {
                    std::swap(csr.colIndices[a], csr.colIndices[b]);
                    std::swap(csr.values[a], csr.values[b]);
                }
            }
        }
    }
}

inline CsrMatrix cooToCsr(const CooMatrix& coo) {
    CsrMatrix out;
    out.rows = coo.rows;
    out.cols = coo.cols;
    out.nnz = coo.nnz;
    out.rowOffsets.assign(coo.rows + 1, 0);
    for (int64_t k = 0; k < coo.nnz; k++) {
        out.rowOffsets[coo.rowIndices[k] + 1]++;
    }
    for (int64_t i = 0; i < coo.rows; i++) {
        out.rowOffsets[i + 1] += out.rowOffsets[i];
    }
    out.colIndices.resize(coo.nnz);
    out.values.resize(coo.nnz);
    std::vector<int32_t> counter(out.rowOffsets.begin(), out.rowOffsets.end() - 1);
    for (int64_t k = 0; k < coo.nnz; k++) {
        int32_t r = coo.rowIndices[k];
        int32_t pos = counter[r]++;
        out.colIndices[pos] = coo.colIndices[k];
        out.values[pos] = coo.values[k];
    }
    sortCsrRowIndices(out);
    return out;
}

inline CsrMatrix cscToCsr(const CscMatrix& csc) {
    CsrMatrix out;
    out.rows = csc.rows;
    out.cols = csc.cols;
    out.nnz = csc.nnz;
    out.rowOffsets.assign(csc.rows + 1, 0);
    for (int64_t k = 0; k < csc.nnz; k++) {
        out.rowOffsets[csc.rowIndices[k] + 1]++;
    }
    for (int64_t i = 0; i < csc.rows; i++) {
        out.rowOffsets[i + 1] += out.rowOffsets[i];
    }
    out.colIndices.resize(csc.nnz);
    out.values.resize(csc.nnz);
    std::vector<int32_t> counter(out.rowOffsets.begin(), out.rowOffsets.end() - 1);
    for (int64_t c = 0; c < csc.cols; c++) {
        for (int32_t k = csc.colOffsets[c]; k < csc.colOffsets[c + 1]; k++) {
            int32_t r = csc.rowIndices[k];
            int32_t pos = counter[r]++;
            out.colIndices[pos] = static_cast<int32_t>(c);
            out.values[pos] = csc.values[k];
        }
    }
    sortCsrRowIndices(out);
    return out;
}

inline void FinalizeCsr(CsrMatrix& out, int64_t m, int64_t currentNnz) {
    out.rowOffsets[m] = static_cast<int32_t>(currentNnz);
    out.nnz = currentNnz;
}

inline CsrMatrix makeTriangularCsr(int64_t m, bool lower, bool unitDiag,
                                    double sparsity, uint32_t seed) {
    CsrMatrix out;
    out.rows = m;
    out.cols = m;
    out.rowOffsets.assign(m + 1, 0);
    if (m <= 0) { out.nnz = 0; return out; }
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> diagDist(1.0f, 5.0f);
    std::uniform_real_distribution<float> offDist(-2.0f, 2.0f);
    std::uniform_real_distribution<float> probDist(0.0f, 1.0f);
    double density = 1.0 - sparsity;
    int64_t currentNnz = 0;
    for (int64_t i = 0; i < m; i++) {
        out.rowOffsets[i] = static_cast<int32_t>(currentNnz);
        int64_t jStart = lower ? 0 : i;
        int64_t jEnd = lower ? i : m - 1;
        for (int64_t j = jStart; j <= jEnd; j++) {
            if (j == i) {
                out.colIndices.push_back(static_cast<int32_t>(j));
                out.values.push_back(unitDiag ? 1.0f : diagDist(rng));
                currentNnz++;
            } else if (probDist(rng) < static_cast<float>(density)) {
                out.colIndices.push_back(static_cast<int32_t>(j));
                out.values.push_back(offDist(rng));
                currentNnz++;
            }
        }
    }
    FinalizeCsr(out, m, currentNnz);
    return out;
}

inline CsrMatrix makeDiagDominantTriangularCsr(int64_t m, bool lower, uint32_t seed) {
    CsrMatrix out;
    out.rows = m;
    out.cols = m;
    out.rowOffsets.assign(m + 1, 0);
    if (m <= 0) { out.nnz = 0; return out; }
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> offDist(-2.0f, 2.0f);
    std::uniform_real_distribution<float> biasDist(0.5f, 2.0f);
    int64_t currentNnz = 0;
    for (int64_t i = 0; i < m; i++) {
        out.rowOffsets[i] = static_cast<int32_t>(currentNnz);
        float rowAbsSum = 0.0f;
        int64_t jStart = lower ? 0 : i + 1;
        int64_t jEnd = lower ? i - 1 : m - 1;
        std::vector<int32_t> rowCols;
        std::vector<float> rowVals;
        for (int64_t j = jStart; j <= jEnd; j++) {
            float v = offDist(rng);
            rowCols.push_back(static_cast<int32_t>(j));
            rowVals.push_back(v);
            rowAbsSum += std::abs(v);
        }
        float diagVal = rowAbsSum + biasDist(rng);
        if (lower) {
            for (size_t k = 0; k < rowCols.size(); k++) {
                out.colIndices.push_back(rowCols[k]);
                out.values.push_back(rowVals[k]);
                currentNnz++;
            }
            out.colIndices.push_back(static_cast<int32_t>(i));
            out.values.push_back(diagVal);
            currentNnz++;
        } else {
            out.colIndices.push_back(static_cast<int32_t>(i));
            out.values.push_back(diagVal);
            currentNnz++;
            for (size_t k = 0; k < rowCols.size(); k++) {
                out.colIndices.push_back(rowCols[k]);
                out.values.push_back(rowVals[k]);
                currentNnz++;
            }
        }
    }
    out.rowOffsets[m] = static_cast<int32_t>(currentNnz);
    out.nnz = currentNnz;
    return out;
}

inline CsrMatrix makeTridiagTriangularCsr(int64_t m, bool lower, uint32_t seed) {
    CsrMatrix out;
    out.rows = m;
    out.cols = m;
    out.rowOffsets.assign(m + 1, 0);
    if (m <= 0) { out.nnz = 0; return out; }
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> diagDist(3.0f, 6.0f);
    std::uniform_real_distribution<float> offDist(-1.0f, 1.0f);
    int64_t currentNnz = 0;
    for (int64_t i = 0; i < m; i++) {
        out.rowOffsets[i] = static_cast<int32_t>(currentNnz);
        if (lower) {
            if (i > 0) {
                out.colIndices.push_back(static_cast<int32_t>(i - 1));
                out.values.push_back(offDist(rng));
                currentNnz++;
            }
            out.colIndices.push_back(static_cast<int32_t>(i));
            out.values.push_back(diagDist(rng));
            currentNnz++;
        } else {
            out.colIndices.push_back(static_cast<int32_t>(i));
            out.values.push_back(diagDist(rng));
            currentNnz++;
            if (i < m - 1) {
                out.colIndices.push_back(static_cast<int32_t>(i + 1));
                out.values.push_back(offDist(rng));
                currentNnz++;
            }
        }
    }
    out.rowOffsets[m] = static_cast<int32_t>(currentNnz);
    out.nnz = currentNnz;
    return out;
}

inline CsrMatrix makeIdentityCsr(int64_t m) {
    return makeDiagCsr(m, 1.0f);
}

inline CsrMatrix makeEmptyTriangularCsr(int64_t m) {
    CsrMatrix out;
    out.rows = m;
    out.cols = m;
    out.nnz = 0;
    out.rowOffsets.assign(m + 1, 0);
    return out;
}

inline SlicedEllMatrix csrToSlicedEll(const CsrMatrix& csr, int32_t sliceWidth = 1) {
    if (sliceWidth <= 0) sliceWidth = 1;
    const int32_t sw = sliceWidth;  // 帮助静态分析工具理解始终为正
    SlicedEllMatrix out;
    out.rows = csr.rows;
    out.cols = csr.cols;
    out.nnz = csr.nnz;
    out.sliceWidth = sw;
    int32_t maxNnzPerRow = 0;
    for (int64_t i = 0; i < csr.rows; i++) {
        int32_t rowNnz = csr.rowOffsets[i + 1] - csr.rowOffsets[i];
        if (rowNnz > maxNnzPerRow) maxNnzPerRow = rowNnz;
    }
    int32_t numSlices = (maxNnzPerRow + sw - 1) / sw;
    if (numSlices == 0) numSlices = 1;
    out.sliceOffsets.resize(numSlices + 1, 0);
    for (int32_t s = 0; s < numSlices; s++) {
        out.sliceOffsets[s + 1] = out.sliceOffsets[s] +
            static_cast<int32_t>(csr.rows) * sw;
    }
    int32_t totalSlots = out.sliceOffsets[numSlices];
    out.colIndices.assign(totalSlots, -1);
    out.values.assign(totalSlots, 0.0f);
    for (int64_t i = 0; i < csr.rows; i++) {
        int32_t idx = 0;
        for (int32_t k = csr.rowOffsets[i]; k < csr.rowOffsets[i + 1]; k++) {
            int32_t s = idx / sw;
            int32_t off = idx % sw;
            int32_t pos = out.sliceOffsets[s] + static_cast<int32_t>(i) * sw + off;
            out.colIndices[pos] = csr.colIndices[k];
            out.values[pos] = csr.values[k];
            idx++;
        }
    }
    return out;
}

inline CsrMatrix slicedEllToCsr(const SlicedEllMatrix& ell) {
    CsrMatrix out;
    out.rows = ell.rows;
    out.cols = ell.cols;
    out.rowOffsets.assign(ell.rows + 1, 0);
    int32_t numSlices = static_cast<int32_t>(ell.sliceOffsets.size()) - 1;
    for (int64_t i = 0; i < ell.rows; i++) {
        int32_t rowNnz = 0;
        for (int32_t s = 0; s < numSlices; s++) {
            for (int32_t k = 0; k < ell.sliceWidth; k++) {
                int32_t pos = ell.sliceOffsets[s] + static_cast<int32_t>(i) * ell.sliceWidth + k;
                if (pos < static_cast<int32_t>(ell.colIndices.size()) && ell.colIndices[pos] >= 0) {
                    rowNnz++;
                }
            }
        }
        out.rowOffsets[i + 1] = rowNnz;
    }
    for (int64_t i = 0; i < ell.rows; i++) {
        out.rowOffsets[i + 1] += out.rowOffsets[i];
    }
    out.nnz = out.rowOffsets[ell.rows];
    out.colIndices.resize(out.nnz);
    out.values.resize(out.nnz);
    for (int64_t i = 0; i < ell.rows; i++) {
        int32_t idx = out.rowOffsets[i];
        for (int32_t s = 0; s < numSlices; s++) {
            for (int32_t k = 0; k < ell.sliceWidth; k++) {
                int32_t pos = ell.sliceOffsets[s] + static_cast<int32_t>(i) * ell.sliceWidth + k;
                if (pos < static_cast<int32_t>(ell.colIndices.size()) && ell.colIndices[pos] >= 0) {
                    out.colIndices[idx] = ell.colIndices[pos];
                    out.values[idx] = ell.values[pos];
                    idx++;
                }
            }
        }
    }
    return out;
}

inline CsrMatrix unsortCsrRowIndices(const CsrMatrix& csr, uint32_t seed) {
    CsrMatrix out = csr;
    std::mt19937 rng(seed);
    for (int64_t i = 0; i < csr.rows; i++) {
        int32_t begin = out.rowOffsets[i];
        int32_t end = out.rowOffsets[i + 1];
        int32_t len = end - begin;
        for (int32_t a = len - 1; a > 0; a--) {
            std::uniform_int_distribution<int32_t> dist(0, a);
            int32_t b = dist(rng);
            std::swap(out.colIndices[begin + a], out.colIndices[begin + b]);
            std::swap(out.values[begin + a], out.values[begin + b]);
        }
    }
    return out;
}

inline CsrMatrix makeSingularTriangularCsr(int64_t m, bool lower, uint32_t seed) {
    CsrMatrix out = makeTriangularCsr(m, lower, false, 0.5, seed);
    for (int64_t i = 0; i < m; i++) {
        for (int32_t k = out.rowOffsets[i]; k < out.rowOffsets[i + 1]; k++) {
            if (out.colIndices[k] == static_cast<int32_t>(i)) {
                out.values[k] = 0.0f;
            }
        }
    }
    return out;
}

inline CsrMatrix makeMissingDiagTriangularCsr(int64_t m, bool lower, uint32_t seed) {
    CsrMatrix out;
    out.rows = m;
    out.cols = m;
    out.rowOffsets.assign(m + 1, 0);
    if (m <= 0) { out.nnz = 0; return out; }
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> diagDist(2.0f, 5.0f);
    std::uniform_real_distribution<float> offDist(-2.0f, 2.0f);
    std::uniform_real_distribution<float> probDist(0.0f, 1.0f);
    int64_t currentNnz = 0;
    for (int64_t i = 0; i < m; i++) {
        out.rowOffsets[i] = static_cast<int32_t>(currentNnz);
        if (i == 1 && m > 2) {
            if (lower && i > 0) {
                out.colIndices.push_back(0);
                out.values.push_back(offDist(rng));
                currentNnz++;
            } else if (!lower && i < m - 1) {
                out.colIndices.push_back(static_cast<int32_t>(m - 1));
                out.values.push_back(offDist(rng));
                currentNnz++;
            }
            continue;
        }
        if (i == 3 && m > 4) {
            continue;
        }
        int64_t jStart = lower ? 0 : i;
        int64_t jEnd = lower ? i : m - 1;
        for (int64_t j = jStart; j <= jEnd; j++) {
            if (j == i) {
                out.colIndices.push_back(static_cast<int32_t>(j));
                out.values.push_back(diagDist(rng));
                currentNnz++;
            } else if (probDist(rng) < 0.5f) {
                out.colIndices.push_back(static_cast<int32_t>(j));
                out.values.push_back(offDist(rng));
                currentNnz++;
            }
        }
    }
    FinalizeCsr(out, m, currentNnz);
    return out;
}

}

#endif
