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

#ifndef TEST_GEBSR2GEBSC_GEBSR2GEBSC_GOLDEN_H_
#define TEST_GEBSR2GEBSC_GEBSR2GEBSC_GOLDEN_H_

#include <cstdint>
#include <vector>
#include <stdexcept>
#include <string>

#ifdef SPARSE_TEST_USE_EIGEN
#include <Eigen/Sparse>
#include <Eigen/Dense>
#endif

#include "fill.h"
#include "cann_ops_sparse.h"

namespace sparse_test {

struct Gebsr2GebscGoldenResult {
    std::vector<int32_t> bscColPtr;
    std::vector<int32_t> bscRowInd;
    std::vector<uint8_t> bscVal;
    int32_t nnzb;
};

inline int DetermineCopyMode(int rA, int cA, int rC, int cC)
{
    if (rC == rA && cC == cA) return 0;
    if (rC == cA && cC == rA) return 1;
    return -1;
}

inline void CopyBlockDirect(
    const uint8_t* src, uint8_t* dst,
    int rA, int cA, int valSize)
{
    int totalBytes = rA * cA * valSize;
    for (int b = 0; b < totalBytes; b++) {
        dst[b] = src[b];
    }
}

inline void CopyBlockTranspose(
    const uint8_t* src, uint8_t* dst,
    int rA, int cA,
    int dirA, int valSize)
{
    for (int i = 0; i < rA; i++) {
        for (int j = 0; j < cA; j++) {
            int srcOff;
            int dstOff;
            if (dirA == 0) {
                srcOff = (i * cA + j) * valSize;
                dstOff = (j * rA + i) * valSize;
            } else {
                srcOff = (j * rA + i) * valSize;
                dstOff = (i * cA + j) * valSize;
            }
            for (int b = 0; b < valSize; b++) {
                dst[dstOff + b] = src[srcOff + b];
            }
        }
    }
}

inline void ComputeBscColPtr(
    const CsrMatrix& csr, int nb, int nnzb, int idxBase,
    std::vector<int32_t>& bscColPtr, std::vector<int32_t>& colOffset)
{
    std::vector<int32_t> colCount(nb, 0);
    for (int k = 0; k < nnzb; k++) {
        int col = csr.colIndices[k] - idxBase;
        colCount[col]++;
    }

    bscColPtr[0] = idxBase;
    for (int j = 0; j < nb; j++) {
        bscColPtr[j + 1] = bscColPtr[j] + colCount[j];
    }

    colOffset.resize(nb);
    for (int j = 0; j < nb; j++) {
        colOffset[j] = bscColPtr[j] - idxBase;
    }
}

inline void ScatterBlocks(
    const CsrMatrix& csr, int mb, int idxBase, int copyValues,
    int blockSizeA, int blockSizeC, int valSize,
    int copyMode, int rowBlockDimA, int colBlockDimA,
    int rowBlockDimC, int colBlockDimC, int dirA,
    const std::vector<uint8_t>& blockValBytes,
    std::vector<int32_t>& colOffset,
    Gebsr2GebscGoldenResult& result)
{
    for (int i = 0; i < mb; i++) {
        int rowStart = csr.rowOffsets[i] - idxBase;
        int rowEnd = csr.rowOffsets[i + 1] - idxBase;
        for (int k = rowStart; k < rowEnd; k++) {
            int col = csr.colIndices[k] - idxBase;
            int dest = colOffset[col];
            result.bscRowInd[dest] = i + idxBase;

            if (copyValues == 1) {
                const uint8_t* src = &blockValBytes[static_cast<size_t>(k) * blockSizeA * valSize];
                uint8_t* dst = &result.bscVal[static_cast<size_t>(dest) * blockSizeC * valSize];

                if (copyMode == 0) {
                    CopyBlockDirect(src, dst, rowBlockDimA, colBlockDimA, valSize);
                } else {
                    CopyBlockTranspose(src, dst,
                        rowBlockDimA, colBlockDimA,
                        dirA, valSize);
                }
            }
            colOffset[col]++;
        }
    }
}

#ifdef SPARSE_TEST_USE_EIGEN

inline float DecodeValF32(const uint8_t* p, int valSize, int elemIdx)
{
    size_t off = static_cast<size_t>(elemIdx) * valSize;
    if (valSize == 4) {
        union { float f; uint32_t u; } cvt;
        cvt.u = static_cast<uint32_t>(p[off])
              | (static_cast<uint32_t>(p[off+1]) << 8)
              | (static_cast<uint32_t>(p[off+2]) << 16)
              | (static_cast<uint32_t>(p[off+3]) << 24);
        return cvt.f;
    }
    if (valSize == 2) {
        uint16_t h = static_cast<uint16_t>(p[off])
                   | (static_cast<uint16_t>(p[off+1]) << 8);
        uint32_t bits = static_cast<uint32_t>(h & 0x8000) << 16;
        int exp = (h >> 10) & 0x1F;
        int mant = h & 0x3FF;
        if (exp == 0) {
            if (mant != 0) {
                exp = 1;
                while ((mant & 0x400) == 0) { mant <<= 1; exp--; }
                mant &= 0x3FF;
                bits |= ((exp + 127 - 15) << 23) | (mant << 13);
            }
        } else if (exp == 31) {
            bits |= 0x7F800000 | (mant << 13);
        } else {
            bits |= ((exp + 127 - 15) << 23) | (mant << 13);
        }
        union { float f; uint32_t u; } cvt;
        cvt.u = bits;
        return cvt.f;
    }
    if (valSize == 1) {
        return static_cast<float>(static_cast<int8_t>(p[off]));
    }
    return 0.0f;
}

inline void EigenStructCrossCheck(
    const CsrMatrix& csr, int mb, int nb, int nnzb, int idxBase,
    const Gebsr2GebscGoldenResult& result)
{
    using Triplet = Eigen::Triplet<double>;
    std::vector<Triplet> triplets;
    triplets.reserve(static_cast<size_t>(nnzb));
    for (int i = 0; i < mb; ++i) {
        int rowStart = csr.rowOffsets[i] - idxBase;
        int rowEnd = csr.rowOffsets[i + 1] - idxBase;
        for (int k = rowStart; k < rowEnd; ++k) {
            int j = csr.colIndices[k] - idxBase;
            triplets.emplace_back(static_cast<double>(i),
                                  static_cast<double>(j),
                                  static_cast<double>(k + 1));
        }
    }
    Eigen::SparseMatrix<double, Eigen::RowMajor> bsrEigen(mb, nb);
    bsrEigen.setFromTriplets(triplets.begin(), triplets.end());
    if (bsrEigen.nonZeros() != nnzb) {
        throw std::logic_error("Eigen struct cross-check: duplicate block coords");
    }
    Eigen::SparseMatrix<double, Eigen::ColMajor> bscEigen(bsrEigen);
    for (int j = 0; j <= nb; ++j) {
        int32_t expect = static_cast<int32_t>(bscEigen.outerIndexPtr()[j]) + idxBase;
        if (result.bscColPtr[static_cast<size_t>(j)] != expect) {
            throw std::logic_error("Eigen bscColPtr cross-check failed at col "
                + std::to_string(j));
        }
    }
    for (int j = 0; j < nb; ++j) {
        int start = bscEigen.outerIndexPtr()[j];
        int end = bscEigen.outerIndexPtr()[j + 1];
        for (int p = start; p < end; ++p) {
            int expectRow = static_cast<int>(bscEigen.innerIndexPtr()[p]) + idxBase;
            if (result.bscRowInd[static_cast<size_t>(p)] != expectRow) {
                throw std::logic_error("Eigen bscRowInd cross-check failed at pos "
                    + std::to_string(p));
            }
        }
    }
}

inline Eigen::MatrixXd EigenBuildDenseIn(
    const CsrMatrix& csr, const std::vector<uint8_t>& blockValBytes,
    int mb, int nb, int idxBase, int dirA,
    int rowBlockDimA, int colBlockDimA, int valSize)
{
    int blockSizeA = rowBlockDimA * colBlockDimA;
    Eigen::MatrixXd dense = Eigen::MatrixXd::Zero(mb * rowBlockDimA, nb * colBlockDimA);
    for (int i = 0; i < mb; ++i) {
        int rowStart = csr.rowOffsets[i] - idxBase;
        int rowEnd = csr.rowOffsets[i + 1] - idxBase;
        for (int k = rowStart; k < rowEnd; ++k) {
            int bj = csr.colIndices[k] - idxBase;
            const uint8_t* blk = &blockValBytes[static_cast<size_t>(k) * blockSizeA * valSize];
            for (int ii = 0; ii < rowBlockDimA; ++ii) {
                for (int jj = 0; jj < colBlockDimA; ++jj) {
                    int elemIdx = (dirA == 0) ? ii * colBlockDimA + jj : jj * rowBlockDimA + ii;
                    dense(i * rowBlockDimA + ii, bj * colBlockDimA + jj) =
                        static_cast<double>(DecodeValF32(blk, valSize, elemIdx));
                }
            }
        }
    }
    return dense;
}

inline Eigen::MatrixXd EigenBuildDenseOut(
    const Gebsr2GebscGoldenResult& result,
    int mb, int nb, int idxBase, int dirA,
    int rowBlockDimC, int colBlockDimC, int valSize)
{
    int blockSizeC = rowBlockDimC * colBlockDimC;
    Eigen::MatrixXd dense = Eigen::MatrixXd::Zero(mb * rowBlockDimC, nb * colBlockDimC);
    for (int bc = 0; bc < nb; ++bc) {
        for (int b = result.bscColPtr[bc] - idxBase;
             b < result.bscColPtr[bc + 1] - idxBase; ++b) {
            int br = result.bscRowInd[static_cast<size_t>(b)] - idxBase;
            const uint8_t* blk = &result.bscVal[static_cast<size_t>(b) * blockSizeC * valSize];
            for (int ii = 0; ii < rowBlockDimC; ++ii) {
                for (int jj = 0; jj < colBlockDimC; ++jj) {
                    int elemIdx = (dirA == 0) ? ii * colBlockDimC + jj : jj * rowBlockDimC + ii;
                    dense(br * rowBlockDimC + ii, bc * colBlockDimC + jj) =
                        static_cast<double>(DecodeValF32(blk, valSize, elemIdx));
                }
            }
        }
    }
    return dense;
}

inline Eigen::MatrixXd EigenBuildTransposeRef(
    const CsrMatrix& csr, const Eigen::MatrixXd& denseIn,
    int mb, int idxBase,
    int rowBlockDimA, int colBlockDimA,
    int rowBlockDimC, int colBlockDimC, int nb)
{
    Eigen::MatrixXd denseRef = Eigen::MatrixXd::Zero(mb * rowBlockDimC, nb * colBlockDimC);
    for (int i = 0; i < mb; ++i) {
        int rowStart = csr.rowOffsets[i] - idxBase;
        int rowEnd = csr.rowOffsets[i + 1] - idxBase;
        for (int k = rowStart; k < rowEnd; ++k) {
            int bj = csr.colIndices[k] - idxBase;
            for (int ii = 0; ii < rowBlockDimA; ++ii)
                for (int jj = 0; jj < colBlockDimA; ++jj)
                    denseRef(i * rowBlockDimC + jj, bj * colBlockDimC + ii) =
                        denseIn(i * rowBlockDimA + ii, bj * colBlockDimA + jj);
        }
    }
    return denseRef;
}

inline void EigenValueCrossCheck(
    const CsrMatrix& csr, const std::vector<uint8_t>& blockValBytes,
    int mb, int nb, int idxBase, int dirA, int copyValues,
    int rowBlockDimA, int colBlockDimA,
    int rowBlockDimC, int colBlockDimC, int valSize,
    const Gebsr2GebscGoldenResult& result)
{
    if (copyValues != 1) return;

    Eigen::MatrixXd denseIn = EigenBuildDenseIn(
        csr, blockValBytes, mb, nb, idxBase, dirA,
        rowBlockDimA, colBlockDimA, valSize);

    Eigen::MatrixXd denseOut = EigenBuildDenseOut(
        result, mb, nb, idxBase, dirA,
        rowBlockDimC, colBlockDimC, valSize);

    int copyMode = DetermineCopyMode(rowBlockDimA, colBlockDimA, rowBlockDimC, colBlockDimC);
    Eigen::MatrixXd denseRef = denseIn;
    if (copyMode == 1) {
        denseRef = EigenBuildTransposeRef(csr, denseIn, mb, idxBase,
            rowBlockDimA, colBlockDimA, rowBlockDimC, colBlockDimC, nb);
    }
    if (!denseOut.isApprox(denseRef, 1e-3)) {
        throw std::logic_error("Eigen value cross-check failed");
    }
}

inline void Gebsr2GebscEigenCrossCheck(
    const CsrMatrix& csr,
    const std::vector<uint8_t>& blockValBytes,
    int mb, int nb, int nnzb,
    int rowBlockDimA, int colBlockDimA,
    int rowBlockDimC, int colBlockDimC,
    int idxBase, int dirA, int copyValues,
    int valSize,
    const Gebsr2GebscGoldenResult& result)
{
    EigenStructCrossCheck(csr, mb, nb, nnzb, idxBase, result);
    EigenValueCrossCheck(csr, blockValBytes,
        mb, nb, idxBase, dirA, copyValues,
        rowBlockDimA, colBlockDimA, rowBlockDimC, colBlockDimC,
        valSize, result);
}

#endif  // SPARSE_TEST_USE_EIGEN

inline Gebsr2GebscGoldenResult Gebsr2GebscGolden(
    const CsrMatrix& csr,
    const std::vector<uint8_t>& blockValBytes,
    int mb, int nb,
    int rowBlockDimA, int colBlockDimA,
    int rowBlockDimC, int colBlockDimC,
    int idxBase, int dirA, int copyValues,
    int valSize)
{
    int nnzb = static_cast<int>(csr.nnz);
    Gebsr2GebscGoldenResult result;
    result.nnzb = nnzb;
    result.bscColPtr.assign(nb + 1, idxBase);

    if (nnzb == 0) {
        return result;
    }

    result.bscRowInd.resize(nnzb);
    int blockSizeA = rowBlockDimA * colBlockDimA;
    int blockSizeC = rowBlockDimC * colBlockDimC;
    result.bscVal.resize(static_cast<size_t>(nnzb) * blockSizeC * valSize);

    int copyMode = DetermineCopyMode(
        rowBlockDimA, colBlockDimA, rowBlockDimC, colBlockDimC);

    std::vector<int32_t> colOffset;
    ComputeBscColPtr(csr, nb, nnzb, idxBase, result.bscColPtr, colOffset);

    ScatterBlocks(csr, mb, idxBase, copyValues,
        blockSizeA, blockSizeC, valSize,
        copyMode, rowBlockDimA, colBlockDimA,
        rowBlockDimC, colBlockDimC, dirA,
        blockValBytes, colOffset, result);

#ifdef SPARSE_TEST_USE_EIGEN
    Gebsr2GebscEigenCrossCheck(csr, blockValBytes,
        mb, nb, nnzb,
        rowBlockDimA, colBlockDimA,
        rowBlockDimC, colBlockDimC,
        idxBase, dirA, copyValues, valSize,
        result);
#endif

    return result;
}

}  // namespace sparse_test

#endif
