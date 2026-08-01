/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TEST_SPSV_NPU_WRAPPER_H_
#define TEST_SPSV_NPU_WRAPPER_H_

#include <cstdint>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "descriptor_manager.h"
#include "fill.h"
#include "sparse_test.h"

namespace sparse_test {

struct SpSVNpuResult {
    std::vector<float> y;
    std::vector<float> y2;
    bool hasSecondSolve = false;
};

struct SpSVNpuConfig {
    std::string format = "CSR";
    bool isI64 = false;
    bool isRowPtrI64 = false;  // effective rowPtr (or colOffset for CSC) index width
    bool isColIndI64 = false;  // effective colInd (or rowInd for CSC/COO) index width
    int32_t idxBase = 0;  // 0=INDEX_BASE_ZERO, 1=INDEX_BASE_ONE
    bool lower = true;
    bool unitDiag = false;
    aclsparseOperation_t opA = ACL_SPARSE_OP_NON_TRANSPOSE;
    bool inPlace = false;
    bool nullVec = false;
    std::string updateMode = "NONE";
    int32_t sliceWidth = 1;
};

struct SpSVDeviceSetup {
    DeviceBuffer dVals;
    DeviceBuffer dIdx0;
    DeviceBuffer dIdx1;
    SpMatManager matA;
};

static inline SpSVDeviceSetup SetupCsrDescriptor(const CsrMatrix& csr, int64_t m,
                                                   aclsparseIndexType_t rowPtrType,
                                                   aclsparseIndexType_t colIndType,
                                                   aclsparseIndexBase_t idxBase) {
    SpSVDeviceSetup setup;
    // Row offsets: independently sized by rowPtrType
    if (rowPtrType == ACL_SPARSE_INDEX_64I) {
        std::vector<int64_t> rowOff64(csr.rowOffsets.begin(), csr.rowOffsets.end());
        if (idxBase == ACL_SPARSE_INDEX_BASE_ONE) {
            for (auto& v : rowOff64) v++;
        }
        setup.dIdx0 = DeviceBuffer::copyFrom(rowOff64.data(), rowOff64.size() * sizeof(int64_t));
    } else {
        std::vector<int32_t> rowOff(csr.rowOffsets);
        if (idxBase == ACL_SPARSE_INDEX_BASE_ONE) {
            for (auto& v : rowOff) v++;
        }
        setup.dIdx0 = DeviceBuffer::copyFrom(rowOff.data(), rowOff.size() * sizeof(int32_t));
    }
    // Column indices: independently sized by colIndType
    if (colIndType == ACL_SPARSE_INDEX_64I) {
        std::vector<int64_t> colIdx64(csr.colIndices.begin(), csr.colIndices.end());
        if (idxBase == ACL_SPARSE_INDEX_BASE_ONE) {
            for (auto& v : colIdx64) v++;
        }
        setup.dIdx1 = DeviceBuffer::copyFrom(colIdx64.data(), colIdx64.size() * sizeof(int64_t));
    } else {
        std::vector<int32_t> colIdx(csr.colIndices);
        if (idxBase == ACL_SPARSE_INDEX_BASE_ONE) {
            for (auto& v : colIdx) v++;
        }
        setup.dIdx1 = DeviceBuffer::copyFrom(colIdx.data(), colIdx.size() * sizeof(int32_t));
    }
    setup.dVals = DeviceBuffer::copyFrom(csr.values.data(), csr.values.size() * sizeof(float));
    setup.matA = SpMatManager::createCsr(m, m, csr.nnz, setup.dIdx0.get(), setup.dIdx1.get(),
                                           setup.dVals.get(), rowPtrType, colIndType,
                                           idxBase, ACL_FLOAT);
    return setup;
}

static inline SpSVDeviceSetup SetupCscDescriptor(const CsrMatrix& csr, int64_t m,
                                                   aclsparseIndexType_t colOffType,
                                                   aclsparseIndexType_t rowIndType,
                                                   aclsparseIndexBase_t idxBase) {
    CscMatrix csc;
    csc.rows = m; csc.cols = m; csc.nnz = csr.nnz;
    csc.colOffsets.assign(m + 1, 0);
    csc.rowIndices.resize(csr.nnz);
    csc.values.resize(csr.nnz);
    for (int32_t c : csr.colIndices) csc.colOffsets[c + 1]++;
    for (int64_t c = 0; c < m; c++) csc.colOffsets[c + 1] += csc.colOffsets[c];
    std::vector<int32_t> counter(csc.colOffsets.begin(), csc.colOffsets.end() - 1);
    for (int64_t i = 0; i < m; i++) {
        for (int32_t k = csr.rowOffsets[i]; k < csr.rowOffsets[i + 1]; k++) {
            int32_t c = csr.colIndices[k];
            int32_t dest = counter[c]++;
            csc.rowIndices[dest] = static_cast<int32_t>(i);
            csc.values[dest] = csr.values[k];
        }
    }
    if (idxBase == ACL_SPARSE_INDEX_BASE_ONE) {
        for (auto& v : csc.colOffsets) v++;
        for (auto& v : csc.rowIndices) v++;
    }
    SpSVDeviceSetup setup;
    // Column offsets: independently sized by colOffType
    if (colOffType == ACL_SPARSE_INDEX_64I) {
        std::vector<int64_t> colOff64(csc.colOffsets.begin(), csc.colOffsets.end());
        setup.dIdx0 = DeviceBuffer::copyFrom(colOff64.data(), colOff64.size() * sizeof(int64_t));
    } else {
        setup.dIdx0 = DeviceBuffer::copyFrom(csc.colOffsets.data(), csc.colOffsets.size() * sizeof(int32_t));
    }
    // Row indices: independently sized by rowIndType
    if (rowIndType == ACL_SPARSE_INDEX_64I) {
        std::vector<int64_t> rowIdx64(csc.rowIndices.begin(), csc.rowIndices.end());
        setup.dIdx1 = DeviceBuffer::copyFrom(rowIdx64.data(), rowIdx64.size() * sizeof(int64_t));
    } else {
        setup.dIdx1 = DeviceBuffer::copyFrom(csc.rowIndices.data(), csc.rowIndices.size() * sizeof(int32_t));
    }
    setup.dVals = DeviceBuffer::copyFrom(csc.values.data(), csc.values.size() * sizeof(float));
    setup.matA = SpMatManager::createCsc(m, m, csc.nnz, setup.dIdx0.get(), setup.dIdx1.get(),
                                           setup.dVals.get(), colOffType, rowIndType,
                                           idxBase, ACL_FLOAT);
    return setup;
}

static inline SpSVDeviceSetup SetupCooDescriptor(const CsrMatrix& csr, int64_t m,
                                                   aclsparseIndexType_t idxType,
                                                   aclsparseIndexBase_t idxBase) {
    CooMatrix coo;
    coo.rows = m; coo.cols = m; coo.nnz = csr.nnz;
    coo.rowIndices.reserve(csr.nnz);
    coo.colIndices.reserve(csr.nnz);
    coo.values.reserve(csr.nnz);
    for (int64_t i = 0; i < m; i++) {
        for (int32_t k = csr.rowOffsets[i]; k < csr.rowOffsets[i + 1]; k++) {
            coo.rowIndices.push_back(static_cast<int32_t>(i));
            coo.colIndices.push_back(csr.colIndices[k]);
            coo.values.push_back(csr.values[k]);
        }
    }
    if (idxBase == ACL_SPARSE_INDEX_BASE_ONE) {
        for (auto& v : coo.rowIndices) v++;
        for (auto& v : coo.colIndices) v++;
    }
    SpSVDeviceSetup setup;
    if (idxType == ACL_SPARSE_INDEX_64I) {
        std::vector<int64_t> rowIdx64(coo.rowIndices.begin(), coo.rowIndices.end());
        std::vector<int64_t> colIdx64(coo.colIndices.begin(), coo.colIndices.end());
        setup.dIdx0 = DeviceBuffer::copyFrom(rowIdx64.data(), rowIdx64.size() * sizeof(int64_t));
        setup.dIdx1 = DeviceBuffer::copyFrom(colIdx64.data(), colIdx64.size() * sizeof(int64_t));
    } else {
        setup.dIdx0 = DeviceBuffer::copyFrom(coo.rowIndices.data(), coo.rowIndices.size() * sizeof(int32_t));
        setup.dIdx1 = DeviceBuffer::copyFrom(coo.colIndices.data(), coo.colIndices.size() * sizeof(int32_t));
    }
    setup.dVals = DeviceBuffer::copyFrom(coo.values.data(), coo.values.size() * sizeof(float));
    setup.matA = SpMatManager::createCoo(m, m, coo.nnz, setup.dIdx0.get(), setup.dIdx1.get(),
                                           setup.dVals.get(), idxType,
                                           idxBase, ACL_FLOAT);
    return setup;
}

static inline SpSVDeviceSetup SetupSlicedEllDescriptor(const CsrMatrix& csr, int64_t m,
                                                       aclsparseIndexType_t idxType,
                                                       aclsparseIndexBase_t idxBase,
                                                       int32_t sliceWidth = 1) {
    SlicedEllMatrix ell = csrToSlicedEll(csr, sliceWidth);
    if (idxBase == ACL_SPARSE_INDEX_BASE_ONE) {
        for (auto& v : ell.sliceOffsets) v++;
        for (auto& v : ell.colIndices) {
            if (v >= 0) v++;
        }
    }
    SpSVDeviceSetup setup;
    if (idxType == ACL_SPARSE_INDEX_64I) {
        std::vector<int64_t> sliceOff64(ell.sliceOffsets.begin(), ell.sliceOffsets.end());
        std::vector<int64_t> colIdx64(ell.colIndices.begin(), ell.colIndices.end());
        setup.dIdx0 = DeviceBuffer::copyFrom(sliceOff64.data(), sliceOff64.size() * sizeof(int64_t));
        setup.dIdx1 = DeviceBuffer::copyFrom(colIdx64.data(), colIdx64.size() * sizeof(int64_t));
    } else {
        setup.dIdx0 = DeviceBuffer::copyFrom(ell.sliceOffsets.data(), ell.sliceOffsets.size() * sizeof(int32_t));
        setup.dIdx1 = DeviceBuffer::copyFrom(ell.colIndices.data(), ell.colIndices.size() * sizeof(int32_t));
    }
    setup.dVals = DeviceBuffer::copyFrom(ell.values.data(), ell.values.size() * sizeof(float));
    setup.matA = SpMatManager::createSlicedEll(m, m, ell.nnz, setup.dIdx0.get(), setup.dIdx1.get(),
                                                setup.dVals.get(),
                                                static_cast<int64_t>(ell.sliceWidth),
                                                static_cast<int64_t>(ell.sliceOffsets.size()) - 1,
                                                idxType,
                                                idxBase, ACL_FLOAT);
    return setup;
}

static inline SpSVDeviceSetup SetupMatrixDescriptor(const CsrMatrix& csr, const SpSVNpuConfig& cfg) {
    int64_t m = csr.rows;
    aclsparseIndexType_t rowPtrType = cfg.isRowPtrI64 ? ACL_SPARSE_INDEX_64I : ACL_SPARSE_INDEX_32I;
    aclsparseIndexType_t colIndType = cfg.isColIndI64 ? ACL_SPARSE_INDEX_64I : ACL_SPARSE_INDEX_32I;
    aclsparseIndexBase_t idxBase = (cfg.idxBase == 1) ? ACL_SPARSE_INDEX_BASE_ONE : ACL_SPARSE_INDEX_BASE_ZERO;
    if (cfg.format == "CSR") return SetupCsrDescriptor(csr, m, rowPtrType, colIndType, idxBase);
    if (cfg.format == "CSC") return SetupCscDescriptor(csr, m, rowPtrType, colIndType, idxBase);
    // COO and SLICED_ELL use a single index type for both arrays; map via colIndType
    if (cfg.format == "COO") return SetupCooDescriptor(csr, m, colIndType, idxBase);
    if (cfg.format == "SLICED_ELL") return SetupSlicedEllDescriptor(csr, m, colIndType, idxBase, cfg.sliceWidth);
    throw std::runtime_error("Unsupported format: " + cfg.format);
}

static inline DeviceBuffer RunAnalysis(aclsparseHandle_t handle, const SpSVNpuConfig& cfg,
                                        float alpha, SpMatManager& matA,
                                        DnVecManager& vecX, DnVecManager& vecY,
                                        SpSVDescrManager& spsvDescr) {
    size_t bufferSize = 0;
    auto ret = aclsparseSpSV_bufferSize(
        handle, cfg.opA, &alpha, matA.cget(),
        cfg.nullVec ? nullptr : vecX.cget(),
        cfg.nullVec ? nullptr : vecY.get(),
        ACL_FLOAT, ACL_SPARSE_SPSV_ALG_DEFAULT, spsvDescr.get(), &bufferSize);
    if (ret != ACL_SPARSE_STATUS_SUCCESS) {
        throw std::runtime_error("aclsparseSpSV_bufferSize failed, status=" + std::to_string(ret));
    }
    DeviceBuffer dWorkspace;
    if (bufferSize > 0) dWorkspace = DeviceBuffer::alloc(bufferSize);
    ret = aclsparseSpSV_analysis(
        handle, cfg.opA, &alpha, matA.cget(),
        cfg.nullVec ? nullptr : vecX.cget(),
        cfg.nullVec ? nullptr : vecY.get(),
        ACL_FLOAT, ACL_SPARSE_SPSV_ALG_DEFAULT, spsvDescr.get(),
        bufferSize > 0 ? dWorkspace.get() : nullptr);
    if (ret != ACL_SPARSE_STATUS_SUCCESS) {
        throw std::runtime_error("aclsparseSpSV_analysis failed, status=" + std::to_string(ret));
    }
    return dWorkspace;
}

static inline void RunSolve(aclsparseHandle_t handle, aclrtStream stream,
                             const SpSVNpuConfig& cfg, float alpha,
                             SpMatManager& matA, DnVecManager& vecX, DnVecManager& vecY,
                             SpSVDescrManager& spsvDescr, DeviceBuffer& dY,
                             std::vector<float>& outY) {
    auto ret = aclsparseSpSV_solve(
        handle, cfg.opA, &alpha, matA.cget(), vecX.cget(), vecY.get(),
        ACL_FLOAT, ACL_SPARSE_SPSV_ALG_DEFAULT, spsvDescr.get());
    if (ret != ACL_SPARSE_STATUS_SUCCESS) {
        throw std::runtime_error("aclsparseSpSV_solve failed, status=" + std::to_string(ret));
    }
    if (aclrtSynchronizeStream(stream) != ACL_SUCCESS) {
        throw std::runtime_error("aclrtSynchronizeStream failed after solve");
    }
    int64_t m = static_cast<int64_t>(outY.size());
    dY.copyToHost(outY.data(), m * sizeof(float));
}

/**
 * @brief SELL + GENERAL 模式更新：newVals 按 SELL 槽位布局组织
 * @details SELL 槽位布局与 CSR 顺序不同（除非 maxRowNnz=1），
 *          kernel 通过 perm[csrIdx] 映射 CSR 位置 → SELL 槽位位置，
 *          因此 updateMatrix 读取的是 SELL 槽位索引的值.
 */
static inline void RunUpdateSellGeneral(aclsparseHandle_t handle,
                                         SpSVDescrManager& spsvDescr,
                                         int32_t sliceWidth, const CsrMatrix& csr) {
    std::mt19937 rng(9999);
    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);

    const int32_t sw = sliceWidth;
    int32_t maxRowNnz = 0;
    for (int64_t i = 0; i < csr.rows; i++) {
        int32_t rNnz = csr.rowOffsets[i + 1] - csr.rowOffsets[i];
        if (rNnz > maxRowNnz) maxRowNnz = rNnz;
    }
    int32_t numSlices = (maxRowNnz + sw - 1) / sw;
    if (numSlices == 0) numSlices = 1;
    // sliceOffsets[s] = s * rows * sw (均匀切片)
    const size_t totalSlots = static_cast<size_t>(numSlices) * csr.rows * sw;
    std::vector<float> newVals(totalSlots, 0.0f);

    for (int64_t i = 0; i < csr.rows; i++) {
        float rowAbsSum = 0.0f;
        int32_t diagSellPos = -1;
        int32_t localIdx = 0;
        for (int32_t k = csr.rowOffsets[i]; k < csr.rowOffsets[i + 1]; k++) {
            int32_t s = localIdx / sw;
            int32_t off = localIdx % sw;
            int32_t sellPos = s * static_cast<int32_t>(csr.rows) * sw +
                              static_cast<int32_t>(i) * sw + off;
            if (csr.colIndices[k] == static_cast<int32_t>(i)) {
                diagSellPos = sellPos;
            } else {
                float v = dist(rng);
                newVals[sellPos] = v;
                rowAbsSum += std::abs(v);
            }
            localIdx++;
        }
        if (diagSellPos >= 0) {
            newVals[diagSellPos] = rowAbsSum + 1.0f;
        }
    }

    auto dNewVals = DeviceBuffer::copyFrom(newVals.data(), newVals.size() * sizeof(float));
    auto ret = aclsparseSpSV_updateMatrix(handle, spsvDescr.get(), dNewVals.get(),
                                           ACL_SPARSE_SPSV_UPDATE_GENERAL);
    if (ret != ACL_SPARSE_STATUS_SUCCESS) {
        throw std::runtime_error("aclsparseSpSV_updateMatrix failed, status=" + std::to_string(ret));
    }
}

/**
 * @brief CSR/COO 格式更新（GENERAL 或 DIAGONAL 模式）
 * @details newVals 按 CSR 顺序索引；GENERAL 模式构造对角占优保证收敛性，
 *          DIAGONAL 模式生成随机值（仅对角线位置被 kernel 读取）.
 */
static inline void RunUpdateCsrLayout(aclsparseHandle_t handle,
                                       SpSVDescrManager& spsvDescr,
                                       const std::string& updateMode,
                                       const CsrMatrix& csr) {
    std::mt19937 rng(9999);
    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);

    std::vector<float> newVals(csr.values.size());
    if (updateMode == "GENERAL") {
        for (int64_t i = 0; i < csr.rows; i++) {
            float rowAbsSum = 0.0f;
            int32_t diagIdx = -1;
            for (int32_t k = csr.rowOffsets[i]; k < csr.rowOffsets[i + 1]; k++) {
                if (csr.colIndices[k] == static_cast<int32_t>(i)) {
                    diagIdx = k;
                } else {
                    float v = dist(rng);
                    newVals[k] = v;
                    rowAbsSum += std::abs(v);
                }
            }
            if (diagIdx >= 0) {
                newVals[diagIdx] = rowAbsSum + 1.0f;
            }
        }
    } else {
        // DIAGONAL 模式: 全量生成随机值, 仅对角线位置被 kernel 读取
        for (size_t i = 0; i < newVals.size(); i++) newVals[i] = dist(rng);
    }
    auto dNewVals = DeviceBuffer::copyFrom(newVals.data(), newVals.size() * sizeof(float));
    aclsparseSpSVUpdate_t updatePart = (updateMode == "DIAGONAL")
        ? ACL_SPARSE_SPSV_UPDATE_DIAGONAL : ACL_SPARSE_SPSV_UPDATE_GENERAL;
    auto ret = aclsparseSpSV_updateMatrix(handle, spsvDescr.get(), dNewVals.get(), updatePart);
    if (ret != ACL_SPARSE_STATUS_SUCCESS) {
        throw std::runtime_error("aclsparseSpSV_updateMatrix failed, status=" + std::to_string(ret));
    }
}

/**
 * @brief CSC 格式更新（GENERAL 或 DIAGONAL 模式）
 * @details newVals 按 CSC 布局（column-major of A^T, 即按列存储 A^T,
 *          等价于按行存储 A 的转置）索引。cuSPARSE 规范要求 CSC 的 newValues
 *          必须按 CSC 列主序排列，才能正确触发 perm 映射路径。
 *          实现采用"列计数 - 前缀和 - scatter"三步法：从 CSR 表示的矩阵遍历，
 *          按列散射到 CSC 顺序的 newVals 数组。
 */
static inline void RunUpdateCsc(aclsparseHandle_t handle,
                                 SpSVDescrManager& spsvDescr,
                                 const std::string& updateMode,
                                 const CsrMatrix& csr) {
    std::mt19937 rng(9999);
    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);

    const int64_t m = csr.rows;
    const size_t nnz = csr.nnz;
    std::vector<float> newVals(nnz);

    if (updateMode == "GENERAL") {
        // Pass 1: 统计每列（A^T 的列 = A 的行）的非零元个数
        std::vector<int64_t> colCount(m, 0);
        for (int64_t i = 0; i < m; i++) {
            for (int32_t k = csr.rowOffsets[i]; k < csr.rowOffsets[i + 1]; k++) {
                colCount[csr.colIndices[k]]++;
            }
        }
        // Pass 2: 计算每列的起始偏移 (exclusive prefix sum)
        std::vector<int64_t> colOffset(m + 1, 0);
        for (int64_t j = 0; j < m; j++) {
            colOffset[j + 1] = colOffset[j] + colCount[j];
        }
        // Pass 3: Scatter - 为每个非零元生成新值，按 CSC 顺序写入 newVals
        // 遍历时维护每行（原始矩阵）的对角占优约束：对非对角元生成随机值，最后设置对角元
        std::vector<int64_t> writePos(colOffset.begin(), colOffset.end() - 1);
        std::vector<float> rowAbsSum(m, 0.0f);
        std::vector<int64_t> diagCscPos(m, -1);

        // 先写入非对角元
        for (int64_t i = 0; i < m; i++) {
            for (int32_t k = csr.rowOffsets[i]; k < csr.rowOffsets[i + 1]; k++) {
                int64_t j = csr.colIndices[k];
                int64_t pos = writePos[j]++;
                if (static_cast<int32_t>(i) == csr.colIndices[k]) {
                    // 对角元，暂时记录位置
                    diagCscPos[i] = pos;
                } else {
                    float v = dist(rng);
                    newVals[pos] = v;
                    rowAbsSum[i] += std::abs(v);
                }
            }
        }
        // 设置对角元，保证对角占优
        for (int64_t i = 0; i < m; i++) {
            if (diagCscPos[i] >= 0) {
                newVals[diagCscPos[i]] = rowAbsSum[i] + 1.0f;
            }
        }
    } else {
        // DIAGONAL 模式: 全量生成 CSC 顺序的随机值
        for (size_t i = 0; i < nnz; i++) newVals[i] = dist(rng);
    }

    auto dNewVals = DeviceBuffer::copyFrom(newVals.data(), newVals.size() * sizeof(float));
    aclsparseSpSVUpdate_t updatePart = (updateMode == "DIAGONAL")
        ? ACL_SPARSE_SPSV_UPDATE_DIAGONAL : ACL_SPARSE_SPSV_UPDATE_GENERAL;
    auto ret = aclsparseSpSV_updateMatrix(handle, spsvDescr.get(), dNewVals.get(), updatePart);
    if (ret != ACL_SPARSE_STATUS_SUCCESS) {
        throw std::runtime_error("aclsparseSpSV_updateMatrix failed, status=" + std::to_string(ret));
    }
}

/**
 * @brief RunUpdate 主调度器：根据 format/mode 路由至对应更新实现，
 *        完成后调用 RunSolve 获取更新后的解.
 */
static inline void RunUpdate(aclsparseHandle_t handle, aclrtStream stream,
                              const SpSVNpuConfig& cfg, float alpha,
                              const CsrMatrix& csr, SpMatManager& matA,
                              DnVecManager& vecX, DnVecManager& vecY,
                              SpSVDescrManager& spsvDescr, DeviceBuffer& dY,
                              SpSVNpuResult& result) {
    if (cfg.updateMode == "GENERAL" && cfg.format == "SLICED_ELL") {
        RunUpdateSellGeneral(handle, spsvDescr, cfg.sliceWidth, csr);
    } else if (cfg.format == "CSC") {
        RunUpdateCsc(handle, spsvDescr, cfg.updateMode, csr);
    } else {
        RunUpdateCsrLayout(handle, spsvDescr, cfg.updateMode, csr);
    }

    int64_t m = csr.rows;
    result.y2.resize(m);
    RunSolve(handle, stream, cfg, alpha, matA, vecX, vecY, spsvDescr, dY, result.y2);
    result.hasSecondSolve = true;
}

inline SpSVNpuResult SpSVNpuWrapper(
    aclsparseHandle_t handle, aclrtStream stream,
    const CsrMatrix& csr,
    const std::vector<float>& xVec,
    float alpha,
    const SpSVNpuConfig& cfg)
{
    int64_t m = csr.rows;
    SpSVNpuResult result;
    result.y.resize(m, 0.0f);
    if (m <= 0) return result;

    auto setup = SetupMatrixDescriptor(csr, cfg);
    aclsparseFillMode_t fillMode = cfg.lower ? ACL_SPARSE_FILL_MODE_LOWER : ACL_SPARSE_FILL_MODE_UPPER;
    aclsparseDiagType_t diagType = cfg.unitDiag ? ACL_SPARSE_DIAG_TYPE_UNIT : ACL_SPARSE_DIAG_TYPE_NON_UNIT;
    setup.matA.setAttribute(ACL_SPARSE_SPMAT_FILL_MODE, &fillMode, sizeof(fillMode));
    setup.matA.setAttribute(ACL_SPARSE_SPMAT_DIAG_TYPE, &diagType, sizeof(diagType));

    auto dX = DeviceBuffer::copyFrom(xVec.data(), xVec.size() * sizeof(float));
    DeviceBuffer dY;
    if (cfg.inPlace) {
        dY = DeviceBuffer::alloc(m * sizeof(float));
        auto ret = aclrtMemcpy(dY.get(), m * sizeof(float), dX.get(), m * sizeof(float), ACL_MEMCPY_DEVICE_TO_DEVICE);
        if (ret != ACL_SUCCESS) throw std::runtime_error("aclrtMemcpy D2D failed");
    } else {
        std::vector<float> yInit(m, 0.0f);
        dY = DeviceBuffer::copyFrom(yInit.data(), yInit.size() * sizeof(float));
    }

    auto vecX = DnVecManager::createConst(m, dX.get(), ACL_FLOAT);
    auto vecY = DnVecManager::create(m, dY.get(), ACL_FLOAT);
    SpSVDescrManager spsvDescr;

    auto dWorkspace = RunAnalysis(handle, cfg, alpha, setup.matA, vecX, vecY, spsvDescr);
    RunSolve(handle, stream, cfg, alpha, setup.matA, vecX, vecY, spsvDescr, dY, result.y);

    if (cfg.updateMode == "GENERAL" || cfg.updateMode == "DIAGONAL") {
        RunUpdate(handle, stream, cfg, alpha, csr, setup.matA, vecX, vecY, spsvDescr, dY, result);
    }

    return result;
}

}

#endif
