// ----------------------------------------------------------------------------------------------------------
// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software; you can redistribute it and/or modify it under the terms of conditions of
// CANN Open Software License Agreement Version 2 (the "License").
// You may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License is
// distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and limitations under the License.
// ----------------------------------------------------------------------------------------------------------
// DenseToSparse arch22 NPU wrapper: device/handle environment and the
// three-stage C-API protocol executor (GetBufferSize -> Analysis ->
// Convert) used by the UT.
#ifndef DENSE_TO_SPARSE_ARCH22_NPU_WRAPPER_H
#define DENSE_TO_SPARSE_ARCH22_NPU_WRAPPER_H
#include <acl/acl.h>
#include "cann_ops_sparse.h"

#include <cstdint>
#include <random>
#include <string>

#include "densetosparse_ref.h"

static aclrtStream dts_stream = nullptr;
static aclsparseHandle_t dts_handle = nullptr;

// ===================== 协议执行器 =====================

struct RunOutcome {
    bool ok = false;
    std::string why;
    int64_t nnz = 0;
};

// BELL pattern for the gate cases: deterministic pseudo-random block
// columns with a -1 empty slot on every odd row's tail.
std::vector<int32_t> MakeBellPattern(const CaseCfg &cfg)
{
    const int blockRows = cfg.rows / cfg.bellB;
    const int slots = cfg.ellCols / cfg.bellB;
    const int colBlocks = cfg.cols / cfg.bellB;
    std::vector<int32_t> pattern(static_cast<size_t>(blockRows) * slots);
    std::mt19937 prng(cfg.seed + 7);
    for (int br = 0; br < blockRows; ++br) {
        for (int s = 0; s < slots; ++s) {
            if (s == slots - 1 && (br & 1)) {
                pattern[br * slots + s] = -1;
            } else {
                pattern[br * slots + s] =
                    static_cast<int32_t>((br + s * 3) % colBlocks) +
                    static_cast<int32_t>(cfg.base);
            }
        }
    }
    return pattern;
}

// Device pattern/values buffers + BlockedEll descriptor.
bool SetupBellCase(const CaseCfg &cfg, const Dense64 &d, void **devPattern,
    void **devBellValues, aclsparseSpMatDescr_t *matB)
{
    std::vector<int32_t> pattern = MakeBellPattern(cfg);
    const size_t valCount = static_cast<size_t>(cfg.rows) * cfg.ellCols;
    aclrtMalloc(devPattern, pattern.size() * 4, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMemcpy(*devPattern, pattern.size() * 4, pattern.data(),
                pattern.size() * 4, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMalloc(devBellValues, valCount * d.elemBytes,
                ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMemset(*devBellValues, valCount * d.elemBytes, 0xAA,
                valCount * d.elemBytes);
    return aclsparseCreateBlockedEll(matB, cfg.rows, cfg.cols, cfg.bellB,
                                     cfg.ellCols, *devPattern,
                                     *devBellValues, ACL_SPARSE_INDEX_32I,
                                     cfg.base,
                                     cfg.dtype) == ACL_SPARSE_STATUS_SUCCESS;
}

// CSR/CSC/COO descriptor (offsets buffer for the CSR/CSC runs).
bool SetupCompressedCase(const CaseCfg &cfg, const Golden &g,
    void **devOffsets, aclsparseSpMatDescr_t *matB)
{
    if (cfg.format != ACL_SPARSE_FORMAT_COO) {
        aclrtMalloc(devOffsets, (g.majorDim + 1) * sizeof(int32_t),
                    ACL_MEM_MALLOC_HUGE_FIRST);
    }
    const aclsparseStatus_t cs =
        cfg.format == ACL_SPARSE_FORMAT_CSR
            ? aclsparseCreateCsr(matB, cfg.rows, cfg.cols, 0, *devOffsets,
                                 nullptr, nullptr, ACL_SPARSE_INDEX_32I,
                                 ACL_SPARSE_INDEX_32I, cfg.base, cfg.dtype)
            : cfg.format == ACL_SPARSE_FORMAT_CSC
                  ? aclsparseCreateCsc(matB, cfg.rows, cfg.cols, 0,
                                       *devOffsets, nullptr, nullptr,
                                       ACL_SPARSE_INDEX_32I,
                                       ACL_SPARSE_INDEX_32I, cfg.base,
                                       cfg.dtype)
                  : aclsparseCreateCoo(matB, cfg.rows, cfg.cols, 0, nullptr,
                                       nullptr, nullptr,
                                       ACL_SPARSE_INDEX_32I, cfg.base,
                                       cfg.dtype);
    return cs == ACL_SPARSE_STATUS_SUCCESS;
}

// One block's value check (tail slots emit zeros, -1 slots all-zero).
bool BellBlockMatches(const CaseCfg &cfg, const Dense64 &d,
    const std::vector<uint8_t> &hostVals,
    const std::vector<int32_t> &pattern, int slots, int colBlocks, int br,
    int s)
{
    const int b = cfg.bellB;
    const int t = br * slots + s;
    const int32_t enc = pattern[t];
    const bool valid =
        enc >= (int)cfg.base &&
        (uint32_t)(enc - (int)cfg.base) < (uint32_t)colBlocks;
    for (int ic = 0; ic < b; ++ic) {
        for (int ir = 0; ir < b; ++ir) {
            const size_t pos =
                ((size_t)t * b * b + ic * b + ir) * d.elemBytes;
            const uint8_t *exp;
            uint8_t zero[8] = {0};
            if (valid) {
                exp = d.At(br * b + ir, (enc - (int)cfg.base) * b + ic);
            } else {
                exp = zero;
            }
            if (std::memcmp(hostVals.data() + pos, exp, d.elemBytes) != 0) {
                return false;
            }
        }
    }
    return true;
}

// BELL verification: download pattern + values and compare every block
// element (tail slots emit zeros, -1 slots are all-zero).
bool CheckBellResult(const CaseCfg &cfg, const Dense64 &d, void *devPattern,
    void *devBellValues)
{
    const int blockRows = cfg.rows / cfg.bellB;
    const int slots = cfg.ellCols / cfg.bellB;
    const int colBlocks = cfg.cols / cfg.bellB;
    std::vector<int32_t> pattern(static_cast<size_t>(blockRows) * slots);
    aclrtMemcpy(pattern.data(), pattern.size() * 4, devPattern,
                pattern.size() * 4, ACL_MEMCPY_DEVICE_TO_HOST);
    const size_t valCount = static_cast<size_t>(cfg.rows) * cfg.ellCols;
    std::vector<uint8_t> hostVals(valCount * d.elemBytes);
    aclrtMemcpy(hostVals.data(), hostVals.size(), devBellValues,
                hostVals.size(), ACL_MEMCPY_DEVICE_TO_HOST);
    for (int br = 0; br < blockRows; ++br) {
        for (int s = 0; s < slots; ++s) {
            if (!BellBlockMatches(cfg, d, hostVals, pattern, slots,
                                  colBlocks, br, s)) {
                return false;
            }
        }
    }
    return true;
}

// Compressed-format payload allocation, pointer rebind and Convert.
bool RunCompressedConvert(const CaseCfg &cfg, const Dense64 &d, int64_t nnz,
    aclsparseDnMatDescr_t matA, aclsparseSpMatDescr_t matB, void *workspace,
    void *devOffsets, void **devIndices, void **devRows, void **devCols,
    void **devValues)
{
    if (nnz > 0) {
        aclrtMalloc(devIndices, nnz * 4, ACL_MEM_MALLOC_HUGE_FIRST);
        aclrtMalloc(devValues, nnz * d.elemBytes,
                    ACL_MEM_MALLOC_HUGE_FIRST);
    }
    if (cfg.format == ACL_SPARSE_FORMAT_CSR) {
        aclsparseCsrSetPointers(matB, devOffsets, *devIndices, *devValues);
    } else if (cfg.format == ACL_SPARSE_FORMAT_CSC) {
        aclsparseCscSetPointers(matB, devOffsets, *devIndices, *devValues);
    } else {
        aclrtMalloc(devRows, nnz * 4, ACL_MEM_MALLOC_HUGE_FIRST);
        aclrtMalloc(devCols, nnz * 4, ACL_MEM_MALLOC_HUGE_FIRST);
        aclsparseCooSetPointers(matB, *devRows, *devCols, *devValues);
    }
    return aclsparseDenseToSparseConvert(
               dts_handle, matA, matB, ACL_SPARSE_DENSETOSPARSE_ALG_DEFAULT,
               workspace) == ACL_SPARSE_STATUS_SUCCESS &&
           aclrtSynchronizeStream(dts_stream) == ACL_SUCCESS;
}

// Compressed-format verification against the golden (offsets, values,
// per-format index arrays).
bool CheckCompressedResult(const CaseCfg &cfg, const Dense64 &d,
    const Golden &g, int64_t nnz, void *devOffsets, void *devIndices,
    void *devRows, void *devCols, void *devValues)
{
    if (cfg.format != ACL_SPARSE_FORMAT_COO) {
        std::vector<int32_t> hostOff(g.majorDim + 1);
        aclrtMemcpy(hostOff.data(), hostOff.size() * 4, devOffsets,
                    hostOff.size() * 4, ACL_MEMCPY_DEVICE_TO_HOST);
        if (hostOff != g.offsets) {
            return false;
        }
    }
    if (nnz <= 0) {
        return true;
    }
    std::vector<uint8_t> hostVal(nnz * d.elemBytes);
    aclrtMemcpy(hostVal.data(), hostVal.size(), devValues, hostVal.size(),
                ACL_MEMCPY_DEVICE_TO_HOST);
    if (std::memcmp(hostVal.data(), g.values.data(), hostVal.size()) != 0) {
        return false;
    }
    if (cfg.format == ACL_SPARSE_FORMAT_COO) {
        std::vector<int32_t> hostR(nnz), hostC(nnz);
        aclrtMemcpy(hostR.data(), nnz * 4, devRows, nnz * 4,
                    ACL_MEMCPY_DEVICE_TO_HOST);
        aclrtMemcpy(hostC.data(), nnz * 4, devCols, nnz * 4,
                    ACL_MEMCPY_DEVICE_TO_HOST);
        return hostR == g.rows && hostC == g.cols;
    }
    std::vector<int32_t> hostI(nnz);
    aclrtMemcpy(hostI.data(), nnz * 4, devIndices, nnz * 4,
                ACL_MEMCPY_DEVICE_TO_HOST);
    return hostI == (cfg.format == ACL_SPARSE_FORMAT_CSC ? g.cols : g.rows);
}

// BELL branch: Convert then verify the pattern + values round trip.
void RunBellVerify(const CaseCfg &cfg, const Dense64 &d,
    aclsparseDnMatDescr_t matA, aclsparseSpMatDescr_t matB, void *workspace,
    void *devPattern, void *devBellValues, RunOutcome &out)
{
    if (aclsparseDenseToSparseConvert(
            dts_handle, matA, matB, ACL_SPARSE_DENSETOSPARSE_ALG_DEFAULT,
            workspace) != ACL_SPARSE_STATUS_SUCCESS ||
        aclrtSynchronizeStream(dts_stream) != ACL_SUCCESS) {
        out.why = "BELL convert";
        return;
    }
    out.ok = CheckBellResult(cfg, d, devPattern, devBellValues);
    out.why = out.ok ? "" : "bell value mismatch";
    out.nnz = 0;
}

// CSR/CSC/COO branch: read nnz, rebind the payload, Convert, verify.
void RunCompressedVerify(const CaseCfg &cfg, const Dense64 &d,
    const Golden &g, aclsparseDnMatDescr_t matA, aclsparseSpMatDescr_t matB,
    void *workspace, void *devOffsets, void *devIndices, void *devRows,
    void *devCols, void *devValues, RunOutcome &out)
{
    int64_t nnz = 0;
    aclsparseSpMatGetSize(matB, nullptr, nullptr, &nnz);
    out.nnz = nnz;
    out.ok = nnz == g.nnz;
    if (!out.ok) {
        out.why = "nnz " + std::to_string(nnz) + " != " +
                  std::to_string(g.nnz);
    }
    if (!RunCompressedConvert(cfg, d, nnz, matA, matB, workspace,
                              devOffsets, &devIndices, &devRows, &devCols,
                              &devValues)) {
        out.why = "Convert";
        return;
    }
    if (out.ok && !CheckCompressedResult(cfg, d, g, nnz, devOffsets,
                                         devIndices, devRows, devCols,
                                         devValues)) {
        out.ok = false;
        out.why = "result mismatch";
    }
}

// Device buffers and descriptors of one protocol case.
struct CaseMem {
    void *devDense = nullptr;
    void *devOffsets = nullptr;
    void *devIndices = nullptr;
    void *devRows = nullptr;
    void *devCols = nullptr;
    void *devValues = nullptr;
    void *devPattern = nullptr;
    void *devBellValues = nullptr;
    void *workspace = nullptr;
    aclsparseDnMatDescr_t matA = nullptr;
    aclsparseSpMatDescr_t matB = nullptr;
};

// Dense upload, descriptor pair, workspace and Analysis stage.
bool SetupCase(const CaseCfg &cfg, const Dense64 &d, const Golden &g,
    CaseMem &m, RunOutcome &out)
{
    if (aclrtMalloc(&m.devDense, d.bytes.size() ? d.bytes.size() : 32,
                    ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS ||
        aclrtMemcpy(m.devDense, d.bytes.size(), d.bytes.data(),
                    d.bytes.size(),
                    ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS) {
        out.why = "dense upload";
        return false;
    }
    if (aclsparseCreateDnMat(&m.matA, cfg.rows, cfg.cols, d.ld, m.devDense,
                             cfg.dtype, cfg.order) !=
        ACL_SPARSE_STATUS_SUCCESS) {
        out.why = "CreateDnMat";
        return false;
    }
    if (cfg.format == ACL_SPARSE_FORMAT_BLOCKED_ELL) {
        if (!SetupBellCase(cfg, d, &m.devPattern, &m.devBellValues,
                           &m.matB)) {
            out.why = "CreateBlockedEll";
            return false;
        }
    } else if (!SetupCompressedCase(cfg, g, &m.devOffsets, &m.matB)) {
        out.why = "CreateSpMat";
        return false;
    }
    size_t bufferSize = 0;
    aclsparseDenseToSparseGetBufferSize(
        dts_handle, m.matA, m.matB, ACL_SPARSE_DENSETOSPARSE_ALG_DEFAULT,
        &bufferSize);
    if (bufferSize > 0) {
        aclrtMalloc(&m.workspace, bufferSize, ACL_MEM_MALLOC_HUGE_FIRST);
    }
    if (aclsparseDenseToSparseAnalysis(
            dts_handle, m.matA, m.matB, ACL_SPARSE_DENSETOSPARSE_ALG_DEFAULT,
            m.workspace) != ACL_SPARSE_STATUS_SUCCESS) {
        out.why = "Analysis";
        return false;
    }
    return true;
}

// 输入只读校验 + descriptor/buffer teardown.
void FinishCase(const Dense64 &d, CaseMem &m, RunOutcome &out)
{
    std::vector<uint8_t> back(d.bytes.size());
    aclrtMemcpy(back.data(), back.size(), m.devDense, back.size(),
                ACL_MEMCPY_DEVICE_TO_HOST);
    if (back != d.bytes) {
        out.ok = false;
        out.why = "dense input mutated";
    }
    aclsparseDestroySpMat(m.matB);
    aclsparseDestroyDnMat(m.matA);
    if (m.workspace) aclrtFree(m.workspace);
    if (m.devBellValues) aclrtFree(m.devBellValues);
    if (m.devPattern) aclrtFree(m.devPattern);
    if (m.devValues) aclrtFree(m.devValues);
    if (m.devRows) aclrtFree(m.devRows);
    if (m.devCols) aclrtFree(m.devCols);
    if (m.devIndices) aclrtFree(m.devIndices);
    if (m.devOffsets) aclrtFree(m.devOffsets);
    aclrtFree(m.devDense);
}

RunOutcome RunConvertCase(const CaseCfg &cfg)
{
    RunOutcome out;
    std::mt19937 rng(cfg.seed);
    Dense64 d = MakeDense(cfg, rng);
    Golden g = MakeGolden(cfg, d);

    CaseMem m;
    if (!SetupCase(cfg, d, g, m, out)) {
        return out;
    }
    if (cfg.format == ACL_SPARSE_FORMAT_BLOCKED_ELL) {
        RunBellVerify(cfg, d, m.matA, m.matB, m.workspace, m.devPattern,
                      m.devBellValues, out);
    } else {
        RunCompressedVerify(cfg, d, g, m.matA, m.matB, m.workspace,
                            m.devOffsets, m.devIndices, m.devRows,
                            m.devCols, m.devValues, out);
    }
    FinishCase(d, m, out);
    return out;
}

#endif  // DENSE_TO_SPARSE_ARCH22_NPU_WRAPPER_H
