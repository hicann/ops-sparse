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

// End-to-end test for aclsparseSpMM with FP32 / FP16 / INT8 data types.
//
//   C = alpha * A * B + beta * C
//
// Three test cases:
//   FP32: A(fp32) * B(fp32) -> C(fp32),  computeType=fp32
//   FP16: A(fp16) * B(fp16) -> C(fp16),  computeType=fp32  (accumulate in fp32)
//   INT8: A(int8) * B(int8) -> C(int32), computeType=int32 (accumulate in int32)

#include <acl/acl.h>
#include <cann_ops_sparse.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <new>
#include <set>
#include <string>
#include <vector>

using Clock = std::chrono::high_resolution_clock;
using TimePoint = std::chrono::time_point<Clock>;

static double ElapsedMs(const TimePoint &start, const TimePoint &end)
{
    return std::chrono::duration<double, std::milli>(end - start).count();
}

struct VerifyResult {
    bool pass;
    double mere;
    double mare;
    double mae;
};

static VerifyResult FailVerify()
{
    return VerifyResult{false, 0.0, 0.0, 0.0};
}

#define CHECK_RET(cond, return_expr) \
    do                               \
    {                                \
        if (!(cond))                 \
        {                            \
            return_expr;             \
        }                            \
    } while (0)

#define LOG_PRINT(message, ...)     \
    do                              \
    {                               \
        printf(message, ##__VA_ARGS__); \
    } while (0)

namespace {

int Init(int32_t deviceId, aclrtStream *stream)
{
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    return 0;
}

void Finalize(int32_t deviceId, aclrtStream stream)
{
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
}

void GenerateRandomCsr(int32_t m, int32_t k, int32_t nnz,
                      std::vector<int32_t> *rowOff,
                      std::vector<int32_t> *colInd,
                      std::vector<float>   *vals)
{
    rowOff->assign(static_cast<size_t>(m) + 1, 0);
    colInd->clear();
    vals->clear();
    colInd->reserve(nnz);
    vals->reserve(nnz);

    if (m <= 0 || k <= 0) {
        return;
    }

    int32_t produced = 0;
    for (int32_t i = 0; i < m; ++i) {
        int32_t rowNnz = nnz / m + (i < (nnz % m) ? 1 : 0);
        if (rowNnz > k) {
            rowNnz = k;
        }
        std::set<int32_t> picked;
        while (static_cast<int32_t>(picked.size()) < rowNnz) {
            picked.insert(std::rand() % k);
        }
        for (int32_t col : picked) {
            colInd->push_back(col);
            vals->push_back(-5.0f + 10.0f * static_cast<float>(produced % 10001) / 10000.0f);
            ++produced;
        }
        (*rowOff)[i + 1] = produced;
    }
}

void SpmmCpuFp32(int32_t m, int32_t n,
                 const std::vector<int32_t> &rowOff,
                 const std::vector<int32_t> &colInd,
                 const std::vector<float>   &vals,
                 const std::vector<float>   &B,
                 int32_t ldb, bool bRowMajor, bool opBTranspose,
                 std::vector<float>         *C,
                 int32_t ldc, bool cRowMajor,
                 float alpha, float beta)
{
    for (int32_t r = 0; r < m; ++r) {
        for (int32_t j = 0; j < n; ++j) {
            float acc = 0.0f;
            int32_t s = rowOff[r];
            int32_t e = rowOff[r + 1];
            for (int32_t p = s; p < e; ++p) {
                int32_t c = colInd[p];
                int64_t bIdx;
                if (!opBTranspose) {
                    bIdx = bRowMajor
                        ? static_cast<int64_t>(c) * ldb + j
                        : static_cast<int64_t>(j) * ldb + c;
                } else {
                    bIdx = bRowMajor
                        ? static_cast<int64_t>(j) * ldb + c
                        : static_cast<int64_t>(c) * ldb + j;
                }
                acc += vals[p] * B[static_cast<size_t>(bIdx)];
            }
            int64_t cIdx = cRowMajor
                ? static_cast<int64_t>(r) * ldc + j
                : static_cast<int64_t>(j) * ldc + r;
            float cOld = (*C)[static_cast<size_t>(cIdx)];
            (*C)[static_cast<size_t>(cIdx)] = alpha * acc + beta * cOld;
        }
    }
}

void SpmmCpuInt8(int32_t m, int32_t n,
                 const std::vector<int32_t> &rowOff,
                 const std::vector<int32_t> &colInd,
                 const std::vector<int8_t>  &vals,
                 const std::vector<int8_t>  &B,
                 int32_t ldb, bool bRowMajor, bool opBTranspose,
                 std::vector<int32_t>       *C,
                 int32_t ldc, bool cRowMajor,
                 int32_t alpha, int32_t beta)
{
    for (int32_t r = 0; r < m; ++r) {
        for (int32_t j = 0; j < n; ++j) {
            int32_t acc = 0;
            int32_t s = rowOff[r];
            int32_t e = rowOff[r + 1];
            for (int32_t p = s; p < e; ++p) {
                int32_t c = colInd[p];
                int64_t bIdx;
                if (!opBTranspose) {
                    bIdx = bRowMajor
                        ? static_cast<int64_t>(c) * ldb + j
                        : static_cast<int64_t>(j) * ldb + c;
                } else {
                    bIdx = bRowMajor
                        ? static_cast<int64_t>(j) * ldb + c
                        : static_cast<int64_t>(c) * ldb + j;
                }
                acc += static_cast<int32_t>(vals[p]) *
                       static_cast<int32_t>(B[static_cast<size_t>(bIdx)]);
            }
            int64_t cIdx = cRowMajor
                ? static_cast<int64_t>(r) * ldc + j
                : static_cast<int64_t>(j) * ldc + r;
            int32_t cOld = (*C)[static_cast<size_t>(cIdx)];
            (*C)[static_cast<size_t>(cIdx)] = alpha * acc + beta * cOld;
        }
    }
}

VerifyResult VerifyFloat(const std::vector<float> &got, const std::vector<float> &expect,
                 int32_t m, int32_t n, int32_t ldc, bool cRowMajor, const char *dtypeLabel,
                 double mereThreshold, double mareThreshold)
{
    const int32_t total = m * n;
    double mereSum = 0.0;
    double mare = 0.0;
    double mae = 0.0;
    int32_t shownErrors = 0;

    for (int32_t r = 0; r < m; ++r) {
        for (int32_t j = 0; j < n; ++j) {
            int64_t idx = cRowMajor
                ? static_cast<int64_t>(r) * ldc + j
                : static_cast<int64_t>(j) * ldc + r;
            float g = got[static_cast<size_t>(idx)];
            float e = expect[static_cast<size_t>(idx)];
            double absDiff = std::fabs(static_cast<double>(g) - static_cast<double>(e));
            double absExpect = std::fabs(static_cast<double>(e));
            double relErr = absDiff / (absExpect + 1e-7);
            mereSum += relErr;
            if (relErr > mare) {
                mare = relErr;
            }
            if (absDiff > mae) {
                mae = absDiff;
            }
            if (relErr > mareThreshold && shownErrors < 10) {
                std::printf("  diff at (%d, %d): expected %.8f, got %.8f, relErr=%.6e\n",
                            r, j, e, g, relErr);
                ++shownErrors;
            }
        }
    }
    double mere = mereSum / static_cast<double>(total);

    bool pass = (mere < mereThreshold) && (mare < mareThreshold);
    std::printf("[%s] MERE=%.6e (threshold=%.6e) %s\n",
                dtypeLabel, mere, mereThreshold,
                mere < mereThreshold ? "PASS" : "FAIL");
    std::printf("[%s] MARE=%.6e (threshold=%.6e) %s\n",
                dtypeLabel, mare, mareThreshold,
                mare < mareThreshold ? "PASS" : "FAIL");
    std::printf("[%s] MAE =%.6e\n", dtypeLabel, mae);
    if (pass) {
        std::printf("[%s] Overall: PASS\n", dtypeLabel);
    } else {
        std::printf("[%s] Overall: FAIL\n", dtypeLabel);
    }
    return {pass, mere, mare, mae};
}

VerifyResult VerifyInt32(const std::vector<int32_t> &got, const std::vector<int32_t> &expect,
                 int32_t m, int32_t n, int32_t ldc, bool cRowMajor)
{
    int32_t errors = 0;
    int32_t shownErrors = 0;
    for (int32_t r = 0; r < m; ++r) {
        for (int32_t j = 0; j < n; ++j) {
            int64_t idx = cRowMajor
                ? static_cast<int64_t>(r) * ldc + j
                : static_cast<int64_t>(j) * ldc + r;
            int32_t g = got[static_cast<size_t>(idx)];
            int32_t e = expect[static_cast<size_t>(idx)];
            if (g != e) {
                if (shownErrors < 10) {
                    std::printf("  diff at (%d, %d): expected %d, got %d\n", r, j, e, g);
                    ++shownErrors;
                }
                ++errors;
            }
        }
    }
    if (errors > 0) {
        std::printf("[INT8] %d / %d elements mismatch (exact match) FAIL\n", errors, m * n);
        double errRate = static_cast<double>(errors) / static_cast<double>(m * n);
        return {false, errRate, errRate, errRate};
    }
    std::printf("[INT8] all %d elements pass (exact match) PASS\n", m * n);
    return {true, 0.0, 0.0, 0.0};
}

void PrintTiming(const char *label, double msDataGen, double msCpuRef,
                 double msDevAlloc, double msH2D, double msGetBuf,
                 double msPreprocess, double msSpmm, double msD2H, double msVerify)
{
    double msTotal = msDataGen + msCpuRef + msDevAlloc + msH2D +
                     msGetBuf + msPreprocess + msSpmm + msD2H + msVerify;
    std::printf("\n--- Timing: %s ---\n", label);
    std::printf("  Data Generation   : %10.3f ms\n", msDataGen);
    std::printf("  CPU Reference     : %10.3f ms\n", msCpuRef);
    std::printf("  Device Alloc      : %10.3f ms\n", msDevAlloc);
    std::printf("  H2D Copy          : %10.3f ms\n", msH2D);
    std::printf("  GetBufferSize     : %10.3f ms\n", msGetBuf);
    std::printf("  Preprocess        : %10.3f ms\n", msPreprocess);
    std::printf("  Spmm Execute      : %10.3f ms\n", msSpmm);
    std::printf("  D2H Copy          : %10.3f ms\n", msD2H);
    std::printf("  Verify            : %10.3f ms\n", msVerify);
    std::printf("  Total             : %10.3f ms\n", msTotal);
}

inline uint16_t Fp32ToFp16Bits(float v)
{
    uint32_t bits;
    __builtin_memcpy(&bits, &v, sizeof(float));
    uint32_t sign = (bits >> 16) & 0x8000u;
    int32_t exp = static_cast<int32_t>((bits >> 23) & 0xFFu) - 127;
    uint32_t mant = bits & 0x007FFFFFu;
    if (exp >= 16) {
        return static_cast<uint16_t>(sign | 0x7C00u);
    }
    if (exp <= -15) {
        if (exp >= -25) {
            uint32_t shift = static_cast<uint32_t>(14 - exp);
            uint32_t frac = (0x00400000u | mant) >> shift;
            return static_cast<uint16_t>(sign | frac);
        }
        return static_cast<uint16_t>(sign);
    }
    uint32_t newExp = static_cast<uint32_t>(exp + 15);
    uint32_t newMant = mant >> 13;
    return static_cast<uint16_t>(sign | (newExp << 10) | newMant);
}

inline float Fp16BitsToFp32(uint16_t h)
{
    uint32_t sign = static_cast<uint32_t>(h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1Fu;
    uint32_t mant = h & 0x03FFu;
    uint32_t f;
    if (exp == 0) {
        if (mant == 0) {
            f = sign;
        } else {
            uint32_t shift = __builtin_clz(mant) - 21;
            mant <<= shift;
            exp = 1;
            f = sign | ((exp + 127 - 15 - shift) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        f = sign | 0x7F800000u | (mant << 13);
    } else {
        f = sign | ((exp + 127 - 15) << 23) | (mant << 13);
    }
    float v;
    __builtin_memcpy(&v, &f, sizeof(float));
    return v;
}

} // namespace

// ============================================================================
// FP32 test: A(fp32) * B(fp32) -> C(fp32), computeType=ACL_FLOAT
// ============================================================================
static VerifyResult RunSpmmTestFp32(int32_t deviceId, aclrtStream stream,
                            int32_t m, int32_t k, int32_t n, int32_t nnz,
                            const std::vector<int32_t> &hRowOff,
                            const std::vector<int32_t> &hColInd,
                            const std::vector<float>   &hValsFp32,
                            float alpha, float beta,
                            aclsparseOperation_t opB,
                            aclsparseOrder_t orderB,
                            aclsparseOrder_t orderC)
{
    TimePoint t0, t1;
    double msDataGen = 0, msCpuRef = 0, msDevAlloc = 0, msH2D = 0;
    double msGetBuf = 0, msPreprocess = 0, msSpmm = 0, msD2H = 0, msVerify = 0;

    const bool bRowMajor = (orderB == ACL_SPARSE_ORDER_ROW);
    const bool cRowMajor = (orderC == ACL_SPARSE_ORDER_ROW);
    const bool opBTranspose = (opB == ACL_SPARSE_OP_TRANSPOSE);
    const int32_t ldb = bRowMajor ? n : k;
    const int32_t ldc = cRowMajor ? n : m;

    const char *opBStr = opBTranspose ? "T" : "N";
    const char *obStr = bRowMajor ? "R" : "C";
    const char *ocStr = cRowMajor ? "R" : "C";
    std::printf("\n====== SPMM FP32 Test [opB=%s, B=%s, C=%s] ======\n", opBStr, obStr, ocStr);

    t0 = Clock::now();
    std::vector<float> hB(static_cast<size_t>(k) * ldb);
    for (size_t i = 0; i < hB.size(); ++i) {
        hB[i] = -5.0f + 10.0f * static_cast<float>(i % 10001) / 10000.0f;
    }
    std::vector<float> hC(static_cast<size_t>(m) * ldc, 0.0f);
    if (beta != 0.0f) {
        for (size_t i = 0; i < hC.size(); ++i) {
            hC[i] = -5.0f + 10.0f * static_cast<float>(i % 10001) / 10000.0f;
        }
    }
    std::vector<float> hCRef(hC);
    t1 = Clock::now();
    msDataGen = ElapsedMs(t0, t1);

    t0 = Clock::now();
    SpmmCpuFp32(m, n, hRowOff, hColInd, hValsFp32, hB, ldb, bRowMajor, opBTranspose, &hCRef, ldc, cRowMajor, alpha, beta);
    t1 = Clock::now();
    msCpuRef = ElapsedMs(t0, t1);

    t0 = Clock::now();
    int32_t *dRowOff = nullptr;
    int32_t *dColInd = nullptr;
    float   *dVals   = nullptr;
    float   *dB      = nullptr;
    float   *dC      = nullptr;
    aclError aclRet = aclrtMalloc((void **)&dRowOff, sizeof(int32_t) * (m + 1), ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", aclRet); return FailVerify());
    aclRet = aclrtMalloc((void **)&dColInd, sizeof(int32_t) * nnz, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", aclRet); return FailVerify());
    aclRet = aclrtMalloc((void **)&dVals, sizeof(float) * nnz, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", aclRet); return FailVerify());
    aclRet = aclrtMalloc((void **)&dB, sizeof(float) * k * ldb, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", aclRet); return FailVerify());
    aclRet = aclrtMalloc((void **)&dC, sizeof(float) * m * ldc, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", aclRet); return FailVerify());
    t1 = Clock::now();
    msDevAlloc = ElapsedMs(t0, t1);

    t0 = Clock::now();
    aclRet = aclrtMemcpy(dRowOff, sizeof(int32_t) * (m + 1), hRowOff.data(), sizeof(int32_t) * (m + 1), ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", aclRet); return FailVerify());
    aclRet = aclrtMemcpy(dColInd, sizeof(int32_t) * nnz, hColInd.data(), sizeof(int32_t) * nnz, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", aclRet); return FailVerify());
    aclRet = aclrtMemcpy(dVals, sizeof(float) * nnz, hValsFp32.data(), sizeof(float) * nnz, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", aclRet); return FailVerify());
    aclRet = aclrtMemcpy(dB, sizeof(float) * k * ldb, hB.data(), sizeof(float) * k * ldb, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", aclRet); return FailVerify());
    aclRet = aclrtMemcpy(dC, sizeof(float) * m * ldc, hC.data(), sizeof(float) * m * ldc, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", aclRet); return FailVerify());
    t1 = Clock::now();
    msH2D = ElapsedMs(t0, t1);

    aclsparseHandle_t handle = nullptr;
    aclsparseStatus_t sparseRet = aclsparseCreate(&handle);
    CHECK_RET(sparseRet == ACL_SPARSE_STATUS_SUCCESS,
              LOG_PRINT("aclsparseCreate failed. ERROR: %d\n", sparseRet);
              return FailVerify());
    sparseRet = aclsparseSetStream(handle, stream);
    CHECK_RET(sparseRet == ACL_SPARSE_STATUS_SUCCESS,
              LOG_PRINT("aclsparseSetStream failed. ERROR: %d\n", sparseRet);
              return FailVerify());

    aclsparseSpMatDescr_t matA = nullptr;
    sparseRet = aclsparseCreateCsr(&matA, m, k, nnz, dRowOff, dColInd, dVals,
                                   ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_32I,
                                   ACL_SPARSE_INDEX_BASE_ZERO, ACL_FLOAT);
    CHECK_RET(sparseRet == ACL_SPARSE_STATUS_SUCCESS,
              LOG_PRINT("aclsparseCreateCsr failed. ERROR: %d\n", sparseRet);
              return FailVerify());

    aclsparseDnMatDescr_t matB = nullptr;
    aclsparseDnMatDescr_t matC = nullptr;
    sparseRet = aclsparseCreateDnMat(&matB, k, n, ldb, dB, ACL_FLOAT, orderB);
    CHECK_RET(sparseRet == ACL_SPARSE_STATUS_SUCCESS, return FailVerify());
    sparseRet = aclsparseCreateDnMat(&matC, m, n, ldc, dC, ACL_FLOAT, orderC);
    CHECK_RET(sparseRet == ACL_SPARSE_STATUS_SUCCESS, return FailVerify());

    t0 = Clock::now();
    size_t bufferSize = 0;
    sparseRet = aclsparseSpMMGetBufferSize(handle,
        ACL_SPARSE_OP_NON_TRANSPOSE, opB,
        &alpha, matA, matB, &beta, matC,
        ACL_FLOAT, ACL_SPARSE_SPMM_CSR_ALG1, &bufferSize);
    CHECK_RET(sparseRet == ACL_SPARSE_STATUS_SUCCESS, return FailVerify());
    t1 = Clock::now();
    msGetBuf = ElapsedMs(t0, t1);
    std::printf("Workspace bytes: %zu\n", bufferSize);

    void *dBuffer = nullptr;
    aclRet = aclrtMalloc(&dBuffer, bufferSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", aclRet); return FailVerify());

    t0 = Clock::now();
    sparseRet = aclsparseSpMMPreprocess(handle,
        ACL_SPARSE_OP_NON_TRANSPOSE, opB,
        &alpha, matA, matB, &beta, matC,
        ACL_FLOAT, ACL_SPARSE_SPMM_CSR_ALG1, dBuffer);
    CHECK_RET(sparseRet == ACL_SPARSE_STATUS_SUCCESS, return FailVerify());
    t1 = Clock::now();
    msPreprocess = ElapsedMs(t0, t1);

    t0 = Clock::now();
    sparseRet = aclsparseSpMM(handle,
        ACL_SPARSE_OP_NON_TRANSPOSE, opB,
        &alpha, matA, matB, &beta, matC,
        ACL_FLOAT, ACL_SPARSE_SPMM_CSR_ALG1, dBuffer);
    CHECK_RET(sparseRet == ACL_SPARSE_STATUS_SUCCESS, return FailVerify());
    aclRet = aclrtSynchronizeStream(stream);
    CHECK_RET(aclRet == ACL_SUCCESS,
              LOG_PRINT("[ERROR] aclsparseSpMM: aclrtSynchronizeStream failed, ret=%d\n", aclRet);
              return FailVerify());
    t1 = Clock::now();
    msSpmm = ElapsedMs(t0, t1);

    t0 = Clock::now();
    aclRet = aclrtMemcpy(hC.data(), sizeof(float) * m * ldc, dC, sizeof(float) * m * ldc, ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", aclRet); return FailVerify());
    t1 = Clock::now();
    msD2H = ElapsedMs(t0, t1);

    t0 = Clock::now();
    VerifyResult vr = VerifyFloat(hC, hCRef, m, n, ldc, cRowMajor, "FP32",
                          1.0 / (1 << 13), 10.0 / (1 << 13));
    t1 = Clock::now();
    msVerify = ElapsedMs(t0, t1);

    PrintTiming("FP32", msDataGen, msCpuRef, msDevAlloc, msH2D, msGetBuf, msPreprocess, msSpmm, msD2H, msVerify);

    aclsparseDestroyDnMat(matB);
    aclsparseDestroyDnMat(matC);
    aclsparseDestroySpMat(matA);
    aclsparseDestroy(handle);
    aclrtFree(dBuffer);
    aclrtFree(dRowOff);
    aclrtFree(dColInd);
    aclrtFree(dVals);
    aclrtFree(dB);
    aclrtFree(dC);

    return vr;
}

// ============================================================================
// FP16 test: A(fp16) * B(fp16) -> C(fp16), computeType=ACL_FLOAT
//   Host uses uint16_t to store IEEE 754 half-precision bit patterns.
//   CPU reference computed in fp32 for accuracy; fp16 tolerance is looser.
// ============================================================================
static VerifyResult RunSpmmTestFp16(int32_t deviceId, aclrtStream stream,
                            int32_t m, int32_t k, int32_t n, int32_t nnz,
                            const std::vector<int32_t> &hRowOff,
                            const std::vector<int32_t> &hColInd,
                            const std::vector<float>   &hValsFp32,
                            float alpha, float beta,
                            aclsparseOperation_t opB,
                            aclsparseOrder_t orderB,
                            aclsparseOrder_t orderC)
{
    TimePoint t0, t1;
    double msDataGen = 0, msCpuRef = 0, msDevAlloc = 0, msH2D = 0;
    double msGetBuf = 0, msPreprocess = 0, msSpmm = 0, msD2H = 0, msVerify = 0;

    const bool bRowMajor = (orderB == ACL_SPARSE_ORDER_ROW);
    const bool cRowMajor = (orderC == ACL_SPARSE_ORDER_ROW);
    const bool opBTranspose = (opB == ACL_SPARSE_OP_TRANSPOSE);
    const int32_t ldb = bRowMajor ? n : k;
    const int32_t ldc = cRowMajor ? n : m;

    const char *opBStr = opBTranspose ? "T" : "N";
    const char *obStr = bRowMajor ? "R" : "C";
    const char *ocStr = cRowMajor ? "R" : "C";
    std::printf("\n====== SPMM FP16 Test [opB=%s, B=%s, C=%s] ======\n", opBStr, obStr, ocStr);

    t0 = Clock::now();
    std::vector<float> hBFp32(static_cast<size_t>(k) * ldb);
    for (size_t i = 0; i < hBFp32.size(); ++i) {
        hBFp32[i] = -5.0f + 10.0f * static_cast<float>(i % 10001) / 10000.0f;
    }
    std::vector<float> hCRefFp32(static_cast<size_t>(m) * ldc, 0.0f);
    if (beta != 0.0f) {
        for (size_t i = 0; i < hCRefFp32.size(); ++i) {
            hCRefFp32[i] = -5.0f + 10.0f * static_cast<float>(i % 10001) / 10000.0f;
        }
    }

    std::vector<uint16_t> hValsFp16(static_cast<size_t>(nnz));
    for (int32_t i = 0; i < nnz; ++i) {
        hValsFp16[i] = Fp32ToFp16Bits(hValsFp32[i]);
    }
    std::vector<uint16_t> hBFp16(static_cast<size_t>(k) * ldb);
    for (size_t i = 0; i < hBFp16.size(); ++i) {
        hBFp16[i] = Fp32ToFp16Bits(hBFp32[i]);
    }
    std::vector<uint16_t> hCFp16(static_cast<size_t>(m) * ldc);
    for (size_t i = 0; i < hCFp16.size(); ++i) {
        hCFp16[i] = Fp32ToFp16Bits(hCRefFp32[i]);
    }
    t1 = Clock::now();
    msDataGen = ElapsedMs(t0, t1);

    t0 = Clock::now();
    std::vector<float> hValsFp16RoundTrip(static_cast<size_t>(nnz));
    for (int32_t i = 0; i < nnz; ++i) {
        hValsFp16RoundTrip[i] = Fp16BitsToFp32(hValsFp16[i]);
    }
    std::vector<float> hBFp16RoundTrip(static_cast<size_t>(k) * ldb);
    for (size_t i = 0; i < hBFp16RoundTrip.size(); ++i) {
        hBFp16RoundTrip[i] = Fp16BitsToFp32(hBFp16[i]);
    }
    std::vector<float> hCRefFp16RoundTrip(static_cast<size_t>(m) * ldc);
    for (size_t i = 0; i < hCRefFp16RoundTrip.size(); ++i) {
        hCRefFp16RoundTrip[i] = Fp16BitsToFp32(hCFp16[i]);
    }
    SpmmCpuFp32(m, n, hRowOff, hColInd, hValsFp16RoundTrip, hBFp16RoundTrip, ldb, bRowMajor, opBTranspose, &hCRefFp16RoundTrip, ldc, cRowMajor, alpha, beta);
    constexpr float kFp16Max = 65504.0f;
    for (size_t i = 0; i < hCRefFp16RoundTrip.size(); ++i) {
        if (hCRefFp16RoundTrip[i] > kFp16Max) hCRefFp16RoundTrip[i] = kFp16Max;
        if (hCRefFp16RoundTrip[i] < -kFp16Max) hCRefFp16RoundTrip[i] = -kFp16Max;
        hCRefFp16RoundTrip[i] = Fp16BitsToFp32(Fp32ToFp16Bits(hCRefFp16RoundTrip[i]));
    }
    t1 = Clock::now();
    msCpuRef = ElapsedMs(t0, t1);

    t0 = Clock::now();
    int32_t  *dRowOff = nullptr;
    int32_t  *dColInd = nullptr;
    void     *dVals   = nullptr;
    void     *dB      = nullptr;
    void     *dC      = nullptr;
    aclError aclRet = aclrtMalloc((void **)&dRowOff, sizeof(int32_t) * (m + 1), ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", aclRet); return FailVerify());
    aclRet = aclrtMalloc((void **)&dColInd, sizeof(int32_t) * nnz, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", aclRet); return FailVerify());
    aclRet = aclrtMalloc(&dVals, sizeof(uint16_t) * nnz, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", aclRet); return FailVerify());
    aclRet = aclrtMalloc(&dB, sizeof(uint16_t) * k * ldb, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", aclRet); return FailVerify());
    aclRet = aclrtMalloc(&dC, sizeof(uint16_t) * m * ldc, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", aclRet); return FailVerify());
    t1 = Clock::now();
    msDevAlloc = ElapsedMs(t0, t1);

    t0 = Clock::now();
    aclRet = aclrtMemcpy(dRowOff, sizeof(int32_t) * (m + 1), hRowOff.data(), sizeof(int32_t) * (m + 1), ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", aclRet); return FailVerify());
    aclRet = aclrtMemcpy(dColInd, sizeof(int32_t) * nnz, hColInd.data(), sizeof(int32_t) * nnz, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", aclRet); return FailVerify());
    aclRet = aclrtMemcpy(dVals, sizeof(uint16_t) * nnz, hValsFp16.data(), sizeof(uint16_t) * nnz, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", aclRet); return FailVerify());
    aclRet = aclrtMemcpy(dB, sizeof(uint16_t) * k * ldb, hBFp16.data(), sizeof(uint16_t) * k * ldb, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", aclRet); return FailVerify());
    aclRet = aclrtMemcpy(dC, sizeof(uint16_t) * m * ldc, hCFp16.data(), sizeof(uint16_t) * m * ldc, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", aclRet); return FailVerify());
    t1 = Clock::now();
    msH2D = ElapsedMs(t0, t1);

    aclsparseHandle_t handle = nullptr;
    aclsparseStatus_t sparseRet = aclsparseCreate(&handle);
    CHECK_RET(sparseRet == ACL_SPARSE_STATUS_SUCCESS,
              LOG_PRINT("aclsparseCreate failed. ERROR: %d\n", sparseRet);
              return FailVerify());
    sparseRet = aclsparseSetStream(handle, stream);
    CHECK_RET(sparseRet == ACL_SPARSE_STATUS_SUCCESS,
              LOG_PRINT("aclsparseSetStream failed. ERROR: %d\n", sparseRet);
              return FailVerify());

    aclsparseSpMatDescr_t matA = nullptr;
    sparseRet = aclsparseCreateCsr(&matA, m, k, nnz, dRowOff, dColInd, dVals,
                                   ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_32I,
                                   ACL_SPARSE_INDEX_BASE_ZERO, ACL_FLOAT16);
    CHECK_RET(sparseRet == ACL_SPARSE_STATUS_SUCCESS,
              LOG_PRINT("aclsparseCreateCsr failed. ERROR: %d\n", sparseRet);
              return FailVerify());

    aclsparseDnMatDescr_t matB = nullptr;
    aclsparseDnMatDescr_t matC = nullptr;
    sparseRet = aclsparseCreateDnMat(&matB, k, n, ldb, dB, ACL_FLOAT16, orderB);
    CHECK_RET(sparseRet == ACL_SPARSE_STATUS_SUCCESS, return FailVerify());
    sparseRet = aclsparseCreateDnMat(&matC, m, n, ldc, dC, ACL_FLOAT16, orderC);
    CHECK_RET(sparseRet == ACL_SPARSE_STATUS_SUCCESS, return FailVerify());

    t0 = Clock::now();
    size_t bufferSize = 0;
    sparseRet = aclsparseSpMMGetBufferSize(handle,
        ACL_SPARSE_OP_NON_TRANSPOSE, opB,
        &alpha, matA, matB, &beta, matC,
        ACL_FLOAT, ACL_SPARSE_SPMM_CSR_ALG1, &bufferSize);
    CHECK_RET(sparseRet == ACL_SPARSE_STATUS_SUCCESS, return FailVerify());
    t1 = Clock::now();
    msGetBuf = ElapsedMs(t0, t1);
    std::printf("Workspace bytes: %zu\n", bufferSize);

    void *dBuffer = nullptr;
    aclRet = aclrtMalloc(&dBuffer, bufferSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", aclRet); return FailVerify());

    t0 = Clock::now();
    sparseRet = aclsparseSpMMPreprocess(handle,
        ACL_SPARSE_OP_NON_TRANSPOSE, opB,
        &alpha, matA, matB, &beta, matC,
        ACL_FLOAT, ACL_SPARSE_SPMM_CSR_ALG1, dBuffer);
    CHECK_RET(sparseRet == ACL_SPARSE_STATUS_SUCCESS, return FailVerify());
    t1 = Clock::now();
    msPreprocess = ElapsedMs(t0, t1);

    t0 = Clock::now();
    sparseRet = aclsparseSpMM(handle,
        ACL_SPARSE_OP_NON_TRANSPOSE, opB,
        &alpha, matA, matB, &beta, matC,
        ACL_FLOAT, ACL_SPARSE_SPMM_CSR_ALG1, dBuffer);
    CHECK_RET(sparseRet == ACL_SPARSE_STATUS_SUCCESS, return FailVerify());
    aclRet = aclrtSynchronizeStream(stream);
    CHECK_RET(aclRet == ACL_SUCCESS,
              LOG_PRINT("[ERROR] aclsparseSpMM: aclrtSynchronizeStream failed, ret=%d\n", aclRet);
              return FailVerify());
    t1 = Clock::now();
    msSpmm = ElapsedMs(t0, t1);

    t0 = Clock::now();
    aclRet = aclrtMemcpy(hCFp16.data(), sizeof(uint16_t) * m * ldc, dC, sizeof(uint16_t) * m * ldc, ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", aclRet); return FailVerify());
    t1 = Clock::now();
    msD2H = ElapsedMs(t0, t1);

    t0 = Clock::now();
    std::vector<float> hCFp32FromFp16(static_cast<size_t>(m) * ldc);
    for (size_t i = 0; i < hCFp32FromFp16.size(); ++i) {
        hCFp32FromFp16[i] = Fp16BitsToFp32(hCFp16[i]);
    }
    VerifyResult vr = VerifyFloat(hCFp32FromFp16, hCRefFp16RoundTrip, m, n, ldc, cRowMajor, "FP16",
                          1.0 / (1 << 10), 10.0 / (1 << 10));
    t1 = Clock::now();
    msVerify = ElapsedMs(t0, t1);

    PrintTiming("FP16", msDataGen, msCpuRef, msDevAlloc, msH2D, msGetBuf, msPreprocess, msSpmm, msD2H, msVerify);

    aclsparseDestroyDnMat(matB);
    aclsparseDestroyDnMat(matC);
    aclsparseDestroySpMat(matA);
    aclsparseDestroy(handle);
    aclrtFree(dBuffer);
    aclrtFree(dRowOff);
    aclrtFree(dColInd);
    aclrtFree(dVals);
    aclrtFree(dB);
    aclrtFree(dC);

    return vr;
}

// ============================================================================
// INT8 test: A(int8) * B(int8) -> C(int32), computeType=ACL_INT32
//   CPU reference computed in int32; exact match required.
// ============================================================================
static VerifyResult RunSpmmTestInt8(int32_t deviceId, aclrtStream stream,
                            int32_t m, int32_t k, int32_t n, int32_t nnz,
                            const std::vector<int32_t> &hRowOff,
                            const std::vector<int32_t> &hColInd,
                            int32_t alpha, int32_t beta,
                            aclsparseOperation_t opB,
                            aclsparseOrder_t orderB,
                            aclsparseOrder_t orderC)
{
    TimePoint t0, t1;
    double msDataGen = 0, msCpuRef = 0, msDevAlloc = 0, msH2D = 0;
    double msGetBuf = 0, msPreprocess = 0, msSpmm = 0, msD2H = 0, msVerify = 0;

    const bool bRowMajor = (orderB == ACL_SPARSE_ORDER_ROW);
    const bool cRowMajor = (orderC == ACL_SPARSE_ORDER_ROW);
    const bool opBTranspose = (opB == ACL_SPARSE_OP_TRANSPOSE);
    const int32_t ldb = bRowMajor ? n : k;
    const int32_t ldc = cRowMajor ? n : m;

    const char *opBStr = opBTranspose ? "T" : "N";
    const char *obStr = bRowMajor ? "R" : "C";
    const char *ocStr = cRowMajor ? "R" : "C";
    std::printf("\n====== SPMM INT8 Test [opB=%s, B=%s, C=%s] ======\n", opBStr, obStr, ocStr);

    t0 = Clock::now();
    std::vector<int8_t> hValsInt8(static_cast<size_t>(nnz));
    for (int32_t i = 0; i < nnz; ++i) {
        hValsInt8[i] = static_cast<int8_t>((i % 11) - 5);
    }
    std::vector<int8_t> hBInt8(static_cast<size_t>(k) * ldb);
    for (size_t i = 0; i < hBInt8.size(); ++i) {
        hBInt8[i] = static_cast<int8_t>((i % 11) - 5);
    }
    std::vector<int32_t> hCInt32(static_cast<size_t>(m) * ldc, 0);
    if (beta != 0) {
        for (size_t i = 0; i < hCInt32.size(); ++i) {
            hCInt32[i] = static_cast<int32_t>((i % 11) - 5);
        }
    }
    std::vector<int32_t> hCRefInt32(hCInt32);
    t1 = Clock::now();
    msDataGen = ElapsedMs(t0, t1);

    t0 = Clock::now();
    SpmmCpuInt8(m, n, hRowOff, hColInd, hValsInt8, hBInt8, ldb, bRowMajor, opBTranspose, &hCRefInt32, ldc, cRowMajor, alpha, beta);
    t1 = Clock::now();
    msCpuRef = ElapsedMs(t0, t1);

    t0 = Clock::now();
    int32_t  *dRowOff = nullptr;
    int32_t  *dColInd = nullptr;
    int8_t   *dVals   = nullptr;
    int8_t   *dB      = nullptr;
    int32_t  *dC      = nullptr;
    aclError aclRet = aclrtMalloc((void **)&dRowOff, sizeof(int32_t) * (m + 1), ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", aclRet); return FailVerify());
    aclRet = aclrtMalloc((void **)&dColInd, sizeof(int32_t) * nnz, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", aclRet); return FailVerify());
    aclRet = aclrtMalloc((void **)&dVals, sizeof(int8_t) * nnz, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", aclRet); return FailVerify());
    aclRet = aclrtMalloc((void **)&dB, sizeof(int8_t) * k * ldb, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", aclRet); return FailVerify());
    aclRet = aclrtMalloc((void **)&dC, sizeof(int32_t) * m * ldc, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", aclRet); return FailVerify());
    t1 = Clock::now();
    msDevAlloc = ElapsedMs(t0, t1);

    t0 = Clock::now();
    aclRet = aclrtMemcpy(dRowOff, sizeof(int32_t) * (m + 1), hRowOff.data(), sizeof(int32_t) * (m + 1), ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", aclRet); return FailVerify());
    aclRet = aclrtMemcpy(dColInd, sizeof(int32_t) * nnz, hColInd.data(), sizeof(int32_t) * nnz, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", aclRet); return FailVerify());
    aclRet = aclrtMemcpy(dVals, sizeof(int8_t) * nnz, hValsInt8.data(), sizeof(int8_t) * nnz, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", aclRet); return FailVerify());
    aclRet = aclrtMemcpy(dB, sizeof(int8_t) * k * ldb, hBInt8.data(), sizeof(int8_t) * k * ldb, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", aclRet); return FailVerify());
    aclRet = aclrtMemcpy(dC, sizeof(int32_t) * m * ldc, hCInt32.data(), sizeof(int32_t) * m * ldc, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", aclRet); return FailVerify());
    t1 = Clock::now();
    msH2D = ElapsedMs(t0, t1);

    aclsparseHandle_t handle = nullptr;
    aclsparseStatus_t sparseRet = aclsparseCreate(&handle);
    CHECK_RET(sparseRet == ACL_SPARSE_STATUS_SUCCESS,
              LOG_PRINT("aclsparseCreate failed. ERROR: %d\n", sparseRet);
              return FailVerify());
    sparseRet = aclsparseSetStream(handle, stream);
    CHECK_RET(sparseRet == ACL_SPARSE_STATUS_SUCCESS,
              LOG_PRINT("aclsparseSetStream failed. ERROR: %d\n", sparseRet);
              return FailVerify());

    aclsparseSpMatDescr_t matA = nullptr;
    sparseRet = aclsparseCreateCsr(&matA, m, k, nnz, dRowOff, dColInd, dVals,
                                   ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_32I,
                                   ACL_SPARSE_INDEX_BASE_ZERO, ACL_INT8);
    CHECK_RET(sparseRet == ACL_SPARSE_STATUS_SUCCESS,
              LOG_PRINT("aclsparseCreateCsr failed. ERROR: %d\n", sparseRet);
              return FailVerify());

    aclsparseDnMatDescr_t matB = nullptr;
    aclsparseDnMatDescr_t matC = nullptr;
    sparseRet = aclsparseCreateDnMat(&matB, k, n, ldb, dB, ACL_INT8, orderB);
    CHECK_RET(sparseRet == ACL_SPARSE_STATUS_SUCCESS, return FailVerify());
    sparseRet = aclsparseCreateDnMat(&matC, m, n, ldc, dC, ACL_INT32, orderC);
    CHECK_RET(sparseRet == ACL_SPARSE_STATUS_SUCCESS, return FailVerify());

    t0 = Clock::now();
    size_t bufferSize = 0;
    sparseRet = aclsparseSpMMGetBufferSize(handle,
        ACL_SPARSE_OP_NON_TRANSPOSE, opB,
        &alpha, matA, matB, &beta, matC,
        ACL_INT32, ACL_SPARSE_SPMM_CSR_ALG1, &bufferSize);
    CHECK_RET(sparseRet == ACL_SPARSE_STATUS_SUCCESS, return FailVerify());
    t1 = Clock::now();
    msGetBuf = ElapsedMs(t0, t1);
    std::printf("Workspace bytes: %zu\n", bufferSize);

    void *dBuffer = nullptr;
    aclRet = aclrtMalloc(&dBuffer, bufferSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", aclRet); return FailVerify());

    t0 = Clock::now();
    sparseRet = aclsparseSpMMPreprocess(handle,
        ACL_SPARSE_OP_NON_TRANSPOSE, opB,
        &alpha, matA, matB, &beta, matC,
        ACL_INT32, ACL_SPARSE_SPMM_CSR_ALG1, dBuffer);
    CHECK_RET(sparseRet == ACL_SPARSE_STATUS_SUCCESS, return FailVerify());
    t1 = Clock::now();
    msPreprocess = ElapsedMs(t0, t1);

    t0 = Clock::now();
    sparseRet = aclsparseSpMM(handle,
        ACL_SPARSE_OP_NON_TRANSPOSE, opB,
        &alpha, matA, matB, &beta, matC,
        ACL_INT32, ACL_SPARSE_SPMM_CSR_ALG1, dBuffer);
    CHECK_RET(sparseRet == ACL_SPARSE_STATUS_SUCCESS, return FailVerify());
    aclRet = aclrtSynchronizeStream(stream);
    CHECK_RET(aclRet == ACL_SUCCESS,
              LOG_PRINT("[ERROR] aclsparseSpMM: aclrtSynchronizeStream failed, ret=%d\n", aclRet);
              return FailVerify());
    t1 = Clock::now();
    msSpmm = ElapsedMs(t0, t1);

    t0 = Clock::now();
    aclRet = aclrtMemcpy(hCInt32.data(), sizeof(int32_t) * m * ldc, dC, sizeof(int32_t) * m * ldc, ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", aclRet); return FailVerify());
    t1 = Clock::now();
    msD2H = ElapsedMs(t0, t1);

    t0 = Clock::now();
    VerifyResult vr = VerifyInt32(hCInt32, hCRefInt32, m, n, ldc, cRowMajor);
    t1 = Clock::now();
    msVerify = ElapsedMs(t0, t1);

    PrintTiming("INT8", msDataGen, msCpuRef, msDevAlloc, msH2D, msGetBuf, msPreprocess, msSpmm, msD2H, msVerify);

    aclsparseDestroyDnMat(matB);
    aclsparseDestroyDnMat(matC);
    aclsparseDestroySpMat(matA);
    aclsparseDestroy(handle);
    aclrtFree(dBuffer);
    aclrtFree(dRowOff);
    aclrtFree(dColInd);
    aclrtFree(dVals);
    aclrtFree(dB);
    aclrtFree(dC);

    return vr;
}

// ============================================================================
// main: one functional case per dtype (FP32 / FP16 / INT8)
// ============================================================================
int main()
{
    std::srand(42);

    int32_t deviceId = 0;
    aclrtStream stream = nullptr;
    int ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, return EXIT_FAILURE);

    constexpr int32_t kM   = 256;
    constexpr int32_t kK   = 256;
    constexpr int32_t kN   = 64;
    constexpr int32_t kNnz = 256 * 256 / 10;

    // Default layout: opB = N, B / C row-major, alpha = 1, beta = 0.
    const aclsparseOperation_t opB = ACL_SPARSE_OP_NON_TRANSPOSE;
    const aclsparseOrder_t orderB = ACL_SPARSE_ORDER_ROW;
    const aclsparseOrder_t orderC = ACL_SPARSE_ORDER_ROW;

    std::printf("\n========== SPMM Test (one case per dtype) ==========\n");
    std::vector<int32_t> hRowOff, hColInd;
    std::vector<float>   hValsFp32;
    GenerateRandomCsr(kM, kK, kNnz, &hRowOff, &hColInd, &hValsFp32);
    const int32_t actualNnz = hRowOff.back();
    std::printf("Generated CSR: m=%d, k=%d, nnz=%d, density=%.4f\n",
                kM, kK, actualNnz,
                static_cast<float>(actualNnz) / (static_cast<float>(kM) * kK));

    VerifyResult rFp32 = RunSpmmTestFp32(deviceId, stream, kM, kK, kN, actualNnz,
        hRowOff, hColInd, hValsFp32, 1.0f, 0.0f, opB, orderB, orderC);
    VerifyResult rFp16 = RunSpmmTestFp16(deviceId, stream, kM, kK, kN, actualNnz,
        hRowOff, hColInd, hValsFp32, 1.0f, 0.0f, opB, orderB, orderC);
    VerifyResult rInt8 = RunSpmmTestInt8(deviceId, stream, kM, kK, kN, actualNnz,
        hRowOff, hColInd, 1, 0, opB, orderB, orderC);

    std::printf("\n========== Results ==========\n");
    std::printf("  FP32: %s  FP16: %s  INT8: %s\n",
                rFp32.pass ? "PASS" : "FAIL",
                rFp16.pass ? "PASS" : "FAIL",
                rInt8.pass ? "PASS" : "FAIL");

    bool allPass = rFp32.pass && rFp16.pass && rInt8.pass;
    std::printf("  Overall: %s\n", allPass ? "PASS" : "FAIL");

    Finalize(deviceId, stream);
    return allPass ? EXIT_SUCCESS : EXIT_FAILURE;
}
