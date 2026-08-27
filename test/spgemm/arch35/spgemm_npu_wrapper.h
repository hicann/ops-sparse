/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * SpGEMM NPU wrapper (uses RAII managers from test/frame/descriptor_manager.h).
 * Calls the 7-interface API: CreateDescr → WorkEstimation → Compute →
 * GetNumProducts → Copy → DestroyDescr.
 */

#ifndef TEST_SPGEMM_NPU_WRAPPER_H_
#define TEST_SPGEMM_NPU_WRAPPER_H_

#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "acl/acl.h"
#include "cann_ops_sparse.h"
#include "descriptor_manager.h"
#include "fill.h"
#include "sparse_test.h"

namespace sparse_test {

// RAII guard for SpGEMM descriptor
struct SpGEMMDescrGuard {
    aclsparseSpGEMMDescr_t d{nullptr};
    explicit SpGEMMDescrGuard(aclsparseSpGEMMDescr_t p) : d(p) {}
    ~SpGEMMDescrGuard() { if (d) aclsparseSpGEMMDestroyDescr(d); }
    SpGEMMDescrGuard(const SpGEMMDescrGuard&) = delete;
    SpGEMMDescrGuard& operator=(const SpGEMMDescrGuard&) = delete;
};

// Host-side IEEE 754 conversion utilities (fp32 ↔ fp16/bf16)
inline uint16_t HostFloatToHalf(float f) {
    uint32_t bits;
    __builtin_memcpy(&bits, &f, sizeof(float));
    uint32_t sign = (bits >> 31) & 1u;
    int32_t  exp  = static_cast<int32_t>((bits >> 23) & 0xFFu);
    uint32_t mant = bits & 0x7FFFFFu;
    if (exp == 0xFF) {
        return static_cast<uint16_t>((sign << 15) | 0x7C00u | (mant ? 1u : 0u));
    }
    int32_t newExp = exp - 127 + 15;
    if (newExp >= 0x1F) return static_cast<uint16_t>((sign << 15) | 0x7C00u);
    if (newExp <= 0) {
        if (newExp < -10) return static_cast<uint16_t>(sign << 15);
        uint32_t m = mant | 0x800000u;
        return static_cast<uint16_t>((sign << 15) | (m >> (14 - newExp)));
    }
    uint32_t roundingBias = 0x1000u + ((mant >> 13) & 1u);
    uint32_t rounded = (mant + roundingBias) >> 13;
    if (rounded > 0x3FFu) { rounded = 0u; newExp += 1; }
    if (newExp >= 0x1F) return static_cast<uint16_t>((sign << 15) | 0x7C00u);
    return static_cast<uint16_t>((sign << 15) | (static_cast<uint32_t>(newExp) << 10) | rounded);
}

inline float HostHalfToFloat(uint16_t h) {
    uint32_t sign = (static_cast<uint32_t>(h) >> 15) & 1u;
    uint32_t exp  = (static_cast<uint32_t>(h) >> 10) & 0x1Fu;
    uint32_t mant = static_cast<uint32_t>(h) & 0x3FFu;
    uint32_t f;
    if (exp == 0u) {
        if (mant == 0u) { f = sign << 31; }
        else {
            int32_t e = -1; uint32_t m = mant;
            do { ++e; m <<= 1; } while ((m & 0x400u) == 0u);
            f = (sign << 31) | ((static_cast<uint32_t>(127 - 15 - e)) << 23) | ((m & 0x3FFu) << 13);
        }
    } else if (exp == 0x1Fu) {
        f = (sign << 31) | (0xFFu << 23) | (mant << 13);
    } else {
        f = (sign << 31) | ((exp + 127u - 15u) << 23) | (mant << 13);
    }
    float result; __builtin_memcpy(&result, &f, sizeof(float));
    return result;
}

inline uint16_t HostFloatToBf16(float f) {
    uint32_t bits;
    __builtin_memcpy(&bits, &f, sizeof(float));
    uint32_t lsb = (bits >> 16) & 1u;
    uint32_t bias = 0x7FFFu + lsb;
    return static_cast<uint16_t>((bits + bias) >> 16);
}

inline float HostBf16ToFloat(uint16_t b) {
    uint32_t f = static_cast<uint32_t>(b) << 16;
    float result; __builtin_memcpy(&result, &f, sizeof(float));
    return result;
}

// Map algorithm string to enum
inline aclsparseSpGEMMAlg_t ParseSpGEMMAlg(const std::string& s) {
    if (s == "ALG2")  return ACL_SPARSE_SPGEMM_ALG2;
    if (s == "ALG3")  return ACL_SPARSE_SPGEMM_ALG3;
    return ACL_SPARSE_SPGEMM_ALG_DEFAULT;
}

// Convert float values to target dtype bytes on host
inline std::vector<uint8_t> ConvertValues(
    const std::vector<float>& vals, aclDataType dtype)
{
    if (dtype == ACL_FLOAT) {
        const auto* src = reinterpret_cast<const uint8_t*>(vals.data());
        return std::vector<uint8_t>(src, src + vals.size() * sizeof(float));
    }
    if (dtype == ACL_FLOAT16) {
        std::vector<uint8_t> out(vals.size() * sizeof(uint16_t));
        auto* dst = reinterpret_cast<uint16_t*>(out.data());
        for (size_t i = 0; i < vals.size(); i++) {
            dst[i] = HostFloatToHalf(vals[i]);
        }
        return out;
    }
    // ACL_BF16: same storage as fp16, different mantissa mapping
    if (dtype == ACL_BF16) {
        std::vector<uint8_t> out(vals.size() * sizeof(uint16_t));
        auto* dst = reinterpret_cast<uint16_t*>(out.data());
        for (size_t i = 0; i < vals.size(); i++) {
            dst[i] = HostFloatToBf16(vals[i]);
        }
        return out;
    }
    throw std::runtime_error("Unsupported dtype for ConvertValues");
}

// Convert uint16 bytes back to float on host
inline std::vector<float> ConvertToFloat(
    const uint8_t* bytes, size_t count, aclDataType dtype)
{
    std::vector<float> out(count);
    if (dtype == ACL_FLOAT) {
        const auto* src = reinterpret_cast<const float*>(bytes);
        out.assign(src, src + count);
    } else if (dtype == ACL_FLOAT16 || dtype == ACL_BF16) {
        const auto* src = reinterpret_cast<const uint16_t*>(bytes);
        for (size_t i = 0; i < count; i++) {
            out[i] = (dtype == ACL_FLOAT16)
                ? HostHalfToFloat(src[i])
                : HostBf16ToFloat(src[i]);
        }
    }
    return out;
}

// SpGEMM NPU wrapper: C = alpha * A * B + beta * C_in
//
// Parameters:
//   spHandle    - aclsparse handle (RAII)
//   stream      - ACL stream
//   csrA, csrB  - input CSR matrices A (m×k) and B (k×n), host FP32 values
//   alpha, beta - scalars
//   dtype       - compute/output dtype (ACL_FLOAT / ACL_FLOAT16 / ACL_BF16)
//   algEnum     - algorithm (DEFAULT/ALG2/ALG3)
//   caseId      - for error messages
//
// Returns: C as CsrMatrix (host, FP32 values).
// Throws std::runtime_error on any ACL API failure.

// Helper: throw on ACL status failure
inline void SpgemmCheckStatus(aclsparseStatus_t st, const std::string& caseId,
                               const std::string& step) {
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        throw std::runtime_error("[" + caseId + "] " + step + " failed, status=" +
                                  std::to_string(static_cast<int>(st)));
    }
}

// Aggregated device buffers + descriptors for SpGEMM 7-interface flow
struct SpGemmDeviceCtx {
    float alphaVal = 0.0f;
    float betaVal  = 0.0f;
    int64_t m = 0, n = 0;
    DeviceBuffer dARowOff, dAColInd, dAVals;
    DeviceBuffer dBRowOff, dBColInd, dBVals;
    DeviceBuffer dCRowOff, dCColInd, dCVals;
    DeviceBuffer buffer1, buffer2;
    SpMatManager matA, matB, matC;
};

// Step 1: prepare device buffers + matrix descriptors for A, B, C
inline SpGemmDeviceCtx PrepareSpgemmInputs(
    const CsrMatrix& csrA, const CsrMatrix& csrB,
    float alpha, float beta, aclDataType dtype)
{
    SpGemmDeviceCtx ctx;
    ctx.m = csrA.rows;
    ctx.n = csrB.cols;
    int64_t k = csrA.cols;
    ctx.alphaVal = alpha;
    ctx.betaVal  = beta;

    auto bytesA = ConvertValues(csrA.values, dtype);
    ctx.dARowOff = DeviceBuffer::copyFrom(csrA.rowOffsets.data(), (ctx.m + 1) * sizeof(int32_t));
    ctx.dAColInd = (csrA.nnz > 0)
        ? DeviceBuffer::copyFrom(csrA.colIndices.data(), csrA.nnz * sizeof(int32_t))
        : DeviceBuffer::alloc(1);
    ctx.dAVals = (csrA.nnz > 0)
        ? DeviceBuffer::copyFrom(bytesA.data(), bytesA.size())
        : DeviceBuffer::alloc(1);

    auto bytesB = ConvertValues(csrB.values, dtype);
    ctx.dBRowOff = DeviceBuffer::copyFrom(csrB.rowOffsets.data(), (k + 1) * sizeof(int32_t));
    ctx.dBColInd = (csrB.nnz > 0)
        ? DeviceBuffer::copyFrom(csrB.colIndices.data(), csrB.nnz * sizeof(int32_t))
        : DeviceBuffer::alloc(1);
    ctx.dBVals = (csrB.nnz > 0)
        ? DeviceBuffer::copyFrom(bytesB.data(), bytesB.size())
        : DeviceBuffer::alloc(1);

    std::vector<int32_t> hCRowOff(ctx.m + 1, 0);
    ctx.dCRowOff = DeviceBuffer::copyFrom(hCRowOff.data(), (ctx.m + 1) * sizeof(int32_t));
    ctx.dCColInd = DeviceBuffer::alloc(1);
    ctx.dCVals   = DeviceBuffer::alloc(1);

    ctx.matA = SpMatManager::createConstCsr(
        ctx.m, k, csrA.nnz, ctx.dARowOff.get(), ctx.dAColInd.get(), ctx.dAVals.get(),
        ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_BASE_ZERO, dtype);
    ctx.matB = SpMatManager::createConstCsr(
        k, ctx.n, csrB.nnz, ctx.dBRowOff.get(), ctx.dBColInd.get(), ctx.dBVals.get(),
        ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_BASE_ZERO, dtype);
    ctx.matC = SpMatManager::createCsr(
        ctx.m, ctx.n, 0, ctx.dCRowOff.get(), ctx.dCColInd.get(), ctx.dCVals.get(),
        ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_BASE_ZERO, dtype);
    return ctx;
}

// Step 2: run 7-interface API (CreateDescr → WorkEstimation → GetNumProducts
//         → allocate C → Compute → Copy)
inline void RunSpgemm7Interface(
    HandleManager& spHandle, SpGemmDeviceCtx& ctx,
    aclDataType dtype, aclsparseSpGEMMAlg_t algEnum,
    const std::string& caseId)
{
    const float* alphaPtr = &ctx.alphaVal;
    const float* betaPtr  = &ctx.betaVal;

    aclsparseSpGEMMDescr_t descr = nullptr;
    SpgemmCheckStatus(aclsparseSpGEMMCreateDescr(&descr), caseId, "CreateDescr");
    SpGEMMDescrGuard descrGuard(descr);

    size_t buffer1Size = 0;
    SpgemmCheckStatus(aclsparseSpGEMMWorkEstimation(
        spHandle.get(), descr, &buffer1Size,
        ACL_SPARSE_OP_NON_TRANSPOSE, ACL_SPARSE_OP_NON_TRANSPOSE,
        alphaPtr, ctx.matA.cget(), ctx.matB.cget(), betaPtr,
        ctx.matC.get(), dtype, algEnum, nullptr), caseId, "WorkEstimation(size)");

    ctx.buffer1 = (buffer1Size > 0) ? DeviceBuffer::alloc(buffer1Size) : DeviceBuffer::alloc(1);

    SpgemmCheckStatus(aclsparseSpGEMMWorkEstimation(
        spHandle.get(), descr, &buffer1Size,
        ACL_SPARSE_OP_NON_TRANSPOSE, ACL_SPARSE_OP_NON_TRANSPOSE,
        alphaPtr, ctx.matA.cget(), ctx.matB.cget(), betaPtr,
        ctx.matC.get(), dtype, algEnum, ctx.buffer1.get()), caseId, "WorkEstimation");

    int64_t numProducts = 0;
    SpgemmCheckStatus(aclsparseSpGEMMGetNumProducts(descr, &numProducts), caseId, "GetNumProducts");

    if (numProducts > 0) {
        ctx.dCColInd = DeviceBuffer::alloc(numProducts * sizeof(int32_t));
        size_t valBytes = (dtype == ACL_FLOAT)
            ? numProducts * sizeof(float) : numProducts * sizeof(uint16_t);
        ctx.dCVals = DeviceBuffer::alloc(valBytes);
        ctx.matC = SpMatManager::createCsr(
            ctx.m, ctx.n, 0, ctx.dCRowOff.get(), ctx.dCColInd.get(), ctx.dCVals.get(),
            ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_BASE_ZERO, dtype);
    }

    size_t buffer2Size = buffer1Size;
    ctx.buffer2 = (buffer2Size > 0) ? DeviceBuffer::alloc(buffer2Size) : DeviceBuffer::alloc(1);

    SpgemmCheckStatus(aclsparseSpGEMMCompute(
        spHandle.get(), descr,
        ACL_SPARSE_OP_NON_TRANSPOSE, ACL_SPARSE_OP_NON_TRANSPOSE,
        alphaPtr, ctx.matA.cget(), ctx.matB.cget(), betaPtr,
        ctx.matC.get(), dtype, algEnum, ctx.buffer1.get(), ctx.buffer2.get()),
        caseId, "Compute");

    SpgemmCheckStatus(aclsparseSpGEMMCopy(
        spHandle.get(), descr,
        ACL_SPARSE_OP_NON_TRANSPOSE, ACL_SPARSE_OP_NON_TRANSPOSE,
        alphaPtr, ctx.matA.cget(), ctx.matB.cget(), betaPtr,
        ctx.matC.get(), dtype, algEnum, ctx.buffer2.get()), caseId, "Copy");
}

// Step 3: read back CSR result from device
inline CsrMatrix ReadSpgemmResult(
    const SpGemmDeviceCtx& ctx, aclDataType dtype)
{
    CsrMatrix result;
    result.rows = ctx.m;
    result.cols = ctx.n;

    result.rowOffsets.resize(ctx.m + 1);
    ctx.dCRowOff.copyToHost(result.rowOffsets.data(), (ctx.m + 1) * sizeof(int32_t));

    int32_t nnzC = result.rowOffsets[ctx.m];
    result.nnz = nnzC;

    if (nnzC > 0) {
        result.colIndices.resize(nnzC);
        ctx.dCColInd.copyToHost(result.colIndices.data(), nnzC * sizeof(int32_t));
        size_t valBytes = (dtype == ACL_FLOAT)
            ? nnzC * sizeof(float) : nnzC * sizeof(uint16_t);
        std::vector<uint8_t> rawVals(valBytes);
        ctx.dCVals.copyToHost(rawVals.data(), valBytes);
        result.values = ConvertToFloat(rawVals.data(), nnzC, dtype);
    }
    return result;
}

inline CsrMatrix SpGEMMNpuWrapper(
    HandleManager& spHandle,
    aclrtStream stream,
    const CsrMatrix& csrA,
    const CsrMatrix& csrB,
    float alpha, float beta,
    aclDataType dtype,
    aclsparseSpGEMMAlg_t algEnum,
    const std::string& caseId)
{
    if (csrA.rows <= 0 || csrB.cols <= 0) {
        CsrMatrix empty;
        empty.rows = csrA.rows;
        empty.cols = csrB.cols;
        empty.nnz = 0;
        empty.rowOffsets.assign(csrA.rows + 1, 0);
        return empty;
    }

    SpGemmDeviceCtx ctx = PrepareSpgemmInputs(csrA, csrB, alpha, beta, dtype);
    RunSpgemm7Interface(spHandle, ctx, dtype, algEnum, caseId);

    if (aclrtSynchronizeStream(stream) != ACL_SUCCESS) {
        throw std::runtime_error("[" + caseId + "] SynchronizeStream failed");
    }
    return ReadSpgemmResult(ctx, dtype);
}

}  // namespace sparse_test

#endif
