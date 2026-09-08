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

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "securec.h"
#include "test_common.h"

using sparse_test::AclEnvScope;
using sparse_test::DeviceBuffer;
using sparse_test::HandleManager;
using sparse_test::SpMatManager;

namespace {

struct TypeParam {
    const char *name;
    aclDataType type;
    float tolerance;
};

static void CheckedCopy(void *destination, size_t destinationSize,
    const void *source, size_t sourceSize)
{
    if (memcpy_s(destination, destinationSize, source, sourceSize) != EOK) {
        throw std::runtime_error("failed to copy SpGEMM test data");
    }
}

static uint32_t FloatBits(float value)
{
    uint32_t bits = 0;
    CheckedCopy(&bits, sizeof(bits), &value, sizeof(value));
    return bits;
}

static uint16_t FloatToHalf(float value)
{
    uint32_t bits = FloatBits(value);
    uint32_t sign = (bits >> 16U) & 0x8000U;
    int32_t exponent = static_cast<int32_t>((bits >> 23U) & 0xFFU) - 127 + 15;
    uint32_t mantissa = bits & 0x7FFFFFU;
    if (exponent <= 0) { return static_cast<uint16_t>(sign); }
    if (exponent >= 31) { return static_cast<uint16_t>(sign | 0x7C00U); }
    uint32_t rounded = mantissa + 0x1000U;
    if ((rounded & 0x800000U) != 0U) {
        rounded = 0;
        ++exponent;
    }
    if (exponent >= 31) { return static_cast<uint16_t>(sign | 0x7C00U); }
    return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exponent) << 10U) | (rounded >> 13U));
}

static float HalfToFloat(uint16_t bits)
{
    constexpr uint16_t kSignMask = 0x8000U;
    constexpr uint16_t kMantissaMask = 0x03FFU;
    const uint32_t sign = static_cast<uint32_t>(bits & kSignMask) << 16U;
    const uint32_t exponent = (bits >> 10U) & 0x1FU;
    uint32_t mantissa = bits & 0x03FFU;
    uint32_t result = 0;
    if (exponent == 0U) {
        if (mantissa == 0U) {
            result = sign;
        } else {
            int32_t shift = 0;
            while ((mantissa & 0x0400U) == 0U) {
                mantissa <<= 1U;
                ++shift;
            }
            mantissa &= kMantissaMask;
            result = sign | (static_cast<uint32_t>(113 - shift) << 23U) | (mantissa << 13U);
        }
    } else if (exponent == 31U) {
        result = sign | 0x7F800000U | (mantissa << 13U);
    } else {
        result = sign | ((exponent + 112U) << 23U) | (mantissa << 13U);
    }
    float value = 0.0F;
    CheckedCopy(&value, sizeof(value), &result, sizeof(result));
    return value;
}

static uint16_t FloatToBFloat16(float value)
{
    uint32_t bits = FloatBits(value);
    uint32_t lsb = (bits >> 16U) & 1U;
    return static_cast<uint16_t>((bits + 0x7FFFU + lsb) >> 16U);
}

static float BFloat16ToFloat(uint16_t value)
{
    uint32_t bits = static_cast<uint32_t>(value) << 16U;
    float result = 0.0F;
    CheckedCopy(&result, sizeof(result), &bits, sizeof(bits));
    return result;
}

static size_t TypeSize(aclDataType type)
{
    if (type == ACL_FLOAT16 || type == ACL_BF16) { return 2U; }
    if (type == ACL_FLOAT) { return 4U; }
    return sizeof(aclsparseComplex);
}

static std::vector<uint8_t> EncodeValues(
    const std::vector<std::complex<float>> &values, aclDataType type)
{
    std::vector<uint8_t> bytes(values.size() * TypeSize(type));
    for (size_t i = 0; i < values.size(); ++i) {
        if (type == ACL_FLOAT16 || type == ACL_BF16) {
            uint16_t bits = type == ACL_FLOAT16 ? FloatToHalf(values[i].real()) :
                FloatToBFloat16(values[i].real());
            size_t offset = i * sizeof(bits);
            CheckedCopy(bytes.data() + offset, bytes.size() - offset, &bits, sizeof(bits));
        } else if (type == ACL_FLOAT) {
            float value = values[i].real();
            size_t offset = i * sizeof(value);
            CheckedCopy(bytes.data() + offset, bytes.size() - offset, &value, sizeof(value));
        } else {
            aclsparseComplex value{values[i].real(), values[i].imag()};
            size_t offset = i * sizeof(value);
            CheckedCopy(bytes.data() + offset, bytes.size() - offset, &value, sizeof(value));
        }
    }
    return bytes;
}

static std::vector<std::complex<float>> DecodeValues(
    const std::vector<uint8_t> &bytes, size_t count, aclDataType type)
{
    std::vector<std::complex<float>> values(count);
    for (size_t i = 0; i < count; ++i) {
        if (type == ACL_FLOAT16 || type == ACL_BF16) {
            uint16_t bits = 0;
            CheckedCopy(&bits, sizeof(bits), bytes.data() + i * sizeof(bits), sizeof(bits));
            float value = type == ACL_FLOAT16 ? HalfToFloat(bits) : BFloat16ToFloat(bits);
            values[i] = {value, 0.0F};
        } else if (type == ACL_FLOAT) {
            float value = 0.0F;
            CheckedCopy(&value, sizeof(value), bytes.data() + i * sizeof(value), sizeof(value));
            values[i] = {value, 0.0F};
        } else {
            aclsparseComplex value{};
            CheckedCopy(&value, sizeof(value), bytes.data() + i * sizeof(value), sizeof(value));
            values[i] = {value.x, value.y};
        }
    }
    return values;
}

class SpGemmDescrGuard {
public:
    SpGemmDescrGuard()
    {
        if (aclsparseSpGEMMCreateDescr(&descr_) != ACL_SPARSE_STATUS_SUCCESS) {
            throw std::runtime_error("aclsparseSpGEMMCreateDescr failed");
        }
    }
    ~SpGemmDescrGuard() { aclsparseSpGEMMDestroyDescr(descr_); }
    aclsparseSpGEMMDescr_t get() const { return descr_; }
private:
    aclsparseSpGEMMDescr_t descr_ = nullptr;
};

struct RunResult {
    std::vector<int32_t> rows;
    std::vector<int32_t> cols;
    std::vector<std::complex<float>> values;
    int64_t products = 0;
};

static void RequireStatus(aclsparseStatus_t status, const char *stage)
{
    if (status != ACL_SPARSE_STATUS_SUCCESS) {
        throw std::runtime_error(std::string(stage) + " failed, status=" +
            std::to_string(static_cast<int>(status)));
    }
}

struct HostProblem {
    int32_t m = 3;
    int32_t k = 4;
    int32_t n = 3;
    std::vector<int32_t> rowsA{0, 3, 4, 6};
    std::vector<int32_t> colsA{0, 1, 1, 2, 1, 3};
    std::vector<int32_t> rowsB{0, 1, 3, 5, 6};
    std::vector<int32_t> colsB{2, 0, 1, 1, 2, 0};
    std::vector<std::complex<float>> valsA{
        {1.0F, 0.25F}, {2.0F, 0.5F}, {-2.0F, -0.5F},
        {3.0F, 0.75F}, {4.0F, 1.0F}, {5.0F, 1.25F}};
    std::vector<std::complex<float>> valsB{
        {7.0F, -3.5F}, {2.0F, -1.0F}, {3.0F, -1.5F},
        {4.0F, -2.0F}, {5.0F, -2.5F}, {6.0F, -3.0F}};
};

static void MakeRegularProblem(HostProblem &problem)
{
    constexpr int32_t kSize = 67;
    constexpr int32_t kDegree = 8;
    problem.m = kSize;
    problem.k = kSize;
    problem.n = kSize;
    problem.rowsA.resize(kSize + 1);
    problem.rowsB.resize(kSize + 1);
    problem.colsA.resize(kSize * kDegree);
    problem.colsB.resize(kSize * kDegree);
    for (int32_t row = 0; row < kSize; ++row) {
        problem.rowsA[row] = row * kDegree;
        problem.rowsB[row] = row * kDegree;
        auto beginA = problem.colsA.begin() + row * kDegree;
        auto beginB = problem.colsB.begin() + row * kDegree;
        for (int32_t j = 0; j < kDegree; ++j) {
            beginA[j] = (row + j) % kSize;
            beginB[j] = (row + j * kDegree) % kSize;
        }
        std::sort(beginA, beginA + kDegree);
        std::sort(beginB, beginB + kDegree);
    }
    problem.rowsA[kSize] = kSize * kDegree;
    problem.rowsB[kSize] = kSize * kDegree;
    problem.valsA.assign(problem.colsA.size(), {1.0F, 0.0F});
    problem.valsB.assign(problem.colsB.size(), {1.0F, 0.0F});
}

static HostProblem MakeHostProblem(const TypeParam &param, bool regularInput)
{
    HostProblem problem;
    if (regularInput) {
        MakeRegularProblem(problem);
    }
    if (param.type != ACL_COMPLEX64) {
        for (auto &value : problem.valsA) {
            value.imag(0.0F);
        }
        for (auto &value : problem.valsB) {
            value.imag(0.0F);
        }
    }
    return problem;
}

struct DeviceProblem {
    DeviceBuffer rowsA;
    DeviceBuffer colsA;
    DeviceBuffer valsA;
    DeviceBuffer rowsB;
    DeviceBuffer colsB;
    DeviceBuffer valsB;
    DeviceBuffer rowsC;
    DeviceBuffer colsCInput;
    DeviceBuffer valsCInput;
    SpMatManager matA;
    SpMatManager matB;
    SpMatManager matC;
};

static DeviceProblem UploadProblem(
    const HostProblem &problem, aclDataType type, const RunResult *inputC = nullptr)
{
    auto bytesA = EncodeValues(problem.valsA, type);
    auto bytesB = EncodeValues(problem.valsB, type);
    DeviceProblem device;
    device.rowsA = DeviceBuffer::copyFrom(problem.rowsA.data(), problem.rowsA.size() * sizeof(int32_t));
    device.colsA = DeviceBuffer::copyFrom(problem.colsA.data(), problem.colsA.size() * sizeof(int32_t));
    device.valsA = DeviceBuffer::copyFrom(bytesA.data(), bytesA.size());
    device.rowsB = DeviceBuffer::copyFrom(problem.rowsB.data(), problem.rowsB.size() * sizeof(int32_t));
    device.colsB = DeviceBuffer::copyFrom(problem.colsB.data(), problem.colsB.size() * sizeof(int32_t));
    device.valsB = DeviceBuffer::copyFrom(bytesB.data(), bytesB.size());
    int64_t nnzC = 0;
    void *colsC = nullptr;
    void *valsC = nullptr;
    if (inputC == nullptr) {
        device.rowsC = DeviceBuffer::alloc((static_cast<size_t>(problem.m) + 1U) * sizeof(int32_t));
    } else {
        if (inputC->rows.size() != static_cast<size_t>(problem.m) + 1U ||
            inputC->cols.size() != inputC->values.size()) {
            throw std::runtime_error("invalid prepopulated C input");
        }
        std::vector<uint8_t> bytesC = EncodeValues(inputC->values, type);
        device.rowsC = DeviceBuffer::copyFrom(
            inputC->rows.data(), inputC->rows.size() * sizeof(int32_t));
        device.colsCInput = DeviceBuffer::copyFrom(
            inputC->cols.data(), inputC->cols.size() * sizeof(int32_t));
        device.valsCInput = DeviceBuffer::copyFrom(bytesC.data(), bytesC.size());
        nnzC = static_cast<int64_t>(inputC->cols.size());
        colsC = device.colsCInput.raw();
        valsC = device.valsCInput.raw();
    }
    device.matA = SpMatManager::createConstCsr(
        problem.m, problem.k, static_cast<int64_t>(problem.colsA.size()),
        device.rowsA.raw(), device.colsA.raw(), device.valsA.raw(),
        ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_BASE_ZERO, type);
    device.matB = SpMatManager::createConstCsr(
        problem.k, problem.n, static_cast<int64_t>(problem.colsB.size()),
        device.rowsB.raw(), device.colsB.raw(), device.valsB.raw(),
        ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_BASE_ZERO, type);
    device.matC = SpMatManager::createCsr(problem.m, problem.n, nnzC, device.rowsC.raw(), colsC, valsC,
        ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_BASE_ZERO, type);
    return device;
}

struct StageBuffers {
    DeviceBuffer buffer1;
    DeviceBuffer buffer2;
    size_t size2 = 0;
    int64_t products = 0;
};

struct ScalarPointers {
    DeviceBuffer alphaDevice;
    DeviceBuffer betaDevice;
    const void *alpha = nullptr;
    const void *beta = nullptr;
};

static ScalarPointers PrepareScalarPointers(
    const std::vector<uint8_t> &alphaBytes, const std::vector<uint8_t> &betaBytes,
    aclsparsePointerMode_t pointerMode)
{
    ScalarPointers scalars;
    scalars.alpha = alphaBytes.data();
    scalars.beta = betaBytes.data();
    if (pointerMode == ACL_SPARSE_POINTER_MODE_DEVICE) {
        scalars.alphaDevice = DeviceBuffer::copyFrom(alphaBytes.data(), alphaBytes.size());
        scalars.betaDevice = DeviceBuffer::copyFrom(betaBytes.data(), betaBytes.size());
        scalars.alpha = scalars.alphaDevice.raw();
        scalars.beta = scalars.betaDevice.raw();
    }
    return scalars;
}

static StageBuffers RunStages(
    HandleManager &handle, DeviceProblem &device, const void *alpha,
    const void *beta, aclDataType type, aclsparseSpGEMMAlg_t alg,
    SpGemmDescrGuard &descr)
{
    StageBuffers stage;
    size_t size1 = 0;
    RequireStatus(aclsparseSpGEMMWorkEstimation(handle.get(), ACL_SPARSE_OP_NON_TRANSPOSE,
        ACL_SPARSE_OP_NON_TRANSPOSE, alpha, device.matA.cget(), device.matB.cget(), beta,
        device.matC.get(), type, alg, descr.get(), &size1, nullptr), "work query");
    stage.buffer1 = DeviceBuffer::alloc(size1);
    size_t shortSize1 = size1 - 1U;
    if (aclsparseSpGEMMWorkEstimation(handle.get(), ACL_SPARSE_OP_NON_TRANSPOSE,
        ACL_SPARSE_OP_NON_TRANSPOSE, alpha, device.matA.cget(), device.matB.cget(), beta,
        device.matC.get(), type, alg, descr.get(), &shortSize1, stage.buffer1.raw()) !=
        ACL_SPARSE_STATUS_INSUFFICIENT_RESOURCES) {
        throw std::runtime_error("undersized work buffer was not rejected");
    }
    RequireStatus(aclsparseSpGEMMWorkEstimation(handle.get(), ACL_SPARSE_OP_NON_TRANSPOSE,
        ACL_SPARSE_OP_NON_TRANSPOSE, alpha, device.matA.cget(), device.matB.cget(), beta,
        device.matC.get(), type, alg, descr.get(), &size1, stage.buffer1.raw()), "work execute");
    RequireStatus(aclsparseSpGEMMGetNumProducts(descr.get(), &stage.products), "get products");
    size_t size3 = 0;
    RequireStatus(aclsparseSpGEMMEstimateMemory(handle.get(), ACL_SPARSE_OP_NON_TRANSPOSE,
        ACL_SPARSE_OP_NON_TRANSPOSE, alpha, device.matA.cget(), device.matB.cget(), beta,
        device.matC.get(), type, alg, descr.get(), 1.0F,
        &size3, nullptr, &stage.size2), "estimate memory");
    if (size3 != 0U) {
        throw std::runtime_error("unexpected bufferSize3");
    }
    stage.buffer2 = DeviceBuffer::alloc(stage.size2);
    size_t shortSize2 = stage.size2 - 1U;
    if (aclsparseSpGEMMCompute(handle.get(), ACL_SPARSE_OP_NON_TRANSPOSE,
        ACL_SPARSE_OP_NON_TRANSPOSE, alpha, device.matA.cget(), device.matB.cget(), beta,
        device.matC.get(), type, alg, descr.get(), &shortSize2, stage.buffer2.raw()) !=
        ACL_SPARSE_STATUS_INSUFFICIENT_RESOURCES) {
        throw std::runtime_error("undersized compute buffer was not rejected");
    }
    RequireStatus(aclsparseSpGEMMCompute(handle.get(), ACL_SPARSE_OP_NON_TRANSPOSE,
        ACL_SPARSE_OP_NON_TRANSPOSE, alpha, device.matA.cget(), device.matB.cget(), beta,
        device.matC.get(), type, alg, descr.get(), &stage.size2, stage.buffer2.raw()), "compute");
    return stage;
}

static RunResult CollectResult(
    aclrtStream stream, HandleManager &handle, DeviceProblem &device,
    const HostProblem &problem, const void *alpha, const void *beta,
    aclDataType type, aclsparseSpGEMMAlg_t alg, SpGemmDescrGuard &descr,
    StageBuffers &stage, bool repeatWithOutput)
{
    int64_t rows = 0;
    int64_t cols = 0;
    int64_t nnz = 0;
    RequireStatus(aclsparseSpMatGetSize(device.matC.cget(), &rows, &cols, &nnz), "get size");
    DeviceBuffer colsC = DeviceBuffer::alloc(static_cast<size_t>(nnz) * sizeof(int32_t));
    DeviceBuffer valsC = DeviceBuffer::alloc(static_cast<size_t>(nnz) * TypeSize(type));
    RequireStatus(aclsparseCsrSetPointers(
        device.matC.get(), device.rowsC.raw(), colsC.raw(), valsC.raw()), "set C pointers");
    if (repeatWithOutput) {
        RequireStatus(aclsparseSpGEMMCompute(handle.get(), ACL_SPARSE_OP_NON_TRANSPOSE,
            ACL_SPARSE_OP_NON_TRANSPOSE, alpha, device.matA.cget(), device.matB.cget(), beta,
            device.matC.get(), type, alg, descr.get(), &stage.size2, stage.buffer2.raw()),
            "repeat compute");
    }
    RequireStatus(aclsparseSpGEMMCopy(handle.get(), ACL_SPARSE_OP_NON_TRANSPOSE,
        ACL_SPARSE_OP_NON_TRANSPOSE, alpha, device.matA.cget(), device.matB.cget(), beta,
        device.matC.get(), type, alg, descr.get()), "copy");
    if (aclrtSynchronizeStream(stream) != ACL_SUCCESS) {
        throw std::runtime_error("stream synchronize failed");
    }
    RunResult result;
    result.products = stage.products;
    result.rows.resize(static_cast<size_t>(problem.m) + 1U);
    result.cols.resize(static_cast<size_t>(nnz));
    std::vector<uint8_t> resultBytes(static_cast<size_t>(nnz) * TypeSize(type));
    device.rowsC.copyToHost(result.rows.data(), result.rows.size() * sizeof(int32_t));
    colsC.copyToHost(result.cols.data(), result.cols.size() * sizeof(int32_t));
    valsC.copyToHost(resultBytes.data(), resultBytes.size());
    result.values = DecodeValues(resultBytes, static_cast<size_t>(nnz), type);
    return result;
}

static RunResult RunSpGemm(
    aclrtStream stream, const TypeParam &param,
    bool regularInput = false, bool repeatWithOutput = false,
    aclsparseSpGEMMAlg_t alg = ACL_SPARSE_SPGEMM_DEFAULT,
    aclsparsePointerMode_t pointerMode = ACL_SPARSE_POINTER_MODE_HOST)
{
    HostProblem problem = MakeHostProblem(param, regularInput);
    DeviceProblem device = UploadProblem(problem, param.type);
    HandleManager handle;
    handle.setStream(stream);
    RequireStatus(aclsparseSetPointerMode(handle.get(), pointerMode), "set pointer mode");
    SpGemmDescrGuard descr;
    std::complex<float> alpha = param.type == ACL_COMPLEX64 ?
        std::complex<float>(0.5F, 0.25F) : std::complex<float>(1.0F, 0.0F);
    std::vector<uint8_t> alphaBytes = EncodeValues({alpha}, param.type);
    std::vector<uint8_t> betaBytes = EncodeValues({{0.0F, 0.0F}}, param.type);
    ScalarPointers scalars = PrepareScalarPointers(alphaBytes, betaBytes, pointerMode);
    StageBuffers stage = RunStages(handle, device, scalars.alpha, scalars.beta, param.type, alg, descr);
    return CollectResult(stream, handle, device, problem, scalars.alpha, scalars.beta,
        param.type, alg, descr, stage, repeatWithOutput);
}

static RunResult MakeInputC(aclDataType type)
{
    RunResult inputC;
    inputC.rows = {0, 3, 5, 7};
    inputC.cols = {0, 1, 2, 1, 2, 0, 1};
    inputC.values = {
        {1.0F, 0.5F}, {-2.0F, 1.0F}, {3.0F, -1.5F}, {-4.0F, 2.0F},
        {5.0F, -2.5F}, {-6.0F, 3.0F}, {7.0F, -3.5F}};
    if (type != ACL_COMPLEX64) {
        for (auto &value : inputC.values) {
            value.imag(0.0F);
        }
    }
    return inputC;
}

static std::vector<std::complex<float>> ExpectedProduct(aclDataType type)
{
    if (type == ACL_COMPLEX64) {
        return {{0.0F, 0.0F}, {0.0F, 0.0F}, {7.875F, -1.75F},
            {13.5F, -3.0F}, {16.875F, -3.75F}, {42.75F, -9.5F},
            {13.5F, -3.0F}};
    }
    return {{0.0F, 0.0F}, {0.0F, 0.0F}, {7.0F, 0.0F}, {12.0F, 0.0F},
        {15.0F, 0.0F}, {38.0F, 0.0F}, {12.0F, 0.0F}};
}

static RunResult RunSpGemmWithInputC(
    aclrtStream stream, const TypeParam &param, const HostProblem &problem,
    const RunResult &inputC, const std::vector<uint8_t> &alphaBytes,
    const std::vector<uint8_t> &betaBytes, aclsparsePointerMode_t pointerMode)
{
    DeviceProblem device = UploadProblem(problem, param.type, &inputC);
    HandleManager handle;
    handle.setStream(stream);
    RequireStatus(aclsparseSetPointerMode(handle.get(), pointerMode), "set pointer mode");
    SpGemmDescrGuard descr;
    ScalarPointers scalars = PrepareScalarPointers(alphaBytes, betaBytes, pointerMode);
    StageBuffers stage = RunStages(handle, device, scalars.alpha, scalars.beta,
        param.type, ACL_SPARSE_SPGEMM_DEFAULT, descr);
    return CollectResult(stream, handle, device, problem, scalars.alpha, scalars.beta,
        param.type, ACL_SPARSE_SPGEMM_DEFAULT, descr, stage, false);
}

static RunResult CollectLegacyResult(
    DeviceProblem &device, DeviceBuffer &colsC, DeviceBuffer &valsC,
    int64_t rows, int64_t nnz, aclDataType type)
{
    RunResult result;
    result.rows.resize(static_cast<size_t>(rows) + 1U);
    result.cols.resize(static_cast<size_t>(nnz));
    std::vector<uint8_t> valueBytes(static_cast<size_t>(nnz) * TypeSize(type));
    device.rowsC.copyToHost(result.rows.data(), result.rows.size() * sizeof(int32_t));
    colsC.copyToHost(result.cols.data(), result.cols.size() * sizeof(int32_t));
    valsC.copyToHost(valueBytes.data(), valueBytes.size());
    result.values = DecodeValues(valueBytes, static_cast<size_t>(nnz), type);
    return result;
}

static RunResult RunLegacySpGemm(aclrtStream stream, const TypeParam &param)
{
    HostProblem problem = MakeHostProblem(param, false);
    DeviceProblem device = UploadProblem(problem, param.type);
    HandleManager handle;
    handle.setStream(stream);
    RequireStatus(aclsparseSetPointerMode(
        handle.get(), ACL_SPARSE_POINTER_MODE_HOST), "set pointer mode");
    const std::complex<float> alpha = param.type == ACL_COMPLEX64 ?
        std::complex<float>(0.5F, 0.25F) : std::complex<float>(1.0F, 0.0F);
    std::vector<uint8_t> alphaBytes = EncodeValues({alpha}, param.type);
    std::vector<uint8_t> betaBytes = EncodeValues({{0.0F, 0.0F}}, param.type);
    size_t workspaceSize = 0;
    RequireStatus(aclsparseSpGEMMGetBufferSize(
        handle.get(), ACL_SPARSE_OP_NON_TRANSPOSE, ACL_SPARSE_OP_NON_TRANSPOSE,
        alphaBytes.data(), device.matA.cget(), device.matB.cget(), betaBytes.data(),
        device.matC.get(), param.type, ACL_SPARSE_SPGEMM_ALG_DEFAULT, &workspaceSize),
        "legacy workspace query");
    DeviceBuffer workspace = DeviceBuffer::alloc(workspaceSize);
    RequireStatus(aclsparseSpGEMMPreprocess(
        handle.get(), ACL_SPARSE_OP_NON_TRANSPOSE, ACL_SPARSE_OP_NON_TRANSPOSE,
        alphaBytes.data(), device.matA.cget(), device.matB.cget(), betaBytes.data(),
        device.matC.get(), param.type, ACL_SPARSE_SPGEMM_ALG_DEFAULT, workspace.raw()),
        "legacy preprocess");
    int64_t rows = 0;
    int64_t cols = 0;
    int64_t nnz = 0;
    RequireStatus(aclsparseSpMatGetSize(device.matC.cget(), &rows, &cols, &nnz), "legacy get size");
    DeviceBuffer colsC = DeviceBuffer::alloc(static_cast<size_t>(nnz) * sizeof(int32_t));
    DeviceBuffer valsC = DeviceBuffer::alloc(static_cast<size_t>(nnz) * TypeSize(param.type));
    RequireStatus(aclsparseCsrSetPointers(
        device.matC.get(), device.rowsC.raw(), colsC.raw(), valsC.raw()), "legacy set C pointers");
    RequireStatus(aclsparseSpGEMM(
        handle.get(), ACL_SPARSE_OP_NON_TRANSPOSE, ACL_SPARSE_OP_NON_TRANSPOSE,
        alphaBytes.data(), device.matA.cget(), device.matB.cget(), betaBytes.data(),
        device.matC.get(), param.type, ACL_SPARSE_SPGEMM_ALG_DEFAULT, workspace.raw()),
        "legacy execute");
    if (aclrtSynchronizeStream(stream) != ACL_SUCCESS) {
        throw std::runtime_error("legacy stream synchronize failed");
    }
    return CollectLegacyResult(device, colsC, valsC, rows, nnz, param.type);
}

static void ExpectSameResult(
    const RunResult &actual, const RunResult &expected, float tolerance)
{
    EXPECT_EQ(actual.rows, expected.rows);
    EXPECT_EQ(actual.cols, expected.cols);
    ASSERT_EQ(actual.values.size(), expected.values.size());
    for (size_t i = 0; i < actual.values.size(); ++i) {
        EXPECT_NEAR(actual.values[i].real(), expected.values[i].real(), tolerance) << "i=" << i;
        EXPECT_NEAR(actual.values[i].imag(), expected.values[i].imag(), tolerance) << "i=" << i;
    }
}

class SpGemmTest : public testing::TestWithParam<TypeParam> {
public:
    static void SetUpTestSuite() { env_ = std::make_unique<AclEnvScope>(); }
    static void TearDownTestSuite() { env_.reset(); }
protected:
    inline static std::unique_ptr<AclEnvScope> env_;
};

TEST_P(SpGemmTest, CanonicalCsrAndValues)
{
    const TypeParam param = GetParam();
    std::vector<std::complex<float>> expected;
    if (param.type == ACL_COMPLEX64) {
        const std::complex<float> alpha(0.5F, 0.25F);
        expected = {{0.0F, 0.0F}, {0.0F, 0.0F}, alpha * std::complex<float>(7.875F, -1.75F),
            alpha * std::complex<float>(13.5F, -3.0F), alpha * std::complex<float>(16.875F, -3.75F),
            alpha * std::complex<float>(42.75F, -9.5F), alpha * std::complex<float>(13.5F, -3.0F)};
    } else {
        expected = {{0.0F, 0.0F}, {0.0F, 0.0F}, {7.0F, 0.0F}, {12.0F, 0.0F},
            {15.0F, 0.0F}, {38.0F, 0.0F}, {12.0F, 0.0F}};
    }
    const std::vector<aclsparseSpGEMMAlg_t> algorithms{
        ACL_SPARSE_SPGEMM_DEFAULT, ACL_SPARSE_SPGEMM_ALG1,
        ACL_SPARSE_SPGEMM_ALG2, ACL_SPARSE_SPGEMM_ALG3};
    for (aclsparseSpGEMMAlg_t alg : algorithms) {
        RunResult result = RunSpGemm(env_->stream(), param, false, false, alg);
        EXPECT_EQ(result.products, 10) << "alg=" << static_cast<int>(alg);
        EXPECT_EQ(result.rows, (std::vector<int32_t>{0, 3, 5, 7}))
            << "alg=" << static_cast<int>(alg);
        EXPECT_EQ(result.cols, (std::vector<int32_t>{0, 1, 2, 1, 2, 0, 1}))
            << "alg=" << static_cast<int>(alg);
        ASSERT_EQ(result.values.size(), expected.size());
        for (size_t i = 0; i < expected.size(); ++i) {
            EXPECT_NEAR(result.values[i].real(), expected[i].real(), param.tolerance)
                << "alg=" << static_cast<int>(alg) << ", i=" << i;
            EXPECT_NEAR(result.values[i].imag(), expected[i].imag(), param.tolerance)
                << "alg=" << static_cast<int>(alg) << ", i=" << i;
        }
    }
}

TEST_P(SpGemmTest, NonzeroBetaWithPrepopulatedC)
{
    const TypeParam param = GetParam();
    HostProblem problem = MakeHostProblem(param, false);
    RunResult inputC = MakeInputC(param.type);
    const std::complex<float> alpha = param.type == ACL_COMPLEX64 ?
        std::complex<float>(0.5F, 0.25F) : std::complex<float>(1.0F, 0.0F);
    const std::complex<float> beta = param.type == ACL_COMPLEX64 ?
        std::complex<float>(0.25F, -0.5F) : std::complex<float>(0.5F, 0.0F);
    std::vector<uint8_t> alphaBytes = EncodeValues({alpha}, param.type);
    std::vector<uint8_t> betaBytes = EncodeValues({beta}, param.type);
    std::vector<std::complex<float>> product = ExpectedProduct(param.type);
    const std::vector<aclsparsePointerMode_t> pointerModes{
        ACL_SPARSE_POINTER_MODE_HOST, ACL_SPARSE_POINTER_MODE_DEVICE};
    for (aclsparsePointerMode_t pointerMode : pointerModes) {
        SCOPED_TRACE(pointerMode == ACL_SPARSE_POINTER_MODE_HOST ? "host scalars" : "device scalars");
        RunResult result = RunSpGemmWithInputC(
            env_->stream(), param, problem, inputC, alphaBytes, betaBytes, pointerMode);
        ASSERT_EQ(result.rows, inputC.rows);
        ASSERT_EQ(result.cols, inputC.cols);
        ASSERT_EQ(result.values.size(), inputC.values.size());
        for (size_t i = 0; i < result.values.size(); ++i) {
            const std::complex<float> expected = alpha * product[i] + beta * inputC.values[i];
            EXPECT_NEAR(result.values[i].real(), expected.real(), param.tolerance) << "i=" << i;
            EXPECT_NEAR(result.values[i].imag(), expected.imag(), param.tolerance) << "i=" << i;
        }
    }
}

TEST_P(SpGemmTest, DevicePointerModeZeroBetaUsesGenericPath)
{
    const TypeParam param = GetParam();
    RunResult hostResult = RunSpGemm(env_->stream(), param, true, false,
        ACL_SPARSE_SPGEMM_DEFAULT, ACL_SPARSE_POINTER_MODE_HOST);
    RunResult deviceResult = RunSpGemm(env_->stream(), param, true, false,
        ACL_SPARSE_SPGEMM_DEFAULT, ACL_SPARSE_POINTER_MODE_DEVICE);
    EXPECT_EQ(deviceResult.products, hostResult.products);
    ExpectSameResult(deviceResult, hostResult, param.tolerance);
}

TEST_P(SpGemmTest, LegacyThreeStageApiCompatibility)
{
    const TypeParam param = GetParam();
    RunResult expected = RunSpGemm(env_->stream(), param);
    RunResult actual = RunLegacySpGemm(env_->stream(), param);
    ExpectSameResult(actual, expected, param.tolerance);
}

TEST_P(SpGemmTest, RegularWrapAndDirectOutput)
{
    const TypeParam param = GetParam();
    RunResult result = RunSpGemm(env_->stream(), param, true, true);
    constexpr int32_t kSize = 67;
    constexpr int32_t kProductsPerRow = 64;
    ASSERT_EQ(result.products, static_cast<int64_t>(kSize) * kProductsPerRow);
    ASSERT_EQ(result.rows.size(), static_cast<size_t>(kSize + 1));
    ASSERT_EQ(result.cols.size(), static_cast<size_t>(kSize * kProductsPerRow));
    for (int32_t row = 0; row <= kSize; ++row) {
        EXPECT_EQ(result.rows[row], row * kProductsPerRow) << "row=" << row;
    }
    std::complex<float> expectedValue = param.type == ACL_COMPLEX64 ?
        std::complex<float>(0.5F, 0.25F) : std::complex<float>(1.0F, 0.0F);
    for (int32_t row = 0; row < kSize; ++row) {
        std::vector<int32_t> expectedCols(kProductsPerRow);
        for (int32_t j = 0; j < kProductsPerRow; ++j) {
            expectedCols[j] = (row + j) % kSize;
        }
        std::sort(expectedCols.begin(), expectedCols.end());
        for (int32_t j = 0; j < kProductsPerRow; ++j) {
            size_t index = static_cast<size_t>(row) * kProductsPerRow + j;
            EXPECT_EQ(result.cols[index], expectedCols[j]) << "row=" << row << ", j=" << j;
            EXPECT_NEAR(result.values[index].real(), expectedValue.real(), param.tolerance)
                << "row=" << row << ", j=" << j;
            EXPECT_NEAR(result.values[index].imag(), expectedValue.imag(), param.tolerance)
                << "row=" << row << ", j=" << j;
        }
    }
}

INSTANTIATE_TEST_SUITE_P(RequiredTypes, SpGemmTest, testing::Values(
    TypeParam{"fp16", ACL_FLOAT16, 0.05F},
    TypeParam{"bf16", ACL_BF16, 0.5F},
    TypeParam{"fp32", ACL_FLOAT, 1.0E-5F},
    TypeParam{"complex64", ACL_COMPLEX64, 1.0E-4F}),
    [](const testing::TestParamInfo<TypeParam> &info) { return info.param.name; });

TEST(SpGemmDescriptorTest, LifecycleAndStageOrder)
{
    EXPECT_EQ(aclsparseSpGEMMCreateDescr(nullptr), ACL_SPARSE_STATUS_INVALID_VALUE);
    EXPECT_EQ(aclsparseSpGEMMDestroyDescr(nullptr), ACL_SPARSE_STATUS_SUCCESS);
    aclsparseSpGEMMDescr_t descr = nullptr;
    ASSERT_EQ(aclsparseSpGEMMCreateDescr(&descr), ACL_SPARSE_STATUS_SUCCESS);
    EXPECT_EQ(aclsparseSpGEMMSetCInValid(descr, 1), ACL_SPARSE_STATUS_INVALID_VALUE);
    int64_t products = -1;
    EXPECT_EQ(aclsparseSpGEMMGetNumProducts(descr, &products), ACL_SPARSE_STATUS_INVALID_VALUE);
    EXPECT_EQ(aclsparseSpGEMMDestroyDescr(descr), ACL_SPARSE_STATUS_SUCCESS);
}

}  // namespace
