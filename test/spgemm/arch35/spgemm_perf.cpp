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
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "securec.h"
#include "test_common.h"

using sparse_test::DeviceBuffer;
using sparse_test::HandleManager;
using sparse_test::SpMatManager;

namespace {

class SpGemmDescrGuard {
public:
    SpGemmDescrGuard()
    {
        Check(aclsparseSpGEMMCreateDescr(&descr_), "create descriptor");
    }
    ~SpGemmDescrGuard() { aclsparseSpGEMMDestroyDescr(descr_); }
    aclsparseSpGEMMDescr_t get() const { return descr_; }
    static void Check(aclsparseStatus_t status, const char *stage)
    {
        if (status != ACL_SPARSE_STATUS_SUCCESS) {
            throw std::runtime_error(std::string(stage) + " failed: " +
                std::to_string(static_cast<int>(status)));
        }
    }
private:
    aclsparseSpGEMMDescr_t descr_ = nullptr;
};

struct Options {
    int32_t n = 19717;
    int32_t d = 4;
    int32_t warmup = 10;
    int32_t samples = 30;
    aclDataType type = ACL_FLOAT;
    std::string typeName = "float32";
};

static void CheckedCopy(void *destination, size_t destinationSize,
    const void *source, size_t sourceSize)
{
    if (memcpy_s(destination, destinationSize, source, sourceSize) != EOK) {
        throw std::runtime_error("failed to copy SpGEMM benchmark data");
    }
}

static aclDataType ParseType(const std::string &name)
{
    if (name == "float16") {
        return ACL_FLOAT16;
    }
    if (name == "bfloat16") {
        return ACL_BF16;
    }
    if (name == "float32") {
        return ACL_FLOAT;
    }
    if (name == "complex64") {
        return ACL_COMPLEX64;
    }
    throw std::runtime_error("unsupported dtype");
}

static void ParseOption(
    const std::string &arg, int argc, char **argv, int &index, Options &options)
{
    if (arg == "--n" && index + 1 < argc) {
        options.n = std::stoi(argv[++index]);
    } else if (arg == "--d" && index + 1 < argc) {
        options.d = std::stoi(argv[++index]);
    } else if (arg == "--warmup" && index + 1 < argc) {
        options.warmup = std::stoi(argv[++index]);
    } else if (arg == "--samples" && index + 1 < argc) {
        options.samples = std::stoi(argv[++index]);
    } else if (arg == "--dtype" && index + 1 < argc) {
        options.typeName = argv[++index];
        options.type = ParseType(options.typeName);
    } else {
        throw std::runtime_error("invalid benchmark argument: " + arg);
    }
}

static Options ParseOptions(int argc, char **argv)
{
    Options options;
    for (int i = 1; i < argc; ++i) {
        std::string arg(argv[i]);
        ParseOption(arg, argc, argv, i, options);
    }
    if (options.n <= 0 || options.d <= 0 || options.d > options.n ||
        options.warmup < 0 || options.samples <= 0) {
        throw std::runtime_error("invalid benchmark dimensions/counts");
    }
    return options;
}

static size_t TypeSize(aclDataType type)
{
    if (type == ACL_FLOAT16 || type == ACL_BF16) { return 2U; }
    if (type == ACL_FLOAT) { return 4U; }
    return sizeof(aclsparseComplex);
}

static std::vector<uint8_t> Ones(size_t count, aclDataType type)
{
    std::vector<uint8_t> values(count * TypeSize(type));
    for (size_t i = 0; i < count; ++i) {
        if (type == ACL_FLOAT16) {
            uint16_t one = 0x3C00U;
            size_t offset = i * sizeof(one);
            CheckedCopy(values.data() + offset, values.size() - offset, &one, sizeof(one));
        } else if (type == ACL_BF16) {
            uint16_t one = 0x3F80U;
            size_t offset = i * sizeof(one);
            CheckedCopy(values.data() + offset, values.size() - offset, &one, sizeof(one));
        } else if (type == ACL_FLOAT) {
            float one = 1.0F;
            size_t offset = i * sizeof(one);
            CheckedCopy(values.data() + offset, values.size() - offset, &one, sizeof(one));
        } else {
            aclsparseComplex one{1.0F, 0.0F};
            size_t offset = i * sizeof(one);
            CheckedCopy(values.data() + offset, values.size() - offset, &one, sizeof(one));
        }
    }
    return values;
}

static std::vector<uint8_t> Scalar(float real, aclDataType type)
{
    std::vector<uint8_t> value(TypeSize(type), 0U);
    if (type == ACL_FLOAT16) {
        uint16_t bits = real == 0.0F ? 0U : 0x3C00U;
        CheckedCopy(value.data(), value.size(), &bits, sizeof(bits));
    } else if (type == ACL_BF16) {
        uint16_t bits = real == 0.0F ? 0U : 0x3F80U;
        CheckedCopy(value.data(), value.size(), &bits, sizeof(bits));
    } else if (type == ACL_FLOAT) {
        CheckedCopy(value.data(), value.size(), &real, sizeof(real));
    } else {
        aclsparseComplex scalar{real, 0.0F};
        CheckedCopy(value.data(), value.size(), &scalar, sizeof(scalar));
    }
    return value;
}

static void BuildInputs(
    int32_t n, int32_t d, std::vector<int32_t> &rowOffsets,
    std::vector<int32_t> &colsA, std::vector<int32_t> &colsB)
{
    if (n <= 0 || d <= 0 || d > n) {
        throw std::runtime_error("invalid benchmark matrix dimensions");
    }
    int64_t nnz64 = static_cast<int64_t>(n) * d;
    if (nnz64 > std::numeric_limits<int32_t>::max()) {
        throw std::runtime_error("input nnz exceeds int32");
    }
    rowOffsets.resize(static_cast<size_t>(n) + 1U);
    colsA.resize(static_cast<size_t>(nnz64));
    colsB.resize(static_cast<size_t>(nnz64));
    for (int32_t row = 0; row < n; ++row) {
        rowOffsets[static_cast<size_t>(row)] = row * d;
        auto beginA = colsA.begin() + static_cast<int64_t>(row) * d;
        auto beginB = colsB.begin() + static_cast<int64_t>(row) * d;
        int32_t columnA = row;
        int32_t columnB = row;
        for (int32_t a = 0; a < d; ++a) {
            beginA[a] = columnA;
            beginB[a] = columnB;
            columnA = (columnA + 1 == n) ? 0 : columnA + 1;
            int64_t nextColumnB = static_cast<int64_t>(columnB) + d;
            columnB = static_cast<int32_t>(nextColumnB >= n ? nextColumnB - n : nextColumnB);
        }
        std::sort(beginA, beginA + d);
        std::sort(beginB, beginB + d);
    }
    rowOffsets[static_cast<size_t>(n)] = static_cast<int32_t>(nnz64);
}

static double Percentile(std::vector<double> values, double q)
{
    if (values.empty()) {
        throw std::invalid_argument("percentile requires non-empty values");
    }
    std::sort(values.begin(), values.end());
    size_t index = static_cast<size_t>(q * static_cast<double>(values.size() - 1U));
    return values[index];
}

struct Timings {
    std::vector<double> workUs;
    std::vector<double> memoryUs;
    std::vector<double> computeUs;
    std::vector<double> copyUs;
    std::vector<double> totalUs;
};

static void RunOneIteration(
    HandleManager &handle, aclrtStream stream, const void *alpha,
    SpMatManager &matA, SpMatManager &matB, const void *beta,
    SpMatManager &matC, aclDataType type, SpGemmDescrGuard &descr,
    size_t size1, DeviceBuffer &buffer1,
    size_t size2, DeviceBuffer &buffer2, Timings *timings)
{
    auto start = std::chrono::steady_clock::now();
    SpGemmDescrGuard::Check(aclsparseSpGEMMWorkEstimation(
        handle.get(), ACL_SPARSE_OP_NON_TRANSPOSE, ACL_SPARSE_OP_NON_TRANSPOSE,
        alpha, matA.cget(), matB.cget(), beta, matC.get(), type,
        ACL_SPARSE_SPGEMM_DEFAULT, descr.get(), &size1, buffer1.raw()), "work execute");
    auto afterWork = std::chrono::steady_clock::now();
    size_t size3 = 0;
    SpGemmDescrGuard::Check(aclsparseSpGEMMEstimateMemory(
        handle.get(), ACL_SPARSE_OP_NON_TRANSPOSE, ACL_SPARSE_OP_NON_TRANSPOSE,
        alpha, matA.cget(), matB.cget(), beta, matC.get(), type,
        ACL_SPARSE_SPGEMM_DEFAULT, descr.get(), 1.0F, &size3, nullptr, &size2),
        "estimate memory");
    auto afterMemory = std::chrono::steady_clock::now();
    SpGemmDescrGuard::Check(aclsparseSpGEMMCompute(
        handle.get(), ACL_SPARSE_OP_NON_TRANSPOSE, ACL_SPARSE_OP_NON_TRANSPOSE,
        alpha, matA.cget(), matB.cget(), beta, matC.get(), type,
        ACL_SPARSE_SPGEMM_DEFAULT, descr.get(), &size2, buffer2.raw()), "compute");
    if (aclrtSynchronizeStream(stream) != ACL_SUCCESS) {
        throw std::runtime_error("compute synchronization failed");
    }
    auto afterCompute = std::chrono::steady_clock::now();
    SpGemmDescrGuard::Check(aclsparseSpGEMMCopy(
        handle.get(), ACL_SPARSE_OP_NON_TRANSPOSE, ACL_SPARSE_OP_NON_TRANSPOSE,
        alpha, matA.cget(), matB.cget(), beta, matC.get(), type,
        ACL_SPARSE_SPGEMM_DEFAULT, descr.get()), "copy");
    if (aclrtSynchronizeStream(stream) != ACL_SUCCESS) {
        throw std::runtime_error("stream synchronization failed");
    }
    auto end = std::chrono::steady_clock::now();
    if (timings != nullptr) {
        double workUs = std::chrono::duration<double, std::micro>(afterWork - start).count();
        double memoryUs = std::chrono::duration<double, std::micro>(afterMemory - afterWork).count();
        double computeUs = std::chrono::duration<double, std::micro>(afterCompute - afterMemory).count();
        double copyUs = std::chrono::duration<double, std::micro>(end - afterCompute).count();
        timings->workUs.push_back(workUs);
        timings->memoryUs.push_back(memoryUs);
        timings->computeUs.push_back(computeUs);
        timings->copyUs.push_back(copyUs);
        timings->totalUs.push_back(workUs + memoryUs + computeUs + copyUs);
    }
}

struct PerfHostData {
    std::vector<int32_t> rows;
    std::vector<int32_t> colsA;
    std::vector<int32_t> colsB;
    std::vector<uint8_t> values;
    std::vector<uint8_t> alpha;
    std::vector<uint8_t> beta;
};

static PerfHostData MakePerfHostData(const Options &options)
{
    PerfHostData host;
    BuildInputs(options.n, options.d, host.rows, host.colsA, host.colsB);
    host.values = Ones(host.colsA.size(), options.type);
    host.alpha = Scalar(1.0F, options.type);
    host.beta = Scalar(0.0F, options.type);
    return host;
}

struct PerfDeviceData {
    DeviceBuffer rowsA;
    DeviceBuffer rowsB;
    DeviceBuffer colsA;
    DeviceBuffer colsB;
    DeviceBuffer valsA;
    DeviceBuffer valsB;
    DeviceBuffer rowsC;
    SpMatManager matA;
    SpMatManager matB;
    SpMatManager matC;
};

static PerfDeviceData UploadPerfData(const PerfHostData &host, const Options &options)
{
    PerfDeviceData device;
    device.rowsA = DeviceBuffer::copyFrom(host.rows.data(), host.rows.size() * sizeof(int32_t));
    device.rowsB = DeviceBuffer::copyFrom(host.rows.data(), host.rows.size() * sizeof(int32_t));
    device.colsA = DeviceBuffer::copyFrom(host.colsA.data(), host.colsA.size() * sizeof(int32_t));
    device.colsB = DeviceBuffer::copyFrom(host.colsB.data(), host.colsB.size() * sizeof(int32_t));
    device.valsA = DeviceBuffer::copyFrom(host.values.data(), host.values.size());
    device.valsB = DeviceBuffer::copyFrom(host.values.data(), host.values.size());
    device.rowsC = DeviceBuffer::alloc(host.rows.size() * sizeof(int32_t));
    int64_t nnzInput = static_cast<int64_t>(host.colsA.size());
    device.matA = SpMatManager::createConstCsr(options.n, options.n, nnzInput,
        device.rowsA.raw(), device.colsA.raw(), device.valsA.raw(), ACL_SPARSE_INDEX_32I,
        ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_BASE_ZERO, options.type);
    device.matB = SpMatManager::createConstCsr(options.n, options.n, nnzInput,
        device.rowsB.raw(), device.colsB.raw(), device.valsB.raw(), ACL_SPARSE_INDEX_32I,
        ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_BASE_ZERO, options.type);
    device.matC = SpMatManager::createCsr(options.n, options.n, 0,
        device.rowsC.raw(), nullptr, nullptr, ACL_SPARSE_INDEX_32I,
        ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_BASE_ZERO, options.type);
    return device;
}

struct PerfStage {
    DeviceBuffer buffer1;
    DeviceBuffer buffer2;
    DeviceBuffer colsC;
    DeviceBuffer valsC;
    size_t size1 = 0;
    size_t size2 = 0;
    int64_t products = 0;
    int64_t nnzC = 0;
};

static PerfStage InitializePerfStage(
    HandleManager &handle, PerfDeviceData &device,
    const PerfHostData &host, const Options &options, SpGemmDescrGuard &descr)
{
    PerfStage stage;
    SpGemmDescrGuard::Check(aclsparseSpGEMMWorkEstimation(
        handle.get(), ACL_SPARSE_OP_NON_TRANSPOSE, ACL_SPARSE_OP_NON_TRANSPOSE,
        host.alpha.data(), device.matA.cget(), device.matB.cget(), host.beta.data(),
        device.matC.get(), options.type, ACL_SPARSE_SPGEMM_DEFAULT,
        descr.get(), &stage.size1, nullptr), "work query");
    stage.buffer1 = DeviceBuffer::alloc(stage.size1);
    SpGemmDescrGuard::Check(aclsparseSpGEMMWorkEstimation(
        handle.get(), ACL_SPARSE_OP_NON_TRANSPOSE, ACL_SPARSE_OP_NON_TRANSPOSE,
        host.alpha.data(), device.matA.cget(), device.matB.cget(), host.beta.data(),
        device.matC.get(), options.type, ACL_SPARSE_SPGEMM_DEFAULT,
        descr.get(), &stage.size1, stage.buffer1.raw()), "work execute");
    SpGemmDescrGuard::Check(
        aclsparseSpGEMMGetNumProducts(descr.get(), &stage.products), "get products");
    size_t size3 = 0;
    SpGemmDescrGuard::Check(aclsparseSpGEMMEstimateMemory(
        handle.get(), ACL_SPARSE_OP_NON_TRANSPOSE, ACL_SPARSE_OP_NON_TRANSPOSE,
        host.alpha.data(), device.matA.cget(), device.matB.cget(), host.beta.data(),
        device.matC.get(), options.type, ACL_SPARSE_SPGEMM_DEFAULT,
        descr.get(), 1.0F, &size3, nullptr, &stage.size2), "estimate memory");
    stage.buffer2 = DeviceBuffer::alloc(stage.size2);
    SpGemmDescrGuard::Check(aclsparseSpGEMMCompute(
        handle.get(), ACL_SPARSE_OP_NON_TRANSPOSE, ACL_SPARSE_OP_NON_TRANSPOSE,
        host.alpha.data(), device.matA.cget(), device.matB.cget(), host.beta.data(),
        device.matC.get(), options.type, ACL_SPARSE_SPGEMM_DEFAULT,
        descr.get(), &stage.size2, stage.buffer2.raw()), "initial compute");
    int64_t outRows = 0;
    int64_t outCols = 0;
    SpGemmDescrGuard::Check(aclsparseSpMatGetSize(
        device.matC.cget(), &outRows, &outCols, &stage.nnzC), "get C size");
    stage.colsC = DeviceBuffer::alloc(static_cast<size_t>(stage.nnzC) * sizeof(int32_t));
    stage.valsC = DeviceBuffer::alloc(static_cast<size_t>(stage.nnzC) * TypeSize(options.type));
    SpGemmDescrGuard::Check(aclsparseCsrSetPointers(
        device.matC.get(), device.rowsC.raw(), stage.colsC.raw(), stage.valsC.raw()), "set C pointers");
    return stage;
}

static Timings RunSamples(
    HandleManager &handle, aclrtStream stream, PerfDeviceData &device,
    const PerfHostData &host, const Options &options,
    SpGemmDescrGuard &descr, PerfStage &stage)
{
    for (int32_t i = 0; i < options.warmup; ++i) {
        RunOneIteration(handle, stream, host.alpha.data(), device.matA, device.matB, host.beta.data(),
            device.matC, options.type, descr, stage.size1, stage.buffer1,
            stage.size2, stage.buffer2, nullptr);
    }
    Timings timings;
    for (int32_t i = 0; i < options.samples; ++i) {
        RunOneIteration(handle, stream, host.alpha.data(), device.matA, device.matB, host.beta.data(),
            device.matC, options.type, descr, stage.size1, stage.buffer1,
            stage.size2, stage.buffer2, &timings);
    }
    return timings;
}

static void PrintSummary(
    const Options &options, int64_t nnzInput,
    const PerfStage &stage, const Timings &timings)
{
    std::cout << std::fixed << std::setprecision(3)
        << "{\"n\":" << options.n
        << ",\"d\":" << options.d
        << ",\"dtype\":\"" << options.typeName << "\""
        << ",\"nnz_a\":" << nnzInput
        << ",\"products\":" << stage.products
        << ",\"nnz_c\":" << stage.nnzC
        << ",\"buffer1_bytes\":" << stage.size1
        << ",\"buffer2_bytes\":" << stage.size2
        << ",\"work_median_us\":" << Percentile(timings.workUs, 0.5)
        << ",\"work_p90_us\":" << Percentile(timings.workUs, 0.9)
        << ",\"memory_median_us\":" << Percentile(timings.memoryUs, 0.5)
        << ",\"memory_p90_us\":" << Percentile(timings.memoryUs, 0.9)
        << ",\"compute_median_us\":" << Percentile(timings.computeUs, 0.5)
        << ",\"compute_p90_us\":" << Percentile(timings.computeUs, 0.9)
        << ",\"copy_median_us\":" << Percentile(timings.copyUs, 0.5)
        << ",\"copy_p90_us\":" << Percentile(timings.copyUs, 0.9)
        << ",\"total_median_us\":" << Percentile(timings.totalUs, 0.5)
        << ",\"total_p90_us\":" << Percentile(timings.totalUs, 0.9)
        << ",\"warmup\":" << options.warmup
        << ",\"samples\":" << options.samples << "}" << std::endl;
}

static int Benchmark(const Options &options)
{
    PerfHostData host = MakePerfHostData(options);
    sparse_test::AclEnvScope env;
    HandleManager handle;
    handle.setStream(env.stream());
    SpGemmDescrGuard::Check(
        aclsparseSetPointerMode(handle.get(), ACL_SPARSE_POINTER_MODE_HOST), "pointer mode");
    PerfDeviceData device = UploadPerfData(host, options);
    SpGemmDescrGuard descr;
    PerfStage stage = InitializePerfStage(handle, device, host, options, descr);
    Timings timings = RunSamples(handle, env.stream(), device, host, options, descr, stage);
    int64_t expectedNnzC = static_cast<int64_t>(options.n) * options.d * options.d;
    if (stage.nnzC != expectedNnzC || stage.products != expectedNnzC) {
        throw std::runtime_error("unexpected product/output nnz");
    }
    PrintSummary(options, static_cast<int64_t>(host.colsA.size()), stage, timings);
    return 0;
}

}  // namespace

int main(int argc, char **argv)
{
    try {
        return Benchmark(ParseOptions(argc, argv));
    } catch (const std::exception &error) {
        std::cerr << "spgemm_perf: " << error.what() << std::endl;
        return 1;
    }
}
