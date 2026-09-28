/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under
 * the terms and conditions of CANN Open Software License Agreement Version 2.0
 * (the "License"). Please refer to the License for details. You may not use
 * this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
 * AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
 * FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
 * for the full text of the License.
 */

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include "descriptor_manager.h"
#include "sparse_test.h"

using sparse_test::DeviceBuffer;
using sparse_test::DnMatManager;
using sparse_test::HandleManager;
using sparse_test::SpMatManager;
using sparse_test::SpSMDescrManager;

namespace
{

struct Options
{
    int32_t m = 4096;
    int32_t rhs = 32;
    int32_t nnzPerRow = 8;
    int32_t warmup = 5;
    int32_t samples = 30;
    int32_t device = 0;
    std::string format = "csr";
    std::string dtype = "fp32";
};

struct Triplet
{
    int32_t row;
    int32_t column;
    float real;
    float imag;
};

struct SparseHostData
{
    std::vector<int32_t> primary;
    std::vector<int32_t> secondary;
    std::vector<float> values;
    std::vector<float> updatedValues;
    int64_t nnz = 0;
};

struct Statistics
{
    double minimum;
    double median;
    double mean;
    double p90;
};

struct BenchmarkContext
{
    aclrtStream stream;
    HandleManager& handle;
    SpMatManager& matA;
    DnMatManager& matB;
    DnMatManager& matC;
    SpSMDescrManager& plan;
    DeviceBuffer& updatedValues;
    const void* alpha;
    aclDataType type;
};

struct BenchmarkStatistics
{
    size_t workspaceBytes;
    Statistics analysis;
    Statistics update;
    Statistics solve;
};

static void PrintUsage(const char* program)
{
    std::cout << "Usage: " << program
              << " [--m rows] [--rhs columns] [--nnz-per-row count]"
                 " [--format csr|csc|coo] [--dtype fp32|complex64]"
                 " [--warmup count] [--samples count] [--device id]\n";
}

static void ParseOption(const std::string& argument, int argc, char** argv, int& index, Options& options)
{
    if (index + 1 >= argc)
    {
        throw std::runtime_error("missing value for " + argument);
    }
    const std::string value = argv[++index];
    if (argument == "--m")
    {
        options.m = std::stoi(value);
    }
    else if (argument == "--rhs")
    {
        options.rhs = std::stoi(value);
    }
    else if (argument == "--nnz-per-row")
    {
        options.nnzPerRow = std::stoi(value);
    }
    else if (argument == "--warmup")
    {
        options.warmup = std::stoi(value);
    }
    else if (argument == "--samples")
    {
        options.samples = std::stoi(value);
    }
    else if (argument == "--device")
    {
        options.device = std::stoi(value);
    }
    else if (argument == "--format")
    {
        options.format = value;
    }
    else if (argument == "--dtype")
    {
        options.dtype = value;
    }
    else
    {
        throw std::runtime_error("unsupported argument: " + argument);
    }
}

static Options ParseOptions(int argc, char** argv)
{
    Options options;
    for (int index = 1; index < argc; ++index)
    {
        const std::string argument = argv[index];
        if (argument == "--help" || argument == "-h")
        {
            PrintUsage(argv[0]);
            std::exit(0);
        }
        ParseOption(argument, argc, argv, index, options);
    }
    if (options.m <= 0 || options.rhs <= 0 || options.nnzPerRow <= 0 || options.nnzPerRow > options.m
        || options.warmup < 0 || options.samples <= 0 || options.device < 0)
    {
        throw std::runtime_error("invalid matrix dimensions, device, warmup, or sample count");
    }
    if (options.format != "csr" && options.format != "csc" && options.format != "coo")
    {
        throw std::runtime_error("format must be csr, csc, or coo");
    }
    if (options.dtype != "fp32" && options.dtype != "complex64")
    {
        throw std::runtime_error("dtype must be fp32 or complex64");
    }
    return options;
}

static aclDataType DataType(const Options& options)
{
    return options.dtype == "fp32" ? ACL_FLOAT : ACL_COMPLEX64;
}

static size_t Components(const Options& options)
{
    return options.dtype == "fp32" ? 1U : 2U;
}

static std::vector<Triplet> BuildTriplets(const Options& options)
{
    const int64_t maximumNnz = static_cast<int64_t>(options.m) * options.nnzPerRow;
    if (maximumNnz > std::numeric_limits<int32_t>::max())
    {
        throw std::runtime_error("nnz exceeds the int32 index range");
    }
    std::vector<Triplet> triplets;
    triplets.reserve(static_cast<size_t>(maximumNnz));
    for (int32_t row = 0; row < options.m; ++row)
    {
        const int32_t firstColumn = std::max(0, row - options.nnzPerRow + 1);
        for (int32_t column = firstColumn; column <= row; ++column)
        {
            const bool diagonal = column == row;
            const float real = diagonal ? 2.0F : 0.001F * static_cast<float>(1 + (row + column) % 7);
            const float imag = options.dtype == "complex64" ? (diagonal ? 0.01F : 0.0001F) : 0.0F;
            triplets.push_back({ row, column, real, imag });
        }
    }
    return triplets;
}

static void AppendValue(
    const Triplet& entry, size_t components, std::vector<float>& values, std::vector<float>& updatedValues)
{
    values.push_back(entry.real);
    updatedValues.push_back(entry.real * 1.01F);
    if (components == 2U)
    {
        values.push_back(entry.imag);
        updatedValues.push_back(entry.imag * 1.01F);
    }
}

static SparseHostData BuildSparseData(const Options& options)
{
    auto triplets = BuildTriplets(options);
    const size_t components = Components(options);
    SparseHostData data;
    data.nnz = static_cast<int64_t>(triplets.size());
    data.secondary.resize(triplets.size());
    data.values.reserve(triplets.size() * components);
    data.updatedValues.reserve(triplets.size() * components);

    if (options.format == "coo")
    {
        data.primary.resize(triplets.size());
        for (size_t index = 0; index < triplets.size(); ++index)
        {
            data.primary[index] = triplets[index].row;
            data.secondary[index] = triplets[index].column;
            AppendValue(triplets[index], components, data.values, data.updatedValues);
        }
        return data;
    }

    const bool csc = options.format == "csc";
    std::sort(triplets.begin(), triplets.end(),
        [csc](const Triplet& left, const Triplet& right)
        {
            const int32_t leftPrimary = csc ? left.column : left.row;
            const int32_t rightPrimary = csc ? right.column : right.row;
            if (leftPrimary != rightPrimary)
            {
                return leftPrimary < rightPrimary;
            }
            return (csc ? left.row : left.column) < (csc ? right.row : right.column);
        });
    data.primary.assign(static_cast<size_t>(options.m) + 1U, 0);
    for (const auto& entry : triplets)
    {
        const int32_t primaryIndex = csc ? entry.column : entry.row;
        ++data.primary[static_cast<size_t>(primaryIndex) + 1U];
    }
    std::partial_sum(data.primary.begin(), data.primary.end(), data.primary.begin());
    for (size_t index = 0; index < triplets.size(); ++index)
    {
        data.secondary[index] = csc ? triplets[index].row : triplets[index].column;
        AppendValue(triplets[index], components, data.values, data.updatedValues);
    }
    return data;
}

static std::vector<float> BuildDenseInput(const Options& options)
{
    const size_t components = Components(options);
    const size_t elements = static_cast<size_t>(options.m) * static_cast<size_t>(options.rhs);
    std::vector<float> values(elements * components);
    for (size_t index = 0; index < elements; ++index)
    {
        values[index * components] = 1.0F + 0.001F * static_cast<float>(index % 17U);
        if (components == 2U)
        {
            values[index * components + 1U] = 0.0001F * static_cast<float>(index % 11U);
        }
    }
    return values;
}

static void CheckSparse(aclsparseStatus_t status, const char* stage)
{
    if (status != ACL_SPARSE_STATUS_SUCCESS)
    {
        throw std::runtime_error(std::string(stage) + " failed: " + std::to_string(static_cast<int>(status)));
    }
}

static void CheckAcl(aclError status, const char* stage)
{
    if (status != ACL_SUCCESS)
    {
        throw std::runtime_error(std::string(stage) + " failed: " + std::to_string(static_cast<int>(status)));
    }
}

class EventPair
{
public:
    EventPair()
    {
        CheckAcl(aclrtCreateEvent(&start_), "aclrtCreateEvent(start)");
        try
        {
            CheckAcl(aclrtCreateEvent(&stop_), "aclrtCreateEvent(stop)");
        }
        catch (...)
        {
            aclrtDestroyEvent(start_);
            start_ = nullptr;
            throw;
        }
    }

    ~EventPair()
    {
        if (start_ != nullptr)
        {
            aclrtDestroyEvent(start_);
        }
        if (stop_ != nullptr)
        {
            aclrtDestroyEvent(stop_);
        }
    }

    EventPair(const EventPair&) = delete;
    EventPair& operator=(const EventPair&) = delete;

    template <typename Function> double Measure(aclrtStream stream, Function function, const char* stage)
    {
        CheckAcl(aclrtRecordEvent(start_, stream), "aclrtRecordEvent(start)");
        CheckSparse(function(), stage);
        CheckAcl(aclrtRecordEvent(stop_, stream), "aclrtRecordEvent(stop)");
        CheckAcl(aclrtSynchronizeEvent(stop_), "aclrtSynchronizeEvent");
        float elapsedMs = 0.0F;
        CheckAcl(aclrtEventElapsedTime(&elapsedMs, start_, stop_), "aclrtEventElapsedTime");
        return static_cast<double>(elapsedMs) * 1000.0;
    }

private:
    aclrtEvent start_ = nullptr;
    aclrtEvent stop_ = nullptr;
};

template <typename Function>
static std::vector<double> MeasureStage(
    aclrtStream stream, int32_t warmup, int32_t samples, Function function, const char* stage)
{
    for (int32_t iteration = 0; iteration < warmup; ++iteration)
    {
        CheckSparse(function(), stage);
    }
    CheckAcl(aclrtSynchronizeStream(stream), "warmup synchronization");
    EventPair events;
    std::vector<double> measurements;
    measurements.reserve(static_cast<size_t>(samples));
    for (int32_t iteration = 0; iteration < samples; ++iteration)
    {
        measurements.push_back(events.Measure(stream, function, stage));
    }
    return measurements;
}

static Statistics Summarize(std::vector<double> values)
{
    if (values.empty())
    {
        throw std::runtime_error("cannot summarize an empty measurement set");
    }
    const double sum = std::accumulate(values.begin(), values.end(), 0.0);
    std::sort(values.begin(), values.end());
    const size_t middle = values.size() / 2U;
    const double median = values.size() % 2U == 0U ? (values[middle - 1U] + values[middle]) * 0.5 : values[middle];
    const size_t p90Index = (9U * values.size() + 9U) / 10U - 1U;
    return { values.front(), median, sum / static_cast<double>(values.size()), values[p90Index] };
}

static SpMatManager CreateSparseDescriptor(const Options& options, const SparseHostData& host, DeviceBuffer& primary,
    DeviceBuffer& secondary, DeviceBuffer& values)
{
    const aclDataType type = DataType(options);
    if (options.format == "csr")
    {
        return SpMatManager::createCsr(options.m, options.m, host.nnz, primary.get(), secondary.get(), values.get(),
            ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_BASE_ZERO, type);
    }
    if (options.format == "csc")
    {
        return SpMatManager::createCsc(options.m, options.m, host.nnz, primary.get(), secondary.get(), values.get(),
            ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_BASE_ZERO, type);
    }
    return SpMatManager::createCoo(options.m, options.m, host.nnz, primary.get(), secondary.get(), values.get(),
        ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_BASE_ZERO, type);
}

static void PrintStatistics(const char* name, const Statistics& statistics)
{
    std::cout << "\"" << name << "_min_us\":" << statistics.minimum << ","
              << "\"" << name << "_median_us\":" << statistics.median << ","
              << "\"" << name << "_mean_us\":" << statistics.mean << ","
              << "\"" << name << "_p90_us\":" << statistics.p90;
}

static BenchmarkStatistics MeasureOperations(const Options& options, const BenchmarkContext& context)
{
    size_t workspaceBytes = 0;
    CheckSparse(
        aclsparseSpSMBufferSize(context.handle.get(), ACL_SPARSE_OP_NON_TRANSPOSE, ACL_SPARSE_OP_NON_TRANSPOSE,
            context.alpha, context.matA.cget(), context.matB.cget(), context.matC.get(), context.type,
            ACL_SPARSE_SPSM_ALG_DEFAULT, context.plan.get(), &workspaceBytes),
        "aclsparseSpSMBufferSize");
    DeviceBuffer workspace = DeviceBuffer::alloc(std::max(workspaceBytes, static_cast<size_t>(16U)));

    auto analysis = [&]()
    {
        return aclsparseSpSMAnalysis(context.handle.get(), ACL_SPARSE_OP_NON_TRANSPOSE, ACL_SPARSE_OP_NON_TRANSPOSE,
            context.alpha, context.matA.cget(), context.matB.cget(), context.matC.get(), context.type,
            ACL_SPARSE_SPSM_ALG_DEFAULT, context.plan.get(), workspace.get());
    };
    auto update = [&]()
    {
        return aclsparseSpSMUpdateMatrix(context.handle.get(), context.plan.get(), context.updatedValues.get(),
            ACL_SPARSE_SPSM_UPDATE_GENERAL);
    };
    auto solve = [&]()
    {
        return aclsparseSpSM(context.handle.get(), ACL_SPARSE_OP_NON_TRANSPOSE, ACL_SPARSE_OP_NON_TRANSPOSE,
            context.alpha, context.matA.cget(), context.matB.cget(), context.matC.get(), context.type,
            ACL_SPARSE_SPSM_ALG_DEFAULT, context.plan.get());
    };

    CheckSparse(analysis(), "initial aclsparseSpSMAnalysis");
    return { workspaceBytes,
        Summarize(MeasureStage(context.stream, options.warmup, options.samples, analysis, "Analysis")),
        Summarize(MeasureStage(context.stream, options.warmup, options.samples, update, "Update")),
        Summarize(MeasureStage(context.stream, options.warmup, options.samples, solve, "Solve")) };
}

static void PrintResult(const Options& options, const SparseHostData& host, size_t valueBytes, size_t denseBytes,
    const BenchmarkStatistics& statistics)
{
    const size_t sparseBytes
        = host.primary.size() * sizeof(int32_t) + host.secondary.size() * sizeof(int32_t) + valueBytes;
    const size_t inputOutputBytes = sparseBytes + denseBytes * 2U;
    std::cout << std::fixed << std::setprecision(3) << "{";
    std::cout << "\"m\":" << options.m << ",\"rhs\":" << options.rhs << ",\"nnz\":" << host.nnz << ",";
    std::cout << "\"format\":\"" << options.format << "\",\"dtype\":\"" << options.dtype << "\",";
    std::cout << "\"warmup\":" << options.warmup << ",\"samples\":" << options.samples << ",";
    std::cout << "\"workspace_bytes\":" << statistics.workspaceBytes
              << ",\"input_output_bytes\":" << inputOutputBytes << ",";
    PrintStatistics("analysis", statistics.analysis);
    std::cout << ",";
    PrintStatistics("update", statistics.update);
    std::cout << ",";
    PrintStatistics("solve", statistics.solve);
    std::cout << "}" << std::endl;
}

static int Run(const Options& options)
{
    sparse_test::AclEnvScope environment(options.device);
    HandleManager handle;
    handle.setStream(environment.stream());

    const SparseHostData host = BuildSparseData(options);
    const std::vector<float> denseInput = BuildDenseInput(options);
    const size_t valueBytes = host.values.size() * sizeof(float);
    const size_t denseBytes = denseInput.size() * sizeof(float);
    DeviceBuffer primary = DeviceBuffer::copyFrom(host.primary.data(), host.primary.size() * sizeof(int32_t));
    DeviceBuffer secondary = DeviceBuffer::copyFrom(host.secondary.data(), host.secondary.size() * sizeof(int32_t));
    DeviceBuffer values = DeviceBuffer::copyFrom(host.values.data(), valueBytes);
    DeviceBuffer updatedValues = DeviceBuffer::copyFrom(host.updatedValues.data(), valueBytes);
    DeviceBuffer bValues = DeviceBuffer::copyFrom(denseInput.data(), denseBytes);
    DeviceBuffer cValues = DeviceBuffer::alloc(denseBytes);

    SpMatManager matA = CreateSparseDescriptor(options, host, primary, secondary, values);
    const aclsparseFillMode_t fillMode = ACL_SPARSE_FILL_MODE_LOWER;
    const aclsparseDiagType_t diagonalType = ACL_SPARSE_DIAG_TYPE_NON_UNIT;
    matA.setAttribute(ACL_SPARSE_SPMAT_FILL_MODE, &fillMode, sizeof(fillMode));
    matA.setAttribute(ACL_SPARSE_SPMAT_DIAG_TYPE, &diagonalType, sizeof(diagonalType));
    const aclDataType type = DataType(options);
    DnMatManager matB
        = DnMatManager::createConst(options.m, options.rhs, options.rhs, bValues.get(), type, ACL_SPARSE_ORDER_ROW);
    DnMatManager matC
        = DnMatManager::create(options.m, options.rhs, options.rhs, cValues.get(), type, ACL_SPARSE_ORDER_ROW);
    SpSMDescrManager plan;
    const float alphaFp32 = 1.0F;
    const aclsparseComplex alphaComplex { 1.0F, 0.0F };
    const void* alpha
        = options.dtype == "fp32" ? static_cast<const void*>(&alphaFp32) : static_cast<const void*>(&alphaComplex);
    BenchmarkContext context { environment.stream(), handle, matA, matB, matC, plan, updatedValues, alpha, type };
    PrintResult(options, host, valueBytes, denseBytes, MeasureOperations(options, context));
    return 0;
}

} // namespace

int main(int argc, char** argv)
{
    try
    {
        return Run(ParseOptions(argc, argv));
    }
    catch (const std::exception& exception)
    {
        std::cerr << "spsm_perf: " << exception.what() << std::endl;
        PrintUsage(argv[0]);
        return 1;
    }
}
