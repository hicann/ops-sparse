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
#include "spmma_compress_npu_wrapper.h"
#include <chrono>
#include <fstream>
#include <cmath>
#include <numeric>
using namespace sparse_test;
static double Quantile(std::vector<double> v, double q)
{
    if (v.empty() || !std::isfinite(q) || q <= 0 || q > 1) {
        throw std::invalid_argument("invalid quantile input");
    }
    std::sort(v.begin(), v.end());
    if (q == 0.5 && v.size() % 2 == 0)
        return (v[v.size() / 2 - 1] + v[v.size() / 2]) / 2;
    return v[static_cast<size_t>(std::ceil(q * v.size())) - 1];
}
static void WriteSummary(const CompressParam& p, int warmup, const std::vector<double>& eventTimes,
    const std::vector<double>& hostTimes, std::ofstream& summary)
{
    const size_t count = eventTimes.size();
    if (count == 0 || hostTimes.size() != count) {
        throw std::invalid_argument("invalid performance samples");
    }
    const auto& dtype = p.dtype;
    const bool alongRow = p.alongRow();
    const double mean = std::accumulate(eventTimes.begin(), eventTimes.end(), 0.0) / count;
    if (!std::isfinite(mean) || mean <= 0) {
        throw std::invalid_argument("invalid mean event duration");
    }
    double variance = 0;
    for (double t : eventTimes)
        variance += (t - mean) * (t - mean);
    summary << p.name << ',' << dtype << ',' << (alongRow ? "row" : "col") << ',' << p.rows << ',' << p.cols << ','
            << warmup << ',' << count << ',' << Quantile(eventTimes, .5) << ',' << Quantile(eventTimes, .95) << ','
            << *std::min_element(eventTimes.begin(), eventTimes.end()) << ','
            << *std::max_element(eventTimes.begin(), eventTimes.end()) << ',' << std::sqrt(variance / count) / mean
            << ',' << Quantile(hostTimes, .5) << ',' << Quantile(hostTimes, .95) << ','
            << p.elements() * p.bytes() + p.compressedBytes() << ',' << p.pattern << '\n';
}

static void RunCase(const CompressParam& p, AclEnvScope& env, std::ofstream& samples, std::ofstream& summary)
{
    const int warmup = 5;
    const int count = 30;
    const auto& dtype = p.dtype;
    const bool alongRow = p.alongRow();
    CompressPlan plan(p);
    const auto input = MakeCompressInput(p);
    const auto expected = CompressGolden(p, input);
    CompressBuffers buffers(p, input);
    auto launch = [&]() {
        CompressCheck(aclsparseLtSpMMACompress(&plan.handle.value, &plan.plan.value, buffers.dense(),
            buffers.compressed(), nullptr, env.stream()),
            "Perf Compress");
    };
    for (int i = 0; i < warmup; ++i)
        launch();
    CompressAclCheck(aclrtSynchronizeStream(env.stream()), "Warmup sync");
    CheckCompressResult(p, buffers, expected);
    std::vector<double> eventTimes;
    std::vector<double> hostTimes;
    CompressEvent start;
    CompressEvent end;
    for (int i = 0; i < count; ++i) {
        CompressAclCheck(aclrtRecordEvent(start.value, env.stream()), "Start record");
        const auto before = std::chrono::steady_clock::now();
        launch();
        const auto after = std::chrono::steady_clock::now();
        CompressAclCheck(aclrtRecordEvent(end.value, env.stream()), "End record");
        CompressAclCheck(aclrtSynchronizeStream(env.stream()), "Sample sync");
        float ms = 0;
        CompressAclCheck(aclrtEventElapsedTime(&ms, start.value, end.value), "Event elapsed");
        if (ms <= 0 || !std::isfinite(ms))
            throw std::runtime_error("invalid event duration");
        const double host = std::chrono::duration<double, std::micro>(after - before).count();
        eventTimes.push_back(ms * 1000.0);
        hostTimes.push_back(host);
        samples << p.name << ',' << dtype << ',' << (alongRow ? "row" : "col") << ',' << p.rows << ',' << p.cols << ','
                << i << ',' << ms * 1000 << ',' << host << '\n';
    }
    CheckCompressResult(p, buffers, expected);
    WriteSummary(p, warmup, eventTimes, hostTimes, summary);
    samples.flush();
    summary.flush();
    if (!samples || !summary)
        throw std::runtime_error("cannot write result CSV");
}
static CompressParam MakePerfCase(const std::string& dtype, bool alongRow, bool large, bool tail)
{
    const int alignment = dtype == "FP32" ? 8 : dtype == "INT8" ? 32 : 16;
    CompressParam param;
    param.dtype = dtype;
    param.op = "N";
    param.order = alongRow ? "ROW" : "COL";
    param.side = alongRow ? "A" : "B";
    const int64_t physicalRows = large ? 1024 : 4 * alignment;
    const int64_t physicalCols = (large ? 4096 : 256 / param.bytes()) + (tail ? alignment : 0);
    param.rows = alongRow ? physicalRows : physicalCols;
    param.cols = alongRow ? physicalCols : physicalRows;
    param.ld = physicalCols;
    param.name =
        dtype + std::string(alongRow ? "_row" : "_col") + (large ? "_large" : "_small") + (tail ? "_tail" : "_full");
    return param;
}

int main(int argc, char** argv)
{
    if (argc != 2) {
        std::cerr << "usage: spmma_compress_perf output_prefix\n";
        return 2;
    }
    try {
        AclEnvScope env;
        const std::string prefix = argv[1];
        std::ofstream samples(prefix + "-samples.csv");
        std::ofstream summary(prefix + "-summary.csv");
        if (!samples || !summary)
            throw std::runtime_error("cannot create result CSV");
        samples << "case,dtype,direction,rows,cols,iteration,event_us,host_enqueue_us\n";
        summary << "case,dtype,direction,rows,cols,warmup,samples,event_median_us,event_p95_us,event_min_us,event_max_"
                   "us,event_cv,host_median_us,host_p95_us,bytes_moved,pattern\n";
        for (const std::string dtype : {"FP32", "FP16", "BF16", "INT8"}) {
            for (bool alongRow : {true, false})
                for (bool large : {false, true})
                    for (bool tail : {false, true}) {
                        RunCase(MakePerfCase(dtype, alongRow, large, tail), env, samples, summary);
                    }
        }
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
    return 0;
}
