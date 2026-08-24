/**
 * ----------------------------------------------------------------------------------------------------------
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ----------------------------------------------------------------------------------------------------------
 */

/**
 * @file matmul_perf_test.cpp
 * @brief Performance baseline benchmark for aclsparseLtMatmul (v2: 4 dtype × 2 path).
 *
 * Reuses the existing npu_wrapper (RunMatmulNpu) which already records kernel
 * execution time via EventGuard (aclrtEvent). This benchmark:
 *   - Runs each configuration with WARMUP iterations (excluded from stats)
 *   - Then MEASURE iterations, reporting mean / min / max / median (ms)
 *   - Outputs a parseable line per config: PERF|tag|dtype|path|m|k|n|mean_ms|min_ms|max_ms|median_ms
 *
 * Demand source: §2-B.7 "no explicit perf requirement; baseline data collection only".
 * No pass/fail threshold; this is reference data for future optimization.
 *
 * Build: standalone executable (own main), linked against the same libs as matmul_test.
 */

#include "test_common.h"

#include "ltmatmul_golden.h"
#include "ltmatmul_npu_wrapper.h"
#include "ltmatmul_param.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <random>
#include <string>
#include <vector>

using namespace sparse_test;

namespace {

// ============================================================================
// Iteration counts: warmup excluded, then measured.
// Small kernels => more iterations for stable timing.
// ============================================================================
constexpr int kWarmupIters = 5;
constexpr int kMeasureIters = 20;

// ============================================================================
// Stats helpers
// ============================================================================
struct Stats {
    double mean = 0.0;
    double min = 0.0;
    double max = 0.0;
    double median = 0.0;
};

inline Stats ComputeStats(std::vector<double>& samples) {
    Stats s{};
    if (samples.empty()) { return s; }
    std::sort(samples.begin(), samples.end());
    const size_t n = samples.size();
    s.min = samples.front();
    s.max = samples.back();
    double sum = 0.0;
    for (double v : samples) { sum += v; }
    s.mean = sum / static_cast<double>(n);
    s.median = (n % 2 == 0)
        ? (samples[n / 2 - 1] + samples[n / 2]) / 2.0
        : samples[n / 2];
    return s;
}

// ============================================================================
// Dispatch: single dtype×path run, returns npuMs (kernel time only).
// Mirrors matmul_test.cpp RunOneCase but WITHOUT golden/verify (perf-only).
// ============================================================================
template <typename InT, typename OutT>
inline double RunOnceTiming(int32_t m, int32_t k, int32_t n,
                             float alpha, float beta,
                             int32_t alg_config_id, int32_t split_k, int32_t split_k_mode,
                             bool isSparseA, bool isDensePath,
                             aclsparseOrder_t order,
                             aclsparseOperation_t opA, aclsparseOperation_t opB,
                             aclsparseLtPruneAlg_t pruneAlg,
                             aclrtStream /*stream*/)
{
    const uint32_t seed = 7777u;
    auto hA = MatmulGenTestMatrix<InT>(m, k, -1.0f, 1.0f, seed);
    auto hB = MatmulGenTestMatrix<InT>(k, n, -1.0f, 1.0f, seed + 1);
    auto hC = MatmulGenTestMatrix<InT>(m, n, -1.0f, 1.0f, seed + 2);

    std::vector<OutT> hD;
    auto npu = RunMatmulNpu<InT, OutT>(m, k, n, hA, hB, hC, hD,
                                        alpha, beta, alg_config_id, split_k, split_k_mode,
                                        isSparseA, isDensePath, order, opA, opB, pruneAlg);
    if (npu.matmulRet != ACL_SPARSE_STATUS_SUCCESS) {
        std::fprintf(stderr, "[PERF] matmul failed ret=%d (m=%d k=%d n=%d)\n",
                     static_cast<int>(npu.matmulRet), m, k, n);
        return -1.0;
    }
    return npu.npuMs;
}

// ============================================================================
// Config descriptor
// ============================================================================
struct PerfConfig {
    std::string tag;
    std::string dtype;     // FP32 / FP16 / BF16 / INT8_I32 / INT8_I8
    std::string path;      // sparse×dense / dense×dense
    int32_t m;
    int32_t k;
    int32_t n;
    int32_t split_k = 1;
    int32_t split_k_mode = 0;  // 0=fused(ONE_KERNEL), 1=TWO_KERNELS
    // dispatch key
    int dtype_key = 0;  // 0=FP32 1=FP16 2=BF16 3=INT8_I32 4=INT8_I8
    bool dense = false;
};

inline void PrintHeader() {
    std::printf("# aclsparseLtMatmul performance baseline\n");
    std::printf("# warmup=%d measure=%d (kernel time via aclrtEvent, ms)\n",
                kWarmupIters, kMeasureIters);
    std::printf("# columns: tag|dtype|path|m|k|n|splitK|mode|mean_ms|min_ms|max_ms|median_ms\n");
}

inline void PrintResult(const PerfConfig& c, const Stats& s) {
    std::printf("PERF|%s|%s|%s|%d|%d|%d|%d|%d|%.4f|%.4f|%.4f|%.4f\n",
                c.tag.c_str(), c.dtype.c_str(), c.path.c_str(),
                c.m, c.k, c.n, c.split_k, c.split_k_mode,
                s.mean, s.min, s.max, s.median);
    std::fflush(stdout);
}

// ============================================================================
// Dispatch by dtype_key: single run, returns npuMs or -1.0 on failure.
// Extracted to avoid duplicated switch in warmup/measure loops
// [codecheck: cyclomatic complexity].
// ============================================================================
inline double RunOnceByDtypeKey(int dtype_key, int32_t m, int32_t k, int32_t n,
    float alpha, float beta, int32_t algCfgId, int32_t split_k, int32_t split_k_mode,
    bool isSparseA, bool isDensePath, aclsparseOrder_t order,
    aclsparseOperation_t opA, aclsparseOperation_t opB,
    aclsparseLtPruneAlg_t pruneAlg, aclrtStream stream)
{
    switch (dtype_key) {
        case 0:
            return RunOnceTiming<float, float>(m, k, n, alpha, beta,
                algCfgId, split_k, split_k_mode, isSparseA, isDensePath,
                order, opA, opB, pruneAlg, stream);
        case 1:
            return RunOnceTiming<uint16_t, uint16_t>(m, k, n, alpha, beta,
                algCfgId, split_k, split_k_mode, isSparseA, isDensePath,
                order, opA, opB, pruneAlg, stream);
        case 2:
            return RunOnceTiming<bf16_bits_t, bf16_bits_t>(m, k, n, alpha, beta,
                algCfgId, split_k, split_k_mode, isSparseA, isDensePath,
                order, opA, opB, pruneAlg, stream);
        case 3:
            return RunOnceTiming<int8_t, int32_t>(m, k, n, alpha, beta,
                algCfgId, split_k, split_k_mode, isSparseA, isDensePath,
                order, opA, opB, pruneAlg, stream);
        case 4:
            return RunOnceTiming<int8_t, int8_t>(m, k, n, alpha, beta,
                algCfgId, split_k, split_k_mode, isSparseA, isDensePath,
                order, opA, opB, pruneAlg, stream);
        default:
            std::fprintf(stderr, "[PERF] unknown dtype_key=%d\n", dtype_key);
            return -1.0;
    }
}

// ============================================================================
// Run a config: warmup + measure
// ============================================================================
inline bool RunConfig(const PerfConfig& c, aclrtStream stream) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    const int32_t algCfgId = 0;
    const auto order = ACL_SPARSE_ORDER_ROW;
    const auto opA = ACL_SPARSE_OP_NON_TRANSPOSE;
    const auto opB = ACL_SPARSE_OP_NON_TRANSPOSE;
    const auto pruneAlg = ACLSPARSELT_PRUNE_SPMMA_STRIP;
    const bool isSparseA = !c.dense;  // sparse×dense => A-sparse
    const bool isDensePath = c.dense;

    std::vector<double> samples;
    samples.reserve(kMeasureIters);

    // Warmup
    for (int i = 0; i < kWarmupIters; ++i) {
        double ms = RunOnceByDtypeKey(c.dtype_key, c.m, c.k, c.n, alpha, beta,
            algCfgId, c.split_k, c.split_k_mode, isSparseA, isDensePath,
            order, opA, opB, pruneAlg, stream);
        if (ms < 0) { return false; }
    }

    // Measure
    for (int i = 0; i < kMeasureIters; ++i) {
        double ms = RunOnceByDtypeKey(c.dtype_key, c.m, c.k, c.n, alpha, beta,
            algCfgId, c.split_k, c.split_k_mode, isSparseA, isDensePath,
            order, opA, opB, pruneAlg, stream);
        if (ms < 0) { return false; }
        samples.push_back(ms);
    }

    Stats s = ComputeStats(samples);
    PrintResult(c, s);
    return true;
}

// ============================================================================
// Config list: covers task §1 (dtype × shape) + §2 (splitK path compare)
// ============================================================================
inline std::vector<PerfConfig> BuildConfigList() {
    std::vector<PerfConfig> v;
    auto add = [&](std::string tag, std::string dtype, std::string path,
                   int32_t m, int32_t k, int32_t n, int dk, bool dense,
                   int32_t splitK = 1, int32_t mode = 0) {
        v.push_back({std::move(tag), std::move(dtype), std::move(path),
                     m, k, n, splitK, mode, dk, dense});
    };

    // ---- §1 sparse×dense (A-sparse, STRIP, fused splitK=1) ----
    // shapes: 128×128×128, 256×256×256
    add("S-FP32-128",  "FP32",     "sparse×dense", 128, 128, 128, 0, false);
    add("S-FP32-256",  "FP32",     "sparse×dense", 256, 256, 256, 0, false);
    add("S-FP16-128",  "FP16",     "sparse×dense", 128, 128, 128, 1, false);
    add("S-FP16-256",  "FP16",     "sparse×dense", 256, 256, 256, 1, false);
    add("S-BF16-128",  "BF16",     "sparse×dense", 128, 128, 128, 2, false);
    add("S-BF16-256",  "BF16",     "sparse×dense", 256, 256, 256, 2, false);
    add("S-I8I32-128", "INT8_I32", "sparse×dense", 128, 128, 128, 3, false);
    add("S-I8I32-256", "INT8_I32", "sparse×dense", 256, 256, 256, 3, false);
    add("S-I8I8-128",  "INT8_I8",  "sparse×dense", 128, 128, 128, 4, false);
    add("S-I8I8-256",  "INT8_I8",  "sparse×dense", 256, 256, 256, 4, false);

    // ---- §1 dense×dense (fused splitK=1) ----
    add("D-FP32-128",  "FP32",     "dense×dense",  128, 128, 128, 0, true);
    add("D-FP32-256",  "FP32",     "dense×dense",  256, 256, 256, 0, true);
    add("D-FP16-128",  "FP16",     "dense×dense",  128, 128, 128, 1, true);
    add("D-FP16-256",  "FP16",     "dense×dense",  256, 256, 256, 1, true);
    add("D-BF16-128",  "BF16",     "dense×dense",  128, 128, 128, 2, true);
    add("D-BF16-256",  "BF16",     "dense×dense",  256, 256, 256, 2, true);
    add("D-I8I32-128", "INT8_I32", "dense×dense",  128, 128, 128, 3, true);
    add("D-I8I32-256", "INT8_I32", "dense×dense",  256, 256, 256, 3, true);
    add("D-I8I8-128",  "INT8_I8",  "dense×dense",  128, 128, 128, 4, true);
    add("D-I8I8-256",  "INT8_I8",  "dense×dense",  256, 256, 256, 4, true);

    // ---- §2 splitK path compare (same shape 128×128×128) ----
    // fused = splitK=1 mode=0 (ONE_KERNEL); TWO_KERNELS = splitK=2 mode=1
    add("K-FP16-fused",  "FP16", "sparse×dense", 128, 128, 128, 1, false, 1, 0);
    add("K-FP16-split2", "FP16", "sparse×dense", 128, 128, 128, 1, false, 2, 1);
    add("K-FP32-fused",  "FP32", "sparse×dense", 128, 128, 128, 0, false, 1, 0);
    add("K-FP32-split2", "FP32", "sparse×dense", 128, 128, 128, 0, false, 2, 1);

    return v;
}

}  // namespace

// ============================================================================
// Standalone main (separate executable, no GTest)
// ============================================================================
int main(int argc, char** argv) {
    // Optional: select a single tag via argv[1] for debugging
    std::string onlyTag = (argc > 1) ? std::string(argv[1]) : "";

    AclEnvScope env;
    aclrtStream stream = env.stream();

    PrintHeader();
    auto configs = BuildConfigList();
    int okCount = 0, failCount = 0;
    for (const auto& c : configs) {
        if (!onlyTag.empty() && c.tag != onlyTag) { continue; }
        std::printf("# running %s ...\n", c.tag.c_str());
        std::fflush(stdout);
        if (RunConfig(c, stream)) {
            ++okCount;
        } else {
            ++failCount;
            std::printf("PERF|%s|%s|%s|%d|%d|%d|%d|%d|FAILED|0|0|0\n",
                        c.tag.c_str(), c.dtype.c_str(), c.path.c_str(),
                        c.m, c.k, c.n, c.split_k, c.split_k_mode);
        }
    }
    std::printf("# done: ok=%d fail=%d\n", okCount, failCount);
    return (failCount == 0) ? 0 : 1;
}
