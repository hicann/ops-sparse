/**
 * ----------------------------------------------------------------------------------------------------------
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software: you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ----------------------------------------------------------------------------------------------------------
 */

/**
 * @file spmm_op_perf.cpp
 * @brief Standalone performance collector for aclsparseSpMMOp (Ascend 950 / arch35).
 *
 * Reuses the golden sparsity generator (spmm_op_golden.h) and the SpMMOp
 * descriptor/plan lifecycle, but pre-allocates all device buffers / descriptors
 * / plan once and times ONLY the Execute stage (aclsparseSpMMOp) with
 * aclrtEvent across warmup + measured iterations. This isolates the kernel
 * execution time from host-side setup / H2D copy overhead.
 *
 * Per representative shape/dtype/alg/sparsity it reports:
 *   - kernel time (min / median / mean, us)
 *   - FLOPs (2*nnz*n dot-product MADs) and achieved GFLOPS
 *   - algorithmic HBM bytes moved and achieved bandwidth (GB/s)
 *   - bandwidth utilization vs a configurable peak HBM BW
 *
 * This program does NOT modify the operator (sparse/) — it only links against
 * the pre-built libops_sparse.so. Built manually (not via the gtest CMake target).
 */

#include "test_common.h"
#include "spmm_op_golden.h"
#include "spmm_op_npu_wrapper.h"  // SpmmOpDescrGuard / SpmmOpPlanGuard / ParseSpmmOpAlg
#include "spmm_op_param.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <iomanip>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <vector>

using namespace sparse_test;

// ============================================================================
// Host data conversion helpers: DoublesToFp32 / DoublesToFp16 are provided by
// spmm_op_golden.h (shared with spmm_op_test.cpp).
// ============================================================================

// ============================================================================
// Performance case descriptor
// Shape notation: M×N×K (standard SpMM convention).
//   A is M×K (sparse CSR), B is K×N (dense), C is M×N (dense output).
// In CSV (m,k,n) order used by the wrapper: m=M, k=K, n=N.
// ============================================================================

struct PerfCase {
    std::string label;
    int64_t m;              // A rows / C rows
    int64_t k;              // A cols / reduction dim
    int64_t n;              // C cols / B cols
    double sparsity_ratio;
    double alpha;
    double beta;
    aclDataType dtype;          // ACL_FLOAT / ACL_FLOAT16
    std::string dtype_name;
    aclsparseSpMMOpAlg_t alg;
    std::string alg_name;
    double value_lo;
    double value_hi;
    uint32_t seed;
};

// ============================================================================
// Timed SpMMOp lifecycle.
// Setup (once): allocate device buffers + descriptors, run BufferSize +
// createDescr + createPlan. The createDescr stage binds matA (CSR pattern)
// and performs ALG2 preprocessing (device-side sort + bin_edge). createPlan
// finalizes the execution plan. Execute only launches the kernel, so timing
// Execute in a loop gives a clean kernel-time measurement.
// PerfWorkspace aggregates non-copyable RAII handles (HandleManager /
// DnMatManager / SpMatManager / DeviceBuffer / SpmmOpDescrGuard /
// SpmmOpPlanGuard). HandleManager has no move ctor, so PerfWorkspace is
// non-movable — therefore it is heap-allocated via unique_ptr and returned by
// pointer (no copy/move needed).
// T: float (FP32) or uint16_t (FP16 bit patterns).
// ============================================================================
template <typename T>
struct PerfWorkspace {
    HandleManager handle;
    DeviceBuffer dRowOff;
    DeviceBuffer dColInd;
    DeviceBuffer dVals;
    DeviceBuffer dB;
    DeviceBuffer dC;
    DeviceBuffer dBuffer;
    DeviceBuffer dAlpha;
    DeviceBuffer dBeta;
    DnMatManager matB;
    DnMatManager matC;
    SpMatManager matA;
    aclDataType dtype;
    aclDataType computeType;
    aclsparseSpMMOpAlg_t alg;
    aclsparseOperation_t opA;
    aclsparseOperation_t opB;
    aclsparseOrder_t orderB;
    aclsparseOrder_t orderC;
    aclsparsePointerMode_t pointerMode;
    float alphaHost;
    float betaHost;
    const void* alphaPtr;
    const void* betaPtr;
    size_t bufferSize;
    int64_t nnz;
    int64_t m, k, n;
    bool ok;
    std::string err;
    // descr/plan managed via guard (non-copyable, non-movable)
    std::unique_ptr<SpmmOpDescrGuard> descrGuard;
    std::unique_ptr<SpmmOpPlanGuard> planGuard;
    aclsparseSpMMOpPlan_t plan;
};

template <typename T>
static void SetupPerfCsrBuffers(PerfWorkspace<T> *ws, const SpmmCsr &csrA,
    const std::vector<T>& hAValues)
{
    // --- Device buffers for CSR data (matA) --------------------------------
    // rowOffsets always present (m+1 ints).
    size_t roSize = static_cast<size_t>(ws->m + 1) * sizeof(int32_t);
    ws->dRowOff = DeviceBuffer::copyFrom(csrA.rowOffsets.data(), roSize);
    if (csrA.nnz > 0) {
        ws->dColInd = DeviceBuffer::copyFrom(
            csrA.colIndices.data(), static_cast<size_t>(csrA.nnz) * sizeof(int32_t));
        ws->dVals = DeviceBuffer::copyFrom(
            hAValues.data(), static_cast<size_t>(csrA.nnz) * sizeof(T));
    }

    // --- Create matA (CSR, const) ------------------------------------------
    ws->matA = SpMatManager::createConstCsr(
        ws->m, ws->k, ws->nnz, ws->dRowOff.get(), ws->dColInd.get(), ws->dVals.get(),
        ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_32I,
        ACL_SPARSE_INDEX_BASE_ZERO, ws->dtype);
}

template <typename T>
static void SetupPerfDnMatBuffers(PerfWorkspace<T> *ws,
    const std::vector<T>& hB,
    const std::vector<T>& hCInit)
{
    // --- Device buffers for B (k×n row-major) and C (m×n row-major) --------
    size_t bElemCount = static_cast<size_t>(ws->k) * static_cast<size_t>(ws->n);
    size_t cElemCount = static_cast<size_t>(ws->m) * static_cast<size_t>(ws->n);
    ws->dB = (bElemCount > 0)
        ? DeviceBuffer::copyFrom(hB.data(), bElemCount * sizeof(T))
        : DeviceBuffer::alloc(sizeof(T));
    ws->dC = (cElemCount > 0)
        ? DeviceBuffer::copyFrom(hCInit.data(), cElemCount * sizeof(T))
        : DeviceBuffer::alloc(sizeof(T));

    // --- Create matB (dense, const) and matC (dense, non-const) ------------
    // opB=NON_TRANSPOSE: B is k×n row-major, ld=n
    int64_t bLd = ws->n;
    int64_t cLd = ws->n;
    ws->matB = DnMatManager::createConst(ws->k, ws->n, bLd, ws->dB.raw(), ws->dtype, ws->orderB);
    ws->matC = DnMatManager::create(ws->m, ws->n, cLd, ws->dC.raw(), ws->dtype, ws->orderC);
}

template <typename T>
static void SetupPerfDescriptors(PerfWorkspace<T> *ws, aclrtStream stream, const PerfCase& pc,
    const SpmmCsr &csrA,
    const std::vector<T>& hAValues,
    const std::vector<T>& hB,
    const std::vector<T>& hCInit)
{
    ws->dtype = pc.dtype;
    ws->computeType = ACL_FLOAT;  // always FP32 accumulation
    ws->alg = pc.alg;
    ws->opA = ACL_SPARSE_OP_NON_TRANSPOSE;
    ws->opB = ACL_SPARSE_OP_NON_TRANSPOSE;
    ws->orderB = ACL_SPARSE_ORDER_ROW;
    ws->orderC = ACL_SPARSE_ORDER_ROW;
    ws->pointerMode = ACL_SPARSE_POINTER_MODE_DEVICE;
    ws->alphaHost = static_cast<float>(pc.alpha);
    ws->betaHost = static_cast<float>(pc.beta);
    ws->m = pc.m;
    ws->k = pc.k;
    ws->n = pc.n;
    ws->nnz = csrA.nnz;
    ws->plan = nullptr;
    ws->handle.setStream(stream);

    SetupPerfCsrBuffers<T>(ws, csrA, hAValues);
    SetupPerfDnMatBuffers<T>(ws, hB, hCInit);

    // --- alpha/beta (DEVICE pointer mode) ----------------------------------
    ws->dAlpha = DeviceBuffer::copyFrom(&ws->alphaHost, sizeof(float));
    ws->dBeta = DeviceBuffer::copyFrom(&ws->betaHost, sizeof(float));
    ws->alphaPtr = ws->dAlpha.get();
    ws->betaPtr = ws->dBeta.get();
    auto st = aclsparseSetPointerMode(ws->handle.get(), ws->pointerMode);
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        ws->err = "SetPointerMode failed: " + std::to_string(st);
    }
}

template <typename T>
static bool SetupPerfPlan(PerfWorkspace<T> *ws, aclrtStream stream)
{
    // --- Stage 1: bufferSize -------------------------------------------------
    size_t bs = 0;
    auto r1 = aclsparseSpMMOp_bufferSize(
        ws->handle.get(), ws->opA, ws->opB,
        ws->matA.cget(), ws->matB.cget(), ws->matC.get(),
        ws->computeType, ws->alg, &bs);
    if (r1 != ACL_SPARSE_STATUS_SUCCESS) {
        ws->err = "bufferSize failed: " + std::to_string(r1);
        return false;
    }
    ws->bufferSize = bs;
    if (bs > 0) {
        ws->dBuffer = DeviceBuffer::alloc(bs);
    }

    // --- Stage 2: createDescr (binds matA, runs ALG2 preprocess) ------------
    aclsparseSpMMOpDescr_t descr = nullptr;
    auto r2 = aclsparseSpMMOp_createDescr(
        ws->handle.get(), &descr,
        ws->opA, ws->opB,
        ws->matA.cget(), ws->matB.cget(), ws->matC.get(),
        ws->computeType, ws->alg, ws->dBuffer.get());
    if (r2 != ACL_SPARSE_STATUS_SUCCESS) {
        ws->err = "createDescr failed: " + std::to_string(r2);
        return false;
    }
    ws->descrGuard = std::make_unique<SpmmOpDescrGuard>(descr);

    // --- Stage 3: createPlan ------------------------------------------------
    auto r3 = aclsparseSpMMOp_createPlan(
        ws->handle.get(), descr, &ws->plan, nullptr, 0);
    if (r3 != ACL_SPARSE_STATUS_SUCCESS) {
        ws->err = "createPlan failed: " + std::to_string(r3);
        return false;
    }
    ws->planGuard = std::make_unique<SpmmOpPlanGuard>(ws->plan);

    // sync after plan creation to ensure workspace is ready before timing
    aclrtSynchronizeStream(stream);
    return true;
}

template <typename T>
static std::unique_ptr<PerfWorkspace<T>> SetupPerf(aclrtStream stream, const PerfCase& pc,
    const SpmmCsr &csrA,
    const std::vector<T>& hAValues,
    const std::vector<T>& hB,
    const std::vector<T>& hCInit)
{
    auto ws = std::make_unique<PerfWorkspace<T>>();
    ws->ok = false;
    SetupPerfDescriptors<T>(ws.get(), stream, pc, csrA, hAValues, hB, hCInit);
    if (!ws->err.empty()) {
        return ws;
    }
    ws->ok = SetupPerfPlan<T>(ws.get(), stream);
    return ws;
}

// Time Execute (kernel) over warmup + iters with aclrtEvent. Returns times (us).
// exec is called once per iteration (must launch the kernel asynchronously).
template <typename ExecFn>
static std::vector<double> TimeExecute(aclrtStream stream, ExecFn exec, int warmup, int iters)
{
    std::vector<double> times;
    times.reserve(static_cast<size_t>(iters));

    aclrtEvent startEv = nullptr;
    aclrtEvent stopEv = nullptr;
    if (aclrtCreateEvent(&startEv) != ACL_SUCCESS || aclrtCreateEvent(&stopEv) != ACL_SUCCESS) {
        if (startEv != nullptr) {
            aclrtDestroyEvent(startEv);
        }
        if (stopEv != nullptr) {
            aclrtDestroyEvent(stopEv);
        }
        return times;
    }

    // Warmup
    for (int i = 0; i < warmup; ++i) {
        exec();
    }
    aclrtSynchronizeStream(stream);

    // Measured iterations: one event pair per iteration
    for (int i = 0; i < iters; ++i) {
        aclrtRecordEvent(startEv, stream);
        exec();
        aclrtRecordEvent(stopEv, stream);
        aclrtSynchronizeStream(stream);
        float ms = 0.0f;
        aclrtEventElapsedTime(&ms, startEv, stopEv);
        times.push_back(static_cast<double>(ms) * 1000.0);  // us
    }

    aclrtDestroyEvent(startEv);
    aclrtDestroyEvent(stopEv);
    return times;
}

// ============================================================================
// Algorithmic metrics
// ============================================================================
// SpMM: C[i,j] = alpha * sum_p A[i,p] * B[p,j] + beta * C[i,j]
// FLOPs: each nonzero A[i,p] contributes to n dot-product accumulations
//        (one per j column). Dominant term: 2 * nnz * n.
//        Plus alpha-scale + beta-fma per output element: 2 * m * n.
// HBM bytes (algorithmic lower bound the kernel must move):
//   A values read : nnz * elem
//   A colInd read : nnz * 4
//   A rowOff read : (m + 1) * 4
//   B read        : k * n * elem        (dense, each element read >= 1x)
//   C read        : beta != 0 ? m * n * elem : 0
//   C write       : m * n * elem
// elem = sizeof(T): FP32=4, FP16=2.

struct AlgoMetrics {
    int64_t nnz;
    double flops;
    double bytes;
};

static AlgoMetrics ComputeAlgoMetrics(const PerfCase& pc, int64_t nnz)
{
    AlgoMetrics m;
    m.nnz = nnz;
    double elem = (pc.dtype == ACL_FLOAT) ? 4.0 : 2.0;

    // FLOPs: 2*nnz*n (dominant) + 2*m*n (alpha-scale + beta-fma per output)
    double flops = 2.0 * static_cast<double>(nnz) * static_cast<double>(pc.n);
    flops += 2.0 * static_cast<double>(pc.m) * static_cast<double>(pc.n);
    m.flops = flops;

    double bytes = 0.0;
    bytes += static_cast<double>(nnz) * elem;                  // A values
    bytes += static_cast<double>(nnz) * 4.0;                   // A colInd
    bytes += static_cast<double>(pc.m + 1) * 4.0;              // A rowOff
    bytes += static_cast<double>(pc.k) * static_cast<double>(pc.n) * elem;  // B read
    if (pc.beta != 0.0) {
        bytes += static_cast<double>(pc.m) * static_cast<double>(pc.n) * elem;  // C read
    }
    bytes += static_cast<double>(pc.m) * static_cast<double>(pc.n) * elem;      // C write
    m.bytes = bytes;
    return m;
}

// ============================================================================
// HBM bandwidth benchmark (aclrtMemcpy D2D, DMA engine).
// Measures the practical peak HBM bandwidth achievable on this device using
// device-to-device memcpy over a large buffer. A D2D copy reads `size` bytes
// from HBM and writes `size` bytes to HBM, so total HBM traffic = 2*size per
// iteration. For large transfers the DMA engine (MTE2/MTE3) approaches the
// device peak HBM bandwidth, giving a defensible practical-peak denominator
// for utilization computation (msprof aic-metrics is unavailable for custom
// op loaded via shared library on this CANN build).
// ============================================================================

static void PrintBwBenchResult(double tMin, double tMed, size_t sizeMiB, size_t size)
{
    double bytesTraffic = 2.0 * static_cast<double>(size);
    double bwMin = (tMin > 0.0) ? bytesTraffic / (tMin / 1000.0) / 1e9 : 0.0;
    double bwMed = (tMed > 0.0) ? bytesTraffic / (tMed / 1000.0) / 1e9 : 0.0;
    std::cout << "=== HBM Bandwidth Benchmark (aclrtMemcpy D2D, " << sizeMiB
              << " MiB) ===\n";
    std::cout << "  copy time: min=" << std::fixed << std::setprecision(3) << tMin
              << " ms  median=" << tMed << " ms\n";
    std::cout << "  HBM BW (2*size/time): min=" << bwMin << " GB/s  median=" << bwMed
              << " GB/s\n";
    std::cout << "  -> use median " << bwMed << " GB/s as practical peak HBM BW\n";
    std::cout << std::defaultfloat;
}

static bool PrepareBwBenchResources(
    aclrtStream stream, size_t size,
    void *&dSrc, void *&dDst,
    aclrtEvent &startEv, aclrtEvent &stopEv)
{
    auto r1 = aclrtMalloc(&dSrc, size, ACL_MEM_MALLOC_HUGE_FIRST);
    auto r2 = aclrtMalloc(&dDst, size, ACL_MEM_MALLOC_HUGE_FIRST);
    if (r1 != ACL_SUCCESS || r2 != ACL_SUCCESS) {
        std::cerr << "bwbench: aclrtMalloc failed r1=" << r1 << " r2=" << r2 << std::endl;
        if (r1 == ACL_SUCCESS) { aclrtFree(dSrc); }
        if (r2 == ACL_SUCCESS) { aclrtFree(dDst); }
        return false;
    }
    aclrtMemset(dSrc, size, 0, size);
    aclrtMemset(dDst, size, 0, size);
    aclrtSynchronizeStream(stream);

    if (aclrtCreateEvent(&startEv) != ACL_SUCCESS ||
        aclrtCreateEvent(&stopEv) != ACL_SUCCESS) {
        if (startEv != nullptr) { aclrtDestroyEvent(startEv); }
        if (stopEv != nullptr) { aclrtDestroyEvent(stopEv); }
        aclrtFree(dSrc);
        aclrtFree(dDst);
        return false;
    }
    return true;
}

static int RunBwBench(aclrtStream stream, int warmup, int iters)
{
    // 1 GiB buffer — large enough to saturate HBM, fits in 123 GiB HBM.
    const size_t sizeMiB = 1024;
    const size_t size = sizeMiB * 1024ULL * 1024ULL;

    void *dSrc = nullptr;
    void *dDst = nullptr;
    aclrtEvent startEv = nullptr;
    aclrtEvent stopEv = nullptr;
    if (!PrepareBwBenchResources(stream, size, dSrc, dDst, startEv, stopEv)) {
        return 1;
    }

    for (int i = 0; i < warmup; ++i) {
        aclrtMemcpy(dDst, size, dSrc, size, ACL_MEMCPY_DEVICE_TO_DEVICE);
    }
    aclrtSynchronizeStream(stream);

    std::vector<double> times;
    for (int i = 0; i < iters; ++i) {
        aclrtRecordEvent(startEv, stream);
        aclrtMemcpy(dDst, size, dSrc, size, ACL_MEMCPY_DEVICE_TO_DEVICE);
        aclrtRecordEvent(stopEv, stream);
        aclrtSynchronizeStream(stream);
        float ms = 0.0f;
        aclrtEventElapsedTime(&ms, startEv, stopEv);
        times.push_back(static_cast<double>(ms));
    }
    std::sort(times.begin(), times.end());
    double tMin = times.front();
    double tMed = times[times.size() / 2];
    PrintBwBenchResult(tMin, tMed, sizeMiB, size);

    aclrtDestroyEvent(startEv);
    aclrtDestroyEvent(stopEv);
    aclrtFree(dSrc);
    aclrtFree(dDst);
    return 0;
}

// ============================================================================
// Main
// ============================================================================

struct RowOut {
    std::string label;
    std::string dtype_name;
    std::string alg_name;
    int64_t nnz;
    double tMin, tMed, tMean;
    double bwGbs, gflops, bwUtil;
    std::string note;
};

static std::string Fmt(double v, int width)
{
    if (v <= 0) {
        return std::string("-");
    }
    std::ostringstream os;
    os << std::fixed << std::setprecision(2) << v;
    std::string s = os.str();
    if (static_cast<int>(s.length()) > width) {
        s = s.substr(0, width);
    }
    return s;
}

struct PerfArgs {
    int warmup = 5;
    int iters = 20;
    double peakHbmGbps = 0.0;  // 0 = unknown
    std::string only;          // if non-empty, run only cases whose label contains this substring
    bool bwbench = false;      // if true, run HBM bandwidth benchmark and exit
};

static PerfArgs ParsePerfArgs(int argc, char** argv)
{
    PerfArgs args;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto pos = a.find('=');
        if (pos == std::string::npos) {
            if (a == "--bwbench") { args.bwbench = true; }
            continue;
        }
        std::string key = a.substr(0, pos);
        std::string val = a.substr(pos + 1);
        if (key == "--warmup") {
            args.warmup = std::stoi(val);
        } else if (key == "--iters") {
            args.iters = std::stoi(val);
        } else if (key == "--peak") {
            args.peakHbmGbps = std::stod(val);
        } else if (key == "--only") {
            args.only = val;
        }
    }
    return args;
}

template <typename T>
static void RunPerfTyped(aclrtStream stream, const PerfCase& pc, const SpmmCsr &csrA,
    const std::vector<T>& hAValues, const std::vector<T>& hB,
    const std::vector<T>& hCInit, int warmup, int iters, RowOut& row)
{
    auto ws = SetupPerf<T>(stream, pc, csrA, hAValues, hB, hCInit);
    if (!ws->ok) {
        row.note = "SETUP_FAIL:" + ws->err;
        return;
    }
    auto exec = [&]() {
        return aclsparseSpMMOp(ws->handle.get(), ws->plan,
                               ws->alphaPtr, ws->betaPtr,
                               ws->matB.cget(), ws->matC.get());
    };
    auto times = TimeExecute(stream, exec, warmup, iters);
    std::sort(times.begin(), times.end());
    row.tMin = times.front();
    row.tMed = times[times.size() / 2];
    double sum = 0;
    for (double t : times) sum += t;
    row.tMean = sum / times.size();
    row.note = "ok";
}

static RowOut RunOneCase(aclrtStream stream, const PerfCase& pc, int warmup, int iters,
    double peakHbmGbps)
{
    RowOut row;
    row.label = pc.label;
    row.dtype_name = pc.dtype_name;
    row.alg_name = pc.alg_name;
    row.tMin = row.tMed = row.tMean = 0;
    row.bwGbs = row.gflops = 0;
    row.bwUtil = -1;
    row.note = "";

    SpmmCsr csrA = MakeSpmmSparsity(pc.m, pc.k, pc.sparsity_ratio,
        pc.value_lo, pc.value_hi, pc.seed,
        false, 0);
    int64_t nnz = csrA.nnz;
    row.nnz = nnz;

    // Generate dense B (k×n) and C_init (m×n) in FP64
    std::mt19937 rngB(pc.seed + 1);
    std::mt19937 rngC(pc.seed + 2);
    std::uniform_real_distribution<double> dist(pc.value_lo, pc.value_hi);
    std::vector<double> Bf64(static_cast<size_t>(pc.k) * static_cast<size_t>(pc.n));
    std::vector<double> Cf64(static_cast<size_t>(pc.m) * static_cast<size_t>(pc.n));
    for (size_t i = 0; i < Bf64.size(); i++) Bf64[i] = dist(rngB);
    for (size_t i = 0; i < Cf64.size(); i++) Cf64[i] = dist(rngC);

    if (pc.dtype == ACL_FLOAT) {
        RunPerfTyped<float>(stream, pc, csrA, DoublesToFp32(csrA.values),
            DoublesToFp32(Bf64), DoublesToFp32(Cf64),
            warmup, iters, row);
    } else {
        RunPerfTyped<uint16_t>(stream, pc, csrA, DoublesToFp16(csrA.values),
            DoublesToFp16(Bf64), DoublesToFp16(Cf64),
            warmup, iters, row);
    }

    auto am = ComputeAlgoMetrics(pc, nnz);
    double tUs = row.tMed;
    if (tUs > 0) {
        double tSec = tUs / 1e6;
        row.bwGbs = am.bytes / tSec / 1e9;
        row.gflops = am.flops / tSec / 1e9;
        row.bwUtil = (peakHbmGbps > 0) ? (row.bwGbs / peakHbmGbps * 100.0) : -1.0;
    }
    return row;
}

static bool InitAclEnv(aclrtStream &stream)
{
    if (aclInit(nullptr) != ACL_SUCCESS) {
        std::cerr << "aclInit failed" << std::endl;
        return false;
    }
    if (aclrtSetDevice(0) != ACL_SUCCESS) {
        std::cerr << "aclrtSetDevice failed" << std::endl;
        return false;
    }
    aclrtCreateStream(&stream);
    return true;
}

static void CleanupAclEnv(aclrtStream stream)
{
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();
}

static void PrintHeader(int64_t arch, size_t totalMem, size_t freeMem,
    int warmup, int iters, double peakHbmGbps)
{
    std::cout << "============================================================\n";
    std::cout << "SpMMOp Performance Collection (aclsparseSpMMOp, Execute stage)\n";
    std::cout << "Device: Ascend arch=" << arch << " (DAV_3510 / Ascend950)\n";
    std::cout << "HBM total=" << (totalMem >> 20) << " MiB  free=" << (freeMem >> 20) << " MiB\n";
    std::cout << "warmup=" << warmup << " iters=" << iters
              << " peakHbmGbps=" << (peakHbmGbps > 0 ? std::to_string(peakHbmGbps) : std::string("N/A"))
              << "\n";
    std::cout << "Timing: aclrtEvent around aclsparseSpMMOp (Execute), per-iter sync\n";
    std::cout << "Shape convention: MxNxK (A: MxK sparse, B: KxN dense, C: MxN dense)\n";
    std::cout << "============================================================\n";

    printf("+----------------------------+--------+--------+----------+--------+--------+--------+"
           "----------+----------+--------+\n");
    printf("| %-26s | %-6s | %-6s | %8s | %6s | %6s | %6s | %8s | %8s | %6s |\n",
           "case", "dtype", "alg", "nnz", "min(us)", "med(us)", "mean(u)", "BW(GB/s)", "GFLOPS", "BWutil");
    printf("+----------------------------+--------+--------+----------+--------+--------+--------+"
           "----------+----------+--------+\n");
}

static void RunAllCases(aclrtStream stream, const std::vector<PerfCase>& cases,
    const std::string& only, int warmup, int iters, double peakHbmGbps,
    std::vector<RowOut>& rows, std::vector<PerfCase>& filtered_cases)
{
    for (const auto& pc : cases) {
        if (!only.empty() && pc.label.find(only) == std::string::npos) {
            continue;
        }
        RowOut row = RunOneCase(stream, pc, warmup, iters, peakHbmGbps);
        int64_t nnz = row.nnz;

        rows.push_back(row);
        filtered_cases.push_back(pc);

        std::string bwutilStr = (row.bwUtil < 0) ? "N/A" : Fmt(row.bwUtil, 6);
        printf("| %-26s | %-6s | %-6s | %8lld | %6s | %6s | %6s | %8s | %8s | %6s |\n",
               row.label.c_str(), row.dtype_name.c_str(), row.alg_name.c_str(),
               static_cast<long long>(nnz),
               Fmt(row.tMin, 6).c_str(), Fmt(row.tMed, 6).c_str(), Fmt(row.tMean, 6).c_str(),
               Fmt(row.bwGbs, 8).c_str(), Fmt(row.gflops, 8).c_str(),
               bwutilStr.c_str());
        fflush(stdout);
    }
    printf("+----------------------------+--------+--------+----------+--------+--------+--------+"
           "----------+----------+--------+\n");
}

static void PrintDetailedBreakdown(const std::vector<PerfCase>& filtered_cases,
    const std::vector<RowOut>& rows)
{
    // Detailed per-case breakdown
    std::cout << "\n=== Detailed breakdown (median time) ===\n";
    for (size_t i = 0; i < filtered_cases.size(); ++i) {
        const auto& pc = filtered_cases[i];
        const auto& row = rows[i];
        auto am = ComputeAlgoMetrics(pc, row.nnz);
        std::cout << "[" << row.label << "]\n";
        std::cout << "  m=" << pc.m << " k=" << pc.k << " n=" << pc.n
                  << " dtype=" << pc.dtype_name
                  << " alg=" << pc.alg_name
                  << " alpha=" << pc.alpha << " beta=" << pc.beta
                  << " ratio=" << pc.sparsity_ratio << "\n";
        std::cout << "  nnz=" << row.nnz
                  << " (density=" << std::fixed << std::setprecision(4)
                  << (pc.k > 0 ? static_cast<double>(row.nnz) / static_cast<double>(pc.m * pc.k) : 0.0)
                  << ")\n";
        std::cout << std::defaultfloat;
        std::cout << "  FLOPs=" << static_cast<uint64_t>(am.flops)
                  << "  HBM_bytes=" << static_cast<uint64_t>(am.bytes) << "\n";
        std::cout << "  time: min=" << std::fixed << std::setprecision(2) << row.tMin
                  << "us  median=" << row.tMed << "us  mean=" << row.tMean << "us\n";
        std::cout << "  achieved BW=" << row.bwGbs << " GB/s  GFLOPS=" << row.gflops;
        if (row.bwUtil >= 0) {
            std::cout << "  BWutil=" << row.bwUtil << "%";
        }
        std::cout << "  [" << row.note << "]\n\n";
        std::cout << std::defaultfloat;
    }
}

static void PrintJsonSummary(const std::vector<PerfCase>& filtered_cases,
    const std::vector<RowOut>& rows,
    double peakHbmGbps, int warmup, int iters)
{
    // Machine-readable JSON summary
    std::cout << "=== JSON ===\n";
    std::cout << "{\"device\":\"Ascend950_dav3510\",\"peak_hbm_gbps\":" << peakHbmGbps
              << ",\"warmup\":" << warmup << ",\"iters\":" << iters << ",\"cases\":[";
    for (size_t i = 0; i < filtered_cases.size(); ++i) {
        const auto& pc = filtered_cases[i];
        const auto& row = rows[i];
        auto am = ComputeAlgoMetrics(pc, row.nnz);
        if (i) {
            std::cout << ",";
        }
        std::cout << "{\"label\":\"" << row.label << "\""
                  << ",\"m\":" << pc.m << ",\"k\":" << pc.k << ",\"n\":" << pc.n
                  << ",\"dtype\":\"" << pc.dtype_name << "\""
                  << ",\"alg\":\"" << pc.alg_name << "\""
                  << ",\"nnz\":" << row.nnz
                  << ",\"alpha\":" << pc.alpha << ",\"beta\":" << pc.beta
                  << ",\"ratio\":" << pc.sparsity_ratio
                  << ",\"flops\":" << static_cast<uint64_t>(am.flops)
                  << ",\"hbm_bytes\":" << static_cast<uint64_t>(am.bytes)
                  << ",\"t_min_us\":" << row.tMin
                  << ",\"t_med_us\":" << row.tMed
                  << ",\"t_mean_us\":" << row.tMean
                  << ",\"bw_gbs\":" << row.bwGbs
                  << ",\"gflops\":" << row.gflops
                  << ",\"bw_util_pct\":" << row.bwUtil
                  << ",\"status\":\"" << row.note << "\"}";
    }
    std::cout << "]}\n";
}

// ============================================================================
// Build the full performance case matrix:
//   4 shapes × 2 dtypes × 2 ALGs × 3 sparsities = 48 cases
// + 4 shapes × 1 dtype(FP32) × 1 ALG(HIGH_PRECISION) × 3 sparsities = 12 cases
//   (HIGH_PRECISION 仅 FP32 有意义；FP16 静默忽略 Kahan，与普通 ALG1 等价，跳过)
// = 60 cases total
// Shape convention: M×N×K (standard SpMM notation)
//   256×256×64, 512×512×128, 1024×1024×64, 2048×2048×128
//   In CSV (m,k,n) order: m=M, k=K, n=N
// ============================================================================
static std::vector<PerfCase> BuildPerfCases()
{
    // Shapes: {M, N, K} in standard SpMM notation
    struct Shape { int64_t m; int64_t k; int64_t n; const char* tag; };
    const Shape shapes[] = {
        {256,  64, 256,  "256x256x64"},
        {512, 128, 512,  "512x512x128"},
        {1024, 64, 1024, "1024x1024x64"},
        {2048, 128, 2048, "2048x2048x128"},
    };

    const double sparsities[] = {0.10, 0.30, 0.50};
    const char* sp_tags[] = {"sp10", "sp30", "sp50"};

    struct DTypeCfg { aclDataType dt; const char* name; double lo; double hi; };
    const DTypeCfg dtypes[] = {
        {ACL_FLOAT,   "FP32", -1.0, 1.0},
        {ACL_FLOAT16, "FP16", -0.5, 0.5},
    };

    struct AlgCfg { aclsparseSpMMOpAlg_t alg; const char* name; };
    const AlgCfg algs[] = {
        {ACL_SPARSE_SPMMOP_ALG1, "ALG1"},
        {ACL_SPARSE_SPMMOP_ALG2, "ALG2"},
        {ACL_SPARSE_SPMMOP_ALG1_HIGH_PRECISION, "HIGH_PREC"},
    };

    std::vector<PerfCase> cases;
    uint32_t seed = 1000;
    for (const auto& sh : shapes) {
        for (const auto& dt : dtypes) {
            for (const auto& al : algs) {
                // HIGH_PRECISION 仅对 FP32 有意义，FP16 静默忽略（与普通 ALG1 等价）
                if (al.alg == ACL_SPARSE_SPMMOP_ALG1_HIGH_PRECISION && dt.dt == ACL_FLOAT16) {
                    continue;
                }
                for (size_t s = 0; s < sizeof(sparsities)/sizeof(sparsities[0]); ++s) {
                    std::ostringstream lbl;
                    lbl << sh.tag << "_" << dt.name << "_" << al.name << "_" << sp_tags[s];
                    cases.push_back({
                        lbl.str(),
                        sh.m, sh.k, sh.n,
                        sparsities[s],
                        1.0, 0.0,    // alpha=1, beta=0 (standard SpMM)
                        dt.dt, dt.name,
                        al.alg, al.name,
                        dt.lo, dt.hi,
                        seed++
                    });
                }
            }
        }
    }
    return cases;
}

int main(int argc, char** argv)
{
    PerfArgs args = ParsePerfArgs(argc, argv);
    int warmup = args.warmup;
    int iters = args.iters;
    double peakHbmGbps = args.peakHbmGbps;
    const std::string& only = args.only;
    bool bwbench = args.bwbench;

    std::vector<PerfCase> cases = BuildPerfCases();

    aclrtStream stream = nullptr;
    if (!InitAclEnv(stream)) {
        return 1;
    }

    if (bwbench) {
        int rc = RunBwBench(stream, warmup, iters);
        CleanupAclEnv(stream);
        return rc;
    }

    int64_t arch = 0;
    aclrtGetDeviceInfo(0, ACL_DEV_ATTR_NPU_ARCH, &arch);
    size_t totalMem = 0;
    size_t freeMem = 0;
    aclrtGetMemInfo(ACL_HBM_MEM, &freeMem, &totalMem);

    PrintHeader(arch, totalMem, freeMem, warmup, iters, peakHbmGbps);

    std::vector<RowOut> rows;
    std::vector<PerfCase> filtered_cases;
    RunAllCases(stream, cases, only, warmup, iters, peakHbmGbps, rows, filtered_cases);

    PrintDetailedBreakdown(filtered_cases, rows);
    PrintJsonSummary(filtered_cases, rows, peakHbmGbps, warmup, iters);

    CleanupAclEnv(stream);
    return 0;
}
