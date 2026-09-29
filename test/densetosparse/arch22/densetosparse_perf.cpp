// DenseToSparse arch22 C++ caliber harness: Blocked-ELL P-cases.
//
// Purpose (task-book 3.3 + 性能口径说明 §4/§7.4): the cuSPARSE C++ baseline
// must be joined against the NPU C++ three-stage window. This harness times
// the native GetBufferSize->Analysis->Convert flow with device events
// (median of 30 after 10 warmup, descriptors/workspace reused) and compares
// against the OFFICIAL per-case medians from the task package
// (test_cases/baseline_results/gpu_full_results.tsv, cuSPARSE segment).
//
// The baseline table below carries the official per-case median_us values
// (scene x dtype x base) verbatim -- NOT per-dtype aggregates. base0/base1
// differ by a few us in the official data and are kept separate.
//
// csr/csc/coo C++-window numbers are informational only (D2S_INFO_ALL=1):
// they MUST NOT be joined against the PyTorch-GPU segment (task-book 3.3
// forbids mixing calibers; PyTorch caliber uses the official runner).
//
// Build (after `bash build.sh --ops=densetosparse`):
//   g++ -O2 -std=c++17 test/densetosparse/arch22/densetosparse_perf.cpp \
//       -Iinclude -I$ASCEND_HOME_PATH/include \
//       -Lbuild_out/lib64 -L$ASCEND_HOME_PATH/lib64 -lops_sparse -lascendcl \
//       -Wl,-rpath,$PWD/build_out/lib64 -Wl,-rpath,$ASCEND_HOME_PATH/lib64 \
//       -o build/densetosparse_perf
// Run (BELL verdict caliber, 3 scenes x 5 dtypes x base{0,1} = 30 cases):
//   ./build/densetosparse_perf blocked_ell
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "acl/acl.h"
#include "cann_ops_sparse.h"

#define CK(x) do { auto s=(x); if (s) { std::printf(#x " -> %d\n",(int)s); return 1; } } while(0)

namespace {

struct PerfCase {
    const char *scene;
    int rows, cols;
    int nnzPerRow;
    int bellB, ellCols;
    aclDataType dtype;
    aclsparseFormat_t format;
    aclsparseIndexBase_t base;
    aclsparseOrder_t order;
};

int ElemBytes(aclDataType t) {
    if (t == ACL_INT8) return 1;
    if (t == ACL_FLOAT16 || t == ACL_BF16) return 2;
    if (t == ACL_FLOAT) return 4;
    return 8;
}

const char *DtName(aclDataType t) {
    if (t == ACL_INT8) return "int8";
    if (t == ACL_FLOAT16) return "float16";
    if (t == ACL_BF16) return "bfloat16";
    if (t == ACL_FLOAT) return "float32";
    return "complex64";
}

const char *FmtName(aclsparseFormat_t f) {
    switch (f) {
    case ACL_SPARSE_FORMAT_CSR: return "csr";
    case ACL_SPARSE_FORMAT_CSC: return "csc";
    case ACL_SPARSE_FORMAT_COO: return "coo";
    default: return "blocked_ell";
    }
}

// Official per-case median_us (gpu_full_results.tsv, cuSPARSE segment),
// [scene 0..2 = P-01/02/03][dtype 0..4 = int8/fp16/bf16/fp32/c64]
// [base 0..1]. Verbatim transcription; keep in sync with the task package.
const double kBellOfficial[3][5][2] = {
    {{38.720, 34.688}, {39.232, 34.816}, {39.136, 34.880},
     {38.816, 35.168}, {39.808, 35.552}},
    {{35.744, 31.872}, {36.448, 31.712}, {36.256, 31.424},
     {36.384, 31.488}, {37.312, 32.384}},
    {{38.240, 34.656}, {38.560, 34.816}, {38.688, 34.432},
     {38.432, 34.816}, {39.616, 35.200}},
};

int DtypeIndex(aclDataType t) {
    return t == ACL_INT8 ? 0 : t == ACL_FLOAT16 ? 1 : t == ACL_BF16 ? 2
         : t == ACL_FLOAT ? 3 : 4;
}

// Dense data with nnzPerRow nonzeros per row (deterministic).
std::vector<uint8_t> MakeDense(const PerfCase &c, int ld, size_t pitch) {
    std::mt19937 rng(20260912);
    const int E = ElemBytes(c.dtype);
    std::vector<uint8_t> host(pitch * ld * E, 0);
    std::uniform_int_distribution<int> colRng(0, c.cols - 1);
    for (int r = 0; r < c.rows; ++r) {
        for (int k = 0; k < c.nnzPerRow; ++k) {
            const int col = colRng(rng);
            uint8_t *p = &host[static_cast<size_t>(
                c.order == ACL_SPARSE_ORDER_ROW ? r : col) * ld *
                E + static_cast<size_t>(
                c.order == ACL_SPARSE_ORDER_ROW ? col : r) * E];
            for (int b = 0; b < E; ++b) {
                p[b] = static_cast<uint8_t>(0x3F + ((r + k + b) & 0x3F));
            }
        }
    }
    return host;
}

int gFail = 0;

// Helper-form error check: prints the failing expression and unwinds the
// setup helper (RunCase still returns 1).
#define CK_B(x) do { auto s=(x); if (s) { std::printf(#x " -> %d\n",(int)s); return false; } } while(0)

// Blocked-ELL pattern + descriptor for one perf case.
bool SetupBellPerf(const PerfCase &c, int E, aclsparseSpMatDescr_t *B,
    void **dPat, void **dBell)
{
    const int blockRows = c.rows / c.bellB;
    const int slots = c.ellCols / c.bellB;
    const int colBlocks = c.cols / c.bellB;
    std::vector<int32_t> pat(static_cast<size_t>(blockRows) * slots);
    std::mt19937 prng(7);
    for (int br = 0; br < blockRows; ++br) {
        for (int s = 0; s < slots; ++s) {
            pat[static_cast<size_t>(br) * slots + s] =
                (br + s * 3 + static_cast<int>(prng() % 2)) % colBlocks +
                static_cast<int32_t>(c.base);
        }
    }
    CK_B(aclrtMalloc(dPat, pat.size() * 4, ACL_MEM_MALLOC_HUGE_FIRST));
    CK_B(aclrtMemcpy(*dPat, pat.size() * 4, pat.data(), pat.size() * 4,
                     ACL_MEMCPY_HOST_TO_DEVICE));
    CK_B(aclrtMalloc(dBell, static_cast<size_t>(c.rows) * c.ellCols * E,
                     ACL_MEM_MALLOC_HUGE_FIRST));
    return aclsparseCreateBlockedEll(B, c.rows, c.cols, c.bellB, c.ellCols,
                                     *dPat, *dBell, ACL_SPARSE_INDEX_32I,
                                     c.base, c.dtype) == ACL_SPARSE_STATUS_SUCCESS;
}

// CSR/CSC/COO descriptor (and payload buffers) for one perf case.
bool SetupCompressedPerf(const PerfCase &c, int64_t nnz, int E,
    aclsparseSpMatDescr_t *B, void **dOff, void **dIdx, void **dRows,
    void **dCols, void **dVals)
{
    if (c.format == ACL_SPARSE_FORMAT_COO) {
        CK_B(aclrtMalloc(dRows, nnz * 4, ACL_MEM_MALLOC_HUGE_FIRST));
        CK_B(aclrtMalloc(dCols, nnz * 4, ACL_MEM_MALLOC_HUGE_FIRST));
        CK_B(aclrtMalloc(dVals, nnz * E, ACL_MEM_MALLOC_HUGE_FIRST));
        return aclsparseCreateCoo(B, c.rows, c.cols, nnz, *dRows, *dCols,
                                  *dVals, ACL_SPARSE_INDEX_32I, c.base,
                                  c.dtype) == ACL_SPARSE_STATUS_SUCCESS;
    }
    CK_B(aclrtMalloc(dOff,
                     (static_cast<size_t>(
                          c.format == ACL_SPARSE_FORMAT_CSC ? c.cols
                                                            : c.rows) +
                      1) * 4,
                     ACL_MEM_MALLOC_HUGE_FIRST));
    CK_B(aclrtMalloc(dIdx, nnz * 4, ACL_MEM_MALLOC_HUGE_FIRST));
    CK_B(aclrtMalloc(dVals, nnz * E, ACL_MEM_MALLOC_HUGE_FIRST));
    if (c.format == ACL_SPARSE_FORMAT_CSR) {
        return aclsparseCreateCsr(B, c.rows, c.cols, nnz, *dOff, *dIdx,
                                  *dVals, ACL_SPARSE_INDEX_32I,
                                  ACL_SPARSE_INDEX_32I, c.base,
                                  c.dtype) == ACL_SPARSE_STATUS_SUCCESS;
    }
    return aclsparseCreateCsc(B, c.rows, c.cols, nnz, *dOff, *dIdx, *dVals,
                              ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_32I,
                              c.base, c.dtype) == ACL_SPARSE_STATUS_SUCCESS;
}

// Warmup + timed three-stage window (median of 30 after 10 warmup,
// descriptors/workspace reused).
bool TimeConvertCase(aclrtStream stream, aclsparseHandle_t handle,
    aclsparseDnMatDescr_t A, aclsparseSpMatDescr_t B, void *ws, float *median)
{
    CK_B(aclsparseDenseToSparseAnalysis(handle, A, B,
                                        ACL_SPARSE_DENSETOSPARSE_ALG_DEFAULT,
                                        ws));
    CK_B(aclrtSynchronizeStream(stream));
    for (int i = 0; i < 10; ++i) {
        CK_B(aclsparseDenseToSparseAnalysis(handle, A, B,
                                            ACL_SPARSE_DENSETOSPARSE_ALG_DEFAULT,
                                            ws));
        CK_B(aclsparseDenseToSparseConvert(handle, A, B,
                                           ACL_SPARSE_DENSETOSPARSE_ALG_DEFAULT,
                                           ws));
    }
    CK_B(aclrtSynchronizeStream(stream));

    aclrtEvent start = nullptr, stop = nullptr;
    CK_B(aclrtCreateEvent(&start));
    CK_B(aclrtCreateEvent(&stop));
    std::vector<float> samples;
    for (int i = 0; i < 30; ++i) {
        CK_B(aclrtRecordEvent(start, stream));
        CK_B(aclsparseDenseToSparseAnalysis(handle, A, B,
                                            ACL_SPARSE_DENSETOSPARSE_ALG_DEFAULT,
                                            ws));
        CK_B(aclsparseDenseToSparseConvert(handle, A, B,
                                           ACL_SPARSE_DENSETOSPARSE_ALG_DEFAULT,
                                           ws));
        CK_B(aclrtRecordEvent(stop, stream));
        CK_B(aclrtSynchronizeEvent(stop));
        float ms = 0.0f;
        CK_B(aclrtEventElapsedTime(&ms, start, stop));
        samples.push_back(ms * 1000.0f);
    }
    aclrtDestroyEvent(start);
    aclrtDestroyEvent(stop);
    std::sort(samples.begin(), samples.end());
    *median = samples[samples.size() / 2];
    return true;
}

// BELL verdict against the official per-case cuSPARSE medians;
// csr/csc/coo prints are informational only.
void ReportCase(const PerfCase &c, float median)
{
    const bool bell = c.format == ACL_SPARSE_FORMAT_BLOCKED_ELL;
    if (bell) {
        const int si = c.scene[3] == '1' ? 0 : c.scene[3] == '2' ? 1 : 2;
        const double gpuUs =
            kBellOfficial[si][DtypeIndex(c.dtype)][c.base == ACL_SPARSE_INDEX_BASE_ONE ? 1 : 0];
        // A 0-median would mean a sub-ns conversion: keep the guarded
        // divisor and score it as an unbounded speedup.
        const double ratio = gpuUs / (median > 0.0f ? median : 1.0e-9f);
        const bool pass = ratio >= 0.25;
        if (!pass) {
            ++gFail;
        }
        std::printf("%s %-11s %-4s base=%d ord=%s | npu=%9.1fus gpu=%8.3fus "
                    "ratio=%6.3f %s\n",
                    c.scene, FmtName(c.format), DtName(c.dtype), (int)c.base,
                    c.order == ACL_SPARSE_ORDER_ROW ? "row" : "col", median,
                    gpuUs, ratio, pass ? "PASS" : "FAIL");
    } else {
        // Informational only: the C++ window must not be joined against the
        // PyTorch-GPU baseline segment (task-book 3.3, 口径说明 §7.4).
        std::printf("%s %-6s %-11s base=%d ord=%s | npu=%9.1fus "
                    "(info: C++ window, no verdict)\n",
                    c.scene, FmtName(c.format), DtName(c.dtype), (int)c.base,
                    c.order == ACL_SPARSE_ORDER_ROW ? "row" : "col", median);
    }
}

int RunCase(aclrtStream stream, aclsparseHandle_t handle, const PerfCase &c) {
    const int E = ElemBytes(c.dtype);
    const int ld = (c.order == ACL_SPARSE_ORDER_ROW ? c.cols : c.rows);
    const size_t pitch =
        c.order == ACL_SPARSE_ORDER_ROW ? c.rows : c.cols;
    std::vector<uint8_t> host = MakeDense(c, ld, pitch);

    void *dA = nullptr, *dOff = nullptr, *dIdx = nullptr, *dRows = nullptr,
         *dCols = nullptr, *dVals = nullptr, *dPat = nullptr, *dBell = nullptr,
         *ws = nullptr;
    CK(aclrtMalloc(&dA, host.size(), ACL_MEM_MALLOC_HUGE_FIRST));
    CK(aclrtMemcpy(dA, host.size(), host.data(), host.size(),
                   ACL_MEMCPY_HOST_TO_DEVICE));
    aclsparseDnMatDescr_t A = nullptr;
    aclsparseSpMatDescr_t B = nullptr;
    CK(aclsparseCreateDnMat(&A, c.rows, c.cols, ld, dA, c.dtype, c.order));
    const int64_t nnz = static_cast<int64_t>(c.rows) * c.nnzPerRow;
    if (c.format == ACL_SPARSE_FORMAT_BLOCKED_ELL) {
        if (!SetupBellPerf(c, E, &B, &dPat, &dBell)) {
            return 1;
        }
    } else if (!SetupCompressedPerf(c, nnz, E, &B, &dOff, &dIdx, &dRows,
                                    &dCols, &dVals)) {
        return 1;
    }

    size_t bs = 0;
    CK(aclsparseDenseToSparseGetBufferSize(
        handle, A, B, ACL_SPARSE_DENSETOSPARSE_ALG_DEFAULT, &bs));
    if (bs > 0) {
        CK(aclrtMalloc(&ws, bs, ACL_MEM_MALLOC_HUGE_FIRST));
    }
    float median = 0.0f;
    if (!TimeConvertCase(stream, handle, A, B, ws, &median)) {
        return 1;
    }
    ReportCase(c, median);

    aclsparseDestroySpMat(B);
    aclsparseDestroyDnMat(A);
    if (ws) aclrtFree(ws);
    if (dBell) aclrtFree(dBell);
    if (dPat) aclrtFree(dPat);
    if (dVals) aclrtFree(dVals);
    if (dCols) aclrtFree(dCols);
    if (dRows) aclrtFree(dRows);
    if (dIdx) aclrtFree(dIdx);
    if (dOff) aclrtFree(dOff);
    aclrtFree(dA);
    return 0;
}

} // namespace

namespace {

struct Scene {
    const char *name;
    int rows, cols, nnzPerRow, bellB, ellCols;
};

PerfCase MakeVerdictCase(const Scene &sc, aclDataType dt, int base)
{
    PerfCase c{};
    c.scene = sc.name;
    c.rows = sc.rows;
    c.cols = sc.cols;
    c.nnzPerRow = sc.nnzPerRow;
    c.bellB = sc.bellB;
    c.ellCols = sc.ellCols;
    c.dtype = dt;
    c.format = ACL_SPARSE_FORMAT_BLOCKED_ELL;
    c.base = static_cast<aclsparseIndexBase_t>(base);
    c.order = ACL_SPARSE_ORDER_ROW;
    return c;
}

// The official cuSPARSE BELL baseline is row-only.
void RunVerdictSweep(const Scene &sc, aclrtStream stream,
    aclsparseHandle_t handle, const aclDataType *dtypes, int nDtypes,
    const char *only)
{
    for (int base = 0; base < 2; ++base) {
        for (int di = 0; di < nDtypes; ++di) {
            const aclDataType dt = dtypes[di];
            PerfCase c = MakeVerdictCase(sc, dt, base);
            if (only && std::string(c.scene) != only &&
                std::string(FmtName(c.format)) != only &&
                std::string(DtName(dt)) != only) {
                continue;
            }
            if (RunCase(stream, handle, c)) {
                std::exit(1);
            }
        }
    }
}

// Informational sweep of the other formats (no verdict): the acceptance
// caliber for csr/csc/coo is the official runner.
void RunInfoSweep(const Scene &sc, aclrtStream stream,
    aclsparseHandle_t handle, const aclDataType *dtypes, int nDtypes,
    const char *only)
{
    const aclsparseFormat_t others[] = {ACL_SPARSE_FORMAT_CSR,
                                        ACL_SPARSE_FORMAT_CSC,
                                        ACL_SPARSE_FORMAT_COO};
    for (aclsparseFormat_t fmt : others) {
        for (int di = 0; di < nDtypes; ++di) {
            const aclDataType dt = dtypes[di];
            if (only && std::string(FmtName(fmt)) != only &&
                std::string(DtName(dt)) != only &&
                std::string(sc.name) != only) {
                continue;
            }
            PerfCase c = MakeVerdictCase(sc, dt, 0);
            c.format = fmt;
            if (RunCase(stream, handle, c)) {
                std::exit(1);
            }
        }
    }
}

} // namespace

int main(int argc, char **argv) {
    const char *only = argc > 1 ? argv[1] : nullptr;
    CK(aclInit(nullptr));
    CK(aclrtSetDevice(0));
    aclrtStream stream = nullptr;
    CK(aclrtCreateStream(&stream));
    aclsparseHandle_t handle = nullptr;
    CK(aclsparseCreate(&handle));
    CK(aclsparseSetStream(handle, stream));

    const Scene scenes[] = {
        {"P-01", 8192, 28672, 64, 16, 64},
        {"P-02", 4096, 1536, 64, 32, 64},
        {"P-03", 7168, 2048, 64, 64, 64},
    };
    const aclDataType dtypes[] = {ACL_INT8, ACL_FLOAT16, ACL_BF16, ACL_FLOAT,
                                  ACL_COMPLEX64};
    // Read once at startup before worker threads exist; the perf harness
    // is single-threaded up to this point.
    const bool infoAll = std::getenv("D2S_INFO_ALL") != nullptr;
    for (const Scene &sc : scenes) {
        RunVerdictSweep(sc, stream, handle, dtypes, 5, only);
        if (infoAll) {
            RunInfoSweep(sc, stream, handle, dtypes, 5, only);
        }
    }
    std::printf("perf: %d failures (target ratio >= 0.25, official per-case "
                "medians)\n", gFail);
    aclsparseDestroy(handle);
    aclrtDestroyStream(stream);
    CK(aclrtResetDevice(0));
    CK(aclFinalize());
    return gFail ? 2 : 0;
}
