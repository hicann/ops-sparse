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

#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <random>
#include <cstring>
#include <sstream>
#include <cmath>
#include <string>
#include <memory>
#include <limits>
#include <numeric>
#include <type_traits>
#include "acl/acl.h"
#include "cann_ops_sparse.h"

// Host-side IEEE 754 binary16 storage. x86_64 GCC does not provide arm_fp16.h.
struct half {
    uint16_t bits;

    half() : bits(0)
    {}
    half(float value)
    {
        *this = value;
    }
    operator float() const
    {
        uint32_t sign = static_cast<uint32_t>(bits & 0x8000u) << 16;
        uint32_t exp = (bits >> 10) & 0x1fu;
        uint32_t mant = bits & 0x03ffu;
        uint32_t result;
        if (exp == 0u) {
            if (mant == 0u) {
                result = sign;
            } else {
                uint32_t shift = static_cast<uint32_t>(__builtin_clz(mant)) - 21u;
                mant <<= shift;
                result = sign | ((1u + 127u - 15u - shift) << 23) | (mant << 13);
            }
        } else if (exp == 31u) {
            result = sign | 0x7f800000u | (mant << 13);
        } else {
            result = sign | ((exp + 127u - 15u) << 23) | (mant << 13);
        }
        float value;
        __builtin_memcpy(&value, &result, sizeof(value));
        return value;
    }
    half& operator=(float value)
    {
        uint32_t raw;
        __builtin_memcpy(&raw, &value, sizeof(raw));
        uint32_t sign = (raw >> 16) & 0x8000u;
        uint32_t rawExp = (raw >> 23) & 0xffu;
        int32_t exp = static_cast<int32_t>(rawExp) - 127;
        uint32_t mant = raw & 0x007fffffu;
        if (rawExp == 0xffu) {
            bits = static_cast<uint16_t>(sign | 0x7c00u | (mant != 0u ? 0x0200u : 0u));
        } else if (exp >= 16) {
            bits = static_cast<uint16_t>(sign | 0x7c00u);
        } else if (exp >= -14) {
            uint32_t mant10 = mant >> 13;
            uint32_t remainder = mant & 0x1fffu;
            uint32_t encoded = (static_cast<uint32_t>(exp + 15) << 10) | mant10;
            if (remainder > 0x1000u || (remainder == 0x1000u && (mant10 & 1u) != 0u)) {
                encoded += 1u;
            }
            bits = static_cast<uint16_t>(sign | encoded);
        } else if (exp >= -25) {
            uint32_t significand = 0x00800000u | mant;
            uint32_t shift = static_cast<uint32_t>(-exp - 1);
            uint32_t frac = significand >> shift;
            uint32_t remainder = significand & ((1u << shift) - 1u);
            uint32_t midpoint = 1u << (shift - 1u);
            if (remainder > midpoint || (remainder == midpoint && (frac & 1u) != 0u)) {
                frac += 1u;
            }
            bits = static_cast<uint16_t>(sign | frac);
        } else {
            bits = static_cast<uint16_t>(sign);
        }
        return *this;
    }
};
static_assert(sizeof(half) == 2, "half must be 2 bytes");

struct bfloat16_t {
    uint16_t bits;
    bfloat16_t() : bits(0)
    {}
    bfloat16_t(float f)
    {
        *this = f;
    }
    operator float() const
    {
        uint32_t u = static_cast<uint32_t>(bits) << 16;
        union {
            uint32_t u;
            float f;
        } x = {u};
        return x.f;
    }
    bfloat16_t& operator=(float f)
    {
        union {
            float f;
            uint32_t u;
        } x = {f};
        bits = static_cast<uint16_t>(x.u >> 16);
        return *this;
    }
};
static_assert(sizeof(bfloat16_t) == 2, "bfloat16_t must be 2 bytes");

// ===================== 工具宏 =====================
#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)

#define LOG_PRINT(message, ...)         \
    do {                                \
        printf(message, ##__VA_ARGS__); \
    } while (0)

uint32_t StableSeed(const std::string& text)
{
    uint32_t hash = 2166136261u;
    for (unsigned char value : text) {
        hash ^= value;
        hash *= 16777619u;
    }
    return hash;
}

// ===================== 数据生成 =====================

/* 为某一行无重复地从 [0, numCols) 中采样 nnz 个列号, 返回升序列表 */
void SampleRowColumnsUnique(
    int nnz, uint32_t numCols, uint32_t rowIdx, std::vector<int>& visited, std::mt19937& rng, std::vector<int>& outCols)
{
    std::uniform_int_distribution<int> colDist(0, static_cast<int>(numCols) - 1);
    outCols.clear();
    outCols.reserve(nnz);
    while (static_cast<int>(outCols.size()) < nnz) {
        int c = colDist(rng);
        if (visited[c] != static_cast<int>(rowIdx)) {
            visited[c] = static_cast<int>(rowIdx);
            outCols.push_back(c);
        }
    }
    std::sort(outCols.begin(), outCols.end());
}

template <typename T>
void GenerateDenseVector(uint32_t size, std::vector<T>& x, std::mt19937& rng, float valueLimit = 10.0f)
{
    std::uniform_real_distribution<float> valDist(-valueLimit, valueLimit);
    x.assign(size, T{});
    for (uint32_t i = 0; i < size; ++i) {
        x[i] = static_cast<T>(valDist(rng));
    }
}

/**
 * @brief 生成 CSR 稀疏矩阵
 *
 * @param sparsity 零元素的比例, [0,1]
 */
template <typename T>
void GenerateCsr(
    uint32_t numRows, uint32_t numCols, float sparsity, std::vector<int32_t>& csrRowPtr,
    std::vector<int32_t>& csrColInd, std::vector<T>& csrVal, uint32_t seed, float valueLimit = 10.0f,
    float emptyRowProb = 0.0f)
{
    if (sparsity < 0.0f || sparsity > 1.0f) {
        std::cerr << "[ERROR] sparsity must be in [0, 1], got " << sparsity << "\n";
        return;
    }
    const float density = 1.0f - sparsity;

    csrRowPtr.assign(numRows + 1, 0);
    csrColInd.clear();
    csrVal.clear();

    const size_t expectedNNZ = static_cast<size_t>(
        static_cast<float>(numRows) * static_cast<float>(numCols) * density * (1.0f - emptyRowProb));
    csrColInd.reserve(expectedNNZ);
    csrVal.reserve(expectedNNZ);

    std::mt19937 rng(seed);

    std::binomial_distribution<int> nnzDist(numCols, density);
    std::uniform_real_distribution<float> valDist(-valueLimit, valueLimit);
    std::uniform_real_distribution<float> emptyDist(0.0f, 1.0f);

    std::vector<int> visited(numCols, -1);
    std::vector<int> rowCols;

    for (uint32_t i = 0; i < numRows; ++i) {
        csrRowPtr[i] = static_cast<uint32_t>(csrColInd.size());

        if (emptyRowProb > 0.0f && emptyDist(rng) < emptyRowProb) {
            continue;
        }

        const int nnz = nnzDist(rng);
        if (nnz <= 0) {
            continue;
        }

        SampleRowColumnsUnique(nnz, numCols, i, visited, rng, rowCols);
        for (int c : rowCols) {
            csrColInd.push_back(static_cast<uint32_t>(c));
            csrVal.push_back(static_cast<T>(valDist(rng)));
        }
    }
    csrRowPtr[numRows] = static_cast<uint32_t>(csrColInd.size());
}

// ===================== CPU 参考实现 =====================

int32_t ClampInt32Reference(__int128 value)
{
    const int64_t low = static_cast<int64_t>(std::numeric_limits<int32_t>::min());
    const int64_t high = static_cast<int64_t>(std::numeric_limits<int32_t>::max());
    return static_cast<int32_t>(std::max<__int128>(low, std::min<__int128>(high, value)));
}

template <typename OutT, typename AccT>
OutT CastReferenceOutput(AccT value)
{
    if constexpr (std::is_same_v<OutT, int32_t>) {
        return ClampInt32Reference(value);
    } else {
        return static_cast<OutT>(value);
    }
}

template <typename CompT, typename ValT = CompT, typename OutT = CompT>
std::vector<OutT> SpmvCpu(
    const std::vector<int32_t>& csrRowPtr, const std::vector<int32_t>& csrColInd, const std::vector<ValT>& csrVal,
    const std::vector<ValT>& xVec, const std::vector<OutT>& yVec, CompT alpha = static_cast<CompT>(1),
    CompT beta = static_cast<CompT>(0))
{
    // Host-only exact oracle: the scaled INT8 dot product can exceed INT64.
    using RefT = std::conditional_t<std::is_same_v<CompT, int32_t>, __int128, double>;
    uint32_t M = csrRowPtr.size() - 1;
    std::vector<OutT> z(M);
    for (uint32_t i = 0; i < M; ++i) {
        RefT sum = 0;
        for (int32_t j = csrRowPtr[i]; j < csrRowPtr[i + 1]; ++j) {
            sum += static_cast<RefT>(csrVal[j]) * static_cast<RefT>(xVec[csrColInd[j]]);
        }
        RefT zm = static_cast<RefT>(alpha) * sum + static_cast<RefT>(beta) * static_cast<RefT>(yVec[i]);
        z[i] = CastReferenceOutput<OutT>(zm);
    }
    return z;
}

// ===================== CPU 参考实现（转置） =====================

template <typename CompT, typename ValT = CompT, typename OutT = CompT>
std::vector<OutT> SpmvTransCpu(
    const std::vector<int32_t>& csrRowPtr, const std::vector<int32_t>& csrColInd, const std::vector<ValT>& csrVal,
    const std::vector<ValT>& xVec, const std::vector<OutT>& yVec, uint32_t numRows, uint32_t numCols,
    CompT alpha = static_cast<CompT>(1), CompT beta = static_cast<CompT>(0))
{
    using RefT = std::conditional_t<std::is_same_v<CompT, int32_t>, __int128, double>;
    std::vector<RefT> accum(numCols, static_cast<RefT>(0));
    for (uint32_t i = 0; i < numRows; ++i) {
        for (int32_t k = csrRowPtr[i]; k < csrRowPtr[i + 1]; ++k) {
            accum[csrColInd[k]] += static_cast<RefT>(xVec[i]) * static_cast<RefT>(csrVal[k]);
        }
    }
    std::vector<OutT> z(numCols);
    for (uint32_t j = 0; j < numCols; ++j) {
        const RefT result =
            static_cast<RefT>(alpha) * accum[j] + static_cast<RefT>(beta) * static_cast<RefT>(yVec[j]);
        z[j] = CastReferenceOutput<OutT>(result);
    }
    return z;
}

// ===================== 精度验证（生态算子开源精度标准：混合容差）
// =====================

struct MixedToleranceParams {
    float rtol;
    float atol;
    float fixedAbsErrorLimit;
    float requiredMatchedRatio;
    int32_t mantissaBits;
    int32_t minExponent;
};

template <typename T>
MixedToleranceParams GetMixedToleranceParams()
{
    if constexpr (std::is_same_v<T, float>) {
        return {std::ldexp(1.0f, -10), std::ldexp(1.0f, -16), 1e-2f, 0.99f, 23, -126};
    } else if constexpr (std::is_same_v<T, half>) {
        return {std::ldexp(1.0f, -9), std::ldexp(1.0f, -9), 1e-1f, 0.99f, 10, -14};
    } else if constexpr (std::is_same_v<T, bfloat16_t>) {
        return {std::ldexp(1.0f, -6), std::ldexp(1.0f, -6), 1e-0f, 0.99f, 7, -126};
    } else if constexpr (std::is_same_v<T, int32_t>) {
        return {0.0f, 0.0f, 0.0f, 1.0f, 0, 0};
    }
    return {std::ldexp(1.0f, -10), std::ldexp(1.0f, -16), 1e-2f, 0.99f, 23, -126};
}

float GetUlpAt(float magnitude, const MixedToleranceParams& params)
{
    const float absolute = std::fabs(magnitude);
    if (absolute == 0.0f) {
        return std::ldexp(1.0f, params.minExponent - params.mantissaBits);
    }
    int32_t exponent = 0;
    std::frexp(absolute, &exponent);
    return std::ldexp(1.0f, std::max(exponent - 1, params.minExponent) - params.mantissaBits);
}

template <typename T>
struct VerificationState {
    size_t passCount = 0;
    size_t worstIdx = 0;
    float worstGolden = 0.0f;
    float worstNpu = 0.0f;
    float maxAbsError = 0.0f;
    float worstMargin = 0.0f;
    size_t worstAbsLimitIdx = 0;
    float worstAbsLimitGolden = 0.0f;
    float worstAbsLimitNpu = 0.0f;
    float worstAbsElementLimit = 0.0f;
    float worstAbsLimitMargin = -std::numeric_limits<float>::infinity();
    bool absErrorLimitFailed = false;

    void ObserveInteger(size_t i, T golden, T result, float cpuVal, float npuVal, float aError)
    {
        if (result == golden) {
            passCount++;
        } else {
            std::cout << "[WARNING] value[" << i << "] in result is not equal to golden, the value is: " << result
                      << " while the golden is: " << golden << "\n";
            const float margin = aError;
            if (margin > worstMargin) {
                worstMargin = margin;
                worstIdx = i;
                worstGolden = cpuVal;
                worstNpu = npuVal;
            }
        }
    }

    void ObserveFloat(size_t i, float cpuVal, float npuVal, float aError, const MixedToleranceParams& params)
    {
        const float tolerance = params.atol + params.rtol * std::fabs(cpuVal);
        const float elementLimit = std::max(params.fixedAbsErrorLimit, 32.0f * GetUlpAt(cpuVal, params));
        if (!std::isfinite(npuVal) || !std::isfinite(cpuVal) || aError > elementLimit) {
            absErrorLimitFailed = true;
            const float absLimitMargin =
                std::isfinite(aError) ? aError - elementLimit : std::numeric_limits<float>::infinity();
            if (absLimitMargin > worstAbsLimitMargin) {
                worstAbsLimitMargin = absLimitMargin;
                worstAbsElementLimit = elementLimit;
                worstAbsLimitIdx = i;
                worstAbsLimitGolden = cpuVal;
                worstAbsLimitNpu = npuVal;
            }
        }
        if (aError <= tolerance) {
            passCount++;
        } else {
            const float margin = aError - tolerance;
            if (margin > worstMargin) {
                worstMargin = margin;
                worstIdx = i;
                worstGolden = cpuVal;
                worstNpu = npuVal;
            }
        }
    }

    void Observe(size_t i, T golden, T result, const MixedToleranceParams& params)
    {
        const float npuVal = static_cast<float>(result);
        const float cpuVal = static_cast<float>(golden);
        if constexpr (!std::is_same_v<T, int32_t>) {
            const bool equivalentNan = std::isnan(npuVal) && std::isnan(cpuVal);
            const bool equivalentInf =
                std::isinf(npuVal) && std::isinf(cpuVal) && std::signbit(npuVal) == std::signbit(cpuVal);
            if (equivalentNan || equivalentInf) {
                ++passCount;
                return;
            }
        }
        const float aError = std::fabs(npuVal - cpuVal);
        if (aError > maxAbsError) {
            maxAbsError = aError;
        }
        if constexpr (std::is_same_v<T, int32_t>) {
            ObserveInteger(i, golden, result, cpuVal, npuVal, aError);
        } else {
            ObserveFloat(i, cpuVal, npuVal, aError, params);
        }
    }

    int32_t Report(float matchedRatio, const MixedToleranceParams& params) const
    {
        int32_t status = 0;
        if constexpr (std::is_same_v<T, int32_t>) {
            if (matchedRatio < params.requiredMatchedRatio) {
                std::cout << "[WARNING] Integer exact match check fail!\n";
                status = 1;
            }
            std::cout << "Matched Ratio = " << matchedRatio << "; Max Absolute Error = " << maxAbsError << "\n";
        } else {
            if (matchedRatio < params.requiredMatchedRatio) {
                std::cout << "[WARNING] Matched ratio check fail! matched_ratio=" << matchedRatio
                          << " required>=" << params.requiredMatchedRatio << "\n";
                status = 1;
            }
            if (absErrorLimitFailed) {
                std::cout << "[WARNING] Per-element absolute error check fail! worst_index=" << worstAbsLimitIdx
                          << " error=" << std::fabs(worstAbsLimitNpu - worstAbsLimitGolden)
                          << " limit=" << worstAbsElementLimit << "\n";
                status = 1;
            }
            std::cout << "Matched Ratio = " << matchedRatio << "; Max Absolute Error = " << maxAbsError
                      << " (rtol=" << params.rtol << ", atol=" << params.atol << ", abs_limit=max("
                      << params.fixedAbsErrorLimit << ", 32*ULP))\n";
        }

        return status;
    }

    void DescribeWorst(std::string* worstInfo) const
    {
        if (worstInfo) {
            std::ostringstream oss;
            oss << "worst[" << worstIdx << "] golden=" << worstGolden << " npu=" << worstNpu;
            if constexpr (!std::is_same_v<T, int32_t>) {
                oss << " margin=" << worstMargin;
            }
            *worstInfo = oss.str();
        }
    }
};

template <typename T>
int32_t Verification(
    const std::vector<T>& cpuGolden, const std::vector<T>& npuRet, float& MARE, float& MERE,
    const MixedToleranceParams& params, std::string* worstInfo = nullptr)
{
    if (npuRet.size() != cpuGolden.size()) {
        std::cout << "[ERROR] The size of npuRet and cpuGolden is not equal!\n";
        return 1;
    }
    std::cout << "Verification...\n";
    for (int i = 0; i < std::min(static_cast<int64_t>(npuRet.size()), static_cast<int64_t>(10)); ++i) {
        std::cout << "golden[" << i << "]=" << cpuGolden[i] << " npu_result[" << i << "]=" << npuRet[i] << "\n";
    }
    VerificationState<T> state;
    for (size_t i = 0; i < npuRet.size(); ++i) {
        state.Observe(i, cpuGolden[i], npuRet[i], params);
    }
    MARE = state.maxAbsError;
    MERE = npuRet.empty() ? 1.0f : static_cast<float>(state.passCount) / static_cast<float>(npuRet.size());
    state.DescribeWorst(worstInfo);
    return state.Report(MERE, params);
}

// ===================== 设备资源创建 =====================

// ===================== Init / Finalize =====================

int Init(int32_t deviceId, aclrtStream* stream)
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

// ===================== 类型映射 =====================

template <typename T>
aclDataType AclTypeOf();
template <>
aclDataType AclTypeOf<float>()
{
    return ACL_FLOAT;
}
template <>
aclDataType AclTypeOf<int32_t>()
{
    return ACL_INT32;
}
template <>
aclDataType AclTypeOf<int8_t>()
{
    return ACL_INT8;
}
template <>
aclDataType AclTypeOf<half>()
{
    return ACL_FLOAT16;
}
template <>
aclDataType AclTypeOf<bfloat16_t>()
{
    return ACL_BF16;
}

// ===================== 主测试函数 =====================

enum class SpecialValueKind {
    None,
    PositiveInfinity,
    NegativeInfinity,
    NaN,
};

/**
 * @brief SpMV 单用例测试（使用 aclsparse 新接口）
 *
 * @tparam CompT     计算类型 (float / int32_t)
 * @tparam ValT      输入值类型（默认=CompT）
 * @tparam OutT      输出类型（默认=CompT）
 */
class SpmvDeviceBuffer {
public:
    SpmvDeviceBuffer() = default;
    SpmvDeviceBuffer(const SpmvDeviceBuffer&) = delete;
    SpmvDeviceBuffer& operator=(const SpmvDeviceBuffer&) = delete;
    ~SpmvDeviceBuffer()
    {
        if (data_ != nullptr) {
            aclrtFree(data_);
        }
    }

    void* Get() const { return data_; }

    int Allocate(size_t bytes) { return aclrtMalloc(&data_, std::max<size_t>(bytes, 1), ACL_MEM_MALLOC_HUGE_FIRST); }

    int Upload(const void* source, size_t bytes)
    {
        const int ret = Allocate(bytes);
        if (ret != ACL_SUCCESS || bytes == 0) {
            return ret;
        }
        return aclrtMemcpy(data_, bytes, source, bytes, ACL_MEMCPY_HOST_TO_DEVICE);
    }

    template <typename T>
    int Upload(const std::vector<T>& host)
    {
        return Upload(host.data(), host.size() * sizeof(T));
    }

    template <typename T>
    int Download(std::vector<T>& host) const
    {
        const size_t bytes = host.size() * sizeof(T);
        if (bytes == 0) {
            return ACL_SUCCESS;
        }
        return aclrtMemcpy(host.data(), bytes, data_, bytes, ACL_MEMCPY_DEVICE_TO_HOST);
    }

    template <typename T>
    int Reset(const std::vector<T>& host)
    {
        const size_t bytes = host.size() * sizeof(T);
        if (bytes == 0) {
            return ACL_SUCCESS;
        }
        return aclrtMemcpy(data_, bytes, host.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE);
    }

private:
    void* data_{nullptr};
};

template <typename CompT, typename ValT, typename OutT>
struct SpmvTestData {
    size_t rows{0};
    size_t cols{0};
    size_t nnz{0};
    size_t xSize{0};
    size_t ySize{0};
    bool transpose{false};
    CompT alpha{};
    CompT beta{};
    std::vector<int32_t> rowOffsets;
    std::vector<int32_t> colIndices;
    std::vector<ValT> values;
    std::vector<ValT> x;
    std::vector<OutT> y;
    int64_t rowStride{1};
    int64_t colStride{1};
    int64_t valueStride{1};
    int64_t xStride{1};
    int64_t yStride{1};

    void Generate(float sparsity, uint32_t seed, float valueLimit)
    {
        GenerateCsr<ValT>(rows, cols, sparsity, rowOffsets, colIndices, values, seed, valueLimit);
        nnz = colIndices.size();
        xSize = transpose ? rows : cols;
        ySize = transpose ? cols : rows;
        std::mt19937 rng(seed ^ 0x9e3779b9u);
        GenerateDenseVector<ValT>(xSize, x, rng, valueLimit);
        GenerateDenseVector<OutT>(ySize, y, rng, valueLimit);
    }

    int Inject(SpecialValueKind specialValue)
    {
        if (specialValue == SpecialValueKind::None) {
            return 0;
        }
        if constexpr (!std::is_same_v<CompT, float>) {
            LOG_PRINT("[ERROR] Non-finite injection requires floating-point compute.\n");
            return 1;
        } else {
            if (values.empty() || x.empty() || y.empty()) {
                LOG_PRINT("[ERROR] Non-finite injection requires a non-empty 1x1 case.\n");
                return 1;
            }
            float injected = std::numeric_limits<float>::quiet_NaN();
            if (specialValue == SpecialValueKind::PositiveInfinity) {
                injected = std::numeric_limits<float>::infinity();
            } else if (specialValue == SpecialValueKind::NegativeInfinity) {
                injected = -std::numeric_limits<float>::infinity();
            }
            values[0] = static_cast<ValT>(injected);
            x[0] = static_cast<ValT>(1.0f);
            y[0] = static_cast<OutT>(0.0f);
            return 0;
        }
    }

    std::vector<OutT> Reference() const
    {
        if (transpose) {
            return SpmvTransCpu<CompT, ValT, OutT>(rowOffsets, colIndices, values, x, y, rows, cols, alpha, beta);
        }
        return SpmvCpu<CompT, ValT, OutT>(rowOffsets, colIndices, values, x, y, alpha, beta);
    }
};

template <typename CompT, typename ValT, typename OutT>
class SpmvTestContext {
public:
    using Data = SpmvTestData<CompT, ValT, OutT>;
    explicit SpmvTestContext(const Data& data, aclrtStream stream, aclsparseSpMVAlg_t alg = ACL_SPARSE_SPMV_ALG_DEFAULT)
        : data_(data),
          stream_(stream),
          alg_(alg),
          op_(data.transpose ? ACL_SPARSE_OP_TRANSPOSE : ACL_SPARSE_OP_NON_TRANSPOSE),
          alpha_(&data.alpha),
          beta_(&data.beta)
    {}
    SpmvTestContext(const SpmvTestContext&) = delete;
    SpmvTestContext& operator=(const SpmvTestContext&) = delete;

    ~SpmvTestContext()
    {
        if (vecY_ != nullptr) {
            aclsparseDestroyDnVec(vecY_);
        }
        if (vecX_ != nullptr) {
            aclsparseDestroyDnVec(vecX_);
        }
        if (mat_ != nullptr) {
            aclsparseDestroySpMat(mat_);
        }
        if (handle_ != nullptr) {
            aclsparseDestroy(handle_);
        }
    }

    int Initialize(bool devicePointerMode = false)
    {
        CHECK_RET(row_.Upload(data_.rowOffsets) == ACL_SUCCESS, return 1);
        CHECK_RET(col_.Upload(data_.colIndices) == ACL_SUCCESS, return 1);
        CHECK_RET(values_.Upload(data_.values) == ACL_SUCCESS, return 1);
        CHECK_RET(x_.Upload(data_.x) == ACL_SUCCESS, return 1);
        CHECK_RET(y_.Upload(data_.y) == ACL_SUCCESS, return 1);
        int ret = CreateDescriptors();
        CHECK_RET(ret == 0, return ret);
        if (devicePointerMode) {
            ret = ConfigureDeviceScalars();
        }
        return ret;
    }

    int Prepare()
    {
        size_t bytes = 0;
        aclsparseStatus_t status = aclsparseSpMVGetBufferSize(
            handle_, op_, alpha_, mat_, vecX_, beta_, vecY_, AclTypeOf<CompT>(), alg_, &bytes);
        CHECK_RET(status == ACL_SPARSE_STATUS_SUCCESS, return status);
        if (bytes > 0 && workspace_.Allocate(bytes) != ACL_SUCCESS) {
            return ACL_SPARSE_STATUS_ALLOC_FAILED;
        }
        return aclsparseSpMVPreprocess(
            handle_, op_, alpha_, mat_, vecX_, beta_, vecY_, AclTypeOf<CompT>(), alg_, workspace_.Get());
    }

    aclsparseStatus_t Execute()
    {
        return aclsparseSpMV(
            handle_, op_, alpha_, mat_, vecX_, beta_, vecY_, AclTypeOf<CompT>(), alg_, workspace_.Get());
    }

    int Read(std::vector<OutT>& result)
    {
        CHECK_RET(aclrtSynchronizeStream(stream_) == ACL_SUCCESS, return ACL_SPARSE_STATUS_EXECUTION_FAILED);
        return y_.Download(result);
    }

    int CheckDeterminism(const std::vector<OutT>& first)
    {
        constexpr int kExecutions = 20;
        std::vector<OutT> repeated(first.size());
        for (int repeat = 1; repeat < kExecutions; ++repeat) {
            if (y_.Reset(data_.y) != ACL_SUCCESS || Execute() != ACL_SPARSE_STATUS_SUCCESS || Read(repeated) != 0) {
                LOG_PRINT("[ERROR] Determinism execution %d failed.\n", repeat + 1);
                return 1;
            }
            if (!first.empty() && std::memcmp(repeated.data(), first.data(), first.size() * sizeof(OutT)) != 0) {
                LOG_PRINT("[ERROR] Determinism execution %d is not bitwise identical.\n", repeat + 1);
                return 1;
            }
        }
        std::cout << "Determinism check: " << kExecutions << "/" << kExecutions
                  << " executions are bitwise identical.\n";
        return 0;
    }

    int CheckStrides() const
    {
        int64_t row = 0, col = 0, value = 0, x = 0, y = 0;
        CHECK_RET(aclsparseCsrGetStrides(mat_, &row, &col, &value) == ACL_SPARSE_STATUS_SUCCESS, return 1);
        CHECK_RET(aclsparseDnVecGetStride(vecX_, &x) == ACL_SPARSE_STATUS_SUCCESS, return 1);
        CHECK_RET(aclsparseDnVecGetStride(vecY_, &y) == ACL_SPARSE_STATUS_SUCCESS, return 1);
        return row == data_.rowStride && col == data_.colStride && value == data_.valueStride && x == data_.xStride &&
                       y == data_.yStride ?
                   0 :
                   1;
    }

private:
    int CreateDescriptors()
    {
        aclsparseStatus_t status = aclsparseCreate(&handle_);
        CHECK_RET(status == ACL_SPARSE_STATUS_SUCCESS, return status);
        status = aclsparseSetStream(handle_, stream_);
        CHECK_RET(status == ACL_SPARSE_STATUS_SUCCESS, return status);
        status = aclsparseCreateCsr(
            &mat_, data_.rows, data_.cols, data_.nnz, row_.Get(), col_.Get(), values_.Get(), ACL_SPARSE_INDEX_32I,
            ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_BASE_ZERO, AclTypeOf<ValT>());
        CHECK_RET(status == ACL_SPARSE_STATUS_SUCCESS, return status);
        status = aclsparseCreateDnVec(&vecX_, data_.xSize, x_.Get(), AclTypeOf<ValT>());
        CHECK_RET(status == ACL_SPARSE_STATUS_SUCCESS, return status);
        status = aclsparseCreateDnVec(&vecY_, data_.ySize, y_.Get(), AclTypeOf<OutT>());
        CHECK_RET(status == ACL_SPARSE_STATUS_SUCCESS, return status);
        return ConfigureStrides();
    }

    int ConfigureStrides()
    {
        aclsparseStatus_t status = aclsparseCsrSetStrides(mat_, data_.rowStride, data_.colStride, data_.valueStride);
        CHECK_RET(status == ACL_SPARSE_STATUS_SUCCESS, return status);
        status = aclsparseDnVecSetStride(vecX_, data_.xStride);
        CHECK_RET(status == ACL_SPARSE_STATUS_SUCCESS, return status);
        return aclsparseDnVecSetStride(vecY_, data_.yStride);
    }

    int ConfigureDeviceScalars()
    {
        const aclsparseStatus_t status = aclsparseSetPointerMode(handle_, ACL_SPARSE_POINTER_MODE_DEVICE);
        CHECK_RET(status == ACL_SPARSE_STATUS_SUCCESS, return status);
        CHECK_RET(alphaDevice_.Upload(&data_.alpha, sizeof(CompT)) == ACL_SUCCESS, return 1);
        CHECK_RET(betaDevice_.Upload(&data_.beta, sizeof(CompT)) == ACL_SUCCESS, return 1);
        alpha_ = alphaDevice_.Get();
        beta_ = betaDevice_.Get();
        return 0;
    }

    const Data& data_;
    aclrtStream stream_;
    aclsparseSpMVAlg_t alg_;
    aclsparseOperation_t op_;
    const void* alpha_;
    const void* beta_;
    SpmvDeviceBuffer row_, col_, values_, x_, y_, alphaDevice_, betaDevice_, workspace_;
    aclsparseHandle_t handle_{nullptr};
    aclsparseSpMatDescr_t mat_{nullptr};
    aclsparseDnVecDescr_t vecX_{nullptr};
    aclsparseDnVecDescr_t vecY_{nullptr};
};

template <
    typename CompT, typename ValT = std::conditional_t<std::is_same_v<CompT, int32_t>, int8_t, CompT>,
    typename OutT = CompT>
int Test(
    const size_t M, const size_t N, const float sparsity, CompT alphaVal, CompT betaVal, bool transpose, float& MARE,
    float& MERE, aclrtStream stream, std::string* worstInfo = nullptr, bool devicePointerMode = false,
    bool verifyDeterminism = false, aclsparseSpMVAlg_t alg = ACL_SPARSE_SPMV_ALG_DEFAULT, uint32_t dataSeed = 20260826u,
    SpecialValueKind specialValue = SpecialValueKind::None, float valueLimit = 10.0f)
{
    const char* label = transpose ? " Transpose" : "";
    std::cout << "====Test" << label << " case: row num = " << M << " col num = " << N
              << " sparsity (zero ratio) = " << sparsity << " alpha = " << alphaVal << " beta = " << betaVal
              << "====\n";
    SpmvTestData<CompT, ValT, OutT> data;
    data.rows = M;
    data.cols = N;
    data.transpose = transpose;
    data.alpha = alphaVal;
    data.beta = betaVal;
    data.Generate(sparsity, dataSeed, valueLimit);
    CHECK_RET(data.Inject(specialValue) == 0, return 1);
    const std::vector<OutT> expected = data.Reference();
    SpmvTestContext<CompT, ValT, OutT> context(data, stream, alg);
    int ret = context.Initialize(devicePointerMode);
    CHECK_RET(ret == 0, return ret);
    ret = context.Prepare();
    CHECK_RET(ret == 0, return ret);
    ret = context.Execute();
    CHECK_RET(ret == 0, return ret);
    std::vector<OutT> result(data.ySize);
    ret = context.Read(result);
    CHECK_RET(ret == 0, return ret);
    ret = Verification<OutT>(expected, result, MARE, MERE, GetMixedToleranceParams<OutT>(), worstInfo);
    if (ret == 0 && verifyDeterminism) {
        ret = context.CheckDeterminism(result);
    }
    CHECK_RET(ret == 0, LOG_PRINT("====Test%s case fail!====\n\n", label); return ret);
    std::cout << "====Test" << label << " case pass!====\n\n";
    return 0;
}

// ===================== 测试统计 =====================

struct TestStats {
    int total = 0;
    int passed = 0;
    int failed = 0;
    std::vector<std::string> failedCases;
};

template <
    typename CompT, typename ValT = std::conditional_t<std::is_same_v<CompT, int32_t>, int8_t, CompT>,
    typename OutT = CompT>
int RunAndTrack(
    size_t M, size_t N, float sparsity, CompT alpha, CompT beta, bool transpose, TestStats& stats, aclrtStream stream,
    const std::string& tag = "", bool devicePointerMode = false, bool verifyDeterminism = false,
    aclsparseSpMVAlg_t alg = ACL_SPARSE_SPMV_ALG_DEFAULT,
    SpecialValueKind specialValue = SpecialValueKind::None, float valueLimit = 10.0f)
{
    float MARE = 0, MERE = 0;
    std::string worstInfo;
    std::ostringstream seedKey;
    seedKey << tag << ':' << M << ':' << N << ':' << sparsity << ':' << transpose;
    int ret = Test<CompT, ValT, OutT>(
        M, N, sparsity, alpha, beta, transpose, MARE, MERE, stream, &worstInfo, devicePointerMode, verifyDeterminism,
        alg, StableSeed(seedKey.str()), specialValue, valueLimit);

    stats.total++;
    if (ret == 0) {
        stats.passed++;
    } else {
        stats.failed++;
        std::ostringstream oss;
        if (!tag.empty()) {
            oss << "[" << tag << "] ";
        }
        oss << "type=" << (std::is_same_v<CompT, float> ? "float" : "int32")
            << (std::is_same_v<ValT, CompT> ? "" : " MixPrec") << (transpose ? " Transpose" : "") << " M=" << M
            << " N=" << N << " sparsity=" << sparsity << " alpha=" << alpha << " beta=" << beta
            << " max_abs_error=" << MARE << " matched_ratio=" << MERE << "\n"
            << "          " << worstInfo << "\n"
            << "          status=" << ret;
        stats.failedCases.push_back(oss.str());
    }
    return ret;
}

// ===================== 随机测试辅助函数 =====================

template <typename T, typename ADist, typename BDist>
void RunRandomParams(
    int numCases, const std::string& tagPrefix, bool transpose, TestStats& stats, aclrtStream stream, ADist alphaDist,
    BDist betaDist)
{
    std::cout << "\n======== " << (std::is_same_v<T, float> ? "Float" : "Int32") << (transpose ? " Transpose" : "")
              << " Random Tests ========\n";

    std::mt19937 rng(StableSeed(tagPrefix));
    std::uniform_int_distribution<int> rowDist(1, 2048);
    std::uniform_int_distribution<int> colDist(1, 4096);
    std::uniform_real_distribution<float> sparsityDist(0.5f, 0.999f);

    for (int i = 0; i < numCases; ++i) {
        size_t M = static_cast<size_t>(rowDist(rng));
        size_t N = static_cast<size_t>(colDist(rng));
        float sp = sparsityDist(rng);
        T a = static_cast<T>(alphaDist(rng));
        T b = static_cast<T>(betaDist(rng));
        std::cout << "--- Random " << (std::is_same_v<T, float> ? "float" : "int32") << (transpose ? " trans" : "")
                  << " case " << (i + 1) << "/" << numCases << " ---\n";
        RunAndTrack<T>(M, N, sp, a, b, transpose, stats, stream, tagPrefix + std::to_string(i + 1));
    }
}

struct PerformanceCase {
    uint32_t rows;
    uint32_t cols;
    float sparsity;
    float benchmarkUs;
};

class SpmvPerformanceTimer {
public:
    SpmvPerformanceTimer() = default;
    SpmvPerformanceTimer(const SpmvPerformanceTimer&) = delete;
    SpmvPerformanceTimer& operator=(const SpmvPerformanceTimer&) = delete;
    ~SpmvPerformanceTimer()
    {
        if (start_ != nullptr) {
            aclrtDestroyEvent(start_);
        }
        if (stop_ != nullptr) {
            aclrtDestroyEvent(stop_);
        }
    }

    int Initialize()
    {
        if (aclrtCreateEvent(&start_) != ACL_SUCCESS || aclrtCreateEvent(&stop_) != ACL_SUCCESS) {
            return 1;
        }
        return 0;
    }

    int Measure(SpmvTestContext<float, float, float>& context, aclrtStream stream, std::vector<float>& timesUs)
    {
        constexpr int32_t kMeasuredIterations = 100;
        timesUs.reserve(kMeasuredIterations);
        for (int32_t i = 0; i < kMeasuredIterations; ++i) {
            CHECK_RET(aclrtRecordEvent(start_, stream) == ACL_SUCCESS, return 1);
            const aclsparseStatus_t status = context.Execute();
            CHECK_RET(aclrtRecordEvent(stop_, stream) == ACL_SUCCESS, return 1);
            if (status != ACL_SPARSE_STATUS_SUCCESS || aclrtSynchronizeEvent(stop_) != ACL_SUCCESS) {
                return 1;
            }
            float elapsedMs = 0.0f;
            CHECK_RET(aclrtEventElapsedTime(&elapsedMs, start_, stop_) == ACL_SUCCESS, return 1);
            timesUs.push_back(elapsedMs * 1000.0f);
        }
        return 0;
    }

private:
    aclrtEvent start_{nullptr};
    aclrtEvent stop_{nullptr};
};

SpmvTestData<float, float, float> GeneratePerformanceData(const PerformanceCase& perfCase)
{
    SpmvTestData<float, float, float> data;
    data.rows = perfCase.rows;
    data.cols = perfCase.cols;
    data.xSize = perfCase.cols;
    data.ySize = perfCase.rows;
    data.alpha = 1.0f;
    data.beta = 0.0f;
    GenerateCsr<float>(
        perfCase.rows, perfCase.cols, perfCase.sparsity, data.rowOffsets, data.colIndices, data.values,
        StableSeed(std::to_string(perfCase.rows) + ":" + std::to_string(perfCase.cols)));
    data.nnz = data.colIndices.size();
    std::mt19937 rng(20260826u);
    GenerateDenseVector<float>(perfCase.cols, data.x, rng);
    GenerateDenseVector<float>(perfCase.rows, data.y, rng);
    return data;
}

int ReportPerformance(const PerformanceCase& perfCase, size_t nnz, std::vector<float>& timesUs)
{
    if (timesUs.size() != 100u) {
        std::cerr << "[ERROR] incomplete performance measurements\n";
        return 1;
    }
    std::sort(timesUs.begin(), timesUs.end());
    double totalUs = 0.0;
    for (float timeUs : timesUs) {
        totalUs += timeUs;
    }
    const double averageUs = totalUs / static_cast<double>(timesUs.size());
    const double score = static_cast<double>(perfCase.benchmarkUs) / averageUs;
    const bool pass = score >= 0.5;
    const double actualSparsity =
        1.0 - static_cast<double>(nnz) / (static_cast<double>(perfCase.rows) * static_cast<double>(perfCase.cols));
    std::cout << "PERF rows=" << perfCase.rows << " cols=" << perfCase.cols << " nnz=" << nnz
              << " sparsity=" << actualSparsity << " avg_us=" << averageUs << " p50_us=" << timesUs[timesUs.size() / 2]
              << " p90_us=" << timesUs[timesUs.size() * 9 / 10] << " min_us=" << timesUs.front()
              << " benchmark_us=" << perfCase.benchmarkUs << " score=" << score
              << " result=" << (pass ? "PASS" : "FAIL") << "\n";
    return pass ? 0 : 1;
}

int RunPerformanceCase(const PerformanceCase& perfCase, aclrtStream stream)
{
    const auto data = GeneratePerformanceData(perfCase);
    SpmvTestContext<float, float, float> context(data, stream);
    CHECK_RET(context.Initialize() == 0, return 1);
    CHECK_RET(context.Prepare() == 0, return 1);
    constexpr int32_t kWarmupIterations = 20;
    for (int32_t i = 0; i < kWarmupIterations; ++i) {
        CHECK_RET(context.Execute() == ACL_SPARSE_STATUS_SUCCESS, return 1);
    }
    CHECK_RET(aclrtSynchronizeStream(stream) == ACL_SUCCESS, return 1);
    SpmvPerformanceTimer timer;
    CHECK_RET(timer.Initialize() == 0, return 1);
    std::vector<float> timesUs;
    CHECK_RET(timer.Measure(context, stream, timesUs) == 0, return 1);
    return ReportPerformance(perfCase, data.nnz, timesUs);
}

int RunPerformanceSuite(aclrtStream stream)
{
    const PerformanceCase cases[] = {
        {128u, 128u, 0.95f, 43.9f},
        {1024u, 1024u, 0.99f, 46.3f},
        {2048u, 4096u, 0.975f, 45.4f},
        {160220u, 68750u, 0.999f, 193.2f},
    };
    int failed = 0;
    for (const PerformanceCase& perfCase : cases) {
        failed += RunPerformanceCase(perfCase, stream);
    }
    return failed == 0 ? 0 : 1;
}

struct TaskCase {
    int32_t id;
    uint32_t rows;
    uint32_t cols;
    float sparsity;
    float alpha;
    float beta;
    bool transpose;
    std::string valueType;
    std::string outputType;
    int32_t computeType;
    int32_t alg;
};

bool ParseTaskCase(const std::string& line, TaskCase& taskCase)
{
    std::stringstream stream(line);
    std::vector<std::string> fields;
    std::string field;
    while (std::getline(stream, field, ',')) {
        fields.push_back(field);
    }
    if (fields.size() != 11u) {
        return false;
    }
    try {
        taskCase.id = std::stoi(fields[0]);
        taskCase.rows = static_cast<uint32_t>(std::stoul(fields[1]));
        taskCase.cols = static_cast<uint32_t>(std::stoul(fields[2]));
        taskCase.sparsity = std::stof(fields[3]);
        taskCase.alpha = std::stof(fields[4]);
        taskCase.beta = std::stof(fields[5]);
        taskCase.transpose = std::stoi(fields[6]) != 0;
        taskCase.valueType = fields[7];
        taskCase.outputType = fields[8];
        taskCase.computeType = std::stoi(fields[9]);
        taskCase.alg = std::stoi(fields[10]);
    } catch (const std::exception&) {
        return false;
    }
    return true;
}

template <typename CompT, typename ValT, typename OutT>
void RunTypedTaskCase(const TaskCase& taskCase, TestStats& stats, aclrtStream stream, const std::string& tag)
{
    RunAndTrack<CompT, ValT, OutT>(
        taskCase.rows, taskCase.cols, taskCase.sparsity, static_cast<CompT>(taskCase.alpha),
        static_cast<CompT>(taskCase.beta), taskCase.transpose, stats, stream, tag, false, false,
        ACL_SPARSE_SPMV_ALG_DEFAULT, SpecialValueKind::None, 5.0f);
}

void RunTaskCase(const TaskCase& taskCase, TestStats& stats, aclrtStream stream)
{
    const std::string tag = "task-case-" + std::to_string(taskCase.id);
    if (taskCase.alg != 0) {
        ++stats.total;
        ++stats.failed;
        stats.failedCases.push_back(tag + " unsupported alg=" + std::to_string(taskCase.alg));
        return;
    }
    using Runner = void (*)(const TaskCase&, TestStats&, aclrtStream, const std::string&);
    struct TaskType {
        int32_t compute;
        const char* value;
        const char* output;
        Runner run;
    };
    const TaskType types[] = {
        {3, "int8", "int32", RunTypedTaskCase<int32_t, int8_t, int32_t>},
        {0, "fp32", "fp32", RunTypedTaskCase<float, float, float>},
        {0, "int8", "fp32", RunTypedTaskCase<float, int8_t, float>},
        {0, "fp16", "fp32", RunTypedTaskCase<float, half, float>},
        {0, "fp16", "fp16", RunTypedTaskCase<float, half, half>},
        {0, "bf16", "fp32", RunTypedTaskCase<float, bfloat16_t, float>},
        {0, "bf16", "bf16", RunTypedTaskCase<float, bfloat16_t, bfloat16_t>},
    };
    for (const auto& type : types) {
        if (taskCase.computeType == type.compute && taskCase.valueType == type.value &&
            taskCase.outputType == type.output) {
            type.run(taskCase, stats, stream, tag);
            return;
        }
    }
    ++stats.total;
    ++stats.failed;
    stats.failedCases.push_back(
        tag + " unsupported combination=" + taskCase.valueType + "/" + taskCase.outputType + "/" +
        std::to_string(taskCase.computeType));
}

int RunTaskCaseFile(const std::string& path, aclrtStream stream)
{
    std::ifstream input(path);
    if (!input.is_open()) {
        std::cerr << "[ERROR] unable to open task case file: " << path << "\n";
        return 1;
    }
    std::string line;
    std::getline(input, line);
    TestStats stats;
    while (std::getline(input, line)) {
        if (line.empty()) {
            continue;
        }
        TaskCase taskCase{};
        if (!ParseTaskCase(line, taskCase)) {
            ++stats.total;
            ++stats.failed;
            stats.failedCases.push_back("malformed CSV row: " + line);
            continue;
        }
        RunTaskCase(taskCase, stats, stream);
    }
    std::cout << "TASK_CASE_SUMMARY total=" << stats.total << " passed=" << stats.passed << " failed=" << stats.failed
              << "\n";
    for (const std::string& failure : stats.failedCases) {
        std::cout << "TASK_CASE_FAILURE " << failure << "\n";
    }
    return stats.total == 200 && stats.failed == 0 ? 0 : 1;
}

class SpmvApiValidation {
public:
    SpmvApiValidation() = default;
    SpmvApiValidation(const SpmvApiValidation&) = delete;
    SpmvApiValidation& operator=(const SpmvApiValidation&) = delete;
    ~SpmvApiValidation()
    {
        if (mat != nullptr) {
            aclsparseDestroySpMat(mat);
        }
        if (matNoOffsets != nullptr) {
            aclsparseDestroySpMat(matNoOffsets);
        }
        if (matCoo != nullptr) {
            aclsparseDestroySpMat(matCoo);
        }
        if (vecX != nullptr) {
            aclsparseDestroyDnVec(vecX);
        }
        if (vecXShort != nullptr) {
            aclsparseDestroyDnVec(vecXShort);
        }
        if (vecY != nullptr) {
            aclsparseDestroyDnVec(vecY);
        }
        if (handle != nullptr) {
            aclsparseDestroy(handle);
        }
        if (pinnedHost != nullptr) {
            aclrtFreeHost(pinnedHost);
        }
    }

    int Run(aclrtStream stream)
    {
        CHECK_RET(Allocate() == 0, return 1);
        CHECK_RET(CreateDescriptors(stream) == 0, return 1);
        CheckNullInputs();
        CheckUnsupportedInputs();
        CheckWorkspace();
        CheckDelayedBinding();
        CheckAsyncExecution(stream);
        std::cout << "API_VALIDATION_SUMMARY failed=" << failed << "\n";
        return failed == 0 ? 0 : 1;
    }

private:
    int Allocate()
    {
        CHECK_RET(rowBuffer.Upload(rowPtr) == ACL_SUCCESS, return 1);
        CHECK_RET(colBuffer.Upload(colInd) == ACL_SUCCESS, return 1);
        CHECK_RET(valuesBuffer.Upload(values) == ACL_SUCCESS, return 1);
        CHECK_RET(xBuffer.Upload(x) == ACL_SUCCESS, return 1);
        CHECK_RET(yBuffer.Upload(y) == ACL_SUCCESS, return 1);
        rowPtrDevice = rowBuffer.Get();
        colIndDevice = colBuffer.Get();
        valuesDevice = valuesBuffer.Get();
        xDevice = xBuffer.Get();
        yDevice = yBuffer.Get();
        return 0;
    }

    void Expect(const char* name, aclsparseStatus_t actual, aclsparseStatus_t expected)
    {
        const bool pass = actual == expected;
        std::cout << "API_CHECK " << name << " actual=" << actual << " expected=" << expected << " "
                  << (pass ? "PASS" : "FAIL") << "\n";
        failed += pass ? 0 : 1;
    }

    int CreateDescriptors(aclrtStream stream)
    {
        Expect("create_handle", aclsparseCreate(&handle), ACL_SPARSE_STATUS_SUCCESS);
        if (handle == nullptr) {
            return 1;
        }
        Expect("set_stream", aclsparseSetStream(handle, stream), ACL_SPARSE_STATUS_SUCCESS);
        Expect(
            "create_csr",
            aclsparseCreateCsr(
                &mat, 2, 3, 3, rowPtrDevice, colIndDevice, valuesDevice, ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_32I,
                ACL_SPARSE_INDEX_BASE_ZERO, ACL_FLOAT),
            ACL_SPARSE_STATUS_SUCCESS);
        Expect(
            "create_delayed_csr",
            aclsparseCreateCsr(
                &matNoOffsets, 2, 3, 3, nullptr, nullptr, nullptr, ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_32I,
                ACL_SPARSE_INDEX_BASE_ZERO, ACL_FLOAT),
            ACL_SPARSE_STATUS_SUCCESS);
        Expect(
            "create_coo",
            aclsparseCreateCoo(
                &matCoo, 2, 3, 3, colIndDevice, colIndDevice, valuesDevice, ACL_SPARSE_INDEX_32I,
                ACL_SPARSE_INDEX_BASE_ZERO, ACL_FLOAT),
            ACL_SPARSE_STATUS_SUCCESS);
        Expect("create_x", aclsparseCreateDnVec(&vecX, 3, xDevice, ACL_FLOAT), ACL_SPARSE_STATUS_SUCCESS);
        Expect("create_short_x", aclsparseCreateDnVec(&vecXShort, 2, xDevice, ACL_FLOAT), ACL_SPARSE_STATUS_SUCCESS);
        Expect("create_y", aclsparseCreateDnVec(&vecY, 3, yDevice, ACL_FLOAT), ACL_SPARSE_STATUS_SUCCESS);
        Expect("reject_zero_dn_stride", aclsparseDnVecSetStride(vecX, 0), ACL_SPARSE_STATUS_INVALID_VALUE);
        Expect("reject_zero_csr_stride", aclsparseCsrSetStrides(mat, 1, 0, 1), ACL_SPARSE_STATUS_INVALID_VALUE);

        return mat != nullptr && matNoOffsets != nullptr && matCoo != nullptr && vecX != nullptr &&
                       vecXShort != nullptr && vecY != nullptr ?
                   0 :
                   1;
    }

    void CheckNullInputs()
    {
        Expect(
            "null_handle",
            aclsparseSpMVGetBufferSize(
                nullptr, ACL_SPARSE_OP_NON_TRANSPOSE, &alpha, mat, vecX, &beta, vecY, ACL_FLOAT,
                ACL_SPARSE_SPMV_ALG_DEFAULT, &bufferSize),
            ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR);
        Expect(
            "null_buffer_size",
            aclsparseSpMVGetBufferSize(
                handle, ACL_SPARSE_OP_NON_TRANSPOSE, &alpha, mat, vecX, &beta, vecY, ACL_FLOAT,
                ACL_SPARSE_SPMV_ALG_DEFAULT, nullptr),
            ACL_SPARSE_STATUS_INVALID_VALUE);
        Expect(
            "null_alpha",
            aclsparseSpMVGetBufferSize(
                handle, ACL_SPARSE_OP_NON_TRANSPOSE, nullptr, mat, vecX, &beta, vecY, ACL_FLOAT,
                ACL_SPARSE_SPMV_ALG_DEFAULT, &bufferSize),
            ACL_SPARSE_STATUS_INVALID_VALUE);
    }

    void CheckUnsupportedInputs()
    {
        Expect(
            "unbound_csr_buffer_query",
            aclsparseSpMVGetBufferSize(
                handle, ACL_SPARSE_OP_NON_TRANSPOSE, &alpha, matNoOffsets, vecX, &beta, vecY, ACL_FLOAT,
                ACL_SPARSE_SPMV_ALG_DEFAULT, &bufferSize),
            ACL_SPARSE_STATUS_SUCCESS);
        Expect(
            "coo_format",
            aclsparseSpMVGetBufferSize(
                handle, ACL_SPARSE_OP_NON_TRANSPOSE, &alpha, matCoo, vecX, &beta, vecY, ACL_FLOAT,
                ACL_SPARSE_SPMV_ALG_DEFAULT, &bufferSize),
            ACL_SPARSE_STATUS_NOT_SUPPORTED);
        Expect(
            "conjugate_transpose",
            aclsparseSpMVGetBufferSize(
                handle, ACL_SPARSE_OP_CONJUGATE_TRANSPOSE, &alpha, mat, vecX, &beta, vecY, ACL_FLOAT,
                ACL_SPARSE_SPMV_ALG_DEFAULT, &bufferSize),
            ACL_SPARSE_STATUS_NOT_SUPPORTED);
        Expect(
            "unsupported_algorithm",
            aclsparseSpMVGetBufferSize(
                handle, ACL_SPARSE_OP_NON_TRANSPOSE, &alpha, mat, vecX, &beta, vecY, ACL_FLOAT,
                ACL_SPARSE_SPMV_COO_ALG1, &bufferSize),
            ACL_SPARSE_STATUS_NOT_SUPPORTED);
        Expect(
            "short_x",
            aclsparseSpMVGetBufferSize(
                handle, ACL_SPARSE_OP_NON_TRANSPOSE, &alpha, mat, vecXShort, &beta, vecY, ACL_FLOAT,
                ACL_SPARSE_SPMV_ALG_DEFAULT, &bufferSize),
            ACL_SPARSE_STATUS_INVALID_VALUE);
    }

    void CheckWorkspace()
    {
        Expect(
            "non_transpose_buffer",
            aclsparseSpMVGetBufferSize(
                handle, ACL_SPARSE_OP_NON_TRANSPOSE, &alpha, mat, vecX, &beta, vecY, ACL_FLOAT,
                ACL_SPARSE_SPMV_ALG_DEFAULT, &bufferSize),
            ACL_SPARSE_STATUS_SUCCESS);
        if (bufferSize != 0) {
            std::cout << "API_CHECK non_transpose_buffer_size actual=" << bufferSize << " expected=0 FAIL\n";
            failed++;
        } else {
            std::cout << "API_CHECK non_transpose_buffer_size actual=0 expected=0 PASS\n";
        }
        Expect(
            "transpose_buffer_query",
            aclsparseSpMVGetBufferSize(
                handle, ACL_SPARSE_OP_TRANSPOSE, &alpha, mat, vecXShort, &beta, vecY, ACL_FLOAT,
                ACL_SPARSE_SPMV_ALG_DEFAULT, &bufferSize),
            ACL_SPARSE_STATUS_SUCCESS);
        Expect(
            "transpose_preprocess_null_workspace",
            aclsparseSpMVPreprocess(
                handle, ACL_SPARSE_OP_TRANSPOSE, &alpha, mat, vecXShort, &beta, vecY, ACL_FLOAT,
                ACL_SPARSE_SPMV_ALG_DEFAULT, nullptr),
            ACL_SPARSE_STATUS_INSUFFICIENT_RESOURCES);
        Expect(
            "transpose_execute_null_workspace",
            aclsparseSpMV(
                handle, ACL_SPARSE_OP_TRANSPOSE, &alpha, mat, vecXShort, &beta, vecY, ACL_FLOAT,
                ACL_SPARSE_SPMV_ALG_DEFAULT, nullptr),
            ACL_SPARSE_STATUS_INSUFFICIENT_RESOURCES);
    }

    aclsparseStatus_t QueryDelayed(aclsparseOperation_t op)
    {
        return aclsparseSpMVGetBufferSize(
            handle, op, &alpha, matNoOffsets, vecX, &beta, vecY, ACL_FLOAT,
            ACL_SPARSE_SPMV_ALG_DEFAULT, &bufferSize);
    }

    aclsparseStatus_t ExecuteDelayed(aclsparseOperation_t op, void* workspace)
    {
        return aclsparseSpMV(
            handle, op, &alpha, matNoOffsets, vecX, &beta, vecY, ACL_FLOAT,
            ACL_SPARSE_SPMV_ALG_DEFAULT, workspace);
    }

    aclsparseStatus_t PreprocessDelayed(aclsparseOperation_t op, void* workspace)
    {
        return aclsparseSpMVPreprocess(
            handle, op, &alpha, matNoOffsets, vecX, &beta, vecY, ACL_FLOAT,
            ACL_SPARSE_SPMV_ALG_DEFAULT, workspace);
    }

    void CheckDelayedBinding()
    {
        Expect("unbind_x", aclsparseDnVecSetValues(vecX, nullptr), ACL_SPARSE_STATUS_SUCCESS);
        Expect("unbind_y", aclsparseDnVecSetValues(vecY, nullptr), ACL_SPARSE_STATUS_SUCCESS);
        const size_t boundTransposeBytes = bufferSize;
        Expect("unbound_all_transpose_query", QueryDelayed(ACL_SPARSE_OP_TRANSPOSE), ACL_SPARSE_STATUS_SUCCESS);
        if (bufferSize == 0 || bufferSize != boundTransposeBytes) {
            ++failed;
        }
        Expect("unbound_all_non_transpose_query", QueryDelayed(ACL_SPARSE_OP_NON_TRANSPOSE),
               ACL_SPARSE_STATUS_SUCCESS);
        if (bufferSize != 0) {
            ++failed;
        }
        Expect("unbound_non_transpose_noop", PreprocessDelayed(ACL_SPARSE_OP_NON_TRANSPOSE, nullptr),
               ACL_SPARSE_STATUS_SUCCESS);
        Expect("unbound_execute_rejected", ExecuteDelayed(ACL_SPARSE_OP_NON_TRANSPOSE, nullptr),
               ACL_SPARSE_STATUS_INVALID_VALUE);
        Expect("restore_x", aclsparseDnVecSetValues(vecX, xDevice), ACL_SPARSE_STATUS_SUCCESS);
        Expect("restore_y", aclsparseDnVecSetValues(vecY, yDevice), ACL_SPARSE_STATUS_SUCCESS);
        Expect("missing_csr_execute_rejected", ExecuteDelayed(ACL_SPARSE_OP_NON_TRANSPOSE, nullptr),
               ACL_SPARSE_STATUS_INVALID_VALUE);
    }

    template <typename T>
    int QueueUpload(const std::vector<T>& data, void* destination, aclrtStream stream, size_t& offset)
    {
        const size_t bytes = data.size() * sizeof(T);
        auto* source = reinterpret_cast<T*>(static_cast<uint8_t*>(pinnedHost) + offset);
        std::copy(data.begin(), data.end(), source);
        offset += bytes;
        return aclrtMemcpyAsync(destination, bytes, source, bytes, ACL_MEMCPY_HOST_TO_DEVICE, stream);
    }

    int QueueInputs(aclrtStream stream)
    {
        size_t offset = 0;
        CHECK_RET(QueueUpload(rowPtr, rowPtrDevice, stream, offset) == ACL_SUCCESS, return 1);
        CHECK_RET(QueueUpload(colInd, colIndDevice, stream, offset) == ACL_SUCCESS, return 1);
        CHECK_RET(QueueUpload(values, valuesDevice, stream, offset) == ACL_SUCCESS, return 1);
        CHECK_RET(QueueUpload(x, xDevice, stream, offset) == ACL_SUCCESS, return 1);
        CHECK_RET(QueueUpload(y, yDevice, stream, offset) == ACL_SUCCESS, return 1);
        return 0;
    }

    void CheckIndexOnlyPreprocess(aclsparseOperation_t op, void* workspace)
    {
        Expect("unbind_matrix_values", aclsparseSpMatSetValues(matNoOffsets, nullptr), ACL_SPARSE_STATUS_SUCCESS);
        Expect("unbind_preprocess_x", aclsparseDnVecSetValues(vecX, nullptr), ACL_SPARSE_STATUS_SUCCESS);
        Expect("unbind_preprocess_y", aclsparseDnVecSetValues(vecY, nullptr), ACL_SPARSE_STATUS_SUCCESS);
        Expect("index_only_preprocess", PreprocessDelayed(op, workspace), ACL_SPARSE_STATUS_SUCCESS);
        Expect("bind_matrix_values", aclsparseSpMatSetValues(matNoOffsets, valuesDevice), ACL_SPARSE_STATUS_SUCCESS);
        Expect("bind_execute_x", aclsparseDnVecSetValues(vecX, xDevice), ACL_SPARSE_STATUS_SUCCESS);
        Expect("bind_execute_y", aclsparseDnVecSetValues(vecY, yDevice), ACL_SPARSE_STATUS_SUCCESS);
    }

    void CheckAsyncCase(aclsparseOperation_t op, bool preprocess, void* workspace, aclrtStream stream)
    {
        Expect("bind_delayed_csr", aclsparseCsrSetPointers(matNoOffsets, rowPtrDevice, colIndDevice, valuesDevice),
               ACL_SPARSE_STATUS_SUCCESS);
        const int uploadStatus = QueueInputs(stream);
        aclsparseStatus_t executeStatus = ACL_SPARSE_STATUS_INVALID_VALUE;
        if (uploadStatus == 0) {
            // No synchronization between H2D, optional preprocess and execute.
            if (preprocess) {
                CheckIndexOnlyPreprocess(op, workspace);
            }
            executeStatus = ExecuteDelayed(op, workspace);
            Expect("async_execute", executeStatus, ACL_SPARSE_STATUS_SUCCESS);
        }
        const int syncStatus = aclrtSynchronizeStream(stream);
        std::vector<float> result(y.size());
        const int copyStatus = yBuffer.Download(result);
        const std::vector<float> expected =
            op == ACL_SPARSE_OP_TRANSPOSE ? std::vector<float>{1.0f, 6.0f, 2.0f} :
                                           std::vector<float>{7.0f, 6.0f, 6.0f};
        const bool pass = uploadStatus == 0 && executeStatus == ACL_SPARSE_STATUS_SUCCESS &&
                          syncStatus == ACL_SUCCESS && copyStatus == ACL_SUCCESS && result == expected;
        failed += pass ? 0 : 1;
        std::cout << "ASYNC_CASE transpose=" << (op == ACL_SPARSE_OP_TRANSPOSE)
                  << " preprocess=" << preprocess << " result=" << (pass ? "PASS" : "FAIL") << "\n";
    }

    void CheckAsyncExecution(aclrtStream stream)
    {
        Expect("delayed_transpose_workspace", QueryDelayed(ACL_SPARSE_OP_TRANSPOSE), ACL_SPARSE_STATUS_SUCCESS);
        SpmvDeviceBuffer workspace;
        const size_t hostBytes = (rowPtr.size() + colInd.size()) * sizeof(int32_t) +
                                 (values.size() + x.size() + y.size()) * sizeof(float);
        if (workspace.Allocate(bufferSize) != ACL_SUCCESS ||
            aclrtMallocHost(&pinnedHost, hostBytes) != ACL_SUCCESS) {
            ++failed;
            return;
        }
        Expect("missing_csr_preprocess_rejected", PreprocessDelayed(ACL_SPARSE_OP_TRANSPOSE, workspace.Get()),
               ACL_SPARSE_STATUS_INVALID_VALUE);
        for (const auto op : {ACL_SPARSE_OP_NON_TRANSPOSE, ACL_SPARSE_OP_TRANSPOSE}) {
            for (const bool preprocess : {false, true}) {
                CheckAsyncCase(op, preprocess, workspace.Get(), stream);
            }
        }
    }

    std::vector<int32_t> rowPtr{0, 2, 3};
    std::vector<int32_t> colInd{0, 2, 1};
    std::vector<float> values{1.0f, 2.0f, 3.0f};
    std::vector<float> x{1.0f, 2.0f, 3.0f};
    std::vector<float> y{4.0f, 5.0f, 6.0f};

    aclsparseHandle_t handle = nullptr;
    aclsparseSpMatDescr_t mat = nullptr;
    aclsparseSpMatDescr_t matNoOffsets = nullptr;
    void* pinnedHost = nullptr;
    aclsparseSpMatDescr_t matCoo = nullptr;
    aclsparseDnVecDescr_t vecX = nullptr;
    aclsparseDnVecDescr_t vecXShort = nullptr;
    aclsparseDnVecDescr_t vecY = nullptr;
    int failed = 0;
    float alpha = 1.0f;
    float beta = 0.0f;
    size_t bufferSize = 0;
    SpmvDeviceBuffer rowBuffer, colBuffer, valuesBuffer, xBuffer, yBuffer;
    void *rowPtrDevice = nullptr, *colIndDevice = nullptr, *valuesDevice = nullptr;
    void *xDevice = nullptr, *yDevice = nullptr;
};

int RunApiValidationSuite(aclrtStream stream)
{
    SpmvApiValidation suite;
    return suite.Run(stream);
}

template <typename T>
std::vector<T> ExpandWithStride(const std::vector<T>& logical, int64_t stride, T sentinel)
{
    if (logical.empty()) {
        return {};
    }
    std::vector<T> physical((logical.size() - 1u) * static_cast<size_t>(stride) + 1u, sentinel);
    for (size_t i = 0; i < logical.size(); ++i) {
        physical[i * static_cast<size_t>(stride)] = logical[i];
    }
    return physical;
}

template <typename CompT, typename ValT, typename OutT>
SpmvTestData<CompT, ValT, OutT> GenerateStridedData(bool transpose)
{
    SpmvTestData<CompT, ValT, OutT> data;
    data.rows = 3;
    data.cols = 4;
    data.nnz = 6;
    data.transpose = transpose;
    data.alpha = static_cast<CompT>(1);
    data.beta = static_cast<CompT>(1);
    data.rowOffsets = {0, 2, 3, 6};
    data.colIndices = {0, 2, 1, 0, 2, 3};
    data.values = {static_cast<ValT>(1.0f), static_cast<ValT>(2.0f), static_cast<ValT>(3.0f),
                   static_cast<ValT>(4.0f), static_cast<ValT>(5.0f), static_cast<ValT>(6.0f)};
    data.x = transpose ? std::vector<ValT>{static_cast<ValT>(2.0f), static_cast<ValT>(3.0f), static_cast<ValT>(4.0f)} :
                         std::vector<ValT>{
                             static_cast<ValT>(10.0f), static_cast<ValT>(20.0f), static_cast<ValT>(30.0f),
                             static_cast<ValT>(40.0f)};
    data.y =
        transpose ?
            std::vector<OutT>{
                static_cast<OutT>(5.0f), static_cast<OutT>(6.0f), static_cast<OutT>(7.0f), static_cast<OutT>(8.0f)} :
            std::vector<OutT>{static_cast<OutT>(1.0f), static_cast<OutT>(2.0f), static_cast<OutT>(3.0f)};
    data.xSize = data.x.size();
    data.ySize = data.y.size();
    return data;
}

template <typename CompT, typename ValT, typename OutT>
void ApplyTestStrides(SpmvTestData<CompT, ValT, OutT>& data)
{
    data.rowStride = 2;
    data.colStride = 3;
    data.valueStride = 2;
    data.xStride = 4;
    data.yStride = 3;
    constexpr int32_t kIndexSentinel = -777777;
    const ValT dataSentinel = static_cast<ValT>(-7.0f);
    const OutT ySentinel = static_cast<OutT>(-9.0f);
    data.rowOffsets = ExpandWithStride(data.rowOffsets, data.rowStride, kIndexSentinel);
    data.colIndices = ExpandWithStride(data.colIndices, data.colStride, kIndexSentinel);
    data.values = ExpandWithStride(data.values, data.valueStride, dataSentinel);
    data.x = ExpandWithStride(data.x, data.xStride, dataSentinel);
    data.y = ExpandWithStride(data.y, data.yStride, ySentinel);
}

template <typename OutT>
int VerifyStridedOutput(const std::vector<OutT>& expected, const std::vector<OutT>& physical, int64_t stride)
{
    if (stride <= 0) {
        LOG_PRINT("[ERROR] Invalid output stride.\n");
        return 1;
    }
    std::vector<OutT> logical(expected.size());
    for (size_t i = 0; i < logical.size(); ++i) {
        logical[i] = physical[i * static_cast<size_t>(stride)];
    }
    float maxError = 0.0f;
    float matchedRatio = 0.0f;
    int failed = Verification<OutT>(expected, logical, maxError, matchedRatio, GetMixedToleranceParams<OutT>());
    const OutT sentinel = static_cast<OutT>(-9.0f);
    for (size_t i = 0; i < physical.size(); ++i) {
        if (i % static_cast<size_t>(stride) != 0u && std::memcmp(&physical[i], &sentinel, sizeof(OutT)) != 0) {
            LOG_PRINT("[ERROR] Strided Y sentinel[%zu] was overwritten.\n", i);
            ++failed;
        }
    }
    return failed == 0 ? 0 : 1;
}

template <typename CompT, typename ValT, typename OutT>
int RunStridedCase(bool transpose, aclrtStream stream)
{
    auto data = GenerateStridedData<CompT, ValT, OutT>(transpose);
    const std::vector<OutT> expected = data.Reference();
    ApplyTestStrides(data);
    SpmvTestContext<CompT, ValT, OutT> context(data, stream);
    CHECK_RET(context.Initialize() == 0, return 1);
    CHECK_RET(context.CheckStrides() == 0, LOG_PRINT("[ERROR] Stride getter validation failed.\n"); return 1);
    CHECK_RET(context.Prepare() == 0, return 1);
    CHECK_RET(context.Execute() == ACL_SPARSE_STATUS_SUCCESS, return 1);
    std::vector<OutT> result(data.y.size());
    CHECK_RET(context.Read(result) == 0, return 1);
    const int failed = VerifyStridedOutput(expected, result, data.yStride);
    std::cout << "STRIDED_CASE op=" << (transpose ? "transpose" : "non_transpose")
              << " input_dtype=" << AclTypeOf<ValT>() << " output_dtype=" << AclTypeOf<OutT>()
              << " compute_dtype=" << AclTypeOf<CompT>() << " row_stride=" << data.rowStride
              << " col_stride=" << data.colStride << " values_stride=" << data.valueStride
              << " x_stride=" << data.xStride << " y_stride=" << data.yStride
              << " result=" << (failed == 0 ? "PASS" : "FAIL") << "\n";
    return failed;
}

int RunStridedSuite(aclrtStream stream)
{
    int failed = 0;
    auto runPair = [&](auto compTag, auto valTag, auto outTag) {
        using CompT = decltype(compTag);
        using ValT = decltype(valTag);
        using OutT = decltype(outTag);
        failed += RunStridedCase<CompT, ValT, OutT>(false, stream) != 0 ? 1 : 0;
        failed += RunStridedCase<CompT, ValT, OutT>(true, stream) != 0 ? 1 : 0;
    };
    runPair(float{}, float{}, float{});
    runPair(int32_t{}, int8_t{}, int32_t{});
    runPair(float{}, int8_t{}, float{});
    runPair(float{}, half{}, float{});
    runPair(float{}, half{}, half{});
    runPair(float{}, bfloat16_t{}, float{});
    runPair(float{}, bfloat16_t{}, bfloat16_t{});
    constexpr int total = 14;
    std::cout << "STRIDED_SUMMARY total=" << total << " passed=" << total - failed << " failed=" << failed << "\n";
    return failed == 0 ? 0 : 1;
}

struct IntegerOverflowCase {
    const char* name;
    int64_t dot;
    int32_t alpha;
    int32_t beta;
    int32_t oldY;
    int32_t expected;
};

SpmvTestData<int32_t, int8_t, int32_t> MakeIntegerOverflowData(const IntegerOverflowCase& testCase, bool transpose)
{
    SpmvTestData<int32_t, int8_t, int32_t> data;
    const int64_t magnitude = testCase.dot < 0 ? -testCase.dot : testCase.dot;
    const int32_t sign = testCase.dot < 0 ? -1 : 1;
    data.values.assign(static_cast<size_t>(magnitude / (127 * 127)), static_cast<int8_t>(sign * 127));
    data.x.assign(data.values.size(), static_cast<int8_t>(127));
    const int32_t remainder = static_cast<int32_t>(magnitude % (127 * 127));
    // At most two tail entries represent the remaining dot product exactly.
    if (remainder / 127 != 0) {
        data.values.push_back(static_cast<int8_t>(sign * (remainder / 127)));
        data.x.push_back(static_cast<int8_t>(127));
    }
    if (remainder % 127 != 0) {
        data.values.push_back(static_cast<int8_t>(sign * (remainder % 127)));
        data.x.push_back(static_cast<int8_t>(1));
    }
    data.nnz = data.values.size();
    data.rows = transpose ? data.nnz : 2;
    data.cols = transpose ? 2 : data.nnz;
    data.transpose = transpose;
    data.alpha = testCase.alpha;
    data.beta = testCase.beta;
    data.y = {testCase.oldY, 0};
    data.xSize = data.x.size();
    data.ySize = data.y.size();
    data.colIndices.resize(data.nnz, 0);
    data.rowOffsets.resize(data.rows + 1);
    if (transpose) {
        std::iota(data.rowOffsets.begin(), data.rowOffsets.end(), 0);
    } else {
        data.rowOffsets = {0, static_cast<int32_t>(data.nnz), static_cast<int32_t>(data.nnz)};
        std::iota(data.colIndices.begin(), data.colIndices.end(), 0);
    }
    return data;
}

int RunIntegerOverflowCase(const IntegerOverflowCase& testCase, bool transpose, bool strided,
                           bool deviceScalars, aclrtStream stream)
{
    auto data = MakeIntegerOverflowData(testCase, transpose);
    const std::vector<int32_t> expected{testCase.expected, 0};
    CHECK_RET(data.Reference() == expected, LOG_PRINT("[ERROR] Integer overflow CPU oracle mismatch.\n"); return 1);
    if (strided) {
        ApplyTestStrides(data);
    }
    SpmvTestContext<int32_t, int8_t, int32_t> context(data, stream);
    CHECK_RET(context.Initialize(deviceScalars) == 0, return 1);
    CHECK_RET(context.CheckStrides() == 0, return 1);
    CHECK_RET(context.Prepare() == 0, return 1);
    CHECK_RET(context.Execute() == ACL_SPARSE_STATUS_SUCCESS, return 1);
    std::vector<int32_t> result(data.y.size());
    CHECK_RET(context.Read(result) == 0, return 1);
    const auto expectedPhysical = ExpandWithStride(expected, data.yStride, static_cast<int32_t>(-9));
    const bool pass = result == expectedPhysical;
    std::cout << "INTEGER_OVERFLOW_CASE name=" << testCase.name << " transpose=" << transpose
              << " strided=" << strided << " device_scalars=" << deviceScalars
              << " expected=" << testCase.expected << " actual=" << result[0]
              << " result=" << (pass ? "PASS" : "FAIL") << "\n";
    return pass ? 0 : 1;
}

int RunIntegerOverflowSuite(aclrtStream stream)
{
    constexpr int32_t max = std::numeric_limits<int32_t>::max();
    constexpr int32_t min = std::numeric_limits<int32_t>::min();
    const IntegerOverflowCase cases[] = {
        {"review-positive-product", 4294975281LL, max, 0, 123, max},
        {"negative-dot-product", -4294975281LL, max, 0, 0, min},
        {"negative-alpha-product", 4294975281LL, min, 0, 0, min},
        {"both-negative-product", -4294975281LL, min, 0, 0, max},
        {"positive-add-overflow", 4294967296LL, max, max, max, max},
        {"negative-add-overflow", -4294967296LL, max, min, max, min},
        {"overflow-with-opposite-beta", 4294975281LL, max, min, max, max},
        {"exact-int64-min-product", 4294967296LL, min, max, max, min},
        {"large-cancellation-zero", 2147483648LL, max, min, max, 0},
        {"large-cancellation-plus-one", 2147483647LL, max, min, max - 1, 1},
        {"large-cancellation-minus-one", -2147483647LL, max, min, -(max - 1), -1},
        {"alpha-zero-beta-positive", 4294975281LL, 0, min, min, max},
        {"alpha-zero-beta-negative", 0, 0, min, max, min},
        {"int32-max-boundary", 1, max, 0, 0, max},
        {"int32-min-boundary", 1, min, 0, 0, min},
        {"small-cancellation", 2, max, -2, max, 0},
    };
    int failed = 0;
    int total = 0;
    for (const auto& testCase : cases) {
        for (const bool transpose : {false, true}) {
            for (const bool strided : {false, true}) {
                for (const bool deviceScalars : {false, true}) {
                    ++total;
                    failed += RunIntegerOverflowCase(testCase, transpose, strided, deviceScalars, stream) != 0 ? 1 : 0;
                }
            }
        }
    }
    std::cout << "INTEGER_OVERFLOW_SUMMARY total=" << total << " passed=" << total - failed
              << " failed=" << failed << "\n";
    return failed == 0 ? 0 : 1;
}

void RunDeviceScalarCases(TestStats& stats, aclrtStream stream)
{
    std::cout << "======== Device scalar / determinism / algorithm acceptance "
                 "========\n";
    RunAndTrack<float>(
        257, 509, 0.83f, 1.25f, -0.5f, false, stats, stream, "device-f32-alg1-deterministic", true, true,
        ACL_SPARSE_SPMV_CSR_ALG1);
    RunAndTrack<float>(
        509, 257, 0.87f, -0.75f, 0.25f, true, stats, stream, "device-trans-f32-alg2-deterministic", true, true,
        ACL_SPARSE_SPMV_CSR_ALG2);
    RunAndTrack<int32_t>(
        257, 509, 0.91f, 3, -2, false, stats, stream, "device-int32-deterministic", true, true,
        ACL_SPARSE_SPMV_ALG_DEFAULT);
    RunAndTrack<float, int8_t, float>(
        257, 509, 0.88f, 1.25f, -0.5f, true, stats, stream, "device-trans-int8-f32-deterministic", true, true,
        ACL_SPARSE_SPMV_CSR_ALG1);
    RunAndTrack<float, half, float>(
        509, 257, 0.86f, -0.75f, 0.25f, false, stats, stream, "host-f16-f32-deterministic", false, true,
        ACL_SPARSE_SPMV_CSR_ALG2);
    RunAndTrack<float, half, half>(
        257, 509, 0.86f, 1.5f, 0.5f, false, stats, stream, "host-f16-output-deterministic", false, true,
        ACL_SPARSE_SPMV_ALG_DEFAULT);
    RunAndTrack<float, bfloat16_t, float>(
        257, 509, 0.89f, 1.25f, -0.5f, false, stats, stream, "host-bf16-f32-deterministic", false, true,
        ACL_SPARSE_SPMV_CSR_ALG1);
    RunAndTrack<float, bfloat16_t, bfloat16_t>(
        509, 257, 0.89f, -1.0f, 1.0f, true, stats, stream, "host-trans-bf16-output-deterministic", false, true,
        ACL_SPARSE_SPMV_ALG_DEFAULT);

}

int RunAcceptanceSuite(aclrtStream stream)
{
    TestStats stats;
    RunDeviceScalarCases(stats, stream);
    std::cout << "======== Empty / zero-NNZ / scalar / non-finite boundary "
                 "acceptance ========\n";
    RunAndTrack<float>(0, 128, 1.0f, 1.0f, 0.0f, false, stats, stream, "empty-output-m0");
    RunAndTrack<float>(128, 0, 1.0f, 1.0f, 2.0f, false, stats, stream, "empty-input-n0");
    RunAndTrack<float>(128, 128, 1.0f, 1.0f, -0.5f, true, stats, stream, "zero-nnz-transpose");
    RunAndTrack<float>(1, 1, 0.0f, 1.0f, 0.0f, false, stats, stream, "scalar-1x1");
    RunAndTrack<float>(
        1, 1, 0.0f, 1.0f, 0.0f, false, stats, stream, "positive-infinity", false, true, ACL_SPARSE_SPMV_ALG_DEFAULT,
        SpecialValueKind::PositiveInfinity);
    RunAndTrack<float>(
        1, 1, 0.0f, 1.0f, 0.0f, false, stats, stream, "negative-infinity", false, true, ACL_SPARSE_SPMV_ALG_DEFAULT,
        SpecialValueKind::NegativeInfinity);
    RunAndTrack<float>(
        1, 1, 0.0f, 1.0f, 0.0f, false, stats, stream, "nan", false, true, ACL_SPARSE_SPMV_ALG_DEFAULT,
        SpecialValueKind::NaN);

    const int stridedRet = RunStridedSuite(stream);
    const int apiRet = RunApiValidationSuite(stream);
    const int integerRet = RunIntegerOverflowSuite(stream);
    std::cout << "ACCEPTANCE_SUMMARY total=" << stats.total << " passed=" << stats.passed << " failed=" << stats.failed
              << " strided=" << (stridedRet == 0 ? "PASS" : "FAIL") << " api=" << (apiRet == 0 ? "PASS" : "FAIL")
              << " integer_overflow=" << (integerRet == 0 ? "PASS" : "FAIL")
              << "\n";
    return stats.failed == 0 && stridedRet == 0 && apiRet == 0 && integerRet == 0 ? 0 : 1;
}

// ===================== main =====================

void RunFloatBasicSuite(TestStats& stats, aclrtStream stream)
{
    // ============ Float 基础用例 ============
    std::cout << "======== Float Basic Tests (alpha=1.0, beta=0.0) ========\n";
    RunAndTrack<float>(512, 1024, 0.9f, 1.0f, 0.0f, false, stats, stream, "float-default");
    RunAndTrack<float>(512, 1024, 0.0f, 1.0f, 0.0f, false, stats, stream, "float-dense");
    RunAndTrack<float>(512, 1024, 0.999f, 1.0f, 0.0f, false, stats, stream, "float-sparse");
    RunAndTrack<float>(10, 1024, 0.999f, 1.0f, 0.0f, false, stats, stream, "float-fewrow");
    RunAndTrack<float>(1, 1024, 0.9f, 1.0f, 0.0f, false, stats, stream, "float-1row");
    RunAndTrack<float>(512, 8192, 0.1f, 1.0f, 0.0f, false, stats, stream, "float-wide");

    // ============ Float Alpha/Beta 组合测试 ============
    std::cout << "\n======== Float Alpha/Beta Tests ========\n";
    RunAndTrack<float>(512, 1024, 0.9f, 2.0f, 0.0f, false, stats, stream,
                       "float-alpha2"); // y = 2*A*x
    RunAndTrack<float>(512, 1024, 0.9f, 1.0f, 1.0f, false, stats, stream,
                       "float-beta1"); // y = A*x + y
    RunAndTrack<float>(512, 1024, 0.9f, 0.5f, 0.5f, false, stats, stream,
                       "float-half"); // y = 0.5*A*x + 0.5*y
    RunAndTrack<float>(512, 1024, 0.9f, -1.0f, 0.0f, false, stats, stream,
                       "float-negA"); // y = -A*x
    RunAndTrack<float>(256, 2048, 0.95f, 1.5f, 0.2f, false, stats, stream,
                       "float-mix"); // y = 1.5*A*x + 0.2*y

    // ============ Float Alpha=0, beta=1 (纯 pass-through) ============
    RunAndTrack<float>(512, 1024, 0.9f, 0.0f, 1.0f, false, stats, stream, "float-betaOnly");
    RunAndTrack<float>(512, 1024, 0.0f, 0.0f, 2.0f, false, stats, stream, "float-beta2");

    // ============ Float 随机参数采样 ============
    RunRandomParams<float>(
        30, "float-rand-", false, stats, stream, std::uniform_real_distribution<float>(-2.0f, 2.0f),
        std::uniform_real_distribution<float>(-1.0f, 1.0f));
}

void RunIntegerBasicSuite(TestStats& stats, aclrtStream stream)
{
    // ============ Int32 基础用例 ============
    std::cout << "\n======== Int32 Basic Tests (alpha=1, beta=0) ========\n";
    RunAndTrack<int32_t>(512, 1024, 0.9f, 1, 0, false, stats, stream, "int32-default");
    RunAndTrack<int32_t>(512, 1024, 0.0f, 1, 0, false, stats, stream, "int32-dense");
    RunAndTrack<int32_t>(512, 1024, 0.999f, 1, 0, false, stats, stream, "int32-sparse");
    RunAndTrack<int32_t>(10, 1024, 0.999f, 1, 0, false, stats, stream, "int32-fewrow");
    RunAndTrack<int32_t>(1, 1024, 0.9f, 1, 0, false, stats, stream, "int32-1row");

    // ============ Int32 Alpha/Beta 组合测试 ============
    std::cout << "\n======== Int32 Alpha/Beta Tests ========\n";
    RunAndTrack<int32_t>(512, 1024, 0.9f, 2, 0, false, stats, stream,
                         "int32-alpha2"); // y = 2*A*x
    RunAndTrack<int32_t>(512, 1024, 0.9f, 1, 1, false, stats, stream,
                         "int32-beta1"); // y = A*x + y
    RunAndTrack<int32_t>(512, 1024, 0.9f, -1, 0, false, stats, stream,
                         "int32-negA"); // y = -A*x
    RunAndTrack<int32_t>(256, 2048, 0.95f, 3, -2, false, stats, stream,
                         "int32-mix"); // y = 3*A*x - 2*y

    // ============ Int32 随机参数采样 ============
    RunRandomParams<int32_t>(
        30, "int32-rand-", false, stats, stream, std::uniform_int_distribution<int>(-5, 5),
        std::uniform_int_distribution<int>(-5, 5));
}

void RunFloatTransposeSuite(TestStats& stats, aclrtStream stream)
{
    // ====================== Transpose 测试 ======================
    std::cout << "\n"
              << "###########################################################\n"
              << "##            Transpose SpMV Test Suite                   ##\n"
              << "###########################################################\n\n";

    // ============ Float Transpose 基础用例 ============
    std::cout << "======== Float Transpose Basic Tests (alpha=1.0, beta=0.0) "
                 "========\n";
    RunAndTrack<float>(512, 1024, 0.9f, 1.0f, 0.0f, true, stats, stream, "trans-float-default");
    RunAndTrack<float>(512, 1024, 0.0f, 1.0f, 0.0f, true, stats, stream, "trans-float-dense");
    RunAndTrack<float>(512, 1024, 0.999f, 1.0f, 0.0f, true, stats, stream, "trans-float-sparse");
    RunAndTrack<float>(10, 1024, 0.999f, 1.0f, 0.0f, true, stats, stream, "trans-float-fewrow");
    RunAndTrack<float>(1, 1024, 0.9f, 1.0f, 0.0f, true, stats, stream, "trans-float-1row");
    RunAndTrack<float>(512, 8192, 0.1f, 1.0f, 0.0f, true, stats, stream, "trans-float-wide");
    // 方阵
    RunAndTrack<float>(1024, 1024, 0.9f, 1.0f, 0.0f, true, stats, stream, "trans-float-square");
    // 行列不等，交换测试 —— 用于覆盖 M < N 和 M > N 场景
    RunAndTrack<float>(2048, 512, 0.9f, 1.0f, 0.0f, true, stats, stream, "trans-float-mgreater");

    // ============ Float Transpose Alpha/Beta 组合 ============
    std::cout << "\n======== Float Transpose Alpha/Beta Tests ========\n";
    RunAndTrack<float>(512, 1024, 0.9f, 2.0f, 0.0f, true, stats, stream, "trans-float-alpha2");
    RunAndTrack<float>(512, 1024, 0.9f, 1.0f, 1.0f, true, stats, stream, "trans-float-beta1");
    RunAndTrack<float>(512, 1024, 0.9f, 0.5f, 0.5f, true, stats, stream, "trans-float-half");
    RunAndTrack<float>(512, 1024, 0.9f, -1.0f, 0.0f, true, stats, stream, "trans-float-negA");
    RunAndTrack<float>(256, 2048, 0.95f, 1.5f, 0.2f, true, stats, stream, "trans-float-mix");
    RunAndTrack<float>(512, 1024, 0.9f, 0.0f, 1.0f, true, stats, stream, "trans-float-betaOnly");
    RunAndTrack<float>(512, 1024, 0.0f, 0.0f, 2.0f, true, stats, stream, "trans-float-beta2");

    // ============ Float Transpose 随机参数采样 ============
    RunRandomParams<float>(
        20, "trans-float-rand-", true, stats, stream, std::uniform_real_distribution<float>(-2.0f, 2.0f),
        std::uniform_real_distribution<float>(-1.0f, 1.0f));
}

void RunIntegerTransposeSuite(TestStats& stats, aclrtStream stream)
{
    // ============ Int32 Transpose 基础用例 ============
    std::cout << "\n======== Int32 Transpose Basic Tests (alpha=1, beta=0) ========\n";
    RunAndTrack<int32_t>(512, 1024, 0.9f, 1, 0, true, stats, stream, "trans-int32-default");
    RunAndTrack<int32_t>(512, 1024, 0.0f, 1, 0, true, stats, stream, "trans-int32-dense");
    RunAndTrack<int32_t>(512, 1024, 0.999f, 1, 0, true, stats, stream, "trans-int32-sparse");
    RunAndTrack<int32_t>(10, 1024, 0.999f, 1, 0, true, stats, stream, "trans-int32-fewrow");
    RunAndTrack<int32_t>(1, 1024, 0.9f, 1, 0, true, stats, stream, "trans-int32-1row");
    RunAndTrack<int32_t>(1024, 1024, 0.9f, 1, 0, true, stats, stream, "trans-int32-square");
    RunAndTrack<int32_t>(2048, 512, 0.9f, 1, 0, true, stats, stream, "trans-int32-mgreater");

    // ============ Int32 Transpose Alpha/Beta 组合 ============
    std::cout << "\n======== Int32 Transpose Alpha/Beta Tests ========\n";
    RunAndTrack<int32_t>(512, 1024, 0.9f, 2, 0, true, stats, stream, "trans-int32-alpha2");
    RunAndTrack<int32_t>(512, 1024, 0.9f, 1, 1, true, stats, stream, "trans-int32-beta1");
    RunAndTrack<int32_t>(512, 1024, 0.9f, -1, 0, true, stats, stream, "trans-int32-negA");
    RunAndTrack<int32_t>(256, 2048, 0.95f, 3, -2, true, stats, stream, "trans-int32-mix");

    // ============ Int32 Transpose 随机参数采样 ============
    RunRandomParams<int32_t>(
        20, "trans-int32-rand-", true, stats, stream, std::uniform_int_distribution<int>(-3, 5),
        std::uniform_int_distribution<int>(-2, 3));
}

void RunHalfBasicSuite(TestStats& stats, aclrtStream stream)
{
    // ====================== 混合精度用例 ======================
    std::cout << "\n"
              << "###########################################################\n"
              << "##              Mixed-Precision Test Suite               ##\n"
              << "###########################################################\n\n";

    // ---- Float16→Float32 Non-Transpose ----
    std::cout << "======== Float16→Float32 Non-Transpose ========\n";
    RunAndTrack<float, half, float>(256, 512, 0.9f, 1.0f, 0.0f, false, stats, stream, "f16-f32-default");
    RunAndTrack<float, half, float>(512, 1024, 0.9f, 2.0f, 0.0f, false, stats, stream, "f16-f32-alpha2");
    RunAndTrack<float, half, float>(512, 1024, 0.9f, 1.0f, 1.0f, false, stats, stream, "f16-f32-beta1");
    RunAndTrack<float, half, float>(256, 2048, 0.95f, 1.5f, 0.2f, false, stats, stream, "f16-f32-mix");
    RunAndTrack<float, half, float>(512, 1024, 0.999f, 1.0f, 0.0f, false, stats, stream, "f16-f32-sparse");
    RunAndTrack<float, half, float>(512, 1024, 0.1f, 1.0f, 0.0f, false, stats, stream, "f16-f32-dense");
    RunAndTrack<float, half, float>(10, 1024, 0.9f, 1.0f, 0.0f, false, stats, stream, "f16-f32-fewrow");
    RunAndTrack<float, half, float>(1, 1024, 0.9f, 1.0f, 0.0f, false, stats, stream, "f16-f32-1row");
}

void RunHalfTransposeSuite(TestStats& stats, aclrtStream stream)
{
    // ---- Float16→Float32 Transpose ----
    std::cout << "\n======== Float16→Float32 Transpose ========\n";
    RunAndTrack<float, half, float>(256, 512, 0.9f, 1.0f, 0.0f, true, stats, stream, "trans-f16-f32-def");
    RunAndTrack<float, half, float>(512, 1024, 0.9f, 2.0f, 0.0f, true, stats, stream, "trans-f16-f32-alpha2");
    RunAndTrack<float, half, float>(512, 1024, 0.9f, 1.0f, 1.0f, true, stats, stream, "trans-f16-f32-beta1");
    RunAndTrack<float, half, float>(1024, 1024, 0.9f, 1.0f, 0.0f, true, stats, stream, "trans-f16-f32-square");
    RunAndTrack<float, half, float>(2048, 512, 0.9f, 1.0f, 0.0f, true, stats, stream, "trans-f16-f32-mgreater");
    RunAndTrack<float, half, float>(512, 1024, 0.999f, 1.0f, 0.0f, true, stats, stream, "trans-f16-f32-sparse");
    RunAndTrack<float, half, float>(512, 1024, 0.1f, 0.0f, 1.0f, true, stats, stream, "trans-f16-f32-betaOnly");
    RunAndTrack<float, half, float>(10, 1024, 0.9f, 1.0f, 0.0f, true, stats, stream, "trans-f16-f32-fewrow");
}

void RunBfloatBasicSuite(TestStats& stats, aclrtStream stream)
{
    // ---- BF16→Float32 Non-Transpose ----
    std::cout << "\n======== BF16→Float32 Non-Transpose ========\n";
    RunAndTrack<float, bfloat16_t, float>(256, 512, 0.9f, 1.0f, 0.0f, false, stats, stream, "bf16-f32-default");
    RunAndTrack<float, bfloat16_t, float>(512, 1024, 0.9f, 2.0f, 0.0f, false, stats, stream, "bf16-f32-alpha2");
    RunAndTrack<float, bfloat16_t, float>(512, 1024, 0.9f, 1.0f, 1.0f, false, stats, stream, "bf16-f32-beta1");
    RunAndTrack<float, bfloat16_t, float>(256, 2048, 0.95f, 1.5f, 0.2f, false, stats, stream, "bf16-f32-mix");
    RunAndTrack<float, bfloat16_t, float>(512, 1024, 0.999f, 1.0f, 0.0f, false, stats, stream, "bf16-f32-sparse");
    RunAndTrack<float, bfloat16_t, float>(512, 1024, 0.1f, 1.0f, 0.0f, false, stats, stream, "bf16-f32-dense");
    RunAndTrack<float, bfloat16_t, float>(10, 1024, 0.9f, 1.0f, 0.0f, false, stats, stream, "bf16-f32-fewrow");
    RunAndTrack<float, bfloat16_t, float>(1, 1024, 0.9f, 1.0f, 0.0f, false, stats, stream, "bf16-f32-1row");
}

void RunBfloatTransposeSuite(TestStats& stats, aclrtStream stream)
{
    // ---- BF16→Float32 Transpose ----
    std::cout << "\n======== BF16→Float32 Transpose ========\n";
    RunAndTrack<float, bfloat16_t, float>(256, 512, 0.9f, 1.0f, 0.0f, true, stats, stream, "trans-bf16-f32-def");
    RunAndTrack<float, bfloat16_t, float>(512, 1024, 0.9f, 2.0f, 0.0f, true, stats, stream, "trans-bf16-f32-alpha2");
    RunAndTrack<float, bfloat16_t, float>(512, 1024, 0.9f, 1.0f, 1.0f, true, stats, stream, "trans-bf16-f32-beta1");
    RunAndTrack<float, bfloat16_t, float>(1024, 1024, 0.9f, 1.0f, 0.0f, true, stats, stream, "trans-bf16-f32-square");
    RunAndTrack<float, bfloat16_t, float>(2048, 512, 0.9f, 1.0f, 0.0f, true, stats, stream, "trans-bf16-f32-mgreater");
    RunAndTrack<float, bfloat16_t, float>(512, 1024, 0.999f, 1.0f, 0.0f, true, stats, stream, "trans-bf16-f32-sparse");
    RunAndTrack<float, bfloat16_t, float>(512, 1024, 0.1f, 0.0f, 1.0f, true, stats, stream, "trans-bf16-f32-betaOnly");
    RunAndTrack<float, bfloat16_t, float>(10, 1024, 0.9f, 1.0f, 0.0f, true, stats, stream, "trans-bf16-f32-fewrow");
}

void RunInt8FloatSuite(TestStats& stats, aclrtStream stream)
{
    // ---- Int8→Float32 ----
    std::cout << "\n======== Int8→Float32 ========\n";
    RunAndTrack<float, int8_t, float>(512, 1024, 0.9f, 1.0f, 0.0f, false, stats, stream, "i8-f32-default");
    RunAndTrack<float, int8_t, float>(512, 1024, 0.5f, 0.5f, 1.0f, false, stats, stream, "i8-f32-dense");
    RunAndTrack<float, int8_t, float>(512, 1024, 0.999f, 2.0f, 0.5f, false, stats, stream, "i8-f32-sparse");
    RunAndTrack<float, int8_t, float>(512, 1024, 0.9f, 1.0f, 0.0f, true, stats, stream, "trans-i8-f32-default");
    RunAndTrack<float, int8_t, float>(2048, 512, 0.95f, 0.5f, 1.0f, true, stats, stream, "trans-i8-f32-tall");
}

void PrintTestSummary(const TestStats& stats)
{
    // ====================== 汇总 ======================
    std::cout << "\n========================================\n";
    std::cout << "              Test Summary\n";
    std::cout << "========================================\n";
    std::cout << "Total cases  : " << stats.total << "\n";
    std::cout << "Passed       : " << stats.passed << "\n";
    std::cout << "Failed       : " << stats.failed << "\n";
    if (stats.total > 0) {
        std::cout << "Pass rate    : " << (100.0 * stats.passed / stats.total) << "%\n";
    }
    if (!stats.failedCases.empty()) {
        std::cout << "----------------------------------------\n";
        std::cout << "Failed case details:\n";
        for (const auto& s : stats.failedCases) {
            std::cout << "  - " << s << "\n";
        }
    }
    std::cout << "========================================\n";
}

int RunSelectedSuite(int32_t argc, char* argv[], aclrtStream stream)
{
    if (argc == 2 && std::string(argv[1]) == "--integer-overflow") {
        return RunIntegerOverflowSuite(stream);
    }
    if (argc == 2 && std::string(argv[1]) == "--perf") {
        return RunPerformanceSuite(stream);
    }
    if (argc == 3 && std::string(argv[1]) == "--task-cases") {
        return RunTaskCaseFile(argv[2], stream);
    }
    if (argc == 2 && std::string(argv[1]) == "--strided") {
        return RunStridedSuite(stream);
    }
    if (argc == 2 && std::string(argv[1]) == "--acceptance") {
        return RunAcceptanceSuite(stream);
    }

    TestStats stats;

    std::cout << "\n"
              << "###########################################################\n"
              << "##               SpMV Test Suite (aclsparse)              ##\n"
              << "###########################################################\n\n";

    RunFloatBasicSuite(stats, stream);
    RunIntegerBasicSuite(stats, stream);
    RunFloatTransposeSuite(stats, stream);
    RunIntegerTransposeSuite(stats, stream);
    RunHalfBasicSuite(stats, stream);
    RunHalfTransposeSuite(stats, stream);
    RunBfloatBasicSuite(stats, stream);
    RunBfloatTransposeSuite(stats, stream);
    RunInt8FloatSuite(stats, stream);
    PrintTestSummary(stats);

    return (stats.failed == 0) ? 0 : 1;
}

int main(int32_t argc, char* argv[])
{
    constexpr int32_t deviceId = 0;
    aclrtStream stream = nullptr;
    int ret = Init(deviceId, &stream);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("Init acl failed. ERROR: %d\n", ret);
        return ret;
    }
    ret = RunSelectedSuite(argc, argv, stream);
    Finalize(deviceId, stream);
    return ret;
}
