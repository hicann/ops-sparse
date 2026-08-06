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

/*
 * Scatter 算子测试 —— 结构对齐 spmv_test.cpp，精度采用 bit-exact 判定。
 *
 * scatter 是非计算类算子（纯数据搬移：y[indices[i]] = values[i]），按
 * /ops-precision-standard 精度标准，非计算类算子要求逐位一致（bit-exact）。
 * 因此 Verification 对 float 做严格逐位匹配（memcmp），任何一位不同即判失败。
 * 同时保留 MARE/MERE 作为信息性指标打印（便于观察），但不参与达标判定。
 *
 * 测试链路：aclsparseCreate -> aclsparseSetStream ->
 *           aclsparseCreateSpVec / aclsparseCreateDnVec ->
 *           aclsparseScatter -> aclsparseDestroy* / aclsparseDestroy。
 *
 * dtype 约束：Host 实现对 idxType/valueType 做了 NOT_SUPPORTED 校验，scatter 仅支持
 *   idxType = ACL_SPARSE_INDEX_32I、valueType = ACL_FLOAT。因此 ValT 实际只有 float，
 * 保留模板骨架以对齐 spmv 风格，但模板参数固定为 float。
 *
 * 索引约束（互异）：本测试只覆盖互异索引场景，所有用例的 indices 均无重复，确保
 *   kernel 写入的 GM 地址互不冲突，结果具有确定性。这与 cuSPARSE scatter 语义下
 *   "互异索引 → 唯一确定写入" 的约定一致，也与原 B 版测试只用互异索引的做法一致。
 *   含重复索引的场景下，kernel 跨核并发写同一 GM 地址会导致 last-write-wins 顺序
 *   非确定（cuSPARSE 语义下本就未定义），不在本测试覆盖范围内。
 *
 * NaN 处理策略：bit-exact 下 NaN 的位模式可能不同（kernel 不保证 NaN payload 一致）。
 *   对含 NaN 的 value 用例：CPU golden 与 NPU 结果在 NaN 位置只要双方都是 NaN 即视为
 *   通过（isnan 判定），其余元素仍严格 bit-exact。NaN 用例在 worstInfo 中单独标注。
 *   +0.0 / -0.0 仍按严格 bit-exact（区分符号位），因为 scatter 是纯搬移，符号位必须保留。
 *
 * 用例分区（对齐 spmv 的 main 组织方式）：
 *   1. 基础用例：小/中/大 N、nnz=1、nnz=N（全覆盖）、nnz=0（早退）
 *   2. 索引模式：contiguous / reverse / stride / random-unique（均为互异索引）
 *   3. 特殊 value：含 0 / 负数 / 大值 / Inf / NaN
 *   4. 非对齐 nnz（考验 DataCopyPad 边界）
 *   5. 随机参数采样（RunRandomParams，~30 例）
 *   6. 大规模（N=1M+，nnz=1M 随机互异 / 连续）
 *   7. 汇总块（Total/Passed/Failed/Pass rate/失败明细）
 */

#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <limits>
#include <string>
#include <memory>
#include <sstream>
#include <type_traits>
#include "acl/acl.h"
#include "cann_ops_sparse.h"

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

// ===================== 索引分布生成 =====================

/*
 * 索引分布模式枚举，对齐 spmv GenerateCsr 的多种数据生成风格。
 * scatter 的核心维度是 (N, nnz, 索引分布)，索引分布决定 kernel 内
 * run-length 写入快路径的命中率，是覆盖 kernel 分支的关键。
 */
enum class IndexPattern {
    CONTIGUOUS,    // 连续：0,1,2,...（触发 run-length 快路径）
    REVERSE,       // 逆序：nnz-1,...,1,0（runLen=1 逐元素写）
    STRIDE,        // 等步长：0,stride,2*stride,...（runLen=1）
    RANDOM_UNIQUE  // 随机互异（覆盖一般场景，确定性强）
};

const char *IndexPatternName(IndexPattern p) {
    switch (p) {
        case IndexPattern::CONTIGUOUS:    return "contiguous";
        case IndexPattern::REVERSE:       return "reverse";
        case IndexPattern::STRIDE:        return "stride";
        case IndexPattern::RANDOM_UNIQUE: return "random-unique";
    }
    return "unknown";
}

/*
 * 为 scatter 生成互异索引（升序）：从 [0, N) 中无放回采样 nnz 个。
 * 思路对齐 spmv 的 SampleRowColumnsUnique（visited 数组去重）。
 */
std::vector<int32_t> GenUniqueIndices(int64_t N, int64_t nnz, std::mt19937 &rng) {
    if (nnz <= 0) return {};
    if (nnz > N) nnz = N;
    std::vector<int32_t> pool(static_cast<size_t>(N));
    for (int64_t i = 0; i < N; ++i) pool[static_cast<size_t>(i)] = static_cast<int32_t>(i);
    std::shuffle(pool.begin(), pool.end(), rng);
    std::vector<int32_t> idx(pool.begin(), pool.begin() + nnz);
    std::sort(idx.begin(), idx.end());
    return idx;
}

/*
 * 按指定 IndexPattern 生成索引序列（长度=nnz），所有模式均保证索引互异。
 * - CONTIGUOUS：从 startStart 起连续递增（startStart 用于错开起点）
 * - REVERSE：[startStart+nnz-1, ..., startStart]
 * - STRIDE：startStart, startStart+stride, ...（保证不越界 N）
 * - RANDOM_UNIQUE：随机互异升序
 *
 * 返回的索引顺序即为 kernel 处理顺序（cuSPARSE 模式：不去重不排序）。
 * 所有模式均产生互异索引，确保 kernel 写入地址不冲突，结果确定。
 */
std::vector<int32_t> GenIndicesByPattern(int64_t N, int64_t nnz, IndexPattern pattern,
                                         std::mt19937 &rng, int64_t startStart = 0,
                                         int64_t stride = 8) {
    if (nnz <= 0) return {};
    std::vector<int32_t> idx;
    idx.reserve(static_cast<size_t>(nnz));

    switch (pattern) {
        case IndexPattern::CONTIGUOUS: {
            int64_t start = startStart;
            if (start + nnz > N) start = std::max<int64_t>(0, N - nnz);
            for (int64_t i = 0; i < nnz; ++i) idx.push_back(static_cast<int32_t>(start + i));
            break;
        }
        case IndexPattern::REVERSE: {
            int64_t start = startStart;
            if (start + nnz > N) start = std::max<int64_t>(0, N - nnz);
            for (int64_t i = 0; i < nnz; ++i) idx.push_back(static_cast<int32_t>(start + nnz - 1 - i));
            break;
        }
        case IndexPattern::STRIDE: {
            if (stride < 1) stride = 1;
            int64_t start = startStart;
            // 保证 start + (nnz-1)*stride < N
            int64_t maxStart = N - 1 - (nnz - 1) * stride;
            if (maxStart < 0) { // stride 太大无法容纳 nnz 个，退化为 stride=1
                stride = 1;
                maxStart = N - nnz;
            }
            if (start > maxStart) start = std::max<int64_t>(0, maxStart);
            for (int64_t i = 0; i < nnz; ++i) idx.push_back(static_cast<int32_t>(start + i * stride));
            break;
        }
        case IndexPattern::RANDOM_UNIQUE: {
            idx = GenUniqueIndices(N, nnz, rng);
            break;
        }
    }
    return idx;
}

// ===================== 数据生成（values / yInit） =====================

/*
 * 生成随机 float 值，对齐 spmv GenerateDenseVector 的风格。
 */
template <typename T>
void GenerateValues(size_t nnz, std::vector<T> &vals, std::mt19937 &rng,
                    float lo = -10.0f, float hi = 10.0f) {
    std::uniform_real_distribution<float> dist(lo, hi);
    vals.assign(nnz, T{});
    for (size_t i = 0; i < nnz; ++i) {
        vals[i] = static_cast<T>(dist(rng));
    }
}

template <typename T>
void GenerateDenseVector(int64_t N, std::vector<T> &y, std::mt19937 &rng,
                         float lo = -5.0f, float hi = 5.0f) {
    std::uniform_real_distribution<float> dist(lo, hi);
    y.assign(N, T{});
    for (int64_t i = 0; i < N; ++i) {
        y[i] = static_cast<T>(dist(rng));
    }
}

// ===================== CPU 参考实现 =====================

/*
 * scatter CPU golden：y_out[indices[i]] = values[i]（i = 0..nnz-1）。
 * cuSPARSE 模式：不去重不排序。本测试 indices 均互异，无 last-write-wins 歧义。
 * yOut 初始值来自 yInit（模拟 device 上 vecY 的初始内容）。
 */
template <typename ValT>
void ScatterCpu(const std::vector<int32_t> &indices,
                const std::vector<ValT> &values,
                const std::vector<ValT> &yInit,
                std::vector<ValT> &yOut) {
    yOut = yInit;
    for (size_t i = 0; i < indices.size(); ++i) {
        yOut[indices[i]] = values[i];
    }
}

// ===================== bit-exact 精度验证 =====================

/*
 * 非计算类算子：逐位一致（bit-exact）。返回 0=通过，1=失败。
 * 同时计算 MARE/MERE 作为信息性指标打印（对齐 spmv Verification 的输出风格），
 * 但达标判定严格基于 bit-exact。
 *
 * NaN 处理：双方均为 NaN 视为匹配（isnan 判定），不计入失败；
 *   其余元素（含 +0.0/-0.0）严格 memcmp 逐位比较。
 */
template <typename T>
int32_t Verification(const std::vector<T> &cpuGolden,
                     const std::vector<T> &npuRet,
                     float &MARE, float &MERE,
                     std::string *worstInfo = nullptr) {
    if (npuRet.size() != cpuGolden.size()) {
        std::ostringstream oss;
        oss << "[ERROR] size mismatch: golden=" << cpuGolden.size()
            << " npu=" << npuRet.size();
        if (worstInfo) *worstInfo = oss.str();
        std::cout << oss.str() << "\n";
        return 1;
    }

    int32_t status = 0;
    std::cout << "Verification...\n";
    for (int i = 0; i < std::min(static_cast<int64_t>(npuRet.size()), static_cast<int64_t>(10)); ++i) {
        std::cout << "golden[" << i << "]=" << cpuGolden[i]
                  << " npu_result[" << i << "]=" << npuRet[i] << "\n";
    }

    size_t worstIdx = 0;
    float worstGolden = 0, worstNpu = 0;
    uint32_t bitDiffCount = 0;
    uint32_t nanSkipped = 0;
    bool worstIsNaN = false;

    for (size_t i = 0; i < npuRet.size(); ++i) {
        float npuVal = static_cast<float>(npuRet[i]);
        float cpuVal = static_cast<float>(cpuGolden[i]);

        // 信息性误差度量（NaN 不计入 MARE/MERE）
        if (!std::isnan(npuVal) && !std::isnan(cpuVal)) {
            float aError = std::fabs(npuVal - cpuVal);
            float rError = aError / (std::fabs(cpuVal) + 1e-7f);
            if (rError > MARE) { worstIdx = i; worstGolden = cpuVal; worstNpu = npuVal; worstIsNaN = false; }
            MERE += rError;
        }

        // ---- bit-exact 判定 ----
        // NaN：双方均为 NaN 视为匹配（payload 可能不同，但语义一致）
        bool goldenIsNaN = std::isnan(cpuVal);
        bool npuIsNaN = std::isnan(npuVal);
        if (goldenIsNaN && npuIsNaN) {
            nanSkipped++;
            continue;
        }
        // 严格逐位比较（区分 +0.0/-0.0/Inf 符号等）
        if (std::memcmp(&npuRet[i], &cpuGolden[i], sizeof(T)) != 0) {
            if (bitDiffCount == 0) {
                worstIdx = i;
                worstGolden = cpuVal;
                worstNpu = npuVal;
                worstIsNaN = (goldenIsNaN != npuIsNaN);
            }
            bitDiffCount++;
            status = 1;
            if (bitDiffCount <= 5) {
                std::cout << "[WARNING] bit-exact FAIL at [" << i
                          << "] golden=" << cpuVal << " npu=" << npuVal << "\n";
            }
        }
    }

    MERE /= (npuRet.size() > 0) ? npuRet.size() : 1;
    std::cout << "Mean Relative Error = " << MERE
              << "; Max Relative Error = " << MARE
              << " (informational only; pass/fail by bit-exact)\n";
    if (nanSkipped > 0) {
        std::cout << "NaN positions skipped (both NaN): " << nanSkipped << "\n";
    }

    if (worstInfo) {
        std::ostringstream oss;
        if (status != 0) {
            oss << "bit-exact FAIL: " << bitDiffCount << "/" << npuRet.size()
                << " elements differ. first mismatch at [" << worstIdx
                << "] golden=" << worstGolden << " npu=" << worstNpu;
            if (worstIsNaN) oss << " (NaN mismatch)";
        } else {
            oss << "bit-exact PASS (all " << npuRet.size() << " elements identical";
            if (nanSkipped > 0) oss << ", " << nanSkipped << " NaN matched";
            oss << ")";
        }
        *worstInfo = oss.str();
    }
    return status;
}

// ===================== 设备资源创建 =====================

int CreateDeviceTensor(uint8_t *hostData, const size_t size, uint8_t **deviceAddr) {
    // size == 0 时不分配（aclrtMalloc 0 字节会返回 ACL_ERROR_INVALID_PARAM），
    // 保持 *deviceAddr = nullptr。scatter 在 nnz==0 时允许 indices/values 为空。
    if (size == 0) {
        *deviceAddr = nullptr;
        return ACL_SUCCESS;
    }
    auto ret = aclrtMalloc((void **)deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    ret = aclrtMemcpy(*deviceAddr, size, hostData, size, ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret);
        aclrtFree(*deviceAddr);
        *deviceAddr = nullptr;
        return ret;
    }
    return ACL_SUCCESS;
}

// ===================== Init / Finalize =====================

int Init(int32_t deviceId, aclrtStream *stream) {
    auto ret = aclInit(nullptr);
    // aclInit 重复调用返回非 0，测试进程内只 Init 一次；首次失败才报错
    if (ret != ACL_SUCCESS) {
        // 若已初始化，继续；否则报错
    }
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    return 0;
}

void Finalize(int32_t deviceId, aclrtStream stream) {
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
}

// ===================== 类型映射 =====================

template <typename T>
aclDataType AclTypeOf();
template <>
aclDataType AclTypeOf<float>() { return ACL_FLOAT; }

// ===================== 主测试函数 =====================

/*
 * @brief Scatter 单用例测试（使用 aclsparseScatter 接口）
 *
 * @tparam ValT   值类型（scatter 仅支持 float，保留模板以对齐 spmv 风格）
 *
 * @param N        稠密向量 vecY 的长度
 * @param nnz      非零元素个数
 * @param pattern  索引分布模式
 * @param fillVal  value 的填充策略（用于特殊值用例：0/负数/大值/Inf/NaN）
 *                 fillVal == nullptr 时使用随机生成
 * @param MARE     OUT，最大相对误差（信息性）
 * @param MERE     OUT，平均相对误差（信息性）
 * @param stream   ACL stream
 * @param worstInfo OUT，验证详情
 * @return 0=通过，非 0=失败
 */
template <typename ValT>
int Test(const size_t N, const size_t nnz, IndexPattern pattern,
         const std::vector<float> *fillVal,
         float &MARE, float &MERE, aclrtStream stream,
         std::string *worstInfo = nullptr) {
    std::cout << "====Test case: N = " << N
              << " nnz = " << nnz
              << " pattern = " << IndexPatternName(pattern)
              << " type = " << (std::is_same_v<ValT, float> ? "float" : "unknown")
              << "====\n";

    std::mt19937 rng(static_cast<uint32_t>(
        std::chrono::steady_clock::now().time_since_epoch().count()));

    // 生成索引
    std::vector<int32_t> indices = GenIndicesByPattern(
        static_cast<int64_t>(N), static_cast<int64_t>(nnz), pattern, rng);
    const size_t realNnz = indices.size();

    // 生成 values
    std::vector<ValT> values;
    if (fillVal != nullptr && !fillVal->empty()) {
        // 用 fillVal 填充到 realNnz 长度（循环复用）
        values.assign(realNnz, ValT{});
        for (size_t i = 0; i < realNnz; ++i) {
            values[i] = static_cast<ValT>((*fillVal)[i % fillVal->size()]);
        }
    } else {
        GenerateValues<ValT>(realNnz, values, rng);
    }

    // 生成 yInit
    std::vector<ValT> yInit;
    GenerateDenseVector<ValT>(static_cast<int64_t>(N), yInit, rng);

    // CPU golden
    std::vector<ValT> output_cpu;
    ScatterCpu<ValT>(indices, values, yInit, output_cpu);

    // 设备内存
    size_t idxBytes = realNnz * sizeof(int32_t);
    size_t valBytes = realNnz * sizeof(ValT);
    size_t yBytes   = N * sizeof(ValT);

    uint8_t *idxDev = nullptr, *valDev = nullptr, *yDev = nullptr;

    int aclRet = CreateDeviceTensor((uint8_t *)indices.data(), idxBytes, &idxDev);
    CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);
    std::unique_ptr<void, aclError (*)(void *)> idp(idxDev, aclrtFree);

    aclRet = CreateDeviceTensor((uint8_t *)values.data(), valBytes, &valDev);
    CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);
    std::unique_ptr<void, aclError (*)(void *)> vdp(valDev, aclrtFree);

    aclRet = CreateDeviceTensor((uint8_t *)yInit.data(), yBytes, &yDev);
    CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);
    std::unique_ptr<void, aclError (*)(void *)> ydp(yDev, aclrtFree);

    // handle + set stream
    aclsparseHandle_t spHandle = nullptr;
    aclsparseStatus_t sparseRet = aclsparseCreate(&spHandle);
    CHECK_RET(sparseRet == ACL_SPARSE_STATUS_SUCCESS,
              LOG_PRINT("aclsparseCreate failed. ERROR: %d\n", sparseRet);
              return sparseRet);
    sparseRet = aclsparseSetStream(spHandle, stream);
    CHECK_RET(sparseRet == ACL_SPARSE_STATUS_SUCCESS,
              LOG_PRINT("aclsparseSetStream failed. ERROR: %d\n", sparseRet);
              return sparseRet);

    // 描述符：SpVec（indices+values）+ DnVec（y）
    // idxType 固定 32I，valueType 固定 FLOAT（Host 实现对其它类型返回 NOT_SUPPORTED）
    aclsparseSpVecDescr_t vecXDesc = nullptr;
    sparseRet = aclsparseCreateSpVec(&vecXDesc, static_cast<int64_t>(N),
                                     static_cast<int64_t>(realNnz), idxDev, valDev,
                                     ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_BASE_ZERO,
                                     AclTypeOf<ValT>());
    CHECK_RET(sparseRet == ACL_SPARSE_STATUS_SUCCESS,
              LOG_PRINT("aclsparseCreateSpVec failed. ERROR: %d\n", sparseRet);
              return sparseRet);

    aclsparseDnVecDescr_t vecYDesc = nullptr;
    sparseRet = aclsparseCreateDnVec(&vecYDesc, static_cast<int64_t>(N), yDev,
                                     AclTypeOf<ValT>());
    CHECK_RET(sparseRet == ACL_SPARSE_STATUS_SUCCESS,
              LOG_PRINT("aclsparseCreateDnVec failed. ERROR: %d\n", sparseRet);
              aclsparseDestroySpVec(vecXDesc); return sparseRet);

    // 执行 scatter
    sparseRet = aclsparseScatter(spHandle, vecXDesc, vecYDesc);
    CHECK_RET(sparseRet == ACL_SPARSE_STATUS_SUCCESS,
              LOG_PRINT("aclsparseScatter failed. ERROR: %d\n", sparseRet);
              aclsparseDestroySpVec(vecXDesc); aclsparseDestroyDnVec(vecYDesc);
              aclsparseDestroy(spHandle); return sparseRet);

    // 同步 + D2H
    aclRet = aclrtSynchronizeStream(stream);
    CHECK_RET(aclRet == ACL_SUCCESS,
              LOG_PRINT("[ERROR] aclsparseScatter: aclrtSynchronizeStream failed, ret=%d\n", aclRet);
              aclsparseDestroySpVec(vecXDesc); aclsparseDestroyDnVec(vecYDesc);
              aclsparseDestroy(spHandle); return ACL_SPARSE_STATUS_EXECUTION_FAILED);

    std::vector<ValT> yResult(N);
    aclRet = aclrtMemcpy(yResult.data(), yBytes, yDev, yBytes, ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(aclRet == ACL_SUCCESS,
              LOG_PRINT("aclrtMemcpy D2H failed. ERROR: %d\n", aclRet);
              aclsparseDestroySpVec(vecXDesc); aclsparseDestroyDnVec(vecYDesc);
              aclsparseDestroy(spHandle); return aclRet);

    // bit-exact 验证（MARE/MERE 信息性，达标判定严格 bit-exact）
    int verifyRet = Verification<ValT>(output_cpu, yResult, MARE, MERE, worstInfo);

    aclsparseDestroySpVec(vecXDesc);
    aclsparseDestroyDnVec(vecYDesc);
    aclsparseDestroy(spHandle);

    CHECK_RET(verifyRet == 0, LOG_PRINT("====Test case fail!====\n\n"); return verifyRet);
    std::cout << "====Test case pass!====\n\n";
    return 0;
}

// ===================== 测试统计 =====================

struct TestStats {
    int total = 0;
    int passed = 0;
    int failed = 0;
    std::vector<std::string> failedCases;
};

/*
 * RunAndTrack：执行单个 Test 用例并统计结果，对齐 spmv 的 RunAndTrack 骨架。
 * 适配 scatter 维度：(N, nnz, pattern, fillVal)。
 */
template <typename ValT>
int RunAndTrack(size_t N, size_t nnz, IndexPattern pattern,
                const std::vector<float> *fillVal,
                TestStats &stats, aclrtStream stream, const std::string &tag = "") {
    float MARE = 0, MERE = 0;
    std::string worstInfo;
    int ret = Test<ValT>(N, nnz, pattern, fillVal, MARE, MERE, stream, &worstInfo);

    stats.total++;
    if (ret == 0) {
        stats.passed++;
    } else {
        stats.failed++;
        std::ostringstream oss;
        if (!tag.empty()) oss << "[" << tag << "] ";
        oss << "type=" << (std::is_same_v<ValT, float> ? "float" : "unknown")
            << " N=" << N << " nnz=" << nnz
            << " pattern=" << IndexPatternName(pattern)
            << " MARE=" << MARE << " MERE=" << MERE << "\n"
            << "          " << worstInfo << "\n"
            << "          status=" << ret;
        stats.failedCases.push_back(oss.str());
    }
    return ret;
}

// ===================== 随机测试辅助函数 =====================

/*
 * RunRandomParams：随机采样 (N, nnz, pattern) 执行多例，对齐 spmv 的 RunRandomParams。
 */
template <typename ValT>
void RunRandomParams(int numCases, const std::string &tagPrefix,
                     TestStats &stats, aclrtStream stream) {
    std::cout << "\n======== Float Random Tests ========\n";

    std::mt19937 rng(static_cast<uint32_t>(
        std::chrono::steady_clock::now().time_since_epoch().count()));
    std::uniform_int_distribution<int64_t> nDist(1024, 65536);
    // 索引模式：偏向 RANDOM_UNIQUE / CONTIGUOUS / STRIDE（覆盖主要分支）
    std::uniform_int_distribution<int> patDist(0, 3);

    for (int i = 0; i < numCases; ++i) {
        size_t N = static_cast<size_t>(nDist(rng));
        std::uniform_int_distribution<int64_t> nnzDist(1, static_cast<int64_t>(N));
        size_t nnz = static_cast<size_t>(nnzDist(rng));
        IndexPattern pat;
        switch (patDist(rng)) {
            case 0: pat = IndexPattern::RANDOM_UNIQUE; break;
            case 1: pat = IndexPattern::CONTIGUOUS; break;
            case 2: pat = IndexPattern::STRIDE; break;
            default: pat = IndexPattern::REVERSE; break;
        }
        std::cout << "--- Random float case " << (i + 1) << "/" << numCases
                  << " (N=" << N << " nnz=" << nnz
                  << " pattern=" << IndexPatternName(pat) << ") ---\n";
        RunAndTrack<ValT>(N, nnz, pat, nullptr, stats, stream,
                          tagPrefix + std::to_string(i + 1));
    }
}

// ===================== main =====================

int main(int32_t /*argc*/, char * /*argv*/[]) {
    const int32_t deviceId = 0;  // device 0 空闲；device 1 被他人占用
    aclrtStream stream = nullptr;
    int ret = Init(deviceId, &stream);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("Init acl failed. ERROR: %d\n", ret);
        return ret;
    }

    TestStats stats;

    std::cout << "\n"
              << "###########################################################\n"
              << "##            Scatter Test Suite (bit-exact)             ##\n"
              << "###########################################################\n\n";

    // ============ 基础用例：小/中/大 N、nnz=1、nnz=N、nnz=0 ============
    std::cout << "======== Basic Tests ========\n";
    RunAndTrack<float>(16, 8, IndexPattern::RANDOM_UNIQUE, nullptr, stats, stream, "small-8of16");
    RunAndTrack<float>(16, 12, IndexPattern::RANDOM_UNIQUE, nullptr, stats, stream, "small-12of16");
    RunAndTrack<float>(1024, 1, IndexPattern::RANDOM_UNIQUE, nullptr, stats, stream, "nnz1");
    RunAndTrack<float>(2048, 2048, IndexPattern::CONTIGUOUS, nullptr, stats, stream, "full-density-seq");
    RunAndTrack<float>(4096, 4096, IndexPattern::RANDOM_UNIQUE, nullptr, stats, stream, "full-density-rand");
    RunAndTrack<float>(1024, 0, IndexPattern::RANDOM_UNIQUE, nullptr, stats, stream, "nnz0-early-return");
    RunAndTrack<float>(65536, 1024, IndexPattern::RANDOM_UNIQUE, nullptr, stats, stream, "medium-1k");

    // ============ 索引模式覆盖（均为互异索引）============
    std::cout << "\n======== Index Pattern Tests ========\n";
    RunAndTrack<float>(4096, 1024, IndexPattern::CONTIGUOUS, nullptr, stats, stream, "contiguous-head");
    RunAndTrack<float>(8192, 4096, IndexPattern::CONTIGUOUS, nullptr, stats, stream, "contiguous-large");
    RunAndTrack<float>(2048, 512, IndexPattern::REVERSE, nullptr, stats, stream, "reverse");
    RunAndTrack<float>(8192, 512, IndexPattern::STRIDE, nullptr, stats, stream, "stride8");
    RunAndTrack<float>(8192, 1000, IndexPattern::STRIDE, nullptr, stats, stream, "stride3");
    RunAndTrack<float>(4096, 2048, IndexPattern::RANDOM_UNIQUE, nullptr, stats, stream, "random-unique");

    // ============ 特殊 value：含 0/负数/大值/Inf/NaN ============
    std::cout << "\n======== Special Values Tests (0/neg/large/Inf/NaN) ========\n";
    {
        // 含 +0/-0/负数/大值/Inf/-Inf/NaN 的混合 value
        std::vector<float> specialVals = {
            0.0f,                                    // +0
            -0.0f,                                   // -0（bit-exact 区分符号位）
            -7.5f,                                   // 负数
            1e30f,                                   // 大值
            std::numeric_limits<float>::infinity(),  // +Inf
            -std::numeric_limits<float>::infinity(), // -Inf
            std::numeric_limits<float>::quiet_NaN(), // NaN（isnan 判定）
            42.0f,
            -1e-10f,
            3.0f
        };
        RunAndTrack<float>(1024, 10, IndexPattern::CONTIGUOUS, &specialVals, stats, stream, "special-mixed");
    }
    {
        // 全 0 值
        std::vector<float> zeroVal = {0.0f};
        RunAndTrack<float>(2048, 256, IndexPattern::RANDOM_UNIQUE, &zeroVal, stats, stream, "all-zero-val");
    }
    {
        // 全负值
        std::vector<float> negVal = {-50.0f, -1.0f, -100.0f, -0.5f};
        RunAndTrack<float>(2048, 256, IndexPattern::RANDOM_UNIQUE, &negVal, stats, stream, "all-negative");
    }
    {
        // 全大值
        std::vector<float> largeVal = {1e30f, -1e30f, 1e35f};
        RunAndTrack<float>(2048, 256, IndexPattern::STRIDE, &largeVal, stats, stream, "all-large");
    }
    {
        // 全 Inf
        std::vector<float> infVal = {std::numeric_limits<float>::infinity(),
                                     -std::numeric_limits<float>::infinity()};
        RunAndTrack<float>(2048, 256, IndexPattern::RANDOM_UNIQUE, &infVal, stats, stream, "all-inf");
    }
    {
        // 全 NaN
        std::vector<float> nanVal = {std::numeric_limits<float>::quiet_NaN()};
        RunAndTrack<float>(2048, 256, IndexPattern::RANDOM_UNIQUE, &nanVal, stats, stream, "all-nan");
    }

    // ============ 非对齐 nnz（考验 DataCopyPad 边界）============
    std::cout << "\n======== Non-aligned nnz Tests ========\n";
    RunAndTrack<float>(4096, 7, IndexPattern::RANDOM_UNIQUE, nullptr, stats, stream, "nonalign-7");
    RunAndTrack<float>(4096, 13, IndexPattern::RANDOM_UNIQUE, nullptr, stats, stream, "nonalign-13");
    RunAndTrack<float>(4096, 33, IndexPattern::RANDOM_UNIQUE, nullptr, stats, stream, "nonalign-33");
    RunAndTrack<float>(4096, 65, IndexPattern::RANDOM_UNIQUE, nullptr, stats, stream, "nonalign-65");
    RunAndTrack<float>(4096, 127, IndexPattern::RANDOM_UNIQUE, nullptr, stats, stream, "nonalign-127");

    // ============ 随机参数采样 ============
    RunRandomParams<float>(30, "float-rand-", stats, stream);

    // ============ 大规模测试（互异索引）============
    std::cout << "\n======== Large Scale Tests ========\n";
    RunAndTrack<float>(5 * 1024 * 1024, 1024 * 1024, IndexPattern::RANDOM_UNIQUE,
                       nullptr, stats, stream, "large-1M-rand");
    RunAndTrack<float>(4 * 1024 * 1024, 1024 * 1024, IndexPattern::CONTIGUOUS,
                       nullptr, stats, stream, "large-1M-contiguous");

    // ====================== 汇总 ======================
    std::cout << "\n========================================\n";
    std::cout << "              Test Summary\n";
    std::cout << "========================================\n";
    std::cout << "Total cases  : " << stats.total << "\n";
    std::cout << "Passed       : " << stats.passed << "\n";
    std::cout << "Failed       : " << stats.failed << "\n";
    if (stats.total > 0) {
        std::cout << "Pass rate    : "
                  << (100.0 * stats.passed / stats.total) << "%\n";
    }
    if (!stats.failedCases.empty()) {
        std::cout << "----------------------------------------\n";
        std::cout << "Failed case details:\n";
        for (const auto &s : stats.failedCases) {
            std::cout << "  - " << s << "\n";
        }
    }
    std::cout << "========================================\n";

    Finalize(deviceId, stream);
    return (stats.failed == 0) ? 0 : 1;
}
