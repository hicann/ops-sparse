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

/*!
 * \file spgemm_harness.h
 * \brief SpGEMM UT 的 device 侧执行封装：多阶段流程 + 结构/数值比对。
 */

#ifndef SPGEMM_TEST_HARNESS_H
#define SPGEMM_TEST_HARNESS_H

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "securec.h"
#include "acl/acl.h"
#include "cann_ops_sparse.h"
#include "spgemm_ref.h"

// ---------------------------------------------------------------------------
// 安全函数辅助：memcpy_s 返回值检查
// ---------------------------------------------------------------------------

inline void SpgemmCheckedMemcpy(void *dest, size_t destMax, const void *src, size_t count)
{
    if (memcpy_s(dest, destMax, src, count) != EOK) {
        throw std::runtime_error("memcpy_s failed in SpGEMM test harness");
    }
}

// ---------------------------------------------------------------------------
// dtype 转换：host 侧 double ←→ 目标存储类型的位表示
// ---------------------------------------------------------------------------

inline uint16_t SpgemmF32ToF16Bits(float f)
{
    uint32_t x = 0;
    SpgemmCheckedMemcpy(&x, sizeof(x), &f, sizeof(x));
    const uint32_t sign = (x >> 16) & 0x8000U;
    int32_t exp = static_cast<int32_t>((x >> 23) & 0xFFU) - 127 + 15;
    uint32_t mant = x & 0x7FFFFFU;
    if (((x >> 23) & 0xFFU) == 0xFFU) {  // Inf / NaN
        return static_cast<uint16_t>(sign | 0x7C00U | (mant ? 0x200U : 0U));
    }
    if (exp >= 0x1F) {
        return static_cast<uint16_t>(sign | 0x7C00U);  // 溢出为 Inf
    }
    if (exp <= 0) {
        return static_cast<uint16_t>(sign);  // 下溢为 ±0（UT 用值不触及次正规）
    }
    // 舍到最近偶数，与 NPU 的 CAST_ROUND 语义一致
    const uint32_t lsb = mant >> 13;
    const uint32_t rem = mant & 0x1FFFU;
    uint32_t out = static_cast<uint32_t>(exp) << 10 | lsb;
    if (rem > 0x1000U || (rem == 0x1000U && (lsb & 1U))) {
        out++;
    }
    return static_cast<uint16_t>(sign | out);
}

inline float SpgemmF16BitsToF32(uint16_t h)
{
    const uint32_t sign = static_cast<uint32_t>(h & 0x8000U) << 16;
    const uint32_t exp = (h >> 10) & 0x1FU;
    const uint32_t mant = h & 0x3FFU;
    uint32_t bits = 0;
    if (exp == 0) {
        bits = sign;  // ±0 / 次正规按 0 处理（UT 不构造次正规）
        if (mant != 0) {
            uint32_t e = 127 - 15 + 1;
            uint32_t m = mant;
            while ((m & 0x400U) == 0) {
                m <<= 1;
                e--;
            }
            bits = sign | (e << 23) | ((m & 0x3FFU) << 13);
        }
    } else if (exp == 0x1FU) {
        bits = sign | 0x7F800000U | (mant << 13);
    } else {
        bits = sign | ((exp + 127 - 15) << 23) | (mant << 13);
    }
    float f = 0.0f;
    SpgemmCheckedMemcpy(&f, sizeof(f), &bits, sizeof(f));
    return f;
}

// bf16 就是 fp32 的高 16 位；舍到最近偶数以匹配 NPU 的 CAST_ROUND。
inline uint16_t SpgemmF32ToBf16Bits(float f)
{
    uint32_t x = 0;
    SpgemmCheckedMemcpy(&x, sizeof(x), &f, sizeof(x));
    if (((x >> 23) & 0xFFU) == 0xFFU) {
        return static_cast<uint16_t>(x >> 16);  // Inf / NaN 直接截断
    }
    const uint32_t lsb = (x >> 16) & 1U;
    const uint32_t rounded = x + 0x7FFFU + lsb;
    return static_cast<uint16_t>(rounded >> 16);
}

inline float SpgemmBf16BitsToF32(uint16_t b)
{
    const uint32_t bits = static_cast<uint32_t>(b) << 16;
    float f = 0.0f;
    SpgemmCheckedMemcpy(&f, sizeof(f), &bits, sizeof(f));
    return f;
}

/**
 * 把参考 CSR 的 double values 按目标 dtype 打包成字节串。
 */
inline std::vector<uint8_t> SpgemmPackValues(
    const std::vector<double> &re, const std::vector<double> &im, aclDataType dt)
{
    const size_t n = re.size();
    std::vector<uint8_t> out;
    switch (dt) {
        case ACL_FLOAT: {
            out.resize(n * sizeof(float));
            auto *p = reinterpret_cast<float *>(out.data());
            for (size_t i = 0; i < n; i++) {
                p[i] = static_cast<float>(re[i]);
            }
            break;
        }
        case ACL_FLOAT16: {
            out.resize(n * sizeof(uint16_t));
            auto *p = reinterpret_cast<uint16_t *>(out.data());
            for (size_t i = 0; i < n; i++) {
                p[i] = SpgemmF32ToF16Bits(static_cast<float>(re[i]));
            }
            break;
        }
        case ACL_BF16: {
            out.resize(n * sizeof(uint16_t));
            auto *p = reinterpret_cast<uint16_t *>(out.data());
            for (size_t i = 0; i < n; i++) {
                p[i] = SpgemmF32ToBf16Bits(static_cast<float>(re[i]));
            }
            break;
        }
        case ACL_COMPLEX64: {
            out.resize(n * 2 * sizeof(float));
            auto *p = reinterpret_cast<float *>(out.data());
            for (size_t i = 0; i < n; i++) {
                p[2 * i] = static_cast<float>(re[i]);
                p[2 * i + 1] = static_cast<float>(i < im.size() ? im[i] : 0.0);
            }
            break;
        }
        default:
            break;
    }
    return out;
}

// 把 device 取回的 values 解包成 double（re/im 两路）
inline void SpgemmUnpackValues(const std::vector<uint8_t> &raw, size_t n,
                               aclDataType dt, std::vector<double> &re,
                               std::vector<double> &im)
{
    re.assign(n, 0.0);
    im.assign(n, 0.0);
    switch (dt) {
        case ACL_FLOAT: {
            auto *p = reinterpret_cast<const float *>(raw.data());
            for (size_t i = 0; i < n; i++) {
                re[i] = p[i];
            }
            break;
        }
        case ACL_FLOAT16: {
            auto *p = reinterpret_cast<const uint16_t *>(raw.data());
            for (size_t i = 0; i < n; i++) {
                re[i] = SpgemmF16BitsToF32(p[i]);
            }
            break;
        }
        case ACL_BF16: {
            auto *p = reinterpret_cast<const uint16_t *>(raw.data());
            for (size_t i = 0; i < n; i++) {
                re[i] = SpgemmBf16BitsToF32(p[i]);
            }
            break;
        }
        case ACL_COMPLEX64: {
            auto *p = reinterpret_cast<const float *>(raw.data());
            for (size_t i = 0; i < n; i++) {
                re[i] = p[2 * i];
                im[i] = p[2 * i + 1];
            }
            break;
        }
        default:
            break;
    }
}

inline size_t SpgemmDtypeBytes(aclDataType dt)
{
    switch (dt) {
        case ACL_FLOAT:     return 4;
        case ACL_FLOAT16:   return 2;
        case ACL_BF16:      return 2;
        case ACL_COMPLEX64: return 8;
        default:            return 4;
    }
}

/**
 * 把 double 量化到目标存储 dtype 能精确表示的值，确保 golden 与 device 输入一致。
 */
inline double SpgemmQuantize(double v, aclDataType dt)
{
    switch (dt) {
        case ACL_FLOAT16:
            return static_cast<double>(
                SpgemmF16BitsToF32(SpgemmF32ToF16Bits(static_cast<float>(v))));
        case ACL_BF16:
            return static_cast<double>(
                SpgemmBf16BitsToF32(SpgemmF32ToBf16Bits(static_cast<float>(v))));
        case ACL_FLOAT:
        case ACL_COMPLEX64:
            return static_cast<double>(static_cast<float>(v));
        default:
            return v;
    }
}

inline const char *SpgemmDtypeName(aclDataType dt)
{
    switch (dt) {
        case ACL_FLOAT:     return "fp32";
        case ACL_FLOAT16:   return "fp16";
        case ACL_BF16:      return "bf16";
        case ACL_COMPLEX64: return "c64";
        default:            return "?";
    }
}

/**
 * 把参考 CSR 的 values 量化到目标 dtype，就地修改。
 */
inline void SpgemmQuantizeCsr(SpgemmRefCsr &m, aclDataType dt)
{
    for (double &v : m.valRe) {
        v = SpgemmQuantize(v, dt);
    }
    if (dt == ACL_COMPLEX64) {
        for (double &v : m.valIm) {
            v = SpgemmQuantize(v, dt);
        }
    } else {
        // 实数类型的虚部恒为 0，避免残留值影响参考实现
        for (double &v : m.valIm) {
            v = 0.0;
        }
    }
}

/** 精度阈值：rtol / atol / A（绝对误差硬上限基值）。 */
struct SpgemmTolerance {
    double rtol;
    double atol;
    double hardA;
};

inline SpgemmTolerance SpgemmToleranceFor(aclDataType dt)
{
    switch (dt) {
        // fp16: rtol=atol=2^-9, A=1e-1
        case ACL_FLOAT16:   return {1.0 / 512.0, 1.0 / 512.0, 1e-1};
        // bf16: rtol=atol=2^-6, A=1e0
        case ACL_BF16:      return {1.0 / 64.0, 1.0 / 64.0, 1e0};
        // fp32 / complex64: rtol=2^-10, atol=2^-16, A=1e-2
        default:            return {1.0 / 1024.0, 1.0 / 65536.0, 1e-2};
    }
}

#endif  // SPGEMM_TEST_HARNESS_H
