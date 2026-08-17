/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root directory of the software repository for the full text of the License.
 */

#ifndef TEST_CSR2GEBSR_DTYPE_UTILS_H_
#define TEST_CSR2GEBSR_DTYPE_UTILS_H_

// ============================================================================
// Unified dtype conversion utilities for csr2gebsr tests.
//
// Single source of truth for float <-> FP16/BF16/INT32 conversions used by
// both the CPU golden (round-trip) and the NPU wrapper (byte-level convert).
// All memory copies use the secure memcpy_s.
// ============================================================================

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "securec.h"  // memcpy_s, EOK

namespace sparse_test {

// ============================================================================
// Secure memcpy wrapper: aborts on failure (test-only, fixed-size copies)
// ============================================================================

inline void checkedMemcpy(void* dest, size_t destMax, const void* src, size_t count) {
    if (count == 0) return;  // nothing to copy (handles empty buffers safely)
    if (memcpy_s(dest, destMax, src, count) != EOK) {
        throw std::runtime_error("memcpy_s failed");
    }
}

// ============================================================================
// IEEE 754 half-precision (FP16) conversion helpers
// ============================================================================

inline uint16_t floatToHalf(uint32_t f) {
    uint32_t sign = (f >> 16) & 0x8000;
    int32_t exponent = static_cast<int32_t>((f >> 23) & 0xFF) - 127 + 15;
    uint32_t mantissa = f & 0x007FFFFF;

    if (exponent <= 0) {
        if (exponent < -10) return static_cast<uint16_t>(sign);
        mantissa |= 0x00800000;
        int shift = 14 - exponent;
        uint32_t m = mantissa >> shift;
        uint32_t remainder = mantissa & ((1u << shift) - 1);
        uint32_t halfway = 1u << (shift - 1);
        if (remainder > halfway || (remainder == halfway && (m & 1))) m++;
        return static_cast<uint16_t>(sign | m);
    } else if (exponent >= 31) {
        if (mantissa != 0) return static_cast<uint16_t>(sign | 0x7E00);
        return static_cast<uint16_t>(sign | 0x7C00);
    }

    uint16_t h = static_cast<uint16_t>(sign | (exponent << 10) | (mantissa >> 13));
    if (mantissa & 0x1000) {
        if ((mantissa & 0x1FFF) > 0x1000 || ((mantissa & 0x1FFF) == 0x1000 && (h & 1))) {
            h++;
        }
    }
    return h;
}

inline float halfToFloat(uint16_t h) {
    uint32_t sign = (static_cast<uint32_t>(h) & 0x8000) << 16;
    uint32_t exponent = (h >> 10) & 0x1F;
    uint32_t mantissa = h & 0x03FF;

    uint32_t f;
    if (exponent == 0) {
        if (mantissa == 0) {
            f = sign;
        } else {
            exponent = 1;
            while (!(mantissa & 0x0400)) {
                mantissa <<= 1;
                exponent--;
            }
            mantissa &= 0x03FF;
            f = sign | ((static_cast<uint32_t>(exponent) + 127 - 15) << 23) | (mantissa << 13);
        }
    } else if (exponent == 31) {
        f = sign | 0x7F800000 | (mantissa << 13);
    } else {
        f = sign | ((exponent + 127 - 15) << 23) | (mantissa << 13);
    }

    float result;
    checkedMemcpy(&result, sizeof(result), &f, sizeof(float));
    return result;
}

// ============================================================================
// BFloat16 (BF16) conversion helpers
// ============================================================================

inline uint16_t floatToBf16Bits(float v) {
    uint32_t fbits;
    checkedMemcpy(&fbits, sizeof(fbits), &v, sizeof(float));
    return static_cast<uint16_t>(fbits >> 16);
}

inline float bf16BitsToFloat(uint16_t bfbits) {
    uint32_t fbits = static_cast<uint32_t>(bfbits) << 16;
    float result;
    checkedMemcpy(&result, sizeof(result), &fbits, sizeof(float));
    return result;
}

// ============================================================================
// INT32 conversion helpers
// ============================================================================

inline int32_t floatToInt32(float v) {
    return static_cast<int32_t>(v);
}

inline float int32ToFloat(int32_t v) {
    return static_cast<float>(v);
}

// ============================================================================
// Unified single-value round-trip: float -> dtype -> float
// Used by the CPU golden to match the NPU's effective value representation.
// ============================================================================

inline float roundTripFloat(float v, const std::string& dtype) {
    if (dtype == "FP16") {
        uint32_t fbits;
        checkedMemcpy(&fbits, sizeof(fbits), &v, sizeof(float));
        return halfToFloat(floatToHalf(fbits));
    } else if (dtype == "BF16") {
        return bf16BitsToFloat(floatToBf16Bits(v));
    } else if (dtype == "INT32") {
        return int32ToFloat(floatToInt32(v));
    }
    // FP32: no conversion needed
    return v;
}

// ============================================================================
// Apply round-trip to a vector of float values (in-place)
// Replaces the former ApplyTypeRoundTrip in golden.h.
// ============================================================================

inline void applyTypeRoundTrip(std::vector<float>& values, const std::string& dtype) {
    for (auto& v : values) {
        v = roundTripFloat(v, dtype);
    }
}

// ============================================================================
// Convert float values to target-dtype byte array
// Replaces the former ConvertToDtypeBytes in npu_wrapper.h.
// ============================================================================

inline std::vector<uint8_t> convertFloatToDtypeBytes(
    const std::vector<float>& floatVals, const std::string& dtype)
{
    if (dtype == "FP16") {
        std::vector<uint8_t> out(floatVals.size() * sizeof(uint16_t));
        for (size_t i = 0; i < floatVals.size(); i++) {
            uint32_t fbits;
            checkedMemcpy(&fbits, sizeof(fbits), &floatVals[i], sizeof(float));
            uint16_t hbits = floatToHalf(fbits);
            checkedMemcpy(out.data() + i * sizeof(uint16_t), sizeof(uint16_t),
                          &hbits, sizeof(uint16_t));
        }
        return out;
    } else if (dtype == "BF16") {
        std::vector<uint8_t> out(floatVals.size() * sizeof(uint16_t));
        for (size_t i = 0; i < floatVals.size(); i++) {
            uint16_t bfbits = floatToBf16Bits(floatVals[i]);
            checkedMemcpy(out.data() + i * sizeof(uint16_t), sizeof(uint16_t),
                          &bfbits, sizeof(uint16_t));
        }
        return out;
    } else if (dtype == "INT32") {
        std::vector<uint8_t> out(floatVals.size() * sizeof(int32_t));
        for (size_t i = 0; i < floatVals.size(); i++) {
            int32_t iv = floatToInt32(floatVals[i]);
            checkedMemcpy(out.data() + i * sizeof(int32_t), sizeof(int32_t),
                          &iv, sizeof(int32_t));
        }
        return out;
    }
    // FP32 (default): raw byte copy
    std::vector<uint8_t> out(floatVals.size() * sizeof(float));
    checkedMemcpy(out.data(), out.size(), floatVals.data(), out.size());
    return out;
}

// ============================================================================
// Convert target-dtype byte array back to float vector
// Replaces the former ConvertFromDtypeBytes in npu_wrapper.h.
// ============================================================================

inline std::vector<float> convertDtypeBytesToFloat(
    const uint8_t* data, size_t count, const std::string& dtype)
{
    std::vector<float> out(count);
    if (dtype == "FP16") {
        for (size_t i = 0; i < count; i++) {
            uint16_t hbits;
            checkedMemcpy(&hbits, sizeof(hbits),
                          data + i * sizeof(uint16_t), sizeof(uint16_t));
            out[i] = halfToFloat(hbits);
        }
    } else if (dtype == "BF16") {
        for (size_t i = 0; i < count; i++) {
            uint16_t bfbits;
            checkedMemcpy(&bfbits, sizeof(bfbits),
                          data + i * sizeof(uint16_t), sizeof(uint16_t));
            out[i] = bf16BitsToFloat(bfbits);
        }
    } else if (dtype == "INT32") {
        for (size_t i = 0; i < count; i++) {
            int32_t iv;
            checkedMemcpy(&iv, sizeof(iv),
                          data + i * sizeof(int32_t), sizeof(int32_t));
            out[i] = int32ToFloat(iv);
        }
    } else {
        // FP32 (default): raw byte copy
        checkedMemcpy(out.data(), count * sizeof(float), data, count * sizeof(float));
    }
    return out;
}

}  // namespace sparse_test

#endif  // TEST_CSR2GEBSR_DTYPE_UTILS_H_
