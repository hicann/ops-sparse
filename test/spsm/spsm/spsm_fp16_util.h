/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SPSM_FP16_UTIL_H_
#define SPSM_FP16_UTIL_H_

#include <cstdint>

// ============================================================================
// FP32 -> FP16 bit conversion (IEEE 754, round-to-nearest-even).
//
// Retained solely for the L2_X18 exception test case (ValueTypeMismatch),
// which constructs an FP16-valueType CSR descriptor to verify the NOT_SUPPORTED
// rejection path. The operator itself only supports FP32; the golden/wrapper
// FP16 compute paths have been removed as dead code.
// ============================================================================
inline uint16_t SpsmFp32ToFp16Bits(float v) {
    uint32_t bits = __builtin_bit_cast(uint32_t, v);
    uint32_t sign = (bits >> 16) & 0x8000u;
    uint32_t rawExp = (bits >> 23) & 0xFFu;
    int32_t exp = static_cast<int32_t>(rawExp) - 127;
    uint32_t mant = bits & 0x007FFFFFu;
    if (rawExp == 0xFFu) {
        uint32_t nanBit = (mant != 0u) ? 0x0200u : 0u;
        return static_cast<uint16_t>(sign | 0x7C00u | nanBit);
    }
    if (exp >= 16) return static_cast<uint16_t>(sign | 0x7C00u);
    if (exp >= -14) {
        uint32_t mant10 = mant >> 13;
        uint32_t remainder = mant & 0x1FFFu;
        uint32_t result = (static_cast<uint32_t>(exp + 15) << 10) | mant10;
        if (remainder > 0x1000u || (remainder == 0x1000u && (mant10 & 1u))) result += 1u;
        return static_cast<uint16_t>(sign | result);
    }
    if (exp >= -25) {
        uint32_t significand = 0x00800000u | mant;
        uint32_t shift = static_cast<uint32_t>(-exp - 1);
        uint32_t frac = significand >> shift;
        uint32_t remainder = significand & ((1u << shift) - 1u);
        uint32_t half = 1u << (shift - 1u);
        if (remainder > half || (remainder == half && (frac & 1u))) frac += 1u;
        return static_cast<uint16_t>(sign | frac);
    }
    return static_cast<uint16_t>(sign);
}

#endif  // SPSM_FP16_UTIL_H_
