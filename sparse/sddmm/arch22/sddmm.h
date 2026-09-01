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

#ifndef SDDMM_ARCH22_H
#define SDDMM_ARCH22_H

#include <cstdint>

#ifndef GM_ADDR
#define GM_ADDR uint8_t *
#endif

struct SddmmTilingData {
    uint32_t nnz;
    uint32_t rowCount;
    uint32_t matARows;
    uint32_t matACols;
    uint32_t matBRows;
    uint32_t matBCols;
    uint32_t blockDim;
    uint32_t rowsPerCore;
    uint32_t remainderRows;
    uint32_t kBlockLen;
    uint32_t nnzPartition;
    uint32_t nnzPerCore;
    uint32_t remainderNnz;
    float alphaHost;
    float betaHost;
    uint32_t opX;
    uint32_t opY;
    uint32_t ldx;
    uint32_t ldy;
    uint32_t orderX;
    uint32_t orderY;
    uint32_t dataType;
};

#ifdef HOST_SDDMM

#include "acl/acl.h"
#include "aclsparse_descr_internal.h"
#include "aclsparse_handle_internal.h"

inline struct aclsparseContext *SddmmToInternalHandle(aclsparseHandle_t handle) {
    return reinterpret_cast<struct aclsparseContext *>(handle);
}

inline struct aclsparseSpMatDescr *SddmmToSpMatInner(aclsparseSpMatDescr_t desc) {
    return reinterpret_cast<struct aclsparseSpMatDescr *>(desc);
}

inline struct aclsparseDnMatDescr *SddmmToDnMatInner(aclsparseConstDnMatDescr_t desc) {
    return const_cast<struct aclsparseDnMatDescr *>(
        reinterpret_cast<const struct aclsparseDnMatDescr *>(desc));
}

inline float SddmmReadScalarToF32(const void *ptr, aclDataType computeType) {
    if (ptr == nullptr) {
        return 0.0f;
    }
    if (computeType == ACL_FLOAT16) {
        uint16_t bits;
        __builtin_memcpy(&bits, ptr, sizeof(uint16_t));
        uint32_t sign = static_cast<uint32_t>(bits & 0x8000u) << 16;
        uint32_t exp = (bits >> 10) & 0x1Fu;
        uint32_t mant = bits & 0x03FFu;
        uint32_t f;
        if (exp == 0) {
            if (mant == 0) {
                f = sign;
            } else {
                uint32_t shift = __builtin_clz(mant) - 21;
                mant <<= shift;
                exp = 1;
                f = sign | ((exp + 127 - 15 - shift) << 23) | ((mant << 13) & 0x7FFFFFu);
            }
        } else if (exp == 31) {
            f = sign | 0x7F800000u | (mant << 13);
        } else {
            f = sign | ((exp + 127 - 15) << 23) | (mant << 13);
        }
        float v;
        __builtin_memcpy(&v, &f, sizeof(float));
        return v;
    }
    float result;
    __builtin_memcpy(&result, ptr, sizeof(float));
    return result;
}

#endif // HOST_SDDMM

#endif // SDDMM_ARCH22_H
