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

#ifndef SPSM_TILING_DATA_H_
#define SPSM_TILING_DATA_H_

#include <cstdint>

// SpsmTilingData
// TilingData 全程在 host 构造, by-value 传递给 Solve kernel,
// 不落盘 workspace GM, 不经 H2D 拷贝。L 字段由 host level scheduling 计算
// 并直接填入。
struct SpsmTilingData {
    int32_t m;                  // A 的行/列数（方阵）
    int32_t n;                  // B/C 的列数（nrhs）
    int32_t ldb;                // B 的 leading dimension
    int32_t ldc;                // C 的 leading dimension
    int32_t orderB;             // 0=ROW, 1=COL
    int32_t orderC;             // 0=ROW, 1=COL
    int32_t needTranspose;      // 1=opA==T，使用转置 CSR
    int32_t effectiveFillMode;  // 0=LOWER, 1=UPPER（T 模式已 swap）
    int32_t diagType;           // 0=NON_UNIT, 1=UNIT
    int32_t L;                  // level 总数 (host level scheduling 计算填充)
    int32_t kChunkSize;         // n 维列块大小（UB 向量化单元）
    int32_t maxRowLen;          // 最大行非零元数（UB buffer 依据）
    float   alpha_host;         // alpha 标量（FP32）
    int32_t indexBase;          // 0=ZERO, 1=ONE（kernel 读 colInd 时减去此值, 归一化为 ZERO）
    // workspace 区偏移（字节）
    int64_t levelRowPtrOff;     // levelRowPtr[L+1] 偏移
    int64_t levelRowIdxOff;     // levelRowIdx[m] 偏移
    int64_t diagValOff;         // diagVal[m] 偏移（0=不用）
    int64_t transRowOffOff;     // 转置 CSR rowOff 偏移
    int64_t transColIndOff;     // 转置 CSR colInd 偏移
    int64_t transValOff;        // 转置 CSR values 偏移
    int64_t denseBufOff;        // denseBuf[m*n] 偏移（COL-order 转置缓冲区，ROW-order 存储；未分配时为 0）
};

#endif // SPSM_TILING_DATA_H_
