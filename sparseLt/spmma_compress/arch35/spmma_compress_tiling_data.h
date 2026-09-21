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

#ifndef SPARSELT_SPMMA_COMPRESS_TILING_DATA_H_
#define SPARSELT_SPMMA_COMPRESS_TILING_DATA_H_

#include <cstdint>

// 字节数和全局偏移使用 64 位整数；UB 内部尺寸受 UB 容量限制。
struct SpMmaCompressTilingData {
    // order 转换后的物理行列；ld 和 batchStride 的单位为元素。
    uint64_t rows = 0;
    uint64_t cols = 0;
    uint64_t ld = 0;
    uint64_t batchStride = 0;
    uint64_t numBatches = 0;
    uint64_t valuesBytes = 0;
    uint64_t metadataBytes = 0;
    uint64_t inputBytes = 0;
    uint64_t tilesPerRowBand = 0;
    uint64_t rowBandsPerBatch = 0;
    uint64_t totalTiles = 0;
    uint32_t dataBytes = 0;
    uint32_t groupSize = 0;
    // 非零表示沿物理行分组，否则跨相邻物理行组成列向分组。
    uint32_t alongRow = 0;
    uint32_t tileRows = 0;
    uint32_t tileCols = 0;
    // UB 中相邻输入行或输出片段的字节间隔，均按向量宽度对齐。
    uint32_t inputPitch = 0;
    uint32_t valuePitch = 0;
    uint32_t metadataPitch = 0;
    // 三个独立 UB 缓冲区的分配字节数，包含各自的对齐与预留空间。
    uint32_t inputBufferBytes = 0;
    uint32_t valueBufferBytes = 0;
    uint32_t metadataBufferBytes = 0;
};

#endif
