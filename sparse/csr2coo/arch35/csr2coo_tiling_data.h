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
 * @file csr2coo_tiling_data.h
 * @brief aclsparseXcsr2coo TilingData structure definition (shared between Host and Kernel).
 */

#ifndef CSR2COO_TILING_DATA_H_
#define CSR2COO_TILING_DATA_H_

#include <cstdint>

namespace csr2coo {

// SIMT 路径每核线程数 = 1024（最终值，P2-2 确定）。
// 约束：__launch_bounds__ 与 dim3 必须共用同一编译期常量，故置于共享头供
// Kernel（__launch_bounds__/asc_vf_call）引用。Host 不再存储 nthreads 字段
// （record-only，Kernel 不读），直接使用此常量即可。
// 选择理由：1024 线程，实测 2048 在小矩阵上大量空转但仍消耗调度开销，退化 6%~53%；
// 1024 在各矩阵规模下性能稳定，故确定为最终值。
constexpr uint32_t kCsr2CooSimtThreads = 1024;

struct Csr2CooTilingData {
    int64_t m{0};             // Number of rows in the matrix
    int64_t nnz{0};           // Number of non-zero elements (for kernel boundary check)
    int32_t idxBase{0};       // Index base (0 or 1), stored as int32_t (kernel has no enum)
    int64_t blockSize{0};     // SIMD path: base rows per core = m / numBlocks (floor)
    uint32_t remainder{0};    // SIMD path: first `remainder` cores handle 1 extra row
    uint32_t cooChunkSize{0}; // SIMD path: cooRowInd UB chunk size (elements, 32B aligned)
    uint32_t rowPtrBytes{0};  // SIMD path: rowPtrBuf_ allocation bytes (32B aligned)
    // ---- SIMT hybrid path ----
    uint32_t useSimt{0};      // Path select: 1=SIMT (small-row, thread-parallel), 0=SIMD (large-row)
    uint32_t simtRowsPerBlock{0}; // SIMT path: rows per core for inter-core split (ceil(m/numBlocks))
};

} // namespace csr2coo

#endif // CSR2COO_TILING_DATA_H_
