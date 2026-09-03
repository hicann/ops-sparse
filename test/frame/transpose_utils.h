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

#ifndef TEST_FRAME_TRANSPOSE_UTILS_H_
#define TEST_FRAME_TRANSPOSE_UTILS_H_

// =============================================================================
// Shared transpose utility used by prune and ltmatmul test wrappers.
// Extracted to eliminate duplicate TransposeToRowMajor definitions
// (prune_npu_wrapper.h vs ltmatmul_test_utils.h).
// =============================================================================

#include <cstddef>
#include <cstdint>
#include <vector>

namespace sparse_test {

// Transpose a row-major (origRows x origCols) matrix into a row-major
// (origCols x dstLd) matrix. dstLd >= origRows must hold for the destination
// to fit. The output is zero-padded when dstLd > origRows.
template <typename T>
inline std::vector<T> TransposeToRowMajor(const std::vector<T>& src,
    int32_t origRows, int32_t origCols, int64_t dstLd)
{
    std::vector<T> dst(static_cast<size_t>(origCols) * static_cast<size_t>(dstLd));
    for (int32_t j = 0; j < origCols; ++j) {
        for (int32_t i = 0; i < origRows; ++i) {
            dst[static_cast<size_t>(j) * dstLd + i] = src[static_cast<size_t>(i) * origCols + j];
        }
    }
    return dst;
}

}  // namespace sparse_test

#endif  // TEST_FRAME_TRANSPOSE_UTILS_H_
