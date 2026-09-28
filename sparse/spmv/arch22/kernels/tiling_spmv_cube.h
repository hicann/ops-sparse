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
#pragma once

#include <cstdint>

namespace tcuscan {

/**
 * @brief `spmv` kernel tiling parameter structure.
 */
struct SpMVCubeTiling {
  /// @brief Number of non-zeros elements.
  uint32_t nnz;
  /// @brief Total number of segments.
  uint32_t num_segments;
  /// @brief Length of the dense input vector.
  uint32_t x_len;
  /// @brief Tiling length.
  uint32_t tile_len;
  /// @brief Block length.
  uint32_t block_len;
  /// @brief Scaling factor applied to the SpMV product, i.e. the `alpha` of
  /// `y = alpha * A @ x + beta * y`.
  float alpha;
  /// @brief Scaling factor applied to the incoming output vector, i.e. the
  /// `beta` of `y = alpha * A @ x + beta * y`.
  float beta;
};

} // namespace tcuscan
