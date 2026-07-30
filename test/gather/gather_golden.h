/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once

#include <vector>
#include "cann_ops_sparse.h"

namespace sparse_test {

template <typename ValT, typename IdxT>
std::vector<ValT> GatherGolden(
    const std::vector<ValT>& Y, const std::vector<IdxT>& indices, aclsparseIndexBase_t idxBase)
{
    std::vector<ValT> output(indices.size());
    for (size_t i = 0; i < indices.size(); i++) {
        int64_t pos = static_cast<int64_t>(indices[i]) - idxBase;
        output[i] = Y[pos];
    }
    return output;
}

} // namespace sparse_test
