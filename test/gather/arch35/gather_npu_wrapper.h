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

#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "aclsparse_handle_internal.h"
#include "cann_ops_sparse.h"
#include "descriptor_manager.h"

namespace sparse_test {

template <typename ValT, typename IdxT>
std::vector<ValT> GatherNpu(
    HandleManager& handle, const std::vector<ValT>& Y, const std::vector<IdxT>& indices, aclDataType valType,
    aclsparseIndexType_t idxType, aclsparseIndexBase_t idxBase)
{
    auto dY = DeviceBuffer::copyFrom(Y.data(), Y.size() * sizeof(ValT));
    auto dIdx = DeviceBuffer::copyFrom(indices.data(), indices.size() * sizeof(IdxT));
    auto dXValues = DeviceBuffer::alloc(indices.size() * sizeof(ValT));

    auto dnVecY = DnVecManager::createConst(Y.size(), dY.raw(), valType);
    auto spVecX = SpVecManager::create(Y.size(), indices.size(), dIdx.get(), dXValues.get(), idxType, idxBase, valType);

    if (auto st = aclsparseGather(handle.get(), dnVecY.cget(), spVecX.get()); st != ACL_SPARSE_STATUS_SUCCESS) {
        throw std::runtime_error("aclsparseGather failed, status=" + std::to_string(st));
    }
    if (auto st = aclrtSynchronizeStream(handle.get()->stream); st != 0) {
        throw std::runtime_error("aclrtSynchronizeStream failed, status=" + std::to_string(st));
    }

    std::vector<ValT> result(indices.size());
    dXValues.copyToHost(result.data(), indices.size() * sizeof(ValT));
    return result;
}

} // namespace sparse_test
