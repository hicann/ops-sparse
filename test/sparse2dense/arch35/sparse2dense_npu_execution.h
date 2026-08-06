/**
 * ----------------------------------------------------------------------------------------------------------
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under
 * the terms and conditions of CANN Open Software License Agreement Version 2.0
 * (the "License"). Please refer to the License for details. You may not use
 * this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
 * AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
 * FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
 * for the full text of the License.
 * ----------------------------------------------------------------------------------------------------------
 */

#ifndef TEST_SPARSE2DENSE_ARCH35_NPU_EXECUTION_H_
#define TEST_SPARSE2DENSE_ARCH35_NPU_EXECUTION_H_

namespace sparse_test {

inline SpMatManager CreateSpMatFromParam(const Sparse2DenseParam &p,
    void *offsets, void *indices, void *values, int64_t nnz)
{
    const auto base = Sparse2DenseBase(p.base);
    const auto vt = Sparse2DenseValueType(p.value_type);
    if (p.format == "CSR")
        return SpMatManager::createCsr(p.m, p.n, nnz, offsets, indices, values,
            ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_32I, base, vt);
    if (p.format == "CSC")
        return SpMatManager::createCsc(p.m, p.n, nnz, offsets, indices, values,
            ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_32I, base, vt);
    return SpMatManager::createCoo(p.m, p.n, nnz, offsets, indices, values,
        ACL_SPARSE_INDEX_32I, base, vt);
}

inline void CopyHostToDevice(const Sparse2DenseHostInput &host,
    DeviceBuffer &dOffsets, DeviceBuffer &dIndices, DeviceBuffer &dValues)
{
    if (host.offsetCount > 0)
        dOffsets = DeviceBuffer::copyFrom(host.sparseOffsets.data(),
                                           host.sparseOffsets.size());
    if (host.indexCount > 0)
        dIndices = DeviceBuffer::copyFrom(host.sparseIndices.data(),
                                           host.sparseIndices.size());
    if (host.nnz > 0)
        dValues = DeviceBuffer::copyFrom(host.sparseValues.data(),
                                          host.sparseValues.size());
}

inline Sparse2DenseRunResult
RunSparse2Dense(HandleManager &handle, aclrtStream stream,
                const Sparse2DenseParam &p,
                const Sparse2DenseHostInput &host) {
    Sparse2DenseRunResult result;
    const aclDataType valueType = Sparse2DenseValueType(p.value_type);
    const size_t width = Sparse2DenseElemWidth(valueType);
    const bool rowMajor = p.order == "ROW";
    const size_t physicalElems =
        static_cast<size_t>(rowMajor ? p.m : p.n) * static_cast<size_t>(p.ld);
    const size_t dnBytes = physicalElems * width;

    handle.setStream(stream);

    DeviceBuffer dOffsets, dIndices, dValues, dDense;
    CopyHostToDevice(host, dOffsets, dIndices, dValues);
    dDense = DeviceBuffer::alloc(std::max<size_t>(dnBytes, 1));

    SpMatManager spMat;
    DnMatManager dnMat;
    try {
        spMat = CreateSpMatFromParam(p, dOffsets.get(), dIndices.get(),
                                      dValues.get(), host.nnz);
        dnMat = DnMatManager::create(p.m, p.n, p.ld, dDense.get(), valueType,
                                     Sparse2DenseOrder(p.order));
    } catch (const std::runtime_error &) {
        result.descriptorStatus = ACL_SPARSE_STATUS_INVALID_VALUE;
        return result;
    }

    result.queryStatus = aclsparseSparseToDense_bufferSize(
        handle.get(), spMat.get(), dnMat.get(),
        ACL_SPARSE_SPARSETODENSE_ALG_DEFAULT, &result.workspaceSize);
    if (result.queryStatus != ACL_SPARSE_STATUS_SUCCESS)
        return result;

    result.executeStatus = aclsparseSparseToDense(
        handle.get(), spMat.get(), dnMat.get(),
        ACL_SPARSE_SPARSETODENSE_ALG_DEFAULT, nullptr);
    if (result.executeStatus != ACL_SPARSE_STATUS_SUCCESS)
        return result;

    result.syncStatus = aclrtSynchronizeStream(stream);
    if (result.syncStatus != ACL_SUCCESS)
        return result;

    result.dense.resize(std::max<size_t>(dnBytes, 1));
    dDense.copyToHost(result.dense.data(), result.dense.size());
    return result;
}

} // namespace sparse_test
#endif
