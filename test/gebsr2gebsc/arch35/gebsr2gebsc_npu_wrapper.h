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

#ifndef GEBSR2GEBSC_NPU_WRAPPER_H_
#define GEBSR2GEBSC_NPU_WRAPPER_H_

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <vector>

#include "acl/acl.h"
#include "cann_ops_sparse.h"
#include "descriptor_manager.h"

using sparse_test::DeviceBuffer;

inline size_t GebsrValSize(const std::string& typeStr) {
    if (typeStr == "FP32" || typeStr == "INT32") return 4;
    if (typeStr == "FP16" || typeStr == "BF16") return 2;
    if (typeStr == "INT8") return 1;
    std::cerr << "[NPU] unknown val_type: " << typeStr << ", defaulting to 4" << std::endl;
    return 4;
}

inline aclDataType GebsrAclDataType(const std::string& typeStr) {
    if (typeStr == "FP32") return ACL_FLOAT;
    if (typeStr == "FP16") return ACL_FLOAT16;
    if (typeStr == "BF16") return ACL_BF16;
    if (typeStr == "INT32") return ACL_INT32;
    if (typeStr == "INT8") return ACL_INT8;
    return ACL_FLOAT;
}

struct Gebsr2GebscNpuResult {
    std::vector<int32_t> bscColPtr;
    std::vector<int32_t> bscRowInd;
    std::vector<uint8_t> bscVal;
    int32_t nnzb;

    aclsparseStatus_t bufferSizeRet;
    aclsparseStatus_t computeRet;
};

inline void CopyGebsrResultToHost(
    DeviceBuffer& dBscColPtr, DeviceBuffer& dBscRowInd, DeviceBuffer& dBscVal,
    int nb, int nnzb, size_t blockSizeC, size_t valSize, Gebsr2GebscNpuResult& result)
{
    result.bscColPtr.resize(nb + 1);
    dBscColPtr.copyToHost(result.bscColPtr.data(), (nb + 1) * sizeof(int32_t));
    if (nnzb > 0) {
        result.bscRowInd.resize(nnzb);
        dBscRowInd.copyToHost(result.bscRowInd.data(), nnzb * sizeof(int32_t));
        size_t valBytes = static_cast<size_t>(nnzb) * blockSizeC * valSize;
        result.bscVal.resize(valBytes);
        dBscVal.copyToHost(result.bscVal.data(), valBytes);
    }
}

struct Gebsr2GebscDeviceBuffers {
    DeviceBuffer dRowPtr;
    DeviceBuffer dColInd;
    DeviceBuffer dVal;
    DeviceBuffer dBscColPtr;
    DeviceBuffer dBscRowInd;
    DeviceBuffer dBscVal;
    size_t valSize;
    int blockSizeC;
};

inline Gebsr2GebscDeviceBuffers AllocGebsr2GebscDeviceBuffers(
    int mb, int nb, int nnzb,
    const uint8_t* bsrValAHost,
    const int32_t* bsrRowPtrAHost,
    const int32_t* bsrColIndAHost,
    int rowBlockDimA, int colBlockDimA,
    int rowBlockDimC, int colBlockDimC,
    const std::string& valType,
    aclsparseAction_t copyValues)
{
    Gebsr2GebscDeviceBuffers buf;
    buf.valSize = GebsrValSize(valType);
    int blockSizeA = rowBlockDimA * colBlockDimA;
    buf.blockSizeC = rowBlockDimC * colBlockDimC;

    buf.dRowPtr = DeviceBuffer::copyFrom(bsrRowPtrAHost, (mb + 1) * sizeof(int32_t));
    size_t colIndBytes = std::max(static_cast<size_t>(nnzb) * sizeof(int32_t), size_t(1));
    size_t valInBytes = std::max(static_cast<size_t>(nnzb) * blockSizeA * buf.valSize, size_t(1));
    size_t valOutBytes = std::max(static_cast<size_t>(nnzb) * buf.blockSizeC * buf.valSize, size_t(1));

    buf.dColInd = (nnzb > 0)
        ? DeviceBuffer::copyFrom(bsrColIndAHost, static_cast<size_t>(nnzb) * sizeof(int32_t))
        : DeviceBuffer::alloc(colIndBytes);
    buf.dVal = (nnzb > 0 && bsrValAHost != nullptr)
        ? DeviceBuffer::copyFrom(bsrValAHost, static_cast<size_t>(nnzb) * blockSizeA * buf.valSize)
        : DeviceBuffer::alloc(valInBytes);
    buf.dBscColPtr = DeviceBuffer::alloc((nb + 1) * sizeof(int32_t));
    buf.dBscRowInd = DeviceBuffer::alloc(colIndBytes);
    buf.dBscVal = DeviceBuffer::alloc(valOutBytes);

    if (copyValues == ACL_SPARSE_ACTION_SYMBOLIC && nnzb > 0) {
        std::vector<uint8_t> magic(valOutBytes, 0xDE);
        aclrtMemcpy(buf.dBscVal.get(), valOutBytes, magic.data(), valOutBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    }
    return buf;
}

inline Gebsr2GebscNpuResult Gebsr2GebscNpu(
    sparse_test::HandleManager& handle, aclrtStream stream,
    const std::string& valType,
    int mb, int nb, int nnzb,
    const uint8_t* bsrValAHost,
    const int32_t* bsrRowPtrAHost,
    const int32_t* bsrColIndAHost,
    int rowBlockDimA, int colBlockDimA,
    int rowBlockDimC, int colBlockDimC,
    aclsparseAction_t copyValues, aclsparseIndexBase_t idxBase,
    aclsparseDirection_t dirA)
{
    Gebsr2GebscNpuResult result{};
    result.nnzb = nnzb;

    auto buf = AllocGebsr2GebscDeviceBuffers(
        mb, nb, nnzb, bsrValAHost, bsrRowPtrAHost, bsrColIndAHost,
        rowBlockDimA, colBlockDimA, rowBlockDimC, colBlockDimC,
        valType, copyValues);

    size_t workspaceSize = 0;
    result.bufferSizeRet = aclsparseGebsr2gebsc_bufferSize(
        handle.get(), mb, nb, nnzb,
        buf.dVal.get(),
        reinterpret_cast<const int*>(buf.dRowPtr.get()),
        reinterpret_cast<const int*>(buf.dColInd.get()),
        rowBlockDimA, colBlockDimA, dirA,
        GebsrAclDataType(valType), &workspaceSize);
    if (result.bufferSizeRet != ACL_SPARSE_STATUS_SUCCESS) {
        std::cerr << "[NPU] gebsr2gebsc_bufferSize failed: " << result.bufferSizeRet << std::endl;
        return result;
    }
    if (workspaceSize == 0) workspaceSize = 1;
    auto dWorkspace = DeviceBuffer::alloc(workspaceSize);

    result.computeRet = aclsparseGebsr2gebsc(
        handle.get(), mb, nb, nnzb,
        buf.dVal.get(),
        reinterpret_cast<const int*>(buf.dRowPtr.get()),
        reinterpret_cast<const int*>(buf.dColInd.get()),
        rowBlockDimA, colBlockDimA,
        buf.dBscVal.get(),
        reinterpret_cast<int*>(buf.dBscColPtr.get()),
        reinterpret_cast<int*>(buf.dBscRowInd.get()),
        rowBlockDimC, colBlockDimC,
        copyValues, idxBase, dirA,
        GebsrAclDataType(valType), dWorkspace.get());
    if (result.computeRet != ACL_SPARSE_STATUS_SUCCESS) {
        std::cerr << "[NPU] gebsr2gebsc failed: " << result.computeRet << std::endl;
        return result;
    }

    if (aclrtSynchronizeStream(stream) != ACL_SUCCESS) {
        std::cerr << "[NPU] aclrtSynchronizeStream failed" << std::endl;
        result.computeRet = ACL_SPARSE_STATUS_EXECUTION_FAILED;
        return result;
    }
    CopyGebsrResultToHost(buf.dBscColPtr, buf.dBscRowInd, buf.dBscVal,
        nb, nnzb, static_cast<size_t>(buf.blockSizeC), buf.valSize, result);
    return result;
}

#endif  // GEBSR2GEBSC_NPU_WRAPPER_H_
