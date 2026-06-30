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

#include <cstdint>
#include <iostream>
#include <memory>
#include <new>
#include <vector>
#include <cmath>

#include "acl/acl.h"
#include "tiling/platform/platform_ascendc.h"
#include "cann_ops_sparse.h"
#include "spmv.h"
#include "spmv_tiling_data.h"

// -------------------------------------------------------------------
// 前向声明：AscendC 内核启动分发器（定义在 kernels/spmv_kernel.cpp）
// -------------------------------------------------------------------
extern "C" void spmv_kernel_do(
    GM_ADDR csrRowPtr, GM_ADDR csrColInd, GM_ADDR csrVal,
    GM_ADDR xVec, GM_ADDR yVec, SpmvTilingData tiling,
    const void *alpha, const void *beta,
    int32_t cType, int32_t vType, int32_t oType,
    bool trans, uint32_t numBlocks, void *stream);

// -------------------------------------------------------------------
// 内部辅助函数
// -------------------------------------------------------------------
namespace {

    /**
     * @brief 获取 UB 容量下允许的最大单行长（非零元个数）
     */
    template <typename T>
    uint32_t ComputeMaxRowLength() {
        constexpr uint64_t kYLocalBytes = 32;
        constexpr uint64_t kSystemReserved = 4 * 1024;
        constexpr uint64_t kBufferCount = 5;
        constexpr uint64_t kAlignBytes = 32;
        constexpr uint64_t kAlignSlack = kBufferCount * kAlignBytes;

        auto platform = platform_ascendc::PlatformAscendCManager::GetInstance();
        uint64_t ubSize = 0;
        platform->GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
        CHECK_RET(ubSize > 0,
                  LOG_PRINT("[ERROR] GetCoreMemSize(UB) failed\n");
                  return 0);

        constexpr uint64_t perElem =
            sizeof(int32_t)  // colIdx
            + sizeof(T)      // vals
            + sizeof(T)      // xLocal
            + sizeof(float)  // workReduce
            + sizeof(float); // floatTmp

        constexpr uint64_t kFixed = kYLocalBytes + kSystemReserved + kAlignSlack;

        if (ubSize <= kFixed)
            return 0;

        uint64_t maxLen = (ubSize - kFixed) / perElem;

        constexpr uint64_t alignElems =
            kAlignBytes / sizeof(T) > 8 ? kAlignBytes / sizeof(T) : 8;
        maxLen = (maxLen / alignElems) * alignElems;

        return static_cast<uint32_t>(maxLen);
    }

    /**
     * @brief 从 device 侧 rowPtr 计算最大行长
     */
    uint32_t ComputeMaxRowLengthFromDevice(void *csrRowPtrDevice, uint64_t rows, aclrtStream stream) {
        if (rows == 0)
            return 0;

        std::vector<int32_t> rowPtrHost(rows + 1);
        aclError ret = aclrtMemcpy(rowPtrHost.data(),
                                   (rows + 1) * sizeof(int32_t),
                                   csrRowPtrDevice,
                                   (rows + 1) * sizeof(int32_t),
                                   ACL_MEMCPY_DEVICE_TO_HOST);
        CHECK_RET(ret == ACL_SUCCESS,
                  LOG_PRINT("[ERROR] ComputeMaxRowLengthFromDevice: aclrtMemcpy D2H failed, ret=%d\n", ret);
                  return 0);

        ret = aclrtSynchronizeStream(stream);
        CHECK_RET(ret == ACL_SUCCESS,
                  LOG_PRINT("[ERROR] ComputeMaxRowLengthFromDevice: sync failed, ret=%d\n", ret);
                  return 0);

        uint32_t maxLen = 0;
        for (uint64_t i = 0; i < rows; i++) {
            uint32_t rowLen = rowPtrHost[i + 1] - rowPtrHost[i];
            if (rowLen > maxLen)
                maxLen = rowLen;
        }
        return maxLen;
    }

} // namespace

// -------------------------------------------------------------------
// Public API Implementation
// -------------------------------------------------------------------

extern "C" {
    aclsparseStatus_t aclsparseSpMV(aclsparseHandle_t handle,
                                    aclsparseOperation_t opA,
                                    const void *alpha,
                                    aclsparseConstSpMatDescr_t matA,
                                    aclsparseConstDnVecDescr_t vecX,
                                    const void *beta,
                                    aclsparseDnVecDescr_t vecY,
                                    aclDataType computeType,
                                    aclsparseSpMVAlg_t alg,
                                    void *externalBuffer) {
        // ==================== 参数校验 ====================
        CHECK_RET(handle != nullptr,
                  LOG_PRINT("[ERROR] aclsparseSpMV: handle is nullptr\n");
                  return ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR);
        CHECK_RET(matA != nullptr,
                  LOG_PRINT("[ERROR] aclsparseSpMV: matA is nullptr\n");
                  return ACL_SPARSE_STATUS_INVALID_VALUE);
        CHECK_RET(vecX != nullptr,
                  LOG_PRINT("[ERROR] aclsparseSpMV: vecX is nullptr\n");
                  return ACL_SPARSE_STATUS_INVALID_VALUE);
        CHECK_RET(vecY != nullptr,
                  LOG_PRINT("[ERROR] aclsparseSpMV: vecY is nullptr\n");
                  return ACL_SPARSE_STATUS_INVALID_VALUE);
        CHECK_RET(alg == ACL_SPARSE_SPMV_ALG_DEFAULT,
                  LOG_PRINT("[ERROR] aclsparseSpMV: only default algorithm is supported currently, got alg=%d\n", alg);
                  return ACL_SPARSE_STATUS_NOT_SUPPORTED);

        CHECK_RET(computeType == ACL_FLOAT || computeType == ACL_INT32,
                  LOG_PRINT("[ERROR] aclsparseSpMV: unsupported computeType %d\n", computeType);
                  return ACL_SPARSE_STATUS_NOT_SUPPORTED);

        // ==================== 解包描述符 ====================
        auto *h = ToInternalHandle(handle);
        auto *matInner = ToMatInner(matA);
        auto *xInner = ToVecInner(vecX);
        auto *yInner = ToVecInner(vecY);

        CHECK_RET(matInner->format == ACL_SPARSE_FORMAT_CSR,
                  LOG_PRINT("[ERROR] aclsparseSpMV: unsupported matrix format %d\n", matInner->format);
                  return ACL_SPARSE_STATUS_NOT_SUPPORTED);

        {
            aclsparseStatus_t idxSt =
                AclsparseValidateSupportedCsrIndexTypes(matInner->ptrType, matInner->IdxType);
            CHECK_RET(idxSt == ACL_SPARSE_STATUS_SUCCESS,
                      LOG_PRINT("[ERROR] aclsparseSpMV: unsupported index type ptr=%d idx=%d "
                                "(only ACL_SPARSE_INDEX_32I)\n",
                                matInner->ptrType, matInner->IdxType);
                      return idxSt);
        }

        CHECK_RET(opA == ACL_SPARSE_OP_NON_TRANSPOSE || opA == ACL_SPARSE_OP_TRANSPOSE,
                  LOG_PRINT("[ERROR] aclsparseSpMV: opA conjugate not supported yet\n");
                  return ACL_SPARSE_STATUS_NOT_SUPPORTED);

        // ==================== Alpha / Beta ====================
        // alpha/beta 类型必须与 computeType 一致（任务书要求），调用方负责保证
        float alphaFloat = 1.0f;
        float betaFloat = 0.0f;
        int32_t alphaInt = 1;
        int32_t betaInt = 0;

        if (alpha != nullptr) {
            if (computeType == ACL_FLOAT)
                alphaFloat = *static_cast<const float *>(alpha);
            else
                alphaInt = *static_cast<const int32_t *>(alpha);
        }
        if (beta != nullptr) {
            if (computeType == ACL_FLOAT)
                betaFloat = *static_cast<const float *>(beta);
            else
                betaInt = *static_cast<const int32_t *>(beta);
        }

        // ==================== Stream ====================
        aclrtStream stream = h->stream;
        CHECK_RET(stream != nullptr,
                  LOG_PRINT("[ERROR] aclsparseSpMV: stream is nullptr, please call aclsparseSetStream first\n");
                  return ACL_SPARSE_STATUS_INVALID_VALUE);

        // ==================== 平台参数 ====================
        auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
        uint32_t blockDim = ascendcPlatform->GetCoreNumAiv();
        CHECK_RET(blockDim > 0,
                  LOG_PRINT("[ERROR] aclsparseSpMV: GetCoreNumAiv returned 0\n");
                  return ACL_SPARSE_STATUS_INTERNAL_ERROR);

        // ==================== 提取矩阵参数 ====================
        uint64_t rows = matInner->rows;
        uint64_t cols = matInner->cols;

        void *csrRowPtrDevice = matInner->ptrs;
        void *csrColIndDevice = matInner->idxs;
        void *csrValDevice = matInner->values;

        CHECK_RET(csrRowPtrDevice != nullptr && csrColIndDevice != nullptr && csrValDevice != nullptr,
                  LOG_PRINT("[ERROR] aclsparseSpMV: matrix device pointers are null\n");
                  return ACL_SPARSE_STATUS_INVALID_VALUE);
        CHECK_RET(xInner->values != nullptr && yInner->values != nullptr,
                  LOG_PRINT("[ERROR] aclsparseSpMV: vector device pointers are null\n");
                  return ACL_SPARSE_STATUS_INVALID_VALUE);
        CHECK_RET(xInner->nums >= (opA == ACL_SPARSE_OP_TRANSPOSE ? static_cast<uint64_t>(rows) : static_cast<uint64_t>(cols)),
                  LOG_PRINT("[ERROR] aclsparseSpMV: vecX size %lu < required %lu\n",
                            xInner->nums,
                            (opA == ACL_SPARSE_OP_TRANSPOSE ? static_cast<uint64_t>(rows) : static_cast<uint64_t>(cols)));
                  return ACL_SPARSE_STATUS_INVALID_VALUE);
        CHECK_RET(yInner->nums >= (opA == ACL_SPARSE_OP_TRANSPOSE ? static_cast<uint64_t>(cols) : rows),
                  LOG_PRINT("[ERROR] aclsparseSpMV: vecY size %lu < required %lu\n",
                            yInner->nums,
                            (opA == ACL_SPARSE_OP_TRANSPOSE ? static_cast<uint64_t>(cols) : rows));
                  return ACL_SPARSE_STATUS_INVALID_VALUE);

        // ==================== UB 容量检查 ====================
        uint32_t maxRowLength = ComputeMaxRowLengthFromDevice(csrRowPtrDevice, rows, stream);
        uint32_t maxTileLength = (computeType == ACL_FLOAT) ? ComputeMaxRowLength<float>() : ComputeMaxRowLength<int32_t>();

        CHECK_RET(maxTileLength > 0,
                  LOG_PRINT("[ERROR] aclsparseSpMV: failed to compute max tile length\n");
                  return ACL_SPARSE_STATUS_INTERNAL_ERROR);
        CHECK_RET(maxRowLength <= maxTileLength,
                  LOG_PRINT("[ERROR] aclsparseSpMV: max row length %u exceeds UB capacity %u\n",
                            maxRowLength, maxTileLength);
                  return ACL_SPARSE_STATUS_INSUFFICIENT_RESOURCES);

        // ==================== Tiling 数据准备 ====================
        SpmvTilingData tilingHost = {
            static_cast<uint32_t>(rows),
            static_cast<uint32_t>(cols)};

        // ==================== 初始化输出向量 y（beta == 0 时清零） ====================
        size_t elemSize = (computeType == ACL_FLOAT) ? sizeof(float) : sizeof(int32_t);
        size_t yByteSize = (opA == ACL_SPARSE_OP_TRANSPOSE ? cols : rows) * elemSize;

        // ==================== 启动内核 ====================
        auto valType = xInner->valueType;
        auto outType = yInner->valueType;
        bool trans = (opA == ACL_SPARSE_OP_TRANSPOSE);

        int32_t cType, vType, oType;
        SpmvTypesFromAcl(computeType, valType, outType, &cType, &vType, &oType);

        const void *alphaPtr = (computeType == ACL_FLOAT) ?
                                   static_cast<const void *>(&alphaFloat) :
                                   static_cast<const void *>(&alphaInt);
        const void *betaPtr = (computeType == ACL_FLOAT) ?
                                  static_cast<const void *>(&betaFloat) :
                                  static_cast<const void *>(&betaInt);

        spmv_kernel_do(
            static_cast<GM_ADDR>(csrRowPtrDevice), static_cast<GM_ADDR>(csrColIndDevice),
            static_cast<GM_ADDR>(csrValDevice), static_cast<GM_ADDR>(xInner->values),
            static_cast<GM_ADDR>(yInner->values), tilingHost,
            alphaPtr, betaPtr,
            cType, vType, oType, trans, blockDim, stream);

        return ACL_SPARSE_STATUS_SUCCESS;
    }

} // extern "C"
