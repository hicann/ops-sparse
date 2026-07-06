/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TEST_FRAME_DESCRIPTOR_MANAGER_H_
#define TEST_FRAME_DESCRIPTOR_MANAGER_H_

#include <cstdint>
#include <iostream>
#include <stdexcept>

#include "acl/acl.h"
#include "cann_ops_sparse.h"

namespace sparse_test {

class SpMatManager {
public:
    SpMatManager() = default;

    static SpMatManager createCsr(int64_t rows, int64_t cols, int64_t nnz,
                                   void* rowOffsets, void* colIndices, void* values,
                                   aclsparseIndexType_t rowOffsetType = ACL_SPARSE_INDEX_32I,
                                   aclsparseIndexType_t colIdxType = ACL_SPARSE_INDEX_32I,
                                   aclsparseIndexBase_t idxBase = ACL_SPARSE_INDEX_BASE_ZERO,
                                   aclDataType valueType = ACL_FLOAT) {
        SpMatManager m;
        auto s = aclsparseCreateCsr(&m.descr_, rows, cols, nnz, rowOffsets, colIndices,
                                    values, rowOffsetType, colIdxType, idxBase, valueType);
        if (s != ACL_SPARSE_STATUS_SUCCESS) throw std::runtime_error("aclsparseCreateCsr failed");
        return m;
    }

    static SpMatManager createConstCsr(int64_t rows, int64_t cols, int64_t nnz,
                                        const void* rowOffsets, const void* colIndices, const void* values,
                                        aclsparseIndexType_t rowOffsetType = ACL_SPARSE_INDEX_32I,
                                        aclsparseIndexType_t colIdxType = ACL_SPARSE_INDEX_32I,
                                        aclsparseIndexBase_t idxBase = ACL_SPARSE_INDEX_BASE_ZERO,
                                        aclDataType valueType = ACL_FLOAT) {
        SpMatManager m;
        aclsparseConstSpMatDescr_t constDescr = nullptr;
        auto s = aclsparseCreateConstCsr(&constDescr, rows, cols, nnz, rowOffsets, colIndices,
                                         values, rowOffsetType, colIdxType, idxBase, valueType);
        if (s != ACL_SPARSE_STATUS_SUCCESS) throw std::runtime_error("aclsparseCreateConstCsr failed");
        m.descr_ = const_cast<aclsparseSpMatDescr_t>(constDescr);
        return m;
    }

    aclsparseSpMatDescr_t get() { return descr_; }
    aclsparseConstSpMatDescr_t cget() const { return descr_; }

    ~SpMatManager() {
        if (descr_) aclsparseDestroySpMat(descr_);
    }

    SpMatManager(const SpMatManager&) = delete;
    SpMatManager& operator=(const SpMatManager&) = delete;
    SpMatManager(SpMatManager&& other) noexcept : descr_(other.descr_) {
        other.descr_ = nullptr;
    }
    SpMatManager& operator=(SpMatManager&& other) noexcept {
        if (this != &other) {
            if (descr_) aclsparseDestroySpMat(descr_);
            descr_ = other.descr_;
            other.descr_ = nullptr;
        }
        return *this;
    }

private:
    aclsparseSpMatDescr_t descr_ = nullptr;
};

class DnVecManager {
public:
    DnVecManager() = default;

    static DnVecManager create(int64_t size, void* values, aclDataType valueType) {
        DnVecManager m;
        auto s = aclsparseCreateDnVec(&m.descr_, size, values, valueType);
        if (s != ACL_SPARSE_STATUS_SUCCESS) throw std::runtime_error("aclsparseCreateDnVec failed");
        return m;
    }

    static DnVecManager createConst(int64_t size, const void* values, aclDataType valueType) {
        DnVecManager m;
        aclsparseConstDnVecDescr_t constDescr = nullptr;
        auto s = aclsparseCreateConstDnVec(&constDescr, size, values, valueType);
        if (s != ACL_SPARSE_STATUS_SUCCESS) throw std::runtime_error("aclsparseCreateConstDnVec failed");
        m.descr_ = const_cast<aclsparseDnVecDescr_t>(constDescr);
        return m;
    }

    aclsparseDnVecDescr_t get() { return descr_; }
    aclsparseConstDnVecDescr_t cget() const { return descr_; }

    ~DnVecManager() {
        if (descr_) aclsparseDestroyDnVec(descr_);
    }

    DnVecManager(const DnVecManager&) = delete;
    DnVecManager& operator=(const DnVecManager&) = delete;
    DnVecManager(DnVecManager&& other) noexcept : descr_(other.descr_) { other.descr_ = nullptr; }
    DnVecManager& operator=(DnVecManager&& other) noexcept {
        if (this != &other) {
            if (descr_) aclsparseDestroyDnVec(descr_);
            descr_ = other.descr_;
            other.descr_ = nullptr;
        }
        return *this;
    }

private:
    aclsparseDnVecDescr_t descr_ = nullptr;
};

class DnMatManager {
public:
    DnMatManager() = default;

    static DnMatManager create(int64_t rows, int64_t cols, int64_t ld,
                                void* values, aclDataType valueType, aclsparseOrder_t order) {
        DnMatManager m;
        auto s = aclsparseCreateDnMat(&m.descr_, rows, cols, ld, values, valueType, order);
        if (s != ACL_SPARSE_STATUS_SUCCESS) throw std::runtime_error("aclsparseCreateDnMat failed");
        return m;
    }

    static DnMatManager createConst(int64_t rows, int64_t cols, int64_t ld,
                                     const void* values, aclDataType valueType, aclsparseOrder_t order) {
        DnMatManager m;
        aclsparseConstDnMatDescr_t constDescr = nullptr;
        auto s = aclsparseCreateConstDnMat(&constDescr, rows, cols, ld, values, valueType, order);
        if (s != ACL_SPARSE_STATUS_SUCCESS) throw std::runtime_error("aclsparseCreateConstDnMat failed");
        m.descr_ = const_cast<aclsparseDnMatDescr_t>(constDescr);
        return m;
    }

    aclsparseDnMatDescr_t get() { return descr_; }
    aclsparseConstDnMatDescr_t cget() const { return descr_; }

    ~DnMatManager() {
        if (descr_) aclsparseDestroyDnMat(descr_);
    }

    DnMatManager(const DnMatManager&) = delete;
    DnMatManager& operator=(const DnMatManager&) = delete;
    DnMatManager(DnMatManager&& other) noexcept : descr_(other.descr_) { other.descr_ = nullptr; }
    DnMatManager& operator=(DnMatManager&& other) noexcept {
        if (this != &other) {
            if (descr_) aclsparseDestroyDnMat(descr_);
            descr_ = other.descr_;
            other.descr_ = nullptr;
        }
        return *this;
    }

private:
    aclsparseDnMatDescr_t descr_ = nullptr;
};

class HandleManager {
public:
    HandleManager() {
        auto s = aclsparseCreate(&handle_);
        if (s != ACL_SPARSE_STATUS_SUCCESS) throw std::runtime_error("aclsparseCreate failed");
    }

    void setStream(aclrtStream stream) {
        auto s = aclsparseSetStream(handle_, stream);
        if (s != ACL_SPARSE_STATUS_SUCCESS) throw std::runtime_error("aclsparseSetStream failed");
    }

    aclsparseHandle_t get() { return handle_; }

    ~HandleManager() {
        if (handle_) aclsparseDestroy(handle_);
    }

    HandleManager(const HandleManager&) = delete;
    HandleManager& operator=(const HandleManager&) = delete;

private:
    aclsparseHandle_t handle_ = nullptr;
};

class DeviceBuffer {
public:
    DeviceBuffer() = default;

    static DeviceBuffer alloc(size_t size) {
        DeviceBuffer b;
        auto ret = aclrtMalloc(reinterpret_cast<void**>(&b.ptr_), size, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != ACL_SUCCESS) throw std::runtime_error("aclrtMalloc failed");
        b.size_ = size;
        return b;
    }

    static DeviceBuffer copyFrom(const void* hostPtr, size_t size) {
        auto b = alloc(size);
        auto ret = aclrtMemcpy(b.ptr_, size, hostPtr, size, ACL_MEMCPY_HOST_TO_DEVICE);
        if (ret != ACL_SUCCESS) throw std::runtime_error("aclrtMemcpy H2D failed");
        return b;
    }

    void copyToHost(void* hostPtr, size_t size) const {
        auto ret = aclrtMemcpy(hostPtr, size, ptr_, size, ACL_MEMCPY_DEVICE_TO_HOST);
        if (ret != ACL_SUCCESS) throw std::runtime_error("aclrtMemcpy D2H failed");
    }

    void* get() { return ptr_; }
    void* raw() const { return ptr_; }
    size_t size() const { return size_; }

    ~DeviceBuffer() {
        if (ptr_) aclrtFree(ptr_);
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    DeviceBuffer(DeviceBuffer&& other) noexcept : ptr_(other.ptr_), size_(other.size_) {
        other.ptr_ = nullptr;
        other.size_ = 0;
    }
    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
        if (this != &other) {
            if (ptr_) aclrtFree(ptr_);
            ptr_ = other.ptr_;
            size_ = other.size_;
            other.ptr_ = nullptr;
            other.size_ = 0;
        }
        return *this;
    }

private:
    void* ptr_ = nullptr;
    size_t size_ = 0;
};

}

#endif
