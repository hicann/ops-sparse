/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TEST_FRAME_SPARSE_TEST_H_
#define TEST_FRAME_SPARSE_TEST_H_

#include <cstdint>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "acl/acl.h"
#include "cann_ops_sparse.h"
#include "types.h"

namespace sparse_test {

#ifndef TEST_DEVICE_ID
#define TEST_DEVICE_ID 0
#endif

class AclEnvScope {
public:
    explicit AclEnvScope(int32_t deviceId = TEST_DEVICE_ID) : deviceId_(deviceId) {
        if (aclInit(nullptr) != ACL_SUCCESS) {
            std::cerr << "aclInit failed" << std::endl;
            throw std::runtime_error("aclInit failed");
        }
        if (aclrtSetDevice(deviceId_) != ACL_SUCCESS) {
            std::cerr << "aclrtSetDevice failed" << std::endl;
            throw std::runtime_error("aclrtSetDevice failed");
        }
        if (aclrtCreateStream(&stream_) != ACL_SUCCESS) {
            std::cerr << "aclrtCreateStream failed" << std::endl;
            throw std::runtime_error("aclrtCreateStream failed");
        }
        initialized_ = true;
    }

    ~AclEnvScope() {
        if (!initialized_) return;
        if (stream_) aclrtDestroyStream(stream_);
        aclrtResetDevice(deviceId_);
        aclFinalize();
    }

    AclEnvScope(const AclEnvScope&) = delete;
    AclEnvScope& operator=(const AclEnvScope&) = delete;

    aclrtStream stream() const { return stream_; }

private:
    int32_t deviceId_;
    aclrtStream stream_ = nullptr;
    bool initialized_ = false;
};

class TestRegistry {
public:
    void record(const std::string& name, bool pass) {
        total_++;
        if (pass) {
            pass_++;
        } else {
            fail_++;
            failedNames_.push_back(name);
        }
    }

    void printSummary() const {
        std::cout << "\n============================================================\n";
        std::cout << "Summary: passed=" << pass_ << " failed=" << fail_ << " total=" << total_ << "\n";
        if (!failedNames_.empty()) {
            std::cout << "Failed cases:\n";
            for (const auto& n : failedNames_) std::cout << "  " << n << "\n";
        }
        std::cout << "============================================================\n";
    }

    int failCount() const { return fail_; }
    int totalCount() const { return total_; }

private:
    int total_ = 0;
    int pass_ = 0;
    int fail_ = 0;
    std::vector<std::string> failedNames_;
};

template <typename T>
aclDataType aclDataTypeOf() {
    return ACL_FLOAT;
}

template <>
inline aclDataType aclDataTypeOf<float>() { return ACL_FLOAT; }
template <>
inline aclDataType aclDataTypeOf<int32_t>() { return ACL_INT32; }

#define SPARSE_CHECK_RET(cond, action) \
    do { if (!(cond)) { action; } } while (0)

#define SPARSE_LOG(msg, ...) \
    do { std::printf(msg, ##__VA_ARGS__); } while (0)

}

#endif
