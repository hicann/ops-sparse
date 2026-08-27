# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

# SpGEMM V1 仅在 Ascend 950PR（SIMT 扩展 + dav-3510）上交付，与 src/spgemm/ 的 SOC 网关保持一致。
if(SOC_VERSION_LOWER MATCHES "^ascend950")
    add_executable(spgemm_test
        arch35/spgemm_test.cpp
    )
    target_include_directories(spgemm_test PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        $ENV{LINUX_INCLUDE_PATH}
    )
    target_link_libraries(spgemm_test PRIVATE
        ${OPS_SPARSE}
        $ENV{EAGER_LIBRARY_PATH}/libascendcl.so
    )
else()
    message(STATUS "[test/spgemm] SOC=${SOC_VERSION} not supported, skipping spgemm_test")
    return()
endif()
