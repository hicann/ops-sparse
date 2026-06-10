# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

# SPMM V1 仅在 Ascend 950PR（SIMT 扩展 + dav-3510）上交付，与 src/spmm/ 的 SOC 网关保持一致。
if(NOT SOC_VERSION_LOWER MATCHES "^ascend950")
    message(STATUS "[test/spmm] SOC=${SOC_VERSION} not ascend950*, skipping spmm_test")
    return()
endif()

add_executable(spmm_test
    spmm_test.cpp
)

target_include_directories(spmm_test PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    $ENV{LINUX_INCLUDE_PATH}
)

target_link_libraries(spmm_test PRIVATE
    ${OPS_SPARSE}
    $ENV{EAGER_LIBRARY_PATH}/libascendcl.so
)
