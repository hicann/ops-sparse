# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

if(NOT SOC_VERSION_LOWER MATCHES "^ascend910b")
    message(STATUS "[test/cube_spmm] SOC=${SOC_VERSION} not ascend910b*, skipping cube_spmm_test")
    return()
endif()

add_executable(cube_spmm_test
    cube_spmm_test.cpp
)

target_include_directories(cube_spmm_test PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/sparse/common
    ${CMAKE_SOURCE_DIR}/sparse/cube_spmm/arch22
    $ENV{LINUX_INCLUDE_PATH}
)

target_link_libraries(cube_spmm_test PRIVATE
    ${OPS_SPARSE}
    $ENV{EAGER_LIBRARY_PATH}/libascendcl.so
)
