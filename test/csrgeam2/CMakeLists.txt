# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

# csrgeam2 GTest + CSV-driven test configuration
# Directory layout: test/csrgeam2/archXX/{test,npu_wrapper,csv}
set(_op_name csrgeam2)
set(_variant csrgeam2)
set(_target ${_variant}_test)
set(_test_src "")
set(_src_arch "")

foreach(arch_dir ${SOC_ARCH_DIRS})
    set(_candidate "${CMAKE_CURRENT_SOURCE_DIR}/${arch_dir}/${_target}.cpp")
    if(EXISTS ${_candidate})
        set(_test_src ${_candidate})
        set(_src_arch ${arch_dir})
        break()
    endif()
endforeach()

if(NOT _test_src)
    message(STATUS "[test/${_op_name}/${_variant}] no test sources for SOC=${SOC_VERSION} (SOC_ARCH_DIRS=${SOC_ARCH_DIRS}), skipping ${_target}")
    return()
endif()

if(NOT IS_DIRECTORY "${CMAKE_SOURCE_DIR}/src/${_op_name}/${_src_arch}")
    message(WARNING "[test/${_op_name}/${_variant}] test arch '${_src_arch}' has no matching src at src/${_op_name}/${_src_arch} — csrgeam2 operator development in progress")
endif()

add_executable(${_target} ${_test_src})

target_include_directories(${_target} PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/src/${_op_name}/${_src_arch}
    ${CMAKE_SOURCE_DIR}/src/common
    ${CMAKE_SOURCE_DIR}/test/frame
    $ENV{LINUX_INCLUDE_PATH}
)

target_link_libraries(${_target} PRIVATE
    ${OPS_SPARSE}
    $ENV{EAGER_LIBRARY_PATH}/libascendcl.so
)

# GTest + CSV-driven test configuration
# Note: test file defines its own main() (registers AclTestEnvironment),
# so GTest::gtest_main is not linked — it would be redundant.
find_package(GTest REQUIRED)
target_link_libraries(${_target} PRIVATE GTest::gtest)

# CSV path definition for runtime CSV file location
target_compile_definitions(${_target} PRIVATE
    CSRGEAM2_TEST_CSV_PATH="${CMAKE_CURRENT_SOURCE_DIR}/${_src_arch}/${_variant}_test.csv"
)

# Eigen for golden reference (optional; golden.h currently uses manual FP64 arithmetic)
if(TEST_USE_EIGEN)
    find_package(Eigen3 3.3 QUIET)
    if(Eigen3_FOUND)
        target_link_libraries(${_target} PRIVATE Eigen3::Eigen)
        target_compile_definitions(${_target} PRIVATE SPARSE_TEST_USE_EIGEN)
    endif()
endif()

# ============================================================================
# Performance test target (standalone, no GTest)
# ============================================================================
set(_perf_target ${_variant}_perf)
set(_perf_src "${CMAKE_CURRENT_SOURCE_DIR}/${_src_arch}/${_variant}_perf.cpp")

if(EXISTS ${_perf_src})
    add_executable(${_perf_target} ${_perf_src})

    target_include_directories(${_perf_target} PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}/src/${_op_name}/${_src_arch}
        ${CMAKE_SOURCE_DIR}/src/common
        ${CMAKE_SOURCE_DIR}/test/frame
        $ENV{LINUX_INCLUDE_PATH}
    )

    target_link_libraries(${_perf_target} PRIVATE
        ${OPS_SPARSE}
        $ENV{EAGER_LIBRARY_PATH}/libascendcl.so
    )
endif()
