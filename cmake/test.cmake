# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

if(DEFINED ENV{EAGER_LIBRARY_PATH} AND NOT "$ENV{EAGER_LIBRARY_PATH}" STREQUAL "")
    set(_ops_sparse_ascendcl_lib "$ENV{EAGER_LIBRARY_PATH}/libascendcl.so")
else()
    set(ASCENDCL_PATH "${ASCEND_CANN_PACKAGE_PATH}/lib64" CACHE PATH "Directory containing libascendcl.so")
    set(_ops_sparse_ascendcl_lib "${ASCENDCL_PATH}/libascendcl.so")
endif()

# Register operator test from test/${operator}/${arch_dir}/${operator}_test.cpp.
# Picks the first matching arch dir in SOC_ARCH_DIRS; skips if none found.
function(ops_sparse_add_test operator link_lib)
    set(target "${operator}_test")
    set(_test_src "")
    set(_src_arch "")

    foreach(arch_dir ${SOC_ARCH_DIRS})
        set(_candidate "${CMAKE_CURRENT_SOURCE_DIR}/${arch_dir}/${target}.cpp")
        if(EXISTS ${_candidate})
            set(_test_src ${_candidate})
            set(_src_arch ${arch_dir})
            break()
        endif()
    endforeach()

    if(NOT _test_src)
        message(STATUS "[test/${operator}] no test sources for SOC=${SOC_VERSION} (SOC_ARCH_DIRS=${SOC_ARCH_DIRS}), skipping ${target}")
        return()
    endif()

    if(NOT IS_DIRECTORY "${CMAKE_SOURCE_DIR}/src/${operator}/${_src_arch}")
        message(FATAL_ERROR "[test/${operator}] test arch '${_src_arch}' has no matching src at src/${operator}/${_src_arch}")
    endif()

    add_executable(${target} ${_test_src})
    target_compile_features(${target} PRIVATE cxx_std_17)

    target_include_directories(${target} PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/src/${operator}/${_src_arch}
        ${CMAKE_SOURCE_DIR}/src/common
        ${CMAKE_SOURCE_DIR}/test/frame
        $ENV{LINUX_INCLUDE_PATH}
    )

    target_link_libraries(${target} PRIVATE
        ${link_lib}
        ${_ops_sparse_ascendcl_lib}
    )

    if(TEST_USE_EIGEN)
        find_package(Eigen3 3.3 QUIET)
        if(Eigen3_FOUND)
            target_link_libraries(${target} PRIVATE Eigen3::Eigen)
            target_compile_definitions(${target} PRIVATE SPARSE_TEST_USE_EIGEN)
        endif()
    endif()
endfunction()
