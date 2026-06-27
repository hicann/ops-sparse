# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

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

    target_include_directories(${target} PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/src/${operator}/${_src_arch}
        $ENV{LINUX_INCLUDE_PATH}
    )

    target_link_libraries(${target} PRIVATE
        ${link_lib}
        $ENV{EAGER_LIBRARY_PATH}/libascendcl.so
    )
endfunction()
