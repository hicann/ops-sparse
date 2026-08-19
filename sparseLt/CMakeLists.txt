# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

# sparseLt 源文件收集：host (CXX) 与 kernel (ASC) 分离
# host: *_host.cpp + common/*.cpp；kernel: *_kernel.cpp
file(GLOB_RECURSE SPARSELT_HOST_SRCS CONFIGURE_DEPENDS
    ${CMAKE_CURRENT_SOURCE_DIR}/*_host.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/common/*.cpp)
file(GLOB_RECURSE SPARSELT_KERNEL_SRCS_ALL CONFIGURE_DEPENDS
    ${CMAKE_CURRENT_SOURCE_DIR}/*_kernel.cpp)
set(SPARSELT_KERNEL_SRCS "")
foreach(arch_dir ${SOC_ARCH_DIRS})
    foreach(src ${SPARSELT_KERNEL_SRCS_ALL})
        if(src MATCHES "/${arch_dir}/")
            list(APPEND SPARSELT_KERNEL_SRCS ${src})
        endif()
    endforeach()
endforeach()

# ========= asc-devkit 版本检测（自包含，不依赖 cmake/ 公共文件）=========
# 读取 asc_devkit_version.h，解析 ASC_DEVKIT_MAJOR / ASC_DEVKIT_MINOR。
# sparseLt 的 prune kernel 依赖 kernel_operator.h（CANN 9.1.0+），低版本时
# 清空源文件列表使 sparseLt 库不编译，并通过 ENABLE_SPARSELT 通知 test 子目录跳过。
set(ASC_DEVKIT_MAJOR 0)
set(ASC_DEVKIT_MINOR 0)
set(ENABLE_SPARSELT TRUE)
set(_sparselt_version_header
    "${ASCEND_CANN_PACKAGE_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/include/version/asc_devkit_version.h")

if(EXISTS "${_sparselt_version_header}")
    file(READ "${_sparselt_version_header}" _sparselt_version_content)
    if(_sparselt_version_content MATCHES "#define ASC_DEVKIT_MAJOR ([0-9]+)")
        set(ASC_DEVKIT_MAJOR "${CMAKE_MATCH_1}")
    endif()
    if(_sparselt_version_content MATCHES "#define ASC_DEVKIT_MINOR ([0-9]+)")
        set(ASC_DEVKIT_MINOR "${CMAKE_MATCH_1}")
    endif()
    # arch35 的 kernel 依赖 kernel_operator.h，需 asc-devkit >= 9.1
    if("arch35" IN_LIST SOC_ARCH_DIRS AND NOT (ASC_DEVKIT_MAJOR GREATER_EQUAL 9 AND ASC_DEVKIT_MINOR GREATER_EQUAL 1))
        set(ENABLE_SPARSELT FALSE)
        message(STATUS "[sparseLt] asc-devkit ${ASC_DEVKIT_MAJOR}.${ASC_DEVKIT_MINOR} < 9.1, "
                       "kernel API unavailable, skipping sparseLt library (prune)")
    endif()
else()
    set(ENABLE_SPARSELT FALSE)
    message(WARNING "[sparseLt] asc_devkit_version.h not found: ${_sparselt_version_header}, "
                    "sparseLt library will be skipped")
endif()

if(NOT ENABLE_SPARSELT)
    # 清空源文件列表，使根 CMakeLists.txt 的 if(ALL_SPARSELT_SRC_FILES) 跳过库构建
    set(SPARSELT_HOST_SRCS "")
    set(SPARSELT_KERNEL_SRCS "")
endif()

message(STATUS "[sparseLt] ASC_DEVKIT_MAJOR=${ASC_DEVKIT_MAJOR}, ASC_DEVKIT_MINOR=${ASC_DEVKIT_MINOR}, "
               "ENABLE_SPARSELT=${ENABLE_SPARSELT}")

set(ALL_SPARSELT_SRC_FILES ${SPARSELT_HOST_SRCS} ${SPARSELT_KERNEL_SRCS} PARENT_SCOPE)
set(ALL_SPARSELT_HOST_SRC_FILES ${SPARSELT_HOST_SRCS} PARENT_SCOPE)
set(ALL_SPARSELT_KERNEL_SRC_FILES ${SPARSELT_KERNEL_SRCS} PARENT_SCOPE)
set(ENABLE_SPARSELT ${ENABLE_SPARSELT} PARENT_SCOPE)
