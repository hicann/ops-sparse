# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.16.0)

project(ops_sparse VERSION 1.0.0)

# ========= 基本配置 =========
if(NOT DEFINED SOC_VERSION OR SOC_VERSION STREQUAL "")
  set(SOC_VERSION ascend910b3)
endif()
string(TOLOWER "${SOC_VERSION}" SOC_VERSION_LOWER)
# SOC_VERSION -> NPU_ARCH 映射（ascend910b->dav-2201, ascend950->dav-3101）
if(SOC_VERSION_LOWER MATCHES "^ascend910b")
  set(NPU_ARCH "dav-2201")
elseif(SOC_VERSION_LOWER MATCHES "^ascend910_93")
  set(NPU_ARCH "dav-2201")
elseif(SOC_VERSION_LOWER MATCHES "^ascend950")
  set(NPU_ARCH "dav-3510")
elseif(SOC_VERSION_LOWER MATCHES "^ascend310p")
  set(NPU_ARCH "dav-2002")
else()
  message(FATAL_ERROR "Unsupported SOC_VERSION: ${SOC_VERSION}. Supported: ascend910b*, ascend910_93*, ascend950*, ascend310p*")
endif()
message(STATUS "SOC_VERSION=${SOC_VERSION}, NPU_ARCH=${NPU_ARCH}")

# ========= SOC 架构目录配置 =========
# 根据 SOC 版本返回需要编译的架构目录列表
#
# 支持的映射关系：
#   - ascend950*     -> arch35 (DAV-3510 架构)
#   - ascend310p*    -> arch20 (DAV-2002 架构)
#   - ascend910b*    -> 使用通用实现 (DAV-2201 架构)
#   - ascend910_93*  -> 使用通用实现 (DAV-2201 架构)
#
# 函数参数：
#   soc_version: 输入的 SOC 版本字符串（如 ascend910b3, ascend950）
#   arch_dirs:   输出变量名，返回架构目录列表
# ============================================================================
function(get_soc_arch_dirs soc_version arch_dirs)
  string(TOLOWER "${soc_version}" soc_lower)
  set(dirs "")

  # ascend950 系列使用 arch35 优化
  if(soc_lower MATCHES "^ascend950")
    list(APPEND dirs "arch35")
    message(STATUS "  -> SOC [${soc_version}] matched arch35 (DAV-3510)")
  # ascend310p 系列使用 arch20 优化
  elseif(soc_lower MATCHES "^ascend310p")
    list(APPEND dirs "arch20")
    message(STATUS "  -> SOC [${soc_version}] matched arch20 (DAV-2002)")
  endif()
  # ascend910b/910_93 系列使用通用实现
  # 其他 SOC 可以根据需要添加

  set(${arch_dirs} ${dirs} PARENT_SCOPE)
endfunction()

# 获取当前 SOC 对应的架构目录
get_soc_arch_dirs(${SOC_VERSION} SOC_ARCH_DIRS)
message(STATUS "SOC_ARCH_DIRS=${SOC_ARCH_DIRS}")

# 架构特定目录列表（全局定义）
# 这些目录包含架构特定的优化实现，会从通用源文件中排除，再根据 SOC 选择性添加
set(ARCH_SPECIFIC_DIRS arch35 arch20 CACHE INTERNAL "Architecture specific directories")

set(ASCEND_CANN_PACKAGE_PATH $ENV{ASCEND_HOME_PATH})
set(RUN_MODE "npu" CACHE STRING "run mode: npu")
set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "Build type Release/Debug" FORCE)
set(CMAKE_INSTALL_PREFIX "${CMAKE_CURRENT_LIST_DIR}/build_out" CACHE STRING "install path" FORCE)

# 外部传参
option(ENABLE_PACKAGE "Enable build package" OFF)

# ========= 构建动态库 =========
find_package(ASC REQUIRED)

# sparse operators
set(ALL_SPARSE_SRC_FILES "")
set(OPS_SPARSE ops_sparse)

add_subdirectory(src)

project(${OPS_SPARSE} LANGUAGES ASC CXX)

add_library(${OPS_SPARSE} SHARED ${ALL_SPARSE_SRC_FILES})

set_source_files_properties(
    ${ALL_SPARSE_SRC_FILES}
    PROPERTIES LANGUAGE ASC
)

target_include_directories(${OPS_SPARSE} PRIVATE
    ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/op_common/
    ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/base/
    ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/
    ${ASCEND_CANN_PACKAGE_PATH}/include/op_common
    ${ASCEND_CANN_PACKAGE_PATH}/aarch64-linux/asc/include/utils
    ${ASCEND_CANN_PACKAGE_PATH}/aarch64-linux/asc/include
    ./include
    ./src/common
)

target_compile_options(${OPS_SPARSE} PRIVATE
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=${NPU_ARCH}>
)

target_link_libraries(${OPS_SPARSE} PRIVATE
    tiling_api
    platform
)

# ========= 安装规则（CPack 打包时放入 lib64 和 include） =========
install(TARGETS ${OPS_SPARSE}
    LIBRARY DESTINATION lib64
)
install(FILES
    include/cann_ops_sparse.h
    DESTINATION include
)

# ========= 构建测试程序 =========
option(BUILD_TEST "Build test programs" OFF)
if(BUILD_TEST)
    add_subdirectory(test)
endif()

if(ENABLE_PACKAGE)
    include(cmake/package.cmake)
    add_dependencies(${OPS_SPARSE} gen_ops_sparse_version_info)
    pack(${SOC_VERSION})
endif()
