# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

# =============================================================================
# GTest + CSV parameterized test for aclsparseLtSpMMAPrune (prune-only).
#
# This test exercises only the prune stage of the aclsparseLt chain:
#   aclsparseLtInit
#   -> StructuredDescriptorInit(A) / DenseDescriptorInit(B,C,D)
#   -> MatmulDescriptorInit
#   -> SpMMAPrune        (dA -> dAPruned)   [core API under test]
#
# It does NOT call AlgSelectionInit / AlgSetAttribute / PlanInit / Matmul.
# The pruned matrix dAPruned is copied back to host and compared element-wise
# against the CPU golden reference (deterministic, exact match expected).
# =============================================================================

# ---- 版本拦截：sparseLt 库依赖 kernel API（CANN 9.1.0+），低版本跳过 ----
if(NOT ENABLE_SPARSELT)
    message(STATUS "[test/prune] ENABLE_SPARSELT=FALSE (requires asc-devkit >= 9.1), skipping prune_test")
    file(APPEND ${CMAKE_BINARY_DIR}/test/skipped_tests.list
        "prune|requires asc-devkit >= 9.1 (ENABLE_SPARSELT=FALSE)\n")
    return()
endif()

set(_target prune_test)
set(_src_arch "")
set(_test_src "")

foreach(arch_dir ${SOC_ARCH_DIRS})
    set(_candidate "${CMAKE_CURRENT_SOURCE_DIR}/${arch_dir}/prune_test.cpp")
    if(EXISTS ${_candidate})
        set(_test_src ${_candidate})
        set(_src_arch ${arch_dir})
        break()
    endif()
endforeach()

if(NOT _test_src)
    message(STATUS "[test/prune] no test sources for SOC=${SOC_VERSION} (SOC_ARCH_DIRS=${SOC_ARCH_DIRS}), skipping ${_target}")
    file(APPEND ${CMAKE_BINARY_DIR}/test/skipped_tests.list
        "prune|no test sources for SOC=${SOC_VERSION} (arch=${SOC_ARCH_DIRS})\n")
    return()
endif()

# ---- Locate GTest (main() is provided by test/frame/test_main.cpp) ----
find_package(GTest QUIET)
if(NOT GTest_FOUND)
    find_package(PkgConfig QUIET)
    if(PkgConfig_FOUND)
        pkg_check_modules(GTEST QUIET gtest)
    endif()
    if(NOT GTEST_FOUND)
        message(WARNING "[test/prune] GTest not found, skipping ${_target}")
        file(APPEND ${CMAKE_BINARY_DIR}/test/skipped_tests.list
            "prune|GTest not found\n")
        return()
    endif()
else()
    set(GTEST_LIBRARIES GTest::gtest)
    set(GTEST_INCLUDE_DIRS "")
endif()

add_executable(${_target} ${_test_src} ${CMAKE_SOURCE_DIR}/test/frame/test_main.cpp)
target_compile_features(${_target} PRIVATE cxx_std_17)

# Pin binary output to build/test/prune/ (build.sh expectation).
set_target_properties(${_target} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/test/prune)

target_include_directories(${_target} PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/sparseLt/prune/${_src_arch}
    ${CMAKE_SOURCE_DIR}/sparseLt/shared
    ${CMAKE_SOURCE_DIR}/sparseLt/common
    ${CMAKE_SOURCE_DIR}/test/frame
    ${CMAKE_SOURCE_DIR}/test/sparseLt/matmulDescriptorInit
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/${_src_arch}
    $ENV{LINUX_INCLUDE_PATH}
    ${GTEST_INCLUDE_DIRS}
)

target_link_libraries(${_target} PRIVATE
    ${OPS_SPARSELT}
    ${_ops_sparse_ascendcl_lib}
    ${GTEST_LIBRARIES}
    pthread
)

# ---- Copy CSV next to the binary so the test loads it by relative path ----
set(_csv_file "${CMAKE_CURRENT_SOURCE_DIR}/${_src_arch}/prune_test.csv")
if(EXISTS ${_csv_file})
    add_custom_command(TARGET ${_target} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy ${_csv_file} $<TARGET_FILE_DIR:${_target}>/prune_test.csv
        COMMENT "Copying prune_test.csv to build dir"
    )
endif()
