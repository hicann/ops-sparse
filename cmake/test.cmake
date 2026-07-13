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

# Register GTest + CSV parameterized test for a new operator.
#
# Signature:
#   ops_sparse_add_gtest_tests(<operator> <link_lib>
#       [VARIANT <name>]                  # precision variant subdir (e.g. 'snnz' under 'nnz')
#                                          # - source searched at test/<op>/<variant>/<arch>/<variant>_test.cpp
#                                          # - binary target stays <operator>_test
#                                          # - CSV copied as <variant>_test.csv
#       [WARN_ON_MISSING_SRC]              # emit warning instead of FATAL_ERROR when src/<op>/<arch>/ is missing
#       [EIGEN]                            # link Eigen3 (unconditionally, regardless of TEST_USE_EIGEN cache var)
#       [EXTRA_INCLUDES <dir1> <dir2> ...] # additional include directories
#   )
#
# Always links test/frame/test_main.cpp (shared main); test cpp files MUST NOT define main().
function(ops_sparse_add_gtest_tests operator link_lib)
    cmake_parse_arguments(ARG
        "WARN_ON_MISSING_SRC;EIGEN"
        "VARIANT"
        "EXTRA_INCLUDES"
        ${ARGN})

    # Source layout:
    #   no VARIANT  -> test/<op>/<arch>/<op>_test.cpp,         CSV: <op>_test.csv
    #   VARIANT v   -> test/<op>/v/<arch>/v_test.cpp,         CSV: v_test.csv
    if(ARG_VARIANT)
        set(_subdir "${ARG_VARIANT}")
        set(_source_prefix "${ARG_VARIANT}")
    else()
        set(_subdir "")
        set(_source_prefix "${operator}")
    endif()

    set(target "${operator}_test")
    set(_test_src "")
    set(_src_arch "")

    foreach(arch_dir ${SOC_ARCH_DIRS})
        set(_candidate "${CMAKE_CURRENT_SOURCE_DIR}/${_subdir}/${arch_dir}/${_source_prefix}_test.cpp")
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
        if(ARG_WARN_ON_MISSING_SRC)
            message(WARNING "[test/${operator}] src/${operator}/${_src_arch} not found, skipping ${target}")
            return()
        else()
            message(FATAL_ERROR "[test/${operator}] test arch '${_src_arch}' has no matching src at src/${operator}/${_src_arch}")
        endif()
    endif()

    # Find GTest
    find_package(GTest QUIET)
    if(NOT GTest_FOUND)
        find_package(PkgConfig QUIET)
        if(PkgConfig_FOUND)
            # 仅使用 gtest 本体；main() 由我们自己的 test/frame/test_main.cpp 提供
            pkg_check_modules(GTEST QUIET gtest)
        endif()
        if(NOT GTEST_FOUND)
            message(WARNING "[test/${operator}] GTest not found, skipping ${target}")
            return()
        endif()
    else()
        # 仅引入 GTest::gtest；main() 由 test/frame/test_main.cpp 提供（避免与 gtest_main 的 main() 符号冲突）
        set(GTEST_LIBRARIES GTest::gtest)
        set(GTEST_INCLUDE_DIRS "")
    endif()

    add_executable(${target} ${_test_src} ${CMAKE_SOURCE_DIR}/test/frame/test_main.cpp)

    target_include_directories(${target} PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/src/${operator}/${_src_arch}
        ${CMAKE_SOURCE_DIR}/src/common
        ${CMAKE_SOURCE_DIR}/test/frame
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/${_subdir}
        ${CMAKE_CURRENT_SOURCE_DIR}/${_subdir}/${_src_arch}
        $ENV{LINUX_INCLUDE_PATH}
        ${GTEST_INCLUDE_DIRS}
        ${ARG_EXTRA_INCLUDES}
    )

    target_link_libraries(${target} PRIVATE
        ${link_lib}
        ${_ops_sparse_ascendcl_lib}
        ${GTEST_LIBRARIES}
        pthread
    )

    # Eigen3: explicit EIGEN kwarg OR legacy TEST_USE_EIGEN cache var
    if(ARG_EIGEN OR TEST_USE_EIGEN)
        find_package(Eigen3 3.3 QUIET)
        if(Eigen3_FOUND)
            target_link_libraries(${target} PRIVATE Eigen3::Eigen)
            target_compile_definitions(${target} PRIVATE SPARSE_TEST_USE_EIGEN)
        endif()
    endif()

    # Copy CSV to build dir so binary can find it via relative path.
    set(_csv_file "${CMAKE_CURRENT_SOURCE_DIR}/${_subdir}/${_src_arch}/${_source_prefix}_test.csv")
    if(EXISTS ${_csv_file})
        add_custom_command(TARGET ${target} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy ${_csv_file} $<TARGET_FILE_DIR:${target}>/${_source_prefix}_test.csv
            COMMENT "Copying ${_source_prefix}_test.csv to build dir"
        )
    endif()
endfunction()
