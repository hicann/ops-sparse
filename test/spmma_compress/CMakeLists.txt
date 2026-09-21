#
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software under CANN Open Software License Agreement Version 2.0.
# See LICENSE in the root of the software repository for the full text.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND.


if(NOT ENABLE_SPARSELT)
    message(STATUS "[test/spmma_compress] sparseLt unavailable for the selected CANN/SOC, skipping tests")
    file(APPEND ${CMAKE_BINARY_DIR}/test/skipped_tests.list
        "spmma_compress|sparseLt unavailable for the selected CANN/SOC (ENABLE_SPARSELT=FALSE)\n")
    return()
endif()
ops_sparse_add_gtest_tests(spmma_compress ${OPS_SPARSELT} NO_SRC_DIR
    EXTRA_INCLUDES
        ${CMAKE_SOURCE_DIR}/sparseLt/matmul_descriptor
        ${CMAKE_SOURCE_DIR}/sparseLt/matmul_plan
        ${CMAKE_SOURCE_DIR}/sparseLt/common
        ${CMAKE_SOURCE_DIR}/sparseLt/shared)

# 性能程序需显式构建和运行，不随功能测试自动执行。
if(TARGET spmma_compress_test AND "arch35" IN_LIST SOC_ARCH_DIRS)
    add_executable(spmma_compress_perf EXCLUDE_FROM_ALL arch35/spmma_compress_perf.cpp)
    target_compile_features(spmma_compress_perf PRIVATE cxx_std_17)
    target_include_directories(spmma_compress_perf PRIVATE
        $<TARGET_PROPERTY:spmma_compress_test,INCLUDE_DIRECTORIES>)
    target_link_directories(spmma_compress_perf PRIVATE
        $<TARGET_PROPERTY:spmma_compress_test,LINK_DIRECTORIES>)
    target_link_libraries(spmma_compress_perf PRIVATE
        ${OPS_SPARSELT}
        ${_ops_sparse_ascendc_runtime_lib}
        ${_ops_sparse_ascendcl_lib}
        ${_ops_sparse_libc_sec}
        tiling_api platform unified_dlog mmpa pthread)
endif()
