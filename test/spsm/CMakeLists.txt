# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

# aclsparseSpSM GTest + CSV tests (Generic API, three-stage: BufferSize -> Analysis -> Solve)
#
# Source layout (VARIANT spsm):
#   test/spsm/spsm/arch35/spsm_test.cpp   (GTest TEST_P CSV + TEST_F exception)
#   test/spsm/spsm/arch35/spsm_test.csv   (L0/L1/L2-special cases)
#   test/spsm/spsm/spsm_golden.h          (Eigen FP64 triangular solve)
#   test/spsm/spsm/spsm_param.h           (CSV param parser)
#
# WARN_ON_MISSING_SRC: sparse/spsm/arch35 may not exist yet (developer-code in progress);
#                      the test target is skipped until the operator source lands.
# EIGEN: golden uses Eigen SparseMatrix<double> (FP64).
ops_sparse_add_gtest_tests(spsm ${OPS_SPARSE} VARIANT spsm WARN_ON_MISSING_SRC EIGEN)
if(TARGET spsm_test AND "arch35" IN_LIST SOC_ARCH_DIRS)
    target_sources(spsm_test PRIVATE spsm/arch35/spsm_extended_test.cpp)

    add_executable(spsm_perf spsm/arch35/spsm_perf.cpp)
    target_compile_features(spsm_perf PRIVATE cxx_std_17)
    target_include_directories(spsm_perf PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/test/frame
        ${ASCEND_CANN_PACKAGE_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/include
        $ENV{LINUX_INCLUDE_PATH})
    target_link_libraries(spsm_perf PRIVATE
        ${OPS_SPARSE}
        ${_ops_sparse_ascendcl_lib}
        ${_ops_sparse_libc_sec})
endif()
