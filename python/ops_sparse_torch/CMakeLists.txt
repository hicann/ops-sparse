# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import torch; print(torch.utils.cmake_prefix_path)"
    OUTPUT_STRIP_TRAILING_WHITESPACE
    OUTPUT_VARIABLE TORCH_CMAKE_PREFIX_PATH
    COMMAND_ERROR_IS_FATAL ANY
)
find_package(Torch REQUIRED HINTS ${TORCH_CMAKE_PREFIX_PATH})

execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import os, torch_npu; print(os.path.dirname(torch_npu.__file__))"
    OUTPUT_STRIP_TRAILING_WHITESPACE
    OUTPUT_VARIABLE TORCH_NPU_PATH
    COMMAND_ERROR_IS_FATAL ANY
)

add_library(ops_sparse_torch SHARED csrc/spgemm_torch.cpp)
set_target_properties(ops_sparse_torch PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    BUILD_RPATH "$<TARGET_FILE_DIR:${OPS_SPARSE}>;${TORCH_NPU_PATH}/lib"
    INSTALL_RPATH "$ORIGIN/../../lib64"
)

target_include_directories(ops_sparse_torch PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${TORCH_INCLUDE_DIRS}
    ${TORCH_NPU_PATH}/include
    ${ASCEND_CANN_PACKAGE_PATH}/include
    ${ASCEND_CANN_PACKAGE_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/include
)

target_link_directories(ops_sparse_torch PRIVATE
    ${TORCH_NPU_PATH}/lib
    ${ASCEND_CANN_PACKAGE_PATH}/lib64
    ${ASCEND_CANN_PACKAGE_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/lib64
)

target_link_libraries(ops_sparse_torch PRIVATE
    ${OPS_SPARSE}
    ${TORCH_LIBRARIES}
    torch_npu
    ascendcl
    ${_ops_sparse_libc_sec}
)

separate_arguments(TORCH_CXX_FLAGS_LIST NATIVE_COMMAND "${TORCH_CXX_FLAGS}")
target_compile_options(ops_sparse_torch PRIVATE ${TORCH_CXX_FLAGS_LIST})

install(TARGETS ops_sparse_torch
    LIBRARY DESTINATION python/ops_sparse_torch
)
install(FILES __init__.py
    DESTINATION python/ops_sparse_torch
)

if(BUILD_TEST)
    add_custom_target(spgemm_torch_test
        COMMAND ${CMAKE_COMMAND} -E env
            "PYTHONPATH=${PROJECT_SOURCE_DIR}/python"
            ${Python3_EXECUTABLE}
            ${PROJECT_SOURCE_DIR}/test/spgemm/python/test_spgemm_torch.py
            --library $<TARGET_FILE:ops_sparse_torch>
        DEPENDS ops_sparse_torch
        USES_TERMINAL
    )
endif()
