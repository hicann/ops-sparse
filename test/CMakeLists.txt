# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

if(BUILD_TEST)
    # 初始化 skip 清单（CMake 配置阶段由 test.cmake 中的函数写入被跳过的算子）
    file(WRITE ${CMAKE_BINARY_DIR}/test/skipped_tests.list "")

    # 支持多算子列表（OP_LIST 是分号分隔的 CMake 列表）
    foreach(op ${OP_LIST})
        set(_found FALSE)
        # 直接查找：test/<op>/
        if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${op}/CMakeLists.txt)
            add_subdirectory(${op})
            set(_found TRUE)
        endif()
        # 嵌套查找：test/<group>/<op>/（如 test/sparseLt/matmulDescriptorInit/）
        if(NOT _found)
            file(GLOB _subdirs RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} */)
            foreach(_subdir ${_subdirs})
                if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${_subdir}/${op}/CMakeLists.txt)
                    add_subdirectory(${_subdir}/${op})
                    set(_found TRUE)
                    break()
                endif()
            endforeach()
        endif()
        if(NOT _found)
            message(FATAL_ERROR "TEST directory not found: ${op}")
        endif()
    endforeach()
endif()
