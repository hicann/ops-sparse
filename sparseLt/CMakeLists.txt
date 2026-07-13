# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

# sparseLt 源文件收集（占位）
# 当前 sparseLt 仅有框架，无源文件，导出空列表

file(GLOB_RECURSE SPARSELT_HOST_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
file(GLOB_RECURSE SPARSELT_KERNEL_SRCS "")

set(ALL_SPARSELT_SRC_FILES ${SPARSELT_HOST_SRCS} PARENT_SCOPE)
set(ALL_SPARSELT_HOST_SRC_FILES ${SPARSELT_HOST_SRCS} PARENT_SCOPE)
set(ALL_SPARSELT_KERNEL_SRC_FILES PARENT_SCOPE)
