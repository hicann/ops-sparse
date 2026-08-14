# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

# aclsparseLtMatmulDescriptorInit GTest tests (Host-side descriptor init, no kernel)
#
# 适配说明：
# - 本接口实现位于 sparseLt/common/（Host 侧 C++），不存在 sparse/<op>/<arch>/ 目录。
# - NO_SRC_DIR：跳过 sparse/${operator}/${arch} 源目录存在性检查；测试文件直接位于
#   test/<op>/ 上层（本接口为 Host 侧，与架构无关，无需 arch 子目录），NO_SRC_DIR 直接
#   在上层查找测试源。
# - EXTRA_INCLUDES sparseLt/common：使测试可 include aclsparselt_*_internal.h（白盒字段验证）。
# - 链接 ${OPS_SPARSELT}（libops_sparseLt.so，接口实现在该库内）。
# - operator 名取 matmulDescriptorInit（与 test.cpp/test.csv 文件名前缀一致）。
ops_sparse_add_gtest_tests(matmulDescriptorInit ${OPS_SPARSELT}
    NO_SRC_DIR
    EXTRA_INCLUDES ${CMAKE_SOURCE_DIR}/sparseLt/common)
