# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

# aclsparseLtMatmulPlanInit / aclsparseLtMatmulAlgSelectionInit GTest tests (Host-side, no kernel)
#
# 隔离设计：测试目录独立，不依赖 matmulDescriptorInit 的测试头文件。
# EXTRA_INCLUDES sparseLt/common：使测试可 include aclsparselt_*_internal.h（白盒字段验证）。
ops_sparse_add_gtest_tests(matmulPlanInit ${OPS_SPARSELT}
    NO_SRC_DIR
    EXTRA_INCLUDES ${CMAKE_SOURCE_DIR}/sparseLt/common)
