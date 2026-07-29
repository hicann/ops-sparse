# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

# aclsparseScatter GTest tests (SpVec-DnVec scatter, Generic API)
# Uses WARN_ON_MISSING_SRC because the operator source (sparse/scatter/arch35/)
# may not exist yet when tests are developed in parallel with the operator.
ops_sparse_add_gtest_tests(scatter ${OPS_SPARSE} WARN_ON_MISSING_SRC)
