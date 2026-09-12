# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""注册 SpGEMM 对应的标准 ATen 稀疏算子 NPU 实现。"""

import torch
from torch.library import impl

from cann_ops_sparse.op_builder import OpBuilder


class SpGemmOpBuilder(OpBuilder):
    """按需编译并加载 SpGEMM 的 C++ wrapper。"""

    def __init__(self):
        super().__init__("spgemm", category="sparse")

    def sources(self):
        return ["csrc/sparse/spgemm/spgemm.cpp"]


_SPGEMM_OP_BUILDER = SpGemmOpBuilder()

# ``cann_ops_sparse.ops`` 收集该映射，并在所有算子包导入后创建
# ``torch_npu.sparse`` 中的对应 façade。
TORCH_NPU_SPARSE_FACADE_APIS = {"mm": torch.sparse.mm}


@impl("aten::_sparse_sparse_matmul", "SparseCsrPrivateUse1")
def _sparse_sparse_matmul_csr(mat1, mat2):
    """执行两个 NPU CSR 稀疏矩阵的矩阵乘。"""
    return _SPGEMM_OP_BUILDER.load().sparse_sparse_matmul_csr(mat1, mat2)


@impl("aten::_sparse_sparse_matmul", "SparsePrivateUse1")
def _sparse_sparse_matmul_coo(mat1, mat2):
    """执行两个 NPU COO 稀疏矩阵的矩阵乘。"""
    return _SPGEMM_OP_BUILDER.load().sparse_sparse_matmul_coo(mat1, mat2)


@impl("aten::_sparse_addmm", "SparseCsrPrivateUse1")
def _sparse_addmm_csr(self, mat1, mat2, beta=1, alpha=1):
    """执行 CSR ``addmm`` 的受限 SpGEMM 路径。"""
    return _SPGEMM_OP_BUILDER.load().sparse_addmm_csr(
        self, mat1, mat2, beta, alpha
    )
