# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""SpGEMM（稀疏×稀疏）torch_npu 后端的 ATen 注册。

稀疏-稀疏矩阵乘法涉及两个已有 ATen 算子：

- ``aten::_sparse_addmm`` 为 CSR 操作数提供 ``torch.sparse.mm``/``torch.sparse.addmm``。
  ``torch.sparse.mm(csr, csr)`` 以 ``beta=0, alpha=1`` 进入。
- ``aten::_sparse_sparse_matmul`` 为 COO 操作数提供 ``torch.sparse.mm``，
  同时也为 CSR 注册，因为两种布局都可能路由到此处。

此处不定义 schema，仅注册 NPU 后端实现。
``aten::_sparse_addmm`` 已有 CompositeExplicitAutograd 内核，无需添加 Meta 实现。
"""

__all__ = []

from typing import List, Union

import torch
from torch.library import impl

from ....op_builder.builder import OpBuilder

# ATen schema 中 beta/alpha 类型为 Scalar，此处接受 complex
# 是因为 complex64 是支持的 value dtype 之一。
Scalar = Union[bool, int, float, complex]


class SpGemmOpBuilder(OpBuilder):
    """SpGEMM C++ 封装的 JIT 构建器。"""

    def __init__(self) -> None:
        super().__init__("spgemm", category="sparse")

    def sources(self) -> List[str]:
        return ["csrc/sparse/spgemm/spgemm.cpp"]


_spgemm_op_builder = SpGemmOpBuilder()


# ``load()`` 在每次分发时调用，而非导入时调用，
# 这样导入包既不需要编译器也不会产生构建产物。
# 以下每个签名严格对应其 ATen schema，包括 ``_sparse_addmm`` 的仅关键字参数 beta/alpha。
@impl("aten::_sparse_addmm", "SparseCsrPrivateUse1")
def _sparse_addmm_sparse_csr_privateuse1(self: torch.Tensor, mat1: torch.Tensor, mat2: torch.Tensor,
                                         *, beta: Scalar = 1, alpha: Scalar = 1) -> torch.Tensor:
    return _spgemm_op_builder.load().sparse_addmm(self, mat1, mat2, beta, alpha)


@impl("aten::_sparse_sparse_matmul", "SparseCsrPrivateUse1")
def _sparse_sparse_matmul_sparse_csr_privateuse1(mat1: torch.Tensor, mat2: torch.Tensor) -> torch.Tensor:
    return _spgemm_op_builder.load().sparse_sparse_matmul_csr(mat1, mat2)


@impl("aten::_sparse_sparse_matmul", "SparsePrivateUse1")
def _sparse_sparse_matmul_sparse_privateuse1(mat1: torch.Tensor, mat2: torch.Tensor) -> torch.Tensor:
    return _spgemm_op_builder.load().sparse_sparse_matmul_coo(mat1, mat2)


# façade 重新导出这些注册所服务的 PyTorch 公共 API，
# 使 ``torch_npu.sparse.mm`` 仍经由 ATen dispatcher 路由。
TORCH_NPU_SPARSE_FACADE_APIS = {
    "mm": torch.sparse.mm,
    "addmm": torch.sparse.addmm,
}
