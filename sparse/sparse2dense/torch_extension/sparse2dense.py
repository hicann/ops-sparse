# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""SparseToDense torch_npu 后端的 ATen 注册（``aten::_to_dense``）。

复用 PyTorch 已有 schema，不额外 ``define`` 自定义算子。CSR / CSC 走
``SparseCsrPrivateUse1``，COO 走 ``SparsePrivateUse1``；首次 NPU 调用时 JIT
编译 C++ wrapper 并接入 ``aclsparseSparseToDense``。
"""

__all__ = []

from typing import List, Optional

import torch
from torch.library import impl

from ....op_builder.builder import OpBuilder


class Sparse2DenseOpBuilder(OpBuilder):
    """SparseToDense C++ 封装的 JIT 构建器。"""

    def __init__(self) -> None:
        super().__init__("sparse2dense", category="sparse")

    def sources(self) -> List[str]:
        return ["csrc/sparse/sparse2dense/sparse2dense.cpp"]


_sparse2dense_op_builder = Sparse2DenseOpBuilder()


def _apply_dtype(out: torch.Tensor, dtype: Optional[torch.dtype]) -> torch.Tensor:
    if dtype is None or out.dtype == dtype:
        return out
    return out.to(dtype)


def _csr_on_device_from_cpu_csr(csr: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Rebuild a CPU CSR as an NPU CSR with explicit int32 index copies."""
    return torch.sparse_csr_tensor(
        csr.crow_indices().to(dtype=torch.int32, device=device).contiguous(),
        csr.col_indices().to(dtype=torch.int32, device=device).contiguous(),
        csr.values().to(device=device).contiguous(),
        size=tuple(csr.shape),
        device=device,
    )


def _normalize_for_aclsparse(tensor: torch.Tensor) -> torch.Tensor:
    """Normalize layouts that are unreliable on some Ascend SoCs before ACLSparse.

    - CSC / uncoalesced COO: convert via CPU to CSR, then rebuild with explicit
      int32 index copies on device. Plain ``.to(device)`` sparse tensors can
      mis-bind after prior CSR launches on ascend950; NPU coalesce is unavailable
      (EZ1001 / 161001).
    """
    device = tensor.device
    if tensor.layout == torch.sparse_csc:
        return _csr_on_device_from_cpu_csr(tensor.cpu().to_sparse_csr(), device)
    if tensor.layout == torch.sparse_coo:
        coo = tensor.cpu() if tensor.is_coalesced() else tensor.cpu().coalesce()
        return _csr_on_device_from_cpu_csr(coo.to_sparse_csr(), device)
    return tensor


def _sparse_to_dense(tensor: torch.Tensor, dtype: Optional[torch.dtype]) -> torch.Tensor:
    tensor = _normalize_for_aclsparse(tensor)
    # Prefer values().numel() over Tensor._nnz() (codecheck G.CLS.11 protected-access).
    if tensor.size(0) == 0 or tensor.size(1) == 0 or int(tensor.values().numel()) == 0:
        # Keep empty-nnz on the Python side; avoids fragile empty value storage in the
        # JIT extension after prior sparse launches on ascend950.
        out = torch.zeros(tuple(tensor.shape), dtype=tensor.dtype, device=tensor.device)
        return _apply_dtype(out, dtype)
    return _apply_dtype(_sparse2dense_op_builder.load().sparse_to_dense(tensor), dtype)


@impl("aten::_to_dense", "SparseCsrPrivateUse1")
def _to_dense_sparse_csr_privateuse1(
    self: torch.Tensor, *, dtype: Optional[torch.dtype] = None
) -> torch.Tensor:
    return _sparse_to_dense(self, dtype)


@impl("aten::_to_dense", "SparsePrivateUse1")
def _to_dense_sparse_privateuse1(
    self: torch.Tensor, *, dtype: Optional[torch.dtype] = None
) -> torch.Tensor:
    return _sparse_to_dense(self, dtype)


# ``to_dense`` 是 Tensor 方法，不通过 torch_npu.sparse façade 暴露。
TORCH_NPU_SPARSE_FACADE_APIS = {}
