# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Register the constrained Gather implementation on ``aten::index_select``."""

from __future__ import annotations

__all__ = []

import torch
from torch.library import impl

from ....op_builder.builder import OpBuilder


class GatherOpBuilder(OpBuilder):
    """JIT builder for the Gather ATen wrapper."""

    def __init__(self) -> None:
        super().__init__("gather", category="sparse")

    def sources(self) -> list[str]:
        return ["csrc/sparse/gather/gather.cpp"]


_gather_op_builder = GatherOpBuilder()


@impl("aten::index_select", "PrivateUse1")
def _index_select_privateuse1(self: torch.Tensor, dim: int, index: torch.Tensor) -> torch.Tensor:
    """Gather from a one-dimensional NPU tensor through ``aclsparseGather``."""
    if self.dim() != 1:
        raise RuntimeError(
            f"aclsparseGather index_select requires a 1-D input, got {self.dim()} dimensions"
        )
    if dim not in (0, -1):
        raise IndexError(f"Dimension out of range for 1-D Gather input: {dim}")
    return _gather_op_builder.load().index_select(self, index)


# Gather uses torch.index_select directly and does not add an unrelated sparse façade.
TORCH_NPU_SPARSE_FACADE_APIS = {}
