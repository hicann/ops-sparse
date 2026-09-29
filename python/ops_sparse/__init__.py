# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.

"""
ops_sparse - Python / torch adapter for the CANN ``aclsparseGather`` operator
(Atlas A2/A3, arch22).

Quick start::

    import torch
    import torch_npu                 # registers NPU backend
    import ops_sparse                # registers torch.ops.ops_sparse.gather

    y = torch.randn(1000, dtype=torch.float32, device="npu")
    idx = torch.randint(0, 1000, (256,), dtype=torch.int32, device="npu")

    out = torch.ops.ops_sparse.gather(y, idx, base=0)   # == y[idx]
    out2 = ops_sparse.index_select_npu(y, 0, idx)        # == torch.index_select(y, 0, idx)
"""

__all__ = [
    "gather",
    "index_select_npu",
    "LibOpsSparseError",
    "AclDevice",
    "load_library",
]

from ._capi import (  # noqa: E402, F401
    LibOpsSparseError,
    AclDevice,
    load_library,
)
from .gather import (  # noqa: E402, F401
    gather,
    index_select_npu,
)
