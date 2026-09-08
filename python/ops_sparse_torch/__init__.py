# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""加载ops-sparse ATen注册库。"""

from __future__ import annotations

import os
from pathlib import Path

import torch


def _library_path() -> Path:
    configured = os.environ.get("OPS_SPARSE_TORCH_LIBRARY")
    if configured:
        return Path(configured).expanduser().resolve()
    candidate = Path(__file__).with_name("libops_sparse_torch.so")
    if candidate.is_file():
        return candidate
    raise ImportError(
        "libops_sparse_torch.so was not found; set OPS_SPARSE_TORCH_LIBRARY "
        "to the CMake build output"
    )


torch.ops.load_library(str(_library_path()))
