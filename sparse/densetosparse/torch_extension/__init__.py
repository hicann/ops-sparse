# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software: you can redistribute it and/or modify it under the terms of conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# A copy of the License is located at
# http://www.huawei.com
# This program is distributed in the hope that it will be useful, but WITHOUT any warranty of any kind.
# ----------------------------------------------------------------------------------------------------------

"""DenseToSparse operator package.

Importing this package triggers the ``aten::_to_sparse*`` NPU registrations
only; no other operator is touched, and no JIT compilation happens until the
first NPU call.
"""

__all__ = []

from .dense_to_sparse import TORCH_NPU_SPARSE_FACADE_APIS
