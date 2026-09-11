# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Install the optional ``torch_npu.sparse`` Python façade.

The actual kernel registration remains on the standard ATen operator.  This
module only exposes the matching public Python entry point after confirming
that no other package already owns the namespace.
"""

import sys
from types import ModuleType


_MODULE_NAME = "torch_npu.sparse"
_OWNER_MARKER = "_cann_ops_sparse_owner"


def install_namespace(torch_npu, facade_apis) -> ModuleType:
    """Create ``torch_npu.sparse`` from operator-declared public APIs."""
    current_attribute = getattr(torch_npu, "sparse", None)
    current_module = sys.modules.get(_MODULE_NAME)

    if current_attribute is not None or current_module is not None:
        if (
            current_attribute is current_module
            and current_module is not None
            and getattr(current_module, _OWNER_MARKER, False)
        ):
            return current_module
        raise RuntimeError(
            "cannot register torch_npu.sparse: the namespace is already owned by "
            "another module"
        )

    namespace = ModuleType(
        _MODULE_NAME,
        "Sparse ATen façades provided by the cann_ops_sparse package.",
    )
    for name, function in facade_apis.items():
        setattr(namespace, name, function)
    namespace.__all__ = sorted(facade_apis)
    setattr(namespace, _OWNER_MARKER, True)
    setattr(torch_npu, "sparse", namespace)
    sys.modules[_MODULE_NAME] = namespace
    return namespace
