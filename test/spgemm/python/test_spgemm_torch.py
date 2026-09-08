#!/usr/bin/env python3
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""SpGEMM PyTorch稀疏分发适配层端到端测试。"""

from __future__ import annotations

import argparse
import logging
import os
from dataclasses import dataclass
from pathlib import Path

import torch
import torch_npu  # noqa: F401


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class CsrData:
    """CSR host data used to construct one NPU test tensor."""

    crow: list[int]
    col: list[int]
    values: list[complex | float]
    shape: tuple[int, int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--device", type=int, default=0)
    return parser.parse_args()


def load_adapter(path: Path) -> None:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    os.environ["OPS_SPARSE_TORCH_LIBRARY"] = str(resolved)
    import ops_sparse_torch  # noqa: F401, PLC0415


def csr(data: CsrData, dtype: torch.dtype, device: str,
        index_dtype: torch.dtype = torch.int32) -> torch.Tensor:
    return torch.sparse_csr_tensor(
        torch.tensor(data.crow, dtype=index_dtype, device=device),
        torch.tensor(data.col, dtype=index_dtype, device=device),
        torch.tensor(data.values, dtype=dtype, device=device),
        size=data.shape,
        device=device,
    )


def cpu_parts(tensor: torch.Tensor) -> tuple[list[int], list[int], torch.Tensor]:
    torch.npu.synchronize()
    return (
        tensor.crow_indices().cpu().tolist(),
        tensor.col_indices().cpu().tolist(),
        tensor.values().cpu(),
    )


def test_required_dtypes(device: str) -> None:
    for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.complex64):
        a = csr(CsrData([0, 2, 3], [0, 1, 1], [1, 2, 3], (2, 2)), dtype, device)
        output = torch.sparse.mm(a, a)
        crow, col, values = cpu_parts(output)
        assert output.layout == torch.sparse_csr
        assert crow == [0, 2, 3]
        assert col == [0, 1, 1]
        torch.testing.assert_close(
            values, torch.tensor([1, 8, 9], dtype=dtype), rtol=0, atol=0
        )


def test_structure_edges(device: str) -> None:
    # PyTorch accepts int64 indices; the adapter converts them on NPU before
    # entering the int32 aclsparse contract.
    a = csr(CsrData([0, 2], [0, 1], [1.0, 1.0], (1, 2)),
            torch.float32, device, torch.int64)
    b = csr(CsrData([0, 1, 2], [0, 0], [1.0, -1.0], (2, 1)),
            torch.float32, device, torch.int64)
    output = torch.sparse.mm(a, b)
    crow, col, values = cpu_parts(output)
    assert crow == [0, 1] and col == [0]
    assert values.tolist() == [0.0]  # explicit zero is retained in nnz(C)

    a = csr(CsrData([0, 1], [0], [1.0], (1, 2)), torch.float32, device)
    b = csr(CsrData([0, 0, 1], [0], [1.0], (2, 1)), torch.float32, device)
    output = torch.sparse.mm(a, b)
    crow, col, values = cpu_parts(output)
    assert crow == [0, 0] and col == [] and values.numel() == 0

    a = csr(CsrData([0, 0, 1, 1], [1], [2.0], (3, 3)), torch.float32, device)
    b = csr(CsrData([0, 1, 1, 2], [2, 0], [3.0, 4.0], (3, 3)), torch.float32, device)
    output = torch.sparse.mm(a, b)
    crow, col, values = cpu_parts(output)
    assert crow == [0, 0, 0, 0] and col == [] and values.numel() == 0


def test_coo_public_path(device: str) -> None:
    indices = torch.tensor([[0, 0, 1], [0, 1, 1]], dtype=torch.int64, device=device)
    values = torch.tensor([1.0, 2.0, 3.0], device=device)
    a = torch.sparse_coo_tensor(indices, values, (2, 2), device=device).coalesce()
    output = torch.sparse.mm(a, a)
    assert output.layout == torch.sparse_coo and output.is_coalesced()
    assert output.indices().cpu().tolist() == [[0, 0, 1], [0, 1, 1]]
    assert output.values().cpu().tolist() == [1.0, 8.0, 9.0]


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    torch.npu.set_device(args.device)
    load_adapter(args.library)
    device = f"npu:{args.device}"
    test_required_dtypes(device)
    test_structure_edges(device)
    test_coo_public_path(device)
    LOGGER.info("SpGEMM PyTorch dispatch tests passed")


if __name__ == "__main__":
    main()
