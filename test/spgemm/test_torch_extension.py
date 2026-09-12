# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""SpGEMM Torch Extension 的端到端测试。"""

from dataclasses import dataclass

import torch
import torch_npu  # noqa: F401
import cann_ops_sparse  # noqa: F401


@dataclass(frozen=True)
class CsrData:
    """用于构造 CSR 测试张量的主机数据。"""

    crow: list[int]
    col: list[int]
    values: list[complex | float | int]
    shape: tuple[int, int]


def _csr(data, dtype, device, index_dtype=torch.int32):
    return torch.sparse_csr_tensor(
        torch.tensor(data.crow, dtype=index_dtype, device=device),
        torch.tensor(data.col, dtype=index_dtype, device=device),
        torch.tensor(data.values, dtype=dtype, device=device),
        size=data.shape,
        device=device,
    )


def _parts(tensor):
    torch.npu.synchronize()
    return tensor.crow_indices().cpu().tolist(), tensor.col_indices().cpu().tolist(), tensor.values().cpu()


def test_spgemm_torch_extension():
    """验证 CSR、COO 公开路径、索引转换和支持的数据类型。"""
    device = "npu:{}".format(torch.npu.current_device())
    for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.complex64):
        matrix = _csr(CsrData([0, 2, 3], [0, 1, 1], [1, 2, 3], (2, 2)), dtype, device)
        output = torch.sparse.mm(matrix, matrix)
        crow, col, values = _parts(output)
        assert output.layout == torch.sparse_csr
        assert crow == [0, 2, 3] and col == [0, 1, 1]
        torch.testing.assert_close(values, torch.tensor([1, 8, 9], dtype=dtype), rtol=0, atol=0)

    matrix_a = _csr(
        CsrData([0, 2], [0, 1], [1.0, 1.0], (1, 2)), torch.float32, device, torch.int64
    )
    matrix_b = _csr(
        CsrData([0, 1, 2], [0, 0], [1.0, -1.0], (2, 1)), torch.float32, device, torch.int64
    )
    crow, col, values = _parts(torch.sparse.mm(matrix_a, matrix_b))
    assert crow == [0, 1] and col == [0] and values.tolist() == [0.0]

    matrix_a = _csr(CsrData([0, 1], [0], [1.0], (1, 2)), torch.float32, device)
    matrix_b = _csr(CsrData([0, 0, 1], [0], [1.0], (2, 1)), torch.float32, device)
    crow, col, values = _parts(torch.sparse.mm(matrix_a, matrix_b))
    assert crow == [0, 0] and col == [] and values.numel() == 0

    matrix_a = _csr(CsrData([0, 0, 1, 1], [1], [2.0], (3, 3)), torch.float32, device)
    matrix_b = _csr(
        CsrData([0, 1, 1, 2], [2, 0], [3.0, 4.0], (3, 3)), torch.float32, device
    )
    crow, col, values = _parts(torch.sparse.mm(matrix_a, matrix_b))
    assert crow == [0, 0, 0, 0] and col == [] and values.numel() == 0

    indices = torch.tensor([[0, 0, 1], [0, 1, 1]], dtype=torch.int64, device=device)
    matrix = torch.sparse_coo_tensor(
        indices, torch.tensor([1.0, 2.0, 3.0], device=device), (2, 2), device=device
    ).coalesce()
    output = torch.sparse.mm(matrix, matrix)
    assert output.layout == torch.sparse_coo and output.is_coalesced()
    assert output.indices().cpu().tolist() == [[0, 0, 1], [0, 1, 1]]
    assert output.values().cpu().tolist() == [1.0, 8.0, 9.0]


def test_spgemm_non_default_stream_and_complex_alpha():
    """验证 Torch 队列顺序及 Scalar 到 pybind 的复数透传。"""
    device = "npu:{}".format(torch.npu.current_device())
    stream = torch.npu.Stream(device=device)
    with torch.npu.stream(stream):
        matrix = _csr(
            CsrData([0, 2, 3], [0, 1, 1], [1, 2, 3], (2, 2)),
            torch.complex64,
            device,
            torch.int64,
        )
        empty = _csr(CsrData([0, 0, 0], [], [], (2, 2)), torch.complex64, device)
        output = torch.sparse.addmm(
            empty, matrix, matrix, beta=0, alpha=1 + 2j
        )
    crow, col, values = _parts(output)
    assert crow == [0, 2, 3] and col == [0, 1, 1]
    torch.testing.assert_close(
        values, torch.tensor([1 + 2j, 8 + 16j, 9 + 18j], dtype=torch.complex64), rtol=0, atol=0
    )
