# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""End-to-end tests for the aclsparseGather ATen registration."""

import os
import subprocess
import sys

import pytest

torch = pytest.importorskip("torch")
torch_npu = pytest.importorskip("torch_npu")
cann_ops_sparse = pytest.importorskip("cann_ops_sparse")

DEVICE = "npu:0"
VALUE_DTYPES = (torch.float16, torch.bfloat16, torch.float32, torch.complex64)


def task_distribution(size: int, dtype: torch.dtype, seed: int) -> torch.Tensor:
    """Build the required uniform, normal and boundary-value distribution."""
    generator = torch.Generator().manual_seed(seed)
    uniform_count = size * 7 // 10
    normal_count = size * 2 // 10
    special_count = size - uniform_count - normal_count
    real = torch.empty(size, dtype=torch.float32)
    real[:uniform_count].uniform_(-1.0, 1.0, generator=generator)
    real[uniform_count:uniform_count + normal_count].normal_(0.0, 1.0, generator=generator)
    boundary = torch.tensor(
        [0.0, -0.0, float("inf"), -float("inf"), float("nan"),
         torch.finfo(torch.float32).tiny, torch.finfo(torch.float32).max],
        dtype=torch.float32,
    )
    if special_count:
        repeats = (special_count + boundary.numel() - 1) // boundary.numel()
        real[-special_count:] = boundary.repeat(repeats)[:special_count]
    if dtype == torch.complex64:
        return torch.complex(real, torch.flip(real, dims=(0,)))
    return real.to(dtype)


def assert_bit_exact(actual: torch.Tensor, expected: torch.Tensor) -> None:
    actual_bytes = actual.detach().cpu().contiguous().view(torch.uint8)
    expected_bytes = expected.detach().cpu().contiguous().view(torch.uint8)
    assert torch.equal(actual_bytes, expected_bytes)


def test_aten_index_select_dispatches_on_npu():
    source_cpu = torch.arange(8, dtype=torch.float32)
    index_cpu = torch.tensor([7, 0, 3], dtype=torch.int32)
    output = torch.ops.aten.index_select.default(
        source_cpu.to(DEVICE), 0, index_cpu.to(DEVICE)
    )
    torch.npu.synchronize()
    expected = torch.index_select(source_cpu, 0, index_cpu.to(torch.int64))
    assert_bit_exact(output, expected)


def test_import_does_not_trigger_jit(tmp_path):
    cache = tmp_path / "extensions"
    cache.mkdir()
    environment = dict(os.environ, TORCH_EXTENSIONS_DIR=str(cache))
    script = (
        "import torch, torch_npu, cann_ops_sparse\n"
        "from cann_ops_sparse.ops.sparse.gather.gather import _gather_op_builder as b\n"
        "print('LOADED:', list(type(b)._loaded_ops.keys()))\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script], env=environment, capture_output=True, text=True,
        timeout=600, check=False
    )
    assert completed.returncode == 0, completed.stderr
    assert "LOADED: []" in completed.stdout
    assert not any(cache.iterdir()), "importing the package produced build artifacts"


@pytest.mark.parametrize("dtype", VALUE_DTYPES)
def test_required_dtypes_are_bit_exact(dtype):
    index_cpu = torch.tensor([15, 0, 7, 7, 1, 14, 3, 15], dtype=torch.int32)
    source_cpu = task_distribution(16, dtype, 20260902)
    source = source_cpu.to(DEVICE)
    index = index_cpu.to(DEVICE)
    source_before = source.cpu()
    index_before = index.cpu()

    output = torch.index_select(source, 0, index)
    repeated = torch.index_select(source, -1, index)
    torch.npu.synchronize()
    expected = torch.index_select(source_cpu, 0, index_cpu.to(torch.int64))

    assert_bit_exact(output, expected)
    assert_bit_exact(repeated, expected)
    assert_bit_exact(source, source_before)
    assert torch.equal(index.cpu(), index_before)
    assert output.data_ptr() != source.data_ptr()
    assert output.data_ptr() != index.data_ptr()


@pytest.mark.parametrize(
    "size,index_values",
    ((0, []), (1, [0]), (257, [256, 0, 128, 128, 1]),
     (1024, list(range(1023, 767, -1)))),
)
def test_dynamic_boundaries_repeats_and_tail(size, index_values):
    source_cpu = torch.arange(size, dtype=torch.float32)
    index_cpu = torch.tensor(index_values, dtype=torch.int32)
    output = torch.index_select(source_cpu.to(DEVICE), 0, index_cpu.to(DEVICE))
    torch.npu.synchronize()
    expected = torch.index_select(source_cpu, 0, index_cpu.to(torch.int64))
    assert_bit_exact(output, expected)


@pytest.mark.parametrize("dtype", VALUE_DTYPES)
@pytest.mark.parametrize(
    "size,index_values",
    ((1, [0, 0]), (1, [0] * 257), (3, [2, 0, 2, 1, 0, 2, 2, 1]),
     (7, [6, 0, 3, 3, 1, 6, 2] * 5), (16, [15, 0, 7, 7, 1, 14, 3, 15] * 9)),
)
def test_repeated_indices_can_exceed_input_length(dtype, size, index_values):
    source_cpu = task_distribution(size, dtype, 20260918)
    if size == 1:
        value = 1.25 - 2.5j if dtype == torch.complex64 else -1.25
        source_cpu = torch.tensor([value], dtype=dtype)
    index_cpu = torch.tensor(index_values, dtype=torch.int32)
    source = source_cpu.to(DEVICE)
    index = index_cpu.to(DEVICE)

    output = torch.index_select(source, 0, index)
    repeated = torch.ops.aten.index_select.default(source, -1, index)
    torch.npu.synchronize()
    expected = torch.index_select(source_cpu, 0, index_cpu.to(torch.int64))

    assert output.shape == index.shape
    assert output.dtype == source.dtype
    assert output.device == source.device
    assert_bit_exact(output, expected)
    assert_bit_exact(repeated, expected)
    assert_bit_exact(source, source_cpu)
    assert torch.equal(index.cpu(), index_cpu)
    assert output.data_ptr() != source.data_ptr()
    assert output.data_ptr() != index.data_ptr()


def test_non_default_stream_preserves_ordering():
    stream = torch.npu.Stream()
    with torch.npu.stream(stream):
        source = (torch.arange(64, dtype=torch.float32, device=DEVICE) * 3.0) + 1.0
        index = torch.tensor([63, 0, 17, 17], dtype=torch.int32, device=DEVICE)
        output = torch.index_select(source, 0, index)
    stream.synchronize()
    assert_bit_exact(output, torch.tensor([190.0, 1.0, 52.0, 52.0]))


def test_long_repeated_indices_preserve_non_default_stream_ordering():
    stream = torch.npu.Stream()
    with torch.npu.stream(stream):
        source = (torch.arange(3, dtype=torch.float32, device=DEVICE) * 3.0) + 1.0
        index = torch.tensor([2, 0, 2, 1, 0, 2, 2, 1], dtype=torch.int32, device=DEVICE)
        output = torch.index_select(source, 0, index)
        following = output + 2.0
    stream.synchronize()
    assert_bit_exact(output, torch.tensor([7.0, 1.0, 7.0, 4.0, 1.0, 7.0, 7.0, 4.0]))
    assert_bit_exact(following, torch.tensor([9.0, 3.0, 9.0, 6.0, 3.0, 9.0, 9.0, 6.0]))


@pytest.mark.parametrize("invalid_index", (-1, 4))
def test_rejects_out_of_bounds_index_before_gather_launch(invalid_index):
    source = torch.arange(4, dtype=torch.float32, device=DEVICE)
    index = torch.tensor([invalid_index], dtype=torch.int32, device=DEVICE)
    with pytest.raises(IndexError, match="index out of range"):
        torch.index_select(source, 0, index)


def test_bounds_cache_tracks_index_mutation_and_input_size():
    source = torch.arange(4, dtype=torch.float32, device=DEVICE)
    index = torch.tensor([0], dtype=torch.int32, device=DEVICE)
    assert_bit_exact(torch.index_select(source, 0, index), torch.tensor([0.0]))

    index.fill_(-1)
    with pytest.raises(IndexError, match="index out of range"):
        torch.index_select(source, 0, index)

    index.fill_(3)
    assert_bit_exact(torch.index_select(source, 0, index), torch.tensor([3.0]))
    with pytest.raises(IndexError, match="index out of range"):
        torch.index_select(source[:3], 0, index)


def test_rejects_unsupported_contracts_without_cpu_fallback():
    source = torch.arange(16, dtype=torch.float32, device=DEVICE)
    index = torch.tensor([0, 1, 2], dtype=torch.int32, device=DEVICE)
    invalid_calls = (
        lambda: torch.index_select(source.reshape(4, 4), 0, index),
        lambda: torch.index_select(source, 0, index.reshape(1, 3)),
        lambda: torch.index_select(source, 0, index.to(torch.int64)),
        lambda: torch.index_select(source.to(torch.int32), 0, index),
        lambda: torch.index_select(source, 1, index),
        lambda: torch.index_select(source[::2], 0, index),
        lambda: torch.index_select(source, 0, source[:6].view(torch.int32)[::2]),
        lambda: torch.index_select(source[:0], 0, torch.zeros(1, dtype=torch.int32, device=DEVICE)),
        lambda: torch.index_select(source.requires_grad_(True), 0, index),
    )
    for call in invalid_calls:
        with pytest.raises((RuntimeError, IndexError)):
            call()
