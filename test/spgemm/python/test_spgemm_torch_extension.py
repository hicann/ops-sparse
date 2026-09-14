# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""SpGEMM Torch Extension 注册的端到端测试。

在真实 NPU 上对已安装的 ``cann_ops_sparse`` wheel 运行::

    bash build.sh --torch_extension --ops=spgemm
    python3 -m pip install build_out/cann_ops_sparse-*.whl --force-reinstall --no-deps
    OPS_SPARSE_LIB_DIR=$PWD/build_out/lib64 python3 -m pytest test/spgemm/python/ -v

``OPS_SPARSE_LIB_DIR`` 为 JIT 链接步骤指定构建树，无需预先安装运行包。
"""

import os
import subprocess
import sys
import threading
from dataclasses import dataclass
from typing import List, Tuple

import pytest

torch = pytest.importorskip("torch")
torch_npu = pytest.importorskip("torch_npu")
cann_ops_sparse = pytest.importorskip("cann_ops_sparse")

DEVICE = "npu:0"
VALUE_DTYPES = (torch.float16, torch.bfloat16, torch.float32, torch.complex64)
# NPU complex 张量的 to_dense() 触发 aclnnIndexAdd dtype 缺陷，
# 与 SpGEMM 无关，因此 complex 用例改为比较 CSR 分量。
DENSE_COMPARABLE_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


@dataclass
class CsrData:
    crow: List[int]
    col: List[int]
    values: List[float]
    shape: Tuple[int, int]


def csr(data: CsrData, dtype: torch.dtype, device: str = DEVICE,
        index_dtype: torch.dtype = torch.int32) -> torch.Tensor:
    return torch.sparse_csr_tensor(
        torch.tensor(data.crow, dtype=index_dtype, device=device),
        torch.tensor(data.col, dtype=index_dtype, device=device),
        torch.tensor(data.values, dtype=dtype, device=device),
        size=data.shape,
        device=device,
    )


def cpu_parts(tensor: torch.Tensor) -> Tuple[List[int], List[int], torch.Tensor]:
    torch.npu.synchronize()
    return (
        tensor.crow_indices().cpu().tolist(),
        tensor.col_indices().cpu().tolist(),
        tensor.values().cpu(),
    )


def random_dense(rows: int, cols: int, dtype: torch.dtype, density: float = 0.3,
                 seed: int = 0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    mask = (torch.rand(rows, cols, generator=generator) < density).to(torch.float32)
    dense = mask * torch.randn(rows, cols, generator=generator)
    if dtype == torch.complex64:
        imaginary = mask * torch.randn(rows, cols, generator=generator)
        return (dense + 1j * imaginary).to(dtype)
    return dense.to(dtype)


def tolerance(dtype: torch.dtype) -> float:
    return {torch.float16: 5e-3, torch.bfloat16: 4e-2, torch.float32: 1e-5,
            torch.complex64: 1e-5}[dtype]


# ---------------------------------------------------------------------------
# 1. 注册验证
# ---------------------------------------------------------------------------

def test_dispatch_keys_are_registered():
    """CSR 和 COO 操作数能正确路由到 NPU 实现。"""
    csr_a = csr(CsrData([0, 1], [0], [1.0], (1, 1)), torch.float32)
    csr_out = torch.sparse.mm(csr_a, csr_a)
    torch.npu.synchronize()
    assert csr_out.device == csr_a.device
    assert csr_out.layout == torch.sparse_csr

    coo_a = csr_a.to_sparse(layout=torch.sparse_coo)
    coo_out = torch.sparse.mm(coo_a, coo_a)
    torch.npu.synchronize()
    assert coo_out.device == csr_a.device
    assert coo_out.layout == torch.sparse_coo


def test_import_does_not_trigger_jit(tmp_path):
    """导入包不应触发 JIT 编译（规范 3.1）。"""
    cache = tmp_path / "extensions"
    cache.mkdir()
    environment = dict(os.environ, TORCH_EXTENSIONS_DIR=str(cache))
    script = (
        "import torch, torch_npu, cann_ops_sparse\n"
        "from cann_ops_sparse.ops.sparse.spgemm.spgemm import _spgemm_op_builder as b\n"
        "print('LOADED:', list(type(b)._loaded_ops.keys()))\n"
    )
    completed = subprocess.run([sys.executable, "-c", script], env=environment,
                               capture_output=True, text=True, timeout=600)
    assert completed.returncode == 0, completed.stderr
    assert "LOADED: []" in completed.stdout
    assert not any(cache.iterdir()), "importing the package produced build artifacts"


def test_facade_exposes_public_apis():
    """torch_npu.sparse 转发公共 API，而非私有入口。"""
    assert torch_npu.sparse.mm is torch.sparse.mm
    assert torch_npu.sparse.addmm is torch.sparse.addmm


# ---------------------------------------------------------------------------
# 2. 功能与精度
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dtype", VALUE_DTYPES)
def test_required_dtypes(dtype):
    """迁移自旧适配器测试：精确验证结构和数值。"""
    a = csr(CsrData([0, 2, 3], [0, 1, 1], [1, 2, 3], (2, 2)), dtype)
    output = torch.sparse.mm(a, a)
    crow, col, values = cpu_parts(output)
    assert output.layout == torch.sparse_csr
    assert crow == [0, 2, 3]
    assert col == [0, 1, 1]
    torch.testing.assert_close(values, torch.tensor([1, 8, 9], dtype=dtype), rtol=0, atol=0)


@pytest.mark.parametrize("dtype", VALUE_DTYPES)
def test_matches_dense_reference(dtype):
    a = random_dense(16, 12, dtype, seed=1)
    b = random_dense(12, 10, dtype, seed=2)
    output = torch.sparse.mm(a.to_sparse_csr().to(DEVICE), b.to_sparse_csr().to(DEVICE))
    reference = (a @ b).to_sparse_csr()
    out_crow, out_col, out_values = cpu_parts(output)
    assert out_crow == reference.crow_indices().tolist()
    assert out_col == reference.col_indices().tolist()
    torch.testing.assert_close(out_values, reference.values(), rtol=tolerance(dtype),
                               atol=tolerance(dtype))


@pytest.mark.parametrize("index_dtype", (torch.int32, torch.int64))
def test_both_index_dtypes(index_dtype):
    """PyTorch 允许 int64 索引；封装层在 device 上窄化为 int32。"""
    a = csr(CsrData([0, 2, 3], [0, 1, 1], [1.0, 2.0, 3.0], (2, 2)), torch.float32,
            index_dtype=index_dtype)
    crow, col, values = cpu_parts(torch.sparse.mm(a, a))
    assert crow == [0, 2, 3] and col == [0, 1, 1]
    assert values.tolist() == [1.0, 8.0, 9.0]


@pytest.mark.parametrize("beta,alpha", [(0.0, 1.0), (1.0, 1.0), (2.0, 3.0), (-1.5, 0.5)])
def test_addmm_with_beta(beta, alpha):
    """beta != 0 时累加稀疏加数；旧适配器不支持此功能。"""
    a = random_dense(8, 8, torch.float32, seed=3)
    b = random_dense(8, 8, torch.float32, seed=4)
    c = random_dense(8, 8, torch.float32, seed=5)
    output = torch.sparse.addmm(c.to_sparse_csr().to(DEVICE), a.to_sparse_csr().to(DEVICE),
                                b.to_sparse_csr().to(DEVICE), beta=beta, alpha=alpha)
    reference = beta * c + alpha * (a @ b)
    torch.npu.synchronize()
    torch.testing.assert_close(output.to_dense().cpu(), reference, rtol=1e-4, atol=1e-4)


def test_addmm_does_not_alias_the_addend():
    """Copy 会重写 matC 的 row offsets，因此 self 不能成为输出。"""
    a = random_dense(8, 8, torch.float32, seed=6)
    b = random_dense(8, 8, torch.float32, seed=7)
    c = random_dense(8, 8, torch.float32, seed=8)
    addend = c.to_sparse_csr().to(DEVICE)
    before = addend.crow_indices().cpu().clone()
    output = torch.sparse.addmm(addend, a.to_sparse_csr().to(DEVICE),
                                b.to_sparse_csr().to(DEVICE), beta=1.0, alpha=1.0)
    torch.npu.synchronize()
    assert torch.equal(addend.crow_indices().cpu(), before)
    assert output.values().data_ptr() != addend.values().data_ptr()


@pytest.mark.parametrize("alpha", (1.0, 2.5, -1.0))
def test_alpha_scales_the_product(alpha):
    """空 alpha 会被库读为 0.0，必须确保正确传递。"""
    a = csr(CsrData([0, 2, 3], [0, 1, 1], [1.0, 2.0, 3.0], (2, 2)), torch.float32)
    empty = csr(CsrData([0, 0, 0], [], [], (2, 2)), torch.float32)
    output = torch.sparse.addmm(empty, a, a, beta=0.0, alpha=alpha)
    _, _, values = cpu_parts(output)
    torch.testing.assert_close(values, torch.tensor([1.0, 8.0, 9.0]) * alpha,
                               rtol=1e-6, atol=1e-6)


def test_coo_public_path():
    """迁移自旧测试：COO 上的 torch.sparse.mm 路由到 _sparse_sparse_matmul。"""
    a = csr(CsrData([0, 2, 3], [0, 1, 1], [1.0, 2.0, 3.0], (2, 2)), torch.float32)
    coo = a.to_sparse(layout=torch.sparse_coo)
    output = torch.sparse.mm(coo, coo)
    torch.npu.synchronize()
    assert output.layout == torch.sparse_coo
    assert output.is_coalesced()
    assert output.indices().cpu().tolist() == [[0, 0, 1], [0, 1, 1]]
    assert output.values().cpu().tolist() == [1.0, 8.0, 9.0]


@pytest.mark.parametrize("dtype", VALUE_DTYPES)
def test_coo_matches_csr_path(dtype):
    """两种布局对同一矩阵必须产生相同乘积。"""
    a = random_dense(10, 10, dtype, seed=13).to(DEVICE)
    # 通过 CSR 构建而非 dense.to_sparse()：aclnnNonzero 不支持
    # complex64，这是稠密化缺陷，与 SpGEMM 无关。
    as_csr = a.to_sparse_csr()
    as_coo = as_csr.to_sparse(layout=torch.sparse_coo)
    from_csr = torch.sparse.mm(as_csr, as_csr)
    from_coo = torch.sparse.mm(as_coo, as_coo)
    torch.npu.synchronize()
    reference = from_csr.to_sparse(layout=torch.sparse_coo).coalesce()
    assert from_coo.indices().cpu().tolist() == reference.indices().cpu().tolist()
    torch.testing.assert_close(from_coo.values().cpu(), reference.values().cpu(), rtol=0, atol=0)


def test_complex_scalar_on_complex_dtype():
    a = csr(CsrData([0, 2, 3], [0, 1, 1], [1, 2, 3], (2, 2)), torch.complex64)
    empty = csr(CsrData([0, 0, 0], [], [], (2, 2)), torch.complex64)
    output = torch.sparse.addmm(empty, a, a, beta=0.0, alpha=2j)
    _, _, values = cpu_parts(output)
    torch.testing.assert_close(values,
                               torch.tensor([1, 8, 9], dtype=torch.complex64) * 2j,
                               rtol=1e-6, atol=1e-6)


# ---------------------------------------------------------------------------
# 3. 边界条件
# ---------------------------------------------------------------------------

def test_explicit_zero_is_retained():
    """迁移自旧测试：抵消为零的乘积仍占据一个 nnz 槽位。"""
    a = csr(CsrData([0, 2], [0, 1], [1.0, 1.0], (1, 2)), torch.float32,
            index_dtype=torch.int64)
    b = csr(CsrData([0, 1, 2], [0, 0], [1.0, -1.0], (2, 1)), torch.float32,
            index_dtype=torch.int64)
    crow, col, values = cpu_parts(torch.sparse.mm(a, b))
    assert crow == [0, 1] and col == [0]
    assert values.tolist() == [0.0]


def test_no_matching_products():
    """迁移自旧测试：不相交模式产生空但结构有效的结果。"""
    a = csr(CsrData([0, 1], [0], [1.0], (1, 2)), torch.float32)
    b = csr(CsrData([0, 0, 1], [0], [1.0], (2, 1)), torch.float32)
    crow, col, values = cpu_parts(torch.sparse.mm(a, b))
    assert crow == [0, 0] and col == [] and values.numel() == 0


def test_empty_rows_are_preserved():
    """迁移自旧测试：即使 nnz(C)=0，row offsets 长度仍为 M+1。"""
    a = csr(CsrData([0, 0, 1, 1], [1], [2.0], (3, 3)), torch.float32)
    b = csr(CsrData([0, 1, 1, 2], [2, 0], [3.0, 4.0], (3, 3)), torch.float32)
    crow, col, values = cpu_parts(torch.sparse.mm(a, b))
    assert crow == [0, 0, 0, 0] and col == [] and values.numel() == 0


def test_zero_nnz_operands():
    a = csr(CsrData([0, 0, 0], [], [], (2, 3)), torch.float32)
    b = csr(CsrData([0, 0, 0, 0], [], [], (3, 2)), torch.float32)
    output = torch.sparse.mm(a, b)
    crow, col, values = cpu_parts(output)
    assert output.shape == (2, 2)
    assert crow == [0, 0, 0] and col == [] and values.numel() == 0


def test_minimal_shape():
    a = csr(CsrData([0, 1], [0], [3.0], (1, 1)), torch.float32)
    crow, col, values = cpu_parts(torch.sparse.mm(a, a))
    assert crow == [0, 1] and col == [0] and values.tolist() == [9.0]


def test_zero_row_dimension():
    a = csr(CsrData([0], [], [], (0, 3)), torch.float32)
    b = csr(CsrData([0, 0, 0, 0], [], [], (3, 2)), torch.float32)
    output = torch.sparse.mm(a, b)
    torch.npu.synchronize()
    assert output.shape == (0, 2)
    assert output.values().numel() == 0


# ---------------------------------------------------------------------------
# 4. 非法输入
# ---------------------------------------------------------------------------

def test_mismatched_dtypes_are_rejected():
    a = csr(CsrData([0, 1], [0], [1.0], (1, 1)), torch.float32)
    b = csr(CsrData([0, 1], [0], [1.0], (1, 1)), torch.float16)
    with pytest.raises(RuntimeError, match="dtype"):
        torch.sparse.mm(a, b)


def test_mismatched_shapes_are_rejected():
    a = csr(CsrData([0, 1], [0], [1.0], (1, 2)), torch.float32)
    b = csr(CsrData([0, 1], [0], [1.0], (1, 2)), torch.float32)
    with pytest.raises(RuntimeError, match="cannot be multiplied"):
        torch.sparse.mm(a, b)


def test_strided_operand_is_rejected():
    """Python 侧注册不会穿透，因此需在此处检查布局。"""
    a = csr(CsrData([0, 1], [0], [1.0], (1, 1)), torch.float32)
    dense = torch.ones(1, 1, device=DEVICE)
    with pytest.raises((RuntimeError, NotImplementedError, TypeError)):
        torch.sparse.addmm(a, a, dense)


def test_unsupported_value_dtype_is_rejected():
    a = csr(CsrData([0, 1], [0], [1.0], (1, 1)), torch.float64)
    with pytest.raises(RuntimeError, match="float16, bfloat16, float32 and complex64"):
        torch.sparse.mm(a, a)


def test_complex_scalar_on_real_dtype_is_rejected():
    a = csr(CsrData([0, 1], [0], [1.0], (1, 1)), torch.float32)
    with pytest.raises(RuntimeError, match="complex scalar"):
        torch.sparse.addmm(a, a, a, beta=0.0, alpha=1j)


def test_addend_shape_is_validated():
    a = csr(CsrData([0, 1, 1], [0], [1.0], (2, 2)), torch.float32)
    wrong = csr(CsrData([0, 1], [0], [1.0], (1, 1)), torch.float32)
    with pytest.raises(RuntimeError, match="self must have shape"):
        torch.sparse.addmm(wrong, a, a, beta=1.0, alpha=1.0)


@pytest.mark.parametrize("beta", (0.0, 1.0))
def test_addend_is_validated_for_every_beta(beta):
    """aten::addmm 对所有 beta（含零）都校验 self，封装层不能在 beta=0 路径上跳过检查。"""
    a = csr(CsrData([0, 2, 3], [0, 1, 1], [1.0, 2.0, 3.0], (2, 2)), torch.float32)
    wrong_dtype = csr(CsrData([0, 0, 0], [], [], (2, 2)), torch.float16)
    with pytest.raises(RuntimeError, match="self must share the dtype"):
        torch.sparse.addmm(wrong_dtype, a, a, beta=beta, alpha=1.0)

    wrong_shape = csr(CsrData([0, 1], [0], [1.0], (1, 1)), torch.float32)
    with pytest.raises(RuntimeError, match="self must have shape"):
        torch.sparse.addmm(wrong_shape, a, a, beta=beta, alpha=1.0)

    dense_addend = torch.zeros(2, 2, device=DEVICE)
    with pytest.raises(RuntimeError, match="self must use the sparse CSR layout"):
        torch.sparse.addmm(dense_addend, a, a, beta=beta, alpha=1.0)


def test_cpu_operands_do_not_reach_the_npu_kernel():
    """注册绑定在 PrivateUse1 上，不应捕获 CPU 工作。"""
    coo = torch.tensor([[1.0, 0.0], [0.0, 1.0]]).to_sparse()
    result = torch.sparse.mm(coo, coo)
    assert result.device.type == "cpu"

    # 无 MKL 时 PyTorch 不支持 CPU CSR × CSR；错误应来自 PyTorch 而非本封装层。
    csr_cpu = torch.tensor([[1.0, 0.0], [0.0, 1.0]]).to_sparse_csr()
    with pytest.raises(RuntimeError) as failure:
        torch.sparse.mm(csr_cpu, csr_cpu)
    assert "SpGEMM" not in str(failure.value)


# ---------------------------------------------------------------------------
# 5. 设备与异步
# ---------------------------------------------------------------------------

def test_repeated_calls_are_stable():
    a = random_dense(24, 24, torch.float32, seed=9)
    operand = a.to_sparse_csr().to(DEVICE)
    reference = None
    for _ in range(5):
        _, _, values = cpu_parts(torch.sparse.mm(operand, operand))
        if reference is None:
            reference = values
        else:
            torch.testing.assert_close(values, reference, rtol=0, atol=0)


@pytest.mark.skipif(torch.npu.device_count() < 2, reason="needs a second NPU")
def test_non_default_device():
    """DeviceGuard 必须在分配输出前生效。"""
    a = random_dense(12, 12, torch.float32, seed=10)
    operand = a.to_sparse_csr().to("npu:1")
    output = torch.sparse.mm(operand, operand)
    torch.npu.synchronize("npu:1")
    assert output.device == operand.device
    torch.testing.assert_close(output.to_dense().cpu(), a @ a, rtol=1e-5, atol=1e-5)


def test_non_default_stream():
    a = random_dense(12, 12, torch.float32, seed=11)
    operand = a.to_sparse_csr().to(DEVICE)
    stream = torch.npu.Stream()
    with torch.npu.stream(stream):
        output = torch.sparse.mm(operand, operand)
    stream.synchronize()
    torch.testing.assert_close(output.to_dense().cpu(), a @ a, rtol=1e-5, atol=1e-5)


def test_ordering_with_preceding_npu_work():
    """封装层读取调用方在同一 stream 上刚产生的输入。"""
    a = random_dense(16, 16, torch.float32, seed=12)
    scaled = (a.to(DEVICE) * 2.0).to_sparse_csr()
    output = torch.sparse.mm(scaled, scaled)
    torch.npu.synchronize()
    torch.testing.assert_close(output.to_dense().cpu(), (a * 2.0) @ (a * 2.0),
                               rtol=1e-4, atol=1e-4)


def test_concurrent_calls_from_threads():
    """per-handle 锁必须序列化描述符/workspace 状态，并发调用者不能互相破坏。"""
    a = random_dense(20, 20, torch.float32, seed=14)
    operand = a.to_sparse_csr().to(DEVICE)
    expected = None
    for _ in range(1):
        _, _, expected = cpu_parts(torch.sparse.mm(operand, operand))

    results: List[torch.Tensor] = []
    failures: List[BaseException] = []
    lock = threading.Lock()

    def worker() -> None:
        try:
            for _ in range(6):
                _, _, values = cpu_parts(torch.sparse.mm(operand, operand))
                with lock:
                    results.append(values)
        except BaseException as error:  # noqa: BLE001 - reported to the assertion below
            with lock:
                failures.append(error)

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert not failures, f"concurrent calls raised: {failures[0]!r}"
    assert len(results) == 24
    for values in results:
        torch.testing.assert_close(values, expected, rtol=0, atol=0)


# ---------------------------------------------------------------------------
# 6. 打包验证
# ---------------------------------------------------------------------------

def test_wheel_contents_are_complete():
    root = os.path.dirname(cann_ops_sparse.__file__)
    for relative in ("ops/sparse/spgemm/__init__.py", "ops/sparse/spgemm/spgemm.py",
                     "csrc/sparse/spgemm/spgemm.cpp", "common/aclsparse_common.h",
                     "common/cann_ops_sparse.h"):
        assert os.path.isfile(os.path.join(root, relative)), f"missing {relative}"


def test_builder_sources_resolve_inside_the_package():
    from cann_ops_sparse.ops.sparse.spgemm.spgemm import SpGemmOpBuilder

    builder = SpGemmOpBuilder()
    root = os.path.dirname(cann_ops_sparse.__file__)
    for source in builder.sources():
        assert os.path.isfile(os.path.join(root, source)), f"unresolved source {source}"


def test_extension_is_compiled_once_per_process():
    from cann_ops_sparse.ops.sparse.spgemm.spgemm import _spgemm_op_builder

    a = csr(CsrData([0, 1], [0], [1.0], (1, 1)), torch.float32)
    torch.sparse.mm(a, a)
    first = _spgemm_op_builder.load()
    torch.sparse.mm(a, a)
    assert _spgemm_op_builder.load() is first


# ---------------------------------------------------------------------------
# 7. 命名空间
# ---------------------------------------------------------------------------

def test_namespace_install_is_idempotent():
    """二次导入必须返回已有命名空间，不应抛异常。"""
    import cann_ops_sparse.torch_npu_sparse as facade

    before = torch_npu.sparse
    again = facade.install_namespace(torch_npu, cann_ops_sparse.ops.TORCH_NPU_SPARSE_FACADE_APIS)
    assert again is before
    assert torch_npu.sparse.mm is torch.sparse.mm


def test_facade_declares_the_spgemm_apis():
    from cann_ops_sparse.ops.sparse.spgemm.spgemm import TORCH_NPU_SPARSE_FACADE_APIS

    assert TORCH_NPU_SPARSE_FACADE_APIS == {"mm": torch.sparse.mm, "addmm": torch.sparse.addmm}
    for name in TORCH_NPU_SPARSE_FACADE_APIS:
        assert cann_ops_sparse.ops.TORCH_NPU_SPARSE_FACADE_APIS[name] is (
            TORCH_NPU_SPARSE_FACADE_APIS[name]
        )


def test_third_party_namespace_is_not_overwritten():
    import cann_ops_sparse.torch_npu_sparse as facade

    class Foreign:
        sparse = "owned by someone else"

    with pytest.raises(RuntimeError):
        facade.install_namespace(Foreign, {"mm": torch.sparse.mm})
    assert Foreign.sparse == "owned by someone else"
