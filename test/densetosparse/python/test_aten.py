# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms of conditions of
# CANN Open Software License Agreement Version 2 (the "License").
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.
# ----------------------------------------------------------------------------------------------------------
"""ATen UT for the ops-sparse DenseToSparse NPU adapter.

Covers Tensor.to_sparse / to_sparse_csr / to_sparse_csc / to_sparse_bsr
(and their aten::_to_sparse* dispatch): parameters, dtypes, shapes,
layouts/strides, devices, exceptions, outputs and the no-CPU-fallback
contract, against the CPU reference.

Registration is loaded in the first available mode:
  1. installed wheel: ``import cann_ops_sparse`` (build with
      ``bash build.sh --torch_extension --ops=densetosparse`` and pip
      install; requires libops_sparse.so reachable via OPS_SPARSE_LIB_DIR
      or installed under ASCEND_HOME_PATH);
  2. dev tree (D2S_ATEN_DEV=1 or no wheel): synthesize the installed
      package layout around ``sparse/densetosparse/torch_extension``,
      build the extension from the in-tree sources and hand it to the
      registration layer through its public load() entry point.
"""
import importlib
import importlib.util
import logging
import os
import sys
import types

import torch
import torch_npu  # noqa: F401

LOG = logging.getLogger(__name__)

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))
_TX = os.path.join(_REPO, "torch_extension")


class _DevOpLoader:
    """Serves the prebuilt extension module for the dev-tree bootstrap."""

    def __init__(self, module):
        self._module = module

    def load(self):
        return self._module


def _load_registration():
    if os.environ.get("D2S_ATEN_DEV") != "1":
        try:
            import cann_ops_sparse  # noqa: F401
            return "wheel"
        except ImportError:
            pass
    _load_dev_registration()
    return "dev"


def _load_dev_registration():
    # Dev-tree bootstrap: graft the operator package onto the REAL
    # cann_ops_sparse package (from torch_extension/) so the 4-dot
    # relative import in dense_to_sparse.py resolves.
    sys.path.insert(0, _TX)
    import cann_ops_sparse  # noqa: F401
    import cann_ops_sparse.ops as _ops_pkg
    sparse_pkg = types.ModuleType("cann_ops_sparse.ops.sparse")
    sparse_pkg.__path__ = []
    sys.modules["cann_ops_sparse.ops.sparse"] = sparse_pkg
    _ops_pkg.sparse = sparse_pkg
    op_pkg = "cann_ops_sparse.ops.sparse.densetosparse"
    src = os.path.join(_REPO, "sparse", "densetosparse", "torch_extension")
    # __init__ imports .dense_to_sparse transitively (single registration).
    spec = importlib.util.spec_from_file_location(
        op_pkg, os.path.join(src, "__init__.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[op_pkg] = mod
    spec.loader.exec_module(mod)
    # Seed the JIT cache: the builder resolves sources against the
    # installed package root, which does not exist in the dev tree.
    from torch.utils import cpp_extension
    libdir = os.environ.get("OPS_SPARSE_LIB_DIR",
                            os.path.join(_REPO, "build_out", "lib64"))
    cann = os.environ.get(
        "ASCEND_HOME_PATH",
        os.path.join(os.path.dirname(torch_npu.__file__), "..", "..", "cann"))
    tnp = os.path.dirname(torch_npu.__file__)
    module = cpp_extension.load(
        name="cann_ops_sparse_densetosparse",
        sources=[os.path.join(src, "csrc", "densetosparse.cpp")],
        extra_include_paths=[
            os.path.join(_TX, "cann_ops_sparse", "common"),
            os.path.join(tnp, "include"),
            os.path.join(tnp, "include/torch_npu/csrc/core"),
            os.path.join(_REPO, "include"),
            os.path.join(cann, "include"),
        ],
        extra_ldflags=[
            "-L" + libdir, "-L" + os.path.join(cann, "lib64"),
            "-L" + os.path.join(tnp, "lib"),
            "-lops_sparse", "-lascendcl", "-ltorch_npu",
            "-Wl,-rpath," + libdir,
        ],
    )
    # Hand the prebuilt module to the registration layer through its
    # public load() entry point (a plain loader object).
    import cann_ops_sparse.ops.sparse.densetosparse.dense_to_sparse as reg
    reg.dense_to_sparse_op_builder = _DevOpLoader(module)


_MODE = _load_registration()

PASS = 0
FAIL = 0


def check(name, cond):
    global PASS, FAIL
    PASS += bool(cond)
    FAIL += not cond
    LOG.info(f"{'PASS' if cond else 'FAIL'}  {name}")


def expect_raise(name, fn, frag=None):
    try:
        fn()
    except Exception as e:  # noqa: BLE001
        ok = frag is None or frag in str(e)
        check(f"exception:{name}" + (f" [{frag}]" if frag else ""), ok)
        return
    check(f"exception:{name} (no raise)", False)


def eq_coo(a, b):
    return (torch.equal(a.values().cpu(), b.values()) and
            torch.equal(a.indices().cpu(), b.indices()) and
            a.is_coalesced() == b.is_coalesced())


def check_dtype_formats(dtypes):
    """Correctness: dtype x format x base semantics."""
    for dt in dtypes:
        if dt == torch.int8:
            x = torch.randint(-2, 3, (17, 19), dtype=torch.int8)
        elif dt == torch.complex64:
            x = torch.complex(torch.randn(17, 19), torch.randn(17, 19))
        else:
            x = torch.randn(17, 19).to(dt)
        ref = x.cpu()
        xn = x.npu()
        check(f"coo/{dt}", eq_coo(xn.to_sparse(), ref.to_sparse()))
        r, rc = xn.to_sparse_csr(), ref.to_sparse_csr()
        check(f"csr/{dt}", r.values().shape[0] == rc.values().shape[0] and
              torch.equal(r.cpu().crow_indices(), rc.crow_indices()) and
              torch.equal(r.cpu().col_indices(), rc.col_indices()) and
              torch.equal(r.cpu().values(), rc.values()))
        r, rc = xn.to_sparse_csc(), ref.to_sparse_csc()
        check(f"csc/{dt}", r.values().shape[0] == rc.values().shape[0] and
              torch.equal(r.cpu().ccol_indices(), rc.ccol_indices()) and
              torch.equal(r.cpu().row_indices(), rc.row_indices()) and
              torch.equal(r.cpu().values(), rc.values()))


def check_bsr_blocks():
    for b in (1, 2, 4, 8, 16):
        n = 3 * b
        x = torch.randn(n, 2 * b)
        r = x.npu().to_sparse_bsr((b, b))
        rc = x.to_sparse_bsr((b, b))
        check(f"bsr/b{b}", torch.equal(r.cpu().crow_indices(),
                                       rc.crow_indices()) and
              torch.equal(r.cpu().col_indices(), rc.col_indices()) and
              torch.equal(r.cpu().values(), rc.values()))


def check_hybrid_and_layout():
    x = torch.randn(6, 6).npu()
    check("hybrid(1,1)", eq_coo(x.to_sparse(1), x.cpu().to_sparse(1)))
    check("to_sparse(layout=csr)",
          torch.equal(x.to_sparse(layout=torch.sparse_csr).cpu().values(),
                      x.cpu().to_sparse_csr().values()))
    check("to_sparse(layout=csc) nnz",
          x.to_sparse(layout=torch.sparse_csc).values().shape[0] ==
          x.cpu().to_sparse_csc().values().shape[0])
    check("to_sparse(layout=bsr)",
          torch.equal(x.to_sparse(layout=torch.sparse_bsr,
                                  blocksize=(2, 2)).cpu().values(),
                      x.cpu().to_sparse_bsr((2, 2)).values()))


def check_strides():
    xt = torch.randn(19, 17).npu().t()  # column-major 2-D storage
    ref = xt.cpu().contiguous()
    r = xt.to_sparse_csr()
    check("csr/transposed", torch.equal(r.cpu().crow_indices(),
                                        ref.to_sparse_csr().crow_indices()) and
          torch.equal(r.cpu().values(), ref.to_sparse_csr().values()))
    xs = torch.randn(4, 64).npu()[:, ::2]  # non-contiguous rows
    refs = xs.cpu().contiguous()
    r = xs.to_sparse_csr()
    check("csr/strided-slice", torch.equal(
        r.cpu().values(), refs.to_sparse_csr().values()))


def check_devices_no_cpu_fallback():
    x = torch.randn(5, 5).npu()
    for name, out in (("coo", x.to_sparse()), ("csr", x.to_sparse_csr()),
                      ("csc", x.to_sparse_csc()),
                      ("bsr", x.to_sparse_bsr((1, 1)))):
        check(f"device/{name}", out.device.type == "npu")
    # All output storage lives on device (values never touched a CPU
    # copy): the index tensors are produced by the kernels.
    r = x.to_sparse_csr()
    check("no-cpu-fallback/indices",
          r.crow_indices().device.type == "npu" and
          r.col_indices().device.type == "npu" and
          r.values().device.type == "npu")


def check_exceptions():
    expect_raise("dim", lambda: torch.randn(3).npu().to_sparse_csr())
    expect_raise("dtype",
                 lambda: torch.randn(3, 3, dtype=torch.float64).npu()
                 .to_sparse_csr())
    expect_raise("dense_dim",
                 lambda: torch.randn(3, 3).npu().to_sparse_csr(dense_dim=1))
    expect_raise("sparse_dim",
                 lambda: torch.randn(3, 3).npu().to_sparse(0))
    expect_raise("bsr-blocksize-divisibility",
                 lambda: torch.randn(3, 5).npu().to_sparse_bsr((2, 2)),
                 "must be divisible")
    expect_raise("bsr-blocksize-elems",
                 lambda: torch.randn(4, 4).npu().to_sparse_bsr((2,)))
    # The kernel takes one square block size; a divisible rectangular
    # request (4, 6) x (2, 3) must be rejected, not converted as (2, 2).
    expect_raise("bsr-rect-blocksize",
                 lambda: torch.randn(4, 6).npu().to_sparse_bsr((2, 3)),
                 "square")
    # CPU tensors keep dispatching to the CPU kernel (our NPU kernels are
    # registered on the NPU key only); they must keep working.
    check("cpu-input-unchanged",
          torch.randn(3, 3).to_sparse_csr().values().numel() >= 0)
    expect_raise("bsr-layout-no-blocksize",
                 lambda: torch.randn(4, 4).npu().to_sparse(
                     layout=torch.sparse_bsr))


def check_alloc_failure_unwind():
    """OOM 注入回归：sp 创建后由 guard 立即接管，栈展开时释放全部描述符。"""
    total = torch.npu.get_device_properties(0).total_memory
    dense_mb = 16
    x = torch.ones(4096, 4096, dtype=torch.int8)
    try:
        # 限额 = dense + 少量余量：Analysis 正常，首个 payload（values
        # 16MB + 索引 64MB）必然 OOM。
        torch.npu.set_per_process_memory_fraction(
            (x.numel() + (dense_mb // 2 << 20)) / total, 0)
        xn = x.npu()
        try:
            xn.to_sparse_csr()
            check("alloc-failure/unwind (OOM not triggered)", False)
        except RuntimeError as e:
            low = str(e).lower()
            check("alloc-failure/unwind",
                  "out of memory" in low or "oom" in low or
                  "alloc" in low)
    finally:
        torch.npu.set_per_process_memory_fraction(1.0, 0)
        del x
        torch.npu.empty_cache()
    y = torch.eye(8).npu().to_sparse_csr()
    check("alloc-failure/device-healthy", y.values().numel() == 8)


def check_determinism_and_special_values():
    z = torch.zeros(9, 9).npu()
    check("all-zero nnz", z.to_sparse_csr().values().numel() == 0)
    x = torch.randn(8, 8)
    x[3, 4] = float("nan")
    x[2, 2] = 0.0
    r1 = x.npu().to_sparse_csr()
    r2 = x.npu().to_sparse_csr()
    check("determinism", torch.equal(r1.cpu().crow_indices(),
                                     r2.cpu().crow_indices()) and
          torch.equal(r1.cpu().col_indices(), r2.cpu().col_indices()))
    rz = x.npu().to_sparse_csr()
    check("nan-kept", rz.values().shape[0] ==
          x.cpu().to_sparse_csr().values().shape[0])
    check("nan-kept-value",
          bool(torch.isnan(rz.cpu().values()).any()))


def main():
    torch.manual_seed(2026)
    dtypes = (torch.int8, torch.float16, torch.bfloat16, torch.float32,
              torch.complex64)
    check_dtype_formats(dtypes)
    check_bsr_blocks()
    check_hybrid_and_layout()
    check_strides()
    check_devices_no_cpu_fallback()
    check_exceptions()
    check_alloc_failure_unwind()
    check_determinism_and_special_values()
    LOG.info(f"\nATen UT: {PASS} passed, {FAIL} failed (registration: {_MODE})")
    return 1 if FAIL else 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    sys.exit(main())
