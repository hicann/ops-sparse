# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.

"""
Low-level ctypes binding to the CANN ``ops-sparse`` C API (``libops_sparse.so``)
plus a minimal ACL runtime bootstrap, so the operator can be driven from Python.

This module intentionally depends only on ``ctypes`` + the CANN shared libraries
(``libops_sparse.so`` and ``libascendcl.so``); it does NOT require torch at import
time. The torch / ATen facing layer lives in :mod:`ops_sparse.gather`.

The enum integer values below are the CANN ``aclDataType`` / ``aclsparse*`` values
(verified against ``include/cann_ops_sparse.h``); keep them in sync with the CANN
version used to build ``libops_sparse.so``.
"""

__all__ = [
    "LibOpsSparseError",
    "load_library",
    "gather",
    "GatherRawArgs",
    "AclDevice",
]

import ctypes  # noqa: E402
import os  # noqa: E402
from ctypes import c_int32, c_int64, c_void_p, POINTER  # noqa: E402
from dataclasses import dataclass  # noqa: E402


class LibOpsSparseError(RuntimeError):
    """Raised when an aclsparse* call returns a non-SUCCESS status."""


# ---------------------------------------------------------------------------
# CANN enum values (mirror include/cann_ops_sparse.h)
# ---------------------------------------------------------------------------
ACL_SPARSE_STATUS_SUCCESS = 0

ACL_SPARSE_INDEX_32I = 0
ACL_SPARSE_INDEX_64I = 1
ACL_SPARSE_INDEX_BASE_ZERO = 0
ACL_SPARSE_INDEX_BASE_ONE = 1

# aclDataType (CANN 9.1.0 / acl_base_rt.h values -- keep in sync!)
ACL_FLOAT = 0
ACL_FLOAT16 = 1
ACL_INT8 = 2
ACL_INT32 = 3
ACL_UINT8 = 4
ACL_INT16 = 6
ACL_UINT16 = 7
ACL_UINT32 = 8
ACL_INT64 = 9
ACL_UINT64 = 10
ACL_DOUBLE = 11
ACL_BOOL = 12
ACL_STRING = 13
ACL_COMPLEX64 = 16
ACL_COMPLEX128 = 17
ACL_BF16 = 27
ACL_COMPLEX32 = 33

# torch dtype -> aclDataType
_TORCH_DTYPE_TO_ACL = {
    "float32": ACL_FLOAT,
    "float16": ACL_FLOAT16,
    "bfloat16": ACL_BF16,
    "complex64": ACL_COMPLEX64,
}
# aclDataType -> number of bytes per element
_ACL_ELEM_BYTES = {
    ACL_FLOAT: 4,
    ACL_FLOAT16: 2,
    ACL_BF16: 2,
    ACL_COMPLEX64: 8,
}


def torch_dtype_to_acl(torch_dtype) -> int:
    """Map a torch dtype to a CANN aclDataType integer.

    torch >= 2.x removed the ``.name`` attribute on ``torch.dtype``; ``str()``
    now yields ``'torch.float32'``. Accept both forms by stripping the
    ``torch.`` prefix and keeping the bare type name.
    """
    name = getattr(torch_dtype, "name", None)
    if name is None:
        name = str(torch_dtype)
    name = name.split(".")[-1]
    if name not in _TORCH_DTYPE_TO_ACL:
        raise ValueError(f"aclsparseGather: unsupported torch dtype {torch_dtype} "
                         f"(arch22 supports float32/float16/bfloat16/complex64)")
    return _TORCH_DTYPE_TO_ACL[name]


# ---------------------------------------------------------------------------
# Library loading
# ---------------------------------------------------------------------------
_LIB = None


def load_library(lib_path: str = None):
    """Load ``libops_sparse.so`` (and CANN ascendcl as a side dependency).

    ``lib_path`` may point directly at ``libops_sparse.so``; if omitted we search
    the standard CANN library locations and ``LD_LIBRARY_PATH``.
    """
    global _LIB
    if _LIB is not None:
        return _LIB

    candidates = []
    if lib_path:
        candidates.append(lib_path)
    env = os.environ.get("OPS_SPARSE_LIB", "")
    if env:
        candidates.append(env)
    candidates += ["libops_sparse.so", "libops_sparse.so.1"]

    last_err = None
    for cand in candidates:
        try:
            _LIB = ctypes.CDLL(cand, mode=ctypes.RTLD_GLOBAL)
            break
        except OSError as e:  # noqa: PERF203
            last_err = e
    if _LIB is None:
        raise FileNotFoundError(
            f"Could not load libops_sparse.so (tried {candidates}); "
            f"set OPS_SPARSE_LIB or LIBRARY_PATH. Last error: {last_err}")

    _register_prototypes(_LIB)
    return _LIB


def _register_prototypes(lib):
    lib.aclsparseCreate.restype = c_int32
    lib.aclsparseCreate.argtypes = [POINTER(c_void_p)]

    lib.aclsparseDestroy.restype = c_int32
    lib.aclsparseDestroy.argtypes = [c_void_p]

    lib.aclsparseSetStream.restype = c_int32
    lib.aclsparseSetStream.argtypes = [c_void_p, c_void_p]

    lib.aclsparseCreateDnVec.restype = c_int32
    lib.aclsparseCreateDnVec.argtypes = [POINTER(c_void_p), c_int64, c_void_p, c_int32]

    lib.aclsparseCreateConstDnVec.restype = c_int32
    lib.aclsparseCreateConstDnVec.argtypes = [POINTER(c_void_p), c_int64, c_void_p, c_int32]

    lib.aclsparseCreateSpVec.restype = c_int32
    lib.aclsparseCreateSpVec.argtypes = [
        POINTER(c_void_p), c_int64, c_int64, c_void_p, c_void_p, c_int32, c_int32, c_int32,
    ]

    lib.aclsparseDestroyDnVec.restype = c_int32
    lib.aclsparseDestroyDnVec.argtypes = [c_void_p]

    lib.aclsparseDestroySpVec.restype = c_int32
    lib.aclsparseDestroySpVec.argtypes = [c_void_p]

    lib.aclsparseGather.restype = c_int32
    lib.aclsparseGather.argtypes = [c_void_p, c_void_p, c_void_p]

    lib.aclsparseGatherRaw.restype = c_int32
    lib.aclsparseGatherRaw.argtypes = [
        c_void_p, c_void_p, c_int64, c_int32, c_void_p, c_void_p, c_int64, c_int32,
    ]


def _check(status: int, what: str):
    if status != ACL_SPARSE_STATUS_SUCCESS:
        raise LibOpsSparseError(f"aclsparse {what} failed with status {status}")


# ---------------------------------------------------------------------------
# Handle cache (reuse aclsparseHandle per device instead of create/destroy
# on every call). The official e2e harness times the whole Python/ATen path,
# and aclsparseCreate/Destroy churn dominates (~hundreds of us) for a us-scale
# kernel -- violating the task spec's "复用描述符" requirement. Caching the
# handle (and re-issuing SetStream per call) removes that overhead.
# ---------------------------------------------------------------------------
_handle_cache: dict = {}
_TORCH_NPU = None


def _get_torch_npu():
    """Lazily detect torch_npu (may be unavailable in standalone tests)."""
    global _TORCH_NPU
    if _TORCH_NPU is None:
        try:
            import torch  # noqa: F401
            import torch_npu  # noqa: F401
            import torch as _t
            _TORCH_NPU = _t
        except Exception:  # noqa: BLE001
            _TORCH_NPU = False
    return _TORCH_NPU


def _current_device_id() -> int:
    torch_mod = _get_torch_npu()
    if torch_mod:
        try:
            return int(torch_mod.npu.current_device())
        except Exception:  # noqa: BLE001
            pass
    return 0


def _get_or_create_handle(device_id: int) -> c_void_p:
    global _handle_cache
    h = _handle_cache.get(device_id)
    if h is not None and h.value is not None:
        return h
    lib = load_library()
    handle = c_void_p()
    _check(lib.aclsparseCreate(ctypes.byref(handle)), "Create")
    _handle_cache[device_id] = handle
    return handle


def reset_handles():
    """Destroy any cached handles (call at process exit if desired)."""
    global _handle_cache
    lib = load_library()
    for h in list(_handle_cache.values()):
        if h.value is not None:
            try:
                lib.aclsparseDestroy(h)
            except Exception:  # noqa: BLE001
                pass
    _handle_cache.clear()


# ---------------------------------------------------------------------------
# Core gather call (device pointers already live on NPU)
# ---------------------------------------------------------------------------
@dataclass
class GatherRawArgs:
    """Named argument pack for one raw ``aclsparseGatherRaw`` invocation.

    All pointers are NPU device addresses (int). Layout:
    ``output[i] = Y[idx[i] - base]`` for ``i in [0, nnz)``.
    """

    output_ptr: int   # device pointer to output buffer (nnz * elemBytes), writable
    y_ptr: int        # device pointer to dense Y (y_len * elemBytes), read-only
    y_len: int        # number of elements in Y
    value_acl: int    # aclDataType of Y / output values
    idx_ptr: int      # device pointer to int32 indices (nnz elements)
    nnz: int          # number of output elements
    base: int         # ACL_SPARSE_INDEX_BASE_ZERO (0) or _ONE (1)
    stream_ptr: int   # aclrtStream (int) or 0 for the default stream


def gather(args: GatherRawArgs):
    """Invoke ``aclsparseGatherRaw`` directly on NPU device pointers (no descriptor churn).

    Uses the raw pointer fast-path (``aclsparseGatherRaw``) which skips the DnVec /
    SpVec descriptor create/destroy that ``aclsparseGather`` does per call -- for a
    us-scale kernel the descriptor churn would otherwise dominate the Python/ATen
    e2e time and fail the task spec's "复用描述符" / performance bar (§3.3 / §3.5).
    """
    lib = load_library()
    handle = _get_or_create_handle(_current_device_id())
    if args.stream_ptr:
        _check(lib.aclsparseSetStream(handle, c_void_p(args.stream_ptr)), "SetStream")

    _check(lib.aclsparseGatherRaw(
        handle,
        c_void_p(args.y_ptr), c_int64(args.y_len), c_int32(args.value_acl),
        c_void_p(args.idx_ptr), c_void_p(args.output_ptr), c_int64(args.nnz),
        c_int32(args.base)),
        "GatherRaw")


# ---------------------------------------------------------------------------
# Minimal ACL runtime bootstrap (for standalone e2e tests without torch_npu)
# ---------------------------------------------------------------------------
class AclDevice:
    """Context manager that initialises an ACL device + stream.

    Usage::

        with AclDevice(device_id=0) as dev:
            ...  # dev.stream is the aclrtStream pointer
    """

    def __init__(self, device_id: int = 0, ascendcl_lib: str = "libascendcl.so"):
        self.device_id = device_id
        self._acl = ctypes.CDLL(ascendcl_lib, mode=ctypes.RTLD_GLOBAL)
        self._acl.aclrtCreateStream.restype = c_int32
        self._acl.aclrtCreateStream.argtypes = [POINTER(c_void_p)]
        self._acl.aclrtDestroyStream.restype = c_int32
        self._acl.aclrtDestroyStream.argtypes = [c_void_p]
        self._acl.aclrtSetDevice.restype = c_int32
        self._acl.aclrtSetDevice.argtypes = [c_int32]
        self._acl.aclrtResetDevice.restype = c_int32
        self._acl.aclrtResetDevice.argtypes = [c_int32]
        self._acl.aclInit.restype = c_int32
        self._acl.aclInit.argtypes = [c_void_p]
        self.stream = None

    def __enter__(self):
        if self._acl.aclInit(None) != 0:
            raise LibOpsSparseError("aclInit failed")
        if self._acl.aclrtSetDevice(self.device_id) != 0:
            raise LibOpsSparseError("aclrtSetDevice failed")
        stream = c_void_p()
        if self._acl.aclrtCreateStream(ctypes.byref(stream)) != 0:
            raise LibOpsSparseError("aclrtCreateStream failed")
        self.stream = int(stream.value)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.stream is not None:
            self._acl.aclrtDestroyStream(c_void_p(self.stream))
            self.stream = None
        self._acl.aclrtResetDevice(self.device_id)
        return False
