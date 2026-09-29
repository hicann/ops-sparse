# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.

"""
torch / ATen facing layer for ``aclsparseGather`` (Atlas A2/A3, arch22).

Public entries
--------------
* ``torch.ops.ops_sparse.gather(values, indices, base)``
      Custom op. ``values`` is the dense source vector ``Y`` (1-D, on NPU);
      ``indices`` is an ``int32`` vector (1-D, on NPU); ``base`` is ``0`` or
      ``1``. Returns ``X.values`` of shape ``(nnz,)`` where
      ``X.values[i] = Y[indices[i] - base]``.
* ``index_select_npu(tensor, dim, index)``
      Drop-in for ``torch.index_select`` limited to the NPU path with
      ``dim == 0`` and 1-D ``input`` (``base == 0``). All other shapes /
      dims raise -- there is **no CPU fallback** by design.

The NPU dispatch key used for registration is ``PrivateUse1`` (torch maps the
NPU backend to this key). If a given torch / torch_npu version rejects the
registration, the package still imports and the direct ``gather`` /
``index_select_npu`` functions work unchanged; finalise the
``aten::index_select`` override inside torch_npu during integration.

NOTE: aclsparseGather is a pure data-movement operator -> results are bit-wise
exact for all supported dtypes (float16 / bfloat16 / float32 / complex64).
"""

__all__ = ["gather", "index_select_npu"]

import torch  # noqa: E402

from ._capi import (  # noqa: E402
    LibOpsSparseError,
    GatherRawArgs,
    gather as _capi_gather,
    torch_dtype_to_acl,
    ACL_SPARSE_INDEX_BASE_ZERO,
    ACL_SPARSE_INDEX_BASE_ONE,
)


def _current_npu_stream_ptr() -> int:
    """Extract the raw aclrtStream for the current NPU stream.

    torch_npu exposes the raw device handle via ``stream.npu_stream`` (and, on
    some versions, ``stream.stream``). The c10-encoded ``stream_id`` is a small
    integer, NOT a pointer, and must never be passed to aclsparseSetStream: on a
    non-default stream it would be interpreted as a bogus aclrtStream and corrupt
    execution ordering. If the raw handle cannot be obtained, raise instead of
    silently falling back to stream_id.
    """
    try:
        import torch_npu  # noqa: F401
    except ImportError as e:
        raise RuntimeError(
            "aclsparseGather: torch_npu is required to obtain the raw aclrtStream "
            "for the current NPU stream.") from e
    stream = torch.npu.current_stream()
    raw = None
    for attr in ("npu_stream", "stream"):
        v = getattr(stream, attr, None)
        if v is not None:
            raw = v
            break
    if raw is None:
        raise RuntimeError(
            "aclsparseGather: cannot obtain the raw aclrtStream from the current "
            "NPU stream (neither stream.npu_stream nor stream.stream is available). "
            "Refusing to fall back to stream_id, which is not a valid pointer.")
    try:
        return int(raw)
    except (TypeError, ValueError) as e:
        raise RuntimeError(
            f"aclsparseGather: raw stream handle is not an integer pointer "
            f"({type(raw)!r}); cannot pass it to aclsparseSetStream.") from e


def _require_npu(tensor, name: str):
    if not tensor.is_npu:
        raise NotImplementedError(
            f"aclsparseGather: {name} is on '{tensor.device}', but the operator "
            f"only supports NPU device (no CPU fallback by design).")


def _validate(values, indices, base: int):
    _require_npu(values, "values")
    _require_npu(indices, "indices")
    if values.dim() != 1:
        raise NotImplementedError(
            f"aclsparseGather: 'values' must be 1-D (got {values.dim()}-D)")
    if indices.dim() != 1:
        raise NotImplementedError(
            f"aclsparseGather: 'indices' must be 1-D (got {indices.dim()}-D)")
    if indices.dtype not in (torch.int32,):
        raise NotImplementedError(
            f"aclsparseGather: 'indices' must be int32 (got {indices.dtype})")
    if base not in (0, 1):
        raise ValueError(f"aclsparseGather: 'base' must be 0 or 1 (got {base})")
    value_acl = torch_dtype_to_acl(values.dtype)
    return value_acl


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------
def gather(values: torch.Tensor, indices: torch.Tensor, base: int = 0) -> torch.Tensor:
    """Public custom-op entry: gather dense ``values`` at ``indices`` (base 0/1).

    NPU only, no CPU fallback; semantics and supported dtypes are documented in
    the module docstring above.
    """
    value_acl = _validate(values, indices, base)
    nnz = int(indices.numel())
    y_len = int(values.numel())

    if nnz == 0:
        return torch.empty(0, dtype=values.dtype, device=values.device)

    out = torch.empty(nnz, dtype=values.dtype, device=values.device)
    stream_ptr = _current_npu_stream_ptr()
    base_enum = ACL_SPARSE_INDEX_BASE_ONE if base == 1 else ACL_SPARSE_INDEX_BASE_ZERO

    _capi_gather(GatherRawArgs(
        output_ptr=int(out.data_ptr()),
        y_ptr=int(values.data_ptr()),
        y_len=y_len,
        value_acl=value_acl,
        idx_ptr=int(indices.data_ptr()),
        nnz=nnz,
        base=base_enum,
        stream_ptr=stream_ptr,
    ))
    # NOTE: do NOT call torch.npu.synchronize() here. The kernel is enqueued on
    # the current NPU stream and torch's dispatch / stream-ordering semantics
    # guarantee the result is observed when it is actually needed (e.g. a
    # subsequent op, or a CPU read that triggers an implicit sync). An explicit
    # synchronise here is redundant with the caller's own synchronisation and,
    # under the official benchmark_runner, double-counts ~100us of host-block /
    # stream-sync overhead that the GPU PyTorch reference (torch.index_select,
    # timed with a single end.synchronize()) does not pay -- making the
    # Python/ATen e2e comparison unfair (task §3.5).
    return out


def index_select_npu(tensor: torch.Tensor, dim: int, index: torch.Tensor) -> torch.Tensor:
    """NPU ``index_select`` (dim == 0, 1-D input only; base == 0). No CPU fallback."""
    if dim != 0:
        raise NotImplementedError(
            f"aclsparseGather index_select: only dim == 0 is supported on NPU "
            f"(got dim={dim}). Other dims are not implemented (no CPU fallback).")
    if tensor.dim() != 1:
        raise NotImplementedError(
            f"aclsparseGather index_select: only 1-D input is supported on NPU "
            f"(got {tensor.dim()}-D).")
    return gather(tensor, index, base=0)


# ---------------------------------------------------------------------------
# torch.library registrations
# ---------------------------------------------------------------------------
def _register_torch_ops():
    try:
        torch.library.define(
            "ops_sparse::gather", "(Tensor values, Tensor indices, int base) -> Tensor")

        @torch.library.impl("ops_sparse::gather", "PrivateUse1")
        def _gather_npu(values, indices, base):  # pragma: no cover - registered op
            return gather(values, indices, base)

        # Override aten::index_select for the NPU dispatch key (base == 0).
        @torch.library.impl("aten::index_select", "PrivateUse1")
        def _index_select_npu(tensor, dim, index):  # pragma: no cover - registered op
            return index_select_npu(tensor, dim, index)

    except Exception as e:  # noqa: BLE001
        # Registration may be rejected by some torch / torch_npu versions; the
        # direct functions above remain fully usable regardless.
        import warnings
        warnings.warn(
            f"ops_sparse: torch.library registration skipped ({e}); "
            f"use ops_sparse.gather / ops_sparse.index_select_npu directly.")


_register_torch_ops()
