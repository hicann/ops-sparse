# Torch Extension 接口列表

`cann_ops_sparse` 包通过 `torch.library` 将 ops-sparse 算子注册到 torch_npu 后端。
导入 `cann_ops_sparse` 即完成注册，无需显式调用私有接口；C++ wrapper 在首次 NPU
调用时由 PyTorch JIT 编译。

## ATen 适配

| public PyTorch API | ATen schema | 分发键 | 算子 | 文档 |
| :--- | :--- | :--- | :--- | :--- |
| `torch.sparse.mm` / `torch.sparse.addmm` | `aten::_sparse_addmm` | `SparseCsrPrivateUse1` | spgemm | [spgemm.md](spgemm.md) |
| `torch.sparse.mm` | `aten::_sparse_sparse_matmul` | `SparseCsrPrivateUse1` | spgemm | [spgemm.md](spgemm.md) |
| `torch.sparse.mm` | `aten::_sparse_sparse_matmul` | `SparsePrivateUse1` | spgemm | [spgemm.md](spgemm.md) |

## `torch_npu.sparse` façade

façade 仅转发同名 public PyTorch API，不绕过 ATen 分发器，语义与直接调用完全一致。

| façade API | 转发目标 | 声明算子 |
| :--- | :--- | :--- |
| `torch_npu.sparse.mm` | `torch.sparse.mm` | spgemm |
| `torch_npu.sparse.addmm` | `torch.sparse.addmm` | spgemm |

## custom API

当前无自定义（非 ATen）Python 接口。
