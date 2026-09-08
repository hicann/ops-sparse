# PyTorch适配层

本可选模块用于集中注册ops-sparse算子的PyTorch Ascend NPU实现，后续可按需扩展其他
稀疏算子。当前已接入SpGEMM：注册`aten::_sparse_sparse_matmul`的Ascend NPU实现；
PyTorch 2.7会将公开的CSR×CSR `torch.sparse.mm`调用下沉到`aten::_sparse_addmm`，因此
同时注册该CSR分发路径。两条路径均调用公开的多阶段`aclsparseSpGEMM*`接口，并在调用方的
NPU stream上执行核心计算。

使用安装了相互匹配的PyTorch和torch_npu的Python解释器配置主工程：

```bash
cmake -S . -B build \
  -DSOC_VERSION=ascend950 \
  -DBUILD_TORCH_ADAPTER=ON \
  -DPython3_EXECUTABLE=/path/to/python
cmake --build build --target ops_sparse_torch --parallel
```

调用公开接口前加载注册库：

```bash
export PYTHONPATH=$PWD/python:$PYTHONPATH
export OPS_SPARSE_TORCH_LIBRARY=$PWD/build/python/ops_sparse_torch/libops_sparse_torch.so
python -c 'import ops_sparse_torch; import torch; print(torch.sparse.mm)'
```

同时开启`BUILD_TEST`并选择`spgemm`时，可直接运行端到端回归目标：

```bash
cmake --build build --target spgemm_torch_test
```

安装工程时，动态库和`__init__.py`会安装到`python/ops_sparse_torch`。适配层支持FP16、
BF16、FP32和Complex64。aclsparse接口使用零基int32 CSR索引；PyTorch的int64 CSR索引
在NPU上转换。已合并的COO输入在NPU上转换为CSR，结果以已合并的COO张量返回；未合并的
COO输入会被明确拒绝。
