# CANN Ops Sparse

`cann_ops_sparse` 是面向 Ascend NPU 的稀疏算子 PyTorch 扩展库。它通过
Just-In-Time（JIT）编译，将 PyTorch 接口接入 `ops-sparse` 的 ACLSparse C API；
扩展不修改 `torch_npu` 包本身，而是在导入时为 wheel 中收集到的算子注册标准 ATen
稀疏算子的后端实现。

框架不依赖任何指定算子的注册实现。下文以 `spmm` 算子为例说明构建、调用和开发流程；
其他算子遵循相同目录约定后，可以独立打包、安装和注册。

## 构建与安装

### 前置条件

- 操作系统：Linux
- Python：3.8+
- 编译器：GCC 9.4.0+
- 框架：
  - PyTorch >= 2.6.0
  - 与 PyTorch 版本匹配的 `torch_npu`
- 工具链：Ascend CANN Toolkit

### 安装步骤

1. 安装依赖。

    ```sh
    cd <ops-sparse 仓根目录>
    python3 -m pip install -r torch_extension/requirements.txt
    ```

2. 构建 wheel。可在仓根使用构建脚本；`--ops` 仅将指定算子的 Torch Extension
   源码打入 wheel，不编译主库或 AscendC kernel。以下以 `spmm` 算子为例；替换为
   其他算子名即可得到不依赖 `spmm` 注册的最小 wheel。

    ```sh
    bash build.sh --torch_extension --ops=spmm
    ```

3. 安装 wheel。

    ```sh
    python3 -m pip install build_out/cann_ops_sparse-*.whl --force-reinstall --no-deps
    ```

## 快速开始

以 `spmm` 算子为例，其注册的标准 ATen 算子为 `aten::_sparse_addmm`。用户使用
`torch.sparse.addmm`，而不是调用 `cann_ops_sparse.spmm`；`torch_npu.sparse`
是由本包在导入时创建的 Python façade。创建前会检查该 namespace 未被其他模块占用；
若已被占用，导入将失败而不会覆盖已有实现。

```python
import torch
import torch_npu
import cann_ops_sparse

out = torch.sparse.addmm(input, csr_mat, dense_mat, beta=1.0, alpha=1.0)
# 等价的 torch_npu façade：
out = torch_npu.sparse.addmm(input, csr_mat, dense_mat, beta=1.0, alpha=1.0)
```

导入 `cann_ops_sparse` 时会为 `aten::_sparse_addmm` 注册 `PrivateUse1` 与
`SparseCsrPrivateUse1` 实现，并注册 `torch_npu.sparse` façade。首次以 NPU CSR
Tensor 调用 `torch.sparse.addmm` 时才会 JIT 编译相应的 C++ wrapper。

以 `spmm` 算子构建的 wheel 包含其 Torch Extension 源码。首次 NPU 调用
`torch.sparse.addmm` 时会触发 JIT 编译并加载 NPU 实现。运行前须先构建包含 `spmm`
的 `libops_sparse.so`。若已通过 run 包安装到 `${ASCEND_HOME_PATH}/lib64`，无需额外
配置；源码构建或安装到其他目录时，以 `OPS_SPARSE_LIB_DIR` 指向其所在目录。

具体接口语义、支持范围、约束和调用示例请参阅
[SpMM 接口说明](cann_ops_sparse/docs/zh/spmm.md)。

## 可复现构建与调用

以下命令以 `spmm` 算子为例，在仓根执行；将 CANN 路径替换为本机实际安装路径。
其他算子只需将两处 `spmm` 替换为目标算子名，并使用其自身的测试文件。

```sh
source /path/to/ascend-toolkit/set_env.sh
# 编译 ACLSparse 主库；Torch Extension wheel 的构建不编译该主库或 kernel。
CMAKE_BUILD_TYPE=Release bash build.sh --ops=spmm --soc=<你的SOC型号>

python3 -m pip install -r torch_extension/requirements.txt
bash build.sh --torch_extension --ops=spmm
python3 -m pip install --force-reinstall --no-deps build_out/cann_ops_sparse-*.whl

# 源码构建的 libops_sparse.so 位于 build/，因此显式指定其路径。
# 若已安装 run 包到 ${ASCEND_HOME_PATH}/lib64，则无需设置该变量。
export OPS_SPARSE_LIB_DIR="$PWD/build"
export TORCH_EXTENSIONS_DIR="$PWD/.torch_extensions"
python3 -m pytest -q test/spmm/test_torch_extension.py
```

以 `spmm` 算子为例，测试用例调用 `torch.sparse.addmm`，并以 CPU 结果作为精度基准；
首次运行会在
`TORCH_EXTENSIONS_DIR` 中生成 `cann_ops_sparse_spmm`。

## 开发者指南：新增算子

一个算子的 Torch Extension 源码归属到 `<category>/<op>/torch_extension/`。以 `spmm`
算子为例，其文件位于 `sparse/spmm/torch_extension/`；新增其他算子时使用自身目录，
不修改或依赖 `spmm` 的注册文件。
`ops-sparse` 当前统一使用 `sparse` category；根目录 `torch_extension` 只负责
通用 builder 与 wheel 打包，并在构建时自动提取各算子的注册源码。

```text
ops-sparse/
├── sparse/
│   └── <op>/                         # 算子原有实现
│       ├── archXX/                   # 原有 host / kernel 代码（不改动）
│       └── torch_extension/          # 算子侧 extension 源码
│           ├── __init__.py           # 导出 Python 接口
│           ├── <op>.py               # ATen dispatch、Meta、PrivateUse1 注册
│           └── csrc/
│               └── <op>.cpp          # at::Tensor 到 ACLSparse C API 的桥接
└── torch_extension/                  # wheel 构建入口
    ├── setup.py
    └── cann_ops_sparse/
        ├── op_builder/
        │   └── builder.py            # 通用 JIT builder
        └── ops/
            └── __init__.py           # 发现 wheel 中已提取的算子
```

构建时，`setup.py` 会将上述文件提取到 wheel 内的
`cann_ops_sparse/ops/<category>/<op>/` 和 `cann_ops_sparse/csrc/<category>/<op>/`。
新增算子时，Python 文件应继承 `OpBuilder`，实现以下内容：

1. `sources()`：声明 `csrc/sparse/<op>/<op>.cpp`；
2. 必要时为既有 ATen schema 注册 Meta 实现；新 ATen 算子不得重复定义 schema；
3. 为 `PrivateUse1` 及稀疏布局对应 dispatch key 注册实现；
4. 实现调用 `builder.load()`，加载 C++ wrapper 并执行底层 ACLSparse API。
5. 如需提供 `torch_npu.sparse` 等价入口，在算子注册模块中声明
   `TORCH_NPU_SPARSE_FACADE_APIS = {"接口名": public_pytorch_api}`；包导入时会自动收集
   并安装，不要修改 `torch_npu_sparse.py`。

| 组件 | 职责 |
| --- | --- |
| `OpBuilder` | 使用 JIT/ninja 编译和加载 C++ wrapper。 |
| ATen Dispatch | 将 `torch.sparse.addmm` 路由到 NPU 的 `aten::_sparse_addmm` 实现。 |
| PrivateUse1 | PyTorch 将 NPU Tensor 分发到自定义后端的 dispatch key。 |
