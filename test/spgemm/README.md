# SpGEMM测试

## 测试说明

SpGEMM测试覆盖`aclsparseSpGEMM*`多阶段接口、支持的数据类型与算法、CSR输出结构、
状态机和异常参数，并通过PyTorch公开接口验证NPU端到端调用。算子接口、支持规格和调用
流程见[SpGEMM算子说明](../../sparse/spgemm/README.md)。

## 目录结构

```text
test/spgemm/
├── CMakeLists.txt
├── README.md
├── spgemm_golden.h
├── spgemm_param.h
├── arch35/
│   ├── spgemm_npu_wrapper.h
│   ├── spgemm_test.cpp       # C++功能与接口测试
│   ├── spgemm_test.csv       # 仓库原有参数化回归用例
│   └── spgemm_perf.cpp       # 独立性能测试程序
└── test_torch_extension.py   # PyTorch公开接口端到端测试
```

## C++测试

在已安装CANN开发环境的Ascend 950PR设备上执行：

```bash
source /usr/local/Ascend/cann/set_env.sh

cmake -S . -B build \
  -DSOC_VERSION=ascend950 \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_TEST=ON \
  -DOP_LIST=spgemm
cmake --build build --target spgemm_test spgemm_perf --parallel

./build/test/spgemm/spgemm_test --gtest_color=no
```

`spgemm_test`覆盖FP16、BF16、FP32、Complex64，DEFAULT、ALG1、ALG2、ALG3，
非零beta与预置C累加，以及workspace不足、重复执行和非法阶段调用等场景。

性能程序可通过命令行指定矩阵规模、每行非零元数量、数据类型、预热次数和采样次数。例如：

```bash
./build/test/spgemm/spgemm_perf \
  --n 19717 --d 4 --dtype float32 --warmup 10 --samples 30
```

## PyTorch端到端测试

使用相互匹配的 PyTorch 和 torch_npu 环境。先构建 `spgemm` 主库，再将该算子的
Torch Extension 源码打包、安装：

```bash
CMAKE_BUILD_TYPE=Release bash build.sh --ops=spgemm --soc=ascend950
python3 -m pip install -r torch_extension/requirements.txt
bash build.sh --torch_extension --ops=spgemm
python3 -m pip install --force-reinstall --no-deps build_out/cann_ops_sparse-*.whl

# 源码构建时 libops_sparse.so 位于 build/；安装 run 包后可省略该变量。
export OPS_SPARSE_LIB_DIR="$PWD/build"
python3 -m pytest -q test/spgemm/test_torch_extension.py
```

该测试覆盖CSR/COO公开路径、int32/int64索引转换以及FP16、BF16、FP32、Complex64。
