# SPMV 算子实现

## 概述

ops-sparse 仓库中的 SPMV (Sparse Matrix-Vector Multiplication) 算子实现了稀疏矩阵与向量的乘法运算，是高性能稀疏矩阵计算中的核心算子之一。

该算子针对稀疏矩阵的存储特性进行了优化，采用 CSR (Compressed Sparse Row) 格式存储稀疏矩阵，并通过分块和并行计算提高计算效率。

## 支持的产品

- Atlas A3 训练系列产品/Atlas A3 推理系列产品
- Atlas A2 训练系列产品/Atlas A2 推理系列产品

## 目录结构介绍

```txt
├── spmv
│   ├── CMakeLists.txt      // 编译工程文件
│   ├── README.md           // 说明文档
│   └── spmv_test.cpp       // 算子调用样例
```

## 算子描述

### 功能

SPMV 算子实现了将稀疏矩阵乘以向量的运算。对应的数学表达式为：

$$
z = alpha * A * x + beta * y
$$

其中，A 为稀疏矩阵，x 和 y 是向量，alpha 和 beta 是标量。

### 存储格式

本实现采用 CSR (Compressed Sparse Row) 格式存储稀疏矩阵，该格式由三个数组组成：

- `ptrs`：行偏移数组，存储每行第一个非零元素在 `values` 和 `idxs` 数组中的位置
- `idxs`：列索引数组，存储非零元素的列索引
- `values`：值数组，存储非零元素的值

### 实现原理

1. **矩阵预处理**：在 `spmv_csr_mat.cpp` 中，对输入的 CSR 矩阵进行分块处理，将大矩阵划分为多个子矩阵
2. **数据重排**：对每个子矩阵进行数据重排，优化内存访问模式
3. **并行计算**：在 `spmv_kernel.cpp` 中，使用 AscendC 编程模型实现并行计算
4. **结果验证**：在 `spmv_test.cpp` 中，通过 CPU 计算golden真值，验证 NPU 计算结果的正确性

### 算子规格
- 参数说明：
  <table>
  <tr><td rowspan="1" align="center">算子类型(OpType)</td><td colspan="6" align="center">Spmv</td></tr>
  <tr>
  <tr><td rowspan="6" align="center">算子输入</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
  <tr><td align="center">A</td><td align="center">CSR格式 (ptrs, idxs, values)</td><td align="center">float</td><td align="center">CSR</td></tr>
  <tr><td align="center">x</td><td align="center">cols</td><td align="center">float</td><td align="center">ND</td></tr>
  <tr><td align="center">y</td><td align="center">rows</td><td align="center">float</td><td align="center">ND</td></tr>
  <tr><td align="center">alpha</td><td align="center">1</td><td align="center">float</td><td align="center">scalar</td></tr>
  <tr><td align="center">beta</td><td align="center">1</td><td align="center">float</td><td align="center">scalar</td></tr>
  </tr>
  <tr><td rowspan="1" align="center">算子输出</td><td align="center">z</td><td align="center">rows</td><td align="center">float</td><td align="center">ND</td></tr>
  <tr><td rowspan="1" align="center">核函数名</td><td colspan="6" align="center">spmv_custom</td></tr>
  </table>

- 约束限制：
  - 当前实现仅支持alpha=1, beta=0
  - 输入输出仅支持float类型

### 测试实现

- 测试流程 (`spmv_test.cpp`)

1. **初始化**：初始化 ACL 环境，设置设备和创建流
2. **生成测试数据**：在线生成随机 CSR 矩阵和向量
3. **CPU 参考计算**：使用 `spmv_csr_cpu` 函数计算参考结果
4. **设备内存管理**：分配和拷贝数据到设备内存
5. **稀疏矩阵操作**：使用 ACL Sparse API 创建矩阵和向量描述符
6. **执行 SPMV**：调用 `aclSparseSpmv` 函数执行稀疏矩阵向量乘法
7. **结果验证**：将设备计算结果拷贝回主机，与 CPU 参考结果进行比较
8. **清理资源**：释放设备和主机内存，销毁描述符

- 关键代码片段

```cpp
// 生成随机 CSR 矩阵
generate_random_csr(A_num_rows, A_num_cols, A_nnz, hA_csrOffsets, hA_columns, hA_values);

// 计算 CPU 参考结果
spmv_csr_cpu(A_num_rows, A_num_cols, A_nnz, hA_csrOffsets, hA_columns, hA_values, hX, hYBase);

// 执行 SpMV
CHECK_ACL_SPARSE(aclSparseSpmv(handle,
    ACL_SPARSE_OP_NON_TRANSPOSE,
    &alpha,
    matA,
    vecX,
    &beta,
    vecY,
    ACL_FLOAT,
    ACL_SPARSE_SPMV_ALG_DEFAULT,
    dBuffer))
```

## 编译运行

在 ops-sparse 仓库根目录下执行如下步骤，编译并执行 SPMV 算子测试。

### 配置环境变量

请根据当前环境上 CANN 开发套件包的安装方式，选择对应配置环境变量的命令。

- 默认路径，root 用户安装 CANN 软件包

  ```bash
  source /usr/local/Ascend/cann/set_env.sh
  ```

- 默认路径，非 root 用户安装 CANN 软件包

  ```bash
  source $HOME/Ascend/cann/set_env.sh
  ```

- 指定路径 install_path，安装 CANN 软件包

  ```bash
  source ${install_path}/cann/set_env.sh
  ```

### 样例执行

```bash
bash build.sh --ops=spmv --run # --ops=<算子名> --run 可选参数，执行测试样例
```

执行结果如下，说明精度对比成功：

```bash
[Success] Case accuracy is verification passed.
```

## 接口说明

### aclSparseSpmv

**函数原型**：

```cpp
AclSparseStatus aclSparseSpmv(
    AclSparseHandler handle,
    AclSparseOperation_t transA,
    const void *alpha,
    AclSparseSpMatDesc matA,
    AclSparseDnVecDesc vecX,
    const void *beta,
    AclSparseDnVecDesc vecY,
    AclSparseDataType_t computeType,
    AclSparseSpmvAlg_t alg,
    void *workspace)
```

**参数说明**：

- `handle`：稀疏矩阵操作的句柄
- `transA`：矩阵 A 的转置选项
- `alpha`：标量 alpha
- `matA`：稀疏矩阵 A 的描述符
- `vecX`：输入向量 X 的描述符
- `beta`：标量 beta
- `vecY`：输出向量 Y 的描述符
- `computeType`：计算数据类型
- `alg`：SPMV 算法选择
- `workspace`：工作空间

**返回值**：

- `ACL_SPARSE_STATUS_SUCCESS`：成功
- 其他值：失败
