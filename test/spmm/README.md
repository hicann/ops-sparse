# SPMM 算子实现

## 概述

ops-sparse 仓库中的 SPMM (Sparse Matrix-Dense Matrix Multiplication) 算子实现了稀疏矩阵与稠密矩阵的乘法运算，是高性能稀疏矩阵计算中的核心算子之一。

该算子针对 CSR 格式稀疏矩阵的存储特性进行了优化，采用 SIMT/AIV 并行计算路径，并在 preprocess 阶段完成行重排与分桶（row reorder / binning），以提升后续计算效率。

## 产品支持情况

| 产品                                                         |  是否支持 |
| :----------------------------------------------------------- |:-------:|
| <term>Ascend 950PR/Ascend 950DT</term>                       |    ✓    |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>       |    ✗    |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>       |    ✗    |

> SPMM 当前版本在 Ascend 950PR/950DT 平台交付，源码位于 `src/spmm/arch35/`，与 SOC 架构映射保持一致。非 `ascend950*` 平台编译时会跳过 `spmm_test`。

## 目录结构介绍

```txt
src/spmm/arch35/
├── spmm_host.cpp       // Host 侧 API 实现与 launch 调度
├── spmm_csr_mat.cpp    // CSR 矩阵预处理（行重排、分桶）
├── spmm_kernel.cpp     // Kernel 侧 SIMT 计算实现
└── spmm.h              // 内部头文件与 Tiling 定义

test/spmm/
├── CMakeLists.txt      // 调用 ops_sparse_add_test(spmm ${OPS_SPARSE})
├── README.md           // 说明文档
└── arch35/
    └── spmm_test.cpp   // 950 算子调用样例
```

## 算子描述

### 功能

SPMM 算子实现了将稀疏矩阵乘以稠密矩阵的运算。对应的数学表达式为：

$$
C = \alpha \cdot op(A) \cdot op(B) + \beta \cdot C
$$

其中，A 为 CSR 格式稀疏矩阵，B 和 C 为稠密矩阵，$\alpha$ 和 $\beta$ 是标量。

### 存储格式

- **稀疏矩阵 A**：采用 CSR (Compressed Sparse Row) 格式，由三个数组组成：
  - `csrRowOffsets`：行偏移数组
  - `csrColInd`：列索引数组
  - `csrValues`：非零元素值数组
- **稠密矩阵 B / C**：支持行主序（`ACL_SPARSE_ORDER_ROW`）与列主序（`ACL_SPARSE_ORDER_COL`），通过 `ld`（leading dimension）描述内存布局。

### 实现原理

1. **参数校验与 Tiling 配置**：在 `spmm_host.cpp` 中校验矩阵维度、数据类型与算法组合，并写入 Kernel 所需的 Tiling 信息。
2. **矩阵预处理**：在 `spmm_csr_mat.cpp` 中对 CSR 矩阵进行行重排与分桶，结果写入 workspace。
3. **并行计算**：在 `spmm_kernel.cpp` 中，基于 SIMT/AIV 路径实现稀疏-稠密矩阵乘。
4. **结果验证**：在 `spmm_test.cpp` 中，通过 CPU 参考实现计算 golden 真值，验证 NPU 计算结果的正确性。

### 算子规格

- 参数说明：

  <table>
  <tr><td rowspan="1" align="center">算子类型(OpType)</td><td colspan="6" align="center">Spmm</td></tr>
  <tr><td rowspan="7" align="center">算子输入</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
  <tr><td align="center">A</td><td align="center">CSR 格式 (rows × cols)</td><td align="center">fp32 / fp16 / int8</td><td align="center">CSR</td></tr>
  <tr><td align="center">B</td><td align="center">cols × colsOut</td><td align="center">fp32 / fp16 / int8</td><td align="center">行主序 / 列主序</td></tr>
  <tr><td align="center">C</td><td align="center">rows × colsOut</td><td align="center">fp32 / fp16 / int32</td><td align="center">行主序 / 列主序</td></tr>
  <tr><td align="center">alpha</td><td align="center">1</td><td align="center">fp32 / int32</td><td align="center">scalar</td></tr>
  <tr><td align="center">beta</td><td align="center">1</td><td align="center">fp32 / int32</td><td align="center">scalar</td></tr>
  <tr><td align="center">opA / opB</td><td align="center">-</td><td align="center">-</td><td align="center">opA 仅 N；opB 支持 N / T</td></tr>
  <tr><td rowspan="1" align="center">算子输出</td><td align="center">C</td><td align="center">rows × colsOut</td><td align="center">fp32 / fp16 / int32</td><td align="center">行主序 / 列主序</td></tr>
  <tr><td rowspan="1" align="center">核函数名</td><td colspan="6" align="center">spmm_kernel_launch</td></tr>
  </table>

- 支持的数据类型组合：

  | A | B | C | computeType |
  |---|---|---|-------------|
  | fp32 | fp32 | fp32 | fp32 |
  | fp16 | fp16 | fp16 | fp32（fp32 累加） |
  | int8 | int8 | int32 | int32 |

- 约束限制：
  - 稀疏矩阵 A 当前仅支持 CSR 格式
  - `opA` 当前仅支持 `ACL_SPARSE_OP_NON_TRANSPOSE`（非转置）
  - `opB` 支持 `ACL_SPARSE_OP_NON_TRANSPOSE` 与 `ACL_SPARSE_OP_TRANSPOSE`
  - 索引类型当前仅支持 `ACL_SPARSE_INDEX_32I`（`ACL_SPARSE_INDEX_64I` 暂未支持）
  - `beta = 0` 时会跳过 C 的读取，提供快路径
  - fp32 高精度算法（`ACL_SPARSE_SPMM_CSR_FP32_HIGH_PRECISION_ALG`）仅对 fp32 生效，fp16 / int8 会自动忽略

### 算法说明

| 算法枚举 | 说明 |
|---------|------|
| `ACL_SPARSE_SPMM_ALG_DEFAULT` | 默认算法，推荐使用；当前版本对 CSR 格式走 SIMT/AIV 实现 |
| `ACL_SPARSE_SPMM_CSR_ALG1` | CSR 算法 1，显式指定 CSR 路径；当前版本与 DEFAULT 同一实现 |
| `ACL_SPARSE_SPMM_CSR_FP32_HIGH_PRECISION_ALG` | fp32 高精度算法；同一 SIMT Kernel，fp32 累加使用 Kahan 补偿求和；仅对 fp32 生效 |

### 测试实现

- 测试流程 (`spmm_test.cpp`)

1. **初始化**：初始化 ACL 环境，设置设备和创建 stream
2. **生成测试数据**：在线生成随机 CSR 矩阵与稠密矩阵 B、C
3. **CPU 参考计算**：使用 `SpmmCpuFp32` / `SpmmCpuInt8` 等函数计算参考结果
4. **设备内存管理**：分配和拷贝数据到设备内存
5. **稀疏矩阵操作**：使用 aclsparse API 创建 CSR 矩阵与稠密矩阵描述符
6. **获取 workspace**：调用 `aclsparseSpMMGetBufferSize` 获取缓冲区大小
7. **预处理**：调用 `aclsparseSpMMPreprocess` 完成行重排与分桶
8. **执行 SPMM**：调用 `aclsparseSpMM` 执行稀疏矩阵-稠密矩阵乘法
9. **结果验证**：将设备计算结果拷贝回主机，与 CPU 参考结果进行比较
10. **清理资源**：释放设备和主机内存，销毁描述符

- 关键代码片段

```cpp
// A: rows × cols，B: cols × colsOut，C: rows × colsOut
CHECK_ACL_SPARSE(aclsparseCreateCsr(&matA, rows, cols, nnz, dRowOff, dColInd, dVals,
    ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_32I,
    ACL_SPARSE_INDEX_BASE_ZERO, ACL_FLOAT));
CHECK_ACL_SPARSE(aclsparseCreateDnMat(&matB, cols, colsOut, ldb, dB, ACL_FLOAT, ACL_SPARSE_ORDER_ROW));
CHECK_ACL_SPARSE(aclsparseCreateDnMat(&matC, rows, colsOut, ldc, dC, ACL_FLOAT, ACL_SPARSE_ORDER_ROW));

// 获取 workspace 大小
size_t bufferSize = 0;
CHECK_ACL_SPARSE(aclsparseSpMMGetBufferSize(handle,
    ACL_SPARSE_OP_NON_TRANSPOSE, ACL_SPARSE_OP_NON_TRANSPOSE,
    &alpha, matA, matB, &beta, matC,
    ACL_FLOAT, ACL_SPARSE_SPMM_CSR_ALG1, &bufferSize));

// 预处理 + 执行 SpMM
CHECK_ACL_SPARSE(aclsparseSpMMPreprocess(handle,
    ACL_SPARSE_OP_NON_TRANSPOSE, ACL_SPARSE_OP_NON_TRANSPOSE,
    &alpha, matA, matB, &beta, matC,
    ACL_FLOAT, ACL_SPARSE_SPMM_CSR_ALG1, dBuffer));
CHECK_ACL_SPARSE(aclsparseSpMM(handle,
    ACL_SPARSE_OP_NON_TRANSPOSE, ACL_SPARSE_OP_NON_TRANSPOSE,
    &alpha, matA, matB, &beta, matC,
    ACL_FLOAT, ACL_SPARSE_SPMM_CSR_ALG1, dBuffer));
```

## 编译运行

在 ops-sparse 仓库根目录下执行如下步骤，编译并执行 SPMM 算子测试。

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
bash build.sh --ops=spmm --soc=ascend950 --run
```

执行结果如下，说明精度对比成功：

```bash
========== Results ==========
  FP32: PASS  FP16: PASS  INT8: PASS
  Overall: PASS
```

## 接口说明

SPMM 采用三段式调用接口，完整 API 说明参见 [接口列表](../../docs/zh/api_list.md)。

### aclsparseSpMM

**函数原型**：

```cpp
aclsparseStatus_t aclsparseSpMM(
    aclsparseHandle_t handle,
    aclsparseOperation_t opA,
    aclsparseOperation_t opB,
    const void *alpha,
    aclsparseConstSpMatDescr_t matA,
    aclsparseConstDnMatDescr_t matB,
    const void *beta,
    aclsparseDnMatDescr_t matC,
    aclDataType computeType,
    aclsparseSpMMAlg_t alg,
    void *buffer);
```

**参数说明**：

- `handle`：稀疏矩阵操作的句柄（`aclsparseHandle_t`）
- `opA`：稀疏矩阵 A 的转置选项，当前版本仅支持 `ACL_SPARSE_OP_NON_TRANSPOSE`
- `opB`：稠密矩阵 B 的转置选项（`ACL_SPARSE_OP_NON_TRANSPOSE` / `ACL_SPARSE_OP_TRANSPOSE`）
- `alpha`：标量 alpha
- `matA`：CSR 稀疏矩阵描述符（`aclsparseConstSpMatDescr_t`，只读输入）
- `matB`：输入稠密矩阵 B 描述符（`aclsparseConstDnMatDescr_t`，只读输入）
- `beta`：标量 beta
- `matC`：输入/输出稠密矩阵 C 描述符（`aclsparseDnMatDescr_t`）
- `computeType`：计算数据类型（`aclDataType`）
- `alg`：SPMM 算法选择（`aclsparseSpMMAlg_t`）
- `buffer`：工作空间（DEVICE），需先调用 `aclsparseSpMMGetBufferSize` 分配，并可通过 `aclsparseSpMMPreprocess` 预处理

**返回值**：

- `ACL_SPARSE_STATUS_SUCCESS`：成功
- 其他值：失败
