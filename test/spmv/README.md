# SPMV 算子

## 概述

ops-sparse 仓库中的 SPMV (Sparse Matrix-Vector Multiplication) 算子实现了稀疏矩阵与稠密向量的乘法运算，支持非转置和转置两种模式：

$$
\begin{aligned}
\text{非转置:}\quad y &= \alpha \cdot A \cdot x + \beta \cdot y \\
\text{转置:}\quad y &= \alpha \cdot A^T \cdot x + \beta \cdot y
\end{aligned}
$$

其中 A 为 M×N 稀疏矩阵，x、y 为稠密向量，alpha、beta 为标量。算子基于 aclsparse 统一接口。

## 支持的 AI 处理器

| 产品 | 是否支持 |
| ---- | :----: |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |

## 目录结构介绍

```
src/spmv/arch22
├── kernels/                           // 各类型独立编译单元（每 TU 一个 kernel）
│   ├── spmv_kernel.h                 // 模板类 + DEFINE 宏
│   ├── spmv_kernel_f32.cpp           // <float, float, float>
│   ├── spmv_kernel_f32_half_half.cpp // <float, half, half>
│   ├── ...
├── spmv_host.cpp                     // Host 侧 API 实现
├── spmv_tiling_data.h                // Tiling 数据结构
└── spmv_utils.h                      // 宏和内部辅助

test/spmv/
├── CMakeLists.txt                    // 编译工程文件
├── README.md                         // 说明文档
└── spmv_test.cpp                     // 算子调用样例
```

## 算子描述

### 功能

SPMV 算子实现了 CSR 格式稀疏矩阵与稠密向量的乘法运算，对应的数学表达式为：

$$
y = \alpha \cdot op(A) \cdot x + \beta \cdot y
$$

其中 `op(A)` 可为 A（非转置）或 Aᵀ（转置）。

### 支持的类型组合

算子支持多种输入精度、计算精度与输出精度的组合：

| 输入 A、X 类型 | 计算类型 (computeType) | 输出 Y 类型 | 文件名中的模板参数 |
| :---: | :---: | :---: | :--- |
| float32 | float32 | float32 | `<float>` |
| int32 | int32 | int32 | `<int32_t>` |
| int32 | float32 | float32 | `<float, int32_t, float>` |
| float16 | float32 | float32 | `<float, half, float>` |
| float16 | float32 | float16 | `<float, half, half>` |
| bfloat16 | float32 | float32 | `<float, bfloat16_t, float>` |
| bfloat16 | float32 | bfloat16 | `<float, bfloat16_t, bfloat16_t>` |

> **注意**：`alpha`/`beta` 的类型必须与 `computeType` 一致：`computeType=float32` 时传 `float` 指针，`computeType=int32` 时传 `int32_t` 指针，否则会导致类型双关错误。

### 存储格式

本实现采用 CSR (Compressed Sparse Row) 格式存储稀疏矩阵，该格式由三个数组组成：

- `csrRowPtr`：行偏移数组，存储每行第一个非零元素在 `csrVal` 和 `csrColInd` 数组中的位置
- `csrColInd`：列索引数组，存储非零元素的列索引
- `csrVal`：值数组，存储非零元素的值

### 实现原理

1. **类型独立编译**：模板类 `SpmvKernel` / `SpmvKernelTrans` 定义在 `kernels/spmv_kernel.h` 中，7 个 `.cpp` 文件各自通过 `DEFINE_SPMV_KERNEL_ENTRY` 宏展开一个具体的 `__global__` 核函数和 host 端启动包装，每个 TU 仅一个实例化
2. **统一内核入口**：核函数通过 `trans` 标量参数控制非转置/转置分发，非转置使用 `SpmvKernel`，转置使用 `SpmvKernelTrans`
3. **行并行策略**：每个 AI Core 处理若干整行，核间均分 workload；转置模式下结果通过原子累加写入，避免核间踩踏
4. **CopyIn → Compute → CopyOut 管线**：行内计算通过三级流水完成，UB 缓冲区复用提高效率
5. **混合精度**：当输入/输出类型与计算类型不一致时，通过 `Cast` 矢量指令在 UB 上完成类型转换，ReduceSum 固定使用 float 精度
6. **结果验证**：在 `spmv_test.cpp` 中，通过 CPU 计算 golden 真值，验证 NPU 计算结果的正确性

### 算子规格参数说明

| 参数名 | 输入/输出/属性 | 描述 | 数据类型 | 数据格式 |
| ------ | ------------ | ---- | -------- | ------- |
| csrRowPtr | 输入 | 矩阵中每一行的第一个非零元素在 csrVal 中的位置 | int32_t | ND |
| csrColInd | 输入 | csrVal 中每个非零元素的列索引 | int32_t | ND |
| csrVal | 输入 | 稀疏矩阵中所有非零元素的值 | float / int32_t / half / bfloat16 | ND |
| xVec | 输入 | 被乘稠密向量 | 与 csrVal 一致 | ND |
| yVec | 输入/输出 | 结果向量（也作为 beta*y 的输入） | float / int32_t / half / bfloat16 | ND |
| alpha | 输入 | 标量系数 alpha | 与 computeType 一致 | scalar |
| beta | 输入 | 标量系数 beta | 与 computeType 一致 | scalar |
| opA | 属性 | 矩阵操作类型（NON_TRANSPOSE / TRANSPOSE） | enum | — |

### 约束说明

- **矩阵格式**：当前仅支持 CSR 格式，CSC 格式校验后会直接返回 `ACL_SPARSE_STATUS_NOT_SUPPORTED`
- **算法类型**：当前仅支持 `ACL_SPARSE_SPMV_ALG_DEFAULT`，传入其他算法会提示错误并返回 `ACL_SPARSE_STATUS_NOT_SUPPORTED`
- **转置支持**：支持非转置和转置，暂不支持共轭转置
- **UB 容量限制**：稀疏矩阵单行最大非零元数受 UB 容量限制，计算公式如下：
  $$maxTileLength = \left\lfloor \frac{UB\_SIZE - kFixed}{perElem} \right\rfloor_{\text{align}}$$
  $$perElem = \text{sizeof}(int32\_t) + 2 \cdot \text{sizeof}(CompT) + 2 \cdot \text{sizeof}(float)$$
  $$kFixed = kYLocalBytes + kSystemReserved + kAlignSlack \approx 4KB$$

### 测试实现

- **测试流程** (`spmv_test.cpp`)

1. **生成测试数据**：在线生成随机 CSR 矩阵（`GenerateCsr`）和稠密向量 x、y（`GenerateDenseVector`）
2. **CPU 参考计算**：非转置使用 `SpmvCpu`，转置使用 `SpmvTransCpu`
3. **初始化 ACL 环境**：初始化 ACL 环境，设置设备和创建流
4. **创建设备内存**：使用 `CreateDeviceTensor` 拷贝矩阵和向量数据到 device，用 `unique_ptr` 托管生命周期
5. **创建描述符**：使用 `aclSparseCreateCsr` 和 `aclSparseCreateDnVec` 创建矩阵和向量描述符
6. **查询 Buffer 大小**：调用 `aclsparseSpMVGetBufferSize` 获取所需工作空间大小
7. **执行 SpMV**：调用 `aclsparseSpMV` 执行稀疏矩阵向量乘法
8. **结果验证**：将 device 结果回读到 host，与 CPU 参考结果比较。float 类型使用相对误差（MARE/MERE），int32 类型使用绝对误差（逐元素完全匹配检查）
9. **清理资源**：销毁描述符，unique_ptr 自动释放设备/主机内存

- **测试覆盖**

| 测试组 | 用例数 | 说明 |
| --- | :---: | --- |
| Float 基础测试 | 6 | 不同稀疏度和形状，alpha=1.0, beta=0.0 |
| Float Alpha/Beta 测试 | 7 | alpha/beta 组合，含负值、pass-through 等 |
| Float 随机采样 | 30 | 随机 M/N/sparsity/alpha/beta |
| Int32 基础测试 | 5 | 不同稀疏度和形状，alpha=1, beta=0 |
| Int32 Alpha/Beta 测试 | 4 | alpha/beta 组合，含负值 |
| Int32 随机采样 | 30 | 随机 M/N/sparsity/alpha/beta |
| Int32→Float32 Non-Transpose | 13 | 混合精度，int32 输入 float32 计算 |
| Int32→Float32 Transpose | 12 | 混合精度 + 转置 |
| Float16→Float32 Non-Transpose | 8 | 混合精度，half 输入 float32 输出 |
| Float16→Float32 Transpose | 8 | 混合精度 + 转置 |
| BF16→Float32 Non-Transpose | 8 | 混合精度，bfloat16 输入 float32 输出 |
| BF16→Float32 Transpose | 8 | 混合精度 + 转置 |
| Float Transpose | 17 | 转置全场景覆盖 |
| Int32 Transpose | 11 | int32 转置全场景覆盖 |

### 精度验证方法

- **float / half / bfloat16**：采用相对误差（Relative Error），计算公式：
  $$rError_i = \frac{|npu_i - golden_i|}{|golden_i| + 10^{-7}}$$
  - MARE（Max Absolute Relative Error）：所有元素中最大的 rError
  - MERE（Mean Relative Error）：所有元素 rError 的平均值
  - 半精度类型 cast 到 float 后再计算

- **int32_t**：采用绝对误差（Absolute Error）和逐元素精确匹配：
  $$aError_i = |npu_i - golden_i|$$
  - 所有元素必须精确相等才算通过
  - MARE：最大绝对误差
  - MERE：平均绝对误差

- **关键代码片段**

```cpp
// 生成随机 CSR 矩阵和稠密向量
GenerateCsr<T>(M, N, sparsity, csrRowPtr, csrColInd, csrVal);
GenerateDenseVector<T>(xSize, xVec, rng);
GenerateDenseVector<T>(ySize, yVec, rng);

// 计算 CPU 参考结果
std::vector<OutT> output_cpu;
if (transpose)
    output_cpu = SpmvTransCpu<CompT, ValT, OutT>(csrRowPtr, csrColInd, csrVal, xVec, yVec, M, N, alpha, beta);
else
    output_cpu = SpmvCpu<CompT, ValT, OutT>(csrRowPtr, csrColInd, csrVal, xVec, yVec, alpha, beta);

// 执行 SpMV（alpha/beta 类型必须与 computeType 匹配）
aclsparseOperation_t op = transpose ? ACL_SPARSE_OP_TRANSPOSE : ACL_SPARSE_OP_NON_TRANSPOSE;
sparseRet = aclsparseSpMV(spHandle, op, &alphaTyped,
                           matDesc, vecXDesc, &betaTyped, vecYDesc,
                           compDt, ACL_SPARSE_SPMV_ALG_DEFAULT,
                           externalBuffer);
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
bash build.sh --ops=spmv --run
```

执行结果如下，说明精度对比成功：

```txt
======== Float Basic Tests (alpha=1.0, beta=0.0) ========
====Test case: row num = 512 col num = 1024 sparsity (zero ratio) = 0.9 alpha = 1 beta = 0====
Verification...
...
====Test case pass!====

======== Int32 Basic Tests (alpha=1, beta=0) ========
====Test case: row num = 512 col num = 1024 sparsity (zero ratio) = 0.9 alpha = 1 beta = 0====
Verification...
Mean Absolute Error = 0; Max Absolute Error = 0
====Test case pass!====

======== Int32->Float32 Non-Transpose ========
...
====Test case pass!====
```

## 接口说明

### aclsparseSpMVGetBufferSize

**函数原型**：

```cpp
aclsparseStatus_t aclsparseSpMVGetBufferSize(
    aclsparseHandle_t handle,
    aclsparseOperation_t opA,
    const void *alpha,
    aclsparseConstSpMatDescr_t matA,
    aclsparseConstDnVecDescr_t vecX,
    const void *beta,
    aclsparseDnVecDescr_t vecY,
    aclDataType computeType,
    aclsparseSpMVAlg_t alg,
    size_t *bufferSize);
```

**参数说明**：

| 参数 | 方向 | 描述 |
| --- | :---: | --- |
| handle | IN | aclsparse 句柄 |
| opA | IN | 矩阵操作类型（`ACL_SPARSE_OP_NON_TRANSPOSE` / `ACL_SPARSE_OP_TRANSPOSE`） |
| alpha | IN | 标量 alpha 指针，类型必须与 computeType 一致 |
| matA | IN | 稀疏矩阵描述符 |
| vecX | IN | 输入稠密向量 x 描述符 |
| beta | IN | 标量 beta 指针，类型必须与 computeType 一致 |
| vecY | IN | 输出稠密向量 y 描述符 |
| computeType | IN | 计算数据类型（`ACL_FLOAT` / `ACL_INT32`） |
| alg | IN | 算法类型，当前仅支持 `ACL_SPARSE_SPMV_ALG_DEFAULT` |
| bufferSize | OUT | 所需工作缓冲区大小 |

### aclsparseSpMV

**函数原型**：

```cpp
aclsparseStatus_t aclsparseSpMV(
    aclsparseHandle_t handle,
    aclsparseOperation_t opA,
    const void *alpha,
    aclsparseConstSpMatDescr_t matA,
    aclsparseConstDnVecDescr_t vecX,
    const void *beta,
    aclsparseDnVecDescr_t vecY,
    aclDataType computeType,
    aclsparseSpMVAlg_t alg,
    void *externalBuffer);
```

**参数说明**：

| 参数 | 方向 | 描述 |
| --- | :---: | --- |
| handle | IN | aclsparse 句柄 |
| opA | IN | 矩阵操作类型（`ACL_SPARSE_OP_NON_TRANSPOSE` / `ACL_SPARSE_OP_TRANSPOSE`） |
| alpha | IN | 标量 alpha 指针，类型必须与 computeType 一致 |
| matA | IN | 稀疏矩阵描述符 |
| vecX | IN | 输入稠密向量 x 描述符 |
| beta | IN | 标量 beta 指针，类型必须与 computeType 一致 |
| vecY | IN/OUT | 稠密向量 y 描述符（y 为被乘向量，结果覆盖写入 y） |
| computeType | IN | 计算数据类型（`ACL_FLOAT` / `ACL_INT32`） |
| alg | IN | 算法类型，当前仅支持 `ACL_SPARSE_SPMV_ALG_DEFAULT` |
| externalBuffer | IN | 工作缓冲区（大小由 `GetBufferSize` 获取） |

**返回值**：

| 返回值 | 说明 |
| --- | --- |
| `ACL_SPARSE_STATUS_SUCCESS` | 成功 |
| `ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR` | handle 为空 |
| `ACL_SPARSE_STATUS_INVALID_VALUE` | matA / vecX / vecY / externalBuffer 为空，或向量大小不满足要求 |
| `ACL_SPARSE_STATUS_NOT_SUPPORTED` | 矩阵格式非 CSR、computeType 不支持、算法不支持、或 valType/outType 组合不支持 |
| `ACL_SPARSE_STATUS_INSUFFICIENT_RESOURCES` | 单行非零元数超过 UB 容量 |
| `ACL_SPARSE_STATUS_EXECUTION_FAILED` | stream 同步失败 |
