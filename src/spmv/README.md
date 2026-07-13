# SpMV算子

## 算子概述

SpMV（Sparse Matrix - Dense Vector Multiplication）算子实现稀疏矩阵与稠密向量的乘法运算。核心运算为 y = alpha * op(A) * x + beta * y，其中 A 为 CSR 格式稀疏矩阵，x 和 y 为稠密向量。

数学表达式：

```
y = alpha * op(A) * x + beta * y
```

其中 op(A) 为稀疏矩阵 A 的操作（转置/非转置），x 为输入稠密向量，y 为输入/输出稠密向量。

包含以下接口：

| 接口名 | 功能简述 |
|--------|---------|
| aclsparseSpMV | 执行稀疏矩阵-稠密向量乘法 y = alpha * op(A) * x + beta * y |
| aclsparseSpMVGetBufferSize | 查询所需 workspace 大小（暂未支持） |
| aclsparseSpMVPreprocess | 对稀疏矩阵进行预处理（暂未支持） |

## 算子执行接口

### aclsparseSpMV

#### 产品支持情况

- Ascend 950PR / Ascend 950DT：不支持
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品：支持
- Atlas A2 训练系列产品 / Atlas A2 推理系列产品：支持

#### 函数原型

```cpp
aclsparseStatus_t aclsparseSpMV(aclsparseHandle_t handle, aclsparseOperation_t opA, const void *alpha, aclsparseConstSpMatDescr_t matA, aclsparseConstDnVecDescr_t vecX, const void *beta, aclsparseDnVecDescr_t vecY, aclDataType computeType, aclsparseSpMVAlg_t alg, void *externalBuffer)
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 |
|--------|----------|---------|------|
| handle | 输入 | aclsparseHandle_t | ops-sparse 库上下文句柄，携带 stream，Host 内存 |
| opA | 输入 | aclsparseOperation_t | 稀疏矩阵 A 的操作类型，支持 `ACL_SPARSE_OP_NON_TRANSPOSE` 或 `ACL_SPARSE_OP_TRANSPOSE`，Host 内存 |
| alpha | 输入 | const void* | 标量 alpha 指针，类型须与 computeType 匹配。内存位置由 `aclsparseSetPointerMode` 控制，Host/Device 内存 |
| matA | 输入 | aclsparseConstSpMatDescr_t | 稀疏矩阵 A 的描述符，仅支持 CSR 格式，Host 内存 |
| vecX | 输入 | aclsparseConstDnVecDescr_t | 输入稠密向量 x 的描述符，Host 内存 |
| beta | 输入 | const void* | 标量 beta 指针，类型须与 computeType 匹配。内存位置由 `aclsparseSetPointerMode` 控制，Host/Device 内存 |
| vecY | 输入/输出 | aclsparseDnVecDescr_t | 输入/输出稠密向量 y 的描述符，Host 内存 |
| computeType | 输入 | aclDataType | 计算精度类型，支持 `ACL_FLOAT` 或 `ACL_INT32`，Host 内存 |
| alg | 输入 | aclsparseSpMVAlg_t | 算法类型，仅支持 `ACL_SPARSE_SPMV_ALG_DEFAULT`，Host 内存 |
| externalBuffer | 输入 | void* | 工作缓冲区，当前版本忽略（传 nullptr 即可），Device 内存 |

#### 约束说明

- handle 不可为 nullptr，且须先调用 `aclsparseSetStream` 设置 stream
- matA、vecX、vecY 不可为 nullptr
- matA 仅支持 CSR 格式（`ACL_SPARSE_FORMAT_CSR`）
- matA 的行偏移和列索引类型必须均为 `ACL_SPARSE_INDEX_32I`，且两者类型相同
- matA 的索引基址仅支持 `ACL_SPARSE_INDEX_BASE_ZERO`
- opA 支持 `ACL_SPARSE_OP_NON_TRANSPOSE` 或 `ACL_SPARSE_OP_TRANSPOSE`，不支持共轭转置
- alg 仅支持 `ACL_SPARSE_SPMV_ALG_DEFAULT`
- computeType 仅支持 `ACL_FLOAT` 或 `ACL_INT32`
- vecX 的值类型须与 matA 的值类型一致
- 维度匹配：
  - 非转置模式：vecX 大小 >= A.cols，vecY 大小 >= A.rows
  - 转置模式：vecX 大小 >= A.rows，vecY 大小 >= A.cols
- 支持的值类型组合：
  - matA=ACL_FLOAT, vecX=ACL_FLOAT, vecY=ACL_FLOAT, computeType=ACL_FLOAT
  - matA=ACL_FLOAT16, vecX=ACL_FLOAT16, vecY=ACL_FLOAT 或 ACL_FLOAT16, computeType=ACL_FLOAT
  - matA=ACL_BF16, vecX=ACL_BF16, vecY=ACL_FLOAT 或 ACL_BF16, computeType=ACL_FLOAT
  - matA=ACL_INT32, vecX=ACL_INT32, vecY=ACL_INT32, computeType=ACL_INT32

#### 支持的稀疏格式

| 格式 | 支持 | 说明 |
|------|------|------|
| CSR | ✅ | 稀疏矩阵 A 支持 CSR 格式 |
| COO | ❌ | 不支持 |
| CSC | ❌ | 不支持 |

#### 调用示例

暂无运行示例，具体编译和执行过程请参考[编译与运行样例](../../docs/zh/develop/compile_and_run_example.md)。

### aclsparseSpMVGetBufferSize

暂未支持。

### aclsparseSpMVPreprocess

暂未支持。
