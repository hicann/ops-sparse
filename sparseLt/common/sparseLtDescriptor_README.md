# sparseLtDescriptor 类接口

## 接口概述

aclsparseLt 的 mat 描述符（MatDescriptor）与 matmul 描述符（MatmulDescriptor）类接口，用于创建和销毁矩阵乘法运算所需的描述符。本类接口所描述的矩阵乘法运算语义对应如下公式：

```
D = α · op(A) · op(B) + β · op(C) + bias
```

公式中，A、B 为输入矩阵，C 为累加矩阵，D 为输出矩阵；op() 表示对矩阵施加的转置或非转置操作；α、β 为系数，bias 为偏置。

描述符用于在计算执行前，记录参与运算的矩阵特征：矩阵的形状、内存布局、数据类型、稀疏模式、操作类型（转置 / 非转置）和计算精度等。本类接口只负责描述符的创建与销毁，不执行任何数值计算，实际计算由后续的 matmul 计算接口完成。

本类包含以下接口：

| 接口名 | 功能简述 |
|--------|---------|
| aclsparseLtDenseDescriptorInit | 初始化稠密矩阵描述符 |
| aclsparseLtStructuredDescriptorInit | 初始化结构化稀疏矩阵描述符 |
| aclsparseLtMatDescriptorDestroy | 销毁 mat 描述符 |
| aclsparseLtMatmulDescriptorInit | 初始化 matmul 描述符 |
| aclsparseLtMatmulDescriptorDestroy | 销毁 matmul 描述符 |

> **销毁顺序说明**：`aclsparseLtMatmulDescriptorDestroy` 只销毁 matmul 描述符本身，不会销毁其中引用的 A/B/C/D 的 mat 描述符。使用完毕后，应先销毁 matmul 描述符，再销毁 A/B/C/D 的 mat 描述符。如果顺序颠倒，matmul 描述符中保存的 mat 描述符引用将指向已释放的内存，产生非法访问风险。

## 接口说明

### aclsparseLtDenseDescriptorInit

#### 产品支持情况

- Ascend 950PR / Ascend 950DT：支持
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品：支持
- Atlas A2 训练系列产品 / Atlas A2 推理系列产品：支持

#### 功能描述

初始化稠密矩阵描述符，用于描述参与计算的稠密矩阵的各类属性。

#### 函数原型

```cpp
aclsparseStatus_t aclsparseLtDenseDescriptorInit(
    const aclsparseLtHandle_t* handle, aclsparseLtMatDescriptor_t* matDescr,
    int64_t rows, int64_t cols, int64_t ld, uint32_t alignment,
    aclDataType valueType, aclsparseOrder_t order)
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 | 内存位置 |
|--------|----------|---------|------|----------|
| handle | 输入 | const aclsparseLtHandle_t* | aclsparseLt 库句柄 | Host |
| matDescr | 输入/输出 | aclsparseLtMatDescriptor_t* | 初始化的 mat 描述符 | Host |
| rows | 输入 | int64_t | 矩阵行数 | Host |
| cols | 输入 | int64_t | 矩阵列数 | Host |
| ld | 输入 | int64_t | 矩阵的 leading dimension，即存储矩阵时每行（ROW 布局）或每列（COL 布局）在内存中的实际步长 | Host |
| alignment | 输入 | uint32_t | 内存对齐字节数 | Host |
| valueType | 输入 | aclDataType | 矩阵数据类型，支持的数据类型见[支持数据类型](#支持数据类型)章节 | Host |
| order | 输入 | aclsparseOrder_t | 矩阵在内存中的布局 | Host |

#### 约束说明

- handle 不可为 nullptr
- matDescr 不可为 nullptr，且调用前 *matDescr 必须为 nullptr，防止覆盖已存在的描述符
- rows、cols 必须大于 0，且必须为对应 valueType 对齐倍数的整数倍
- ld 必须大于 0；在 COL 布局下必须大于等于 rows，在 ROW 布局下必须大于等于 cols；且必须为对应 valueType 对齐倍数的整数倍
- alignment 必须为 16 的倍数且大于 0
- order 取值为 ACL_SPARSE_ORDER_ROW（按行存储）或 ACL_SPARSE_ORDER_COL（按列存储）
- valueType 必须是受支持的数据类型

---

### aclsparseLtStructuredDescriptorInit

#### 产品支持情况

- Ascend 950PR / Ascend 950DT：支持
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品：支持
- Atlas A2 训练系列产品 / Atlas A2 推理系列产品：支持

#### 功能描述

初始化结构化稀疏（2:4）矩阵描述符，用于描述参与计算的结构化稀疏矩阵。

#### 函数原型

```cpp
aclsparseStatus_t aclsparseLtStructuredDescriptorInit(
    const aclsparseLtHandle_t* handle, aclsparseLtMatDescriptor_t* matDescr,
    int64_t rows, int64_t cols, int64_t ld, uint32_t alignment,
    aclDataType valueType, aclsparseOrder_t order,
    aclsparseLtSparsity_t sparsity)
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 | 内存位置 |
|--------|----------|---------|------|----------|
| handle | 输入 | const aclsparseLtHandle_t* | aclsparseLt 库句柄 | Host |
| matDescr | 输入/输出 | aclsparseLtMatDescriptor_t* | 初始化的 mat 描述符 | Host |
| rows | 输入 | int64_t | 矩阵行数 | Host |
| cols | 输入 | int64_t | 矩阵列数 | Host |
| ld | 输入 | int64_t | 矩阵的 leading dimension，即存储矩阵时每行（ROW 布局）或每列（COL 布局）在内存中的实际步长 | Host |
| alignment | 输入 | uint32_t | 内存对齐字节数 | Host |
| valueType | 输入 | aclDataType | 矩阵数据类型，支持的数据类型见[支持数据类型](#支持数据类型)章节 | Host |
| order | 输入 | aclsparseOrder_t | 矩阵在内存中的布局 | Host |
| sparsity | 输入 | aclsparseLtSparsity_t | 矩阵的稀疏模式 | Host |

#### 约束说明

- 同 aclsparseLtDenseDescriptorInit 的约束，其中 rows、cols、ld 的对齐倍数采用 structured 档位，为 dense 档位的 2 倍
- sparsity 目前仅支持 ACL_SPARSE_LT_SPARSITY_50_PERCENT

---

### aclsparseLtMatDescriptorDestroy

#### 产品支持情况

- Ascend 950PR / Ascend 950DT：支持
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品：支持
- Atlas A2 训练系列产品 / Atlas A2 推理系列产品：支持

#### 功能描述

销毁 mat 描述符，释放其内存并将 *matDescr 置为 nullptr，销毁后不能再使用该描述符。

#### 函数原型

```cpp
aclsparseStatus_t aclsparseLtMatDescriptorDestroy(
    aclsparseLtMatDescriptor_t* matDescr)
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 | 内存位置 |
|--------|----------|---------|------|----------|
| matDescr | 输入/输出 | aclsparseLtMatDescriptor_t* | 指向要销毁的 mat 描述符的指针 | Host |

#### 约束说明

- 若 matDescr 指针本身为 nullptr，返回 ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR
- 若 *matDescr 为 nullptr，视为空操作，返回 ACL_SPARSE_STATUS_SUCCESS

---

### aclsparseLtMatmulDescriptorInit

#### 产品支持情况

- Ascend 950PR / Ascend 950DT：支持
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品：支持
- Atlas A2 训练系列产品 / Atlas A2 推理系列产品：支持

#### 功能描述

初始化 matmul 描述符，把参与运算的 A/B/C/D 的 mat 描述符、转置操作与计算精度汇总为一个 matmul 运算的完整描述。

#### 函数原型

```cpp
aclsparseStatus_t aclsparseLtMatmulDescriptorInit(
    const aclsparseLtHandle_t* handle, aclsparseLtMatmulDescriptor_t* matmulDescr,
    aclsparseOperation_t opA, aclsparseOperation_t opB,
    const aclsparseLtMatDescriptor_t* matA,
    const aclsparseLtMatDescriptor_t* matB,
    const aclsparseLtMatDescriptor_t* matC,
    const aclsparseLtMatDescriptor_t* matD,
    aclsparseComputeType_t computeType)
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 | 内存位置 |
|--------|----------|---------|------|----------|
| handle | 输入 | const aclsparseLtHandle_t* | aclsparseLt 库句柄 | Host |
| matmulDescr | 输入/输出 | aclsparseLtMatmulDescriptor_t* | 初始化的 matmul 描述符 | Host |
| opA | 输入 | aclsparseOperation_t | 作用于矩阵 A 的转置操作 | Host |
| opB | 输入 | aclsparseOperation_t | 作用于矩阵 B 的转置操作 | Host |
| matA | 输入 | const aclsparseLtMatDescriptor_t* | A 矩阵的 mat 描述符 | Host |
| matB | 输入 | const aclsparseLtMatDescriptor_t* | B 矩阵的 mat 描述符 | Host |
| matC | 输入 | const aclsparseLtMatDescriptor_t* | C 矩阵的 mat 描述符 | Host |
| matD | 输入 | const aclsparseLtMatDescriptor_t* | D 矩阵的 mat 描述符 | Host |
| computeType | 输入 | aclsparseComputeType_t | 计算精度 | Host |

#### 约束说明

- handle 不可为 nullptr
- matmulDescr 不可为 nullptr，且调用前 *matmulDescr 必须为 nullptr（防止覆盖已存在的描述符）
- opA、opB 取值为 ACL_SPARSE_OP_NON_TRANSPOSE（非转置）或 ACL_SPARSE_OP_TRANSPOSE（转置）
- matA、matB、matC、matD 不可为 nullptr，且必须已初始化
- matA 与 matB 中必须有且仅有一个为 structured 类型，另一个为 dense 类型
- matC 和 matD 必须为 dense 类型
- matC 与 matD 的 ld 和 order 必须相同
- matC、matD 的 rows 和 cols 都必须小于等于 2097120
- 当 matA 的数据类型为 INT8/FP8/FP4 时，opA/opB 与 orderA/orderB 的组合仅支持四种合法形式（见[操作与布局组合约束](#操作与布局组合约束)）
- computeType 取值为 ACL_SPARSE_COMPUTE_16F、ACL_SPARSE_COMPUTE_32F 或 ACL_SPARSE_COMPUTE_32I
- computeType 与矩阵数据类型（valueType）相互独立，两者可分别指定、互不影响

---

### aclsparseLtMatmulDescriptorDestroy

#### 产品支持情况

- Ascend 950PR / Ascend 950DT：支持
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品：支持
- Atlas A2 训练系列产品 / Atlas A2 推理系列产品：支持

#### 功能描述

销毁 matmul 描述符，释放其内存并将 *matmulDescr 置为 nullptr；不销毁其中引用的 A/B/C/D 的 mat 描述符。

#### 函数原型

```cpp
aclsparseStatus_t aclsparseLtMatmulDescriptorDestroy(
    aclsparseLtMatmulDescriptor_t* matmulDescr)
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 | 内存位置 |
|--------|----------|---------|------|----------|
| matmulDescr | 输入/输出 | aclsparseLtMatmulDescriptor_t* | 指向要销毁的 matmul 描述符的指针 | Host |

#### 约束说明

- 若 matmulDescr 指针本身为 nullptr，返回 ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR
- 若 *matmulDescr 为 nullptr，视为空操作，返回 ACL_SPARSE_STATUS_SUCCESS
- 本接口不销毁 matA/matB/matC/matD

## 支持数据类型

### valueType（aclDataType）

下表列出了接口支持的数据类型，以及每种类型在 Dense 和 Structured 档位下要求矩阵的 rows、cols 和 ld 满足的对齐倍数。若 rows、cols、ld 不足对齐倍数，算子无法按硬件对齐规则读取数据。

| 数据类型 | aclDataType 枚举 | Dense 对齐倍数 | Structured 对齐倍数 |
|----------|------------------|---------------|---------------------|
| FP32 | ACL_FLOAT | 4 | 8 |
| FP16 | ACL_FLOAT16 | 8 | 16 |
| BF16 | ACL_BF16 | 8 | 16 |
| INT8 | ACL_INT8 | 16 | 32 |
| FP8_E4M3 | ACL_FLOAT8_E4M3FN | 16 | 32 |
| FP8_E5M2 | ACL_FLOAT8_E5M2 | 16 | 32 |
| FP4_E2M1 | ACL_FLOAT4_E2M1 | 16 | 32 |
| HIFLOAT8 | ACL_HIFLOAT8 | 16 | 32 |
| FP8_E8M0 | ACL_FLOAT8_E8M0 | 16 | 32 |
| FP4_E1M2 | ACL_FLOAT4_E1M2 | 16 | 32 |

> 支持列表外的 aclDataType 返回 ACL_SPARSE_STATUS_NOT_SUPPORTED。

### computeType（aclsparseComputeType_t）

| 计算精度 | 枚举 |
|----------|------|
| 16F | ACL_SPARSE_COMPUTE_16F |
| 32F | ACL_SPARSE_COMPUTE_32F |
| 32I | ACL_SPARSE_COMPUTE_32I |

## 约束说明

### 对齐约束

rows、cols、ld 必须为对应 valueType 对齐倍数的整数倍。Dense 与 Structured 的对齐倍数见[支持数据类型](#支持数据类型)表。alignment 必须为 16 的倍数且大于 0。

### 操作与布局组合约束

当 matA 的 valueType 属于 {ACL_INT8, ACL_FLOAT8_E4M3FN, ACL_FLOAT8_E5M2, ACL_FLOAT8_E8M0, ACL_FLOAT4_E2M1, ACL_FLOAT4_E1M2} 时施加此约束。

缩写：N = ACL_SPARSE_OP_NON_TRANSPOSE，T = ACL_SPARSE_OP_TRANSPOSE。

| orderA | orderB | 要求 opA | 要求 opB |
|--------|--------|---------|---------|
| COL | COL | T | N |
| ROW | ROW | N | T |
| ROW | COL | N | N |
| COL | ROW | T | T |

## 返回值

| 返回值 | 含义 |
|--------|------|
| ACL_SPARSE_STATUS_SUCCESS | 操作成功完成。 |
| ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR | 传入的库句柄为 nullptr，或销毁接口传入的描述符指针本身为 nullptr。 |
| ACL_SPARSE_STATUS_INVALID_VALUE | 参数校验失败。 |
| ACL_SPARSE_STATUS_NOT_SUPPORTED | 矩阵数据类型（valueType）不在支持列表内，或计算精度（computeType）非法。 |
| ACL_SPARSE_STATUS_ALLOC_FAILED | 内存分配失败。 |

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../docs/zh/develop/compile_and_run_example.md)。

```cpp
#include "cann_ops_sparseLt.h"
#include <cstdio>

int main()
{
    // 1. 创建库句柄
    aclsparseLtHandle_t handle = nullptr;
    aclsparseStatus_t st = aclsparseLtInit(&handle);
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        printf("aclsparseLtInit failed: %d\n", st);
        return -1;
    }

    // 2. 初始化 mat 描述符
    //    A 为 structured（FP16），B/C/D 为 dense（FP16），row-major
    aclsparseLtMatDescriptor_t matA = nullptr, matB = nullptr, matC = nullptr, matD = nullptr;
    const int64_t rows = 32, cols = 32, ld = 32;
    const uint32_t alignment = 16;

    st = aclsparseLtStructuredDescriptorInit(
        &handle, &matA, rows, cols, ld, alignment,
        ACL_FLOAT16, ACL_SPARSE_ORDER_ROW, ACL_SPARSE_LT_SPARSITY_50_PERCENT);
    if (st != ACL_SPARSE_STATUS_SUCCESS) { printf("matA failed: %d\n", st); return -1; }

    st = aclsparseLtDenseDescriptorInit(&handle, &matB, cols, rows, ld, alignment, ACL_FLOAT16, ACL_SPARSE_ORDER_ROW);
    if (st != ACL_SPARSE_STATUS_SUCCESS) { printf("matB failed: %d\n", st); return -1; }

    st = aclsparseLtDenseDescriptorInit(&handle, &matC, rows, rows, ld, alignment, ACL_FLOAT16, ACL_SPARSE_ORDER_ROW);
    if (st != ACL_SPARSE_STATUS_SUCCESS) { printf("matC failed: %d\n", st); return -1; }

    st = aclsparseLtDenseDescriptorInit(&handle, &matD, rows, rows, ld, alignment, ACL_FLOAT16, ACL_SPARSE_ORDER_ROW);
    if (st != ACL_SPARSE_STATUS_SUCCESS) { printf("matD failed: %d\n", st); return -1; }

    // 3. 组装 matmul 描述符
    aclsparseLtMatmulDescriptor_t matmulDescr = nullptr;
    st = aclsparseLtMatmulDescriptorInit(
        &handle, &matmulDescr,
        ACL_SPARSE_OP_NON_TRANSPOSE, ACL_SPARSE_OP_NON_TRANSPOSE,
        &matA, &matB, &matC, &matD, ACL_SPARSE_COMPUTE_16F);
    if (st != ACL_SPARSE_STATUS_SUCCESS) { printf("matmulDescr failed: %d\n", st); return -1; }

    // 4. 销毁（顺序：先 matmul，再 mat，最后 handle）
    aclsparseLtMatmulDescriptorDestroy(&matmulDescr);
    aclsparseLtMatDescriptorDestroy(&matD);
    aclsparseLtMatDescriptorDestroy(&matC);
    aclsparseLtMatDescriptorDestroy(&matB);
    aclsparseLtMatDescriptorDestroy(&matA);
    aclsparseLtDestroy(&handle);

    return 0;
}
```

接口调用成功即返回 0，无输出。