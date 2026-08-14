# aclsparseLtMatmulDescriptorInit 接口

## 接口概述

aclsparseLtMatmulDescriptorInit 为 Host 侧描述符初始化接口，用于初始化结构化稀疏矩阵乘法运算所需的描述符。语义对应：

```
D = α · op(A) · op(B) + β · op(C) + bias
```

本接口仅初始化描述符，不执行上述计算。描述符记录矩阵形状、布局、数据类型、操作类型、计算精度、稀疏模式等元信息，供后续 matmul 规划/执行接口使用。本接口不涉及 NPU kernel 计算，不执行任何数值运算。

包含以下接口：

| 接口名 | 功能简述 |
|--------|---------|
| aclsparseLtDenseDescriptorInit | 初始化稠密矩阵描述符 |
| aclsparseLtStructuredDescriptorInit | 初始化结构化稀疏矩阵描述符 |
| aclsparseLtMatmulDescriptorInit | 初始化 matmul 描述符（主接口） |
| aclsparseLtMatDescriptorDestroy | 销毁矩阵描述符 |
| aclsparseLtMatmulDescriptorDestroy | 销毁 matmul 描述符 |

调用流程：

1. `aclsparseLtInit` 创建库句柄
2. `aclsparseLtDenseDescriptorInit` / `aclsparseLtStructuredDescriptorInit` 初始化矩阵描述符
3. `aclsparseLtMatmulDescriptorInit` 组装 matmul 描述符，引用上一步的矩阵描述符
4. 使用完毕后按顺序销毁：先 matmul 描述符，再 mat 描述符，最后库句柄

> **生命周期约束**：matmul 描述符中的 matA/matB/matC/matD 为非所有权引用，`aclsparseLtMatmulDescriptorDestroy` 不会销毁它们。mat 描述符须在 matmul 描述符之后销毁，否则 matmul 描述符中残留的指针将变为悬垂指针。

## 接口说明

### aclsparseLtDenseDescriptorInit

#### 产品支持情况

- Ascend 950PR / Ascend 950DT：支持
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品：支持
- Atlas A2 训练系列产品 / Atlas A2 推理系列产品：支持

#### 函数原型

```cpp
aclsparseStatus_t aclsparseLtDenseDescriptorInit(
    const aclsparseLtHandle_t* handle, aclsparseLtMatDescriptor_t* matDescr,
    int64_t rows, int64_t cols, int64_t ld, uint32_t alignment,
    aclDataType valueType, aclsparseOrder_t order)
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 |
|--------|----------|---------|------|
| handle | 输入 | const aclsparseLtHandle_t* | aclsparseLt 库句柄，不可为 nullptr，Host 内存 |
| matDescr | 输出 | aclsparseLtMatDescriptor_t* | 矩阵描述符输出，调用前 *matDescr 须为 nullptr，Host 内存 |
| rows | 输入 | int64_t | 行数，须 > 0 且为对应 valueType 对齐倍数的整数倍，Host 内存 |
| cols | 输入 | int64_t | 列数，须 > 0 且为对应 valueType 对齐倍数的整数倍，Host 内存 |
| ld | 输入 | int64_t | leading dimension，COL 序须 >= rows，ROW 序须 >= cols，须为对齐倍数的整数倍，Host 内存 |
| alignment | 输入 | uint32_t | 内存对齐字节数，须为 16 的倍数且非 0，Host 内存 |
| valueType | 输入 | aclDataType | 矩阵数据存储类型，须在支持列表内，Host 内存 |
| order | 输入 | aclsparseOrder_t | 内存布局（ACL_SPARSE_ORDER_ROW / ACL_SPARSE_ORDER_COL），Host 内存 |

#### 约束说明

- handle 不可为 nullptr
- matDescr 不可为 nullptr，且 *matDescr 须为 nullptr（防覆盖）
- rows、cols、ld 须 > 0
- alignment 须为 16 的倍数且非 0
- order 仅支持 ACL_SPARSE_ORDER_ROW 或 ACL_SPARSE_ORDER_COL
- valueType 须在支持列表内（见"支持数据类型"）
- rows、cols、ld 须为对应 valueType 对齐倍数的整数倍（见"对齐约束"）

---

### aclsparseLtStructuredDescriptorInit

#### 产品支持情况

- Ascend 950PR / Ascend 950DT：支持
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品：支持
- Atlas A2 训练系列产品 / Atlas A2 推理系列产品：支持

#### 函数原型

```cpp
aclsparseStatus_t aclsparseLtStructuredDescriptorInit(
    const aclsparseLtHandle_t* handle, aclsparseLtMatDescriptor_t* matDescr,
    int64_t rows, int64_t cols, int64_t ld, uint32_t alignment,
    aclDataType valueType, aclsparseOrder_t order,
    aclsparseLtSparsity_t sparsity)
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 |
|--------|----------|---------|------|
| handle | 输入 | const aclsparseLtHandle_t* | aclsparseLt 库句柄，不可为 nullptr，Host 内存 |
| matDescr | 输出 | aclsparseLtMatDescriptor_t* | 矩阵描述符输出，调用前 *matDescr 须为 nullptr，Host 内存 |
| rows | 输入 | int64_t | 行数，须 > 0 且为 structured 档位对齐倍数的整数倍，Host 内存 |
| cols | 输入 | int64_t | 列数，须 > 0 且为 structured 档位对齐倍数的整数倍，Host 内存 |
| ld | 输入 | int64_t | leading dimension，COL 序须 >= rows，ROW 序须 >= cols，须为 structured 档位对齐倍数的整数倍，Host 内存 |
| alignment | 输入 | uint32_t | 内存对齐字节数，须为 16 的倍数且非 0，Host 内存 |
| valueType | 输入 | aclDataType | 矩阵数据存储类型，须在支持列表内，Host 内存 |
| order | 输入 | aclsparseOrder_t | 内存布局（ACL_SPARSE_ORDER_ROW / ACL_SPARSE_ORDER_COL），Host 内存 |
| sparsity | 输入 | aclsparseLtSparsity_t | 稀疏模式，当前仅支持 ACL_SPARSE_LT_SPARSITY_50_PERCENT，Host 内存 |

#### 约束说明

- 同 aclsparseLtDenseDescriptorInit 的约束
- sparsity 仅支持 ACL_SPARSE_LT_SPARSITY_50_PERCENT
- 对齐倍数采用 structured 档位（为 dense 档位的 2 倍）

---

### aclsparseLtMatmulDescriptorInit

#### 产品支持情况

- Ascend 950PR / Ascend 950DT：支持
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品：支持
- Atlas A2 训练系列产品 / Atlas A2 推理系列产品：支持

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

| 参数名 | 输入/输出 | 参数类型 | 说明 |
|--------|----------|---------|------|
| handle | 输入 | const aclsparseLtHandle_t* | aclsparseLt 库句柄，不可为 nullptr，Host 内存 |
| matmulDescr | 输出 | aclsparseLtMatmulDescriptor_t* | matmul 描述符输出，调用前 *matmulDescr 须为 nullptr，Host 内存 |
| opA | 输入 | aclsparseOperation_t | 作用于矩阵 A 的操作（NON_TRANSPOSE / TRANSPOSE），Host 内存 |
| opB | 输入 | aclsparseOperation_t | 作用于矩阵 B 的操作，Host 内存 |
| matA | 输入 | const aclsparseLtMatDescriptor_t* | 矩阵 A 描述符引用（structured 或 dense），不可为 nullptr 且须已初始化，Host 内存 |
| matB | 输入 | const aclsparseLtMatDescriptor_t* | 矩阵 B 描述符引用（structured 或 dense），不可为 nullptr 且须已初始化，Host 内存 |
| matC | 输入 | const aclsparseLtMatDescriptor_t* | 矩阵 C 描述符引用（dense），不可为 nullptr 且须已初始化，Host 内存 |
| matD | 输入 | const aclsparseLtMatDescriptor_t* | 矩阵 D 描述符引用（dense），不可为 nullptr 且须已初始化，Host 内存 |
| computeType | 输入 | aclsparseComputeType_t | 计算精度（ACL_SPARSE_COMPUTE_16F / 32F / 32I），与存储类型解耦，Host 内存 |

#### 约束说明

- handle 不可为 nullptr
- matmulDescr 不可为 nullptr，且 *matmulDescr 须为 nullptr（防覆盖）
- matA、matB、matC、matD 不可为 nullptr，且须已初始化（*mat 非 nullptr）
- matA 与 matB 中有且仅有一个为 structured 描述符
- matC 和 matD 必须为 dense 描述符
- matC 与 matD 须具有相同的 ld 和 order
- matC / matD 的 rows、cols 均 ≤ 2097120
- INT8/FP8/FP4 类型下，opA/opB 和 orderA/orderB 的组合受限于四种合法组合（见"操作与布局组合约束"），FP32/FP16/BF16/HIFLOAT8 不受此约束

---

### aclsparseLtMatDescriptorDestroy

#### 产品支持情况

- Ascend 950PR / Ascend 950DT：支持
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品：支持
- Atlas A2 训练系列产品 / Atlas A2 推理系列产品：支持

#### 函数原型

```cpp
aclsparseStatus_t aclsparseLtMatDescriptorDestroy(aclsparseLtMatDescriptor_t* matDescr)
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 |
|--------|----------|---------|------|
| matDescr | 输入/输出 | aclsparseLtMatDescriptor_t* | 指向要销毁的矩阵描述符的指针，销毁成功后 *matDescr 被置为 nullptr，Host 内存 |

#### 约束说明

- 若 matDescr 指针本身为 nullptr，返回 ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR
- 若 *matDescr 为 nullptr，视为空操作，返回 ACL_SPARSE_STATUS_SUCCESS

---

### aclsparseLtMatmulDescriptorDestroy

#### 产品支持情况

- Ascend 950PR / Ascend 950DT：支持
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品：支持
- Atlas A2 训练系列产品 / Atlas A2 推理系列产品：支持

#### 函数原型

```cpp
aclsparseStatus_t aclsparseLtMatmulDescriptorDestroy(aclsparseLtMatmulDescriptor_t* matmulDescr)
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 |
|--------|----------|---------|------|
| matmulDescr | 输入/输出 | aclsparseLtMatmulDescriptor_t* | 指向要销毁的 matmul 描述符的指针，销毁成功后 *matmulDescr 被置为 nullptr，Host 内存 |

#### 约束说明

- 若 matmulDescr 指针本身为 nullptr，返回 ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR
- 若 *matmulDescr 为 nullptr，视为空操作，返回 ACL_SPARSE_STATUS_SUCCESS
- 本接口不销毁 matA/matB/matC/matD（非所有权引用）

## 支持数据类型

### valueType（aclDataType）

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

> computeType 独立于 valueType，表示 matmul 计算精度档位，与存储类型解耦。

## 约束说明

### 对齐约束

rows、cols、ld 须为对应 valueType 对齐倍数的整数倍。Dense 与 Structured 的对齐倍数见上方"支持数据类型"表。alignment 须为 16 的倍数且非 0。

### 操作与布局组合约束

仅当 matA 的 valueType 属于 {ACL_INT8, ACL_FLOAT8_E4M3FN, ACL_FLOAT8_E5M2, ACL_FLOAT8_E8M0, ACL_FLOAT4_E2M1, ACL_FLOAT4_E1M2} 时施加此约束。FP32/FP16/BF16/HIFLOAT8 不受此约束。

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
| ACL_SPARSE_STATUS_SUCCESS | 操作成功完成 |
| ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR | handle 为 nullptr，或 Destroy 时描述符指针为 nullptr |
| ACL_SPARSE_STATUS_INVALID_VALUE | 描述符输出指针为 nullptr / *descr 非空、维度/对齐/布局/约束违反、枚举值非法 |
| ACL_SPARSE_STATUS_NOT_SUPPORTED | valueType 不在支持列表内、computeType 非法 |
| ACL_SPARSE_STATUS_ALLOC_FAILED | 内存分配失败 |

## 范围限制

- `SetAttribute` / `GetAttribute` 接口本期不实现。batch、scale 等属性须通过此接口配置，故本期均不可通过 API 配置。

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

    // 2. 初始化矩阵描述符
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

