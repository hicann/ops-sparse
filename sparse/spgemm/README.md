<!-- 
 * ----------------------------------------------------------------------------------------------------------
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ----------------------------------------------------------------------------------------------------------
 -->

# SpGEMM算子

## 算子概述

SpGEMM（Sparse General Matrix-Matrix Multiplication）算子实现稀疏矩阵与稀疏矩阵的乘法运算，对标 NVIDIA cuSPARSE `cusparseSpGEMM`。核心运算为 C = alpha * op(A) * op(B) + beta * C，其中 A、B、C 均为 CSR 格式稀疏矩阵。

与 SpMM（稀疏×稠密）不同，SpGEMM 的输出矩阵 C 的稀疏结构（nnz、每行非零分布）在运算前未知，必须先通过符号阶段确定结构，再通过数值阶段填值——这是 SpGEMM 的核心难点。

数学表达式：

```
C = alpha * op(A) * op(B) + beta * C
```

其中 A(m×k)、B(k×n)、C(m×n) 均为 CSR 稀疏矩阵，alpha/beta 为标量。

采用经典的两阶段法（对齐 cuSPARSE）：

1. **符号阶段（Symbolic Phase）**：根据 A/B 的稀疏结构（rowPtr + colInd），计算 C 的稀疏结构（rowPtrC、nnzC），为 colIndC/values 分配内存。遵循"宁多不漏"原则——即使数值抵消为 0，也计入结构非零。
2. **数值阶段（Numeric Phase）**：利用符号阶段确定的结构，进行实际乘加运算，填充 C 的 colIdx/values。每行 colIdx 按升序排列，fp32 结果 bit-wise 确定。

支持结构复用（Structure Reuse）：当 A/B 结构不变时，符号阶段只需执行一次，后续数值阶段可复用结构，支持不同 alpha/beta 快速重算。

提供两套 API：

| 接口名 | 功能简述 |
|--------|---------|
| aclsparseSpGEMMGetBufferSize | 查询所需 workspace 大小（3-stage 兼容 API） |
| aclsparseSpGEMMPreprocess | 符号阶段：确定 C 结构 + 分核调度（3-stage 兼容 API） |
| aclsparseSpGEMM | 数值阶段：填充 C 值（3-stage 兼容 API） |
| aclsparseSpGEMMCreateDescr | 创建内部描述符（7-interface API） |
| aclsparseSpGEMMDestroyDescr | 销毁内部描述符 |
| aclsparseSpGEMMWorkEstimation | 估算乘积对数 + buffer1 大小，可选执行符号阶段 |
| aclsparseSpGEMMEstimateMemory | 查询 buffer3 大小（ALG_DEFAULT 返回 0） |
| aclsparseSpGEMMCompute | 符号+数值（调两次：先探测内存上界，再执行） |
| aclsparseSpGEMMGetNumProducts | 返回乘积对总数 |
| aclsparseSpGEMMCopy | 将结果写入 matC 的 CSR 数组 |
| aclsparseSpGEMMSetCInValid | 标记 matC->values 包含有效 C_in 数据（beta≠0 扩展） |

## 算子执行接口

### aclsparseSpGEMMGetBufferSize

#### 产品支持情况

- Ascend 950PR / Ascend 950DT：支持
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品：不支持
- Atlas A2 训练系列产品 / Atlas A2 推理系列产品：不支持

#### 函数原型

```cpp
aclsparseStatus_t aclsparseSpGEMMGetBufferSize(aclsparseHandle_t handle, aclsparseOperation_t opA, aclsparseOperation_t opB, const void *alpha, aclsparseConstSpMatDescr_t matA, aclsparseConstSpMatDescr_t matB, const void *beta, aclsparseSpMatDescr_t matC, aclDataType computeType, aclsparseSpGEMMAlg_t alg, size_t *size)
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 |
|--------|----------|---------|------|
| handle | 输入 | aclsparseHandle_t | ops-sparse 库上下文句柄，携带 stream，Host 内存 |
| opA | 输入 | aclsparseOperation_t | 稀疏矩阵 A 的操作类型，仅支持 `ACL_SPARSE_OP_NON_TRANSPOSE`，Host 内存 |
| opB | 输入 | aclsparseOperation_t | 稀疏矩阵 B 的操作类型，仅支持 `ACL_SPARSE_OP_NON_TRANSPOSE`，Host 内存 |
| alpha | 输入 | const void* | 标量 alpha 指针，类型须与 computeType 匹配。内存位置由 `aclsparseSetPointerMode` 控制，Host/Device 内存 |
| matA | 输入 | aclsparseConstSpMatDescr_t | 稀疏矩阵 A 的描述符，仅支持 CSR 格式，Host 内存 |
| matB | 输入 | aclsparseConstSpMatDescr_t | 稀疏矩阵 B 的描述符，仅支持 CSR 格式，Host 内存 |
| beta | 输入 | const void* | 标量 beta 指针，类型须与 computeType 匹配。Host/Device 内存 |
| matC | 输入/输出 | aclsparseSpMatDescr_t | 输出稀疏矩阵 C 的描述符，CSR 格式，Host 内存 |
| computeType | 输入 | aclDataType | 计算精度类型，支持 `ACL_FLOAT`/`ACL_FLOAT16`/`ACL_BF16`，Host 内存 |
| alg | 输入 | aclsparseSpGEMMAlg_t | 算法类型，支持 `ACL_SPARSE_SPGEMM_ALG_DEFAULT`（=ALG1），Host 内存 |
| size | 输出 | size_t* | 输出所需 workspace 大小（字节），Host 内存 |

#### 约束说明

- handle 不可为 nullptr
- matA、matB、matC 不可为 nullptr
- matA、matB、matC 仅支持 CSR 格式（`ACL_SPARSE_FORMAT_CSR`）
- 索引类型必须均为 `ACL_SPARSE_INDEX_32I`，zero-based
- opA、opB 仅支持 `ACL_SPARSE_OP_NON_TRANSPOSE`
- A/B/C/computeType 四者数据类型必须一致（同精度：fp32/fp16/bf16）
- 维度匹配：A.cols == B.rows，A.rows == C.rows，B.cols == C.cols
- 输入 A/B 列索引须 sorted
- 维度不超过 INT32_MAX
- size 不可为 nullptr

#### 支持的稀疏格式

| 格式 | 支持 | 说明 |
|------|------|------|
| CSR | ✅ | A、B、C 均支持 CSR 格式 |
| COO | ❌ | 不支持 |
| CSC | ❌ | 不支持 |

---

### aclsparseSpGEMMPreprocess

#### 产品支持情况

- Ascend 950PR / Ascend 950DT：支持
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品：不支持
- Atlas A2 训练系列产品 / Atlas A2 推理系列产品：不支持

#### 函数原型

```cpp
aclsparseStatus_t aclsparseSpGEMMPreprocess(aclsparseHandle_t handle, aclsparseOperation_t opA, aclsparseOperation_t opB, const void *alpha, aclsparseConstSpMatDescr_t matA, aclsparseConstSpMatDescr_t matB, const void *beta, aclsparseSpMatDescr_t matC, aclDataType computeType, aclsparseSpGEMMAlg_t alg, void *buffer)
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 |
|--------|----------|---------|------|
| handle | 输入 | aclsparseHandle_t | ops-sparse 库上下文句柄，Host 内存 |
| opA | 输入 | aclsparseOperation_t | 仅支持 `ACL_SPARSE_OP_NON_TRANSPOSE`，Host 内存 |
| opB | 输入 | aclsparseOperation_t | 仅支持 `ACL_SPARSE_OP_NON_TRANSPOSE`，Host 内存 |
| alpha | 输入 | const void* | 标量 alpha 指针，Host/Device 内存 |
| matA | 输入 | aclsparseConstSpMatDescr_t | 稀疏矩阵 A 描述符，CSR 格式，Host 内存 |
| matB | 输入 | aclsparseConstSpMatDescr_t | 稀疏矩阵 B 描述符，CSR 格式，Host 内存 |
| beta | 输入 | const void* | 标量 beta 指针，Host/Device 内存 |
| matC | 输入/输出 | aclsparseSpMatDescr_t | 输出稀疏矩阵 C 描述符，Host 内存 |
| computeType | 输入 | aclDataType | 计算精度类型，Host 内存 |
| alg | 输入 | aclsparseSpGEMMAlg_t | 算法类型，Host 内存 |
| buffer | 输入 | void* | workspace 缓冲区，Device 内存 |

#### 约束说明

- 同 aclsparseSpGEMMGetBufferSize 的约束
- buffer 不可为 nullptr
- 执行后 matC->nnz 更新为实际 nnzC
- 执行后 matC->activeBuffer 标记为当前 buffer（用于结构复用检测）

#### 支持的稀疏格式

| 格式 | 支持 | 说明 |
|------|------|------|
| CSR | ✅ | A、B、C 均支持 CSR 格式 |
| COO | ❌ | 不支持 |
| CSC | ❌ | 不支持 |

---

### aclsparseSpGEMM

#### 产品支持情况

- Ascend 950PR / Ascend 950DT：支持
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品：不支持
- Atlas A2 训练系列产品 / Atlas A2 推理系列产品：不支持

#### 函数原型

```cpp
aclsparseStatus_t aclsparseSpGEMM(aclsparseHandle_t handle, aclsparseOperation_t opA, aclsparseOperation_t opB, const void *alpha, aclsparseConstSpMatDescr_t matA, aclsparseConstSpMatDescr_t matB, const void *beta, aclsparseSpMatDescr_t matC, aclDataType computeType, aclsparseSpGEMMAlg_t alg, void *buffer)
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 |
|--------|----------|---------|------|
| handle | 输入 | aclsparseHandle_t | ops-sparse 库上下文句柄，Host 内存 |
| opA | 输入 | aclsparseOperation_t | 仅支持 `ACL_SPARSE_OP_NON_TRANSPOSE`，Host 内存 |
| opB | 输入 | aclsparseOperation_t | 仅支持 `ACL_SPARSE_OP_NON_TRANSPOSE`，Host 内存 |
| alpha | 输入 | const void* | 标量 alpha 指针，Host/Device 内存 |
| matA | 输入 | aclsparseConstSpMatDescr_t | 稀疏矩阵 A 描述符，CSR 格式，Host 内存 |
| matB | 输入 | aclsparseConstSpMatDescr_t | 稀疏矩阵 B 描述符，CSR 格式，Host 内存 |
| beta | 输入 | const void* | 标量 beta 指针，Host/Device 内存 |
| matC | 输入/输出 | aclsparseSpMatDescr_t | 输出稀疏矩阵 C 描述符，Host 内存 |
| computeType | 输入 | aclDataType | 计算精度类型，Host 内存 |
| alg | 输入 | aclsparseSpGEMMAlg_t | 算法类型，Host 内存 |
| buffer | 输入 | void* | workspace 缓冲区，Device 内存 |

#### 约束说明

- 同 aclsparseSpGEMMGetBufferSize 的约束
- buffer 不可为 nullptr
- 调用前须先调用 `aclsparseSpGEMMPreprocess` 进行符号阶段（或由本接口自动调用）
- 结构复用：若 `matC->activeBuffer == buffer`，跳过符号阶段直接执行数值阶段
- beta=0 时 C_in 视为空（host 侧 memset valuesC 清零，beta·C_in = 0）
- fp32 结果 bit-wise 确定（固定遍历顺序 + 单行单线程）

#### 支持的数据类型组合

| A | B | C | computeType | 累加精度 |
|---|---|---|-------------|---------|
| fp32 | fp32 | fp32 | fp32 | fp32 |
| fp16 | fp16 | fp16 | fp16 | fp32（fp32 累加，末尾转回 fp16） |
| bf16 | bf16 | bf16 | bf16 | fp32（fp32 累加，末尾转回 bf16） |

> A/B/C/computeType 四者必须一致（同精度）。

#### 支持的稀疏格式

| 格式 | 支持 | 说明 |
|------|------|------|
| CSR | ✅ | A、B、C 均支持 CSR 格式 |
| COO | ❌ | 不支持 |
| CSC | ❌ | 不支持 |

---

### 7-interface 完整 API

除上述 3-stage 兼容 API 外，SpGEMM 还提供对齐 cuSPARSE 生命周期的 7-interface API：

| 接口 | 职责 | 对应 cuSPARSE |
|------|------|--------------|
| aclsparseSpGEMMCreateDescr | 创建内部描述符（numProds、buffer 记账、phase 状态机） | cusparseSpGEMM_createDescr |
| aclsparseSpGEMMDestroyDescr | 销毁内部描述符 | cusparseSpGEMM_destroyDescr |
| aclsparseSpGEMMWorkEstimation | 估算 numProds + buffer1 大小；可选执行符号阶段 | cusparseSpGEMM_workEstimation |
| aclsparseSpGEMMEstimateMemory | 查询 buffer3 大小（ALG_DEFAULT 返回 0） | cusparseSpGEMM_estimateMemory |
| aclsparseSpGEMMCompute | 符号+数值（调两次：先探测内存上界，再执行） | cusparseSpGEMM_compute |
| aclsparseSpGEMMGetNumProducts | 返回乘积对总数 | cusparseSpGEMM_numProducts |
| aclsparseSpGEMMCopy | 将结果写入 matC 的 CSR 数组 | cusparseSpGEMM_copy |
| aclsparseSpGEMMSetCInValid | 标记 matC->values 包含有效 C_in 数据（beta≠0 扩展） | — |

Buffer 策略：

| buffer | 使用接口 | ALG_DEFAULT |
|--------|----------|-------------|
| buffer1 | WorkEstimation + Compute（符号阶段） | 需要 |
| buffer2 | Compute + Copy（数值阶段） | 需要 |
| buffer3 | EstimateMemory | 不需要（返回 0） |

### 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../docs/zh/develop/compile_and_run_example.md)。

```cpp
#include <cstdio>
#include <memory>
#include <vector>

#include "acl/acl.h"
#include "cann_ops_sparse.h"

#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)

#define LOG_PRINT(message, ...)         \
    do {                                \
        printf(message, ##__VA_ARGS__); \
    } while (0)

class AclContext {
public:
    explicit AclContext(int32_t deviceId) : deviceId_(deviceId) {}

    ~AclContext()
    {
        if (stream_ != nullptr) {
            aclrtDestroyStream(stream_);
            stream_ = nullptr;
        }
        if (deviceSet_) {
            aclrtResetDevice(deviceId_);
            deviceSet_ = false;
        }
        if (aclInited_) {
            aclFinalize();
            aclInited_ = false;
        }
    }

    int Init()
    {
        auto ret = aclInit(nullptr);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
        aclInited_ = true;

        ret = aclrtSetDevice(deviceId_);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
        deviceSet_ = true;

        ret = aclrtCreateStream(&stream_);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
        return ACL_SUCCESS;
    }

    aclrtStream Stream() const { return stream_; }

private:
    int32_t deviceId_;
    aclrtStream stream_ = nullptr;
    bool aclInited_ = false;
    bool deviceSet_ = false;
};

static void* AllocAndCopyDevice(const void *hostPtr, size_t sizeBytes)
{
    void *dPtr = nullptr;
    aclrtMalloc(&dPtr, sizeBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    if (hostPtr != nullptr && sizeBytes > 0) {
        aclrtMemcpy(dPtr, sizeBytes, hostPtr, sizeBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    }
    return dPtr;
}

int aclsparseSpGEMMTest(AclContext& ctx)
{
    aclrtStream stream = ctx.Stream();

    // 1. 创建 ops-sparse 句柄
    aclsparseHandle_t rawHandle = nullptr;
    auto sparseRet = aclsparseCreate(&rawHandle);
    CHECK_RET(sparseRet == ACL_SPARSE_STATUS_SUCCESS, LOG_PRINT("aclsparseCreate failed. ERROR: %d\n", sparseRet);
              return sparseRet);
    std::unique_ptr<aclsparseContext, aclsparseStatus_t (*)(aclsparseHandle_t)> handlePtr(rawHandle, aclsparseDestroy);

    sparseRet = aclsparseSetStream(static_cast<aclsparseHandle_t>(handlePtr.get()), stream);
    CHECK_RET(sparseRet == ACL_SPARSE_STATUS_SUCCESS, LOG_PRINT("aclsparseSetStream failed. ERROR: %d\n", sparseRet);
              return sparseRet);

    // 2. 准备 Host 端 CSR 数据
    //    A (2x2, nnz=2):    B (2x2, nnz=2):
    //    [1.0  0.0]         [1.0  0.0]
    //    [0.0  2.0]         [0.0  3.0]
    //
    //    C = 1.0 * A * B (2x2, nnz=2):
    //    C[0] = [1.0, 0.0]
    //    C[1] = [0.0, 6.0]
    int64_t m = 2, k = 2, n = 2;
    int64_t nnzA = 2, nnzB = 2;
    float hAlpha = 1.0f;
    float hBeta = 0.0f;

    std::vector<int> hRowPtrA = {0, 1, 2};
    std::vector<int> hColIndA = {0, 1};
    std::vector<float> hValA  = {1.0f, 2.0f};

    std::vector<int> hRowPtrB = {0, 1, 2};
    std::vector<int> hColIndB = {0, 1};
    std::vector<float> hValB  = {1.0f, 3.0f};

    // 3. 拷贝数据到 Device
    void *dRowPtrA = AllocAndCopyDevice(hRowPtrA.data(), (m + 1) * sizeof(int));
    void *dColIndA = AllocAndCopyDevice(hColIndA.data(), nnzA * sizeof(int));
    void *dValA    = AllocAndCopyDevice(hValA.data(),    nnzA * sizeof(float));
    void *dRowPtrB = AllocAndCopyDevice(hRowPtrB.data(), (k + 1) * sizeof(int));
    void *dColIndB = AllocAndCopyDevice(hColIndB.data(), nnzB * sizeof(int));
    void *dValB    = AllocAndCopyDevice(hValB.data(),    nnzB * sizeof(float));

    // 预分配 matC（上限 m*n）
    int64_t maxNnzC = m * n;
    std::vector<int> hRowPtrC(m + 1, 0);
    void *dRowPtrC = AllocAndCopyDevice(hRowPtrC.data(), (m + 1) * sizeof(int));
    void *dColIndC = nullptr;
    aclrtMalloc(&dColIndC, maxNnzC * sizeof(int), ACL_MEM_MALLOC_HUGE_FIRST);
    void *dValC = nullptr;
    aclrtMalloc(&dValC, maxNnzC * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);

    // 4. 创建描述符
    aclsparseConstSpMatDescr_t matA = nullptr, matB = nullptr;
    aclsparseSpMatDescr_t matC = nullptr;

    sparseRet = aclsparseCreateConstCsr(&matA, m, k, nnzA, dRowPtrA, dColIndA, dValA,
                                        ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_32I,
                                        ACL_SPARSE_INDEX_BASE_ZERO, ACL_FLOAT);
    CHECK_RET(sparseRet == ACL_SPARSE_STATUS_SUCCESS, LOG_PRINT("CreateConstCsr A failed. ERROR: %d\n", sparseRet);
              return sparseRet);

    sparseRet = aclsparseCreateConstCsr(&matB, k, n, nnzB, dRowPtrB, dColIndB, dValB,
                                        ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_32I,
                                        ACL_SPARSE_INDEX_BASE_ZERO, ACL_FLOAT);
    CHECK_RET(sparseRet == ACL_SPARSE_STATUS_SUCCESS, LOG_PRINT("CreateConstCsr B failed. ERROR: %d\n", sparseRet);
              return sparseRet);

    sparseRet = aclsparseCreateCsr(&matC, m, n, maxNnzC, dRowPtrC, dColIndC, dValC,
                                   ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_32I,
                                   ACL_SPARSE_INDEX_BASE_ZERO, ACL_FLOAT);
    CHECK_RET(sparseRet == ACL_SPARSE_STATUS_SUCCESS, LOG_PRINT("CreateCsr C failed. ERROR: %d\n", sparseRet);
              return sparseRet);

    // 5. Step 1 — GetBufferSize
    size_t bufferSize = 0;
    sparseRet = aclsparseSpGEMMGetBufferSize(
        static_cast<aclsparseHandle_t>(handlePtr.get()),
        ACL_SPARSE_OP_NON_TRANSPOSE, ACL_SPARSE_OP_NON_TRANSPOSE,
        &hAlpha, matA, matB, &hBeta, matC, ACL_FLOAT,
        ACL_SPARSE_SPGEMM_ALG_DEFAULT, &bufferSize);
    CHECK_RET(sparseRet == ACL_SPARSE_STATUS_SUCCESS, LOG_PRINT("GetBufferSize failed. ERROR: %d\n", sparseRet);
              return sparseRet);
    LOG_PRINT("bufferSize = %zu bytes\n", bufferSize);

    void *dBuffer = nullptr;
    auto aclRet = aclrtMalloc(&dBuffer, bufferSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc buffer failed. ERROR: %d\n", aclRet); return aclRet);

    // 6. Step 2 — Preprocess（符号阶段）
    sparseRet = aclsparseSpGEMMPreprocess(
        static_cast<aclsparseHandle_t>(handlePtr.get()),
        ACL_SPARSE_OP_NON_TRANSPOSE, ACL_SPARSE_OP_NON_TRANSPOSE,
        &hAlpha, matA, matB, &hBeta, matC, ACL_FLOAT,
        ACL_SPARSE_SPGEMM_ALG_DEFAULT, dBuffer);
    CHECK_RET(sparseRet == ACL_SPARSE_STATUS_SUCCESS, LOG_PRINT("Preprocess failed. ERROR: %d\n", sparseRet);
              return sparseRet);
    aclrtSynchronizeStream(stream);

    // 7. Step 3 — SpGEMM（数值阶段）
    sparseRet = aclsparseSpGEMM(
        static_cast<aclsparseHandle_t>(handlePtr.get()),
        ACL_SPARSE_OP_NON_TRANSPOSE, ACL_SPARSE_OP_NON_TRANSPOSE,
        &hAlpha, matA, matB, &hBeta, matC, ACL_FLOAT,
        ACL_SPARSE_SPGEMM_ALG_DEFAULT, dBuffer);
    CHECK_RET(sparseRet == ACL_SPARSE_STATUS_SUCCESS, LOG_PRINT("SpGEMM failed. ERROR: %d\n", sparseRet);
              return sparseRet);
    aclrtSynchronizeStream(stream);

    // 8. 将结果拷贝回 Host 并打印
    aclRet = aclrtMemcpy(hRowPtrC.data(), (m + 1) * sizeof(int),
                         dRowPtrC, (m + 1) * sizeof(int), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("copy rowPtrC failed. ERROR: %d\n", aclRet); return aclRet);

    int32_t nnzC = hRowPtrC[m];
    LOG_PRINT("nnzC = %d\n", nnzC);

    std::vector<int> hColIndC(nnzC);
    std::vector<float> hValC(nnzC);
    aclrtMemcpy(hColIndC.data(), nnzC * sizeof(int), dColIndC, nnzC * sizeof(int), ACL_MEMCPY_DEVICE_TO_HOST);
    aclrtMemcpy(hValC.data(), nnzC * sizeof(float), dValC, nnzC * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);

    for (int32_t i = 0; i < m; i++) {
        LOG_PRINT("C[%d]: ", i);
        for (int32_t p = hRowPtrC[i]; p < hRowPtrC[i + 1]; p++) {
            LOG_PRINT("(col=%d, val=%.1f) ", hColIndC[p], hValC[p]);
        }
        LOG_PRINT("\n");
    }

    // 9. 清理资源
    aclsparseDestroySpMat(matA);
    aclsparseDestroySpMat(matB);
    aclsparseDestroySpMat(matC);
    if (dRowPtrA) aclrtFree(dRowPtrA);
    if (dColIndA) aclrtFree(dColIndA);
    if (dValA)    aclrtFree(dValA);
    if (dRowPtrB) aclrtFree(dRowPtrB);
    if (dColIndB) aclrtFree(dColIndB);
    if (dValB)    aclrtFree(dValB);
    if (dRowPtrC) aclrtFree(dRowPtrC);
    if (dColIndC) aclrtFree(dColIndC);
    if (dValC)    aclrtFree(dValC);
    if (dBuffer)  aclrtFree(dBuffer);

    return ACL_SPARSE_STATUS_SUCCESS;
}

int main()
{
    AclContext ctx(0);
    auto ret = ctx.Init();
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = aclsparseSpGEMMTest(ctx);
    CHECK_RET(ret == ACL_SPARSE_STATUS_SUCCESS, LOG_PRINT("aclsparseSpGEMMTest failed. ERROR: %d\n", ret); return ret);
    return 0;
}
```

预期输出如下：

```
bufferSize = 11264 bytes
nnzC = 2
C[0]: (col=0, val=1.0)
C[1]: (col=1, val=6.0)
```
