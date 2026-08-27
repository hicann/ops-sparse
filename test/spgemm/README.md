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

# SpGEMM 算子实现

## 概述

ops-sparse 仓库中的 SpGEMM (Sparse General Matrix-Matrix Multiplication) 算子实现了稀疏矩阵与稀疏矩阵的乘法运算，是高性能稀疏矩阵计算中的核心算子之一，广泛应用于图计算、代数多重网格和科学计算。

该算子对标 NVIDIA cuSPARSE `cusparseSpGEMM`，采用经典的**符号 + 数值**两阶段法：符号阶段根据 A/B 的稀疏结构确定输出矩阵 C 的稀疏结构（rowPtr、nnz），数值阶段在确定的结构上执行乘加运算填充 C 的值。通过多阶段 API 支持结构复用（Structure Reuse），当 A/B 结构不变时符号阶段只需执行一次。

## 产品支持情况

| 产品                                                         |  是否支持 |
| :----------------------------------------------------------- |:-------:|
| <term>Ascend 950PR/Ascend 950DT</term>                       |    ✓    |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>       |    ✗    |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>       |    ✗    |

> SpGEMM 当前版本在 Ascend 950PR/950DT 平台交付，源码位于 `src/spgemm/arch35/`，与 SOC 架构映射保持一致。非 `ascend950*` 平台编译时会跳过 `spgemm_test`。

## 目录结构介绍

```txt
src/spgemm/arch35/
├── spgemm_host.cpp       // Host 侧 API 实现（3-stage + 7-interface）与 launch 调度
├── spgemm_csr_mat.cpp    // CSR 矩阵预处理（行重排、分桶、sorted 校验）
├── spgemm_csr_mat.h      // CSR 辅助函数声明
├── spgemm_kernel.cpp     // Kernel 侧 SIMT 计算实现（符号/数值两阶段）
└── spgemm.h              // 内部头文件与 Tiling 定义

test/spgemm/
├── CMakeLists.txt        // 测试编译配置
├── README.md             // 说明文档
└── arch35/
    └── spgemm_test.cpp   // 950 算子调用样例
```

## 算子描述

### 功能

SpGEMM 算子实现了将稀疏矩阵乘以稀疏矩阵的运算。对应的数学表达式为：

$$
C = \alpha \cdot op(A) \cdot op(B) + \beta \cdot C
$$

其中，A、B、C 均为 CSR 格式稀疏矩阵，$\alpha$ 和 $\beta$ 是标量。输出矩阵 C 的稀疏结构（nnz、每行分布）在运算前未知，必须先通过符号阶段确定结构，再通过数值阶段填值——这是 SpGEMM 与 SpMM 的核心差异。

### 存储格式

- **稀疏矩阵 A/B/C**：均采用 CSR (Compressed Sparse Row) 格式，由三个数组组成：
  - `rowPtr`：行偏移数组
  - `colInd`：列索引数组
  - `values`：非零元素值数组
- 索引类型：`int32`（`ACL_SPARSE_INDEX_32I`），zero-based
- 输入 A/B 列索引须 sorted；输出 C 列索引 sorted

### 实现原理

1. **参数校验**：在 `spgemm_host.cpp` 中校验矩阵维度、数据类型一致性、CSR 格式、sorted 性等。
2. **分核调度**：在 `spgemm_csr_mat.cpp` 中计算每行乘积对数作为负载权重，贪心装箱分配到 64 个 AIV 核。
3. **符号阶段**（`spgemm_symbolic_kernel`）：遍历 A.row(i) × B.row(k)，用 bitmask 去重统计每行结构非零数（"宁多不漏"），经 prefix sum 得到 rowPtrC 与 nnzC。
4. **数值阶段**（`spgemm_numeric_kernel`）：按固定遍历顺序（A 行序 × B 行内列序）在 fp32 dense accumulator 中乘加累加，按列号升序写回 colIdxC/valuesC，保证 fp32 bit-wise 确定性。
5. **结构复用**：`matC->activeBuffer == buffer` 时跳过符号阶段，仅执行数值阶段，支持不同 alpha/beta 复用同一稀疏结构。

### 算子规格

- 参数说明：

  <table>
  <tr><td rowspan="1" align="center">算子类型(OpType)</td><td colspan="6" align="center">SpGEMM</td></tr>
  <tr><td rowspan="7" align="center">算子输入</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
  <tr><td align="center">matA</td><td align="center">m × k</td><td align="center">fp32 / fp16 / bf16</td><td align="center">CSR</td></tr>
  <tr><td align="center">matB</td><td align="center">k × n</td><td align="center">同 A</td><td align="center">CSR</td></tr>
  <tr><td align="center">matC</td><td align="center">m × n</td><td align="center">同 A</td><td align="center">CSR</td></tr>
  <tr><td align="center">alpha</td><td align="center">1</td><td align="center">同 computeType</td><td align="center">scalar</td></tr>
  <tr><td align="center">beta</td><td align="center">1</td><td align="center">同 computeType</td><td align="center">scalar</td></tr>
  <tr><td align="center">opA / opB</td><td align="center">-</td><td align="center">-</td><td align="center">仅 NON_TRANSPOSE</td></tr>
  <tr><td rowspan="1" align="center">算子输出</td><td align="center">matC</td><td align="center">m × n</td><td align="center">同 A</td><td align="center">CSR（结构由计算确定）</td></tr>
  <tr><td rowspan="1" align="center">核函数名</td><td colspan="6" align="center">spgemm_symbolic_kernel / spgemm_numeric_kernel</td></tr>
  </table>

- 支持的数据类型组合：

  | A | B | C | computeType | 累加精度 |
  |---|---|---|-------------|---------|
  | fp32 | fp32 | fp32 | fp32 | fp32 |
  | fp16 | fp16 | fp16 | fp16 | fp32（fp32 累加，末尾转回 fp16） |
  | bf16 | bf16 | bf16 | bf16 | fp32（fp32 累加，末尾转回 bf16） |

  > A/B/C/computeType 四者必须一致（同精度）。

- 约束限制：
  - 稀疏矩阵 A/B/C 当前仅支持 CSR 格式
  - `opA`/`opB` 当前仅支持 `ACL_SPARSE_OP_NON_TRANSPOSE`（非转置）
  - 索引类型当前仅支持 `ACL_SPARSE_INDEX_32I`（int32），zero-based
  - 输入 A/B 列索引须 sorted；输出 C 列索引 sorted
  - `beta = 0` 时 C_in 视为空（host 侧 memset valuesC 清零）
  - fp32 结果 bit-wise 确定性（固定遍历顺序 + 单行单线程）
  - 规模受 INT32_MAX 上界约束

### 算法说明

| 算法枚举 | 说明 |
|---------|------|
| `ACL_SPARSE_SPGEMM_ALG_DEFAULT` | 默认算法（= ALG1），推荐使用 |
| `ACL_SPARSE_SPGEMM_ALG1` | 与 DEFAULT 同一实现 |
| `ACL_SPARSE_SPGEMM_ALG2` | 预留枚举，当前返回 NOT_SUPPORTED |
| `ACL_SPARSE_SPGEMM_ALG3` | 预留枚举，当前返回 NOT_SUPPORTED |

### 接口说明

SpGEMM 提供两套 API：

**3-stage 兼容 API**（对齐仓内 SpMM 风格）：

```cpp
aclsparseSpGEMMGetBufferSize(handle, opA, opB, alpha, matA, matB, beta, matC, computeType, alg, &size);
aclsparseSpGEMMPreprocess(handle, opA, opB, alpha, matA, matB, beta, matC, computeType, alg, buffer);
aclsparseSpGEMM(handle, opA, opB, alpha, matA, matB, beta, matC, computeType, alg, buffer);
```

**7-interface 完整 API**（对齐 cuSPARSE 生命周期）：

```cpp
aclsparseSpGEMMCreateDescr(&descr);
aclsparseSpGEMMWorkEstimation(handle, descr, &buffer1Size, ..., buffer1);
aclsparseSpGEMMEstimateMemory(handle, descr, &buffer3Size, matC, computeType, alg, buffer3);
aclsparseSpGEMMCompute(handle, descr, ..., buffer1, buffer2);
aclsparseSpGEMMGetNumProducts(descr, &numProducts);
aclsparseSpGEMMCopy(handle, descr, ..., buffer2);
aclsparseSpGEMMDestroyDescr(descr);
```

### 测试实现

- 测试流程 (`spgemm_test.cpp`)

1. **初始化**：初始化 ACL 环境，设置设备和创建 stream
2. **生成测试数据**：生成随机 CSR 矩阵 A/B
3. **CPU 参考计算**：使用 `CpuSpGEMM` 计算 CSR×CSR→CSR golden 真值（"宁多不漏"，sorted 输出）
4. **设备内存管理**：分配和拷贝数据到设备内存
5. **3-stage API 调用**：GetBufferSize → Preprocess → SpGEMM
6. **结果验证**：结构（rowPtr/colInd 精确匹配）+ 精度（MERE/MARE/ATK 三指标）
7. **确定性验证**：同输入运行多次，bit-wise 比对
8. **清理资源**：释放设备和主机内存，销毁描述符

- 关键代码片段

```cpp
// 创建 CSR 矩阵
aclsparseCreateConstCsr(&matA, m, k, nnzA, dARowOff, dAColInd, dAVals,
    ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_32I,
    ACL_SPARSE_INDEX_BASE_ZERO, ACL_FLOAT);

// 3-stage API 调用
aclsparseSpGEMMGetBufferSize(handle, ACL_SPARSE_OP_NON_TRANSPOSE,
    ACL_SPARSE_OP_NON_TRANSPOSE, &alpha, matA, matB, &beta, matC,
    ACL_FLOAT, ACL_SPARSE_SPGEMM_ALG_DEFAULT, &bufSize);

aclsparseSpGEMMPreprocess(handle, ACL_SPARSE_OP_NON_TRANSPOSE,
    ACL_SPARSE_OP_NON_TRANSPOSE, &alpha, matA, matB, &beta, matC,
    ACL_FLOAT, ACL_SPARSE_SPGEMM_ALG_DEFAULT, dBuf);

aclsparseSpGEMM(handle, ACL_SPARSE_OP_NON_TRANSPOSE,
    ACL_SPARSE_OP_NON_TRANSPOSE, &alpha, matA, matB, &beta, matC,
    ACL_FLOAT, ACL_SPARSE_SPGEMM_ALG_DEFAULT, dBuf);
```

## 编译运行

### 配置环境变量

请根据当前环境上 CANN 开发套件包的安装方式，选择对应配置环境变量的命令。

- 默认路径，root 用户安装 CANN 软件包

  ```bash
  source /usr/local/Ascend/cann/set_env.sh
  ```

- 指定路径 install_path，安装 CANN 软件包

  ```bash
  source ${install_path}/cann/set_env.sh
  ```

### 样例执行

```bash
bash build.sh --ops=spgemm --soc=ascend950 --run
```

执行结果如下，说明精度对比成功：

```bash
========== Results Summary ==========
  TC-01-small         : PASS
  TC-02-empty         : PASS
  TC-03-alpha-beta    : PASS
  TC-04-sparse        : PASS
  DET-01-determinism  : PASS
  TC-05-api-complete  : PASS
  EFF-efficiency      : PASS
  PREC-precision      : PASS
  VD-value-domain     : PASS
  STAB-stability      : PASS
  INFNAN-consistency  : PASS
  FIX-verification    : PASS

  Overall: PASS
```

> **说明**：200 组泛化测试和压力测试中，n > 128 的部分用例可能因 arch35 SIMT GM cache coherence 限制而失败，不影响主测试的 PASS 状态。

## 接口说明

完整 API 说明参见 `include/cann_ops_sparse.h` 中 `aclsparseSpGEMM*` 系列声明。
