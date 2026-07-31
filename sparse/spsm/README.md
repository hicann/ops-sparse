# aclsparseSpSM

## 产品支持情况

| 产品 | 是否支持 |
| :----------------------------------------- | :------:|
| <term>Ascend 950PR/Ascend 950DT</term> | √ |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> | × |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> | × |
| <term>Atlas 200I/500 A2 推理产品</term> | × |
| <term>Atlas 推理系列产品</term> | × |
| <term>Atlas 训练系列产品</term> | × |

## 功能说明

- **算子功能**：aclsparseSpSM（Sparse Triangular Solve with Multiple Right-Hand Sides）用于求解稀疏三角线性方程组 `op(A)·X = α·B`。其中 A 为 CSR 格式的稀疏三角方阵（m×m），B 为稠密右端项矩阵（m×n），X 为稠密解矩阵（m×n，可与 B 原地操作），α 为标量系数。该算子对标 cuSPARSE Generic API 中的 `cusparseSpSM`，采用三阶段（BufferSize/Analysis/Solve）执行模型与 Descriptor 描述符模式。
- **目标平台**：<term>Ascend 950PR/Ascend 950DT</term>（arch35 / DAV_3510）。
- **编程模型**：ascendc 低阶 vector API（TQue/LocalTensor/DataCopyPad/Muls/Sub 等），不使用矩阵乘硬件（Cube/Tensor Core）。

### 数学原理

#### 问题定义

求解稀疏三角线性方程组：

```
op(A) · X = α · B
```

| 符号 | 含义 | 维度/类型 |
|------|------|-----------|
| A | 稀疏三角方阵（CSR） | m×m，FP32 |
| B | 稠密右端项矩阵 | m×n（n=nrhs），FP32 |
| X | 稠密解矩阵（可与 B in-place） | m×n，FP32 |
| α | 标量系数 | FP32（Host） |
| op(A) | A（opA=N）或 Aᵀ（opA=T） | — |
| fillMode | A 参与计算的三角部分 | LOWER / UPPER |
| diagType | 对角线类型 | UNIT（A[i,i]≡1.0，不存储）/ NON_UNIT（实际值） |

> cuSPARSE 原始公式为 `op(A)·C = α·op(B)`，含 opB。本期接口签名保留 opB 参数对齐 cuSPARSE，但首期仅实现 opB=NON_TRANSPOSE。

#### 行更新通式

对每一行 i，沿求解方向遍历，已解行构成依赖集 `deps(i)`：

```
             α·B[i, 0:n] − Σ_{j∈deps(i)} A[i,j] · X[j, 0:n]
X[i, 0:n] = ───────────────────────────────────────────────      （NON_UNIT 通式）
                               A[i,i]

X[i, 0:n] = α·B[i, 0:n] − Σ_{j∈deps(i)} A[i,j] · X[j, 0:n]      （UNIT 特例，省除法）
```

关键点：

- **除法作用域**：`/A[i,i]` 作用于整个分子（α·B[i,:] − Σ ...），非仅求和项。
- **UNIT 是特例**：A[i,i]≡1.0 时除法省略，设计以 NON_UNIT 通式为基础、UNIT 作 fast path。
- **除数恒为 A[i,i]**：四种 (opA, fillMode) 模式下都是 A[i,i]（转置不改变对角线，Aᵀ[i,i]=A[i,i]）。

#### 四种 (opA, fillMode) 依赖模式

deps(i) 由 (opA, fillMode) 决定：

| opA | fillMode | 代入方式 | 遍历方向 | deps(i) |
|-----|----------|---------|---------|---------|
| N | LOWER | 前向 | i=0→m−1 | {j<i \| A[i,j]≠0} |
| N | UPPER | 后向 | i=m−1→0 | {j>i \| A[i,j]≠0} |
| T | LOWER | 后向 | i=m−1→0 | {j>i \| A[j,i]≠0} |
| T | UPPER | 前向 | i=0→m−1 | {j<i \| A[j,i]≠0} |

T 模式实现含义：opA=T 时数学上读 Aᵀ，CSR 按行存储无法高效按列访问 A（=按行访问 Aᵀ），故实现需先物理转置为 CSC（=Aᵀ 的 CSR）再按 N 模式处理。转置后三角性翻转，`effectiveFillMode = swap(fillMode)`。

#### level scheduling 并行结构

level 定义：`level[i] = max( level[dep] for dep ∈ deps(i) ) + 1`，无依赖则 level=0。同 level 行无依赖（依赖行都在更低 level），level 间严格串行；level 数 L = max(level[i])+1，L ≤ m。

```
for level k = 0 .. L-1:                       // level 间串行
    对 R_k 中所有行并行做标量乘向量+归约        // level 内并行
    SyncAll / 栅栏                             // 等本 level 完成，X 回写 GM
```

并行度收益随矩阵结构变化：带状矩阵、块对角矩阵、随机稀疏矩阵的 level 数远小于 m，行级并行度显著；最坏情况（严格下三角稠密矩阵）退化为串行，无回退风险。

#### 除法与边界处理

- **NON_UNIT 除法**：行更新最后一步，标量 ÷ 行向量（A[i,i] 作用于长度 n 行向量每个元素），实现上转为乘倒数 `Muls(acc, acc, 1/A[i,i])`。
- **UNIT diag**：省略除法（A[i,i]≡1.0）。
- **奇异矩阵检测**：Analysis 阶段检测对角线零元（显式零或缺失对角线项），若检测到零对角元则返回错误码（对齐 cuSPARSE zeroPivot 机制）。
- **数值稳定性**：不做特殊稳定化处理（对齐 cuSPARSE 行为），稳定性取决于 A 的条件数。
- **in-place 正确性**：由 level scheduling 保证依赖顺序——行 i 自身 B[i] 在覆写前已消费，依赖行 j 的 B[j] 已被 X[j] 覆写且要读的正是 X[j]。

## 接口说明

aclsparseSpSM 采用三阶段执行模型（BufferSize → Analysis → Solve）配合 SpSM 描述符（spsmDescr）跨阶段共享，对标 cuSPARSE cusparseSpSM。

### aclsparseSpSMCreateDescr

#### 函数原型

```cpp
aclsparseStatus_t aclsparseSpSMCreateDescr(aclsparseSpSMDescr_t *spsmDescr);
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 |
|--------|----------|---------|------|
| spsmDescr | 输出 | aclsparseSpSMDescr_t* | SpSM 描述符句柄指针，调用前 `*spsmDescr` 须为 nullptr，Host 内存 |

#### 约束说明

- spsmDescr 不可为 nullptr，否则返回 `ACL_SPARSE_STATUS_INVALID_VALUE`。
- 该描述符在 BufferSize/Analysis/Solve 三阶段间共享，须在调用三阶段接口前创建。

### aclsparseSpSMDestroyDescr

#### 函数原型

```cpp
aclsparseStatus_t aclsparseSpSMDestroyDescr(aclsparseSpSMDescr_t spsmDescr);
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 |
|--------|----------|---------|------|
| spsmDescr | 输入 | aclsparseSpSMDescr_t | 待销毁的 SpSM 描述符句柄，Host 内存 |

#### 约束说明

- spsmDescr 为 nullptr 时直接返回 `ACL_SPARSE_STATUS_SUCCESS`，不报错。
- 销毁后不应再使用该句柄。

### aclsparseSpSMBufferSize

查询 SpSM 所需 workspace 大小（字节）。

#### 函数原型

```cpp
aclsparseStatus_t aclsparseSpSMBufferSize(
    aclsparseHandle_t handle, aclsparseOperation_t opA, aclsparseOperation_t opB,
    const void *alpha, aclsparseConstSpMatDescr_t matA,
    aclsparseConstDnMatDescr_t matB, aclsparseDnMatDescr_t matC,
    aclDataType computeType, aclsparseSpSMAlg_t alg,
    aclsparseSpSMDescr_t spsmDescr, size_t *bufferSize);
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 |
|--------|----------|---------|------|
| handle | 输入 | aclsparseHandle_t | ops-sparse 库上下文句柄，携带 stream，Host 内存 |
| opA | 输入 | aclsparseOperation_t | 稀疏矩阵 A 的操作类型，支持 `ACL_SPARSE_OP_NON_TRANSPOSE` 或 `ACL_SPARSE_OP_TRANSPOSE`，Host 内存 |
| opB | 输入 | aclsparseOperation_t | 稠密矩阵 B 的操作类型，仅支持 `ACL_SPARSE_OP_NON_TRANSPOSE`，Host 内存 |
| alpha | 输入 | const void* | 标量 alpha 指针，类型须与 computeType 匹配。仅支持 HOST 模式（`ACL_SPARSE_POINTER_MODE_HOST`），DEVICE 模式返回 `ACL_SPARSE_STATUS_NOT_SUPPORTED` |
| matA | 输入 | aclsparseConstSpMatDescr_t | 稀疏三角矩阵 A 的描述符，仅支持 CSR 格式，m×m 方阵，Host 内存 |
| matB | 输入 | aclsparseConstDnMatDescr_t | 稠密矩阵 B 的描述符（m×n，右端项），Host 内存 |
| matC | 输出 | aclsparseDnMatDescr_t | 稠密矩阵 C 的描述符（m×n，解 X，可与 B 原地），Host 内存 |
| computeType | 输入 | aclDataType | 计算精度类型，仅支持 `ACL_FLOAT`（FP32），Host 内存 |
| alg | 输入 | aclsparseSpSMAlg_t | 算法类型，仅支持 `ACL_SPARSE_SPSM_ALG_DEFAULT`，Host 内存 |
| spsmDescr | 输入 | aclsparseSpSMDescr_t | SpSM 描述符，Host 内存 |
| bufferSize | 输出 | size_t* | 输出所需 workspace 大小（字节），Host 内存 |

### aclsparseSpSMAnalysis

SpSM 分析阶段：opA=T 时 host 侧执行 CSR→CSC 转置；host CPU 计算 level scheduling 拓扑分层（level 数组、行索引表、对角线值、奇异标志）；缓存 tiling 到描述符；绑定 active buffer 供后续 Solve 复用。

#### 函数原型

```cpp
aclsparseStatus_t aclsparseSpSMAnalysis(
    aclsparseHandle_t handle, aclsparseOperation_t opA, aclsparseOperation_t opB,
    const void *alpha, aclsparseConstSpMatDescr_t matA,
    aclsparseConstDnMatDescr_t matB, aclsparseDnMatDescr_t matC,
    aclDataType computeType, aclsparseSpSMAlg_t alg,
    aclsparseSpSMDescr_t spsmDescr, void *buffer);
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 |
|--------|----------|---------|------|
| handle | 输入 | aclsparseHandle_t | ops-sparse 库上下文句柄，携带 stream，Host 内存 |
| opA | 输入 | aclsparseOperation_t | 稀疏矩阵 A 的操作类型，支持 `ACL_SPARSE_OP_NON_TRANSPOSE` 或 `ACL_SPARSE_OP_TRANSPOSE`，Host 内存 |
| opB | 输入 | aclsparseOperation_t | 稠密矩阵 B 的操作类型，仅支持 `ACL_SPARSE_OP_NON_TRANSPOSE`，Host 内存 |
| alpha | 输入 | const void* | 标量 alpha 指针，类型须与 computeType 匹配。仅支持 HOST 模式（`ACL_SPARSE_POINTER_MODE_HOST`），DEVICE 模式返回 `ACL_SPARSE_STATUS_NOT_SUPPORTED` |
| matA | 输入 | aclsparseConstSpMatDescr_t | 稀疏三角矩阵 A 的描述符，仅支持 CSR 格式，m×m 方阵，Host 内存 |
| matB | 输入 | aclsparseConstDnMatDescr_t | 稠密矩阵 B 的描述符（m×n，右端项），Host 内存 |
| matC | 输出 | aclsparseDnMatDescr_t | 稠密矩阵 C 的描述符（m×n，解 X，可与 B 原地），Host 内存 |
| computeType | 输入 | aclDataType | 计算精度类型，仅支持 `ACL_FLOAT`（FP32），Host 内存 |
| alg | 输入 | aclsparseSpSMAlg_t | 算法类型，仅支持 `ACL_SPARSE_SPSM_ALG_DEFAULT`，Host 内存 |
| spsmDescr | 输入/输出 | aclsparseSpSMDescr_t | SpSM 描述符，分析完成后标记 analyzed 并绑定 active buffer，Host 内存 |
| buffer | 输入 | void* | workspace 缓冲区（由 BufferSize 返回的大小分配），Device 内存 |

### aclsparseSpSM

SpSM 求解阶段：异步执行三角求解 `op(A)·X = α·B`，复用 Analysis 阶段绑定的 active buffer 与 level scheduling 元数据。

#### 函数原型

```cpp
aclsparseStatus_t aclsparseSpSM(
    aclsparseHandle_t handle, aclsparseOperation_t opA, aclsparseOperation_t opB,
    const void *alpha, aclsparseConstSpMatDescr_t matA,
    aclsparseConstDnMatDescr_t matB, aclsparseDnMatDescr_t matC,
    aclDataType computeType, aclsparseSpSMAlg_t alg,
    aclsparseSpSMDescr_t spsmDescr);
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 |
|--------|----------|---------|------|
| handle | 输入 | aclsparseHandle_t | ops-sparse 库上下文句柄，携带 stream，Host 内存 |
| opA | 输入 | aclsparseOperation_t | 稀疏矩阵 A 的操作类型，须与 Analysis 阶段一致，支持 `ACL_SPARSE_OP_NON_TRANSPOSE` 或 `ACL_SPARSE_OP_TRANSPOSE`，Host 内存 |
| opB | 输入 | aclsparseOperation_t | 稠密矩阵 B 的操作类型，须与 Analysis 阶段一致，仅支持 `ACL_SPARSE_OP_NON_TRANSPOSE`，Host 内存 |
| alpha | 输入 | const void* | 标量 alpha 指针，类型须与 computeType 匹配。仅支持 HOST 模式（`ACL_SPARSE_POINTER_MODE_HOST`），DEVICE 模式返回 `ACL_SPARSE_STATUS_NOT_SUPPORTED` |
| matA | 输入 | aclsparseConstSpMatDescr_t | 稀疏三角矩阵 A 的描述符，仅支持 CSR 格式，m×m 方阵，Host 内存 |
| matB | 输入 | aclsparseConstDnMatDescr_t | 稠密矩阵 B 的描述符（m×n，右端项），Host 内存 |
| matC | 输出 | aclsparseDnMatDescr_t | 稠密矩阵 C 的描述符（m×n，解 X，可与 B 原地），Host 内存 |
| computeType | 输入 | aclDataType | 计算精度类型，仅支持 `ACL_FLOAT`（FP32），Host 内存 |
| alg | 输入 | aclsparseSpSMAlg_t | 算法类型，仅支持 `ACL_SPARSE_SPSM_ALG_DEFAULT`，Host 内存 |
| spsmDescr | 输入 | aclsparseSpSMDescr_t | SpSM 描述符，须已完成 Analysis 阶段，Host 内存 |

### 矩阵属性接口

SpSM 通过稀疏矩阵描述符的属性接口 `aclsparseSpMatSetAttribute` 设置三角矩阵的 fillMode 与 diagType，须在调用 BufferSize/Analysis 前设置。

```cpp
aclsparseStatus_t aclsparseSpMatSetAttribute(aclsparseSpMatDescr_t spMatDescr,
                                             aclsparseSpMatAttribute_t attribute,
                                             const void *data, size_t dataSize);
aclsparseStatus_t aclsparseSpMatGetAttribute(aclsparseConstSpMatDescr_t spMatDescr,
                                             aclsparseSpMatAttribute_t attribute,
                                             void *data, size_t dataSize);
```

- **attribute=`ACL_SPARSE_SPMAT_FILL_MODE`**：data 指向 `aclsparseFillMode_t`，dataSize 须为 `sizeof(aclsparseFillMode_t)`，值为 `ACL_SPARSE_FILL_MODE_LOWER`（下三角）或 `ACL_SPARSE_FILL_MODE_UPPER`（上三角）。
- **attribute=`ACL_SPARSE_SPMAT_DIAG_TYPE`**：data 指向 `aclsparseDiagType_t`，dataSize 须为 `sizeof(aclsparseDiagType_t)`，值为 `ACL_SPARSE_DIAG_TYPE_UNIT`（单位对角线，A[i,i]≡1.0 不存储）或 `ACL_SPARSE_DIAG_TYPE_NON_UNIT`（实际对角线值，须存储于 CSR）。

### 返回码

| 返回码 | 说明 |
|--------|------|
| `ACL_SPARSE_STATUS_SUCCESS` | 操作成功 |
| `ACL_SPARSE_STATUS_NOT_INITIALIZED` | Solve 阶段在 Analysis 之前调用，或库未初始化 |
| `ACL_SPARSE_STATUS_INVALID_VALUE` | 参数取值非法（如 spsmDescr/alpha/matA/matB/matC 为 nullptr、维度不匹配、fillMode 非法、opA/opB 跨阶段不一致、ld 不满足布局约束） |
| `ACL_SPARSE_STATUS_ARCH_MISMATCH` | 芯片架构不匹配（算子所需特性在当前设备架构上不可用） |
| `ACL_SPARSE_STATUS_NOT_SUPPORTED` | 不支持的参数组合（如 opA=CONJUGATE_TRANSPOSE、opB=TRANSPOSE、computeType 非 FP32、diagType 非法、alg 非默认、索引类型非 32I、维度超过 INT32_MAX、PointerMode=DEVICE、**奇异矩阵检测到零对角元**） |
| `ACL_SPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED` | matA 非 CSR 格式 |
| `ACL_SPARSE_STATUS_INSUFFICIENT_RESOURCES` | workspace/buffer 为 nullptr |
| `ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR` | handle 为 nullptr |
| `ACL_SPARSE_STATUS_EXECUTION_FAILED` | ACL 运行时调用失败（如 aclrtMemcpy 等调用失败） |
| `ACL_SPARSE_STATUS_INTERNAL_ERROR` | 算子内部错误 |
| `ACL_SPARSE_STATUS_ALLOC_FAILED` | 描述符内存分配失败 |

## 支持规格

| 规格项 | 支持值 | 说明 |
|--------|--------|------|
| 数据类型（computeType） | `ACL_FLOAT`（FP32） | matA/matB/matC 值类型与 computeType 须均为 FP32；FP16/BF16/FP64/Complex 不支持 |
| 稀疏格式 | CSR | COO/CSC 不支持 |
| opA | `ACL_SPARSE_OP_NON_TRANSPOSE`、`ACL_SPARSE_OP_TRANSPOSE` | CONJUGATE_TRANSPOSE 不支持（实数下等价 T） |
| opB | `ACL_SPARSE_OP_NON_TRANSPOSE` | 接口签名保留 opB 对齐 cuSPARSE；首期仅支持 NON_TRANSPOSE，传 T/CONJUGATE 返回 NOT_SUPPORTED |
| fillMode | `ACL_SPARSE_FILL_MODE_LOWER`、`ACL_SPARSE_FILL_MODE_UPPER` | 均支持 |
| diagType | `ACL_SPARSE_DIAG_TYPE_UNIT`、`ACL_SPARSE_DIAG_TYPE_NON_UNIT` | 均支持；UNIT 时 A[i,i]≡1.0 不存储、省除法；NON_UNIT 时对角线值须存储于 CSR |
| order（B/C 布局） | `ACL_SPARSE_ORDER_ROW`、`ACL_SPARSE_ORDER_COL` | 均支持 |
| indexBase | `ACL_SPARSE_INDEX_BASE_ZERO`、`ACL_SPARSE_INDEX_BASE_ONE` | 均支持；ONE 内部归一化为 ZERO |
| 索引类型 | `ACL_SPARSE_INDEX_32I` | 行偏移与列索引类型须均为 32I；64I 不支持 |
| in-place | 支持 | matB.values 与 matC.values 可指向同一 Device 内存，正确性由 level scheduling 保证（详见"约束说明"in-place 布局） |
| alg | `ACL_SPARSE_SPSM_ALG_DEFAULT` | 仅默认算法 |
| 矩阵规模 | m/n/nnz ≤ INT32_MAX | 受 GM workspace 容量约束；m/n/nnz 超过 INT32_MAX 返回 NOT_SUPPORTED |

## 约束说明

- **opB 限制**：首期仅支持 `ACL_SPARSE_OP_NON_TRANSPOSE`。接口签名保留 opB 参数对齐 cuSPARSE 便于后续扩展，传 `ACL_SPARSE_OP_TRANSPOSE` 或 `ACL_SPARSE_OP_CONJUGATE_TRANSPOSE` 返回 `ACL_SPARSE_STATUS_NOT_SUPPORTED`。
- **computeType 限制**：仅支持 `ACL_FLOAT`（FP32）。matA/matB/matC 的值类型与 computeType 须均为 FP32，其他类型组合返回 `ACL_SPARSE_STATUS_NOT_SUPPORTED`。
- **PointerMode 限制**：alpha 仅支持 HOST 模式（`ACL_SPARSE_POINTER_MODE_HOST`），DEVICE 模式（`ACL_SPARSE_POINTER_MODE_DEVICE`）返回 `ACL_SPARSE_STATUS_NOT_SUPPORTED`。
- **索引类型限制**：CSR 的行偏移类型（csrRowOffsetsType）与列索引类型（csrColIndType）须均为 `ACL_SPARSE_INDEX_32I`，否则返回 `ACL_SPARSE_STATUS_NOT_SUPPORTED`。
- **矩阵须方阵**：matA 的行数须等于列数（rows==cols），且 matB/matC 的行数须等于 matA 的行数，matB/matC 的列数须相等，否则返回 `ACL_SPARSE_STATUS_INVALID_VALUE`。
- **leading dimension 约束**：matB/matC 的 leading dimension（ldb/ldc）须满足布局约束——行主序（`ACL_SPARSE_ORDER_ROW`）`ld >= cols`，列主序（`ACL_SPARSE_ORDER_COL`）`ld >= rows`，否则返回 `ACL_SPARSE_STATUS_INVALID_VALUE`。
- **奇异矩阵检测**：diagType=NON_UNIT 时，Analysis 阶段扫描对角线，若检测到零对角元（显式零或 CSR 中缺失对角线项）则判定为奇异矩阵，返回 `ACL_SPARSE_STATUS_NOT_SUPPORTED`。diagType=UNIT 时不检测（对角线隐式为 1.0）。
- **矩阵规模上限**：m/n/nnz 受 INT32_MAX 与 GM workspace 容量约束，超限返回 `ACL_SPARSE_STATUS_NOT_SUPPORTED`。level 数组与 tiling 元数据分别驻留 GM 与 host 描述符，无 UB 容量上限约束。
- **三阶段一致性**：Analysis 与 Solve 之间 matA/matB/matC/buffer 不可变；Solve 阶段传入的 opA/opB 须与 Analysis 阶段一致，否则返回 `ACL_SPARSE_STATUS_INVALID_VALUE`。须先调用 Analysis 完成 analyzed 标记，否则 Solve 返回 `ACL_SPARSE_STATUS_NOT_INITIALIZED`。
- **buffer 生命周期**：externalBuffer（Analysis 阶段传入的 workspace）生命周期须延续至 Solve 完成后方可释放。
- **异步执行**：Solve 阶段 kernel 异步启动，算子内部禁止 `aclrtSynchronizeStream`；调用方如需读取结果，须自行同步 stream。
- **unsorted indices**：支持 CSR 列索引未排序，求解正确性不受影响（加法交换律）。
- **in-place 布局**：任意 orderB/orderC 组合均支持 in-place（matB.values 与 matC.values 指向同一 Device 内存时，orderB!=orderC 时 B 与 X 在 Solve 阶段分离至不同缓冲区，天然安全）。建议 orderB==orderC 以减少 transpose 开销。
- **属性设置时机**：fillMode 与 diagType 须在调用 BufferSize/Analysis 前通过 `aclsparseSpMatSetAttribute` 设置（attribute 分别取 `ACL_SPARSE_SPMAT_FILL_MODE` 与 `ACL_SPARSE_SPMAT_DIAG_TYPE`）。

## 调用说明

调用流程为三阶段法（Generic API Descriptor 模式）：

1. **BufferSize**：查询所需 workspace 大小，分配 Device 内存。
2. **Analysis**：opA=T 时 host 侧执行 CSR→CSC 转置；host CPU 计算 level scheduling 拓扑分层与对角线值；检测奇异矩阵；缓存 tiling 到描述符；绑定 active buffer。
3. **Solve**：异步执行三角求解，复用 Analysis 绑定的 active buffer。

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

// 辅助：分配 Device 内存并拷贝 Host 数据
static void* AllocAndCopyDevice(const void *hostPtr, size_t sizeBytes)
{
    void *dPtr = nullptr;
    aclrtMalloc(&dPtr, sizeBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    if (hostPtr != nullptr && sizeBytes > 0) {
        aclrtMemcpy(dPtr, sizeBytes, hostPtr, sizeBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    }
    return dPtr;
}

int aclsparseSpSMTest(AclContext& ctx)
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

    // 2. 设置 PointerMode
    sparseRet = aclsparseSetPointerMode(static_cast<aclsparseHandle_t>(handlePtr.get()), ACL_SPARSE_POINTER_MODE_HOST);
    CHECK_RET(sparseRet == ACL_SPARSE_STATUS_SUCCESS, LOG_PRINT("aclsparseSetPointerMode failed. ERROR: %d\n", sparseRet);
              return sparseRet);

    // 3. 准备 Host 端 CSR 数据
    //    A (3x3, 下三角, NON_UNIT, nnz=6):     B (3x2):
    //    [2.0  0.0  0.0]      [2.0  4.0]
    //    [1.0  3.0  0.0]      [4.0  8.0]
    //    [4.0  5.0  6.0]      [6.0 12.0]
    //
    //    CSR (含对角线, NON_UNIT):
    //      rowOff = [0, 1, 3, 6]
    //      colInd = [0, 0, 1, 0, 1, 2]
    //      values = [2.0, 1.0, 3.0, 4.0, 5.0, 6.0]
    //
    //    求解 A * X = 1.0 * B (下三角, 前向替换, NON_UNIT):
    //      X[0] = B[0] / A[0,0]                         = [1.0,  2.0]
    //      X[1] = (B[1] - A[1,0]*X[0]) / A[1,1]         = [1.0,  2.0]
    //      X[2] = (B[2] - A[2,0]*X[0] - A[2,1]*X[1]) / A[2,2] = [-0.5, -1.0]
    int64_t m = 3, n = 2;
    int64_t nnzA = 6;
    float hAlpha = 1.0f;

    std::vector<int> hRowPtrA = {0, 1, 3, 6};
    std::vector<int> hColIndA = {0, 0, 1, 0, 1, 2};
    std::vector<float> hValA  = {2.0f, 1.0f, 3.0f, 4.0f, 5.0f, 6.0f};

    // B: 行主序 3x2
    int64_t ldb = n, ldc = n;
    aclsparseOrder_t orderB = ACL_SPARSE_ORDER_ROW;
    aclsparseOrder_t orderC = ACL_SPARSE_ORDER_ROW;
    std::vector<float> hB(static_cast<size_t>(m) * n, 0.0f);
    hB[0 * n + 0] = 2.0f; hB[0 * n + 1] = 4.0f;
    hB[1 * n + 0] = 4.0f; hB[1 * n + 1] = 8.0f;
    hB[2 * n + 0] = 6.0f; hB[2 * n + 1] = 12.0f;

    std::vector<float> hC(static_cast<size_t>(m) * n, 0.0f);

    // 4. 拷贝数据到 Device
    void *dRowPtrA = AllocAndCopyDevice(hRowPtrA.data(), (m + 1) * sizeof(int));
    void *dColIndA = AllocAndCopyDevice(hColIndA.data(), nnzA * sizeof(int));
    void *dValA    = AllocAndCopyDevice(hValA.data(),    nnzA * sizeof(float));
    void *dB       = AllocAndCopyDevice(hB.data(),       static_cast<size_t>(m) * n * sizeof(float));
    void *dC       = AllocAndCopyDevice(hC.data(),       static_cast<size_t>(m) * n * sizeof(float));

    // 5. 创建稀疏矩阵描述符并设置三角属性
    aclsparseSpMatDescr_t matA = nullptr;
    sparseRet = aclsparseCreateCsr(&matA, m, m, nnzA, dRowPtrA, dColIndA, dValA,
                                   ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_32I,
                                   ACL_SPARSE_INDEX_BASE_ZERO, ACL_FLOAT);
    CHECK_RET(sparseRet == ACL_SPARSE_STATUS_SUCCESS, LOG_PRINT("aclsparseCreateCsr failed. ERROR: %d\n", sparseRet);
              return sparseRet);

    // 5.1 设置三角矩阵属性：fillMode=LOWER, diagType=NON_UNIT
    aclsparseFillMode_t fillMode = ACL_SPARSE_FILL_MODE_LOWER;
    aclsparseDiagType_t diagType = ACL_SPARSE_DIAG_TYPE_NON_UNIT;
    sparseRet = aclsparseSpMatSetAttribute(matA, ACL_SPARSE_SPMAT_FILL_MODE, &fillMode, sizeof(fillMode));
    CHECK_RET(sparseRet == ACL_SPARSE_STATUS_SUCCESS, LOG_PRINT("aclsparseSpMatSetAttribute(FILL_MODE) failed. ERROR: %d\n", sparseRet);
              return sparseRet);
    sparseRet = aclsparseSpMatSetAttribute(matA, ACL_SPARSE_SPMAT_DIAG_TYPE, &diagType, sizeof(diagType));
    CHECK_RET(sparseRet == ACL_SPARSE_STATUS_SUCCESS, LOG_PRINT("aclsparseSpMatSetAttribute(DIAG_TYPE) failed. ERROR: %d\n", sparseRet);
              return sparseRet);

    aclsparseDnMatDescr_t matB = nullptr, matC = nullptr;
    sparseRet = aclsparseCreateDnMat(&matB, m, n, ldb, dB, ACL_FLOAT, orderB);
    CHECK_RET(sparseRet == ACL_SPARSE_STATUS_SUCCESS, LOG_PRINT("aclsparseCreateDnMat B failed. ERROR: %d\n", sparseRet);
              return sparseRet);
    sparseRet = aclsparseCreateDnMat(&matC, m, n, ldc, dC, ACL_FLOAT, orderC);
    CHECK_RET(sparseRet == ACL_SPARSE_STATUS_SUCCESS, LOG_PRINT("aclsparseCreateDnMat C failed. ERROR: %d\n", sparseRet);
              return sparseRet);

    // 6. 创建 SpSM 描述符
    aclsparseSpSMDescr_t spsmDescr = nullptr;
    sparseRet = aclsparseSpSMCreateDescr(&spsmDescr);
    CHECK_RET(sparseRet == ACL_SPARSE_STATUS_SUCCESS, LOG_PRINT("aclsparseSpSMCreateDescr failed. ERROR: %d\n", sparseRet);
              return sparseRet);

    // 7. Step 1 — BufferSize
    size_t bufferSize = 0;
    sparseRet = aclsparseSpSMBufferSize(
        static_cast<aclsparseHandle_t>(handlePtr.get()),
        ACL_SPARSE_OP_NON_TRANSPOSE, ACL_SPARSE_OP_NON_TRANSPOSE,
        &hAlpha, matA, matB, matC, ACL_FLOAT,
        ACL_SPARSE_SPSM_ALG_DEFAULT, spsmDescr, &bufferSize);
    CHECK_RET(sparseRet == ACL_SPARSE_STATUS_SUCCESS, LOG_PRINT("SpSMBufferSize failed. ERROR: %d\n", sparseRet);
              return sparseRet);
    LOG_PRINT("bufferSize = %zu bytes\n", bufferSize);

    void *dBuffer = nullptr;
    auto aclRet = aclrtMalloc(&dBuffer, bufferSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc for buffer failed. ERROR: %d\n", aclRet); return aclRet);

    // 8. Step 2 — Analysis (写 tiling + CSR→CSC(若 opA=T) + level scheduling + 奇异检测)
    sparseRet = aclsparseSpSMAnalysis(
        static_cast<aclsparseHandle_t>(handlePtr.get()),
        ACL_SPARSE_OP_NON_TRANSPOSE, ACL_SPARSE_OP_NON_TRANSPOSE,
        &hAlpha, matA, matB, matC, ACL_FLOAT,
        ACL_SPARSE_SPSM_ALG_DEFAULT, spsmDescr, dBuffer);
    CHECK_RET(sparseRet == ACL_SPARSE_STATUS_SUCCESS, LOG_PRINT("SpSMAnalysis failed. ERROR: %d\n", sparseRet);
              return sparseRet);

    // 9. Step 3 — Solve (异步启动, 算子内部不做 stream 同步)
    sparseRet = aclsparseSpSM(
        static_cast<aclsparseHandle_t>(handlePtr.get()),
        ACL_SPARSE_OP_NON_TRANSPOSE, ACL_SPARSE_OP_NON_TRANSPOSE,
        &hAlpha, matA, matB, matC, ACL_FLOAT,
        ACL_SPARSE_SPSM_ALG_DEFAULT, spsmDescr);
    CHECK_RET(sparseRet == ACL_SPARSE_STATUS_SUCCESS, LOG_PRINT("SpSM failed. ERROR: %d\n", sparseRet);
              return sparseRet);

    // 10. 同步等待计算完成（由调用方负责, 算子内部禁止同步）
    aclRet = aclrtSynchronizeStream(stream);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", aclRet); return aclRet);

    // 11. 将结果拷贝回 Host 并打印
    aclRet = aclrtMemcpy(hC.data(), static_cast<size_t>(m) * n * sizeof(float),
                         dC, static_cast<size_t>(m) * n * sizeof(float),
                         ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", aclRet);
              return aclRet);

    for (int64_t i = 0; i < m; i++) {
        LOG_PRINT("C[%lld] = %.1f, %.1f\n", static_cast<long long>(i), hC[i * n + 0], hC[i * n + 1]);
    }

    // 12. 清理资源
    aclsparseSpSMDestroyDescr(spsmDescr);
    aclsparseDestroySpMat(matA);
    aclsparseDestroyDnMat(matB);
    aclsparseDestroyDnMat(matC);
    if (dRowPtrA) aclrtFree(dRowPtrA);
    if (dColIndA) aclrtFree(dColIndA);
    if (dValA)    aclrtFree(dValA);
    if (dB)       aclrtFree(dB);
    if (dC)       aclrtFree(dC);
    if (dBuffer)  aclrtFree(dBuffer);

    return ACL_SPARSE_STATUS_SUCCESS;
}

int main()
{
    AclContext ctx(0);
    auto ret = ctx.Init();
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = aclsparseSpSMTest(ctx);
    CHECK_RET(ret == ACL_SPARSE_STATUS_SUCCESS, LOG_PRINT("aclsparseSpSMTest failed. ERROR: %d\n", ret); return ret);
    return 0;
}
```

预期输出如下：

```
bufferSize = 448 bytes
C[0] = 1.0, 2.0
C[1] = 1.0, 2.0
C[2] = -0.5, -1.0
```

> in-place 用法：将上述示例中 `dC` 改为复用 `dB`（即 `aclsparseCreateDnMat(&matC, m, n, ldc, dB, ...)`，matB 与 matC 共用同一 Device 指针），即可原地求解，X 逐行覆写 B。in-place 支持范围与布局约束详见"约束说明"in-place 布局。

## 实现说明

### level scheduling（Analysis + Solve）

- **Analysis 阶段**：在 host CPU 上执行拓扑分层（Gauss-Seidel 式顺序分层），计算 level 数组 `level[i] = max(level[dep])+1`，构建按 level 分桶的行索引表（levelRowPtr/levelRowIdx），提取对角线值（NON_UNIT）并检测奇异矩阵。opA=T 时先在 host 侧执行 CSR→CSC 转置，Solve 阶段复用转置结果。tiling 数据以 by-value 形式缓存在描述符中，不落盘 workspace。
- **Solve 阶段**：单 kernel 多 pass，按 level 0→L−1 顺序逐 pass 处理；每个 level 内的行跨 AI Core 均分并行求解，level 间通过 SyncAll 栅栏同步，保证依赖行 X[j,:] 已回写 GM 后才被下一 level 读取。
- **双正交并行维度**：level 内行切分（多核）与 n 维列块切分（kChunk，UB 内向量化）正交叠加。n=1（SpSV 退化）时向量化消失，level 行并行是唯一有效并行来源。

### ascendc 低阶 vector API

算子遍历 CSR 非零元做标量乘向量 + 顺序累减（Vector 单元），不使用矩阵乘硬件（Cube/Tensor Core）。核心 API 映射：

- `DataCopyPad`：GM↔UB 批量搬运（非对齐安全），用于加载 colInd/vals/B/X 及回写 X。
- `Muls`：标量×向量（α·B[i,:]、A[i,j]·X[j,:]）。
- `Sub`：向量减法（逐依赖行累减）。
- 除法转乘倒数：`Muls(acc, acc, 1/A[i,i])`（NON_UNIT），1 条指令。
- `SyncAll`：level 间核间同步。
- `PipeBarrier<PIPE_V>`：V→V 流水依赖同步。

### COL/ROW 转置 kernel

B/C 支持 `ACL_SPARSE_ORDER_COL`（列主序）布局，Solve kernel 仅处理 ROW 连续搬运路径，故 orderB=COL 或 orderC=COL 时由独立的 vector transpose kernel 在 Solve 前后做布局转换：

- **transpose-in**（Solve 前）：orderB=COL 时将 B(COL) 转入 denseBuf(ROW)，direction=0（COL→ROW），跨步读 src[k*ld+i]、连续写 dst[i*n+k]。
- **transpose-out**（Solve 后）：orderC=COL 时将 denseBuf(ROW) 转出至 C(COL)，direction=1（ROW→COL），连续读 src[i*n+k]、跨步写 dst[k*ld+i]。

转置 kernel 多核按行切分，每核处理若干行；用 `DataCopyPad` 跨步搬运实现（GM 侧 stride 单位为字节，支持任意 ld），分段处理（kTransSegMax）同时约束 blockCount（≤4095）与 UB 占用。orderB/orderC 均 ROW 时跳过转置、Solve 直接读写原 GM。

### opA=T CSR→CSC 转置

opA=T 时数学上读 Aᵀ，CSR 按行存储无法高效按列访问 A，故 Analysis 阶段在 host 侧先物理转置为 CSC（=Aᵀ 的 CSR），Solve 阶段按 N 模式处理转置结果。转置后三角性翻转（A 下三角 → Aᵀ 上三角），`effectiveFillMode = swap(fillMode)`。除数恒为 A[i,i]（转置不改变对角线）。

## 参考资源

- 对标接口：cuSPARSE [cusparseSpSM](https://docs.nvidia.com/cuda/cusparse/index.html#cusparsespsm)（Generic API，三阶段执行）。
