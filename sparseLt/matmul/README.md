# aclsparseLtMatmul

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

- **算子功能**：aclsparseLtMatmul 执行 2:4 结构化稀疏矩阵乘法 `D = alpha * A_pruned * B + beta * C`，其中 A_pruned 是经 `aclsparseLtSpMMAPrune` 剪枝后的稠密存储矩阵（被置零元素以 0 表示）。该算子对标 cuSPARSELt 中的 `cusparseLtMatmul`，采用描述符（Descriptor）+ 算法选择（AlgSelection）+ 执行计划（Plan）的分段式调用模型。
- **目标平台**：<term>Ascend 950PR/Ascend 950DT</term>（arch35 / DAV_3510）。
- **编程模型**：ascendc Cube（Mmad/Fixpipe）矩阵乘硬件单元执行核心 GEMM，Vector 单元执行 epilogue（alpha·acc + beta·C + 类型转换）。splitK=1 时采用 AIC+AIV 融合单 kernel 路径（L0C→UB→GM）；splitK>1 时采用两段式路径（Cube 写 temp → epilogue 归约写 D）。算子异步启动，内部不执行 stream 同步。
- **支持 dtype**：FP32 / FP16 / BF16 / INT8（v2 扩展 BF16/INT8）。INT8 路径使用 `computeType = ACL_SPARSE_COMPUTE_32I`（int32 累加），支持 INT8/INT32 双输出。
- **支持路径**：sparse×dense（2:4 结构化稀疏 × 稠密）+ dense×dense（稠密 × 稠密，v2 新增）。dense×dense 路径跳过剪枝前置，matA 须显式传入。

### 数学原理

#### 问题定义

```
A_pruned = Prune_2:4(A)
D = alpha * A_pruned * B + beta * C
```

| 符号 | 含义 | 维度/类型 |
|------|------|-----------|
| A_pruned | 剪枝后的结构化稀疏矩阵（稠密存储，由 SpMMAPrune 产出） | m×k，FP32/FP16/BF16/INT8 |
| B | 稠密输入矩阵 | k×n，FP32/FP16/BF16/INT8 |
| C | 稠密输入矩阵（累加项） | m×n，与 A/B 同型（INT8 路径下 C 须为 INT8） |
| D | 稠密输出矩阵 | m×n，与 C 同型（INT8 路径下 D 可为 INT8 或 INT32） |
| alpha | 标量缩放因子 | FP32（Host） |
| beta | 标量缩放因子 | FP32（Host） |

> alpha/beta 始终按 Host 端 float 解析（对齐 cuSPARSELt）。当前支持 FP32/FP16/BF16/INT8 输入。

#### 维度推导

逻辑维度 m/n/k 由矩阵 A、B 的描述符维度与 opA/opB 推导（与 cuSPARSELt 一致）：

| opA | m | k |
|-----|---|---|
| NON_TRANSPOSE | A.rows | A.cols |
| TRANSPOSE | A.cols | A.rows |

| opB | k | n |
|-----|---|---|
| NON_TRANSPOSE | B.rows | B.cols |
| TRANSPOSE | B.cols | B.rows |

> 当前 Matmul 执行阶段支持 opA/opB = NON_TRANSPOSE 与 TRANSPOSE。transA 通过剪枝阶段切换剪枝方向实现（`pruneAlongRow = (transA != isRowOrder)`，详见 [prune/README.md](../prune/README.md)）；transB 通过 matmul kernel 将 GM B 声明为 DNExt(k,n) 列主序视图，CopyGM2L1 自动选择 DN2ZN 转置实现。CONJUGATE_TRANSPOSE 不支持。

#### splitK 切分

当 K 维度较大或 m×n tile 数不足以占满多核时，可通过 `aclsparseLtMatmulAlgSetAttribute` 设置 splitK > 1，将 K 维度切分为 splitK 段并行计算，各段结果写入 temp 缓冲区后由 epilogue 归约求和。splitK=1 时走融合路径，无需 temp 缓冲区。splitK 取值范围为 [1, K]。

## 接口说明

aclsparseLtMatmul 的完整调用链包含：库句柄管理、矩阵描述符初始化、Matmul 描述符初始化、算法选择与属性设置、计划初始化、workspace 查询、执行与销毁。以下按调用顺序说明各接口。

### 库句柄管理

#### aclsparseLtInit

初始化 aclsparseLt 库句柄，须在调用任何其他 aclsparseLt 函数之前调用。

```cpp
aclsparseStatus_t aclsparseLtInit(aclsparseLtHandle_t* handle);
```

| 参数名 | 输入/输出 | 参数类型 | 说明 |
|--------|----------|---------|------|
| handle | 输出 | aclsparseLtHandle_t* | 库句柄输出参数，调用前 *handle 须为 nullptr，Host 内存 |

#### aclsparseLtDestroy

释放库句柄占用的全部资源，调用后该句柄不可再使用。

```cpp
aclsparseStatus_t aclsparseLtDestroy(const aclsparseLtHandle_t* handle);
```

| 参数名 | 输入/输出 | 参数类型 | 说明 |
|--------|----------|---------|------|
| handle | 输入 | const aclsparseLtHandle_t* | 指向要销毁的库句柄的 const 指针。指针本身为 nullptr 返回 HANDLE_IS_NULLPTR；指向的句柄值为 nullptr 视为空操作，Host 内存 |

> 所有公共 API 的 handle 参数均为 `aclsparseLtConstHandle_t`（即 `const aclsparseLtHandle_t*`），调用时传 `&handle`。

### 矩阵描述符初始化

#### aclsparseLtStructuredDescriptorInit

初始化结构化（2:4 稀疏）矩阵描述符，用于矩阵 A。

```cpp
aclsparseStatus_t aclsparseLtStructuredDescriptorInit(
    aclsparseLtConstHandle_t handle,
    aclsparseLtMatDescriptor_t* matDescr,
    int64_t rows, int64_t cols, int64_t ld,
    uint32_t alignment, aclDataType valueType, aclsparseOrder_t order,
    aclsparseLtSparsity_t sparsity);
```

| 参数名 | 输入/输出 | 参数类型 | 说明 |
|--------|----------|---------|------|
| handle | 输入 | aclsparseLtConstHandle_t | 库句柄的 const 指针，Host 内存 |
| matDescr | 输出 | aclsparseLtMatDescriptor_t* | 结构化矩阵描述符输出，Host 内存 |
| rows | 输入 | int64_t | 矩阵行数，须 > 0 且 ≤ INT32_MAX，Host 内存 |
| cols | 输入 | int64_t | 矩阵列数，须 > 0 且 ≤ INT32_MAX，Host 内存 |
| ld | 输入 | int64_t | Leading dimension（行间跨度），须 > 0；ROW order 时 ld ≥ cols，COL order 时 ld ≥ rows，Host 内存 |
| alignment | 输入 | uint32_t | 内存对齐字节数，须为 16 的倍数，Host 内存 |
| valueType | 输入 | aclDataType | 矩阵数据类型，支持 ACL_FLOAT / ACL_FLOAT16 / ACL_BF16 / ACL_INT8，Host 内存 |
| order | 输入 | aclsparseOrder_t | 矩阵存储顺序（ROW / COL），仅影响剪枝方向，Host 内存 |
| sparsity | 输入 | aclsparseLtSparsity_t | 稀疏模式，仅支持 ACL_SPARSE_LT_SPARSITY_50_PERCENT，Host 内存 |

> 对齐 cuSPARSELt：描述符初始化时不绑定数据指针，数据在 Matmul/Prune 执行时通过 matA/d_in 等参数显式传入。

#### aclsparseLtDenseDescriptorInit

初始化稠密矩阵描述符，用于矩阵 B/C/D。

```cpp
aclsparseStatus_t aclsparseLtDenseDescriptorInit(
    aclsparseLtConstHandle_t handle,
    aclsparseLtMatDescriptor_t* matDescr,
    int64_t rows, int64_t cols, int64_t ld,
    uint32_t alignment, aclDataType valueType, aclsparseOrder_t order);
```

参数语义同 aclsparseLtStructuredDescriptorInit，区别为无 sparsity 参数。`valueType` 支持 ACL_FLOAT / ACL_FLOAT16 / ACL_BF16 / ACL_INT8 / ACL_INT32（INT32 仅用于 INT8 路径下的 matC/matD 输出描述符）。数据指针在 Matmul 执行时显式传入。

### Matmul 描述符初始化

#### aclsparseLtMatmulDescriptorInit

初始化 Matmul 操作描述符，绑定 A/B/C/D 与操作类型。

```cpp
aclsparseStatus_t aclsparseLtMatmulDescriptorInit(
    aclsparseLtConstHandle_t handle,
    aclsparseLtMatmulDescriptor_t* matmulDescr,
    aclsparseOperation_t opA, aclsparseOperation_t opB,
    const aclsparseLtMatDescriptor_t* matA,
    const aclsparseLtMatDescriptor_t* matB,
    const aclsparseLtMatDescriptor_t* matC,
    const aclsparseLtMatDescriptor_t* matD,
    aclsparseComputeType_t computeType);
```

| 参数名 | 输入/输出 | 参数类型 | 说明 |
|--------|----------|---------|------|
| handle | 输入 | aclsparseLtConstHandle_t | 库句柄的 const 指针，Host 内存 |
| matmulDescr | 输出 | aclsparseLtMatmulDescriptor_t* | Matmul 描述符输出，Host 内存 |
| opA | 输入 | aclsparseOperation_t | 矩阵 A 的操作类型，支持 NON_TRANSPOSE / TRANSPOSE；CONJUGATE_TRANSPOSE 不支持，Host 内存 |
| opB | 输入 | aclsparseOperation_t | 矩阵 B 的操作类型，支持 NON_TRANSPOSE / TRANSPOSE；CONJUGATE_TRANSPOSE 不支持，Host 内存 |
| matA | 输入 | const aclsparseLtMatDescriptor_t* | 结构化稀疏矩阵 A 描述符，Host 内存 |
| matB | 输入 | const aclsparseLtMatDescriptor_t* | 稠密矩阵 B 描述符，Host 内存 |
| matC | 输入 | const aclsparseLtMatDescriptor_t* | 稠密矩阵 C 描述符（累加项），Host 内存 |
| matD | 输入 | const aclsparseLtMatDescriptor_t* | 稠密矩阵 D 描述符（输出），Host 内存 |
| computeType | 输入 | aclsparseComputeType_t | 计算精度类型，独立于存储 dtype。`ACL_SPARSE_COMPUTE_32F` 对应 FP32/FP16/BF16 输入（FP32 累加）；`ACL_SPARSE_COMPUTE_32I` 对应 INT8 输入（int32 累加）。`ACL_SPARSE_COMPUTE_16F` 当前不支持，Host 内存 |

> computeType 与存储 dtype 解耦：`COMPUTE_32F` 要求 A/B 为 FP32/FP16/BF16（L0C 以 float 累加）；`COMPUTE_32I` 要求 A/B 为 INT8（L0C 以 int32_t 累加）。INT8 路径下 matC 须为 INT8，matD 可为 INT8 或 INT32（双输出，详见下方"INT8 双输出"章节）。

### 算法选择与属性设置

#### aclsparseLtMatmulAlgSelectionInit

初始化算法选择描述符，默认 algConfigId=0、splitK=1、searchIterations=5。

```cpp
aclsparseStatus_t aclsparseLtMatmulAlgSelectionInit(
    aclsparseLtConstHandle_t handle,
    aclsparseLtMatmulAlgSelection_t* algSelection,
    const aclsparseLtMatmulDescriptor_t* matmulDescr,
    aclsparseLtMatmulAlg_t alg);
```

| 参数名 | 输入/输出 | 参数类型 | 说明 |
|--------|----------|---------|------|
| handle | 输入 | aclsparseLtConstHandle_t | 库句柄的 const 指针，Host 内存 |
| algSelection | 输出 | aclsparseLtMatmulAlgSelection_t* | 算法选择描述符输出，Host 内存 |
| matmulDescr | 输入 | aclsparseLtConstMatmulDescriptor_t* | Matmul 描述符，Host 内存 |
| alg | 输入 | aclsparseLtMatmulAlg_t | 算法类型，仅支持 ACL_SPARSE_LT_MATMUL_ALG_DEFAULT，Host 内存 |

#### aclsparseLtMatmulAlgSetAttribute

设置算法选择属性。这是配置 Matmul 执行行为的核心接口，用于切换算法配置、splitK 切分等。

```cpp
aclsparseStatus_t aclsparseLtMatmulAlgSetAttribute(
    aclsparseLtConstHandle_t handle,
    aclsparseLtMatmulAlgSelection_t* algSelection,
    aclsparseLtMatmulAlgAttribute_t attr,
    const void* attrValue,
    size_t attrValueSize);
```

| 参数名 | 输入/输出 | 参数类型 | 说明 |
|--------|----------|---------|------|
| handle | 输入 | aclsparseLtConstHandle_t | 库句柄的 const 指针，Host 内存 |
| algSelection | 输入 | aclsparseLtMatmulAlgSelection_t* | 算法选择描述符，Host 内存 |
| attr | 输入 | aclsparseLtMatmulAlgAttribute_t | 要设置的属性枚举，Host 内存 |
| attrValue | 输入 | const void* | 属性值指针，Host 内存 |
| attrValueSize | 输入 | size_t | attrValue 缓冲区大小（字节），Host 内存 |

可设置的属性如下：

| 属性枚举 | 值类型 | 取值范围 | 说明 |
|----------|--------|---------|------|
| ACLSPARSELT_MATMUL_ALG_CONFIG_ID | int32_t | 0 / 1 | 算法配置 ID，影响 cube tiling（baseM/baseN/baseK）。需在 PlanInit 前设置 |
| ACLSPARSELT_MATMUL_SPLIT_K | int32_t | [1, K] | splitK 切分因子。需在 PlanInit 前设置 |
| ACLSPARSELT_MATMUL_SEARCH_ITERATIONS | int32_t | > 0 | 搜索迭代次数（默认 5），当前仅存储不参与执行 |
| ACLSPARSELT_MATMUL_SPLIT_K_MODE | int32_t | 0 / 1 | splitK 模式（ONE_KERNEL / TWO_KERNELS），参与 effectiveSplitK 计算：ONE_KERNEL 时 effectiveSplitK=1（融合路径），TWO_KERNELS 时 effectiveSplitK=splitK（两段式路径） |
| ACLSPARSELT_MATMUL_SPLIT_K_BUFFERS | int32_t | [0, splitK-1] | splitK 缓冲区数，当前仅存储不参与执行 |
| ACLSPARSELT_MATMUL_ALG_CONFIG_MAX_ID | — | 只读 | 不可设置，返回 `ACL_SPARSE_STATUS_NOT_SUPPORTED` |

> 属性设置须在 `aclsparseLtMatmulPlanInit` 之前完成，PlanInit 会读取 algConfigId 与 splitK 计算 tiling。

#### aclsparseLtMatmulAlgGetAttribute

查询算法选择属性当前值。

```cpp
aclsparseStatus_t aclsparseLtMatmulAlgGetAttribute(
    aclsparseLtConstHandle_t handle,
    aclsparseLtConstMatmulAlgSelection_t* algSelection,
    aclsparseLtMatmulAlgAttribute_t attr,
    void* attrValue,
    size_t attrValueSize);
```

参数语义同 AlgSetAttribute，attrValue 为输出。`ACLSPARSELT_MATMUL_ALG_CONFIG_MAX_ID` 查询返回 2（algConfigId 0 和 1 共 2 种配置）。详细文档见 [alg_get_attribute/README.md](../alg_get_attribute/README.md)。

### 计划初始化与 workspace 查询

#### aclsparseLtMatmulPlanInit

初始化执行计划，计算 cube tiling 参数与 workspace 大小并冻结到计划中。PlanInit 会校验描述符维度一致性、dtype 一致性、algConfigId/splitK 取值合法性等。

```cpp
aclsparseStatus_t aclsparseLtMatmulPlanInit(
    aclsparseLtConstHandle_t handle,
    aclsparseLtMatmulPlan_t* plan,
    const aclsparseLtMatmulDescriptor_t* matmulDescr,
    const aclsparseLtMatmulAlgSelection_t* algSelection);
```

| 参数名 | 输入/输出 | 参数类型 | 说明 |
|--------|----------|---------|------|
| handle | 输入 | aclsparseLtConstHandle_t | 库句柄的 const 指针，Host 内存 |
| plan | 输出 | aclsparseLtMatmulPlan_t* | 执行计划输出，Host 内存 |
| matmulDescr | 输入 | aclsparseLtConstMatmulDescriptor_t* | Matmul 描述符，Host 内存 |
| algSelection | 输入 | aclsparseLtConstMatmulAlgSelection_t* | 算法选择描述符，Host 内存 |

#### aclsparseLtMatmulGetWorkspace

查询 Matmul 所需 workspace 大小（字节）。

```cpp
aclsparseStatus_t aclsparseLtMatmulGetWorkspace(
    aclsparseLtConstHandle_t handle,
    aclsparseLtConstMatmulPlan_t* plan, size_t* workspaceSize);
```

| 参数名 | 输入/输出 | 参数类型 | 说明 |
|--------|----------|---------|------|
| handle | 输入 | aclsparseLtConstHandle_t | 库句柄的 const 指针，Host 内存 |
| plan | 输入 | aclsparseLtConstMatmulPlan_t* | 执行计划，Host 内存 |
| workspaceSize | 输出 | size_t* | 所需 workspace 大小（字节），Host 内存 |

> workspace 布局（各区域 64 字节对齐）：`[ tiling区 .. aPruned区 .. temp区 ]`。splitK=1 时不分配 temp 区（融合路径直接 L0C→UB→GM）；splitK>1 时 temp 区大小为 splitK×m×n×4 字节。

### 执行

#### aclsparseLtMatmul

执行结构化稀疏矩阵乘 `D = alpha * A_pruned * B + beta * C`。

```cpp
aclsparseStatus_t aclsparseLtMatmul(
    aclsparseLtConstHandle_t handle,
    aclsparseLtConstMatmulPlan_t* plan,
    const void* alpha,
    const void* matA, const void* matB,
    const void* beta,
    const void* matC, void* matD,
    void* workspace,
    aclrtStream* streams, int32_t numStreams);
```

| 参数名 | 输入/输出 | 参数类型 | 说明 |
|--------|----------|---------|------|
| handle | 输入 | aclsparseLtConstHandle_t | 库句柄的 const 指针，Host 内存 |
| plan | 输入 | aclsparseLtConstMatmulPlan_t* | 执行计划，Host 内存 |
| alpha | 输入 | const void* | 标量 alpha 指针（float），Host 内存。为 nullptr 时按 1.0 处理 |
| matA | 输入 | const void* | 稀疏侧矩阵 A_pruned 的 Device 内存指针，须 16 字节对齐。A-sparse 时为 A_pruned（由 SpMMAPrune 产出）；B-sparse 时为稠密矩阵 A。为 nullptr 时回退使用 workspace 的 aPruned 区（调用方须已通过 SpMMAPrune 填充该区域），Device 内存 |
| matB | 输入 | const void* | 稠密矩阵 B 的 Device 内存指针，须 16 字节对齐，不可为 nullptr。A-sparse 时为稠密矩阵 B；B-sparse 时为 B_pruned（由 SpMMAPrune 产出）。Device 内存 |
| beta | 输入 | const void* | 标量 beta 指针（float），Host 内存。为 nullptr 时按 0.0 处理 |
| matC | 输入 | const void* | 稠密矩阵 C 的 Device 内存指针，须 16 字节对齐，不可为 nullptr，Device 内存 |
| matD | 输出 | void* | 稠密矩阵 D 的 Device 内存指针，须 16 字节对齐，不可为 nullptr，Device 内存 |
| workspace | 输入 | void* | workspace 的 Device 内存指针，须 16 字节对齐，不可为 nullptr，Device 内存 |
| streams | 输入 | aclrtStream* | ACL 流指针数组，算子在此流上异步执行，streams 不可为 nullptr（数组指针须有效），streams[0] 可为 nullptr（表示使用默认流），Host 内存。当前仅使用 streams[0]（多流暂不支持） |
| numStreams | 输入 | int32_t | 流数量，须 > 0，Host 内存 |

> matA 传 nullptr 的回退路径需调用方自行保证 workspace 的 aPruned 区已通过 SpMMAPrune 填充有效数据，否则将读取未初始化的 Device 内存。推荐路径是显式传入 SpMMAPrune 的 d_out 作为 matA。matB/matC/matD 必须显式传入（对齐 cuSPARSELt，不支持描述符回退）。

### 销毁

销毁接口按依赖逆序调用，不级联失效：

```cpp
aclsparseStatus_t aclsparseLtMatmulPlanDestroy(aclsparseLtMatmulPlan_t* plan);
aclsparseStatus_t aclsparseLtMatmulAlgSelectionDestroy(aclsparseLtMatmulAlgSelection_t* algSelection);
aclsparseStatus_t aclsparseLtMatmulDescriptorDestroy(aclsparseLtMatmulDescriptor_t* matmulDescr);
aclsparseStatus_t aclsparseLtMatDescriptorDestroy(aclsparseLtMatDescriptor_t* matDescr);
```

各销毁接口接收二级指针，销毁后置 nullptr。传入 nullptr 或 *ptr 为 nullptr 时返回 SUCCESS（空操作）。

### 返回码

| 返回码 | 说明 |
|--------|------|
| `ACL_SPARSE_STATUS_SUCCESS` | 操作成功 |
| `ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR` | handle 为 nullptr |
| `ACL_SPARSE_STATUS_INVALID_VALUE` | 参数取值非法（描述符/matmulDescr/plan/stream 为 nullptr、维度不匹配、ld 不满足布局约束、数据指针未 16 字节对齐、attrValueSize 不匹配、algConfigId/splitK/searchIterations/splitKBuffers 取值非法、alg 非默认） |
| `ACL_SPARSE_STATUS_NOT_SUPPORTED` | 不支持的参数组合（数据类型非 FP32/FP16/BF16/INT8、computeType 与 dtype 不匹配、CONJUGATE_TRANSPOSE、sparsity 非 50%、维度超过 2097120） |
| `ACL_SPARSE_STATUS_INSUFFICIENT_RESOURCES` | workspace 为 nullptr 或大小不足 |
| `ACL_SPARSE_STATUS_ALLOC_FAILED` | 描述符内存分配失败 |
| `ACL_SPARSE_STATUS_EXECUTION_FAILED` | ACL 运行时调用失败（如 aclrtMemcpy/aclrtMemsetAsync 失败） |

## 支持规格

| 规格项 | 支持值 | 说明 |
|--------|--------|------|
| 数据类型（存储） | ACL_FLOAT（FP32）、ACL_FLOAT16（FP16）、ACL_BF16（BF16）、ACL_INT8（INT8） | matA/matB 值类型须一致；FP8 不支持。INT8 路径下 matC 须为 INT8，matD 可为 INT8 或 INT32 |
| 计算精度（computeType） | ACL_SPARSE_COMPUTE_32F（FP32 累加）、ACL_SPARSE_COMPUTE_32I（INT8 int32 累加） | COMPUTE_32F 对应 FP32/FP16/BF16 输入；COMPUTE_32I 对应 INT8 输入。COMPUTE_16F 不支持 |
| 计算路径 | sparse×dense、dense×dense | sparse×dense：matA 或 matB 为结构化稀疏（2:4），需先剪枝；dense×dense：matA/matB 均为稠密，跳过剪枝前置 |
| opA / opB（执行阶段） | NON_TRANSPOSE / TRANSPOSE | transA 通过剪枝阶段切换剪枝方向实现；transB 通过 DNExt 列主序视图 + CopyGM2L1 自动转置实现。CONJUGATE_TRANSPOSE 不支持 |
| order | ROW / COL | 仅影响剪枝方向，kernel 始终按行主序访问内存 |
| 稀疏模式 | ACL_SPARSE_LT_SPARSITY_50_PERCENT | 仅支持 2:4（sparse×dense 路径） |
| alg | ACL_SPARSE_LT_MATMUL_ALG_DEFAULT | 仅默认算法 |
| algConfigId | 0 / 1 | 影响 cube tiling |
| splitK | [1, K] | splitK=1 融合路径，splitK>1 两段式路径 |
| 维度上限 | m/n/k ≤ 2097120 | 超限返回 INVALID_VALUE |
| K 维度对齐 | 无对齐约束 | kernel 通过 DataCopyPad 和 tail-clearing 处理非对齐 K |
| 内存对齐 | 16 字节倍数 | 所有数据指针与 alignment 参数 |
| alpha / beta | Host 端 float | 仅支持 Host 模式 |
| in-place | 支持 | matC 与 matD 可指向同一 Device 内存 |

## 约束说明

- **数据类型一致性**：
  - FP32/FP16/BF16 路径：matA/matB/matC/matD 的值类型须一致，否则 PlanInit 返回 `ACL_SPARSE_STATUS_INVALID_VALUE`。
  - INT8 路径（`computeType = ACL_SPARSE_COMPUTE_32I`）：matA/matB 须为 INT8；matC 须为 INT8；matD 可为 INT8 或 INT32（C/D 类型不要求一致，D 支持双输出）。
  - 不支持的类型返回 `ACL_SPARSE_STATUS_NOT_SUPPORTED`。FP8 不支持。
- **computeType 与 dtype 匹配**：`ACL_SPARSE_COMPUTE_32F` 要求 A/B 为 FP32/FP16/BF16；`ACL_SPARSE_COMPUTE_32I` 要求 A/B 为 INT8。不匹配返回 `ACL_SPARSE_STATUS_NOT_SUPPORTED`。`ACL_SPARSE_COMPUTE_16F` 当前不支持。
- **计算路径（sparse×dense / dense×dense）**：
  - **sparse×dense**：matA 或 matB 的 sparsity 为 `ACL_SPARSE_LT_SPARSITY_50_PERCENT`（不可同时为稀疏）。需先调用 `aclsparseLtSpMMAPrune` 剪枝，matA 传剪枝结果。
  - **dense×dense**：matA 与 matB 均为稠密（sparsity=NONE）。跳过剪枝前置（无 prune group 校验），matA 必须显式传入（无 workspace 回退，因无 prune 前端填充 aPruned 区）。
- **维度一致性**：A/B/C/D 的物理维度须与由 opA/opB 推导的逻辑 (m, k, n) 匹配。C 与 D 须具有相同的 ld 与 order。维度超限（m/n/k > 2097120）返回 `ACL_SPARSE_STATUS_INVALID_VALUE`。
- **K 维度无对齐约束**：kernel 通过 DataCopyPad 和 tail-clearing 处理非对齐 K，无需 K 为分组大小的倍数。sparse×dense 与 dense×dense 路径均无此约束。
- **leading dimension 约束**：ROW order 时 ld ≥ 物理列数，COL order 时 ld ≥ 物理行数。ld 为 0 返回 `ACL_SPARSE_STATUS_INVALID_VALUE`。
- **内存对齐**：所有 Device 数据指针（matA/matB/matC/matD/workspace）须 16 字节对齐；alignment 参数须为 16 的倍数。
- **transpose 支持**：Matmul 执行阶段支持 opA/opB = NON_TRANSPOSE 与 TRANSPOSE。transA 通过剪枝阶段切换剪枝方向实现（`pruneAlongRow = (transA != isRowOrder)`，剪枝 kernel 读取物理 (k,m) 布局并输出 (m,k) 行主序的 A_pruned，matmul kernel 读取 A_pruned 无需额外处理）；transB 通过 matmul kernel 将 GM B 声明为 DNExt(k,n) 列主序视图（物理 B 为 (n,k) 行主序），CopyGM2L1 自动选择 DN2ZN 转置完成 ND→ZN fractal 转换，下游 L1/L0B 操作与非 transB 路径一致。CONJUGATE_TRANSPOSE 在描述符初始化阶段即被拒绝。
- **order 语义偏差**：与 NVIDIA cuSPARSELt 不同，CANN 实现中 order 仅切换剪枝方向，不改变内存布局。kernel 始终按行主序访问内存。用户矩阵须以行主序存储，传入真实列主序存储的矩阵会产生错误结果（已知偏差）。
- **algConfigId 语义**：algConfigId 影响 cube tiling（baseM/baseN/baseK），不同配置在不同形状下性能表现不同。取值仅 0 或 1，`ACLSPARSELT_MATMUL_ALG_CONFIG_MAX_ID` 查询返回 2。
- **splitK 语义**：splitK 须在 [1, K] 范围内。splitK=1 走融合路径（无 temp 缓冲区）；splitK>1 走两段式路径（Cube 各段写 temp → epilogue 归约写 D），temp 区大小为 splitK×m×n×4 字节。
- **workspace 生命周期**：workspace 须在 Matmul 执行完成（stream 同步）后方可释放。workspace 不可为 nullptr，大小须 ≥ PlanInit 计算值。
- **matA 回退路径**：matA 为 nullptr 时回退使用 workspace 的 aPruned 区（仅 sparse×dense 且 A-sparse 路径），调用方须自行保证该区域已通过 SpMMAPrune 填充有效数据。dense×dense 路径与 B-sparse 路径 matA 不可为 nullptr。推荐显式传入 SpMMAPrune 的 d_out 作为 matA。
- **销毁顺序**：描述符之间存在原始指针依赖（无引用计数），销毁不级联失效。须按 `plan → algSelection → matmulDesc → mat` 逆序销毁，否则产生悬垂指针。handle 在所有描述符销毁后销毁。
- **异步执行**：Matmul 异步启动，内部不执行 `aclrtSynchronizeStream`；调用方如需读取 matD 结果，须自行同步 stream。
- **属性设置时机**：`aclsparseLtMatmulAlgSetAttribute` 须在 `aclsparseLtMatmulPlanInit` 之前调用，PlanInit 读取 algConfigId 与 splitK 计算 tiling。PlanInit 之后修改属性不会生效（需重新初始化计划）。

### INT8 双输出（INT8/INT32）

INT8 输入路径（`computeType = ACL_SPARSE_COMPUTE_32I`，A/B 为 INT8）支持两种输出 dtype，由 matD 描述符的 `valueType` 决定：

- **INT8 输出**（`matD.valueType = ACL_INT8`）：epilogue 将 int32 累加结果饱和截断到 `[-128, 127]` 后以 int8 写出。使用 `Mins`/`Maxs` 钳位 + `SpltCastFp32ToInt8`（`float→half→int8` 两步 Vector Cast，DAV-3510 不支持直接 `float→int8_t` Cast，也不支持 `CAST_SATURATE`）。
- **INT32 输出**（`matD.valueType = ACL_INT32`）：epilogue 将 int32 累加结果直接写出（无 Cast、无饱和截断），保留完整累加精度。alpha=1+beta=0 快速路径使用 Vector `Add<int32_t>` 累加（splitK>1 归约）+ 直接 int32 UB→GM 输出，完全 bypass float 中间转换。

C 矩阵（累加项）在 INT8 路径下须为 INT8（`matC.valueType = ACL_INT8`）。kernel 通过 `GlobalTensor<int8_t>` 读取 C，不支持 C 为 INT32（与 cuSPARSELt `COMPUTE_32I` 语义不完全对齐：cuSPARSELt 允许 C/D 各自为 INT8 或 INT32，但本实现中 C 必须与输入 dtype 一致）。INT8 路径支持 beta≠0（C 以 int8_t 读取后 Cast 到 float 参与 `beta*C` 累加）。

### 已知限制

| 限制项 | 说明 | 后续计划 |
|--------|------|----------|
| INT8 输出 Cast 两步路径 | DAV-3510 不支持直接 `float→int8_t` Cast，使用 `SpltCastFp32ToInt8`（`float→half→int8` 两步 Vector Cast）+ `Mins`/`Maxs` 饱和截断 | 后续版本探索硬件饱和 Cast 指令 |
| INT8 C→float 标量 Cast | INT8 路径 beta≠0 时，C（int8_t）→ float 的 Cast 使用标量 `SpltScalarCastLoop`（DAV-3510 Vector Cast 不支持 `int8_t→float`），性能未向量化 | 后续版本探索 L0C→UB 直出或硬件 Cast 支持 |
| BF16 Cast 已向量化 | BF16 epilogue 使用 `SpltCastFp32ToBf16Vec`/`SpltCastBf16ToFp32Vec`（Vector `Cast<bfloat16_t,float>`），已向量化优化 | — |
| INT8 splitK 累加已向量化 | INT8 splitK>1 epilogue 归约使用 Vector `Add<int32_t>`（替代标量 `GetValue`/`SetValue` 循环），已向量化优化 | — |

## 调用说明

调用流程为分段式（描述符 + 执行计划模式）：

1. **库句柄**：`aclsparseLtInit`。
2. **矩阵描述符**：`aclsparseLtStructuredDescriptorInit`（A）与 `aclsparseLtDenseDescriptorInit`（B/C/D）。
3. **Matmul 描述符**：`aclsparseLtMatmulDescriptorInit`。
4. **算法选择**：`aclsparseLtMatmulAlgSelectionInit` + `aclsparseLtMatmulAlgSetAttribute`（可选，设置 algConfigId/splitK）。
5. **计划**：`aclsparseLtMatmulPlanInit`。
6. **workspace**：`aclsparseLtMatmulGetWorkspace` 查询大小并分配 Device 内存。
7. **剪枝**：`aclsparseLtSpMMAPrune` 将 A 剪枝为 A_pruned（详见 [prune/README.md](../prune/README.md)）。
8. **执行**：`aclsparseLtMatmul` 执行 `D = alpha * A_pruned * B + beta * C`。
9. **销毁**：按逆序销毁所有描述符与计划。

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../docs/zh/develop/compile_and_run_example.md)。

以下示例演示完整的剪枝 + 结构化稀疏矩阵乘流程。计算 `D = 1.0 * A_pruned * B + 0.0 * C`，其中 A 为 m×k（FP32，行主序），B 为 k×n，C/D 为 m×n。

```cpp
#include <cstdio>
#include <vector>

#include "acl/acl.h"
#include "cann_ops_sparseLt.h"

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

int aclsparseLtMatmulTest()
{
    int32_t deviceId = 0;
    CHECK_RET(aclInit(nullptr) == ACL_SUCCESS, return -1);
    CHECK_RET(aclrtSetDevice(deviceId) == ACL_SUCCESS, aclFinalize(); return -1);
    aclrtStream stream = nullptr;
    CHECK_RET(aclrtCreateStream(&stream) == ACL_SUCCESS, aclrtResetDevice(deviceId); aclFinalize(); return -1);

    // 1. 初始化 aclsparseLt 库句柄
    aclsparseLtHandle_t handle = nullptr;
    auto ret = aclsparseLtInit(&handle);
    CHECK_RET(ret == ACL_SPARSE_STATUS_SUCCESS, LOG_PRINT("aclsparseLtInit failed: %d\n", ret); return ret);

    // 2. 准备矩阵维度与 Host 数据 (FP32, 行主序)
    //    k 须为 2 的倍数（FP32 剪枝分组为 2）
    int64_t m = 64, k = 64, n = 128;
    float alpha = 1.0f, beta = 0.0f;
    std::vector<float> hA(static_cast<size_t>(m) * k);
    std::vector<float> hB(static_cast<size_t>(k) * n);
    std::vector<float> hC(static_cast<size_t>(m) * n, 0.0f);
    for (size_t i = 0; i < hA.size(); i++) { hA[i] = static_cast<float>(i % 11) - 5.0f; }
    for (size_t i = 0; i < hB.size(); i++) { hB[i] = static_cast<float>(i % 7) - 3.0f; }

    // 3. 拷贝数据到 Device
    void *dA = AllocAndCopyDevice(hA.data(), static_cast<size_t>(m) * k * sizeof(float));
    void *dB = AllocAndCopyDevice(hB.data(), static_cast<size_t>(k) * n * sizeof(float));
    void *dC = AllocAndCopyDevice(hC.data(), static_cast<size_t>(m) * n * sizeof(float));
    void *dD = nullptr;
    aclrtMalloc(&dD, static_cast<size_t>(m) * n * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
    // A_pruned 缓冲区：由 SpMMAPrune 写入，供 Matmul 读取
    void *dAPruned = nullptr;
    aclrtMalloc(&dAPruned, static_cast<size_t>(m) * k * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);

    // 4. 创建矩阵描述符
    aclsparseLtMatDescriptor_t matA = nullptr;
    aclsparseLtMatDescriptor_t matB = nullptr, matC = nullptr, matD = nullptr;
    aclsparseLtStructuredDescriptorInit(
        &handle, &matA, m, k, k, 16, ACL_FLOAT,
        ACL_SPARSE_ORDER_ROW, ACL_SPARSE_LT_SPARSITY_50_PERCENT);
    aclsparseLtDenseDescriptorInit(&handle, &matB, k, n, n, 16, ACL_FLOAT, ACL_SPARSE_ORDER_ROW);
    aclsparseLtDenseDescriptorInit(&handle, &matC, m, n, n, 16, ACL_FLOAT, ACL_SPARSE_ORDER_ROW);
    aclsparseLtDenseDescriptorInit(&handle, &matD, m, n, n, 16, ACL_FLOAT, ACL_SPARSE_ORDER_ROW);

    // 5. 创建 Matmul 描述符 (opA=N, opB=N, FP32 累加)
    aclsparseLtMatmulDescriptor_t matmulDesc = nullptr;
    ret = aclsparseLtMatmulDescriptorInit(
        &handle, &matmulDesc,
        ACL_SPARSE_OP_NON_TRANSPOSE, ACL_SPARSE_OP_NON_TRANSPOSE,
        &matA, &matB, &matC, &matD, ACL_SPARSE_COMPUTE_32F);
    CHECK_RET(ret == ACL_SPARSE_STATUS_SUCCESS, LOG_PRINT("MatmulDescriptorInit failed: %d\n", ret); return ret);

    // 6. 创建算法选择描述符并设置属性
    aclsparseLtMatmulAlgSelection_t algSel = nullptr;
    ret = aclsparseLtMatmulAlgSelectionInit(&handle, &algSel, &matmulDesc, ACL_SPARSE_LT_MATMUL_ALG_DEFAULT);
    CHECK_RET(ret == ACL_SPARSE_STATUS_SUCCESS, LOG_PRINT("AlgSelectionInit failed: %d\n", ret); return ret);

    //    设置 algConfigId=0 与 splitK=1（默认值，显式设置以演示用法）
    int32_t algConfigId = 0;
    int32_t splitK = 1;
    ret = aclsparseLtMatmulAlgSetAttribute(&handle, &algSel, ACLSPARSELT_MATMUL_ALG_CONFIG_ID,
                                           &algConfigId, sizeof(int32_t));
    CHECK_RET(ret == ACL_SPARSE_STATUS_SUCCESS, LOG_PRINT("AlgSetAttribute(ALG_CONFIG_ID) failed: %d\n", ret); return ret);
    ret = aclsparseLtMatmulAlgSetAttribute(&handle, &algSel, ACLSPARSELT_MATMUL_SPLIT_K,
                                           &splitK, sizeof(int32_t));
    CHECK_RET(ret == ACL_SPARSE_STATUS_SUCCESS, LOG_PRINT("AlgSetAttribute(SPLIT_K) failed: %d\n", ret); return ret);

    // 7. 创建执行计划
    aclsparseLtMatmulPlan_t plan = nullptr;
    ret = aclsparseLtMatmulPlanInit(&handle, &plan, &matmulDesc, &algSel);
    CHECK_RET(ret == ACL_SPARSE_STATUS_SUCCESS, LOG_PRINT("PlanInit failed: %d\n", ret); return ret);

    // 8. 查询 workspace 大小并分配
    //    注意：aclsparseLtMatmulGetWorkspace / aclsparseLtSpMMAPrune / aclsparseLtMatmul
    //    的描述符/计划参数类型为 aclsparseLtConstXxx_t*（即 const struct Xxx**），
    //    而局部变量声明为 aclsparseLtXxx_t（即 struct Xxx*），取地址后为 struct Xxx**。
    //    C++ 不允许 T** 到 const T** 的隐式转换，需通过 const_cast 显式转换。
    size_t workspaceSize = 0;
    ret = aclsparseLtMatmulGetWorkspace(&handle, const_cast<aclsparseLtConstMatmulPlan_t*>(&plan), &workspaceSize);
    CHECK_RET(ret == ACL_SPARSE_STATUS_SUCCESS, LOG_PRINT("GetWorkspaceSize failed: %d\n", ret); return ret);
    LOG_PRINT("workspaceSize = %zu bytes\n", workspaceSize);
    void *dWorkspace = nullptr;
    if (workspaceSize > 0) {
        aclrtMalloc(&dWorkspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    }

    // 9. 剪枝：dA -> dAPruned
    ret = aclsparseLtSpMMAPrune(
        &handle, const_cast<aclsparseLtConstMatmulDescriptor_t*>(&matmulDesc),
        dA, dAPruned, ACLSPARSELT_PRUNE_SPMMA_STRIP, stream);
    CHECK_RET(ret == ACL_SPARSE_STATUS_SUCCESS, LOG_PRINT("SpMMAPrune failed: %d\n", ret); return ret);

    // 10. 执行 Matmul：D = alpha * A_pruned * B + beta * C
    //     matA 传 dAPruned（剪枝结果），workspace 传 dWorkspace
    ret = aclsparseLtMatmul(
        &handle, const_cast<aclsparseLtConstMatmulPlan_t*>(&plan),
        &alpha,
        dAPruned, dB,
        &beta, dC, dD,
        dWorkspace, &stream, 1);
    CHECK_RET(ret == ACL_SPARSE_STATUS_SUCCESS, LOG_PRINT("Matmul failed: %d\n", ret); return ret);

    // 11. 同步等待完成（调用方负责同步，算子内部不同步）
    aclrtSynchronizeStream(stream);

    // 12. 将结果拷贝回 Host
    std::vector<float> hD(static_cast<size_t>(m) * n, 0.0f);
    aclrtMemcpy(hD.data(), static_cast<size_t>(m) * n * sizeof(float),
                dD, static_cast<size_t>(m) * n * sizeof(float),
                ACL_MEMCPY_DEVICE_TO_HOST);
    LOG_PRINT("matmul done: D[0]=%.4f, D[%zu]=%.4f\n", hD[0], hD.size() - 1, hD.back());

    // 13. 清理资源（按依赖逆序销毁：plan -> algSel -> matmulDesc -> mat -> handle）
    aclsparseLtMatmulPlanDestroy(&plan);
    aclsparseLtMatmulAlgSelectionDestroy(&algSel);
    aclsparseLtMatmulDescriptorDestroy(&matmulDesc);
    aclsparseLtMatDescriptorDestroy(&matA);
    aclsparseLtMatDescriptorDestroy(&matB);
    aclsparseLtMatDescriptorDestroy(&matC);
    aclsparseLtMatDescriptorDestroy(&matD);
    aclsparseLtDestroy(&handle);

    aclrtFree(dA);
    aclrtFree(dB);
    aclrtFree(dC);
    aclrtFree(dD);
    aclrtFree(dAPruned);
    if (dWorkspace) { aclrtFree(dWorkspace); }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}

int main()
{
    return aclsparseLtMatmulTest();
}
```

> splitK > 1 用法：将上述示例中 splitK 改为大于 1 的值（须 ≤ k），PlanInit 会自动分配 temp 缓冲区（splitK×m×n×4 字节）用于各段结果归约。workspace 大小会相应增大。

> in-place 用法：将 matC 与 matD 描述符绑定同一 Device 指针（dD 复用 dC），即 `D = alpha * A_pruned * B + beta * C` 原地写入 C 的内存。

## 参考资源

- 对标接口：cuSPARSELt [cusparseLtMatmul](https://docs.nvidia.com/cuda/cusparselt/index.html#cusparseltmatmul)（描述符 + 执行计划模式）。
- 配套算子：[aclsparseLtSpMMAPrune](../prune/README.md)（剪枝，产出 Matmul 的稀疏侧输入 A_pruned）。
