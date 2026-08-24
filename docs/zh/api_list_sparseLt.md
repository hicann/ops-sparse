# 稀疏结构化算子接口（aclsparseLt）

## 使用说明

为方便调用 2:4 结构化稀疏相关算子，提供一套基于 C 的 API（以 aclsparseLt 为前缀），主要用于结构化稀疏矩阵乘法及其前置剪枝、压缩等场景。

调用 aclsparseLt 算子 API 时，需引用依赖的头文件和库文件。

- 依赖的头文件：`cann_ops_sparseLt.h`
- 依赖的库文件：`libops_sparseLt.so`

## 接口列表

| 接口名 | 说明 |
|--------|------|
| [aclsparseLtInit](#aclsparseltinit) | 初始化 aclsparseLt 库句柄 |
| [aclsparseLtDestroy](#aclsparseltdestroy) | 销毁 aclsparseLt 库句柄 |
| [aclsparseLtGetErrorName](#aclsparseltgeterrorname) | 获取状态码对应的枚举名字符串 |
| [aclsparseLtGetErrorString](#aclsparseltgeterrorstring) | 获取状态码对应的描述性字符串 |
| [aclsparseLtGetVersion](#aclsparseltgetversion) | 获取 aclsparseLt 库版本号（**暂未支持**） |
| [aclsparseLtGetProperty](#aclsparseltgetproperty) | 获取 aclsparseLt 库属性信息（**暂未支持**） |
| [aclsparseLtDenseDescriptorInit](#aclsparseltdensedescriptorinit) | 初始化稠密矩阵描述符 |
| [aclsparseLtStructuredDescriptorInit](#aclsparseltstructureddescriptorinit) | 初始化结构化稀疏矩阵描述符（2:4） |
| [aclsparseLtMatDescriptorDestroy](#aclsparseltmatdescriptordestroy) | 销毁矩阵描述符 |
| [aclsparseLtMatDescSetAttribute](#aclsparseltmatdescsetattribute) | 设置矩阵描述符属性（**暂未支持**） |
| [aclsparseLtMatDescGetAttribute](#aclsparseltmatdescgetattribute) | 获取矩阵描述符属性（**暂未支持**） |
| [aclsparseLtMatmulDescriptorInit](#aclsparseltmatmuldescriptorinit) | 初始化矩阵乘法描述符 |
| [aclsparseLtMatmulDescriptorDestroy](#aclsparseltmatmuldescriptordestroy) | 销毁矩阵乘法描述符 |
| [aclsparseLtMatmulDescSetAttribute](#aclsparseltmatmuldescsetattribute) | 设置 matmul 描述符属性（**暂未支持**） |
| [aclsparseLtMatmulDescGetAttribute](#aclsparseltmatmuldescgetattribute) | 获取 matmul 描述符属性（**暂未支持**） |
| [aclsparseLtMatmulAlgSelectionInit](#aclsparseltmatmulalgselectioninit) | 初始化 matmul 算法选择描述符 |
| [aclsparseLtMatmulAlgSelectionDestroy](#aclsparseltmatmulalgselectiondestroy) | 销毁 matmul 算法选择描述符 |
| [aclsparseLtMatmulAlgSetAttribute](#aclsparseltmatmulalgsetattribute) | 设置算法选择描述符属性（**暂未支持**） |
| [aclsparseLtMatmulAlgGetAttribute](#aclsparseltmatmulalggetattribute) | 获取算法选择描述符属性（**暂未支持**） |
| [aclsparseLtMatmulPlanInit](#aclsparseltmatmulplaninit) | 初始化 matmul 执行计划 |
| [aclsparseLtMatmulPlanDestroy](#aclsparseltmatmulplandestroy) | 销毁 matmul 执行计划 |
| [aclsparseLtMatmulGetWorkspace](#aclsparseltmatmulgetworkspace) | 获取 matmul 所需 workspace 大小（**暂未支持**） |
| [aclsparseLtMatmul](#aclsparseltmatmul) | 执行结构化稀疏矩阵乘法（**暂未支持**） |
| [aclsparseLtMatmulSearch](#aclsparseltmatmulsearch) | 搜索最优 matmul 算法（**暂未支持**） |
| [aclsparseLtSpMMAPrune](#aclsparseltspmmaprune) | 对稠密矩阵执行 2:4 结构化稀疏剪枝 |
| [aclsparseLtSpMMAPruneCheck](#aclsparseltspmmaprunecheck) | 校验稠密矩阵是否已满足 2:4 结构化稀疏约束（**暂未支持**） |
| [aclsparseLtSpMMACompressedSize](#aclsparseltspmmacompressedsize) | 查询压缩后矩阵所需存储大小（**暂未支持**） |
| [aclsparseLtSpMMACompress](#aclsparseltspmmacompress) | 将 2:4 稀疏矩阵压缩为紧凑存储（**暂未支持**） |

## 接口详情

### aclsparseLtInit

```c
aclsparseStatus_t aclsparseLtInit(aclsparseLtHandle_t* handle);
```

**功能**：初始化 aclsparseLt 库句柄。在主机端分配轻量级硬件资源，创建 aclsparseLt 库上下文。必须在调用任何其他 aclsparseLt 函数之前调用。库上下文绑定到当前 NPU 设备，多设备使用需为每个设备创建独立 handle。

**参数说明**：

- `handle`（OUT）：HOST，aclsparseLt 库句柄输出参数。

**返回值**：

- `ACL_SPARSE_STATUS_SUCCESS`：成功
- `ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR`：handle 指针为空
- `ACL_SPARSE_STATUS_INVALID_VALUE`：`*handle` 非空（防止覆盖已有句柄导致泄漏）
- `ACL_SPARSE_STATUS_ALLOC_FAILED`：内存分配失败

---

### aclsparseLtDestroy

```c
aclsparseStatus_t aclsparseLtDestroy(const aclsparseLtHandle_t* handle);
```

**功能**：释放 aclsparseLt 库句柄占用的全部资源。与特定 handle 关联的最后一次调用，调用后该 handle 不可再使用。

**参数说明**：

- `handle`（IN）：HOST，指向要销毁的 aclsparseLt 库句柄的指针。

**返回值**：

- `ACL_SPARSE_STATUS_SUCCESS`：成功
- `ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR`：handle 指针为空

---

### aclsparseLtGetErrorName

```c
const char* aclsparseLtGetErrorName(aclsparseStatus_t status);
```

**功能**：返回状态码对应的枚举名字符串（如 `"ACL_SPARSE_STATUS_SUCCESS"`）。未识别的状态码返回 `"unrecognized error code"`。

**参数说明**：

- `status`（IN）：HOST，要转换的状态码。

**返回值**：

- `const char*`：指向枚举名字符串的指针。

---

### aclsparseLtGetErrorString

```c
const char* aclsparseLtGetErrorString(aclsparseStatus_t status);
```

**功能**：返回状态码对应的描述性字符串。未识别的状态码返回 `"unrecognized error code"`。

**参数说明**：

- `status`（IN）：HOST，要转换的状态码。

**返回值**：

- `const char*`：指向描述性字符串的指针。

---

### aclsparseLtGetVersion

> **支持状态**：暂未支持。当前版本尚未实现。

```c
aclsparseStatus_t aclsparseLtGetVersion(
    aclsparseLtConstHandle_t handle,
    int* version);
```

**功能**：获取 aclsparseLt 库的版本号。

**参数说明**：

- `handle`（IN）：HOST，aclsparseLt 库句柄。
- `version`（OUT）：HOST，返回库的版本号。

**返回值**：

- `ACL_SPARSE_STATUS_SUCCESS`：成功
- `ACL_SPARSE_STATUS_INVALID_VALUE`：version 为空指针

---

### aclsparseLtGetProperty

> **支持状态**：暂未支持。当前版本尚未实现。

```c
aclsparseStatus_t aclsparseLtGetProperty(
    libraryPropertyType propertyType,
    int* value);
```

**功能**：获取 aclsparseLt 库的属性信息（如主版本号、次版本号、补丁版本号）。

**参数说明**：

- `propertyType`（IN）：HOST，请求的属性类型（如 `MAJOR_VERSION` / `MINOR_VERSION` / `PATCH_LEVEL`）。
- `value`（OUT）：HOST，返回请求属性的值。

**返回值**：

- `ACL_SPARSE_STATUS_SUCCESS`：成功
- `ACL_SPARSE_STATUS_INVALID_VALUE`：value 为空指针、propertyType 非法

---

### aclsparseLtDenseDescriptorInit

```c
aclsparseStatus_t aclsparseLtDenseDescriptorInit(
    const aclsparseLtHandle_t*       handle,
    aclsparseLtMatDescriptor_t*      matDescr,
    int64_t                          rows,
    int64_t                          cols,
    int64_t                          ld,
    uint32_t                         alignment,
    aclDataType                      valueType,
    aclsparseOrder_t                 order);
```

**功能**：初始化稠密矩阵描述符。在主机端分配并填充矩阵描述符结构体，记录稠密矩阵的形状、布局、数据类型等信息。用于描述 matmul 中的稠密矩阵（或剪枝场景下的占位稠密矩阵）。

**参数说明**：

- `handle`（IN）：HOST，aclsparseLt 库句柄。
- `matDescr`（OUT）：HOST，矩阵描述符输出。
- `rows`（IN）：HOST，行数。
- `cols`（IN）：HOST，列数。
- `ld`（IN）：HOST，leading dimension。
- `alignment`（IN）：HOST，内存对齐字节数。
- `valueType`（IN）：HOST，矩阵数据存储类型。
- `order`（IN）：HOST，内存布局（`ACL_SPARSE_ORDER_ROW` / `ACL_SPARSE_ORDER_COL`）。

**返回值**：

- `ACL_SPARSE_STATUS_SUCCESS`：成功
- `ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR`：handle 指针为空
- `ACL_SPARSE_STATUS_INVALID_VALUE`：matDescr 为空、维度/对齐/布局非法
- `ACL_SPARSE_STATUS_NOT_SUPPORTED`：valueType 不在支持列表内
- `ACL_SPARSE_STATUS_ALLOC_FAILED`：内存分配失败

---

### aclsparseLtStructuredDescriptorInit

```c
aclsparseStatus_t aclsparseLtStructuredDescriptorInit(
    const aclsparseLtHandle_t*       handle,
    aclsparseLtMatDescriptor_t*      matDescr,
    int64_t                          rows,
    int64_t                          cols,
    int64_t                          ld,
    uint32_t                         alignment,
    aclDataType                      valueType,
    aclsparseOrder_t                 order,
    aclsparseLtSparsity_t            sparsity);
```

**功能**：初始化结构化稀疏矩阵描述符（2:4 结构化稀疏）。与 `aclsparseLtDenseDescriptorInit` 类似，但描述符标记为 structured，对齐倍数采用 structured 档位，并记录稀疏模式。用于描述 matmul 中的稀疏矩阵。

**参数说明**：

- `handle`（IN）：HOST，aclsparseLt 库句柄。
- `matDescr`（OUT）：HOST，矩阵描述符输出。
- `rows`（IN）：HOST，行数。
- `cols`（IN）：HOST，列数。
- `ld`（IN）：HOST，leading dimension。
- `alignment`（IN）：HOST，内存对齐字节数。
- `valueType`（IN）：HOST，矩阵数据存储类型。
- `order`（IN）：HOST，内存布局（`ACL_SPARSE_ORDER_ROW` / `ACL_SPARSE_ORDER_COL`）。
- `sparsity`（IN）：HOST，稀疏模式（`ACL_SPARSE_LT_SPARSITY_50_PERCENT`）。

**返回值**：

- `ACL_SPARSE_STATUS_SUCCESS`：成功
- `ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR`：handle 指针为空
- `ACL_SPARSE_STATUS_INVALID_VALUE`：matDescr 为空、维度/对齐/布局/sparsity 非法
- `ACL_SPARSE_STATUS_NOT_SUPPORTED`：valueType 不在支持列表内
- `ACL_SPARSE_STATUS_ALLOC_FAILED`：内存分配失败

---

### aclsparseLtMatDescriptorDestroy

```c
aclsparseStatus_t aclsparseLtMatDescriptorDestroy(aclsparseLtMatDescriptor_t* matDescr);
```

**功能**：销毁矩阵描述符，释放其占用的主机内存。销毁成功后 `*matDescr` 被置为 nullptr。

**参数说明**：

- `matDescr`（IN/OUT）：HOST，指向要销毁的矩阵描述符的指针。

**返回值**：

- `ACL_SPARSE_STATUS_SUCCESS`：成功
- `ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR`：matDescr 指针为空

---

### aclsparseLtMatDescSetAttribute

> **支持状态**：暂未支持。当前版本尚未实现。

```c
aclsparseStatus_t aclsparseLtMatDescSetAttribute(
    aclsparseLtConstHandle_t handle,
    aclsparseLtMatDescriptor_t* matDescr,
    aclsparseLtMatDescAttribute_t attribute,
    const void* data,
    size_t dataSize);
```

**功能**：设置矩阵描述符的指定属性（如批数、批步长等）。

**参数说明**：

- `handle`（IN）：HOST，aclsparseLt 库句柄。
- `matDescr`（IN/OUT）：HOST，矩阵描述符。
- `attribute`（IN）：HOST，要设置的矩阵描述符属性。
- `data`（IN）：HOST，指向属性值的指针。
- `dataSize`（IN）：HOST，属性值字节数，用于校验。

**返回值**：

- `ACL_SPARSE_STATUS_SUCCESS`：成功
- `ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR`：handle 为空指针
- `ACL_SPARSE_STATUS_INVALID_VALUE`：matDescr 为空、attribute 非法、data 为空、dataSize 与属性不匹配
- `ACL_SPARSE_STATUS_NOT_SUPPORTED`：该属性暂不支持

---

### aclsparseLtMatDescGetAttribute

> **支持状态**：暂未支持。当前版本尚未实现。

```c
aclsparseStatus_t aclsparseLtMatDescGetAttribute(
    aclsparseLtConstHandle_t handle,
    aclsparseLtConstMatDescriptor_t* matDescr,
    aclsparseLtMatDescAttribute_t attribute,
    void* data,
    size_t dataSize);
```

**功能**：获取矩阵描述符的指定属性（如批数、批步长等）。

**参数说明**：

- `handle`（IN）：HOST，aclsparseLt 库句柄。
- `matDescr`（IN）：HOST，矩阵描述符。
- `attribute`（IN）：HOST，要获取的矩阵描述符属性。
- `data`（OUT）：HOST，返回属性值的内存地址。
- `dataSize`（IN）：HOST，属性值字节数，用于校验。

**返回值**：

- `ACL_SPARSE_STATUS_SUCCESS`：成功
- `ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR`：handle 为空指针
- `ACL_SPARSE_STATUS_INVALID_VALUE`：matDescr 为空、attribute 非法、data 为空、dataSize 与属性不匹配
- `ACL_SPARSE_STATUS_NOT_SUPPORTED`：该属性暂不支持

---

### aclsparseLtMatmulDescriptorInit

```c
aclsparseStatus_t aclsparseLtMatmulDescriptorInit(
    const aclsparseLtHandle_t*             handle,
    aclsparseLtMatmulDescriptor_t*         matmulDescr,
    aclsparseOperation_t                   opA,
    aclsparseOperation_t                   opB,
    const aclsparseLtMatDescriptor_t*      matA,
    const aclsparseLtMatDescriptor_t*      matB,
    const aclsparseLtMatDescriptor_t*      matC,
    const aclsparseLtMatDescriptor_t*      matD,
    aclsparseComputeType_t                 computeType);
```

**功能**：初始化矩阵乘法描述符。在主机端分配并填充 matmul 描述符结构体，记录 matmul 运算的完整信息：操作类型、四个矩阵描述符引用、计算精度。matA/matB/matC/matD 为非所有权引用。

**参数说明**：

- `handle`（IN）：HOST，aclsparseLt 库句柄。
- `matmulDescr`（OUT）：HOST，matmul 描述符输出。
- `opA`（IN）：HOST，作用于矩阵 A 的操作（`ACL_SPARSE_OP_NON_TRANSPOSE` / `ACL_SPARSE_OP_TRANSPOSE`）。
- `opB`（IN）：HOST，作用于矩阵 B 的操作。
- `matA`（IN）：HOST，矩阵 A 描述符（structured 或 dense）。
- `matB`（IN）：HOST，矩阵 B 描述符（structured 或 dense）。
- `matC`（IN）：HOST，矩阵 C 描述符（dense）。
- `matD`（IN）：HOST，矩阵 D 描述符（dense）。
- `computeType`（IN）：HOST，计算精度（`ACL_SPARSE_COMPUTE_16F` / `ACL_SPARSE_COMPUTE_32F` / `ACL_SPARSE_COMPUTE_32I`）。

**返回值**：

- `ACL_SPARSE_STATUS_SUCCESS`：成功
- `ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR`：handle 指针为空
- `ACL_SPARSE_STATUS_INVALID_VALUE`：matmulDescr 为空、opA/opB 非法、结构化位置/C/D 一致性/操作布局组合/维度约束违反
- `ACL_SPARSE_STATUS_NOT_SUPPORTED`：computeType 不在支持列表内
- `ACL_SPARSE_STATUS_ALLOC_FAILED`：内存分配失败

---

### aclsparseLtMatmulDescriptorDestroy

```c
aclsparseStatus_t aclsparseLtMatmulDescriptorDestroy(aclsparseLtMatmulDescriptor_t* matmulDescr);
```

**功能**：销毁 matmul 描述符，释放其占用的主机内存。本接口不销毁 matA/matB/matC/matD 描述符，用户须自行调用 `aclsparseLtMatDescriptorDestroy` 释放它们。销毁成功后 `*matmulDescr` 被置为 nullptr。

**参数说明**：

- `matmulDescr`（IN/OUT）：HOST，指向要销毁的 matmul 描述符的指针。

**返回值**：

- `ACL_SPARSE_STATUS_SUCCESS`：成功
- `ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR`：matmulDescr 指针为空

---

### aclsparseLtMatmulDescSetAttribute

> **支持状态**：暂未支持。当前版本尚未实现。

```c
aclsparseStatus_t aclsparseLtMatmulDescSetAttribute(
    aclsparseLtConstHandle_t handle,
    aclsparseLtMatmulDescriptor_t* matmulDescr,
    aclsparseLtMatmulDescAttribute_t attribute,
    const void* data,
    size_t dataSize);
```

**功能**：设置 matmul 描述符的指定属性（如激活函数、偏置等）。

**参数说明**：

- `handle`（IN）：HOST，aclsparseLt 库句柄。
- `matmulDescr`（IN/OUT）：HOST，matmul 描述符。
- `attribute`（IN）：HOST，要设置的 matmul 描述符属性。
- `data`（IN）：HOST，指向属性值的指针。
- `dataSize`（IN）：HOST，属性值字节数，用于校验。

**返回值**：

- `ACL_SPARSE_STATUS_SUCCESS`：成功
- `ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR`：handle 为空指针
- `ACL_SPARSE_STATUS_INVALID_VALUE`：matmulDescr 为空、attribute 非法、data 为空、dataSize 与属性不匹配
- `ACL_SPARSE_STATUS_NOT_SUPPORTED`：该属性暂不支持

---

### aclsparseLtMatmulDescGetAttribute

> **支持状态**：暂未支持。当前版本尚未实现。

```c
aclsparseStatus_t aclsparseLtMatmulDescGetAttribute(
    aclsparseLtConstHandle_t handle,
    aclsparseLtConstMatmulDescriptor_t* matmulDescr,
    aclsparseLtMatmulDescAttribute_t attribute,
    void* data,
    size_t dataSize);
```

**功能**：获取 matmul 描述符的指定属性（如激活函数、偏置等）。

**参数说明**：

- `handle`（IN）：HOST，aclsparseLt 库句柄。
- `matmulDescr`（IN）：HOST，matmul 描述符。
- `attribute`（IN）：HOST，要获取的 matmul 描述符属性。
- `data`（OUT）：HOST，返回属性值的内存地址。
- `dataSize`（IN）：HOST，属性值字节数，用于校验。

**返回值**：

- `ACL_SPARSE_STATUS_SUCCESS`：成功
- `ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR`：handle 为空指针
- `ACL_SPARSE_STATUS_INVALID_VALUE`：matmulDescr 为空、attribute 非法、data 为空、dataSize 与属性不匹配
- `ACL_SPARSE_STATUS_NOT_SUPPORTED`：该属性暂不支持

---

### aclsparseLtMatmulAlgSelectionInit

```c
aclsparseStatus_t aclsparseLtMatmulAlgSelectionInit(
    const aclsparseLtHandle_t*             handle,
    aclsparseLtMatmulAlgSelection_t*       algSelection,
    const aclsparseLtMatmulDescriptor_t*   matmulDescr,
    aclsparseLtMatmulAlg_t                 alg);
```

**功能**：初始化 matmul 算法选择描述符。在主机端分配并填充算法选择描述符结构体，记录算法模式及其关联的 matmul 描述符引用。

**参数说明**：

- `handle`（IN）：HOST，aclsparseLt 库句柄。
- `algSelection`（OUT）：HOST，算法选择描述符输出。
- `matmulDescr`（IN）：HOST，matmul 描述符引用。
- `alg`（IN）：HOST，算法模式（`ACL_SPARSE_LT_MATMUL_ALG_DEFAULT`）。

**返回值**：

- `ACL_SPARSE_STATUS_SUCCESS`：成功
- `ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR`：handle 指针为空
- `ACL_SPARSE_STATUS_INVALID_VALUE`：algSelection 为空、matmulDescr 为空或未初始化、alg 非法
- `ACL_SPARSE_STATUS_ALLOC_FAILED`：内存分配失败

---

### aclsparseLtMatmulAlgSelectionDestroy

```c
aclsparseStatus_t aclsparseLtMatmulAlgSelectionDestroy(aclsparseLtMatmulAlgSelection_t* algSelection);
```

**功能**：销毁 matmul 算法选择描述符，释放其占用的主机内存。本接口不销毁 matmulDescr，用户须自行调用 `aclsparseLtMatmulDescriptorDestroy` 释放。销毁成功后 `*algSelection` 被置为 nullptr。

**参数说明**：

- `algSelection`（IN/OUT）：HOST，指向要销毁的算法选择描述符的指针。

**返回值**：

- `ACL_SPARSE_STATUS_SUCCESS`：成功
- `ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR`：algSelection 指针为空

---

### aclsparseLtMatmulAlgSetAttribute

> **支持状态**：暂未支持。当前版本尚未实现。

```c
aclsparseStatus_t aclsparseLtMatmulAlgSetAttribute(
    aclsparseLtConstHandle_t handle,
    aclsparseLtMatmulAlgSelection_t* algSelection,
    aclsparseLtMatmulAlgAttribute_t attribute,
    const void* data,
    size_t dataSize);
```

**功能**：设置算法选择描述符的指定属性（如算法配置 ID、搜索迭代次数、Split-K 参数等）。

**参数说明**：

- `handle`（IN）：HOST，aclsparseLt 库句柄。
- `algSelection`（IN/OUT）：HOST，算法选择描述符。
- `attribute`（IN）：HOST，要设置的算法选择描述符属性。
- `data`（IN）：HOST，指向属性值的指针。
- `dataSize`（IN）：HOST，属性值字节数，用于校验。

**返回值**：

- `ACL_SPARSE_STATUS_SUCCESS`：成功
- `ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR`：handle 为空指针
- `ACL_SPARSE_STATUS_INVALID_VALUE`：algSelection 为空、attribute 非法、data 为空、dataSize 与属性不匹配
- `ACL_SPARSE_STATUS_NOT_SUPPORTED`：该属性暂不支持

---

### aclsparseLtMatmulAlgGetAttribute

> **支持状态**：暂未支持。当前版本尚未实现。

```c
aclsparseStatus_t aclsparseLtMatmulAlgGetAttribute(
    aclsparseLtConstHandle_t handle,
    aclsparseLtConstMatmulAlgSelection_t* algSelection,
    aclsparseLtMatmulAlgAttribute_t attribute,
    void* data,
    size_t dataSize);
```

**功能**：获取算法选择描述符的指定属性（如算法配置 ID、搜索迭代次数、Split-K 参数等）。

**参数说明**：

- `handle`（IN）：HOST，aclsparseLt 库句柄。
- `algSelection`（IN）：HOST，算法选择描述符。
- `attribute`（IN）：HOST，要获取的算法选择描述符属性。
- `data`（OUT）：HOST，返回属性值的内存地址。
- `dataSize`（IN）：HOST，属性值字节数，用于校验。

**返回值**：

- `ACL_SPARSE_STATUS_SUCCESS`：成功
- `ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR`：handle 为空指针
- `ACL_SPARSE_STATUS_INVALID_VALUE`：algSelection 为空、attribute 非法、data 为空、dataSize 与属性不匹配
- `ACL_SPARSE_STATUS_NOT_SUPPORTED`：该属性暂不支持

---

### aclsparseLtMatmulPlanInit

```c
aclsparseStatus_t aclsparseLtMatmulPlanInit(
    const aclsparseLtHandle_t*             handle,
    aclsparseLtMatmulPlan_t*               plan,
    const aclsparseLtMatmulDescriptor_t*   matmulDescr,
    const aclsparseLtMatmulAlgSelection_t* algSelection);
```

**功能**：初始化 matmul 执行计划。在主机端分配并填充 plan 结构体，绑定 matmul 描述符与算法选择描述符，作为后续 matmul 执行的规划对象。

**参数说明**：

- `handle`（IN）：HOST，aclsparseLt 库句柄。
- `plan`（OUT）：HOST，执行计划输出。
- `matmulDescr`（IN）：HOST，matmul 描述符引用。
- `algSelection`（IN）：HOST，算法选择描述符引用。

**返回值**：

- `ACL_SPARSE_STATUS_SUCCESS`：成功
- `ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR`：handle 指针为空
- `ACL_SPARSE_STATUS_INVALID_VALUE`：plan 为空、matmulDescr 或 algSelection 为空或未初始化
- `ACL_SPARSE_STATUS_ALLOC_FAILED`：内存分配失败

---

### aclsparseLtMatmulPlanDestroy

```c
aclsparseStatus_t aclsparseLtMatmulPlanDestroy(aclsparseLtMatmulPlan_t* plan);
```

**功能**：销毁 matmul 执行计划，释放其占用的主机内存。本接口不销毁 matmulDescr 和 algSelection，用户须自行调用对应 Destroy 释放。plan 须在 algSelection 与 matmul 描述符之前销毁。销毁成功后 `*plan` 被置为 nullptr。

**参数说明**：

- `plan`（IN/OUT）：HOST，指向要销毁的执行计划的指针。

**返回值**：

- `ACL_SPARSE_STATUS_SUCCESS`：成功
- `ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR`：plan 指针为空

---

### aclsparseLtMatmulGetWorkspace

> **支持状态**：暂未支持。当前版本尚未实现。

```c
aclsparseStatus_t aclsparseLtMatmulGetWorkspace(
    aclsparseLtConstHandle_t handle,
    aclsparseLtConstMatmulPlan_t* plan,
    size_t* workspaceSize);
```

**功能**：查询执行 matmul 所需的 workspace 大小（字节）。

**参数说明**：

- `handle`（IN）：HOST，aclsparseLt 库句柄。
- `plan`（IN）：HOST，matmul 执行计划。
- `workspaceSize`（OUT）：HOST，返回所需 workspace 大小（字节）。

**返回值**：

- `ACL_SPARSE_STATUS_SUCCESS`：成功
- `ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR`：handle 为空指针
- `ACL_SPARSE_STATUS_INVALID_VALUE`：plan 为空、workspaceSize 为空

---

### aclsparseLtMatmul

> **支持状态**：暂未支持。当前版本尚未实现。

```c
aclsparseStatus_t aclsparseLtMatmul(
    aclsparseLtConstHandle_t handle,
    aclsparseLtConstMatmulPlan_t* plan,
    const void* alpha,
    const void* d_A,
    const void* d_B,
    const void* beta,
    const void* d_C,
    void* d_D,
    void* workspace,
    aclrtStream* streams,
    int32_t numStreams);
```

**功能**：执行结构化稀疏矩阵乘法，计算 `D = α · op(A) · op(B) + β · op(C)`（含可选的激活与偏置）。A、B 中有且仅有一个为结构化稀疏矩阵，须为 `aclsparseLtSpMMACompress` 的压缩输出。算子在指定 stream 上异步执行。

**参数说明**：

- `handle`（IN）：HOST，aclsparseLt 库句柄。
- `plan`（IN）：HOST，matmul 执行计划。
- `alpha`（IN）：HOST/DEVICE，标量 α（`float` 类型指针）；标量时为 HOST 指针，向量缩放时为 DEVICE 指针。
- `d_A`（IN）：DEVICE，矩阵 A 的指针（结构化稀疏或稠密）。
- `d_B`（IN）：DEVICE，矩阵 B 的指针（结构化稀疏或稠密）。
- `beta`（IN）：HOST/DEVICE，标量 β（`float` 类型指针）；标量时为 HOST 指针，向量缩放时为 DEVICE 指针。
- `d_C`（IN）：DEVICE，稠密矩阵 C 的指针。
- `d_D`（OUT）：DEVICE，稠密矩阵 D 的输出指针。
- `workspace`（IN）：DEVICE，workspace 指针。
- `streams`（IN）：HOST，指向 ACL stream 数组的指针。
- `numStreams`（IN）：HOST，`streams` 数组中的 stream 数量。

**返回值**：

- `ACL_SPARSE_STATUS_SUCCESS`：成功
- `ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR`：handle 为空指针
- `ACL_SPARSE_STATUS_INVALID_VALUE`：plan 为空、矩阵指针为空
- `ACL_SPARSE_STATUS_NOT_SUPPORTED`：不支持的数据类型组合或操作
- `ACL_SPARSE_STATUS_EXECUTION_FAILED`：kernel 执行失败
- `ACL_SPARSE_STATUS_INSUFFICIENT_RESOURCES`：资源不足

---

### aclsparseLtMatmulSearch

> **支持状态**：暂未支持。当前版本尚未实现。

```c
aclsparseStatus_t aclsparseLtMatmulSearch(
    aclsparseLtConstHandle_t handle,
    aclsparseLtMatmulPlan_t* plan,
    const void* alpha,
    const void* d_A,
    const void* d_B,
    const void* beta,
    const void* d_C,
    void* d_D,
    void* workspace,
    aclrtStream* streams,
    int32_t numStreams);
```

**功能**：对 `plan` 描述的 matmul 评估所有可用算法，自动选择最快的算法并更新关联的算法选择描述符，用于同一运算多次执行时的自动调优。语义类似 `aclsparseLtMatmul`，但本接口为阻塞调用，且 in-place（`d_C == d_D`）时 `d_D` 结果可能累积。

**参数说明**：

- `handle`（IN）：HOST，aclsparseLt 库句柄。
- `plan`（IN/OUT）：HOST，matmul 执行计划；搜索完成后内部更新关联的算法选择描述符。
- `alpha`（IN）：HOST，标量 α（`float` 类型指针）。
- `d_A`（IN）：DEVICE，矩阵 A 的指针（结构化稀疏或稠密）。
- `d_B`（IN）：DEVICE，矩阵 B 的指针（结构化稀疏或稠密）。
- `beta`（IN）：HOST，标量 β（`float` 类型指针）。
- `d_C`（IN）：DEVICE，稠密矩阵 C 的指针。
- `d_D`（OUT）：DEVICE，稠密矩阵 D 的输出指针。
- `workspace`（IN）：DEVICE，workspace 指针。
- `streams`（IN）：HOST，指向 ACL stream 数组的指针。
- `numStreams`（IN）：HOST，`streams` 数组中的 stream 数量。

**返回值**：

- `ACL_SPARSE_STATUS_SUCCESS`：成功
- `ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR`：handle 为空指针
- `ACL_SPARSE_STATUS_INVALID_VALUE`：plan 为空、矩阵指针为空
- `ACL_SPARSE_STATUS_NOT_SUPPORTED`：不支持的数据类型组合或操作
- `ACL_SPARSE_STATUS_EXECUTION_FAILED`：kernel 执行失败
- `ACL_SPARSE_STATUS_INSUFFICIENT_RESOURCES`：资源不足

---

### aclsparseLtSpMMAPrune

```c
aclsparseStatus_t aclsparseLtSpMMAPrune(
    aclsparseLtConstHandle_t handle,
    aclsparseLtConstMatmulDescriptor_t* matmulDescr,
    const void* d_in,
    void* d_out,
    aclsparseLtPruneAlg_t pruneAlg,
    aclrtStream stream);
```

**功能**：对稠密矩阵 A 执行 2:4 结构化稀疏剪枝，将每 4 个元素（FP16/BF16/INT8）或每 2 个元素（FP32）中绝对值较小的元素置零，保留绝对值最大的元素，输出与 A 同型的稠密存储矩阵 A_pruned（被置零元素以 0 表示）。直接接收 Matmul 描述符，从中读取矩阵 A 的维度（m、k）、数据类型、order 与 opA 派生剪枝参数。算子异步启动，内部不执行 stream 同步，调用方如需读取结果须自行同步。

> 说明：本算子为软件实现的 2:4 结构化稀疏剪枝，不依赖 Ascend 950 硬件的 2:4 稀疏加速单元，仅通过 Vector API 完成剪枝计算。FP32 统一采用 1:2 模式（分组为 2、保留 1，即每 2 个元素中保留绝对值最大的 1 个），并非"4 选 2"；FP16/BF16/INT8 采用标准 2:4 模式（分组为 4、保留 2）。所有数据类型稀疏度均为 50%。

**剪枝规则**：

| 数据类型 | 分组大小 | 每组保留元素数 | 稀疏度 |
|----------|---------|---------------|--------|
| FP16（ACL_FLOAT16） | 4 | 2（保留绝对值最大的 2 个） | 50%（2:4） |
| BF16（ACL_BF16） | 4 | 2（保留绝对值最大的 2 个） | 50%（2:4） |
| INT8（ACL_INT8） | 4 | 2（保留绝对值最大的 2 个） | 50%（2:4） |
| FP32（ACL_FLOAT） | 2 | 1（保留绝对值最大的 1 个） | 50%（1:2，等价 2:4 密度） |

**产品支持情况**：

| 产品 | 是否支持 |
| :----------------------------------------- | :------:|
| <term>Ascend 950PR/Ascend 950DT</term> | √ |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> | × |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> | × |

> 依赖 CANN asc-devkit >= 9.1.0（`ASC_DEVKIT_MAJOR >= 9 && ASC_DEVKIT_MINOR >= 1`），低于该版本时编译与运行将跳过此算子。

**参数说明**：

- `handle`（IN）：HOST，aclsparseLt 库句柄的 const 指针。
- `matmulDescr`（IN）：HOST，Matmul 操作描述符，算子从中读取矩阵 A 的维度（m、k）、数据类型、order 与 opA。
- `d_in`（IN）：DEVICE，待剪枝的稠密矩阵 A 的指针。
- `d_out`（OUT）：DEVICE，剪枝结果 A_pruned 的指针。支持 in-place（d_in == d_out）。
- `pruneAlg`（IN）：HOST，剪枝算法，支持 `ACLSPARSELT_PRUNE_SPMMA_STRIP` 与 `ACLSPARSELT_PRUNE_SPMMA_TILE`。
- `stream`（IN）：HOST，ACL 流，算子在此流上异步执行，可为 nullptr（表示使用默认流）。

**返回值**：

- `ACL_SPARSE_STATUS_SUCCESS`：成功
- `ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR`：handle 为 nullptr
- `ACL_SPARSE_STATUS_INVALID_VALUE`：matmulDescr/matA/d_in/d_out 为 nullptr、数据指针未 16 字节对齐
- `ACL_SPARSE_STATUS_NOT_SUPPORTED`：pruneAlg 非 STRIP/TILE、数据类型非 FP32/FP16/BF16/INT8、K 维度超过 UB 容量、转置路径 in-place（d_in == d_out）

---

### aclsparseLtSpMMAPruneCheck

> **支持状态**：暂未支持。当前版本尚未实现。

```c
aclsparseStatus_t aclsparseLtSpMMAPruneCheck(
    aclsparseLtConstHandle_t handle,
    aclsparseLtConstMatmulDescriptor_t* matmulDescr,
    const void* d_in,
    int* d_valid,
    aclrtStream stream);
```

**功能**：校验稠密矩阵是否已满足 2:4 结构化稀疏约束。经 `aclsparseLtSpMMAPrune` 剪枝的结果保证正确，可跳过本接口；仅用于校验自定义剪枝结果的合法性。

**参数说明**：

- `handle`（IN）：HOST，aclsparseLt 库句柄。
- `matmulDescr`（IN）：HOST，Matmul 操作描述符，从中读取矩阵 A 的维度、数据类型、order 与 opA。
- `d_in`（IN）：DEVICE，待校验的矩阵指针。
- `d_valid`（OUT）：DEVICE，校验结果（`0` 表示满足约束，`1` 表示不满足）。
- `stream`（IN）：HOST，ACL 流，可为 nullptr（表示使用默认流）。

**返回值**：

- `ACL_SPARSE_STATUS_SUCCESS`：成功
- `ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR`：handle 为空指针
- `ACL_SPARSE_STATUS_INVALID_VALUE`：matmulDescr 为空、d_in 为空、d_valid 为空

---

### aclsparseLtSpMMACompressedSize

> **支持状态**：暂未支持。当前版本尚未实现。

```c
aclsparseStatus_t aclsparseLtSpMMACompressedSize(
    aclsparseLtConstHandle_t handle,
    aclsparseLtConstMatmulDescriptor_t* matmulDescr,
    size_t* compressedSize);
```

**功能**：查询 2:4 结构化稀疏矩阵压缩后所需的存储大小（字节），用于在调用 `aclsparseLtSpMMACompress` 前分配 Device 内存。

**参数说明**：

- `handle`（IN）：HOST，aclsparseLt 库句柄。
- `matmulDescr`（IN）：HOST，Matmul 操作描述符，从中读取稀疏侧矩阵的维度、数据类型等信息。
- `compressedSize`（OUT）：HOST，返回压缩后矩阵所需的存储大小（字节）。

**返回值**：

- `ACL_SPARSE_STATUS_SUCCESS`：成功
- `ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR`：handle 为空指针
- `ACL_SPARSE_STATUS_INVALID_VALUE`：matmulDescr 为空、compressedSize 为空

---

### aclsparseLtSpMMACompress

> **支持状态**：暂未支持。当前版本尚未实现。

```c
aclsparseStatus_t aclsparseLtSpMMACompress(
    aclsparseLtConstHandle_t handle,
    aclsparseLtConstMatmulDescriptor_t* matmulDescr,
    const void* d_in,
    void* d_out,
    aclrtStream stream);
```

**功能**：将已剪枝的 2:4 稀疏矩阵压缩为紧凑存储，压缩结果用作 `aclsparseLtMatmul` / `aclsparseLtMatmulSearch` 中稀疏侧矩阵（A 或 B）的输入。输入须已通过 `aclsparseLtSpMMAPrune` 剪枝或满足 2:4 约束。算子在指定 stream 上异步执行。

**参数说明**：

- `handle`（IN）：HOST，aclsparseLt 库句柄。
- `matmulDescr`（IN）：HOST，Matmul 操作描述符，从中读取稀疏侧矩阵的维度、数据类型、order 与 opA。
- `d_in`（IN）：DEVICE，待压缩的已剪枝稀疏矩阵指针。
- `d_out`（OUT）：DEVICE，压缩结果输出指针。
- `stream`（IN）：HOST，ACL 流，可为 nullptr（表示使用默认流）。

**返回值**：

- `ACL_SPARSE_STATUS_SUCCESS`：成功
- `ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR`：handle 为空指针
- `ACL_SPARSE_STATUS_INVALID_VALUE`：matmulDescr 为空、d_in 为空、d_out 为空
- `ACL_SPARSE_STATUS_NOT_SUPPORTED`：不支持的数据类型

---

## 枚举说明

### aclsparseLtSparsity_t

结构化稀疏模式枚举：

| 枚举值 | 说明 |
|--------|------|
| `ACL_SPARSE_LT_SPARSITY_50_PERCENT` | 2:4 结构化稀疏（50% 稀疏度） |

### aclsparseComputeType_t

计算精度枚举：

| 枚举值 | 说明 |
|--------|------|
| `ACL_SPARSE_COMPUTE_16F` | FP16 计算精度 |
| `ACL_SPARSE_COMPUTE_32F` | FP32 计算精度 |
| `ACL_SPARSE_COMPUTE_32I` | INT32 计算精度 |

### aclsparseLtMatmulAlg_t

matmul 算法模式枚举：

| 枚举值 | 说明 |
|--------|------|
| `ACL_SPARSE_LT_MATMUL_ALG_DEFAULT` | 默认算法 |

### aclsparseLtPruneAlg_t

剪枝算法枚举：

| 枚举值 | 说明 |
|--------|------|
| `ACLSPARSELT_PRUNE_SPMMA_TILE` | 逐 tile 剪枝，以 tile 为单位枚举所有满足 2:4 约束的有效配置，选 L1-norm 最大的配置 |
| `ACLSPARSELT_PRUNE_SPMMA_STRIP` | 逐分组剪枝，每组独立选择绝对值最大的元素，不跨组优化 |

### 公共枚举

以下枚举为 aclsparse 与 aclsparseLt 共用，定义于 `cann_ops_sparse.h`，详细说明请参考 [api_list.md 枚举说明](./api_list.md#枚举说明)：

- `aclsparseStatus_t`：返回状态码
- `aclsparseOperation_t`：稀疏操作类型（`ACL_SPARSE_OP_NON_TRANSPOSE` / `ACL_SPARSE_OP_TRANSPOSE` / `ACL_SPARSE_OP_CONJUGATE_TRANSPOSE`）
- `aclsparseOrder_t`：稠密矩阵布局（`ACL_SPARSE_ORDER_ROW` / `ACL_SPARSE_ORDER_COL`）

---

## 推荐调用流程

aclsparseLt 的完整工作流分为初始化、描述符构建、计划构建、执行、清理五个阶段。

1. 调用 `aclsparseLtInit` 创建库上下文。
2. 对稀疏侧矩阵（A 或 B）调用 `aclsparseLtStructuredDescriptorInit` 创建结构化描述符，对稠密侧矩阵调用 `aclsparseLtDenseDescriptorInit` 创建稠密描述符。
3. 调用 `aclsparseLtMatmulDescriptorInit` 构建 Matmul 描述符，传入 opA/opB、四个矩阵描述符（matA/matB/matC/matD）与 computeType。
4. （可选）调用 `aclsparseLtSpMMAPrune` 对稠密矩阵执行 2:4 剪枝，生成稀疏侧输入。
5. （可选）调用 `aclsparseLtSpMMACompressedSize` 查询压缩后所需存储大小，再调用 `aclsparseLtSpMMACompress` 将稀疏矩阵压缩为紧凑存储。
6. 调用 `aclsparseLtMatmulAlgSelectionInit` 构建算法选择描述符。
7. 调用 `aclsparseLtMatmulPlanInit` 构建执行计划。
8. 调用 `aclsparseLtMatmulGetWorkspace` 查询所需 workspace 大小并分配设备内存。
9. 调用 `aclsparseLtMatmul` 执行结构化稀疏矩阵乘法；或调用 `aclsparseLtMatmulSearch` 自动搜索最优算法。
10. 按依赖逆序销毁执行计划、算法选择描述符、Matmul 描述符、矩阵描述符，最后调用 `aclsparseLtDestroy` 释放库句柄。

---

## 最小示例（伪代码）

```c
#include "acl/acl.h"
#include "cann_ops_sparseLt.h"

int aclsparseLtExample()
{
    int32_t deviceId = 0;
    aclrtStream stream = nullptr;
    aclsparseLtHandle_t handle = nullptr;
    aclsparseLtMatDescriptor_t matA = nullptr, matB = nullptr, matC = nullptr, matD = nullptr;
    aclsparseLtMatmulDescriptor_t matmulDesc = nullptr;
    aclsparseLtMatmulAlgSelection_t algSelection = nullptr;
    aclsparseLtMatmulPlan_t plan = nullptr;
    void *dA = nullptr, *dAPruned = nullptr;

    aclInit(nullptr);
    aclrtSetDevice(deviceId);
    aclrtCreateStream(&stream);

    // 1. 初始化库句柄
    aclsparseLtInit(&handle);

    // 2. 构建矩阵描述符：A 为结构化稀疏（2:4），m×k，FP32，行主序
    int64_t m = 16, k = 32, n = 128;
    aclsparseLtStructuredDescriptorInit(&handle, &matA, m, k, k, 16,
        ACL_FLOAT, ACL_SPARSE_ORDER_ROW, ACL_SPARSE_LT_SPARSITY_50_PERCENT);
    aclsparseLtDenseDescriptorInit(&handle, &matB, k, n, n, 16, ACL_FLOAT, ACL_SPARSE_ORDER_ROW);
    aclsparseLtDenseDescriptorInit(&handle, &matC, m, n, n, 16, ACL_FLOAT, ACL_SPARSE_ORDER_ROW);
    aclsparseLtDenseDescriptorInit(&handle, &matD, m, n, n, 16, ACL_FLOAT, ACL_SPARSE_ORDER_ROW);

    // 3. 构建 Matmul 描述符
    aclsparseLtMatmulDescriptorInit(&handle, &matmulDesc,
        ACL_SPARSE_OP_NON_TRANSPOSE, ACL_SPARSE_OP_NON_TRANSPOSE,
        &matA, &matB, &matC, &matD, ACL_SPARSE_COMPUTE_32F);

    // 4. 执行 2:4 剪枝：dA -> dAPruned
    aclsparseLtSpMMAPrune(&handle, &matmulDesc,
        dA, dAPruned, ACLSPARSELT_PRUNE_SPMMA_STRIP, stream);
    aclrtSynchronizeStream(stream);  // 算子内部不同步，调用方负责同步

    // --- 以下步骤当前版本暂未支持 ---

    // 5.（可选）压缩稀疏矩阵
    // size_t compressedSize = 0;
    // aclsparseLtSpMMACompressedSize(&handle, &matmulDesc, &compressedSize);
    // void *dACompressed = nullptr;
    // aclrtMalloc(&dACompressed, compressedSize, ACL_MEM_MALLOC_HUGE_FIRST);
    // aclsparseLtSpMMACompress(&handle, &matmulDesc, dAPruned, dACompressed, stream);

    // 6. 构建算法选择描述符
    aclsparseLtMatmulAlgSelectionInit(&handle, &algSelection, &matmulDesc,
        ACL_SPARSE_LT_MATMUL_ALG_DEFAULT);

    // 7. 构建执行计划
    aclsparseLtMatmulPlanInit(&handle, &plan, &matmulDesc, &algSelection);

    // 8. 获取 workspace 大小并分配
    // size_t workspaceSize = 0;
    // aclsparseLtMatmulGetWorkspace(&handle, &plan, &workspaceSize);
    // void *workspace = nullptr;
    // aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);

    // 9. 执行结构化稀疏矩阵乘法
    // float alpha = 1.0f, beta = 0.0f;
    // aclrtStream streams[] = {stream};
    // aclsparseLtMatmul(&handle, &plan, &alpha, dAPruned, dB, &beta, dC, dD,
    //     workspace, streams, 1);

    // 10. 清理资源（按依赖逆序销毁）
    aclsparseLtMatmulPlanDestroy(&plan);
    aclsparseLtMatmulAlgSelectionDestroy(&algSelection);
    aclsparseLtMatmulDescriptorDestroy(&matmulDesc);
    aclsparseLtMatDescriptorDestroy(&matA);
    aclsparseLtMatDescriptorDestroy(&matB);
    aclsparseLtMatDescriptorDestroy(&matC);
    aclsparseLtMatDescriptorDestroy(&matD);
    aclsparseLtDestroy(&handle);

    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
```

---

## 备注

- 接口能力与属性支持范围以当前实现版本为准，详细限制请参考各算子目录下的README文档（如`sparseLt/common/sparseLtDescriptor_README.md`、`sparseLt/prune/README.md`）及头文件内注释。
- 若文档描述与头文件声明不一致，请**以头文件声明与实际实现行为为准**。

