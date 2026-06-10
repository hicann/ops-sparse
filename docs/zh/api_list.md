# 稀疏算子接口（aclSparse）

## 使用说明

为方便调用稀疏算子，提供一套基于C的API（以aclSparse为前缀API），主要用于稀疏矩阵运算场景。该API提供稀疏矩阵与向量的乘法（SpMV）等操作。

调用稀疏算子API时，需引用依赖的头文件和库文件。

- 依赖的头文件：`cann_ops_sparse.h`
- 依赖的库文件：稀疏算子库文件

## 接口列表

| 接口名 | 说明 |
|--------|------|
| [aclsparseCreate](#aclsparsecreate) | 创建稀疏矩阵处理器 |
| [aclsparseDestroy](#aclsparsedestroy) | 销毁稀疏矩阵处理器 |
| [aclsparseSetStream](#aclsparsesetstream) | 设置处理器使用的 stream |
| [aclsparseGetStream](#aclsparsegetstream) | 获取处理器当前的 stream |
| [aclSparseCreateDnVec](#aclsparsecreatednvec) | 创建稠密向量 |
| [aclSparseDestroyDnVec](#aclsparsedestroydnvec) | 销毁稀疏向量描述符 |
| [aclsparseCreateCsr](#aclsparsecreatecsr) | 创建CSR格式稀疏矩阵 |
| [aclsparseCreateCsc](#aclsparsecreatecsc) | 创建CSC格式稀疏矩阵 |
| [aclsparseDestroySpMat](#aclsparsedestroyspmat) | 销毁稀疏矩阵对象 |
| [aclSparseSpmvGetBufferSize](#aclsparsespmvgetbuffersize) | 获取SpMV缓冲区大小 |
| [aclSparseSpmvPreprocess](#aclsparsespmvpreprocess) | SpMV预处理 |
| [aclSparseSpmv](#aclsparsespmv) | 稀疏矩阵向量乘法 |
| [aclSparseSpmvShowWorkSpace](#aclsparsespmvshowworkspace) | 显示SpMV工作空间 |

## 接口详情

### aclsparseCreate

```c
aclsparseStatus_t aclsparseCreate(aclsparseHandle_t *handle);
```

**功能**：创建稀疏矩阵处理器。处理器的 stream 默认为 `nullptr`（默认 stream），如需指定请调用 `aclsparseSetStream`。

**参数说明**：

- `handle`（IN/OUT）：HOST，返回创建的稀疏矩阵处理器句柄。调用前 `*handle` 必须为 `nullptr`。

**返回值**：

- `ACL_SPARSE_STATUS_SUCCESS`：成功
- 其他值：失败

---

### aclsparseDestroy

```c
aclsparseStatus_t aclsparseDestroy(aclsparseHandle_t handle);
```

**功能**：销毁稀疏矩阵处理器。stream 由用户托管，此接口不会销毁 stream。

**参数说明**：

- `handle`（IN）：稀疏矩阵处理器句柄。

**返回值**：

- `ACL_SPARSE_STATUS_SUCCESS`：成功
- 其他值：失败

---

### aclsparseSetStream

```c
aclsparseStatus_t aclsparseSetStream(aclsparseHandle_t handle, aclrtStream stream);
```

**功能**：设置处理器后续算子执行所使用的 stream。

**参数说明**：

- `handle`（IN）：稀疏矩阵处理器句柄。
- `stream`（IN）：要绑定的 stream，`nullptr` 表示使用默认 stream。

**返回值**：

- `ACL_SPARSE_STATUS_SUCCESS`：成功
- 其他值：失败

---

### aclsparseGetStream

```c
aclsparseStatus_t aclsparseGetStream(aclsparseHandle_t handle, aclrtStream *stream);
```

**功能**：获取处理器当前绑定的 stream。

**参数说明**：

- `handle`（IN）：稀疏矩阵处理器句柄。
- `stream`（OUT）：返回当前 stream。

**返回值**：

- `ACL_SPARSE_STATUS_SUCCESS`：成功
- 其他值：失败

---

### aclSparseCreateDnVec

```c
aclsparseStatus_t aclSparseCreateDnVec(
    aclsparseDnVecDescr_t *dnVecDescr,
    int64_t size,
    void *values,
    aclDataType valueType
);
```

**功能**：创建一个稠密向量。

**参数说明**：

- `dnVecDescr`（IN/OUT）：HOST，稀疏向量描述符。
- `size`（IN）：HOST，向量的大小。
- `values`（IN）：DEVICE，向量的值。
- `valueType`（IN）：HOST，值的数据类型。

**返回值**：

- `ACL_SPARSE_STATUS_SUCCESS`：成功
- 其他值：失败

---

### aclSparseDestroyDnVec

```c
aclsparseStatus_t aclSparseDestroyDnVec(aclsparseConstDnVecDescr_t dnVecDescr);
```

**功能**：销毁稀疏向量描述符。

**参数说明**：

- `dnVecDescr`（IN）：HOST，要销毁的稀疏向量描述符。

**返回值**：

- `ACL_SPARSE_STATUS_SUCCESS`：成功
- 其他值：失败

---

### aclsparseCreateCsr

```c
aclsparseStatus_t aclsparseCreateCsr(
    aclsparseSpMatDescr_t *spMatDescr,
    int64_t rows,
    int64_t cols,
    int64_t nnz,
    void *csrRowOffsets,
    void *csrColInd,
    void *csrValues,
    aclsparseIndexType_t csrRowOffsetsType,
    aclsparseIndexType_t csrColIndType,
    aclsparseIndexBase_t idxBase,
    aclDataType valueType
);
```

**功能**：创建一个CSR（Compressed Sparse Row）格式的稀疏矩阵。

**参数说明**：

- `spMatDescr`（IN/OUT）：HOST，稀疏矩阵描述符。
- `rows`（IN）：HOST，矩阵的行数。
- `cols`（IN）：HOST，矩阵的列数。
- `nnz`（IN）：HOST，矩阵的非零元素个数。
- `csrRowOffsets`（IN）：DEVICE，指向CSR格式行偏移量数组的指针。
- `csrColInd`（IN）：DEVICE，指向CSR格式列索引数组的指针。
- `csrValues`（IN）：DEVICE，指向CSR格式非零元素数组的指针。
- `csrRowOffsetsType`（IN）：HOST，行偏移量数组的数据类型。
- `csrColIndType`（IN）：HOST，列索引数组的数据类型。
- `idxBase`（IN）：HOST，索引的基值（0或1）。
- `valueType`（IN）：HOST，非零元素的数据类型。

**返回值**：

- `ACL_SPARSE_STATUS_SUCCESS`：成功
- 其他值：失败

---

### aclsparseCreateCsc

```c
aclsparseStatus_t aclsparseCreateCsc(
    aclsparseSpMatDescr_t *spMatDescr,
    int64_t rows,
    int64_t cols,
    int64_t nnz,
    void *cscColOffsets,
    void *cscRowInd,
    void *cscValues,
    aclsparseIndexType_t cscColOffsetsType,
    aclsparseIndexType_t cscRowIndType,
    aclsparseIndexBase_t idxBase,
    aclDataType valueType
);
```

**功能**：创建一个CSC（Compressed Sparse Column）格式的稀疏矩阵。

**参数说明**：

- `spMatDescr`（IN/OUT）：HOST，稀疏矩阵描述符。
- `rows`（IN）：HOST，矩阵的行数。
- `cols`（IN）：HOST，矩阵的列数。
- `nnz`（IN）：HOST，矩阵的非零元素个数。
- `cscColOffsets`（IN）：DEVICE，指向列偏移数组的指针。
- `cscRowInd`（IN）：DEVICE，指向行索引数组的指针。
- `cscValues`（IN）：DEVICE，指向非零元素值数组的指针。
- `cscColOffsetsType`（IN）：HOST，列偏移数组的数据类型。
- `cscRowIndType`（IN）：HOST，行索引数组的数据类型。
- `idxBase`（IN）：HOST，索引的基础值。
- `valueType`（IN）：HOST，非零元素值的数据类型。

**返回值**：

- `ACL_SPARSE_STATUS_SUCCESS`：成功
- 其他值：失败

---

### aclsparseDestroySpMat

```c
aclsparseStatus_t aclsparseDestroySpMat(aclsparseConstSpMatDescr_t spMatDescr);
```

**功能**：销毁稀疏矩阵对象。

**参数说明**：

- `spMatDescr`（IN）：HOST，稀疏矩阵描述符。

**返回值**：

- `ACL_SPARSE_STATUS_SUCCESS`：成功
- 其他值：失败

---

### 只读(const)描述符构造接口

只读(const)构造接口：数据指针为 `const`，构造出的描述符是 `const` 变体，只能传给接收 `const` 形参的接口（如 SpMV/SpMM 的输入 `mat`/`matA`/`x`/`matB`）。销毁接口（`aclSparseDestroyDnVec` / `aclsparseDestroySpMat` / `aclsparseDestroyDnMat`）统一接收 `const` 变体，因此 const 与非 const 描述符都可销毁。

```c
aclsparseStatus_t aclsparseCreateConstDnVec(aclsparseConstDnVecDescr_t *dnVecDescr, int64_t size,
    const void *values, aclDataType valueType);

aclsparseStatus_t aclsparseCreateConstCsr(aclsparseConstSpMatDescr_t *spMatDescr, int64_t rows, int64_t cols, int64_t nnz,
    const void *csrRowOffsets, const void *csrColInd, const void *csrValues, aclsparseIndexType_t csrRowOffsetsType,
    aclsparseIndexType_t csrColIndType, aclsparseIndexBase_t idxBase, aclDataType valueType);

aclsparseStatus_t aclsparseCreateConstCsc(aclsparseConstSpMatDescr_t *spMatDescr, int64_t rows, int64_t cols, int64_t nnz,
    const void *cscColOffsets, const void *cscRowInd, const void *cscValues, aclsparseIndexType_t cscColOffsetsType,
    aclsparseIndexType_t cscRowIndType, aclsparseIndexBase_t idxBase, aclDataType valueType);

aclsparseStatus_t aclsparseCreateConstDnMat(aclsparseConstDnMatDescr_t *dnMatDescr,
    int64_t rows, int64_t cols, int64_t ld, const void *values,
    aclDataType valueType, aclsparseOrder_t order);
```

---

### aclSparseSpmvGetBufferSize

```c
aclsparseStatus_t aclSparseSpmvGetBufferSize(
    aclsparseHandle_t handle,
    aclsparseOperation_t op,
    const void *alpha,
    aclsparseConstSpMatDescr_t mat,
    aclsparseConstDnVecDescr_t x,
    const void *beta,
    aclsparseDnVecDescr_t y,
    aclDataType computeType,
    aclsparseSpMVAlg_t alg,
    size_t *size
);
```

**功能**：获取稀疏矩阵与向量相乘（SpMV）操作所需的缓冲区大小。

**参数说明**：

- `handle`（IN）：HOST，稀疏计算句柄。
- `op`（IN）：HOST，稀疏操作类型。
- `alpha`（IN）：HOST/DEVICE，标量alpha参数。
- `mat`（IN）：HOST，稀疏矩阵描述符。
- `x`（IN）：HOST，输入向量x的描述符。
- `beta`（IN）：HOST/DEVICE，标量beta参数。
- `y`（IN）：HOST，输出向量y的描述符。
- `computeType`（IN）：HOST，计算的数据类型。
- `alg`（IN）：HOST，SpMV算法类型。
- `size`（OUT）：HOST，所需的缓冲区大小。

**返回值**：

- `ACL_SPARSE_STATUS_SUCCESS`：成功
- 其他值：失败

---

### aclSparseSpmvPreprocess

```c
aclsparseStatus_t aclSparseSpmvPreprocess(
    aclsparseHandle_t handle,
    aclsparseOperation_t op,
    const void *alpha,
    aclsparseConstSpMatDescr_t mat,
    aclsparseConstDnVecDescr_t x,
    const void *beta,
    aclsparseDnVecDescr_t y,
    aclDataType computeType,
    aclsparseSpMVAlg_t alg,
    void *buffer
);
```

**功能**：对稀疏矩阵进行预处理，以便在后续的SpMV计算中使用。

**参数说明**：

- `handle`（IN）：HOST，稀疏计算句柄。
- `op`（IN）：HOST，稀疏操作类型。
- `alpha`（IN）：HOST/DEVICE，标量alpha参数。
- `mat`（IN）：HOST，稀疏矩阵描述符。
- `x`（IN）：HOST，密集向量x的描述符。
- `beta`（IN）：HOST/DEVICE，标量beta参数。
- `y`（IN）：HOST，密集向量y的描述符。
- `computeType`（IN）：HOST，计算的数据类型。
- `alg`（IN）：HOST，SpMV算法类型。
- `buffer`（IN）：DEVICE，用于存储预处理结果的缓冲区。

**返回值**：

- `ACL_SPARSE_STATUS_SUCCESS`：成功
- 其他值：失败

---

### aclSparseSpmv

```c
aclsparseStatus_t aclSparseSpmv(
    aclsparseHandle_t handle,
    aclsparseOperation_t op,
    const void *alpha,
    aclsparseConstSpMatDescr_t mat,
    aclsparseConstDnVecDescr_t x,
    const void *beta,
    aclsparseDnVecDescr_t y,
    aclDataType computeType,
    aclsparseSpMVAlg_t alg,
    void *buffer
);
```

**功能**：稀疏矩阵向量乘法（SpMV）。

**参数说明**：

- `handle`（IN）：HOST，稀疏矩阵处理器句柄。
- `op`（IN）：HOST，稀疏矩阵操作符。
- `alpha`（IN）：HOST/DEVICE，标量alpha。
- `mat`（IN）：HOST，稀疏矩阵描述符。
- `x`（IN）：HOST，密集向量x描述符。
- `beta`（IN）：HOST/DEVICE，标量beta。
- `y`（IN/OUT）：HOST，密集向量y描述符。
- `computeType`（IN）：HOST，计算类型。
- `alg`（IN）：HOST，稀疏矩阵向量乘法算法。
- `buffer`（IN）：DEVICE，工作缓冲区指针。

**返回值**：

- `ACL_SPARSE_STATUS_SUCCESS`：成功
- 其他值：失败

---

### aclSparseSpmvShowWorkSpace

```c
aclsparseStatus_t aclSparseSpmvShowWorkSpace(aclsparseHandle_t handle, void *buffer);
```

**功能**：显示SpMV工作空间信息。

**参数说明**：

- `handle`（IN）：HOST，稀疏矩阵处理器句柄。
- `buffer`（IN）：DEVICE，工作缓冲区指针。

**返回值**：

- `ACL_SPARSE_STATUS_SUCCESS`：成功
- 其他值：失败

---

## 枚举说明

### aclsparseOperation_t

稀疏操作类型枚举：

| 枚举值 | 说明 |
|--------|------|
| `ACL_SPARSE_OP_NON_TRANSPOSE` | 非转置操作 |
| `ACL_SPARSE_OP_TRANSPOSE` | 转置操作 |
| `ACL_SPARSE_OP_CONJUGATE_TRANSPOSE` | 共轭转置操作 |

### aclsparseStatus_t

返回状态枚举：

| 枚举值 | 说明 |
|--------|------|
| `ACL_SPARSE_STATUS_SUCCESS` | 操作成功完成 |
| `ACL_SPARSE_STATUS_NOT_INITIALIZED` | aclSPARSE库未初始化 |
| `ACL_SPARSE_STATUS_ALLOC_FAILED` | 资源分配失败 |
| `ACL_SPARSE_STATUS_INVALID_VALUE` | 无效的参数值 |
| `ACL_SPARSE_STATUS_ARCH_MISMATCH` | 设备架构不匹配 |
| `ACL_SPARSE_STATUS_EXECUTION_FAILED` | NPU程序执行失败 |
| `ACL_SPARSE_STATUS_INTERNAL_ERROR` | 内部aclSPARSE操作失败 |
| `ACL_SPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED` | 不支持的矩阵类型 |
| `ACL_SPARSE_STATUS_NOT_SUPPORTED` | 当前不支持该操作或数据类型组合 |
| `ACL_SPARSE_STATUS_INSUFFICIENT_RESOURCES` | 资源不足 |

### aclsparseSpMVAlg_t

SpMV算法枚举：

| 枚举值 | 说明 |
|--------|------|
| `ACL_SPARSE_SPMV_ALG_DEFAULT` | 默认算法 |
| `ACL_SPARSE_SPMV_COO_ALG1` | COO格式默认算法 |
| `ACL_SPARSE_SPMV_COO_ALG2` | COO格式确定性算法 |
| `ACL_SPARSE_SPMV_CSR_ALG1` | CSR/CSC格式默认算法 |
| `ACL_SPARSE_SPMV_CSR_ALG2` | CSR/CSC格式确定性算法 |
| `ACL_SPARSE_SPMV_SELL_ALG1` | SELL格式默认算法 |

### aclsparseIndexType_t

索引类型枚举：

| 枚举值 | 说明 |
|--------|------|
| `ACL_SPARSE_INDEX_32I` | 32位无符号整数 [0, 2^31 - 1] |
| `ACL_SPARSE_INDEX_64I` | 64位无符号整数 [0, 2^63 - 1] |

### aclsparseIndexBase_t

索引基值枚举：

| 枚举值 | 说明 |
|--------|------|
| `ACL_SPARSE_INDEX_BASE_ZERO` | 索引从0开始（C语言兼容） |
| `ACL_SPARSE_INDEX_BASE_ONE` | 索引从1开始（Fortran兼容） |

### aclsparseFormat_t

稀疏矩阵格式枚举：

| 枚举值 | 说明 |
|--------|------|
| `ACL_SPARSE_FORMAT_COO` | COO（Coordinate）格式 |
| `ACL_SPARSE_FORMAT_CSR` | CSR（Compressed Sparse Row）格式 |
| `ACL_SPARSE_FORMAT_CSC` | CSC（Compressed Sparse Column）格式 |
| `ACL_SPARSE_FORMAT_BLOCKED_ELL` | Blocked-ELL格式 |
| `ACL_SPARSE_FORMAT_SLICED_ELL` | Sliced-ELL格式 |
| `ACL_SPARSE_FORMAT_BSR` | BSR（Block Sparse Row）格式 |


## 使用示例

```c
// 初始化ACL
aclInit(nullptr);
aclrtSetDevice(deviceId);
aclrtCreateStream(&stream);

// 创建稀疏矩阵处理器并绑定 stream
aclsparseHandle_t handle = nullptr;
aclsparseCreate(&handle);
aclsparseSetStream(handle, stream);

// 创建CSR格式稀疏矩阵
aclsparseSpMatDescr_t matA;
aclsparseCreateCsr(&matA, rows, cols, nnz,
    dA_csrOffsets, dA_columns, dA_values,
    ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_32I,
    ACL_SPARSE_INDEX_BASE_ZERO, ACL_FLOAT);

// 创建稠密向量
aclsparseDnVecDescr_t vecX, vecY;
aclSparseCreateDnVec(&vecX, cols, dX, ACL_FLOAT);
aclSparseCreateDnVec(&vecY, rows, dY, ACL_FLOAT);

// 获取缓冲区大小
size_t bufferSize;
float alpha = 1.0f, beta = 0.0f;
aclSparseSpmvGetBufferSize(handle, ACL_SPARSE_OP_NON_TRANSPOSE,
    &alpha, matA, vecX, &beta, vecY,
    ACL_FLOAT, ACL_SPARSE_SPMV_ALG_CSR_ALG1, &bufferSize);

// 分配缓冲区
void *dBuffer;
aclrtMalloc(&dBuffer, bufferSize, ACL_MEM_MALLOC_HUGE_FIRST);

// 执行SpMV
aclSparseSpmv(handle, ACL_SPARSE_OP_NON_TRANSPOSE,
    &alpha, matA, vecX, &beta, vecY,
    ACL_FLOAT, ACL_SPARSE_SPMV_ALG_CSR_ALG1, dBuffer);

// 销毁资源
aclsparseDestroySpMat(matA);
aclSparseDestroyDnVec(vecX);
aclSparseDestroyDnVec(vecY);
aclsparseDestroy(handle);

// 清理ACL
aclrtFree(dBuffer);
aclrtDestroyStream(stream);
aclrtResetDevice(deviceId);
aclFinalize();
```
