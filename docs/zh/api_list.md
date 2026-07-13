# 稀疏算子接口（aclsparse）

## 使用说明

为方便调用稀疏算子，提供一套基于 C 的 API（以 aclsparse 为前缀），主要用于稀疏矩阵运算场景。

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
| [aclsparseGetVersion](#aclsparsegetversion) | 获取版本号 |
| [aclsparseCreateDnVec](#aclsparsecreatednvec) | 创建稠密向量 |
| [aclsparseDestroyDnVec](#aclsparsedestroydnvec) | 销毁稠密向量描述符 |
| [aclsparseDnVecGet](#aclsparsednvecget) | 获取稠密向量描述符的全部字段 |
| [aclsparseDnVecGetValues](#aclsparsednvecgetvalues) | 获取稠密向量描述符的 values 指针 |
| [aclsparseDnVecSetValues](#aclsparsednvecsetvalues) | 设置稠密向量描述符的 values 指针 |
| [aclsparseCreateSpVec](#aclsparsecreatespvec) | 创建稀疏向量 |
| [aclsparseDestroySpVec](#aclsparsedestroyspvec) | 销毁稀疏向量描述符 |
| [aclsparseSpVecGet](#aclsparsespvecget) | 获取稀疏向量描述符的全部字段 |
| [aclsparseSpVecGetIndexBase](#aclsparsespvecgetindexbase) | 获取稀疏向量的索引基值 |
| [aclsparseSpVecGetValues](#aclsparsespvecgetvalues) | 获取稀疏向量描述符的 values 指针 |
| [aclsparseSpVecSetValues](#aclsparsespvecsetvalues) | 设置稀疏向量描述符的 values 指针 |
| [aclsparseCreateCsr](#aclsparsecreatecsr) | 创建CSR格式稀疏矩阵 |
| [aclsparseCreateCsc](#aclsparsecreatecsc) | 创建CSC格式稀疏矩阵（**暂未支持**） |
| [aclsparseDestroySpMat](#aclsparsedestroyspmat) | 销毁稀疏矩阵对象 |
| [aclsparseSpMatGetFormat](#aclsparsespmatgetformat) | 获取稀疏矩阵的存储格式 |
| [aclsparseSpMatGetValues](#aclsparsespmatgetvalues) | 获取稀疏矩阵描述符的 values 指针 |
| [aclsparseSpMatSetValues](#aclsparsespmatsetvalues) | 设置稀疏矩阵描述符的 values 指针 |
| [aclsparseSpMatGetSize](#aclsparsespmatgetsize) | 获取稀疏矩阵的行数、列数和非零元素个数 |
| [aclsparseSpMVGetBufferSize](#aclsparsespmvgetbuffersize) | 获取SpMV缓冲区大小（**暂未支持**） |
| [aclsparseSpMVPreprocess](#aclsparsespmvpreprocess) | SpMV预处理（**暂未支持**） |
| [aclsparseSpMV](#aclsparsespmv) | 稀疏矩阵向量乘法 |
| [aclsparseCreateDnMat](#aclsparsecreatednmat) | 创建稠密矩阵 |
| [aclsparseDestroyDnMat](#aclsparsedestroydnmat) | 销毁稠密矩阵描述符 |
| [aclsparseDnMatGet](#aclsparsednmatget) | 获取稠密矩阵描述符的全部字段 |
| [aclsparseDnMatGetValues](#aclsparsednmatgetvalues) | 获取稠密矩阵描述符的 values 指针 |
| [aclsparseDnMatSetValues](#aclsparsednmatsetvalues) | 设置稠密矩阵描述符的 values 指针 |
| [aclsparseSpMMGetBufferSize](#aclsparsespmmgetbuffersize) | 获取SpMM缓冲区大小 |
| [aclsparseSpMMPreprocess](#aclsparsespmmpreprocess) | SpMM预处理 |
| [aclsparseSpMM](#aclsparsespmm) | 稀疏矩阵-稠密矩阵乘法 |

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

### aclsparseGetVersion

```c
aclsparseStatus_t aclsparseGetVersion(aclsparseHandle_t handle, int *version);
```

**功能**：获取 ops-sparse 版本号。版本号编码为 `MAJOR * 10000 + MINOR * 100 + PATCH`，如 `10000` 表示 1.0.0。

**参数说明**：

- `handle`（IN）：HOST，稀疏矩阵处理器句柄，可为 `nullptr`。
- `version`（OUT）：HOST，输出参数，接收版本号。

**返回值**：

- `ACL_SPARSE_STATUS_SUCCESS`：成功
- `ACL_SPARSE_STATUS_INVALID_VALUE`：`version` 为空
- 其他值：失败

---

### aclsparseCreateDnVec

```c
aclsparseStatus_t aclsparseCreateDnVec(
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

### aclsparseDestroyDnVec

```c
aclsparseStatus_t aclsparseDestroyDnVec(aclsparseConstDnVecDescr_t dnVecDescr);
```

**功能**：销毁稠密向量描述符。

**参数说明**：

- `dnVecDescr`（IN）：HOST，要销毁的稠密向量描述符。

**返回值**：

- `ACL_SPARSE_STATUS_SUCCESS`：成功
- 其他值：失败

---

### aclsparseDnVecGet

```c
aclsparseStatus_t aclsparseDnVecGet(
    aclsparseDnVecDescr_t dnVecDescr,
    int64_t *size,
    void **values,
    aclDataType *valueType
);
```

**功能**：获取稠密向量描述符的全部字段。

**参数说明**：

- `dnVecDescr`（IN）：HOST，稠密向量描述符。
- `size`（OUT）：HOST，向量的大小。
- `values`（OUT）：DEVICE，向量的值指针。
- `valueType`（OUT）：HOST，值的数据类型。

**返回值**：

- `ACL_SPARSE_STATUS_SUCCESS`：成功
- `ACL_SPARSE_STATUS_INVALID_VALUE`：`dnVecDescr` 为空指针

---

### aclsparseDnVecGetValues

```c
aclsparseStatus_t aclsparseDnVecGetValues(
    aclsparseDnVecDescr_t dnVecDescr,
    void **values
);
```

**功能**：获取稠密向量描述符的 values 指针。

**参数说明**：

- `dnVecDescr`（IN）：HOST，稠密向量描述符。
- `values`（OUT）：DEVICE，向量的值指针。

**返回值**：

- `ACL_SPARSE_STATUS_SUCCESS`：成功
- `ACL_SPARSE_STATUS_INVALID_VALUE`：`dnVecDescr` 为空指针

---

### aclsparseDnVecSetValues

```c
aclsparseStatus_t aclsparseDnVecSetValues(
    aclsparseDnVecDescr_t dnVecDescr,
    void *values
);
```

**功能**：设置稠密向量描述符的 values 指针。

**参数说明**：

- `dnVecDescr`（IN）：HOST，稠密向量描述符。
- `values`（IN）：DEVICE，向量的值指针。

**返回值**：

- `ACL_SPARSE_STATUS_SUCCESS`：成功
- `ACL_SPARSE_STATUS_INVALID_VALUE`：`dnVecDescr` 为空指针

---

### aclsparseCreateSpVec

```c
aclsparseStatus_t aclsparseCreateSpVec(
    aclsparseSpVecDescr_t *spVecDescr,
    int64_t size,
    int64_t nnz,
    void *indices,
    void *values,
    aclsparseIndexType_t idxType,
    aclsparseIndexBase_t idxBase,
    aclDataType valueType
);
```

**功能**：创建一个稀疏向量描述符。

**参数说明**：

- `spVecDescr`（IN/OUT）：HOST，稀疏向量描述符。
- `size`（IN）：HOST，稀疏向量的大小（元素总数）。
- `nnz`（IN）：HOST，稀疏向量的非零元素个数。
- `indices`（IN）：DEVICE，索引数组，长度为 `nnz`。
- `values`（IN）：DEVICE，值数组，长度为 `nnz`。
- `idxType`（IN）：HOST，`indices` 的数据类型。
- `idxBase`（IN）：HOST，`indices` 的索引基值。
- `valueType`（IN）：HOST，`values` 的数据类型。

**返回值**：

- `ACL_SPARSE_STATUS_SUCCESS`：成功
- `ACL_SPARSE_STATUS_INVALID_VALUE`：`spVecDescr` 为空指针，或 `size < 0`，或 `nnz < 0`，或 `nnz > size`
- `ACL_SPARSE_STATUS_ALLOC_FAILED`：内存分配失败

---

### aclsparseDestroySpVec

```c
aclsparseStatus_t aclsparseDestroySpVec(aclsparseConstSpVecDescr_t spVecDescr);
```

**功能**：销毁稀疏向量描述符。接受 const 描述符，非 const 描述符同样支持。

**参数说明**：

- `spVecDescr`（IN）：HOST，要销毁的稀疏向量描述符。

**返回值**：

- `ACL_SPARSE_STATUS_SUCCESS`：成功

---

### aclsparseSpVecGet

```c
aclsparseStatus_t aclsparseSpVecGet(
    aclsparseSpVecDescr_t spVecDescr,
    int64_t *size,
    int64_t *nnz,
    void **indices,
    void **values,
    aclsparseIndexType_t *idxType,
    aclsparseIndexBase_t *idxBase,
    aclDataType *valueType
);
```

**功能**：获取稀疏向量描述符的全部字段。输出参数允许为 `nullptr`，此时仅返回请求的字段。

**参数说明**：

- `spVecDescr`（IN）：HOST，稀疏向量描述符。
- `size`（OUT）：HOST，稀疏向量的大小。
- `nnz`（OUT）：HOST，非零元素个数。
- `indices`（OUT）：DEVICE，索引数组指针。
- `values`（OUT）：DEVICE，值数组指针。
- `idxType`（OUT）：HOST，索引数据类型。
- `idxBase`（OUT）：HOST，索引基值。
- `valueType`（OUT）：HOST，值数据类型。

**返回值**：

- `ACL_SPARSE_STATUS_SUCCESS`：成功
- `ACL_SPARSE_STATUS_INVALID_VALUE`：`spVecDescr` 为空指针

---

### aclsparseSpVecGetIndexBase

```c
aclsparseStatus_t aclsparseSpVecGetIndexBase(
    aclsparseConstSpVecDescr_t spVecDescr,
    aclsparseIndexBase_t *idxBase
);
```

**功能**：获取稀疏向量描述符的索引基值。接受 const 描述符，非 const 描述符同样支持。

**参数说明**：

- `spVecDescr`（IN）：HOST，稀疏向量描述符。
- `idxBase`（OUT）：HOST，索引基值。

**返回值**：

- `ACL_SPARSE_STATUS_SUCCESS`：成功
- `ACL_SPARSE_STATUS_INVALID_VALUE`：`spVecDescr` 为空指针

---

### aclsparseSpVecGetValues

```c
aclsparseStatus_t aclsparseSpVecGetValues(
    aclsparseSpVecDescr_t spVecDescr,
    void **values
);
```

**功能**：获取稀疏向量描述符的 values 指针。

**参数说明**：

- `spVecDescr`（IN）：HOST，稀疏向量描述符。
- `values`（OUT）：DEVICE，值数组指针，长度为 `nnz`。

**返回值**：

- `ACL_SPARSE_STATUS_SUCCESS`：成功
- `ACL_SPARSE_STATUS_INVALID_VALUE`：`spVecDescr` 为空指针

---

### aclsparseSpVecSetValues

```c
aclsparseStatus_t aclsparseSpVecSetValues(
    aclsparseSpVecDescr_t spVecDescr,
    void *values
);
```

**功能**：设置稀疏向量描述符的 values 指针。

**参数说明**：

- `spVecDescr`（IN）：HOST，稀疏向量描述符。
- `values`（IN）：DEVICE，值数组指针，长度为 `nnz`。

**返回值**：

- `ACL_SPARSE_STATUS_SUCCESS`：成功
- `ACL_SPARSE_STATUS_INVALID_VALUE`：`spVecDescr` 为空指针

---

### 只读(const)稀疏向量描述符构造接口

只读(const)构造接口：数据指针为 `const`，构造出的描述符是 `const` 变体，只能传给接收 `const` 形参的接口。销毁接口（`aclsparseDestroySpVec`）统一接收 `const` 变体，因此 const 与非 const 描述符都可销毁。

```c
aclsparseStatus_t aclsparseCreateConstSpVec(aclsparseConstSpVecDescr_t *spVecDescr, int64_t size,
    int64_t nnz, const void *indices, const void *values, aclsparseIndexType_t idxType,
    aclsparseIndexBase_t idxBase, aclDataType valueType);

aclsparseStatus_t aclsparseConstSpVecGet(aclsparseConstSpVecDescr_t spVecDescr, int64_t *size,
    int64_t *nnz, const void **indices, const void **values, aclsparseIndexType_t *idxType,
    aclsparseIndexBase_t *idxBase, aclDataType *valueType);

aclsparseStatus_t aclsparseConstSpVecGetValues(aclsparseConstSpVecDescr_t spVecDescr,
    const void **values);
```

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

> **索引类型支持**：当前版本 SpMV / SpMM 仅支持 `ACL_SPARSE_INDEX_32I`，且 `csrRowOffsetsType` 与 `csrColIndType` 必须相同。传入 `ACL_SPARSE_INDEX_64I` 或 ptr/col 类型不一致时，`aclsparseCreateCsr` 返回 `ACL_SPARSE_STATUS_NOT_SUPPORTED`。

**参数说明**：

- `spMatDescr`（IN/OUT）：HOST，稀疏矩阵描述符。
- `rows`（IN）：HOST，矩阵的行数。
- `cols`（IN）：HOST，矩阵的列数。
- `nnz`（IN）：HOST，矩阵的非零元素个数。
- `csrRowOffsets`（IN）：DEVICE，指向CSR格式行偏移量数组的指针。
- `csrColInd`（IN）：DEVICE，指向CSR格式列索引数组的指针。
- `csrValues`（IN）：DEVICE，指向CSR格式非零元素数组的指针。
- `csrRowOffsetsType`（IN）：HOST，行偏移量数组的数据类型（当前仅支持 `ACL_SPARSE_INDEX_32I`）。
- `csrColIndType`（IN）：HOST，列索引数组的数据类型（当前仅支持 `ACL_SPARSE_INDEX_32I`，且须与 `csrRowOffsetsType` 一致）。
- `idxBase`（IN）：HOST，索引的基值（0或1）。
- `valueType`（IN）：HOST，非零元素的数据类型。

**返回值**：

- `ACL_SPARSE_STATUS_SUCCESS`：成功
- 其他值：失败

---

### aclsparseCreateCsc

> **支持状态**：当前版本暂未实现，敬请期待。当前版本 SpMV / SpMM 仅实现 CSR 稀疏矩阵；CSC / COO 等格式未提供算子路径。调用本接口返回 `ACL_SPARSE_STATUS_NOT_SUPPORTED`。

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

**功能**：创建一个CSC（Compressed Sparse Column）格式的稀疏矩阵（**当前版本暂未支持**，见上方说明）。

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

### aclsparseSpMatGetFormat

```c
aclsparseStatus_t aclsparseSpMatGetFormat(
    aclsparseConstSpMatDescr_t spMatDescr,
    aclsparseFormat_t *format
);
```

**功能**：获取稀疏矩阵描述符的存储格式。接受 const 描述符，非 const 描述符同样支持。

**参数说明**：

- `spMatDescr`（IN）：HOST，稀疏矩阵描述符。
- `format`（OUT）：HOST，存储格式。

**返回值**：

- `ACL_SPARSE_STATUS_SUCCESS`：成功
- `ACL_SPARSE_STATUS_INVALID_VALUE`：`spMatDescr` 为空指针

---

### aclsparseSpMatGetValues

```c
aclsparseStatus_t aclsparseSpMatGetValues(
    aclsparseSpMatDescr_t spMatDescr,
    void **values
);
```

**功能**：获取稀疏矩阵描述符的 values 指针。

**参数说明**：

- `spMatDescr`（IN）：HOST，稀疏矩阵描述符。
- `values`（OUT）：DEVICE，值数组指针，长度为 `nnz`。

**返回值**：

- `ACL_SPARSE_STATUS_SUCCESS`：成功
- `ACL_SPARSE_STATUS_INVALID_VALUE`：`spMatDescr` 为空指针

---

### aclsparseSpMatSetValues

```c
aclsparseStatus_t aclsparseSpMatSetValues(
    aclsparseSpMatDescr_t spMatDescr,
    void *values
);
```

**功能**：设置稀疏矩阵描述符的 values 指针。

**参数说明**：

- `spMatDescr`（IN）：HOST，稀疏矩阵描述符。
- `values`（IN）：DEVICE，值数组指针，长度为 `nnz`。

**返回值**：

- `ACL_SPARSE_STATUS_SUCCESS`：成功
- `ACL_SPARSE_STATUS_INVALID_VALUE`：`spMatDescr` 为空指针

---

### aclsparseSpMatGetSize

```c
aclsparseStatus_t aclsparseSpMatGetSize(
    aclsparseConstSpMatDescr_t spMatDescr,
    int64_t *rows,
    int64_t *cols,
    int64_t *nnz
);
```

**功能**：获取稀疏矩阵的行数、列数和非零元素个数。接受 const 描述符，非 const 描述符同样支持。输出参数允许为 `nullptr`，此时仅返回请求的字段。

**参数说明**：

- `spMatDescr`（IN）：HOST，稀疏矩阵描述符。
- `rows`（OUT）：HOST，稀疏矩阵的行数。
- `cols`（OUT）：HOST，稀疏矩阵的列数。
- `nnz`（OUT）：HOST，稀疏矩阵的非零元素个数。

**返回值**：

- `ACL_SPARSE_STATUS_SUCCESS`：成功
- `ACL_SPARSE_STATUS_INVALID_VALUE`：`spMatDescr` 为空指针

---

### 只读(const)描述符构造接口

只读(const)构造接口：数据指针为 `const`，构造出的描述符是 `const` 变体，只能传给接收 `const` 形参的接口（如 SpMV/SpMM 的输入 `mat`/`matA`/`x`/`matB`）。销毁接口（`aclsparseDestroyDnVec` / `aclsparseDestroySpMat` / `aclsparseDestroyDnMat` / `aclsparseDestroySpVec`）统一接收 `const` 变体，因此 const 与非 const 描述符都可销毁。`aclsparseCreateConstCsc` 当前版本暂未支持，调用返回 `ACL_SPARSE_STATUS_NOT_SUPPORTED`。

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

aclsparseStatus_t aclsparseConstSpMatGetValues(aclsparseConstSpMatDescr_t spMatDescr,
    const void **values);
```

> **支持状态**：`aclsparseCreateConstCsc` 暂未支持。调用返回 `ACL_SPARSE_STATUS_NOT_SUPPORTED`。

---

### aclsparseSpMVGetBufferSize

> **支持状态**：暂未支持。当前版本仅提供头文件声明，库中尚无实现；调用会导致链接失败。

```c
aclsparseStatus_t aclsparseSpMVGetBufferSize(
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

### aclsparseSpMVPreprocess

> **支持状态**：暂未支持。当前版本仅提供头文件声明，库中尚无实现；调用会导致链接失败。

```c
aclsparseStatus_t aclsparseSpMVPreprocess(
    aclsparseHandle_t handle,
    aclsparseOperation_t opA,
    const void *alpha,
    aclsparseConstSpMatDescr_t matA,
    aclsparseConstDnVecDescr_t vecX,
    const void *beta,
    aclsparseDnVecDescr_t vecY,
    aclDataType computeType,
    aclsparseSpMVAlg_t alg,
    void *externalBuffer
);
```

**功能**：对稀疏矩阵进行预处理，以便在后续的 SpMV 计算中复用工作缓冲区、加速计算。

**参数说明**：

- `handle`（IN）：HOST，稀疏计算句柄。
- `opA`（IN）：HOST，稀疏矩阵操作类型。
- `alpha`（IN）：HOST/DEVICE，标量 alpha 参数。
- `matA`（IN）：HOST，稀疏矩阵描述符。
- `vecX`（IN）：HOST，输入向量 x 的描述符。
- `beta`（IN）：HOST/DEVICE，标量 beta 参数。
- `vecY`（IN）：HOST，输出向量 y 的描述符。
- `computeType`（IN）：HOST，计算的数据类型。
- `alg`（IN）：HOST，SpMV 算法类型。
- `externalBuffer`（IN）：DEVICE，工作缓冲区（需先通过 `aclsparseSpMVGetBufferSize` 分配）。

**返回值**：

- `ACL_SPARSE_STATUS_SUCCESS`：成功
- 其他值：失败

---

### aclsparseSpMV

```c
aclsparseStatus_t aclsparseSpMV(
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

**功能**：稀疏矩阵向量乘法（SpMV）。稀疏矩阵 `mat` 须为 **CSR** 格式（CSC / COO 等格式将返回 `ACL_SPARSE_STATUS_NOT_SUPPORTED`）。

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

### aclsparseCreateDnMat

```c
aclsparseStatus_t aclsparseCreateDnMat(
    aclsparseDnMatDescr_t *dnMatDescr,
    int64_t rows,
    int64_t cols,
    int64_t ld,
    void *values,
    aclDataType valueType,
    aclsparseOrder_t order
);
```

**功能**：创建稠密矩阵描述符，用于描述 SpMM 中的 B / C 稠密矩阵。

**参数说明**：

- `dnMatDescr`（IN/OUT）：HOST，输出的稠密矩阵描述符。
- `rows`（IN）：HOST，矩阵的行数。
- `cols`（IN）：HOST，矩阵的列数。
- `ld`（IN）：HOST，leading dimension。行主序时需 `>= cols`；列主序时需 `>= rows`。
- `values`（IN）：DEVICE，矩阵数据指针。
- `valueType`（IN）：HOST，元素数据类型。作为 B 矩阵时支持 `ACL_FLOAT` / `ACL_FLOAT16` / `ACL_INT8`；作为 C 矩阵时支持 `ACL_FLOAT` / `ACL_FLOAT16` / `ACL_INT32`（`ACL_INT32` 仅在 int8 计算路径中作为 C 矩阵类型使用）。
- `order`（IN）：HOST，布局：`ACL_SPARSE_ORDER_ROW`（行主序）/ `ACL_SPARSE_ORDER_COL`（列主序）。

**返回值**：

- `ACL_SPARSE_STATUS_SUCCESS`：成功
- 其他值：失败

---

### aclsparseDestroyDnMat

```c
aclsparseStatus_t aclsparseDestroyDnMat(aclsparseConstDnMatDescr_t dnMatDescr);
```

**功能**：销毁稠密矩阵描述符。

**参数说明**：

- `dnMatDescr`（IN）：HOST，要销毁的稠密矩阵描述符。

**返回值**：

- `ACL_SPARSE_STATUS_SUCCESS`：成功
- 其他值：失败

---

### aclsparseDnMatGet

```c
aclsparseStatus_t aclsparseDnMatGet(
    aclsparseDnMatDescr_t dnMatDescr,
    int64_t *rows,
    int64_t *cols,
    int64_t *ld,
    void **values,
    aclDataType *valueType,
    aclsparseOrder_t *order
);
```

**功能**：获取稠密矩阵描述符的全部字段。

**参数说明**：

- `dnMatDescr`（IN）：HOST，稠密矩阵描述符。
- `rows`（OUT）：HOST，矩阵的行数。
- `cols`（OUT）：HOST，矩阵的列数。
- `ld`（OUT）：HOST，leading dimension。
- `values`（OUT）：DEVICE，矩阵数据指针。
- `valueType`（OUT）：HOST，元素数据类型。
- `order`（OUT）：HOST，内存布局。

**返回值**：

- `ACL_SPARSE_STATUS_SUCCESS`：成功
- `ACL_SPARSE_STATUS_INVALID_VALUE`：`dnMatDescr` 为空指针

---

### aclsparseDnMatGetValues

```c
aclsparseStatus_t aclsparseDnMatGetValues(
    aclsparseDnMatDescr_t dnMatDescr,
    void **values
);
```

**功能**：获取稠密矩阵描述符的 values 指针。

**参数说明**：

- `dnMatDescr`（IN）：HOST，稠密矩阵描述符。
- `values`（OUT）：DEVICE，矩阵数据指针。

**返回值**：

- `ACL_SPARSE_STATUS_SUCCESS`：成功
- `ACL_SPARSE_STATUS_INVALID_VALUE`：`dnMatDescr` 为空指针

---

### aclsparseDnMatSetValues

```c
aclsparseStatus_t aclsparseDnMatSetValues(
    aclsparseDnMatDescr_t dnMatDescr,
    void *values
);
```

**功能**：设置稠密矩阵描述符的 values 指针。

**参数说明**：

- `dnMatDescr`（IN）：HOST，稠密矩阵描述符。
- `values`（IN）：DEVICE，矩阵数据指针。

**返回值**：

- `ACL_SPARSE_STATUS_SUCCESS`：成功
- `ACL_SPARSE_STATUS_INVALID_VALUE`：`dnMatDescr` 为空指针

---

### aclsparseSpMMGetBufferSize

```c
aclsparseStatus_t aclsparseSpMMGetBufferSize(
    aclsparseHandle_t handle,
    aclsparseOperation_t opA,
    aclsparseOperation_t opB,
    const void *alpha,
    aclsparseConstSpMatDescr_t matA,
    aclsparseConstDnMatDescr_t matB,
    const void *beta,
    aclsparseDnMatDescr_t matC,
    aclDataType computeType,
    aclsparseSpMMAlg_t alg,
    size_t *size
);
```

**功能**：获取稀疏矩阵-稠密矩阵乘法（SpMM）操作所需的 workspace 缓冲区大小。

**参数说明**：

- `handle`（IN）：HOST，稀疏计算句柄。
- `opA`（IN）：HOST，稀疏矩阵 A 的操作类型，当前版本仅支持 `ACL_SPARSE_OP_NON_TRANSPOSE`。
- `opB`（IN）：HOST，稠密矩阵 B 的操作类型，支持 `ACL_SPARSE_OP_NON_TRANSPOSE` / `ACL_SPARSE_OP_TRANSPOSE`。
- `alpha`（IN）：HOST/DEVICE，标量 alpha 参数。
- `matA`（IN）：HOST，CSR 稀疏矩阵 A 的描述符。
- `matB`（IN）：HOST，输入稠密矩阵 B 的描述符。
- `beta`（IN）：HOST/DEVICE，标量 beta 参数。
- `matC`（IN）：HOST，输入/输出稠密矩阵 C 的描述符。
- `computeType`（IN）：HOST，计算的数据类型。
- `alg`（IN）：HOST，SpMM 算法类型。
- `size`（OUT）：HOST，所需的 workspace 缓冲区大小（字节）。

**返回值**：

- `ACL_SPARSE_STATUS_SUCCESS`：成功
- 其他值：失败

---

### aclsparseSpMMPreprocess

```c
aclsparseStatus_t aclsparseSpMMPreprocess(
    aclsparseHandle_t handle,
    aclsparseOperation_t opA,
    aclsparseOperation_t opB,
    const void *alpha,
    aclsparseConstSpMatDescr_t matA,
    aclsparseConstDnMatDescr_t matB,
    const void *beta,
    aclsparseDnMatDescr_t matC,
    aclDataType computeType,
    aclsparseSpMMAlg_t alg,
    void *buffer
);
```

**功能**：对 CSR 稀疏矩阵进行预处理（行重排与分桶），以便在后续的 SpMM 计算中使用。

**参数说明**：

- `handle`（IN）：HOST，稀疏计算句柄。
- `opA`（IN）：HOST，稀疏矩阵 A 的操作类型。
- `opB`（IN）：HOST，稠密矩阵 B 的操作类型。
- `alpha`（IN）：HOST/DEVICE，标量 alpha 参数。
- `matA`（IN）：HOST，CSR 稀疏矩阵 A 的描述符。
- `matB`（IN）：HOST，输入稠密矩阵 B 的描述符。
- `beta`（IN）：HOST/DEVICE，标量 beta 参数。
- `matC`（IN）：HOST，输入/输出稠密矩阵 C 的描述符。
- `computeType`（IN）：HOST，计算的数据类型。
- `alg`（IN）：HOST，SpMM 算法类型。
- `buffer`（IN）：DEVICE，用于存储预处理结果的 workspace 缓冲区。

**返回值**：

- `ACL_SPARSE_STATUS_SUCCESS`：成功
- 其他值：失败

---

### aclsparseSpMM

```c
aclsparseStatus_t aclsparseSpMM(
    aclsparseHandle_t handle,
    aclsparseOperation_t opA,
    aclsparseOperation_t opB,
    const void *alpha,
    aclsparseConstSpMatDescr_t matA,
    aclsparseConstDnMatDescr_t matB,
    const void *beta,
    aclsparseDnMatDescr_t matC,
    aclDataType computeType,
    aclsparseSpMMAlg_t alg,
    void *buffer
);
```

**功能**：稀疏矩阵-稠密矩阵乘法（SpMM），计算 `C = alpha * op(A) * op(B) + beta * C`。稀疏矩阵 A 须为 **CSR** 格式（CSC / COO 等格式传入 `matA` 将返回 `ACL_SPARSE_STATUS_NOT_SUPPORTED`）。

**参数说明**：

- `handle`（IN）：HOST，稀疏矩阵处理器句柄。
- `opA`（IN）：HOST，稀疏矩阵 A 的操作类型，当前版本仅支持 `ACL_SPARSE_OP_NON_TRANSPOSE`。
- `opB`（IN）：HOST，稠密矩阵 B 的操作类型。
- `alpha`（IN）：HOST/DEVICE，标量 alpha。
- `matA`（IN）：HOST，CSR 稀疏矩阵 A 的描述符。
- `matB`（IN）：HOST，输入稠密矩阵 B 的描述符。
- `beta`（IN）：HOST/DEVICE，标量 beta。
- `matC`（IN/OUT）：HOST，输入/输出稠密矩阵 C 的描述符。
- `computeType`（IN）：HOST，计算类型。
- `alg`（IN）：HOST，SpMM 算法类型。
- `buffer`（IN）：DEVICE，工作缓冲区指针（需先调用 `aclsparseSpMMGetBufferSize` 分配，建议先调用 `aclsparseSpMMPreprocess` 预处理）。

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
| `ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR` | handle 为空 |

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

### aclsparseSpMMAlg_t

SpMM算法枚举：

| 枚举值 | 说明 |
|--------|------|
| `ACL_SPARSE_SPMM_ALG_DEFAULT` | 默认算法，推荐使用；当前版本对 CSR 格式走 SIMT/AIV 实现 |
| `ACL_SPARSE_SPMM_CSR_ALG1` | CSR 算法 1，显式指定 CSR 路径；当前版本与 DEFAULT 同一实现 |
| `ACL_SPARSE_SPMM_CSR_FP32_HIGH_PRECISION_ALG` | fp32 高精度算法；同一 SIMT Kernel，fp32 累加使用 Kahan 补偿求和；仅对 fp32 生效 |

### aclsparseOrder_t

稠密矩阵布局枚举（SpMM 中 B / C 使用）：

| 枚举值 | 说明 |
|--------|------|
| `ACL_SPARSE_ORDER_ROW` | 行主序 |
| `ACL_SPARSE_ORDER_COL` | 列主序 |

### aclsparseIndexType_t

索引类型枚举：

| 枚举值 | 说明 |
|--------|------|
| `ACL_SPARSE_INDEX_32I` | 32位有符号整数 [0, 2^31 - 1]；**当前 SpMV / SpMM 已实现** |
| `ACL_SPARSE_INDEX_64I` | 64位有符号整数 [0, 2^63 - 1]；**暂未支持**（`aclsparseCreateCsr` 传入时返回 `ACL_SPARSE_STATUS_NOT_SUPPORTED`） |

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

---

## 推荐调用流程

1. 调用 `aclsparseCreate` 创建句柄，并通过 `aclsparseSetStream` 绑定 stream。
2. 使用 `aclsparseCreateCsr` 创建 CSR 稀疏矩阵 A，使用 `aclsparseCreateDnMat` 创建稠密矩阵 B、C。
3. 调用 `aclsparseSpMMGetBufferSize` 获取 workspace 大小并分配设备内存。
4. 调用 `aclsparseSpMMPreprocess` 对稀疏矩阵进行预处理（同一 sparsity pattern 可复用 workspace）。
5. 调用 `aclsparseSpMM` 执行稀疏矩阵-稠密矩阵乘法。
6. 依次销毁矩阵描述符与句柄，释放 workspace 与 ACL 资源。

---

## 最小示例（伪代码）

```c
aclsparseHandle_t handle = nullptr;
aclsparseCreate(&handle);
aclsparseSetStream(handle, stream);

// C = alpha * A * B + beta * C：A 为 rows × cols，B 为 cols × colsOut，C 为 rows × colsOut
aclsparseSpMatDescr_t matA;
aclsparseCreateCsr(&matA, rows, cols, nnz,
    dA_csrOffsets, dA_columns, dA_values,
    ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_32I,
    ACL_SPARSE_INDEX_BASE_ZERO, ACL_FLOAT);

aclsparseDnMatDescr_t matB, matC;
aclsparseCreateDnMat(&matB, cols, colsOut, colsOut, dB, ACL_FLOAT, ACL_SPARSE_ORDER_ROW);
aclsparseCreateDnMat(&matC, rows, colsOut, colsOut, dC, ACL_FLOAT, ACL_SPARSE_ORDER_ROW);

size_t bufferSize = 0;
float alpha = 1.0f, beta = 0.0f;
aclsparseSpMMGetBufferSize(handle,
    ACL_SPARSE_OP_NON_TRANSPOSE, ACL_SPARSE_OP_NON_TRANSPOSE,
    &alpha, matA, matB, &beta, matC,
    ACL_FLOAT, ACL_SPARSE_SPMM_ALG_DEFAULT, &bufferSize);

void *dBuffer = nullptr;
aclrtMalloc(&dBuffer, bufferSize, ACL_MEM_MALLOC_HUGE_FIRST);
aclsparseSpMMPreprocess(handle,
    ACL_SPARSE_OP_NON_TRANSPOSE, ACL_SPARSE_OP_NON_TRANSPOSE,
    &alpha, matA, matB, &beta, matC,
    ACL_FLOAT, ACL_SPARSE_SPMM_ALG_DEFAULT, dBuffer);
aclsparseSpMM(handle,
    ACL_SPARSE_OP_NON_TRANSPOSE, ACL_SPARSE_OP_NON_TRANSPOSE,
    &alpha, matA, matB, &beta, matC,
    ACL_FLOAT, ACL_SPARSE_SPMM_ALG_DEFAULT, dBuffer);

aclsparseDestroyDnMat(matB);
aclsparseDestroyDnMat(matC);
aclsparseDestroySpMat(matA);
aclsparseDestroy(handle);
aclrtFree(dBuffer);
```

---

## 备注

- 接口能力与属性支持范围以当前实现版本为准，详细限制请参考各算子目录下的README文档（如`test/spmv/README.md`、`test/spmm/README.md`）及头文件内注释。
- 若文档描述与头文件声明不一致，请**以头文件声明与实际实现行为为准**。
