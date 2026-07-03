# Host 侧日志模板

## 模板 1：参数校验失败

```cpp
static aclsparseStatus_t ValidateSpMVParams(aclsparseHandle_t handle,
                                             aclsparseConstSpMatDescr_t matA,
                                             aclsparseConstDnVecDescr_t vecX) {
    if (handle == nullptr) {
        OP_LOGE("aclsparseSpMV", "handle is nullptr");
        return ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR;
    }
    if (matA == nullptr) {
        OP_LOGE("aclsparseSpMV", "matA is nullptr");
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (vecX == nullptr) {
        OP_LOGE("aclsparseSpMV", "vecX is nullptr");
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}
```

## 模板 2：ACL Runtime 失败

```cpp
aclError ret = aclrtMalloc(&buffer, size, ACL_MEM_MALLOC_HUGE_FIRST);
if (ret != ACL_SUCCESS) {
    OP_LOGE("aclsparseSpMV", "aclrtMalloc failed, size=%zu, ret=%d", size, ret);
    return ACL_SPARSE_STATUS_ALLOC_FAILED;
}
```

## 模板 3：不支持的格式/类型

```cpp
auto* matInner = ToMatInner(matA);
if (matInner->format != ACL_SPARSE_FORMAT_CSR) {
    OP_LOGE("aclsparseSpMV", "unsupported format: %d, only CSR supported", matInner->format);
    return ACL_SPARSE_STATUS_NOT_SUPPORTED;
}
```

## 模板 4：Tiling 参数记录

```cpp
static aclsparseStatus_t LaunchSpMVKernel(aclsparseHandle_t handle, ...) {
    SpmvTilingData tiling = CalSpMVTilingData(rows, cols, nnz, numBlocks);

    OP_LOGD("aclsparseSpMV", "tiling: totalRows=%u, totalCols=%u, rowsPerCore=%u, remainder=%u",
            tiling.totalRows, tiling.totalCols, tiling.rowsPerCore, tiling.remainderRows);
    // ...
}
```

## 模板 5：Kernel 启动

```cpp
OP_LOGI("aclsparseSpMV", "launching kernel: blocks=%u, stream=%p", numBlocks, useStream);
spmv_kernel_do(csrRowPtr, csrColInd, csrValues, xVec, yVec,
               numBlocks, tiling, useStream);
```

## 模板 6：早期返回（nnz=0 或空矩阵）

```cpp
// nnz=0 或 rows=0 时直接返回 SUCCESS，**不输出日志**（避免日志噪音）
if (nnz == 0 || rows == 0) {
    return ACL_SPARSE_STATUS_SUCCESS;
}
```
