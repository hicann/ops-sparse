# 日志速查表

## 日志级别选择

| 场景 | 宏 | 示例 |
|------|-----|------|
| 参数校验失败 | `OP_LOGE` | `OP_LOGE("aclsparseSpMV", "handle is nullptr")` |
| ACL Runtime 调用失败 | `OP_LOGE` | `OP_LOGE("aclsparseSpMV", "aclrtMalloc failed, ret=%d", ret)` |
| 不支持的格式/类型 | `OP_LOGE` | `OP_LOGE("aclsparseSpMV", "unsupported format: %d", format)` |
| 潜在问题/降级 | `OP_LOGW` | `OP_LOGW("aclsparseSpMV", "nnz=0, returning early")` |
| Kernel 启动 | `OP_LOGI` | `OP_LOGI("aclsparseSpMV", "launching kernel, blocks=%u", numBlocks)` |
| Tiling 参数 | `OP_LOGD` | `OP_LOGD("aclsparseSpMV", "tiling: rows=%u, cols=%u", rows, cols)` |
| 描述符字段 | `OP_LOGD` | `OP_LOGD("aclsparseSpMV", "matA: format=%d, nnz=%lu", format, nnz)` |

## 格式规范

```cpp
OP_LOGx("API名", "格式化字符串", 参数...);
```

- 第一个参数：**API 名或算子名**（如 `"aclsparseSpMV"`）
- 后续参数：printf 风格格式化
- 禁止在日志中使用 `std::endl`，使用 `\n` 或直接省略

## 稀疏算子特有参数

| 参数 | 格式 | 示例 |
|------|------|------|
| 矩阵维度 | `rows=%lu, cols=%lu, nnz=%lu` | `OP_LOGD("...", "mat: rows=%lu, cols=%lu, nnz=%lu", rows, cols, nnz)` |
| 稀疏格式 | `format=%d (CSR/COO/CSC)` | `OP_LOGD("...", "format=%d", inner->format)` |
| 索引类型 | `idxType=%d, base=%d` | `OP_LOGD("...", "idxType=%d, base=%d", ptrType, baseType)` |
| 数据类型 | `valueType=%d` | `OP_LOGD("...", "valueType=%d", valueType)` |
| 核数 | `blockDim=%u` | `OP_LOGI("...", "blockDim=%u", blockDim)` |
