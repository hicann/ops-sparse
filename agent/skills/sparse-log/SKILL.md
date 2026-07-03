---
name: sparse-log
description: ops-sparse 日志集成规范，定义 dlog 日志使用方式、日志级别、格式要求。
---

# ops-sparse 日志规范

## 日志框架

ops-sparse 使用 **dlog** 作为统一日志框架，对应头文件：

```cpp
#include "log/log.h"
```

## 日志级别

| 宏 | 级别 | 用途 |
|----|------|------|
| `OP_LOGE` | ERROR | 参数校验失败、ACL Runtime 调用失败、不可恢复错误 |
| `OP_LOGW` | WARNING | 潜在问题、降级处理 |
| `OP_LOGI` | INFO | Kernel 启动信息、关键执行节点 |
| `OP_LOGD` | DEBUG | Tiling 数据、详细调试信息 |

## 日志格式

```cpp
OP_LOGE("aclsparseXxx", "Error message: %s, code: %d", detail, code);
OP_LOGD("aclsparseXxx", "Tiling data: rows=%u, cols=%u, blockDim=%u", rows, cols, blockDim);
OP_LOGI("aclsparseXxx", "Kernel launched: blockIdx=%u, stream=%p", blockIdx, stream);
```

**格式规范**：
- 第一个参数为 **API 名或算子名**（如 `"aclsparseSpMV"`）
- 后续参数为格式化字符串 + 可变参数

## Host 侧日志集成

### 强制规则

1. **必须包含** `#include "log/log.h"`
2. **禁止使用** `printf`、`std::cout`、`LOG_PRINT`（旧代码遗留，新代码禁止）
3. 参数校验失败 → `OP_LOGE`
4. ACL Runtime 调用失败 → `OP_LOGE`
5. Tiling 参数 → `OP_LOGD`
6. Kernel launch → `OP_LOGI`

### 示例

```cpp
static aclsparseStatus_t ValidateSpMVParams(aclsparseHandle_t handle, ...) {
    if (handle == nullptr) {
        OP_LOGE("aclsparseSpMV", "handle is nullptr");
        return ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR;
    }
    // ...
}

static aclsparseStatus_t LaunchSpMVKernel(...) {
    OP_LOGD("aclsparseSpMV", "Tiling: rows=%u, cols=%u, blockDim=%u",
            tiling.totalRowsNum, tiling.totalColNum, blockDim);

    spmv_kernel_do(...);

    OP_LOGI("aclsparseSpMV", "Kernel launched, blockDim=%u", blockDim);
    return ACL_SPARSE_STATUS_SUCCESS;
}
```

## Kernel 侧日志

- Kernel 侧通常不使用 dlog（性能考虑）
- 如需调试输出，使用 `AscendC` 内置的 `PRINT` 宏（仅调试模式使用）
- 正式代码中必须移除所有 Kernel 侧调试输出

## 日志级别设置

通过环境变量控制（推荐）：

```bash
export ASCEND_GLOBAL_LOG_LEVEL=0        # 0=DEBUG, 1=INFO, 2=WARNING, 3=ERROR
export ASCEND_SLOG_PRINT_TO_STDOUT=1    # 1=输出到 stdout
```

详见 `references/log-config.md`（含 `CheckLogLevel` 条件调试用法）。

## 参考文档

| 文档 | 说明 |
|------|------|
| `references/log-quickref.md` | 日志级别选择速查表 + 稀疏算子特有参数格式 |
| `references/host-log-templates.md` | 6 种标准 Host 侧日志模板（参数校验/ACL 失败/格式不支持/Tiling/Kernel 启动/早期返回） |
| `references/log-config.md` | `CheckLogLevel` 条件调试 + 环境变量配置 + `aclsparseLoggerSetLevel` 说明 |
| `references/best-practices.md` | printf→OP_LOG 迁移表 + 常见陷阱 + 最佳实践 |
