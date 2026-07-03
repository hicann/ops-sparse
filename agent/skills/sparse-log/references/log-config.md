# 日志配置

## CheckLogLevel（条件调试日志）

在高频调用路径中，`OP_LOGD` 即使被禁用仍有格式化开销。使用 `CheckLogLevel` 包裹可避免：

```cpp
#include "log/log.h"

// 仅在 DEBUG 级别启用时才执行日志输出
if (CheckLogLevel(OP, DLOG_DEBUG)) {
    OP_LOGD("aclsparseSpMV", "detailed tiling: %s", tilingToString(tiling).c_str());
}
```

**适用场景**：
- Tiling 数据详细打印（字段多、格式化开销大）
- 循环内的调试日志
- 大数组/矩阵内容打印

**不适用**：
- `OP_LOGE`（错误日志必须始终输出）
- `OP_LOGI`（启动日志开销可接受）
- 简单的 `OP_LOGD`（如单个整数打印）

## 环境变量配置

| 环境变量 | 说明 | 示例 |
|---------|------|------|
| `ASCEND_GLOBAL_LOG_LEVEL` | 全局日志级别（0=DEBUG, 1=INFO, 2=WARNING, 3=ERROR） | `export ASCEND_GLOBAL_LOG_LEVEL=0` |
| `ASCEND_SLOG_PRINT_TO_STDOUT` | 是否输出到 stdout（1=是, 0=否） | `export ASCEND_SLOG_PRINT_TO_STDOUT=1` |

```bash
# 开发调试：开启 DEBUG + stdout 输出
export ASCEND_GLOBAL_LOG_LEVEL=0
export ASCEND_SLOG_PRINT_TO_STDOUT=1

# 生产环境：仅 ERROR + 文件输出
export ASCEND_GLOBAL_LOG_LEVEL=3
export ASCEND_SLOG_PRINT_TO_STDOUT=0
```

## aclsparseLoggerSetLevel（内部 API）

```cpp
#include "aclsparse_logger_manager.h"

// 注意：这是内部 API（不对外暴露），仅用于测试或内部调试
AclSparse::aclsparseLoggerSetLevel(AclSparse::LOG_LEVEL_DEBUG);
```

**注意**：此 API 不在 `cann_ops_sparse.h` 中声明，不应在用户代码中使用。生产环境通过环境变量控制日志级别。
