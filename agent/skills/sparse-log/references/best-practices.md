# 日志最佳实践与 printf 迁移指南

## printf → OP_LOG 迁移表

仓内现有代码（`src/spmv/`、`src/spmm/`）仍使用 `LOG_PRINT` / `printf`，新代码**禁止使用**，旧代码应逐步迁移：

| ❌ 旧代码 | ✅ 新代码 | 说明 |
|-----------|-----------|------|
| `printf("error: %s\n", msg)` | `OP_LOGE("aclsparseXxx", "error: %s", msg)` | 错误日志 |
| `LOG_PRINT("[ERROR] %s\n", msg)` | `OP_LOGE("aclsparseXxx", "%s", msg)` | LOG_PRINT 是 printf 的包装 |
| `std::cout << "info" << std::endl` | `OP_LOGI("aclsparseXxx", "info")` | C++ 流输出 |
| `fprintf(stderr, "warn\n")` | `OP_LOGW("aclsparseXxx", "warn")` | stderr 输出 |
| `printf("tiling: %u\n", val)` | `OP_LOGD("aclsparseXxx", "tiling: %u", val)` | 调试信息 |

## 迁移步骤

1. **添加头文件**：`#include "log/log.h"`
2. **确定 API 名**：使用当前函数所属的 API 名（如 `"aclsparseSpMV"`）
3. **选择级别**：按 log-quickref.md 的场景表选择
4. **替换格式**：去掉末尾 `\n`（dlog 自动换行），去掉 `std::endl`
5. **删除旧宏定义**：移除 `#define LOG_PRINT(...)` 等旧宏

## 常见陷阱

| 陷阱 | 说明 | 解决 |
|------|------|------|
| 忘记 `#include "log/log.h"` | 编译报错 `OP_LOGE was not declared` | 添加头文件 |
| 日志级别选错 | 错误信息用了 `OP_LOGD`，生产环境看不到 | 错误必须用 `OP_LOGE` |
| 缺少上下文 | `"failed"` 无法定位问题 | 包含 API 名 + 具体参数值 |
| 高频路径无条件 DEBUG | 性能下降 | 用 `CheckLogLevel` 包裹 |
| Kernel 侧使用 dlog | 编译失败或性能严重下降 | Kernel 侧禁止 dlog，仅用 `PRINT` 调试 |

## 最佳实践

1. **每个 API 入口的第一条日志**：`OP_LOGI` 记录 API 名和关键参数
2. **每个错误返回前**：`OP_LOGE` 记录错误原因
3. **Tiling 计算完成后**：`OP_LOGD` 记录 tiling 结构体关键字段
4. **Kernel launch 前**：`OP_LOGI` 记录 numBlocks 和 stream
5. **早期返回（nnz=0）**：不输出日志（避免噪音）
