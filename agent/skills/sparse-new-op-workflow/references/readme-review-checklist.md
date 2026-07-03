# README 审查清单与流程（用于 readme-review 模式）

## 审查清单（9 项）

| 编号 | 检查项 | 检查内容 |
|------|--------|---------|
| 1 | 模板完整性 | 所有章节存在（算子概述、产品支持、函数原型、参数说明、约束、稀疏格式说明、调用示例）；占位符已替换为实际内容 |
| 2 | API 函数签名 | 函数原型与 `cann_ops_sparse.h` 声明逐字匹配；参数名、类型、const 限定符、顺序一致；Generic API 的描述符类型（`aclsparseConstSpMatDescr_t` 等）正确 |
| 3 | 参数类型 | 示例代码中 C++ 类型与函数签名匹配（如 `float*` 非 `uint8_t*`），类型转换使用 `static_cast` |
| 4 | RAII 模式 | 描述符使用 `SpMatManager`/`DnVecManager`/`DnMatManager` 管理，或 `aclsparseDestroy*` 与 `aclsparseCreate*` 严格配对；Device 内存使用 `DeviceBuffer` 或 `aclrtFree` 配对；禁止泄漏 |
| 5 | API 名称正确性 | `aclrtSynchronizeStream`（非 `aclrtStreamSynchronize`）、`aclsparseCreate`、`aclsparseDestroySpMat` 等拼写正确 |
| 6 | 头文件完备 | 仅包含 `<cstdio>`、`<memory>`、`<vector>`、`acl/acl.h`、`cann_ops_sparse.h`；禁止 `<cstdint>` 等冗余头文件 |
| 7 | 交叉引用 | 调用示例标题与代码块之间包含 `compile_and_run_example.md` 链接（格式：`示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](compile_and_run_example.md)。`） |
| 8 | 内存位置标注 | 参数表中每行明确标注 Host 内存 / Device 内存 |
| 9 | 约束描述 | 约束列表与 host.cpp 中 `Validate{Op}Params` 逻辑匹配；无约束时写"无"，不留空；稀疏格式支持列表与实际实现一致 |

## 审查流程（5 阶段）

### 阶段 1：获取输入
- 读取 `src/{operator_name}/README.md`
- 从 `include/cann_ops_sparse.h` 查找对应 API 声明
- 读取 `src/{operator_name}/archXX/{operator_name}_host.cpp` 获取约束验证逻辑

### 阶段 2：创建追踪任务
- 使用 TaskCreate 创建 9 个审查任务（每项一个），初始状态 pending
- 若工具不可用，在上下文中输出检查点清单

### 阶段 3：逐项审查循环
对每个 pending 任务：
1. TaskUpdate 锁定当前任务（status: in_progress）
2. 按清单逐项对照检查，记录证据（代码行号、文档引用）
3. 评定每项状态：通过 / 发现问题（附行号和期望值）
4. TaskUpdate 完成确认（status: completed）

### 阶段 4：生成审查报告
- 输出文件：`.agent/dev-docs/{operator_name}/4.1.1-审查报告.md`
- 格式：每项一行，状态 + 证据（通过则简要说明；发现问题则附行号+期望值）
- 汇总：通过项数 / 总项数，明确标记整体通过或失败

### 阶段 5：输出日志摘要

```markdown
---
## 日志摘要（供任务下发方写入开发日志）
- **状态**: ✅通过 / ❌失败
- **通过项**: N/9
- **发现问题**: （列表，每项一行，附行号）
```
