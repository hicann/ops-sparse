---
name: sparse-ascendc-coding-rules
description: |
  ops-sparse Ascend C 编码规范 + MR 安全编码规则速查索引。
  **非开发期常驻上下文**：作为代码提交前的自查工具，按需读取。
  触发时机：① 2.x.2 联调前 ② 2.x.3 验收前 ③ 3.1/4.2 代码检视前 ④ reviewer 检视 MR 代码时。
  详细规则内容位于 references/ 目录，由调用方按需加载相关文档。
---

# Sparse Ascend C 编码规范

本规则为**自查清单**，不加载为开发期常驻上下文。仅在以下节点由 agent 按需调用：

| 调用时机 | 适用角色 | 调用方式 |
|---------|---------|---------|
| 2.x.1.A 算子初稿完成后 | developer | 加载本 skill，按 `references/checklist.md` 自查一次 |
| 2.x.2 联调前 | developer | 同上 |
| 2.x.3 测试验收前 | developer | 同上 + 加载 `references/mr-rules-essential.md` 做 MR 合规审查 |
| CP2.x 打点前 | developer | 同上 |
| 3.1 / 4.2 代码检视前 | reviewer | 加载本 skill，对照 `references/mr-rules-essential.md` + `references/mr-rules-general.md` 做检视 |

## references 索引

### AscendC 编码规则（R1-R10）

- 每条规则独立成一个 reference 文件，包含错误/正确示例：

| 编号 | 规则 | reference |
|------|------|-----------|
| R1 | 禁止逐元素操作 | `references/R1-禁止逐元素操作.md` |
| R2 | 动态获取 CoreNum（禁止硬编码） | `references/R2-动态获取CoreNum.md` |
| R3 | TPipe 禁止作为成员变量 | `references/R3-TPipe禁止成员变量.md` |
| R4 | TilingData 禁止使用数组做核间分配 | `references/R4-TilingData禁止数组.md` |
| R5-R10 | 圈复杂度/嵌套深度/函数行数/除零防御/许可证头/extern 引用 | `references/ascendc-r5-r10.md` |

### MR 安全编码规则

- 按**严重等级**分组，方便 reviewer 按优先级检视：

| reference | 包含内容 | 适用场景 |
|-----------|---------|---------|
| `references/mr-rules-essential.md` | 严重/致命级规则（G.PRE.05、G.INC.*、G.FUU.09/10/12/13/15、G.MEM.04、G.STD.*、OAT 等 ~18 条） | MR 提交前必查 |
| `references/mr-rules-general.md` | 一般/建议级规则（G.EXP.*、G.CTL.03、G.AST.03、G.FUU.11/14、CQ.*、CIP.01 等 ~17 条） | 代码检视深度审查 |

### 检查流程与修复指南

| reference | 内容 | 适用场景 |
|-----------|------|---------|
| `references/checklist.md` | 8 步检查流程（文件级 → 头文件 → 函数级 → 表达式 → 安全函数 → 内存 → 标准库 → 冗余告警） | 提交前系统自查 |
| `references/fix-guide.md` | 14 种常见违规的修复方法对照表 | 发现违规后查修复方案 |

## 使用示例

### 场景 1：developer 自查

在 2.x.1.A 完成后，调用本 skill 执行自检：

```
1. 加载 `references/checklist.md`，按 8 步流程逐条检查代码
2. 发现 R6 嵌套深度超标 → 加载 `references/ascendc-r5-r10.md` 查看修复方法
3. 发现 G.FUU.09 使用了 realloc → 加载 `references/mr-rules-essential.md` 确认级别与修复方案
4. 修复完成后再次执行 checklist.md 直至通过
```

### 场景 2：reviewer 检视

在 3.1 / 4.2 检视前，加载：

```
1. `references/mr-rules-essential.md` 对照严重/致命规则做检视（必须零违规）
2. `references/mr-rules-general.md` 做深度审查（建议修复）
3. `references/ascendc-r5-r10.md` 检查代码质量指标（圈复杂度/嵌套深度/行数）
```

---

**注意**：本 skill 的 SKILL.md 仅作为索引与触发说明。Agent 在执行自查/检视时，应直接读取相应的 references 文件获取完整规则，**不要**在开发过程中常驻本 skill 的全部内容。

## 代码风格

### 命名规范

| 类别 | 风格 | 示例 |
|------|------|------|
| 目录名（算子名） | snake_case | `spmv`、`spmm`、`sp_geam` |
| 文件名 | snake_case | `{op_name}_host.cpp`、`{op_name}_kernel.h` |
| C++ 类名 | PascalCase | `SpmmCsrMat`、`SpmvKernel` |
| 函数/方法名 | PascalCase / camelCase | `ValidateSpMVParams`、`ComputeMaxRowLength` |
| 常量 | kPascalCase | `kAlignBytes`、`kBufferCount` |
| 宏 | UPPER_SNAKE_CASE | `CHECK_RET`、`GM_ADDR` |
| Tiling 结构体 | PascalCase + TilingData 后缀 | `SpmvTilingData`、`SpmmTilingData` |

### 头文件保护

```cpp
#ifndef SPMV_TILING_DATA_H_
#define SPMV_TILING_DATA_H_
// ...
#endif // SPMV_TILING_DATA_H_
```

### include 顺序

```cpp
// 1. C 标准库
#include <cstdint>
#include <cmath>

// 2. C++ 标准库
#include <memory>
#include <vector>

// 3. ACL/CANN 头文件
#include "acl/acl.h"
#include "log/log.h"

// 4. 算子本地头文件
#include "cann_ops_sparse.h"
#include "{op_name}.h"
#include "{op_name}_tiling_data.h"
#include "{op_name}_kernel.h"
```

## OAT 量化指标

| 指标 | 阈值 | 说明 |
|------|------|------|
| 圈复杂度 | ≤ 20 | 每个函数 |
| 函数深度 | ≤ 5 | 嵌套层级 |
| NBNC (Non-Blank Non-Comment lines) | ≤ 50 | 有效代码行 |
| 除零风险 | 0 | 不允许无保护的除法 |
| extern 引用告警 | 0 | 所有 extern 声明必须在头文件中 |

## OAT 自检 Checklist

开发交付时必须附上各函数的：
- [ ] 圈复杂度
- [ ] 函数深度
- [ ] 有效代码行数
- [ ] 除零校验说明
- [ ] extern 引用检查

## 稀疏算子特有约束

### 描述符操作（按 API 体系）

**通用规则**（两种 API 体系适用）：
- 所有 `new` 操作必须使用 `std::nothrow`
- nullptr 检查必须在所有公开 API 入口处完成

**Generic API**（使用 SpMatDescr/DnVecDescr/DnMatDescr）：
- const 描述符的 `const_cast` 必须在 `ToMatInner`/`ToVecInner`/`ToDnMatInner` 等统一函数中完成，禁止在业务代码中直接 cast
- 描述符的 Create 必须完整初始化所有字段（format、rows、cols、nnz、ptrs、idxs、values、baseType、ptrType、IdxType、valueType）
- Destroy 必须安全处理 nullptr

**Legacy API**（使用扁平参数 + 可选 MatDescr）：
- MatDescr（若接口使用）必须 Create/Destroy 配对
- MatDescr 的 getter（`aclsparseGetMatType` / `aclsparseGetMatIndexBase` 等）在 Host 侧调用，禁止自行存储冗余字段
- 同一算子的多精度版本（S/D 函数）需按精度前缀分别命名，禁止共享实现（避免精度丢失）
- 扁平数据指针校验 nullptr（每个指针独立校验，不能批量）

### 索引运算

- 行偏移数组长度为 `rows + 1`，必须校验一致性
- nnz 必须与值数组/列索引数组的实际长度匹配
- 索引基址（0-based/1-based）处理逻辑必须显式标注

### 数据指针

- Device 指针使用 `GM_ADDR`（`uint8_t *`）传递
- Host 指针使用 `const void *`（alpha/beta 标量，仅 Generic 使用）
- 不允许在 Kernel 侧直接访问 Host 内存

## License Header

所有源文件必须包含标准 License Header：

```cpp
/**
 * ----------------------------------------------------------------------------------------------------------
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ----------------------------------------------------------------------------------------------------------
 */
```
