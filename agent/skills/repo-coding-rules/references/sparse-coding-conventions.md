# 稀疏算子编码约定

> 本文件收录 ops-sparse 仓特有的编码约束，补充 AscendC 通用编码规则（R1-R10）和 MR 安全规则之外的稀疏领域编码约定。

---

## 1. 异步 Kernel 启动与 Tiling 传递

Host 侧 kernel 是**异步**的，调用 `kernel_do(...)` 后立即返回：

- **禁止**在 host 侧调用 `aclrtSynchronizeStream`（同步由调用方负责）
- Tiling 数据传递：host 侧以 **`const` 引用** 传入 `kernel_do`；kernel 侧以 **by value** 接收（运行时 launch 参数自动拷贝）
- **禁止**使用 `aclrtMalloc` + `aclrtMemcpy(H2D)` 传递 tiling（性能差且不必要）

```cpp
// ✅ 正确：const 引用传 tiling
{{op}}_kernel_do(dataPtr1, dataPtr2, numBlocks, tiling, useStream);

// ❌ 错误：禁止 H2D 拷贝 tiling
aclrtMalloc(&tilingGm, sizeof(tiling), ...);
aclrtMemcpy(tilingGm, ..., &tiling, ..., ACL_MEMCPY_HOST_TO_DEVICE);
{{op}}_kernel_do(dataPtr1, ..., numBlocks, tilingGm, useStream);
```

---

## 2. 独立 kernel.h 头文件

每个算子必须有独立的 `{op_name}_kernel.h` 文件，声明 `kernel_do` 函数签名：

- host.cpp 和 kernel.cpp 都 `#include "{op}_kernel.h"`
- **禁止**在 host.cpp 中以 `extern` 前向声明方式声明 `kernel_do`

```cpp
// ✅ 正确：通过 kernel.h 引入签名
#include "{op}_kernel.h"

// ❌ 错误：禁止 extern 前向声明
extern void {{op}}_kernel_do(GM_ADDR data, uint32_t numBlocks, ...);
```

---

## 3. kernel.h + kernel 签名规范

- `kernel.h` 中 `kernel_do` 数据指针参数统一使用 `GM_ADDR`，与 kernel.cpp 签名一致，**禁止**使用 `uint8_t*`
- kernel.cpp 中所有 `__global__` kernel 入口函数必须带 `extern "C"` 修饰，禁止 C++ name mangling（代码检视为 HIGH）

```cpp
// {op}_kernel.h
#define GM_ADDR uint8_t *
void {{op}}_kernel_do(GM_ADDR data1, GM_ADDR data2, ...,
                      uint32_t numBlocks, const {{Op}}TilingData& tiling, void* stream);

// {op}_kernel.cpp
extern "C" __global__ __aicore__ void {{op}}_kernel(GM_ADDR data1, ..., const {{Op}}TilingData tiling)
{
    // ...
}
```

---

## 4. Host include 精简

host.cpp **禁止**引入冗余 include，仅保留必需头文件：

| 头文件 | 是否必需 | 说明 |
|--------|---------|------|
| `<cstdint>` | 是 | 标准整数类型 |
| `<new>` | 是 | `std::nothrow` |
| `log/log.h` | 是 | dlog 日志 |
| `cann_ops_sparse.h` | 是 | 算子公共头文件 |
| `{op}_kernel.h` | 是 | kernel_do 签名 |
| `aclsparse_handle_internal.h` | 是 | Handle 内部结构体 |
| `aclsparse_descr_internal.h` | 是 | 描述符内部结构体 |
| `host_utils.h` | 是 | `GetAivCoreCount` 等工具函数 |
| ~~`acl/acl.h`~~ | **否** | 由公共头文件间接引入 |
| ~~`cann_ops_sparse_common.h`~~ | **否** | 由公共头文件间接引入 |
| ~~`tiling/platform/platform_ascendc.h`~~ | **否** | 由公共头文件间接引入 |

---

## 5. 代码风格

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

---

## 6. 描述符编码约束（补充）

> 领域层面的描述符规范见 `repo-knowledge` skill 的 `references/sparse-domain-rules.md`。此处仅补充编码层面的强制约束。

### 通用规则（两种 API 体系适用）

- 所有 `new` 操作必须使用 `std::nothrow`
- nullptr 检查必须在所有公开 API 入口处完成

### Generic API

- const 描述符的 `const_cast` 必须在 `ToMatInner`/`ToVecInner`/`ToDnMatInner` 等统一函数中完成，**禁止**在业务代码中直接 cast
- 描述符的 Create 必须完整初始化所有字段（format、rows、cols、nnz、ptrs、idxs、values、baseType、ptrType、IdxType、valueType）
- Destroy 必须安全处理 nullptr（直接返回 SUCCESS）

### Legacy API

- MatDescr（若接口使用）必须 Create/Destroy 配对
- MatDescr 的 getter（`aclsparseGetMatType` / `aclsparseGetMatIndexBase` 等）在 Host 侧调用，**禁止**自行存储冗余字段
- 同一算子的多精度版本（S/D 函数）需按精度前缀分别命名，**禁止**共享实现（避免精度丢失）
- 扁平数据指针校验 nullptr（每个指针独立校验，不能批量）

---

## 7. OAT 量化指标

| 指标 | 阈值 | 说明 |
|------|------|------|
| 圈复杂度 | ≤ 20 | 每个函数 |
| 函数深度 | ≤ 5 | 嵌套层级 |
| NBNC (Non-Blank Non-Comment lines) | ≤ 50 | 有效代码行 |
| 除零风险 | 0 | 不允许无保护的除法 |
| extern 引用告警 | 0 | 所有 extern 声明必须在头文件中 |

## 8. OAT 自检 Checklist

开发交付时必须附上各函数的：

- [ ] 圈复杂度
- [ ] 函数深度
- [ ] 有效代码行数
- [ ] 除零校验说明
- [ ] extern 引用检查

---

## 9. License Header

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
