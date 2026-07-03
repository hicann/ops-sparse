---
name: sparse-op-templates
description: ops-sparse 算子代码模板库，提供不同编程模型下的标准化代码骨架。根据目标芯片架构和编程模型选择对应模板。
---

# Sparse 算子代码模板库

## 使用方法

根据目标芯片架构（arch20/arch22/arch35）和编程模型（SIMD/RegBase/SIMT）选择对应的模板文件：

```
templates/
├── simd/                             # 所有架构通用（默认首选）
│   ├── op_tiling_data.h              # TilingData 结构体模板
│   ├── op_kernel.h                   # kernel.h 签名模板
│   ├── op_kernel.cpp                 # Kernel 实现模板
│   └── op_host.cpp                   # Host 侧实现模板
├── regbase/                          # 仅 arch35（Ascend950）可用
│   ├── op_tiling_data.h              # 含 UB buffer 大小字段
│   ├── op_kernel.h                   # kernel.h 签名模板
│   ├── op_kernel.cpp                 # __VEC_SCOPE__ + RegTensor 模板
│   └── op_host.cpp                   # 含 BufferLayout 预计算模板
└── simt/                             # 仅 arch35（Ascend950）可用
    ├── op_tiling_data.h              # 含 nthreads 字段
    ├── op_kernel.h                   # kernel.h 签名模板
    ├── op_kernel.cpp                 # __simt_vf__ + asc_vf_call 模板
    └── op_host.cpp                   # 含 nthreads 计算模板
```

> `{op_name}.h`（公共头文件）的内容（描述符转换函数）见本文"描述符转换函数"章节，直接内联在算子头文件中，无需独立模板文件。

> RegBase 模板与 SIMD 模板的关键区别：使用 `TBuf + LocalTensor`（非 `TQue`）、`__VEC_SCOPE__` 块内 `RegTensor` 寄存器级 API、host 侧预计算 UB buffer 大小。

> SIMT 模板与 SIMD/RegBase 模板的关键区别：使用 `__simt_vf__` 装饰器、`asc_vf_call` 调度、`grid-stride loop` 线程级并行、无 `TPipe/TQue/DataCopyPad`（直接 `__gm__` 指针访问）。

## 编程模型与架构对应关系

| 架构 | 支持的编程模型 | 说明 |
|------|--------------|------|
| arch22（Ascend910B/910_93） | SIMD | 仅支持向量编程模型 |
| arch20（Ascend310P） | SIMD | 仅支持向量编程模型 |
| arch35（Ascend950） | SIMD / RegBase / SIMT | 三种模型均可，由架构师在 1.3.A 决策 |

## 架构无关性

- SIMD 编程模型在所有架构上可用，是默认选择
- Kernel 侧代码的架构差异由 AscendC 编译器屏蔽
- RegBase 和 SIMT **仅 Ascend950 (arch35) 可用**，有独立模板目录（`templates/regbase/`、`templates/simt/`），开发时直接参考对应模板扩展，不需额外技能

## 模板文件说明

### 1. {op_name}.h - 公共头文件

包含类型映射宏、描述符转换函数、公共常量。

### 2. {op_name}_tiling_data.h - TilingData 结构体

定义 Host/Kernel 共享的 Tiling 参数结构体。

### 3. {op_name}_kernel.h - Kernel 头文件

声明 `kernel_do` 函数签名和 Kernel 类声明。

### 4. {op_name}_host.cpp - Host 侧实现

包含参数校验 + Kernel launch 的完整骨架。

## 公共宏定义（来自现有代码）

以下是仓内已有的公共宏，新算子**必须复用**，**禁止重新定义**：

```cpp
#define GM_ADDR uint8_t *

#define CHECK_RET(cond, return_expr) \
    do { if (!(cond)) { return_expr; } } while (0)
```

## 描述符转换函数（来自 aclsparse_descr_internal.h）

新算子应 `#include "aclsparse_descr_internal.h"` 使用内部结构体，并在本地头文件 `{op_name}.h` 中定义转换函数：

```cpp
inline struct aclsparseContext *ToInternalHandle(aclsparseHandle_t handle) {
    return reinterpret_cast<struct aclsparseContext *>(handle);
}

inline struct aclsparseSpMatDescr *ToMatInner(aclsparseConstSpMatDescr_t desc) {
    return const_cast<struct aclsparseSpMatDescr *>(
        reinterpret_cast<const struct aclsparseSpMatDescr *>(desc));
}

inline struct aclsparseDnVecDescr *ToVecInner(aclsparseConstDnVecDescr_t desc) {
    return const_cast<struct aclsparseDnVecDescr *>(
        reinterpret_cast<const struct aclsparseDnVecDescr *>(desc));
}

inline struct aclsparseDnMatDescr *ToDnMatInner(aclsparseConstDnMatDescr_t desc) {
    return const_cast<struct aclsparseDnMatDescr *>(
        reinterpret_cast<const struct aclsparseDnMatDescr *>(desc));
}
```

## kernel_do 签名模式

```cpp
// {op_name}_kernel.h
#ifndef {OP_NAME}_KERNEL_H_
#define {OP_NAME}_KERNEL_H_

#include "{op_name}_tiling_data.h"

#define GM_ADDR uint8_t *

// kernel_do 签名：
// - 数据指针按算子需求填写（GM_ADDR）
// - tiling 以 const 引用传入
// - numBlocks + stream 为通用参数
void {op_name}_kernel_do(
    GM_ADDR dataPtr1, GM_ADDR dataPtr2, ...,
    uint32_t numBlocks, const {Op}TilingData& tiling, void *stream);

#endif
```

> **Generic API 算子**：若需要 alpha/beta/computeType 参数，在 kernel_do 中按算子需求追加（如 `const void *alpha, const void *beta`），并在 host 侧从描述符或接口参数中提取。

## Host 骨架模式

```cpp
// {op_name}_host.cpp
// 禁止冗余 include：acl/acl.h、cann_ops_sparse_common.h、tiling/platform/platform_ascendc.h 均为冗余
#include <cstdint>
#include <new>
#include "log/log.h"
#include "cann_ops_sparse.h"
#include "{op_name}.h"
#include "{op_name}_kernel.h"
#include "aclsparse_handle_internal.h"
#include "aclsparse_descr_internal.h"
#include "host_utils.h"

namespace {
    static aclsparseStatus_t Validate{Op}Params(
        aclsparseHandle_t handle, ...) {
        // 参数校验 - 每个参数 nullptr/value 检查
    }

    static aclsparseStatus_t Launch{Op}Kernel(
        aclsparseHandle_t handle, ...) {
        // 描述符解析 -> TilingData 计算 -> kernel_do launch
    }
} // namespace

extern "C" {
    aclsparseStatus_t aclsparse{Op}(
        aclsparseHandle_t handle, ...) {
        aclsparseStatus_t status = Validate{Op}Params(handle, ...);
        if (status != ACL_SPARSE_STATUS_SUCCESS) return status;
        return Launch{Op}Kernel(handle, ...);
    }
}
```
