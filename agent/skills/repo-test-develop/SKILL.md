---
name: repo-test-develop
description: |
  仓库测试开发指导，介绍 ops-sparse 测试框架的使用与测试代码的开发方法。
  为 Sparse 算子开发 GTest + CSV 驱动的精度 ST：param / golden / npu_wrapper / test.cpp / test.csv / CMake。
  触发：实现 golden、编写分级功能用例、补全白盒测试、执行精度测试时加载。
  代码模板位于 references/ 目录，由调用方按需复制。
---

# Sparse 算子 ST 开发技能

ops-sparse 精度 ST 采用 **GTest 参数化 + CSV 用例表** 驱动，使用 **Eigen SparseMatrix<double>** 作为唯一 CPU golden 参考。开发新算子测试时，以 `references/test/{op}/` 下的模板为起点，复制到算子目录后按 API 填充。

## 测试标杆体系（单层 Eigen Golden）

```
                      ┌─────────────────────────────┐
                      │     NPU 算子执行             │
                      └──────────┬──────────────────┘
                                 │ NPU 结果
                                 ▼
                      ┌─────────────────────────────┐
                      │       精度验证               │
                      └──────────┬──────────────────┘
                                 │
                      ┌──────────▼──────────────────┐
                      │   Eigen Golden (FP64)        │
                      │   （唯一 CPU 参考基准）       │
                      └─────────────────────────────┘
```

| 实现 | 用途 | 依赖 |
|------|------|------|
| Eigen SparseMatrix<double> | 唯一 CPU golden 参考，与 cuSPARSE 结果对齐 | Eigen3 (header-only) |

**Eigen 作为测试标杆的原因**：C++ 原生（与测试代码同语言）、header-only（无复杂构建依赖）、SparseMatrix (CSR) 直接对应 cuSPARSE 的 CSR 格式、社区公认、FP64 精度避免精度损失。

## 测试交付件目录结构

一个算子的测试交付件位于 `test/{op_name}/`，共 6 个文件（3 个芯片无关 + 3 个在 `archXX/`）：

```
test/{op_name}/
├── {op_name}_param.h          -- 参数结构体（与芯片无关），继承 SparseTestParamBase
├── {op_name}_golden.h         -- CPU golden（与芯片无关），使用 Eigen SparseMatrix<double> FP64
├── CMakeLists.txt             -- 测试注册
└── archXX/
    ├── {op_name}_npu_wrapper.h  -- NPU wrapper（芯片相关 ACL 操作），使用 RAII Manager
    ├── {op_name}_test.cpp       -- GTest 入口（TEST_F 异常路径 + TEST_P CSV 驱动）
    └── {op_name}_test.csv       -- CSV 用例表，列名 = API 参数名
```

## 代码模板

模板目录 `references/test/{op}/` 按上表布局组织，文件名以 `op_` 为占位前缀，内容用 `{{op}}` / `{{Op}}` 占位符与 `// TEMPLATE:` 注释标注需替换处：

| 交付文件 | 模板 |
|---------|------|
| `{op}_param.h` | [references/test/{op}/op_param.h](references/test/{op}/op_param.h) |
| `{op}_golden.h` | [references/test/{op}/op_golden.h](references/test/{op}/op_golden.h)（含 SpMV / SpMM 两变体） |
| `CMakeLists.txt` | [references/test/{op}/CMakeLists.txt](references/test/{op}/CMakeLists.txt) |
| `archXX/{op}_npu_wrapper.h` | [references/test/{op}/archXX/op_npu_wrapper.h](references/test/{op}/archXX/op_npu_wrapper.h)（含 Generic RAII + Legacy 两变体） |
| `archXX/{op}_test.cpp` | [references/test/{op}/archXX/op_test.cpp](references/test/{op}/archXX/op_test.cpp)（含 TEST_P CSV 驱动 + TEST_F 异常路径） |
| `archXX/{op}_test.csv` | [references/test/{op}/archXX/op_test.csv](references/test/{op}/archXX/op_test.csv) |

## ST 框架头文件（`test/frame/`）

| 头文件 | 职责 |
|--------|------|
| `types.h` | `PrecisionMode` 枚举（ABS/REL/MERE_MARE/EXACT/INTEGER）+ `VerifyConfig` 精度配置 + `CaseSummary` 统计 |
| `verify.h` | `Verifier` 精度比对类（策略模式：AbsStrategy / RelStrategy / MereMareStrategy / ExactStrategy / IntegerStrategy） |
| `fill.h` | `SparseFillGenerator` + `CsrMatrix/CooMatrix/CscMatrix` 结构体 + `makeSparseCsr/Coo/Csc/makeDense/makeDenseFloat/makeDiagCsr/makeEmptyCsr` 快速填充函数 |
| `descriptor_manager.h` | Generic 描述符 RAII 封装：`SpMatManager/DnVecManager/DnMatManager/HandleManager/DeviceBuffer`，自动 Create/Destroy |
| `sparse_test.h` | `AclEnvScope` RAII 环境初始化 + `SparseTestParamBase` 参数结构体基类 + `aclDataTypeOf<T>()` 模板 + `SPARSE_CHECK_RET/SPARSE_LOG` 宏 |
| `csv_loader.h` | `csv_map` / `ReadMap` / `GetCasesFromCsv<ParamType>` / `parseBool/parseInt/parseFloat` 等 CSV 解析器 |

### 强制规则

- 所有新算子测试**必须**使用 `test/frame/` 公共头文件，禁止在算子测试代码中重新定义相同功能
- 精度配置必须使用 `VerifyConfig` + `Verifier`，禁止手写 `Verify()` 函数
- 描述符生命周期必须使用 RAII Manager（Handle/SpMat/DnVec/DnMat），禁止裸指针手动 destroy
- ACL 环境初始化必须使用 `AclEnvScope`，禁止手写 Init/Finalize
- 若新增通用功能，必须先补充到 `test/frame/` 对应头文件，再在算子测试中调用

## 开发流程

1. **分析 API**：读 `include/cann_ops_sparse.h` 与算子实现，确认签名与 API 体系（Generic/Legacy）。NPU wrapper 封装全部 ACL 操作，测试侧只准备 host `std::vector`。
2. **写 Param**（`{op}_param.h`）：继承 `SparseTestParamBase`，实现 `fillCustom()` + `caseId()`，字段按 API 参数顺序。
3. **写 Golden**（`{op}_golden.h`）：使用 Eigen SparseMatrix<double> FP64 实现，签名与算子 API 一致，保留参数校验。
4. **写 NPU Wrapper**（`{op}_npu_wrapper.h`）：使用 RAII Manager（Handle/SpMat/DnVec/DnMat/DeviceBuffer），禁止裸指针；每个 ACL 调用校验返回值。
5. **写 CSV + GTest**（`{op}_test.csv` / `{op}_test.cpp`）：CSV 分级用例（L0/L1），GTest `TEST_F` 异常路径 + `TEST_P` CSV 驱动 5 步流程。
6. **CMake + 构建验证**：`ops_sparse_add_gtest_tests` 注册，`test/frame/test_main.cpp` 提供统一 main。

   ```bash
   source <CANN>/set_env.sh
   cd ops-sparse
   bash build.sh --ops={op} --run              # 默认卡0
   bash build.sh --ops={op} --run --device=1   # 指定卡1
   ```
   通过标准：`[  PASSED  ] N tests.`，Summary 中 `Failed: 0`。

## 精度模式选择

`VerifyConfig.mode` 必须**显式设置**，不得依赖默认值。

| 算子类型 | 推荐模式 | 配置方式 |
|----------|----------|----------|
| 格式转换 / pack-unpack | EXACT | `TEST_P` 内设 `cfg.mode = PrecisionMode::EXACT` |
| 浮点运算（SpMV/SpMM 等） | MERE_MARE | param 加 `mereThreshold`/`mareMultiplier` 字段，CSV 加 `mere_threshold`/`mare_multiplier` 列 |
| 整数运算（排序/格式转换索引） | INTEGER | `cfg.mode = PrecisionMode::INTEGER` |

### Eigen 精度容差

由于 Eigen 使用 FP64 计算，NPU 使用 T 类型，不同 dtype 的容差不同：

| NPU dtype | Eigen golden rtol | Eigen golden atol | 说明 |
|-----------|-------------------|-------------------|------|
| FP32 | 1e-5 | 1e-5 | FP32 标准 |
| FP16 | 1e-3 | 1e-3 | FP16 标准 |
| BF16 | 5e-3 | 5e-3 | BF16 标准 |
| INT32 | 0 (位精确) | 0 | 整数计算 |

## 测试用例设计指南

### L0 用例（基础功能验证）

| Shape | nnz | dtype | 格式 | 说明 |
|-------|-----|-------|------|------|
| 4×4 | 8 | FP32 | CSR | 基本方阵 |
| 8×16 | 24 | FP32 | CSR | 宽矩阵 |
| 16×8 | 24 | FP32 | CSR | 高矩阵 |

### L1 用例（覆盖 + 边界 + 大 shape）

| Shape | nnz | dtype | 格式 | 说明 |
|-------|-----|-------|------|------|
| 2048×2048 | 32768 | FP32 | CSR | 大 shape |
| 1×100 | 10 | FP32 | CSR | 单行 |
| 100×1 | 10 | FP32 | CSR | 单列 |
| 10×10 | 0 | FP32 | CSR | 空矩阵 |
| 64×64 | 128 | FP16 | CSR | 半精度 |
| 64×64 | 128 | FP32 | CSR(T) | 转置 |

## 老算子（仅 spmv / spmm，保持现状）

老算子保留自定义 main + TestRegistry 统计模式，使用 `ops_sparse_add_test` 注册（非 `ops_sparse_add_gtest_tests`）。新算子必须使用 GTest + CSV 模式。

## 常见问题

| 现象 | 处理 |
|------|------|
| Eigen header 找不到 | 检查 CMake 是否启用 `TEST_USE_EIGEN=ON`；确认 Eigen3 已安装 |
| CSV 读取失败 | 确认 CSV 与 .cpp 同名同目录，`ops_sparse_add_gtest_tests` 宏自动拷贝到 build 目录 |
| null handle 测试多余代码 | 改用 `TEST_F` 单独测，不下 CSV |
| `gtest_main` 链接冲突 | 框架统一用 `test/frame/test_main.cpp`，勿自写 `main()` |
| 描述符泄漏 | 检查所有 `aclsparseCreate*` 是否有配对的 `aclsparseDestroy*`；推荐使用 RAII Manager |
| RAII 双重释放 | 检查是否同时使用了 RAII Manager 和手动 `aclsparseDestroy*`；二者只能选其一 |
| 精度 fail | 看 Verifier 日志中的 MERE/MARE 或 exact mismatch 计数；对比 NPU 输出与 golden 的最大误差位置 |
