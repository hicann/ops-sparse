---
name: sparse-new-op-workflow
description: ops-sparse 算子全流程开发技能，协调 agent 团队完成设计->开发->验收->上库的完整开发链路。同时支持 cuSPARSE Generic API 和 Legacy API 两种体系。触发：用户要求开发新算子或实现某 cuSPARSE 接口时。
---

# 硬规则

以下规则在任何阶段都必须遵守，不可跳过、不可变通。

1. **角色分工** — 主 Agent 只负责调度，禁止亲自动手
   - 每个步骤调用对应 Subagent：编码→`developer`，测试→`tester`，设计→`architect`，检视→`reviewer`，文档→`writer`
   - **禁止**主 Agent 执行：读模板文件（assets/*.md）、读代码、搜索仓内文件、抓取/搜索网页资料、写代码、写文档
   - 主 Agent 唯一例外：读取 references/ 下的流程参考文件、更新 LOG.md、发送 AskUserQuestion
   - 调用 Subagent 时只定义三要素（输入、输出、验收标准），禁止干涉实现细节
   - 若检视/验收报告发现问题，必须通过 developer Subagent 修复，主 Agent 不得直接修改代码

2. **流程控制** — 阶段不可跳过，门控不可绕过
   - 设计→开发→验收→上库，每阶段需用户确认后推进
   - Subagent 报告未通过验证时，禁止进入下一阶段
   - CP1.2 之后的 CP1.4 / CP2.1 / CP2.2：评审/验收通过 → 直接进入下一步；不通过 → 直接打回修订，同一轮超过 3 次仍不通过再生成 CP 问卷。CP3.2 和 CP4.3 除外，必须询问用户

   **CP 否定路由表**：

   | CP | 「需要修改」打回 | 「终止流程」 |
   |----|----------------|------------|
   | CP1.1.A | 重新发送问卷 | — |
   | CP1.1.B | 重新发送问卷 | — |
   | CP1.2 | 打回 1.2 需求分析 | — |
   | CP1.4 | 打回 1.3.A/1.3.B 重新设计 | 终止开发流程 |
   | CP2.1 | 打回 2.1.1.A/2.1.1.B 重新开发 | 终止开发流程 |
   | CP2.2 | 打回 2.2.1.A/2.2.1.B 重新开发 | 终止开发流程 |
   | CP3.2 | 打回 3.1/3.2 重新检视/验收 | — |
   | 4.1.1 / 4.1.2 | 打回 4.1 修复 README | — |
   | CP4.3 | 打回 4.2 代码检视 | — |

3. **问卷处理** — 基于已生成的 json 模板发送 `AskUserQuestion` 问卷时，不得进行任何修改，直接发送。每次问卷得到用户答复后，保存为 `{原问卷名}.ret.json`。

4. **日志与 Git 安全**
   - 每次流程步骤推进后，主 Agent 必须立即更新 LOG.md
   - `.agent/` 目录下的文件已被 `.gitignore` 屏蔽，禁止执行 `git add .agent/`

5. **检视前 diff 预检** — 调用 reviewer（3.1 / 4.2）前，主 Agent 必须先执行：
   - **diff 预检**：执行 `git diff --stat cann/master...HEAD`，若出现非本算子文件（允许白名单：`src/{op_name}/`、`test/{op_name}/`、`include/cann_ops_sparse.h`、`include/cann_ops_sparse_common.h`），必须先还原
   - **OAT 合规扫描**：执行 `sh scripts/oat_check.sh <变更文件列表>`

6. **异步 Kernel 启动与 const 引用 Tiling**
   - Host 侧 kernel 是**异步**的：调用 `kernel_do(...)` 后立即返回，**禁止**在 host 侧调用 `aclrtSynchronizeStream`
   - Tiling 数据传递：host 侧以 **`const` 引用** 传入 `kernel_do`；kernel 侧以 **by value** 接收
   - **禁止**使用 `aclrtMalloc` + `aclrtMemcpy(H2D)` 传递 tiling

7. **描述符/MatDescr 生命周期**（按 CP1.1.A 选择的 API 体系）
    - **Generic 体系**：所有稀疏算子的 Host 侧必须正确管理描述符（Descriptor）的创建和使用；Create/Destroy API 必须配对调用；Host 侧从描述符提取信息时，必须校验格式（format）、维度（rows/cols/nnz）和数据类型
    - **Legacy 体系**：无 SpMatDescr/DnVecDescr/DnMatDescr；若接口需要矩阵格式属性，使用轻量 MatDescr（Create/Destroy 配对）；数据指针扁平传入 Host 侧

8. **2 层目录结构**
   - 算子代码位于 `src/{op_name}/archXX/`
   - 测试代码位于 `test/{op_name}/`
   - `{op_name}` 使用 **snake_case** 格式（如 `spmv`、`spmm`、`sp_geam`）

9. **独立 kernel.h 头文件**
   - 每个算子必须有独立的 `{op_name}_kernel.h` 文件，声明 `kernel_do` 函数签名
   - host.cpp 和 kernel.cpp 都 `#include "{op}_kernel.h"`
   - **禁止**在 host.cpp 中以 `extern` 前向声明方式声明 `kernel_do`

10. **Host 函数结构与强制 dlog 集成**
    - host.cpp 必须拆分为两个静态函数：`Validate{Op}Params(...)`（参数校验）+ `Launch{Op}Kernel(...)`（tiling 计算 + launch）
    - **强制集成 dlog 日志**：`#include "log/log.h"`，使用 `OP_LOGE`/`OP_LOGD`/`OP_LOGI`
    - **禁止**使用 `printf` 或 `std::cout`

11. **cuSPARSE 接口对齐**
    - 新增 API 必须严格对齐 cuSPARSE 对应接口的规范（按 CP1.1.A 选择的 API 体系）
    - **Generic 体系**：接口签名参照 Generic API、使用 `aclsparseSpMatDescr_t` / `aclsparseDnVecDescr_t` / `aclsparseDnMatDescr_t`、三阶段执行（若需要）
    - **Legacy 体系**：接口签名参照 Sparse BLAS 风格、带精度前缀（S/D/C/Z）、使用 `aclsparseMatDescr_t`、扁平参数
    - 两者共用：状态码使用 `aclsparseStatus_t` 枚举值、操作类型使用 `aclsparseOperation_t`、索引类型使用 `aclsparseIndexType_t` / `aclsparseIndexBase_t`

12. **公共代码复用**
    - 优先复用 `src/common/` 下的公共模块（handle_internal、descr_internal、auxiliary）
    - 禁止在算子代码中重复定义公共宏或工具函数

13. **kernel.h + kernel 签名规范**
    - `kernel.h` 中 `kernel_do` 数据指针参数统一使用 `GM_ADDR`，与 kernel.cpp 签名一致，禁止使用 `uint8_t*`
    - kernel.cpp 中所有 `__global__` kernel 入口函数必须带 `extern "C"` 修饰，禁止 C++ name mangling（reviewer 检视为 HIGH）

14. **host include 精简**
    - host.cpp **禁止**引入冗余 include：`acl/acl.h`、`cann_ops_sparse_common.h`、`tiling/platform/platform_ascendc.h` 均为冗余，由公共头文件间接引入
    - 仅保留必需头文件：`log/log.h`、`cann_ops_sparse.h`、`{op}_kernel.h`、`aclsparse_handle_internal.h`、`aclsparse_descr_internal.h`；视算子需求可选其他

15. **README 质量门控**
    - README 必须先通过内容审查（4.1.1）和编译测试（4.1.2）才能进入代码检视（4.2）
    - 4.1.1 或 4.1.2 失败时，打回 4.1 writer 修复文档，最多 2 次；第 2 次仍失败通过 AskUserQuestion 询问用户
    - 4.1.2 编译测试中 NPU 不可用时，标记为「跳过运行时（环境限制）」，编译通过即视为成功

---

# 启动指令

**加载本技能后，主 Agent 必须立即执行以下动作：**

1. 确定当前步骤（首次加载从 1.1 开始，中断恢复从 LOG.md 记录的步骤继续）
2. 查找下方流程表中该步骤对应的"参与角色"
3. 调用该 Subagent，传入 task-prompts.md 中定义的输入/输出/验收标准
4. Subagent 返回后，主 Agent 更新 LOG.md，然后按流程表推进到下一步

---

# 流程全景

## 角色

| 角色 | 职责 |
|------|------|
| 用户 | 需求提出、各确认点审批 |
| writer | 资料准备、文档与问卷整理、文档编写 |
| architect | 需求分析、方案设计、方案评审 |
| developer | 代码开发、编译联调、性能调优 |
| tester | 测试设计、用例开发、测试验收 |
| reviewer | 代码检视：规范、一致性、风险 |

## 流程

| 步骤 | 输入 | 参与角色 | 输出 | 说明 | 并行 |
|------|------|--------|------|------|------|
| **阶段1：设计** | | | | | |
| 1.1.A 资料准备 | 用户需求 | writer | 工作区目录、LOG.md、1.1-参考资料清单.md | 收集 cuSPARSE 对标接口文档 + 用户提供的资料 | |
| 1.1.S 总结 | 1.1-参考资料清单.md | writer | CP1.1.A.json | 整理为基础信息问卷 | |
| ⛔ CP1.1.A | CP1.1.A.json | 用户 | 算子名/dtype/目标芯片对齐 | AskUserQuestion | |
| 1.1.B 环境准备 | CP1.1.A确认的算子名 | developer | 2.0.1-开发环境.md、git 分支 | 环境检查、创建分支 | |
| 1.1.S2 总结 | CP1.1.A结论 + 1.1-参考资料清单.md | writer | CP1.1.B.json | 裁剪问卷 | |
| ⛔ CP1.1.B | CP1.1.B.json | 用户 | 精度标准/编程模型对齐 | AskUserQuestion | |
| 1.2 需求分析 | CP1.1结论 + 1.1-参考资料清单.md | architect | 1.2-需求分析.md、CP1.2.json | 参数约束、可行性评估 | |
| ⛔ CP1.2 | CP1.2.json | 用户 | 需求分析审批 | AskUserQuestion | |
| 1.3.A 开发方案设计 | 1.2-需求分析.md | architect | 1.3.A-开发方案设计.md | Tiling / Kernel / Host 设计 | ┐ |
| 1.4.A 开发方案评审 | 1.3.A-开发方案设计.md | architect | 1.4.A-开发方案评审.md | 不通过 → 打回 1.3.A | │ |
| 1.3.B 测试方案设计 | 1.2-需求分析.md | tester | 1.3.B-测试方案设计.md | 用例表 + 验收标准 | │ |
| 1.4.B 测试方案评审 | 1.3.B-测试方案设计.md | tester | 1.4.B-测试方案评审.md | 不通过 → 打回 1.3.B | ┘ |
| ⚪ CP1.4 | 1.4.A + 1.4.B | 主Agent | 裁定 | 双方通过→阶段2 | |
| **阶段2：开发** | | | | | |
| **2.1 迭代一** | | | | 核心路径 | |
| 2.1.1.A 算子开发 | 1.3.A-开发方案设计.md | developer | 算子代码 | 核心逻辑 + dlog + 描述符解析 | ┐ |
| 2.1.1.B 测试开发 | 1.3.B-测试方案设计.md | tester | L0 用例代码 | 测试用例 + golden 比对 | ┘ |
| 2.1.2 汇合联调 | 算子代码 + 用例代码 | developer | 2.1.2-汇合联调报告.md | 编译 + L0 精度 | |
| 2.1.3 测试验收 | 2.1.2-汇合联调报告.md | tester | 2.1.3-测试验收报告.md | L0 通过率 100% | |
| ⚪ CP2.1 | 2.1.3-测试验收报告.md | 主Agent | 裁定 | 通过→迭代二 | |
| **2.2 迭代二** | | | | 全覆盖 + 边界 | |
| 2.2.1.A 算子开发 | 1.3.A-开发方案设计.md | developer | 完整算子代码 | 补齐分支 + 异常拦截 | ┐ |
| 2.2.1.B 测试开发 | 1.3.B-测试方案设计.md | tester | L0+L1 用例 | 新增用例 + 边界 | ┘ |
| 2.2.2 汇合联调 | 算子代码 + 用例代码 | developer | 2.2.2-汇合联调报告.md | 全量精度 | |
| 2.2.3 测试验收 | 2.2.2-汇合联调报告.md | tester | 2.2.3-测试验收报告.md | 全量通过率 100% | |
| ⚪ CP2.2 | 2.2.3-测试验收报告.md | 主Agent | 裁定 | 通过→阶段3 | |
| **阶段3：验收** | | | | | |
| 3.1 代码检视 | git diff + OAT + 全部变更文件 | reviewer | 3.1-代码检视报告.md | 规范、一致性、风险、日志 | |
| 3.2 性能验收 | 1.2 + 1.3.A 文档 | developer | 3.2-性能报告.md | 性能采集、瓶颈分析 | |
| ⛔ CP3.2 | 3.1 + 3.2 | 用户 | 验收审批 | AskUserQuestion | |
| 3.3 大 shape 精简 | CP3.2 问卷结果 | developer | 精简后的 CSV + ST 通过 | 仅当用户选择「精简为 1 条」时执行 | |
| **阶段4：上库** | | | | | |
| 4.1 编写文档 | 全部代码和设计文档 | writer | README.md | — | |
| 4.1.1 README 内容审查 | README.md + API 声明（cann_ops_sparse.h）+ host.cpp | reviewer (scene: readme-review) | 4.1.1-审查报告.md | 9 项逐项审查，不通过→打回 4.1（≤2 次） | |
| 4.1.2 README 编译测试 | README.md + 2.0.1-开发环境.md | developer (scene: readme-compile-test) | 4.1.2-编译测试报告.md | 提取调用示例→CMake 编译→NPU 可用时运行，不通过→打回 4.1（≤2 次） | |
| 4.2 代码检视 | git diff + OAT + 全部文件 + 文档 | reviewer | 4.2-代码检视报告.md | 最终检视 | |
| 4.3 开发总结 | 全部交付物 | writer | CP4.3.json + Issue + PR 模板 + LOG | 总结 | |
| ⛔ CP4.3 | CP4.3.json | 用户 | 上库审批 | AskUserQuestion，通过后 squash commit | |

**图例**：⛔ 必需确认  ⚪ 仅不通过时直接打回，3次仍失败后询问

---

# 参考资源

| 资源 | 路径 | 说明 |
|-----|------|------|
| Task 调用参数 | [references/task-prompts.md](references/task-prompts.md) | 各阶段 Subagent 的调用参数与验收标准 |
| 数据流说明 | [references/data-flow.md](references/data-flow.md) | 各阶段输入输出文件和数据流向 |
| 错误处理指南 | [references/error-handling.md](references/error-handling.md) | 常见错误类型、重试阈值与回退策略 |
| Issue 模板 | [assets/ISSUE_TEMPLATE.md](assets/ISSUE_TEMPLATE.md) | 上库 Issue 模板 |
| 文档与问卷模板 | [assets/](assets/) | 所有产出物的模板文件 |
| README 标准模板 | [assets/README.md](assets/README.md) | 算子 README 统一模板，4.1 编写文档时使用 |
| ST 测试开发指南 | [agent/skills/sparse-ST-develop/SKILL.md](../../../agent/skills/sparse-ST-develop/SKILL.md) | ST 框架设计与编码规范，2.1.1.B / 2.2.1.B 测试开发时参考 |
| 算子代码模板库 | [agent/skills/sparse-op-templates/SKILL.md](../../../agent/skills/sparse-op-templates/SKILL.md) | 标准化代码骨架（SIMD/RegBase/SIMT），2.1.1.A / 2.2.1.A 算子开发时使用 |
