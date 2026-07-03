# AclSparse Agent

多 Agent 协作框架，将 Sparse 算子开发流程编排为可追溯、可恢复的工程流水线。用户描述算子需求，Agent 团队自动完成从需求分析到代码上库的全流程。

## 设计思想

**结构化问卷，无遗漏需求** — 逐项问透，不留盲区。

**执行不验收，验收不执行** — 质量靠制衡，不靠自觉。

**流程可追溯，文档全记录** — 每步有日志，每阶段有产出。

## 参与角色

| 角色 | 职责 |
|------|------|
| 用户 | 需求提出、各确认点审批 |
| writer | 资料准备、文档与问卷整理、文档编写 |
| architect | 需求分析、方案设计、方案评审 |
| developer | 代码开发、编译联调、性能调优 |
| tester | 测试设计、用例开发、测试验收 |
| reviewer | 代码检视：规范、一致性、风险 |

## 开发流程

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

## 外部参考仓库

Agent 在架构设计、代码开发和性能优化阶段可按需加载以下外部仓库作为参考：

| 仓库 | 本地路径 | 技能 | 用途 |
|------|---------|------|------|
| [cannbot-skills](https://gitcode.com/cann/cannbot-skills.git) | `.opencode/ref-repos/cannbot-skills/` | 通过 `cannbot_references.json` 映射 | Ascend C 通用技能（API 最佳实践、代码检视、精度调试、性能优化等 16 个 skill） |
| [cann-samples](https://gitcode.com/cann/cann-samples.git) | `.agent/cann-samples/` | `op-samples-reference` | 高性能算子样例、端到端调优实践、SIMT 编程模型参考 |
| [asc-devkit](https://gitcode.com/cann/asc-devkit.git) | `.agent/asc-devkit/` | `asc-devkit-reference` | Ascend C 官方 API 文档（1022+）、示例代码（587+）、实现参考、Tiling 配置 |

初始化时通过 `init.sh` 自动克隆，也可通过 `--cannbot`、`--samples`、`--asc` 参数指定本地路径创建软链接。

## 对标说明

本仓库对标 NVIDIA cuSPARSE，**按目标接口的归属选择 API 体系**：

### Generic API 体系（cuSPARSE 主推方向）
- 使用 Descriptor（描述符）模式管理稀疏矩阵和稠密向量/矩阵
- 操作包含三阶段调用：GetBufferSize → Preprocess → Execute
- 适用算子：SpMV/SpMM/SpGEMM/SDDMM/SpSM/SpSV 等

### Legacy API 体系（cuSPARSE Legacy，尚未迁移到 Generic）
- 直传扁平参数 + 轻量 MatDescr 矩阵描述符
- 带精度前缀（S/D/C/Z）的独立函数命名
- 适用算子：gtsv2（三对角求解）、格式转换（coo2csr/csr2coo）、排序（csrsort/coosort）等

**工作流中的决策点**：CP1.1.A 问卷中明确 API 体系，之后 architect / developer / tester / reviewer 均按对应规范执行。
