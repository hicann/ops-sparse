---
name: workflow-doc-templates
description: 交付件模板，提供设计文档、验收报告等中间交付件的格式模板。触发：产出需求/Spec/方案/验收报告/算子文档/开发日志/Issue 等交付件时，先加载对应模板作为格式基准。
---

# 交付件模板索引（ops-sparse 专用）

本仓交付件的格式模板集中在 `references/` 下，文件名按工作流统一流程表编号 + 中文标题命名。产出对应交付件时，先读取模板作为格式基准，按占位符 `{...}` 填充实际内容。

> 本文件只做路由。模板正文在 `references/` 中按需加载，不在此内联。

## 模板索引

| 编号 | 交付件 | 模板 | 说明 |
|------|--------|------|------|
| 0 | 开发环境信息 | [references/0-环境信息.md](references/0-环境信息.md) | 环境检查、NPU 设备、CANN 版本、SOC 确认 |
| 1.1 | 需求分析 | [references/1.1-需求分析.md](references/1.1-需求分析.md) | 数学定义、算子规格、cuSPARSE 对标、API 体系、稀疏格式、参数约束 |
| 1.2 | Spec | [references/1.2-Spec.md](references/1.2-Spec.md) | spec.yaml 生成指引（机器可校验的 L0 数学契约） |
| 2.1 | 测试方案设计 | [references/2.1-测试方案设计.md](references/2.1-测试方案设计.md) | Eigen golden 方案、稀疏格式用例、L0/L1/L2 分级用例 |
| 2.2 | 开发方案设计 | [references/2.2-开发方案设计.md](references/2.2-开发方案设计.md) | 代码架构、Tiling 策略、描述符解析、Kernel 设计、Host 拆分 |
| CP3 | 功能验收报告 | [references/CP3-功能验收报告.md](references/CP3-功能验收报告.md) | 编译验证、精度统计、稀疏专项检查、测试代码完整性 |
| CP4 | 性能验收报告 | [references/CP4-性能验收报告.md](references/CP4-性能验收报告.md) | 性能数据、瓶颈分析、稀疏度/nnz 维度 |
| CP5 | 代码检视报告 | [references/CP5-代码检视报告.md](references/CP5-代码检视报告.md) | OAT 合规、稀疏专项（描述符/日志）、冗余代码、文档检视 |
| 6.1 | 算子文档 | [references/6.1-算子文档.md](references/6.1-算子文档.md) | README 模板（Generic/Legacy API 调用示例、稀疏格式、产品支持） |
| 7.1 | 开发报告 | [references/7.1-开发报告.md](references/7.1-开发报告.md) | 交付物清单、开发过程、关键指标 |
| 7.2 | 经验总结 | [references/7.2-经验总结.md](references/7.2-经验总结.md) | 有效经验、踩坑记录、可复用产物 |
| — | Issue 问题记录 | [references/Issue-问题记录.md](references/Issue-问题记录.md) | 开发中的问题独立成文（Background/Origin/Benefit/Design） |
| — | 开发日志 | [references/LOG-开发日志.md](references/LOG-开发日志.md) | 全程状态跟踪、进度跟踪（按统一流程编号） |
