# 通用修改要点

> 本文件适用于所有文件类型的修改。每次修改完成后，必须逐项检查。

## 修改后必做

### 1. 检查本 skill 是否需要同步更新

修改工作流结构时（如新增/删除阶段、变更部署方式、调整目录结构），检查 `sparse-new-op-workflow-maintain/SKILL.md` 中的描述是否仍然准确：
- 目录结构树
- init.sh 部署原理表
- references/ 下的操作指南

### 2. 询问用户是否需要检视

每次修改完成后，**必须**使用 `question` 工具询问用户：

```
问题：工作流已修改完成，是否需要使用 reviewer 进行一致性检视？
选项：
- 是，启动 reviewer 检视
- 不需要，我手动检查
```

如果用户选择「是」，调用 reviewer subagent 执行工作流一致性检视。

### 3. 检视问题复盘

如果 reviewer 检视发现了问题，**必须**先复盘：对照本 skill 的 references/ 操作指南，分析为什么在 maintain skill 的指引下没有提前发现该问题。

- 如果操作指南中确实存在遗漏（如缺少某个检查步骤、未覆盖某种修改场景），使用 `question` 工具询问用户：

```
问题：reviewer 发现了 {问题摘要}，复盘发现 maintain skill 的 {具体文件} 中缺少 {具体检查项}。是否同步更新 maintain skill？
选项：
- 是，同步更新 maintain skill
- 不需要，仅修复当前问题
```

- 如果操作指南已覆盖但执行时遗漏，则仅修复当前问题，不修改 maintain skill

---

## 检视要点

当 reviewer 检视工作流修改时，应检查：

1. **流程一致性**：SKILL.md、task-prompts.md、data-flow.md、README.md 四处流程表是否一致
2. **引用完整性**：所有模板路径是否可达，所有 skill 名称是否存在
3. **逻辑完整性**：新增步骤是否有完整的输入/输出/验收标准，是否有错误处理路径
4. **CP 问卷一致性**：CP 否定分支表在 SKILL.md 和 error-handling.md 中是否一致
5. **LOG.md 一致性**：assets/LOG.md 步骤跟踪表是否与流程表步骤编号一致
