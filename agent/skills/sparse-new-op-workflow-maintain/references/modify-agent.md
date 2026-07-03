# 修改 Agent 定义

> 适用于：`agent/agents/*.md`、`agent/AGENT.md`

## 修改 Subagent（agents/*.md）

1. 修改 `agent/agents/<agent-name>.md`
2. 检查该 agent 的 `skills:` frontmatter 列表是否与 `task-prompts.md` 中要求其加载的 skill 一致
3. 检查 `agent/AGENT.md` 中的 `agents:` 列表是否需要更新
4. 检查 `task-prompts.md` 中引用该 agent 的地方是否需要更新
5. 执行 `references/common.md` 通用检查

## 修改主 Agent（AGENT.md）

1. 修改 `agent/AGENT.md`
2. 检查 `skills:` 列表是否包含所有主 Agent 直接使用的 skill
3. 检查 `agents:` 列表是否包含所有可用的 subagent
4. **新增全局规则时**，检查下游 skill（如 `gitcode-pr-issue-guide`、`sparse-new-op-workflow`）和 Subagent 定义文件（`agents/*.md`）是否需要配合执行该规则，若需要则在对应文件中增加引用或执行步骤
5. 执行 `references/common.md` 通用检查

## frontmatter 字段说明

```yaml
---
name: <agent-name>          # agent 标识符
description: <描述>          # 触发条件描述
mode: subagent | primary    # subagent 或 primary
skills:                     # 该 agent 可加载的 skill 列表
  - skill-name-1
  - skill-name-2
---
```

**注意**：`skills:` 列表决定了该 agent 在运行时可以通过 `skill` 工具加载哪些 skill。如果 `task-prompts.md` 中要求某 agent「加载 xxx 技能」，则该 skill 必须出现在该 agent 的 `skills:` 列表中。
