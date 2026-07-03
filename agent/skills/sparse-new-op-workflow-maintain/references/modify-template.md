# 修改模板文件

> 适用于：`agent/skills/sparse-new-op-workflow/assets/*.md`、`agent/skills/sparse-new-op-workflow/assets/*.json`

## 修改报告模板（assets/*.md）

1. 修改 `agent/skills/sparse-new-op-workflow/assets/<template>.md`
2. 检查 `task-prompts.md` 中引用该模板的路径是否正确（搜索 `模板路径:` 关键字）
3. 检查模板中的占位符（如 `{operator_name}`、`{Xxx}`）是否与 task-prompts.md 中的输出文件名一致
4. 执行 `references/common.md` 通用检查

## 修改 CP 问卷模板（assets/CP*.json）

1. 修改 `agent/skills/sparse-new-op-workflow/assets/CP*.json`
2. 检查 SKILL.md 中 CP 否定分支表是否需要更新
3. 检查 error-handling.md 中对应的错误处理路径是否需要更新
4. 执行 `references/common.md` 通用检查

## 修改 LOG.md 模板（assets/LOG.md）

1. 修改 `agent/skills/sparse-new-op-workflow/assets/LOG.md`
2. 检查步骤跟踪表是否与 SKILL.md 流程表的步骤编号一致
3. 执行 `references/common.md` 通用检查

## 模板路径引用规范

在 `task-prompts.md` 中引用模板时，使用以下格式：

```yaml
输入:
  - <模板描述> 模板文件路径 (模板路径: agent/skills/sparse-new-op-workflow/assets/<文件名>)
输出:
  - .agent/dev-docs/{operator_name}/<输出文件名> (按模板填写)
```
