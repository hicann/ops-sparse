# 修改 Skill

> 适用于：`agent/skills/*/SKILL.md`、`agent/skills/cannbot_references.json`
>
> `<target>` 根据当前运行环境确定：OpenCode 环境使用 `opencode`，Claude Code 环境使用 `claude`。

## 新增一个 Skill

1. 创建 `agent/skills/<skill-name>/SKILL.md`
2. 若 skill 包含代码文件（`.sh`、`.py` 等），按 [script-coding-rules.md](script-coding-rules.md) 添加版权头并遵守编码规范
3. 若主 Agent 直接使用：在 `agent/AGENT.md` 的 `skills:` 列表中添加
4. 若 Subagent 使用：在对应 `agent/agents/<name>.md` 的 `skills:` 列表中添加
5. 检查 `task-prompts.md` 中是否有步骤需要「加载 <skill-name> 技能」，若有则确认步骤 4 中已添加
6. 如果是 cannbot 外部 skill，更新 `cannbot_references.json`
7. **必须**自行运行 `bash agent/init.sh <target>` 创建软链接使配置生效，**禁止**让用户退出当前会话重新运行 init
8. 执行 `references/common.md` 通用检查

## 修改已有 Skill

1. 修改 `agent/skills/<skill-name>/SKILL.md`
2. 检查是否有 agent 的 `skills:` frontmatter 引用了该 skill
3. 检查 `task-prompts.md` 中是否有步骤加载该 skill
4. 如果修改了 skill 的触发条件或功能范围，检查 `task-prompts.md` 中的加载指令是否需要更新
5. **如果添加了新功能或新流程**，检查所有下游消费方（agent 定义、task-prompts.md 步骤）是否需要在关键决策点主动引用该功能，避免"有工具无触发"的断裂
6. 执行 `references/common.md` 通用检查

## 删除一个 Skill

1. 删除 `agent/skills/<skill-name>/` 目录
2. 从 `agent/AGENT.md` 的 `skills:` 列表中移除
3. 从所有 `agent/agents/*.md` 的 `skills:` 列表中移除
4. 从 `task-prompts.md` 中移除所有「加载 <skill-name> 技能」的指令
5. 如果是 cannbot 外部 skill，从 `cannbot_references.json` 中移除
6. 自行运行 `bash agent/init.sh <target> --clean` 重新初始化，**禁止**让用户退出当前会话重新运行 init
7. 执行 `references/common.md` 通用检查

## cannbot_references.json 格式

```json
{
    "skill-name": ["path/in/cannbot-skills/repo"]
}
```

- **key**：skill 名称（在 `.opencode/skills/` 中显示的目录名）
- **value**：在 cannbot-skills 仓库中的路径数组
- **注意**：`op-samples-reference` 和 `asc-devkit-reference` 是**本地 skill**（在 `agent/skills/` 中维护），不在 `cannbot_references.json` 中。它们由 init.sh 步骤 5 自动创建软链接，与 `sparse-log` 等本地 skill 处理方式相同
