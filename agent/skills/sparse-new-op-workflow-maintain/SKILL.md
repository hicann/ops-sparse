---
name: sparse-new-op-workflow-maintain
description: sparse 工作流文件维护技能，对 agent/ 目录下文件执行新增/修改/删除时的强制流程。
---

# sparse-new-op-workflow 工作流维护指南

## 目录结构

### 仓库布局

```
ops-sparse/
├── agent/                          # 源文件目录（版本控制）
│   ├── AGENT.md                    # 主 Agent 配置
│   ├── README.md                   # 工作流概览
│   ├── QUICKSTART.md               # 快速开始指南
│   ├── init.sh                     # 部署脚本
│   ├── agents/                     # Subagent 定义
│   │   ├── developer.md
│   │   ├── tester.md
│   │   ├── reviewer.md
│   │   ├── architect.md
│   │   └── writer.md
│   └── skills/                     # Skill 定义（每个子目录为一个 skill）
│       ├── sparse-new-op-workflow/ # 主工作流 skill
│       │   ├── SKILL.md            # 工作流主文档（硬规则 + 流程表）
│       │   ├── references/         # 流程参考文档（task-prompts / data-flow / error-handling）
│       │   └── assets/             # 模板文件（报告模板 *.md + CP 问卷 *.json）
│       ├── <skill-name>/           # 其他 skill（本地维护，由 init.sh 步骤 5 软链接）
│       └── cannbot_references.json # cannbot 外部 skill 引用映射
│
├── .opencode/                      # OpenCode 运行时目录（init.sh 生成）
│   ├── AGENTS.md -> agent/AGENT.md
│   ├── agents/                     # 软链接到 agent/agents/
│   └── skills/                     # 软链接到 agent/skills/
│
├── .claude/                        # Claude Code 运行时目录（init.sh 生成）
│   ├── CLAUDE.md -> agent/AGENT.md
│   ├── agents/                     # 软链接到 agent/agents/
│   └── skills/                     # 软链接到 agent/skills/
│
└── .agent/                         # 临时工作目录（gitignore）
    ├── dev-docs/                   # 开发过程文档
    ├── gitcode/                    # PR/Issue 中间文件（gitcode-pr-issue-guide 生成）
    ├── cann-samples/               # 外部仓库（clone 或 symlink）
    └── asc-devkit/                 # 外部仓库（clone 或 symlink）
```

### 部署原理（init.sh）

`agent/init.sh` 负责将 `agent/` 下的源文件部署到运行时目录：

| 步骤 | 操作 | 说明 |
|------|------|------|
| 1 | 创建 `.opencode/` 或 `.claude/` + `.agent/dev-docs/` | 根据 target 参数选择，同时创建临时文档目录 |
| 2 | 软链接 `AGENT.md` | `AGENTS.md -> agent/AGENT.md`（opencode）或 `CLAUDE.md -> agent/AGENT.md`（claude） |
| 3 | 软链接 agents | `.opencode/agents/*.md -> agent/agents/*.md` |
| 4 | 设置 cannbot-skills | clone 或使用本地路径 |
| 5 | 软链接本地 skills | `.opencode/skills/* -> agent/skills/*` |
| 6 | 软链接 cannbot skills | 读取 `cannbot_references.json`，从 cannbot-skills 仓库软链接 |
| 7 | 设置外部参考仓库 | clone cann-samples 和 asc-devkit 到 `.agent/` |
| 8 | 生成 opencode.json | 读取 `model_config.json`，将非 default 的模型配置写入项目级 `opencode.json`（仅 opencode） |

**关键理解**：
- `agent/` 是**源文件目录**，所有修改都在这里进行
- `.opencode/` 和 `.claude/` 是**运行时目录**，由 init.sh 通过软链接生成
- 修改 `agent/skills/xxx/` 后，如果已运行过 init.sh，软链接会自动指向新内容，无需重新运行
- 新增 skill 目录后，需要重新运行 `bash agent/init.sh <target>` 以创建新的软链接

### cannbot_references.json

该文件定义了从 `cannbot-skills` 外部仓库引用的 skill：

```json
{
    "skill-name": ["path/in/cannbot-skills/repo"]
}
```

- **key**：skill 名称（在 `.opencode/skills/` 中显示的目录名）
- **value**：在 cannbot-skills 仓库中的路径数组

**注意**：`op-samples-reference` 和 `asc-devkit-reference` 是**本地 skill**（在 `agent/skills/` 中维护），不在 `cannbot_references.json` 中。它们由 init.sh 步骤 5 自动创建软链接，与 `sparse-log` 等本地 skill 处理方式相同。

### cannbot-skills 默认路径

init.sh 步骤 4 将 cannbot-skills 仓库 clone 到 `$CONFIG_DIR/ref-repos/cannbot-skills`（即 `.opencode/ref-repos/cannbot-skills` 或 `.claude/ref-repos/cannbot-skills`）。步骤 6 从该路径创建软链接到 `.opencode/skills/`。

---

## 修改操作指南

根据修改的文件类型，加载对应的参考文档：

| 修改类型 | 参考文档 | 涉及文件 |
|---------|---------|---------|
| 修改工作流流程 | [references/modify-workflow.md](references/modify-workflow.md) | SKILL.md、task-prompts.md、data-flow.md、error-handling.md、LOG.md、README.md |
| 修改 Agent 定义 | [references/modify-agent.md](references/modify-agent.md) | agents/*.md、AGENT.md |
| 新增/修改/删除 Skill | [references/modify-skill.md](references/modify-skill.md) | skills/*/SKILL.md、cannbot_references.json |
| 修改模板文件 | [references/modify-template.md](references/modify-template.md) | assets/*.md、assets/*.json |
| 修改 init.sh | [references/modify-init.md](references/modify-init.md) | init.sh、QUICKSTART.md |
| Skill 中的脚本编码 | [references/script-coding-rules.md](references/script-coding-rules.md) | skills/*/\*.sh、skills/*/\*.py |

**所有修改完成后**，必须执行 [references/common.md](references/common.md) 中的通用检查。
