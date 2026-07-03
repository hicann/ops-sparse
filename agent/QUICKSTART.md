# Sparse Agent 快速使用

**Step 1**：初始化

```bash
# Claude Code
bash agent/init.sh claude

# OpenCode
bash agent/init.sh opencode
```

**Step 2**：启动

```bash
# Claude Code
claude

# OpenCode
opencode
```

**Step 3**：描述需求

> 帮我开发一个 ascend910b 上的 SpMV 算子，支持 CSR 格式和 FP32 数据类型

---

**特殊情况**

使用本地 cannbot-skills：

```bash
bash agent/init.sh claude --cannbot /path/to/cannbot-skills
```

重新初始化：

```bash
bash agent/init.sh claude --clean
```

仅清理环境：

```bash
bash agent/init.sh --clean
```

**配置 Agent 模型**（仅 OpenCode）：

创建 `agent/agents/model_config.json`（该文件在 `.gitignore` 中）：

```json
{
  "architect": {
    "comment": "架构师",
    "model": "provider/model-id"
  },
  "developer": {
    "comment": "开发工程师",
    "model": "provider/model-id"
  }
}
```

未列出的 agent 默认使用 `"default"`（跟随主 Agent 模型）。

**格式说明**：
- `model` 字段格式为 `provider/model-id`，例如 `alibaba-cn/qwen3-coder-480b-a35b-instruct`、`deepseek/deepseek-reasoner`
- 可通过 `opencode models` 查看当前环境可用的模型列表
- 未列出的 agent（如 tester、reviewer、writer）默认使用 `"default"`，即跟随主 Agent 的模型

**手动配置时**，配置完成后重新运行初始化：

```bash
bash agent/init.sh opencode
```

**Agent 配置时**，Agent 会自动运行 init.sh 使配置生效，无需手动操作。

脚本会自动将非 `"default"` 的配置写入项目根目录的 `opencode.json`（与已有内容合并）。全部为 `"default"` 时不生成 `opencode.json`。

**模型校验**：init 阶段会通过 `opencode models` 获取当前环境可用模型列表，如果配置的模型不在列表中，会自动回退到 `"default"` 并输出 warning。

**重置配置**：如需重置为默认配置，删除 `agent/agents/model_config.json` 后重新运行 `bash agent/init.sh opencode`，脚本会自动清理 `opencode.json` 中的旧配置。
