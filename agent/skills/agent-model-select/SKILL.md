---
name: agent-model-select
description: Agent 模型选择技能，读取 model_config.json 配置并为各 subagent 设置对应模型。触发：配置模型、选模型、model config、换个模型、模型推荐。
---

# Agent 模型选择

## 核心原则

**禁止硬编码 agent 列表**。必须从 `agent/AGENT.md` 的 `agents:` 字段和每个 `agent/agents/<name>.md` 的 frontmatter 动态读取。

## 模型选择原则

### 能力→模型映射

| Agent 角色 | 核心能力需求 | 推荐模型类型 |
|-----------|-------------|-------------|
| architect | 强推理、架构设计 | 旗舰推理模型（reasoning/flagship） |
| developer | 代码生成、调试 | 旗舰代码模型（coder/flagship） |
| reviewer | 代码审查、规范检查 | 中高推理模型 |
| tester | 测试设计、用例生成 | 中高推理模型 |
| writer | 文档撰写、模板填充 | 中端/默认模型 |

### 模型类型识别

通过 `opencode models` 输出识别模型类型：
- **推理模型**：名称含 `max` / `reasoner` / `thinking` / `o1`
- **代码模型**：名称含 `coder` / `code` / `devin`
- **通用模型**：名称含 `plus` / `turbo` / `flash` / `mini`

### 选择策略

| 策略 | 适用场景 | 分配规则 |
|------|---------|---------|
| **性能优先** | 复杂算子、高精度要求 | architect/developer 用旗舰，其余用中高 |
| **平衡** | 常规开发 | architect 用旗舰推理，developer 用旗舰代码，其余用默认 |
| **成本优先** | 简单算子、快速验证 | 仅 architect 用中高，其余用默认 |

## 配置文件

`agent/agents/model_config.json`（该文件在 `.gitignore` 中，本地修改不被 git 追踪）

## 配置格式

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

- `model` 字段格式：`provider/model-id`
- 未列出的 agent 默认使用 `"default"`（跟随主 Agent 模型）
- 可通过 `opencode models` 查看可用模型列表

## 步骤化流程

1. **获取可用模型列表**：执行 `opencode models`
2. **获取 agent 列表**：读取 `agent/AGENT.md` 的 `agents:` 字段
3. **询问用户偏好**：使用 `question` 工具询问选择策略（性能优先/平衡/成本优先）
4. **为每个 agent 选择模型**：根据策略和能力映射表分配
5. **写入配置文件**：生成 `agent/agents/model_config.json`（为**所有** agent 写入，包括使用 default 的）
6. **执行 init.sh**：运行 `bash agent/init.sh opencode` 使配置生效
7. **逐 agent 确认与修改**：使用 `question` 工具发送问卷，动态列出所有 agent 的当前配置，询问用户是否需要修改：
   ```
   问题：以下是各 agent 的模型配置，是否需要修改？

   当前配置：
   - <agent-name>（<角色描述>）: <model>
   - ...

   选项：
   - 确认，无需修改
   - 需要修改 <agent-name> 的模型
   - ...
   ```
   允许多选。若用户选择了需要修改的 agent，则针对选中的 agent 发送新问卷，让用户从可用模型列表中选择新模型，更新配置后重新执行 Step 6。

8. **确认结果**：向用户展示最终配置和生效状态

## 重要说明

- `opencode.json` 修改后**无需重启** opencode，下次调用 subagent 时自动生效
- 主 Agent 的模型在 opencode 启动时选择，不通过此技能配置
- `model_config.json` 已在 `.gitignore` 中，不会被 git 追踪
- 重置配置：删除 `model_config.json` 后重新运行 `bash agent/init.sh opencode`

## 生效方式

配置完成后运行 `bash agent/init.sh opencode` 即可生效。
