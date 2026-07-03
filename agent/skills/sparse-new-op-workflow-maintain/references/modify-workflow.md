# 修改工作流文件

> 适用于：`sparse-new-op-workflow/SKILL.md`、`references/task-prompts.md`、`references/data-flow.md`、`references/error-handling.md`、`assets/LOG.md`、`agent/README.md`

## 核心规则：流程文档四件套必须一致

以下四个文件描述同一流程，修改任一个时必须同步其余三个：

| 文件 | 用途 |
|------|------|
| `sparse-new-op-workflow/SKILL.md` | 主文档，含硬规则 + 流程表 |
| `references/task-prompts.md` | 各阶段调用参数（输入/输出/验收标准） |
| `references/data-flow.md` | 数据流图 |
| `agent/README.md` | 工作流概览（面向用户的精简版） |

**检查清单**：
- [ ] 步骤编号一致
- [ ] 输入/输出一致
- [ ] Subagent 一致
- [ ] 说明列一致

---

## 新增一个步骤

1. **SKILL.md**：在流程表中插入新行
2. **task-prompts.md**：
   - 任务恢复映射表新增行
   - 各阶段调用参数新增节（含输入/输出/验收标准）
3. **data-flow.md**：
   - 数据流图新增节点
   - 交付物目录树新增对应文件（如 `4.3-Issue.md`）
4. **agent/README.md**：流程表新增行
5. **error-handling.md**（可选）：新增错误处理路径
6. **LOG.md 模板**：在 `assets/LOG.md` 的步骤跟踪表中添加新行
7. 执行 `references/common.md` 通用检查

## 修改步骤输出物

当修改某个步骤的输出物（如新增/删除输出文件）时，除四件套流程表同步外，还需检查：

1. **data-flow.md 交付物目录树**：顶部的目录树是否需要同步新增/删除文件
2. **CP 问卷 JSON 交付物清单**：对应阶段的 CP 问卷（如 CP4.3.json）中 `交付物清单` 字段是否需要同步
3. **LOG.md 步骤描述**：`assets/LOG.md` 中对应步骤的括号描述是否需要同步（如新增/删除输出物关键词）
4. 执行 `references/common.md` 通用检查

## 删除或重排步骤

1. **SKILL.md**：删除/移动流程表中的行
2. **task-prompts.md**：同步删除/移动任务恢复映射表行 + 各阶段调用参数节
3. **data-flow.md**：同步删除/移动数据流节点
4. **agent/README.md**：同步删除/移动流程表行
5. **error-handling.md**：检查是否有对应的错误处理路径需要删除/调整
6. **LOG.md 模板**：检查 `assets/LOG.md` 中的步骤跟踪表是否需要同步
7. 执行 `references/common.md` 通用检查

## 修改 CP 问卷

CP 问卷修改涉及模板文件和流程文档，请按 [modify-template.md](modify-template.md) 中「修改 CP 问卷模板」章节操作，然后额外检查以下工作流一致性：
- SKILL.md 中 CP 否定分支表
- error-handling.md 中错误处理路径

## 修改步骤输入/验收标准

当修改 task-prompts.md 中某个步骤的输入或验收标准时，除四件套流程表同步外，还需检查：

1. **模板路径完整性**：输入中引用的所有模板文件（如 `xxx-报告.md 模板文件路径`）是否在 `assets/` 目录中存在，且路径拼写正确
2. **占位符/变量一致性**：若使用占位符（如 `<基准分支>`），必须在首次出现时定义其具体值（如 `<基准分支> = cann/master`），或直接使用具体值保持全文一致
3. **执行主体明确性**：验收标准中涉及"谁执行什么操作"时，必须明确执行主体（如"reviewer 必须自行执行"），避免与硬规则中的分工产生歧义
4. **白名单完整性**：若定义了文件白名单（如"允许修改的共享文件列表"），需查阅 git 历史确认所有相关修改的文件都已纳入，避免遗漏总是同步修改的文件
5. 执行 `references/common.md` 通用检查
