---
name: gitcode-pr-issue-guide
description: 生成 PR 或 Issue 文件，并可通过 GitCode API 直接提交到仓库。仅当用户明确说"提PR"、"创建PR"、"提交Issue"、"推送到PR"、"amend"、"force push"等时才触发，禁止根据上下文推断自动触发。所有操作必须经过用户许可，禁止擅自行动。
---

# PR / Issue 模板填写技能

## 概述

本技能帮助用户快速生成符合仓库规范的 PR 或 Issue 文件。通过读取模板 + 问卷确认的方式，确保内容完整准确。模板直接复用仓库 `.gitcode/` 目录下的官方模板。生成后可通过 GitCode API 直接提交到仓库。

## 硬规则

1. **仅响应明确指令** — 只有当收到明确的"提 PR"、"创建 PR"、"提交 Issue"、"推送到 PR"、"amend"、"force push"等指令（来自用户或工作流显式调用）时，才加载本技能并执行流程。**禁止**根据上下文推断意图并主动触发，**禁止**在未收到明确指令时自动进入 PR/Issue 流程
2. **所有操作必须经过用户许可** — 流程中的每个关键操作（squash、token 使用、PR 创建、触发编译等）都必须发送问卷获得用户确认后才能执行，**禁止**擅自行动。多个阶段可以合并在一次问卷中确认以提高效率，但每个操作都必须有明确的用户许可

## 模板清单

| 模板类型 | 模板路径 | 输出文件名 |
|---------|---------|-----------|
| Pull Request | [assets/PULL_REQUEST_TEMPLATE.zh-CN.md](assets/PULL_REQUEST_TEMPLATE.zh-CN.md) | `PR-{branch-or-title}.md` |
| 缺陷反馈 | [assets/bug-report.yml](assets/bug-report.yml) | `ISSUE-bug-{title}.md` |
| 文档反馈 | [assets/documentation.yml](assets/documentation.yml) | `ISSUE-doc-{title}.md` |
| 需求建议 | [assets/feature-request.yml](assets/feature-request.yml) | `ISSUE-feature-{title}.md` |
| 问题咨询 | [assets/question.yml](assets/question.yml) | `ISSUE-question-{title}.md` |

## 问卷模板清单

| 问卷 | 模板路径 | 触发条件 |
|------|---------|---------|
| PR-Issue 关联 | [assets/questionnaire-pr-issue-link.md](assets/questionnaire-pr-issue-link.md) | 用户只提到提 PR 时 |
| Token 获取方式 | [assets/questionnaire-token-source.md](assets/questionnaire-token-source.md) | 需要调用 GitCode API 时 |
| Squash Commit | [assets/questionnaire-squash-commit.md](assets/questionnaire-squash-commit.md) | 分支有 >1 个 commit 时 |
| Commit Message 选择 | [assets/questionnaire-commit-message.md](assets/questionnaire-commit-message.md) | 用户同意 squash 后 |
| 触发编译 | [assets/questionnaire-trigger-compile.md](assets/questionnaire-trigger-compile.md) | PR 创建成功后 |
| Issue 指派人 | [assets/questionnaire-issue-assignee.md](assets/questionnaire-issue-assignee.md) | 创建 Issue 时（提 PR 同步创建则默认 assign 给用户） |

## 工作流程

### 步骤 1：确定模板类型

根据用户请求判断需要填写的模板类型（PR 或某种 Issue）。如果无法判断，询问用户。

**当用户只提到提 PR 时**，必须使用 `question` 工具发送 [PR-Issue 关联问卷](assets/questionnaire-pr-issue-link.md)。

### 步骤 2：读取模板并预填内容

1. 读取对应的模板文件（`assets/`）
2. 根据当前对话上下文、git 状态、代码变更等信息，预填模板中各字段的候选内容
3. 对于无法预填的字段，标记为"待填写"

### 步骤 3：发送问卷

使用 `AskUserQuestion` 工具，将推断出的内容以问卷形式发送给用户：

- 每个字段作为一个问题
- 提供推断出的内容作为默认选项
- 允许用户选择"使用默认值"或"自定义填写"
- 对于必填字段，明确标注
- **Issue 的"信息来源（Origin）"字段默认值为 `cann 开发者`，无需发问卷询问，直接填入**

**问卷格式示例：**

```
问题 1 - 描述（必填）
推断内容：本次 PR 新增了 XX 算子的实现...
选项：
- [ ] 使用推断内容
- [ ] 自定义填写

问题 2 - 关联的 Issue
推断内容：Issue #123
选项：
- [ ] 使用推断内容
- [ ] 无关联 Issue
- [ ] 自定义填写
```

### 步骤 4：逐项确认

用户逐项确认或补充每个字段的内容。对于用户选择"自定义填写"的字段，等待用户提供具体内容。

### 步骤 4.5：合规审查

在生成文件前，对所有确认后的内容进行合规审查（参见 AGENT.md「公开内容合规限制」规则）：

- 检查是否包含竞品对标、模型暴露、商业敏感信息、舆论风险内容
- 发现违规内容时，自动过滤并告知用户修改原因
- 过滤后无需重新发送问卷，直接在生成文件中体现

### 步骤 5：生成文件

1. 根据用户确认的内容，填充模板
2. 生成 Markdown 文件，保存到仓库根目录的 **`.agent/gitcode/`** 目录下（如目录不存在，先 `mkdir -p .agent/gitcode` 创建）
3. 文件名格式见模板清单
4. 输出生成结果，告知用户文件路径

### 步骤 6：询问是否直接提交

生成文件后，询问用户是否需要通过 GitCode API 直接提交到仓库。如果需要，进入「GitCode API 提交流程」。

### 输出格式规则

**PR 模板**：模板本身是 md 格式，直接填充 HTML 注释占位符即可，保留原有结构。

**Issue 模板（yml 来源）**：yml 文件定义了网页表单的多个文本框（textarea），每个文本框对应一个字段。生成输出时，必须转换为 md 格式，每个字段作为一个独立段落，方便用户逐段复制粘贴到网页表单的对应文本框中。

输出格式示例（以 bug-report.yml 为例）：

```markdown
## 问题描述

本次发现的缺陷是...

## 环境信息

- 芯片型号：Ascend 910B
- CANN 版本：8.0.RC1
- 操作系统：Ubuntu 20.04

## 重现步骤

1. 执行 xxx 命令
2. 观察到 xxx 异常

## 预期结果

期望输出应为...

## 日志 / 截图

（粘贴日志或截图）

## 备注

无
```

规则：
- 每个 yml `textarea` 字段对应一个 `##` 二级标题，标题文字取 yml 中 `label` 的中文部分
- 标题之间用空行分隔，确保每个段落可独立选中复制
- 选填字段如用户未提供内容，填写"无"
- 不输出 yml 原始格式，只输出纯 md

---

## GitCode API 提交流程

当用户要求直接提交 Issue 或 PR 到 GitCode 仓库时，按以下流程操作。

### 前置信息

- 仓库地址：`https://gitcode.com/cann/ops-sparse`
- API 基础路径：`https://gitcode.com/api/v5/repos/cann/ops-sparse`
- 认证方式：URL 参数 `access_token=<token>`

### 获取 Token

**必须先获取用户的 GitCode Access Token，再执行任何 API 调用。**

**按以下顺序获取，禁止跳步：**

1. **自动读取**（静默提取，不打印 token）：执行以下命令从 `~/.git-credentials` 提取 token：
   ```bash
   grep 'gitcode.com' ~/.git-credentials 2>/dev/null | head -1 | sed 's|https://[^:]*:\([^@]*\)@.*|\1|'
   ```
   - 如果提取到非空 token → 进入步骤 2 询问用户
   - 将 token 保存为变量 `TOKEN`，**禁止**打印到输出

2. **用户确认**（必须执行，无论是否读取到 token）：使用 `question` 工具发送 [Token 获取方式问卷](assets/questionnaire-token-source.md)
   - 如果步骤 1 读取到 token → 问卷选项包含"使用从 git-credentials 读取的 token 直接发送"和"手动提供其他 token"
   - 如果步骤 1 未读取到 token → 问卷选项仅包含"手动提供 token"
   - **禁止**在未获得用户确认的情况下使用任何 token 调用 API

3. **每次 API 调用前确认**：每次使用 token 直接发送（创建 PR、提交 Issue、发送评论等）之前，都**必须**发送问卷询问用户是否使用该 token 直接发送，**禁止**擅自使用 token 做任何操作

### 验证 Token（可选）

提交前可先验证 token 是否有效：

```bash
bash scripts/verify_token.sh "<用户token>" "cann/ops-sparse"
```

### 提交 Issue

**步骤 0：确认 Issue 指派人**

创建 Issue 前，需确认 assignee 设置：

- **单独创建 Issue**：使用 `question` 工具发送 [Issue 指派人问卷](assets/questionnaire-issue-assignee.md)，询问用户
- **提 PR 时同步创建 Issue**：**无需发问卷**，默认 assign 给当前提 PR 的用户（从 `~/.git-credentials` 提取用户名）

将用户名保存为变量 `ISSUE_ASSIGNEE`。

使用 `scripts/batch_create_issues.sh` 批量提交：

```bash
bash scripts/batch_create_issues.sh "<用户token>" "cann/ops-sparse" "<issue目录>" "[文件模式]" "[assignee]"
```

**参数说明：**
- `token`：用户的 GitCode access token
- `repo`：仓库路径，如 `cann/ops-sparse`
- `issue_dir`：包含 issue md 文件的目录
- `file_pattern`：文件匹配模式，默认 `ISSUE-bug-*.md`

**示例：**
```bash
bash scripts/batch_create_issues.sh "abc123" "cann/ops-sparse" "/path/to/issues" "ISSUE-bug-*.md"
```

脚本会：
- 遍历匹配的 md 文件
- 提取第一行作为标题（去掉 `# ` 前缀）
- 剩余内容作为正文
- 逐个调用 API 创建 issue
- 每个请求间隔 1 秒避免限流
- 输出每个 issue 的 URL 和最终统计

### 提交 PR

**步骤 0：确认分支已 rebase 最新目标分支**

提交 PR 前，**必须**先确认当前分支已 rebase 了最新的目标分支（通常为 master/main），避免代码冲突：

```bash
git fetch origin <目标分支>
git merge-base --is-ancestor origin/<目标分支> HEAD
```

- 如果命令返回 0（成功）：当前分支已包含最新目标分支的所有提交，可以继续
- 如果命令返回 1（失败）：当前分支落后于目标分支，**必须**先执行 rebase：

```bash
git rebase origin/<目标分支>
```

rebase 完成后，由于提交历史已改写，后续推送需使用 `git push --force-with-lease`。

如果 rebase 过程中出现冲突，**必须**解决冲突后再继续，**禁止**在存在冲突的情况下提交 PR。

如果 rebase 发生在 PR 已创建之后，rebase 后的 force push 触发编译询问由步骤 3 统一处理，无需按 AGENT.md 推送后规则重复询问。

**步骤 1：检查并合并 commit**

确认 rebase 完成后，检查当前分支相对于目标分支的 commit 数量：

```bash
git log --oneline <当前分支> --not <目标分支>
```

如果 commit 数量大于 1，**必须**使用 `question` 工具发送 [Squash Commit 问卷](assets/questionnaire-squash-commit.md)。如果用户选择合并，继续发送 [Commit Message 选择问卷](assets/questionnaire-commit-message.md)。

**步骤 2：创建 PR**

使用 `scripts/create_pr.sh` 创建 PR：

```bash
bash scripts/create_pr.sh "<用户token>" "cann/ops-sparse" "<PR标题>" "<源分支>" "<目标分支>" "<正文文件>"
```

**参数说明：**
- `token`：用户的 GitCode access token
- `repo`：仓库路径，如 `cann/ops-sparse`
- `title`：PR 标题
- `head`：源分支名（包含你的改动）
- `base`：目标分支名（通常为 `main` 或 `master`）
- `body_file`：PR 正文的 md 文件路径

**示例：**
```bash
bash scripts/create_pr.sh "abc123" "cann/ops-sparse" "新增 XX 算子" "feature-xx" "main" "PR-xx.md"
```

**步骤 2.5：更新 PR 描述（可选）**

当需要追加关联 Issue 或修改 PR 描述时，使用 `scripts/update_pr.sh` 更新：

```bash
bash scripts/update_pr.sh "<用户token>" "cann/ops-sparse" "<PR编号>" "<正文文件>"
```

**参数说明：**
- `token`：用户的 GitCode access token
- `repo`：仓库路径，如 `cann/ops-sparse`
- `pr_number`：PR 编号
- `body_file`：新 PR 正文的 md 文件路径

**示例：**
```bash
bash scripts/update_pr.sh "abc123" "cann/ops-sparse" "132" ".agent/gitcode/pr-body-update.md"
```

**步骤 3：询问是否触发编译**

PR 创建成功后，使用 `question` 工具发送 [触发编译问卷](assets/questionnaire-trigger-compile.md)。后续向 PR 分支推送代码时的编译询问由「向已创建的 PR 推送代码」流程的步骤 C 统一处理。

**前提条件：**
1. 代码已 commit 并 push 到远程分支
2. 用户已确认源分支和目标分支名称
3. 用户已确认是否需要合并 commit

### 向已创建的 PR 推送代码

当 PR 已创建，后续需要推送新代码时（如修复 review 意见、补充修改），按以下流程操作：

**步骤 A：询问是否 amend**

使用 `question` 工具询问用户：

```
问题：PR 已存在，新代码如何处理？
选项：
- amend 到最后一个 commit（推荐，保持提交历史简洁）
- 新建 commit（保留独立提交记录）
```

**步骤 B：根据选择执行**

无论选择哪种方式，都必须先检查当前分支相对于目标分支的 commit 数量（同步骤 1）。如果 commit 数量大于 1，**必须**发送 [Squash Commit 问卷](assets/questionnaire-squash-commit.md)。如果用户选择合并，继续发送 [Commit Message 选择问卷](assets/questionnaire-commit-message.md)。

- **选择 amend**：
  ```bash
  git add <修改的文件>
  git commit --amend --no-edit
  git push --force-with-lease
  ```
  如果之前执行了 squash，则 squash 操作已包含 push，无需再次推送。

- **选择新建 commit**：
  ```bash
  git add <修改的文件>
  git commit -m "<commit message>"
  ```
  如果之前执行了 squash，则 squash 操作已包含 push，无需再次推送。如果用户选择不合并或只有 1 个 commit，执行 `git push`。

**步骤 C：询问是否触发编译**

推送完成后，发送 [触发编译问卷](assets/questionnaire-trigger-compile.md)。此步骤已覆盖 AGENT.md「推送后触发编译」规则，无需按 AGENT.md 重复询问。

### 提交前检查清单

提交 Issue 前：
- [ ] 已获取用户 token
- [ ] 已验证 token 有效（可选）
- [ ] md 文件已生成且内容已确认

提交 PR 前：
- [ ] 已获取用户 token
- [ ] 分支已 rebase 最新目标分支（无冲突）
- [ ] 代码已 commit 并 push 到远程分支
- [ ] 已确认 head 分支和 base 分支名称
- [ ] PR 正文已填充完整

### 提交行内评论（代码行评论）

在 PR 的特定代码行上添加评论（diff_comment 类型），常用于代码检视、Review 意见等场景。

**关键概念：**
- `position` 参数是 **diff 相对行号**，不是文件行号
- 脚本内部会自动从文件行号转换为 diff position
- **只能评论 PR 实际修改的行**（diff 中的 `+` 行），未变更的行无法评论

**单条评论：**

使用 `scripts/comment_pr_inline.sh`：

```bash
bash scripts/comment_pr_inline.sh "<token>" "<repo>" "<pr_number>" "<file>" "<line>" "<comment>"
```

**示例：**
```bash
bash scripts/comment_pr_inline.sh "abc123" "cann/ops-sparse" "120" \
    "sparse/csrmv/csrmv/arch22/csrmv_host.cpp" "113" \
    "**[HIGH] SEC-1.2**: handle 未做空指针校验"
```

**批量评论：**

使用 `scripts/comment_pr_inline_batch.py`：

```bash
# 方式 1: 多个 --comment 参数
bash scripts/comment_pr_inline_batch.py \
    --token "abc123" --repo "cann/ops-sparse" --pr "120" \
    --comment "file1.cpp:42:评论内容1" \
    --comment "file2.cpp:100:评论内容2"

# 方式 2: JSON 文件批量输入
bash scripts/comment_pr_inline_batch.py \
    --token "abc123" --repo "cann/ops-sparse" --pr "120" \
    --json comments.json
```

**JSON 文件格式：**
```json
[
  {"file": "path/to/file.cpp", "line": 42, "body": "**[HIGH]** 问题描述..."},
  {"file": "path/to/other.cpp", "line": 100, "body": "**[MED]** 另一条评论..."}
]
```

**注意事项：**
- `line` 必须是 PR 实际修改的行（diff 中的 `+` 行），否则脚本会尝试定位最近的修改行
- 评论内容支持 Markdown 格式
- 每条评论间隔 1 秒避免限流
- 脚本会自动获取 PR 文件列表并构建 position 映射

---

## 脚本清单

| 脚本 | 用途 | 路径 |
|------|------|------|
| `verify_token.sh` | 验证 GitCode token 是否有效 | `scripts/verify_token.sh` |
| `batch_create_issues.sh` | 批量创建 Issue | `scripts/batch_create_issues.sh` |
| `create_pr.sh` | 创建 Pull Request | `scripts/create_pr.sh` |
| `update_pr.sh` | 更新已有 PR 的描述 | `scripts/update_pr.sh` |
| `comment_pr.sh` | 在 PR 评论区添加普通评论 | `scripts/comment_pr.sh` |
| `comment_pr_inline.sh` | 在 PR 代码行上添加单条行内评论 | `scripts/comment_pr_inline.sh` |
| `comment_pr_inline_batch.py` | 批量添加 PR 行内评论（支持 JSON 输入） | `scripts/comment_pr_inline_batch.py` |

所有脚本均位于本 skill 的 `scripts/` 目录下，使用时需提供完整路径或使用相对路径。

---

## 注意事项

- 生成前必须让用户确认所有必填字段
- 推断内容仅供参考，最终以用户确认为准
- 文件生成在 `.agent/gitcode/` 目录下，而非仓库根目录或 `.opencode/` 目录下
- 如果用户中途放弃，不生成文件
- **Token 安全**：token 仅在当前会话中使用，不写入任何文件或代码，不打印到输出中
- **限流保护**：批量提交时每个请求间隔 1 秒（`sleep 1`）
- **错误处理**：API 返回错误时打印完整错误信息，不静默跳过
