# Issue 指派人问卷

**触发条件**：创建 Issue 时，必须发送本问卷询问用户是否要将 Issue assign 给自己。

**场景 A：单独创建 Issue（不是提 PR 时同步创建）**

```
问题：是否将此 Issue assign 给当前提 PR 的用户？
选项：
- 是，assign 给我（默认选项）
- 否，不 assign 任何人
- assign 给其他用户（用户在自定义输入中指定用户名）
```

**场景 B：提 PR 时同步创建 Issue**

**无需发送问卷**，直接默认 assign 给当前提 PR 的用户。

---

## 获取当前用户

从 `~/.git-credentials` 中提取用户名：

```bash
grep 'gitcode.com' ~/.git-credentials 2>/dev/null | head -1 | sed 's|https://\([^:]*\):.*|\1|'
```

将用户名保存为变量 `ISSUE_ASSIGNEE`，**禁止**打印到输出。

---

## 使用场景

- 单独创建 Issue → 发送问卷询问用户
- 提 PR 并同步创建 Issue → 不询问，直接 assign 给当前用户

---

## 后续 API 调用

当用户选择"assign 给我"或"assign 给其他用户"时，在调用 GitCode API 创建 Issue 时需要传递 `assignee` 参数：

```bash
bash scripts/batch_create_issues.sh "<token>" "<repo>" "<issue_dir>" "<pattern>" "<assignee>"
```

**参数说明：**
- `assignee`：GitCode 用户名（如 `xutianze`），不传或传空字符串表示不 assign
