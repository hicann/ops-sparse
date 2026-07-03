# Squash Commit 问卷

**触发条件**：提交 PR 前检查到当前分支相对于目标分支有大于 1 个 commit

```
问题：当前分支有 N 个 commit，不合并会导致 PR 被打上 stat/needs-squash 标签。是否合并为单个 commit？
选项：
- 合并为单个 commit（推荐）
- 保留所有 commit
```

**后续处理**：
- 选择"合并"：继续发送 [commit message 选择问卷](questionnaire-commit-message.md)
- 选择"保留"：直接创建 PR
