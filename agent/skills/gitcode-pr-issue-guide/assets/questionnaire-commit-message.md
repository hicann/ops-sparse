# Commit Message 选择问卷

**触发条件**：用户同意 squash commit 后

**发问卷前**：先根据所有 commit 的内容生成一条推荐的 commit message，然后列出所有现有 commit message 供选择。

**合规过滤**：生成推荐 message 和展示现有 commit message 前，必须按 AGENT.md「公开内容合规限制」规则过滤竞品对标、模型暴露等违规内容。发现违规时自动修改措辞并在问卷中标注已过滤。

```
问题：合并后的 commit message 使用哪个？
推荐：Feat: <根据所有 commit 内容总结的一句话描述>
选项：
- 使用推荐：<推荐的 message>
- 保留 commit 1：<第一个 commit message>
- 保留 commit 2：<第二个 commit message>
- ...（列出所有 commit）
- 自定义填写
```

**后续处理**：用户确认后，**先检查工作区是否有未暂存的修改**：

```bash
git status --short
```

如果有未暂存的修改，必须先 `git add` 暂存所有修改，再执行 squash：

```bash
git add -A
git reset --soft $(git merge-base <当前分支> <目标分支>)
git commit -m "<用户确认的 commit message>"
git push <remote> <当前分支> --force-with-lease
```
