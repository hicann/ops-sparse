# 触发编译问卷

**触发条件**：PR 创建成功后，或每次向 PR 分支推送代码后

```
问题：PR 已创建成功，是否需要在评论区评论 "compile" 触发编译？
选项：
- 是，触发编译（推荐）
- 否，跳过
```

**后续处理**：
- 选择"是"：使用 `scripts/comment_pr.sh` 添加评论

```bash
bash scripts/comment_pr.sh "<用户token>" "cann/ops-sparse" "<PR编号>" "compile"
```

- 选择"否"：跳过
