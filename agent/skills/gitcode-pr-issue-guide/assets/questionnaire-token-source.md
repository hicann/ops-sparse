# Token 获取方式问卷

**触发条件**：需要调用 GitCode API 时，**必须**发送本问卷询问用户，无论是否已从 `~/.git-credentials` 读取到 token。

**安全规则**：获取到 token 之后保存为变量 `TOKEN`，**禁止**将 token 打印到输出或写入任何文件。

---

## 场景 A：已从 `~/.git-credentials` 读取到 token

```
问题：已从 ~/.git-credentials 中找到 GitCode token，是否使用该 token 直接发送？
选项：
- 使用已读取的 token 直接发送
- 手动提供其他 token（用户在自定义输入中粘贴）
```

**后续处理**：
- 用户选择"使用已读取的 token" → 使用已保存的 `TOKEN` 变量
- 用户选择"手动提供" → 等待用户在自定义输入中粘贴 token

---

## 场景 B：未从 `~/.git-credentials` 读取到 token

```
问题：未从 ~/.git-credentials 中找到 GitCode token，请手动提供：
选项：
- 我来手动提供 token（用户在自定义输入中粘贴）
```

**后续处理**：
- 等待用户在自定义输入中粘贴 token

---

## 每次 API 调用前确认

**重要**：每次使用 token 直接发送（创建 PR、提交 Issue、发送评论等）之前，都**必须**发送本问卷询问用户是否使用该 token 直接发送，**禁止**擅自使用 token 做任何操作。
