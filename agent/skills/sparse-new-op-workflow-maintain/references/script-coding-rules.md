# 脚本编码规范

> 适用于 skill 目录下的所有代码文件：`.sh`、`.py`

## 1. 版权头（强制）

所有代码文件必须在文件开头添加以下版权头。

**Shell 脚本（.sh）：**

```bash
#!/bin/bash
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
```

**Python 脚本（.py）：**

```python
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
```

## 2. Python 日志规范（强制）

Python 脚本中**禁止使用 `print`** 输出日志，必须使用 `logging` 模块：

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

logger.info("操作成功: %s", result)
logger.warning("非预期情况: %s", detail)
logger.error("操作失败: %s", error_msg)
```

**原因**：`print` 无法控制日志级别、无法格式化时间戳、无法按模块过滤，不利于问题排查。

## 3. Python 函数体行数限制（强制）

`main()` 函数体**不得超过 50 行**（不含空行和注释），其他函数建议不超过 30 行。超过时必须拆分为子函数。

**原因**：超大函数可读性差，难以测试和维护。

```python
# ❌ main 函数过长
def main():
    # 80 行代码...

# ✅ 拆分为子函数
def parse_args():
    ...

def load_comments(args):
    ...

def post_all_comments(config, comments, pos_maps):
    ...

def main():
    args = parse_args()
    config = build_config(args)
    comments = load_comments(args)
    pos_maps = build_position_maps(config)
    success, failed = post_all_comments(config, comments, pos_maps)
    logger.info("发布完成: 成功 %d, 失败 %d", success, failed)
    sys.exit(0 if failed == 0 else 1)
```

## 4. 异常转换保留原始调用栈（强制）

对异常做类型转换时，**必须使用 `raise ... from e`** 保留原始异常的调用栈信息，禁止直接 `raise` 新异常导致原始堆栈丢失。

```python
# ❌ 丢失原始异常调用栈
try:
    line = int(parts[1])
except ValueError:
    raise ValueError(f"Invalid line number: {parts[1]}")

# ✅ 保留原始异常调用栈
try:
    line = int(parts[1])
except ValueError as e:
    raise ValueError(f"Invalid line number: {parts[1]}") from e
```

## 5. Python 函数参数封装（建议）

当函数参数个数较多（≥4 个）且参数之间存在逻辑相关性时，建议通过具名形式封装：

```python
# ❌ 参数过多，调用时容易传错顺序
def post_comment(token, repo, pr_number, file_path, line, body):
    ...

# ✅ 使用 dataclass 封装相关参数
from dataclasses import dataclass

@dataclass
class CommentRequest:
    token: str
    repo: str
    pr_number: int
    file_path: str
    line: int
    body: str

def post_comment(req: CommentRequest):
    ...
```

**适用场景**：
- 参数 ≥4 个
- 参数之间存在逻辑分组（如 API 认证参数、文件定位参数）
- 同一组参数在多个函数间传递

**可选封装方式**：`dataclass`（推荐）、`namedtuple`、普通 `class`
