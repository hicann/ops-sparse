#!/usr/bin/env python3
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""
comment_pr_inline_batch.py - 批量在 PR 的代码行上添加行内评论 (diff_comment)

用法:
  python3 comment_pr_inline_batch.py --token TOKEN --repo REPO --pr PR_NUMBER \
      --comment "file:line:body" [--comment "file:line:body" ...]

  或使用 JSON 文件批量输入:
  python3 comment_pr_inline_batch.py --token TOKEN --repo REPO --pr PR_NUMBER \
      --json comments.json

JSON 文件格式:
  [
    {"file": "path/to/file.cpp", "line": 42, "body": "评论内容"},
    {"file": "path/to/other.cpp", "line": 100, "body": "另一条评论"}
  ]

注意事项:
  - line 必须是 PR 实际修改的行（diff 中的 + 行），否则评论会创建失败
  - position 参数是 GitCode API 的 diff 相对行号，脚本内部自动从文件行号转换
  - 每条评论间隔 1 秒避免限流
"""

import argparse
import json
import logging
import re
import sys
import time
import urllib.request
import urllib.error
from dataclasses import dataclass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class GitCodeConfig:
    token: str
    repo: str
    pr_number: str


@dataclass
class InlineComment:
    file: str
    line: int
    body: str


def fetch_pr_files(config: GitCodeConfig):
    api_url = f"https://gitcode.com/api/v5/repos/{config.repo}/pulls/{config.pr_number}/files"
    url = f"{api_url}?private_token={config.token}&per_page=100"
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode("utf-8"))


def build_position_map(patch_diff):
    lines = patch_diff.split("\n")
    pos = 0
    new_line = 0
    mapping = {}
    for line in lines:
        if line.startswith("@@"):
            m = re.search(r"\+(\d+)", line)
            if m:
                new_line = int(m.group(1)) - 1
            continue
        pos += 1
        if line.startswith("+"):
            new_line += 1
            mapping[new_line] = pos
        elif line.startswith("-"):
            pass
        else:
            new_line += 1
    return mapping


def post_inline_comment(config: GitCodeConfig, comment: InlineComment, position: int):
    api_url = f"https://gitcode.com/api/v5/repos/{config.repo}/pulls/{config.pr_number}/comments"
    url = f"{api_url}?private_token={config.token}"
    payload = json.dumps({
        "body": comment.body,
        "path": comment.file,
        "position": position
    }).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode("utf-8"))


def parse_comment_arg(arg):
    parts = arg.split(":", 2)
    if len(parts) != 3:
        raise ValueError(f"Invalid comment format: {arg}. Expected 'file:line:body'")
    file_path = parts[0]
    try:
        line = int(parts[1])
    except ValueError as e:
        raise ValueError(f"Invalid line number: {parts[1]}") from e
    body = parts[2]
    return InlineComment(file=file_path, line=line, body=body)


def parse_args():
    parser = argparse.ArgumentParser(description="批量添加 PR 行内评论")
    parser.add_argument("--token", required=True, help="GitCode access token")
    parser.add_argument("--repo", required=True, help="仓库路径，如 cann/ops-sparse")
    parser.add_argument("--pr", required=True, help="PR 编号")
    parser.add_argument("--comment", action="append", help="评论，格式: file:line:body")
    parser.add_argument("--json", help="JSON 文件路径，包含批量评论数据")
    return parser.parse_args()


def load_comments(args):
    comments = []
    if args.json:
        with open(args.json, "r") as f:
            raw = json.load(f)
        comments = [InlineComment(file=c["file"], line=c["line"], body=c["body"]) for c in raw]
    if args.comment:
        for c in args.comment:
            comments.append(parse_comment_arg(c))
    return comments


def resolve_position(comment, pmap, index, total):
    position = pmap.get(comment.line)
    if position is not None:
        return position
    available = sorted(pmap.keys())
    if not available:
        logger.warning("[%d/%d] 跳过: %s 无新增行", index, total, comment.file)
        return None
    nearest = min(available, key=lambda x: abs(x - comment.line))
    logger.warning("[%d/%d] %s:%d 不在 diff 中，使用最近行 %d",
                   index, total, comment.file, comment.line, nearest)
    return pmap[nearest]


def post_all_comments(config, comments, pos_maps):
    success = 0
    failed = 0
    for i, comment in enumerate(comments):
        pmap = pos_maps.get(comment.file)
        if pmap is None:
            logger.warning("[%d/%d] 跳过: %s 不在 PR 变更文件中", i + 1, len(comments), comment.file)
            failed += 1
            continue

        position = resolve_position(comment, pmap, i + 1, len(comments))
        if position is None:
            failed += 1
            continue

        try:
            result = post_inline_comment(config, comment, position)
            note_id = result.get("note_id", "?")
            logger.info("[%d/%d] 成功: %s:%d (position=%d, note_id=%s)",
                        i + 1, len(comments), comment.file, comment.line, position, note_id)
            success += 1
        except urllib.error.HTTPError as e:
            error_body = e.read().decode("utf-8") if e.fp else ""
            logger.error("[%d/%d] 失败: %s:%d - HTTP %d: %s",
                         i + 1, len(comments), comment.file, comment.line, e.code, error_body[:200])
            failed += 1
        except Exception as e:
            logger.error("[%d/%d] 失败: %s:%d - %s",
                         i + 1, len(comments), comment.file, comment.line, e)
            failed += 1

        if i < len(comments) - 1:
            time.sleep(1)

    return success, failed


def main():
    args = parse_args()
    config = GitCodeConfig(token=args.token, repo=args.repo, pr_number=args.pr)

    comments = load_comments(args)
    if not comments:
        logger.error("未提供评论。请使用 --comment 或 --json 参数。")
        sys.exit(1)

    logger.info("正在获取 PR #%s 的文件列表...", config.pr_number)
    files = fetch_pr_files(config)

    pos_maps = {}
    for f in files:
        pos_maps[f["filename"]] = build_position_map(f["patch"]["diff"])

    logger.info("共 %d 条评论待发布", len(comments))
    success, failed = post_all_comments(config, comments, pos_maps)
    logger.info("发布完成: 成功 %d, 失败 %d, 总计 %d", success, failed, len(comments))
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
