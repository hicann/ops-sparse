#!/bin/bash
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
# batch_create_issues.sh - 批量创建 GitCode Issue
# 用法: ./batch_create_issues.sh <token> <repo> <issue_dir> [file_pattern] [assignee]
# 示例: ./batch_create_issues.sh "your_token" "cann/ops-sparse" "/path/to/issues" "ISSUE-bug-*.md" "xutianze"

set -euo pipefail

TOKEN="${1:?用法: $0 <token> <repo> <issue_dir> [file_pattern] [assignee]}"
REPO="${2:?用法: $0 <token> <repo> <issue_dir> [file_pattern] [assignee]}"
ISSUE_DIR="${3:?用法: $0 <token> <repo> <issue_dir> [file_pattern] [assignee]}"
PATTERN="${4:-ISSUE-bug-*.md}"
ASSIGNEE="${5:-}"
API="https://gitcode.com/api/v5/repos/${REPO}/issues"

echo "=========================================="
echo "批量创建 GitCode Issues"
echo "=========================================="
echo "仓库: ${REPO}"
echo "目录: ${ISSUE_DIR}"
echo "模式: ${PATTERN}"
if [[ -n "$ASSIGNEE" ]]; then
  echo "指派人: ${ASSIGNEE}"
fi
echo "=========================================="
echo ""

success_count=0
fail_count=0
declare -a results=()

for f in "${ISSUE_DIR}"/${PATTERN}; do
  [[ -f "$f" ]] || continue
  filename=$(basename "$f")

  # 第一行是标题，去掉开头的 "# "
  title=$(head -1 "$f" | sed 's/^# //')
  # 正文是去掉第一行后的内容
  body=$(tail -n +3 "$f")

  # 用 python 构造 JSON（正确处理特殊字符和换行）
  json=$(python3 -c "
import json, sys
title = sys.argv[1]
body = sys.stdin.read()
assignee = sys.argv[3] if len(sys.argv) > 3 and sys.argv[3] else None
payload = {'title': title, 'body': body, 'access_token': sys.argv[2]}
if assignee:
    payload['assignee'] = assignee
print(json.dumps(payload))
" "$title" "$TOKEN" "$ASSIGNEE" <<< "$body")

  echo "--- 创建: ${title} ---"
  resp=$(curl -s -X POST "$API" \
    -H "Content-Type: application/json" \
    -d "$json")

  url=$(echo "$resp" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    print(d.get('web_url') or d.get('html_url') or 'ERROR: ' + str(d))
except:
    print('ERROR: 无法解析响应')
" 2>/dev/null || echo "ERROR: 请求失败")

  if [[ "$url" == http* ]]; then
    echo "  ✓ ${url}"
    results+=("${filename}|${url}")
    ((success_count++))
  else
    echo "  ✗ ${url}"
    results+=("${filename}|FAILED: ${url}")
    ((fail_count++))
  fi
  echo ""
  sleep 1
done

echo "=========================================="
echo "提交完成"
echo "=========================================="
echo "成功: ${success_count}"
echo "失败: ${fail_count}"
echo ""

if [[ ${#results[@]} -gt 0 ]]; then
  echo "详细结果:"
  for r in "${results[@]}"; do
    IFS='|' read -r fname furl <<< "$r"
    echo "  ${fname} -> ${furl}"
  done
fi
