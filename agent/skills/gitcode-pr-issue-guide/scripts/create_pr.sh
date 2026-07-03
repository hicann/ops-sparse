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
# create_pr.sh - 创建 GitCode Pull Request
# 用法: ./create_pr.sh <token> <repo> <title> <head> <base> <body_file>
# 示例: ./create_pr.sh "your_token" "cann/ops-sparse" "PR标题" "feature-branch" "main" "PR-xxx.md"

set -euo pipefail

TOKEN="${1:?用法: $0 <token> <repo> <title> <head> <base> <body_file>}"
REPO="${2:?用法: $0 <token> <repo> <title> <head> <base> <body_file>}"
TITLE="${3:?用法: $0 <token> <repo> <title> <head> <base> <body_file>}"
HEAD="${4:?用法: $0 <token> <repo> <title> <head> <base> <body_file>}"
BASE="${5:?用法: $0 <token> <repo> <title> <head> <base> <body_file>}"
BODY_FILE="${6:?用法: $0 <token> <repo> <title> <head> <base> <body_file>}"
API="https://gitcode.com/api/v5/repos/${REPO}/pulls"

if [[ ! -f "$BODY_FILE" ]]; then
  echo "错误: 文件不存在 ${BODY_FILE}"
  exit 1
fi

echo "=========================================="
echo "创建 GitCode Pull Request"
echo "=========================================="
echo "仓库: ${REPO}"
echo "标题: ${TITLE}"
echo "分支: ${HEAD} -> ${BASE}"
echo "正文: ${BODY_FILE}"
echo "=========================================="
echo ""

body=$(cat "$BODY_FILE")

# 用 python 构造 JSON
json=$(python3 -c "
import json, sys
body = sys.stdin.read()
print(json.dumps({
    'title': sys.argv[1],
    'head': sys.argv[2],
    'base': sys.argv[3],
    'body': body
}))
" "$TITLE" "$HEAD" "$BASE" <<< "$body")

echo "正在创建 PR..."
resp=$(curl -s -X POST "$API" \
  -H "Content-Type: application/json" \
  -H "private-token: ${TOKEN}" \
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
  echo "✓ PR 创建成功"
  echo "  ${url}"
  exit 0
else
  echo "✗ PR 创建失败"
  echo "  ${url}"
  exit 1
fi
