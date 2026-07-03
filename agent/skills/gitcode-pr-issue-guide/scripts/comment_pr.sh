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
# comment_pr.sh - 在 PR 评论区添加评论
# 用法: ./comment_pr.sh <token> <repo> <pr_number> <comment>
# 示例: ./comment_pr.sh "your_token" "cann/ops-sparse" "97" "compile"

set -euo pipefail

TOKEN="${1:?用法: $0 <token> <repo> <pr_number> <comment>}"
REPO="${2:?用法: $0 <token> <repo> <pr_number> <comment>}"
PR_NUMBER="${3:?用法: $0 <token> <repo> <pr_number> <comment>}"
COMMENT="${4:?用法: $0 <token> <repo> <pr_number> <comment>}"
API="https://gitcode.com/api/v5/repos/${REPO}/pulls/${PR_NUMBER}/comments"

echo "=========================================="
echo "添加 PR 评论"
echo "=========================================="
echo "仓库: ${REPO}"
echo "PR 编号: ${PR_NUMBER}"
echo "评论内容: ${COMMENT}"
echo "=========================================="
echo ""

# 用 python 构造 JSON
json=$(python3 -c "
import json, sys
print(json.dumps({'body': sys.argv[1]}))
" "$COMMENT")

echo "正在添加评论..."
resp=$(curl -s -X POST "$API" \
  -H "Content-Type: application/json" \
  -H "private-token: ${TOKEN}" \
  -d "$json")

comment_id=$(echo "$resp" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    print(d.get('id', 'ERROR: ' + str(d)))
except:
    print('ERROR: 无法解析响应')
" 2>/dev/null || echo "ERROR: 请求失败")

if [[ "$comment_id" != ERROR* ]]; then
  echo "✓ 评论添加成功"
  echo "  评论 ID: ${comment_id}"
  exit 0
else
  echo "✗ 评论添加失败"
  echo "  ${comment_id}"
  exit 1
fi
