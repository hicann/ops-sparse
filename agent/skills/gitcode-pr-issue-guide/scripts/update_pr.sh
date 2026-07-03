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
# update_pr.sh — 更新已有 PR 的描述
# 用法: bash update_pr.sh <token> <repo> <pr_number> <body_file>

set -e

TOKEN="$1"
REPO="$2"
PR_NUMBER="$3"
BODY_FILE="$4"

if [ -z "$TOKEN" ] || [ -z "$REPO" ] || [ -z "$PR_NUMBER" ] || [ -z "$BODY_FILE" ]; then
    echo "用法: bash update_pr.sh <token> <repo> <pr_number> <body_file>"
    echo "  token      - GitCode access_token"
    echo "  repo       - 仓库路径，如 cann/ops-sparse"
    echo "  pr_number  - PR 编号"
    echo "  body_file  - PR 正文的 md 文件路径"
    exit 1
fi

if [ ! -f "$BODY_FILE" ]; then
    echo "错误: 文件不存在: $BODY_FILE"
    exit 1
fi

BODY=$(cat "$BODY_FILE")

echo "=========================================="
echo "更新 PR 描述"
echo "=========================================="
echo "仓库: $REPO"
echo "PR 编号: $PR_NUMBER"
echo "正文文件: $BODY_FILE"
echo "=========================================="

echo ""
echo "正在更新 PR..."

RESPONSE=$(curl -s -w "\n%{http_code}" -X PATCH \
    "https://gitcode.com/api/v5/repos/${REPO}/pulls/${PR_NUMBER}?access_token=${TOKEN}" \
    -H "Content-Type: application/json" \
    -d "$(jq -n --arg body "$BODY" '{body: $body}')")

HTTP_CODE=$(echo "$RESPONSE" | tail -1)
BODY_RESP=$(echo "$RESPONSE" | sed '$d')

if [ "$HTTP_CODE" -ge 200 ] && [ "$HTTP_CODE" -lt 300 ]; then
    echo "✓ PR 更新成功"
    echo "  https://gitcode.com/${REPO}/merge_requests/${PR_NUMBER}"
else
    echo "✗ PR 更新失败 (HTTP $HTTP_CODE)"
    echo "$BODY_RESP"
    exit 1
fi
