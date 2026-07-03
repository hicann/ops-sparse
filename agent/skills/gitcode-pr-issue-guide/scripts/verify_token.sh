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
# verify_token.sh - 验证 GitCode access token 是否有效
# 用法: ./verify_token.sh <token> <repo>
# 示例: ./verify_token.sh "your_token" "cann/ops-sparse"

set -euo pipefail

TOKEN="${1:?用法: $0 <token> <repo>}"
REPO="${2:?用法: $0 <token> <repo>}"
API="https://gitcode.com/api/v5/repos/${REPO}"

echo "正在验证 token..."
resp=$(curl -s -w "\n%{http_code}" -X GET "${API}/issues?state=open&per_page=1&access_token=${TOKEN}")

http_code=$(echo "$resp" | tail -1)
body=$(echo "$resp" | sed '$d')

if [[ "$http_code" == "200" ]]; then
  echo "✓ Token 有效"
  echo "  仓库: ${REPO}"
  exit 0
else
  echo "✗ Token 无效或无权限 (HTTP ${http_code})"
  echo "  响应: ${body}"
  exit 1
fi
