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
# comment_pr_inline.sh - 在 PR 的代码行上添加行内评论 (diff_comment)
#
# 用法:
#   ./comment_pr_inline.sh <token> <repo> <pr_number> <file> <line> <comment>
#
# 参数说明:
#   token      - GitCode access token
#   repo       - 仓库路径，如 cann/ops-sparse
#   pr_number  - PR 编号
#   file       - 文件相对路径（相对于仓库根目录）
#   line       - 文件中的行号（必须是 PR 实际修改的行，即 diff 中的 + 行）
#   comment    - 评论内容（支持 Markdown）
#
# 示例:
#   ./comment_pr_inline.sh "your_token" "cann/ops-sparse" "120" \
#       "sparse/csrmv/csrmv/arch22/csrmv_host.cpp" "113" \
#       "**[HIGH] SEC-1.2**: handle 未做空指针校验"
#
# 注意事项:
#   - line 必须是 PR 实际修改的行（diff 中的 + 行），否则评论会创建失败或定位错误
#   - 如需批量添加多条评论，建议使用 comment_pr_inline_batch.py
#   - position 参数是 diff 相对行号，脚本内部会自动计算（从文件行号转换）

set -euo pipefail

TOKEN="${1:?用法: $0 <token> <repo> <pr_number> <file> <line> <comment>}"
REPO="${2:?用法: $0 <token> <repo> <pr_number> <file> <line> <comment>}"
PR_NUMBER="${3:?用法: $0 <token> <repo> <pr_number> <file> <line> <comment>}"
FILE="${4:?用法: $0 <token> <repo> <pr_number> <file> <line> <comment>}"
LINE="${5:?用法: $0 <token> <repo> <pr_number> <file> <line> <comment>}"
COMMENT="${6:?用法: $0 <token> <repo> <pr_number> <file> <line> <comment>}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=========================================="
echo "添加 PR 行内评论"
echo "=========================================="
echo "仓库: ${REPO}"
echo "PR 编号: ${PR_NUMBER}"
echo "文件: ${FILE}"
echo "行号: ${LINE}"
echo "=========================================="
echo ""

exec python3 "${SCRIPT_DIR}/comment_pr_inline_batch.py" \
    --token "$TOKEN" \
    --repo "$REPO" \
    --pr "$PR_NUMBER" \
    --comment "$FILE:$LINE:$COMMENT"
