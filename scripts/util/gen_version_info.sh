#!/bin/bash
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
# 参考 ops-blas scripts/util/gen_version_info.sh 实现
# 生成 version.info 文件，供 run 包使用

ascend_install_dir=$1
gen_file_dir=$2
run_pkg_version=$3

mkdir -p "${gen_file_dir}"

# create version.info
# Version: run 包版本
echo "Version=${run_pkg_version}" > "${gen_file_dir}/version.info"

# custom_opp_compiler_version: 从 Ascend compiler 读取
if [ -f "${ascend_install_dir}/compiler/version.info" ]; then
  compiler_version=$(grep "Version" -w "${ascend_install_dir}/compiler/version.info" | awk -F = '{print $2}')
  echo "custom_opp_compiler_version=${compiler_version}" >> "${gen_file_dir}/version.info"
fi
