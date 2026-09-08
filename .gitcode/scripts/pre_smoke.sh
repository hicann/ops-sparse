#!/bin/bash
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

echo "start run test case, please wait ..."

log() {
    local dt
    dt=$(date '+%Y%m%d.%H%M%S')
    echo "===================================================================="
    echo "$dt : $*"
    echo "===================================================================="
}

LOG_HEAD() {
    local msg=${1}
    local ts
    ts=$(date +%Y%m%d-%H%M%S)
    echo "[INFO] ${ts} ${msg}"
}

LOG_INFO() {
    local msg=${1}
    local ts
    ts=$(date +%Y%m%d-%H%M%S)
    echo "[INFO] ${ts} ${msg}"
}

LOG_ERROR() {
    local msg=${1}
    local ts
    ts=$(date +%Y%m%d-%H%M%S)
    echo "[ERROR] ${ts} ${msg}"
}

CHECK_ENV_VAR() {
    local var_name=${1}
    local var_value=${!var_name}
    if [[ -z "${var_value}" ]]; then
        LOG_ERROR "Environment variable ${var_name} is not set"
        exit 1
    fi
}

CHECK_ENV_VAR REPOSITORY_NAME
CHECK_ENV_VAR pr_id
CHECK_ENV_VAR WORKSPACE
CHECK_ENV_VAR smoke_run_file_url
CHECK_ENV_VAR obs_smoke_path

LOG_HEAD "REPOSITORY_NAME: ${REPOSITORY_NAME}"
LOG_HEAD "pr_id: ${pr_id}"
LOG_HEAD "smoke_run_file_url: ${smoke_run_file_url}"
LOG_HEAD "WORKSPACE: ${WORKSPACE}"

cd "${WORKSPACE}"

export ASCEND_GLOBAL_LOG_LEVEL=2
export ASCEND_SLOG_PRINT_TO_STDOUT=0
set -o pipefail

log "init test case, please wait ..."
rm -rf /root/ascend/log



# ==============================
# 下载并安装 smoke run 包
# ==============================
log "start downloading packages and running test case ..."

arm_package=$(basename "${smoke_run_file_url}")
echo "Starting to download file: ${arm_package}"
wget -nv --no-clobber "${smoke_run_file_url}"

if [ ! -f "${arm_package}" ]; then
    echo "File ${arm_package} does not exist, no need to execute smoke test task"
    exit 0
fi

FILE_SIZE=$(stat -c%s "${arm_package}" 2>/dev/null || echo 0)
if [ "${FILE_SIZE}" -lt 15360 ]; then
    echo "No compiled operators, no need to execute smoke test task"
    rm -f "${arm_package}"
    touch slog.tar.gz
    exit 0
fi
echo "File download completed, size ${FILE_SIZE}, starting installation."

set +e
yes "y" | bash "${arm_package}" --install --install-path=/usr/local/Ascend --quiet
set -e

if [ -f "/usr/local/Ascend/cann/bin/setenv.bash" ]; then
    source /usr/local/Ascend/cann/bin/setenv.bash
elif [ -f "/usr/local/Ascend/cann/set_env.sh" ]; then
    source /usr/local/Ascend/cann/set_env.sh
fi

# ==============================
# 确定 SOC_VERSION 和 ARCH_DIR
# ==============================
if echo "${arm_package}" | grep -qi "A5"; then
    SOC_VERSION="ascend950"
    ARCH_DIR="arch35"
else
    SOC_VERSION="ascend910b"
    ARCH_DIR="arch22"
fi
LOG_HEAD "Detected SOC_VERSION=${SOC_VERSION}, ARCH_DIR=${ARCH_DIR} from package: ${arm_package}"

# ==============================
# 运行测试主循环
# ==============================
declare -a PASSED_OPS=() SKIPPED_OPS=() FAILED_OPS=()

for test_dir in test/*/; do
    op=$(basename "$test_dir")

    src_path="src/${op}/${ARCH_DIR}"
    test_file="test/${op}/${ARCH_DIR}/${op}_test.cpp"

    if [[ -d "${src_path}" && -f "${test_file}" ]]; then
        echo ">> [RUN] ${op}"
        if bash build.sh --ops="${op}" --run --soc="${SOC_VERSION}" 2>&1 | tee -a ${WORKSPACE}/run_test.log; then
            PASSED_OPS+=("${op}")
        else
            FAILED_OPS+=("${op}")
        fi
    else
        echo ">> [SKIP] ${op} (Missing ${ARCH_DIR} impl)"
        SKIPPED_OPS+=("${op}")
    fi
done

# ==============================
# 打包 slog 日志
# ==============================
mkdir -p /root/ascend/log
slog_name="slog.tar.gz"
tar -zcf "${slog_name}" -C /root/ascend log

rm -rf /root/ascend/log/*

# ==============================
# 检查 NPU 状态
# ==============================
log "checking NPU status ..."
mkdir -p ./npu_log
npu-smi info 2>&1 | tee ./npu_log/npu_info.log
if grep "dcmi module initialize failed" "./npu_log/npu_info.log"; then
    LOG_ERROR "dcmi module initialize failed"
    exit 1
fi

# ==============================
# 检查测试结果
# ==============================
log "checking test results ..."
date_time=$(date +%Y%m%d).$(date +%H%M%S)

echo "========================================"
echo "Test Summary:"
echo "  Passed:  ${#PASSED_OPS[@]} - ${PASSED_OPS[*]}"
echo "  Skipped: ${#SKIPPED_OPS[@]} - ${SKIPPED_OPS[*]}"
echo "  Failed:  ${#FAILED_OPS[@]} - ${FAILED_OPS[*]}"
echo "========================================"

if [ ${#FAILED_OPS[@]} -gt 0 ]; then
    LOG_ERROR "${date_time} : run test case failed"
    exit 1
fi

if grep -E '\b(FAIL|errors|fail|failed|error|ERROR:|Error|error:)\b' "${WORKSPACE}/run_test.log" | grep -v "error)"; then
    LOG_ERROR "${date_time} : run test case failed (error pattern detected in run_test.log)"
    exit 1
fi

LOG_INFO "${date_time} : run test case success"
