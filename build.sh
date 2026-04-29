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

set -e

BUILD_DIR=build
BUILD_OPS=""
RUN_TEST=OFF
ENABLE_PACKAGE=FALSE

export BASE_PATH=$(
  cd "$(dirname $0)"
  pwd
)
export BUILD_PATH="${BASE_PATH}/build"
export BUILD_OUT_PATH="${BASE_PATH}/build_out"
CANN_3RD_LIB_PATH="${BUILD_PATH}/third_party"

ARCH_INFO=$(uname -m)

# 分隔线和错误打印函数（参考 ops-blas build.sh）
dotted_line="----------------------------------------------------------------"
print_error() {
  echo
  echo $dotted_line
  local msg="$1"
  echo -e "\033[31m[ERROR] ${msg}\033[0m"
  echo $dotted_line
  echo
}

# 支持的 SOC 版本
# 按字符串长度从长到短排序，避免前缀匹配时出错
SUPPORT_COMPUTE_UNIT_SHORT=("ascend910_93" "ascend910b" "ascend950" "ascend310p")
SUPPORT_COMPUTE_UNIT_SHORT=($(printf '%s\n' "${SUPPORT_COMPUTE_UNIT_SHORT[@]}" | awk '{print length($0) " " $0}' | sort -rn | cut -d ' ' -f2-))

# ==========================
# 环境检查（Ascend/CANN）
# ==========================
ASCEND_HOME="${ASCEND_HOME_PATH:-${ASCEND_TOOLKIT_HOME}}"
if [ -z "${ASCEND_HOME}" ]; then
    echo "Error: Ascend/CANN environment is not configured."
    echo "Please source the Ascend environment script first, e.g.:"
    echo "  source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh"
    echo "  or source \${HOME}/Ascend/ascend-toolkit/latest/set_env.sh"
    exit 1
fi
# 确保后续使用的 ASCEND_HOME_PATH 有值
export ASCEND_HOME_PATH="${ASCEND_HOME_PATH:-${ASCEND_HOME}}"
# test 使用变量
export LINUX_INCLUDE_PATH="${ASCEND_HOME_PATH}/${ARCH_INFO}-linux/include"
export EAGER_LIBRARY_PATH="${ASCEND_HOME_PATH}/lib64"

# ==========================
# 解析参数
# ==========================
for arg in "$@"; do
    case $arg in
        --ops=*)
            BUILD_OPS="${arg#*=}"
            ;;
        --run)
            RUN_TEST=ON
            ;;
        --soc=*)
            SOC_VERSION="${arg#*=}"
            ;;
        --pkg)
            ENABLE_PACKAGE=TRUE
            ;;
        *)
            echo "Unknown option: $arg"
            echo "Usage:"
            echo "  bash build.sh                                # 只编译库"
            echo "  bash build.sh --ops=spmv                       # 编译指定算子"
            echo "  bash build.sh --ops=spmv,spmm                  # 编译多个算子（逗号分隔）"
            echo "  bash build.sh --run                            # 编译并运行全部 test"
            echo "  bash build.sh --ops=spmv --run                 # 编译并运行指定算子测试"
            echo "  bash build.sh --pkg                            # 编译并打包 run 包"
            echo "  bash build.sh --pkg --soc=ascend950            # 打包指定 SOC 的 run 包"
            exit 1
            ;;
    esac
done

# 校验 --run 和 --pkg 不能同时使用
if [ "${RUN_TEST}" == "ON" ] && [ "${ENABLE_PACKAGE}" == "TRUE" ]; then
  print_error "--run cannot be used with --pkg"
  exit 1
fi

# --run 且未指定 --ops：默认包含 test 下所有含 CMakeLists.txt 的算子子目录
if [ "${RUN_TEST}" == "ON" ] && [ -z "${BUILD_OPS}" ]; then
    TEST_OP_LIST=()
    shopt -s nullglob
    for d in "${BASE_PATH}/test"/*/; do
        [ -d "$d" ] || continue
        name=$(basename "$d")
        if [ -f "${d}/CMakeLists.txt" ]; then
            TEST_OP_LIST+=("$name")
        fi
    done
    shopt -u nullglob
    if [ ${#TEST_OP_LIST[@]} -eq 0 ]; then
        print_error "No tests found under test/ (expected test/<op>/CMakeLists.txt)"
        exit 1
    fi
    BUILD_OPS=$(printf '%s\n' "${TEST_OP_LIST[@]}" | sort | paste -sd, -)
fi

echo "BUILD_OPS=${BUILD_OPS}, RUN_TEST=${RUN_TEST}, ENABLE_PACKAGE=${ENABLE_PACKAGE}"

# ==========================
# 构建
# ==========================
rm -rf ${BUILD_DIR}
mkdir -p ${BUILD_DIR}

# 默认 SOC_VERSION，ASCEND_CANN_PACKAGE_PATH 使用环境变量
if [ -z "${SOC_VERSION}" ]; then
    SOC_VERSION="ascend910b3"
fi

# 校验 SOC 是否在支持列表中（前缀匹配）
soc_lower=$(echo "${SOC_VERSION}" | tr '[:upper:]' '[:lower:]')
matched=""
for support_unit in "${SUPPORT_COMPUTE_UNIT_SHORT[@]}"; do
    if [[ "${soc_lower}" == "${support_unit}"* ]]; then
        matched="${support_unit}"
        break
    fi
done
if [ -z "${matched}" ]; then
    echo "Error: The soc [${SOC_VERSION}] is not supported."
    echo "Supported SOC: ${SUPPORT_COMPUTE_UNIT_SHORT[*]}"
    exit 1
fi

CMAKE_OPTIONS="-DSOC_VERSION=${SOC_VERSION} -DASCEND_CANN_PACKAGE_PATH=${ASCEND_HOME}"

# 帮助 find_package(ASC) 定位 ASCConfig.cmake（CMake 查找 ASC/ASCConfig.cmake 或 ASC/asc-config.cmake）
if [ -z "${ASC_DIR}" ]; then
    for p in "${ASCEND_HOME}/lib64/cmake" "${ASCEND_HOME}/compiler/latest/lib64/cmake" \
             "${ASCEND_HOME}/ascendc/lib64/cmake"; do
        if [ -f "${p}/ASC/ASCConfig.cmake" ] || [ -f "${p}/ASC/asc-config.cmake" ]; then
            CMAKE_OPTIONS="${CMAKE_OPTIONS} -DCMAKE_PREFIX_PATH=${p}"
            break
        fi
    done
fi
[ -n "${ASC_DIR}" ] && CMAKE_OPTIONS="${CMAKE_OPTIONS} -DASC_DIR=${ASC_DIR}"

if [ -n "${BUILD_OPS}" ]; then
    # 将逗号分隔转换为 CMake 列表格式（分号分隔）
    TEST_NAMES_CMAKE="${BUILD_OPS//,/;}"
    CMAKE_OPTIONS="${CMAKE_OPTIONS} -DBUILD_TEST=ON -DTEST_NAMES=${TEST_NAMES_CMAKE}"
fi

if [ "${ENABLE_PACKAGE}" == "TRUE" ]; then
    CMAKE_OPTIONS="${CMAKE_OPTIONS} -DENABLE_PACKAGE=ON -DCANN_3RD_LIB_PATH=${CANN_3RD_LIB_PATH}"
fi

cmake -B ${BUILD_DIR} ${CMAKE_OPTIONS}
cmake --build ${BUILD_DIR} -j
if [ "${ENABLE_PACKAGE}" == "TRUE" ]; then
    cmake --build ${BUILD_DIR} --target package
else
    cmake --install ${BUILD_DIR}
fi


# ==========================
# 运行算子测试
# ==========================
if [ "${RUN_TEST}" == "ON" ]; then
    # 将逗号分隔的算子名转换为数组
    IFS=',' read -ra OP_ARRAY <<< "${BUILD_OPS}"

    FAILED_OPS=()
    PASSED_OPS=()

    for op in "${OP_ARRAY[@]}"; do
        TEST_BIN="${BUILD_DIR}/test/${op}/${op}_test"
        echo ""
        echo "========== Running ${op}_test =========="
        if [ ! -f "${TEST_BIN}" ]; then
            echo "[ERROR] Test binary not found: ${TEST_BIN}"
            FAILED_OPS+=("${op}")
            continue
        fi

        # 临时禁用 errexit 以捕获测试退出码
        set +e
        LD_LIBRARY_PATH="${BUILD_OUT_PATH}/lib64:${LD_LIBRARY_PATH}" "${TEST_BIN}"
        exit_code=$?
        set -e
        if [ $exit_code -eq 0 ]; then
            echo "[PASS] ${op}_test"
            PASSED_OPS+=("${op}")
        else
            echo "[FAIL] ${op}_test (exit code: $exit_code)"
            FAILED_OPS+=("${op}")
        fi
    done

    # 汇总测试结果
    echo ""
    echo "========================================"
    echo "Test Summary:"
    echo "  Passed: ${#PASSED_OPS[@]} - ${PASSED_OPS[*]}"
    echo "  Failed: ${#FAILED_OPS[@]} - ${FAILED_OPS[*]}"
    echo "========================================"

    if [ ${#FAILED_OPS[@]} -gt 0 ]; then
        exit 1
    fi
fi
