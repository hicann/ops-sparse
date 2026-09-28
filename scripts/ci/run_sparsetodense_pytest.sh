#!/usr/bin/env bash
# SparseToDense torch_extension pytest helper.
# Prerequisite: source Ascend set_env.sh (see test/sparse2dense/python/README.md).
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
WORK="${WORK:-${REPO_ROOT}}"
RUN_DIR="${1:?RUN_DIR required}"
mkdir -p "${RUN_DIR}"
LOG="${RUN_DIR}/pytest_sparsetodense.log"

PYTEST_RC=127
: > "${LOG}"

TEST_FILE="${WORK}/test/sparse2dense/python/test_sparse2dense_torch_extension.py"
if [ ! -f "${TEST_FILE}" ]; then
  echo "SKIP pytest: missing ${TEST_FILE}" | tee -a "${LOG}"
  echo "PYTEST_RC=${PYTEST_RC}" > "${RUN_DIR}/pytest_exit_code.txt"
  exit 0
fi

export LD_LIBRARY_PATH="/usr/local/Ascend/cann/lib64:${WORK}/build_out/lib64:${WORK}/build:${LD_LIBRARY_PATH:-}"
export OPS_SPARSE_LIB_DIR="${OPS_SPARSE_LIB_DIR:-${WORK}/build_out/lib64}"
mkdir -p "${WORK}/build_out/lib64" "${WORK}/.torch_extensions"
if [ ! -f "${OPS_SPARSE_LIB_DIR}/libops_sparse.so" ] && [ -f "${WORK}/build/libops_sparse.so" ]; then
  export OPS_SPARSE_LIB_DIR="${WORK}/build"
fi
if [ ! -f "${OPS_SPARSE_LIB_DIR}/libops_sparse.so" ] && [ -f "${WORK}/build/lib64/libops_sparse.so" ]; then
  export OPS_SPARSE_LIB_DIR="${WORK}/build/lib64"
fi
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-${WORK}/.torch_extensions}"
mkdir -p "${TORCH_EXTENSIONS_DIR}"

PYTEST_RC=0
if python3 -c "import torch; import torch_npu; assert torch.npu.device_count()>=1" 2>/dev/null; then
  if ! python3 -c "import cann_ops_sparse" 2>/dev/null; then
    echo "INFO: cann_ops_sparse not installed; building torch_extension wheel" | tee -a "${LOG}"
    (cd "${WORK}" && python3 -m pip install -r torch_extension/requirements.txt -q \
      && bash build.sh --torch_extension --ops=sparse2dense \
      && python3 -m pip install --force-reinstall --no-deps build_out/cann_ops_sparse-*.whl) \
      2>&1 | tee -a "${LOG}" || true
  fi
  (cd "${WORK}" && python3 -m pytest "${TEST_FILE}" -v) \
    2>&1 | tee -a "${LOG}" || PYTEST_RC=${PIPESTATUS[0]}
else
  echo "SKIP pytest: NPU unavailable" | tee -a "${LOG}"
fi

echo "PYTEST_RC=${PYTEST_RC}" | tee "${RUN_DIR}/pytest_exit_code.txt" | tee -a "${LOG}"
exit 0
