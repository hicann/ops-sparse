#!/usr/bin/env bash
# SparseToDense torch_extension: build lib + wheel, then pytest.
# Prerequisite: source Ascend set_env.sh first (see test/sparse2dense/python/README.md).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
WORK="${WORK:-${REPO_ROOT}}"
SOC="${SOC:-ascend950}"
RUN_DIR="${RUN_DIR:-${WORK}/out/sparsetodense_torch_ext_$(date +%Y%m%d_%H%M%S)}"
export SOC

mkdir -p \
  "${WORK}" \
  "${RUN_DIR}" \
  "${WORK}/build_out" \
  "${WORK}/build_out/lib64" \
  "${WORK}/.torch_extensions"

LOG="${RUN_DIR}/console.log"
exec > >(tee -a "${LOG}") 2>&1

echo "WORK=${WORK}"
echo "SOC=${SOC}"
echo "RUN_DIR=${RUN_DIR}"

cd "${WORK}"

if ! command -v bicc >/dev/null 2>&1 && [ -z "${ASCEND_HOME_PATH:-}" ]; then
  echo "ERROR: Ascend env not configured. Source set_env.sh first:"
  echo "  source /usr/local/Ascend/cann/set_env.sh"
  echo "See test/sparse2dense/python/README.md"
  exit 1
fi

echo "==> build libops_sparse (sparse2dense)"
CMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE:-Release}" bash build.sh --ops=sparse2dense --soc="${SOC}"

echo "==> build/install cann_ops_sparse wheel"
python3 -m pip install -r torch_extension/requirements.txt
bash build.sh --torch_extension --ops=sparse2dense
mkdir -p "${WORK}/build_out"
python3 -m pip install --force-reinstall --no-deps build_out/cann_ops_sparse-*.whl

if [ -f "${WORK}/build_out/lib64/libops_sparse.so" ]; then
  export OPS_SPARSE_LIB_DIR="${WORK}/build_out/lib64"
elif [ -f "${WORK}/build/lib64/libops_sparse.so" ]; then
  export OPS_SPARSE_LIB_DIR="${WORK}/build/lib64"
elif [ -f "${WORK}/build/libops_sparse.so" ]; then
  export OPS_SPARSE_LIB_DIR="${WORK}/build"
else
  echo "ERROR: libops_sparse.so not found under build_out/lib64 or build/"
  find "${WORK}/build_out" "${WORK}/build" -name 'libops_sparse.so' 2>/dev/null || true
  exit 1
fi

export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-${WORK}/.torch_extensions}"
mkdir -p "${TORCH_EXTENSIONS_DIR}" "${OPS_SPARSE_LIB_DIR}"
export LD_LIBRARY_PATH="${OPS_SPARSE_LIB_DIR}:${ASCEND_HOME_PATH:-/usr/local/Ascend/ascend-toolkit/latest}/lib64:${LD_LIBRARY_PATH:-}"

echo "OPS_SPARSE_LIB_DIR=${OPS_SPARSE_LIB_DIR}"
echo "TORCH_EXTENSIONS_DIR=${TORCH_EXTENSIONS_DIR}"

TEST_FILE="${WORK}/test/sparse2dense/python/test_sparse2dense_torch_extension.py"
if [ ! -f "${TEST_FILE}" ]; then
  echo "ERROR: missing ${TEST_FILE}"
  exit 1
fi

echo "==> pytest"
python3 -m pytest -q "${TEST_FILE}" -v
echo "PYTEST_RC=$?" | tee "${RUN_DIR}/pytest_exit_code.txt"
echo "DONE. logs: ${LOG}"
