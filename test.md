# Build Test Results

本文档记录了稀疏算子库的构建测试结果，包括基础构建、功能测试、单算子测试、多算子测试以及不同 SOC 架构的打包验证。

## 目录

- [基础构建测试](#基础构建测试)
- [功能测试（全算子）](#功能测试全算子)
- [单算子测试](#单算子测试)
- [多算子测试](#多算子测试)
- [打包测试](#打包测试)
- [arch35 架构验证（ascend950）](#arch35-架构验证ascend950)

---

## 基础构建测试

### 执行命令

```bash
bash build.sh
```

### 构建输出

```text
BUILD_OPS=, RUN_TEST=OFF, ENABLE_PACKAGE=FALSE
-- The C compiler identification is GNU 9.4.0
-- The CXX compiler identification is GNU 9.4.0
-- Detecting C compiler ABI info
-- Detecting C compiler ABI info - done
-- Check for working C compiler: /usr/bin/cc/cc - skipped
-- Detecting C compile features
-- Detecting C compile features - done
-- Detecting CXX compiler ABI info
-- Detecting CXX compiler ABI info - done
-- Check for working CXX compiler: /usr/bin/c++ - skipped
-- Detecting CXX compile features
-- Detecting CXX compile features - done
-- SOC_VERSION=ascend910b3, NPU_ARCH=dav-2201
-- SOC_ARCH_DIRS=
-- System processer: aarch64
-- CMAKE_ASC_COMPILER: /home/developer/Ascend/cann-9.0.0/bin/bisheng
-- ASCEND_CANN_PACKAGE_LINUX_PATH: /home/developer/Ascend/cann-9.0.0/aarch64-linux
-- CMAKE_ASC_LLD_LINKER: /home/developer/Ascend/cann-9.0.0/aarch64-linux/ccec_compiler/bin/ld.lld
-- Configuring done
-- Generating done
-- Build files have been written to: /mnt/workspace/gitCode/cann/ops-sparse/build
[ 20%] Building ASC object CMakeFiles/ops_sparse.dir/src/spmm/spmm_host.cpp.o
[ 60%] Building ASC object CMakeFiles/ops_sparse.dir/src/spmm/spmm_kernel.cpp.o
[ 60%] Building ASC object CMakeFiles/ops_sparse.dir/src/spmv/spmv_host.cpp.o
[ 80%] Building ASC object CMakeFiles/ops_sparse.dir/src/spmv/spmv_kernel.cpp.o
[100%] Linking ASC shared library libops_sparse.so
[100%] Built target ops_sparse
-- Install configuration: "Debug"
-- Installing: /mnt/workspace/gitCode/cann/ops-sparse/build_out/lib64/libops_sparse.so
-- Installing: /mnt/workspace/gitCode/cann/ops-sparse/build_out/include/cann_ops_sparse.h
```

### 构建摘要

| 项目     | 结果                           |
| -------- | ------------------------------ |
| 状态     | 成功                           |
| 编译器   | bisheng (Ascend C)             |
| 目标架构 | ascend910b3 (dav-2201)         |
| 构建产物 | `libops_sparse.so`             |
|          | `cann_ops_sparse.h`            |
| 安装路径 | `build_out/`                   |

---

## 功能测试（全算子）

### 执行命令

```bash
bash build.sh --run
```

### 构建输出

```text
BUILD_OPS=spmm,spmv, RUN_TEST=ON, ENABLE_PACKAGE=FALSE
-- The C compiler identification is GNU 9.4.0
-- The CXX compiler identification is GNU 9.4.0
-- Detecting C compiler ABI info
-- Detecting C compiler ABI info - done
-- Check for working C compiler: /usr/bin/cc - skipped
-- Detecting C compile features
-- Detecting C compile features - done
-- Detecting CXX compiler ABI info
-- Detecting CXX compiler ABI info - done
-- Check for working CXX compiler: /usr/bin/c++ - skipped
-- Detecting CXX compile features
-- Detecting CXX compile features - done
-- SOC_VERSION=ascend910b3, NPU_ARCH=dav-2201
-- SOC_ARCH_DIRS=
-- System processer: aarch64
-- CMAKE_ASC_COMPILER: /home/developer/Ascend/cann-9.0.0/bin/bisheng
-- ASCEND_CANN_PACKAGE_LINUX_PATH: /home/developer/Ascend/cann-9.0.0/aarch64-linux
-- CMAKE_ASC_LLD_LINKER: /home/developer/Ascend/cann-9.0.0/aarch64-linux/ccec_compiler/bin/ld.lld
-- Configuring done
-- Generating done
-- Build files have been written to: /mnt/workspace/gitCode/cann/ars-sparse/build
[ 22%] Building ASC object CMakeFiles/ops_sparse.dir/src/spmm/spmm_host.cpp.o
[ 22%] Building ASC object CMakeFiles/ops_sparse.dir/src/spmm/spmm_kernel.cpp.o
[ 33%] Building ASC object CMakeFiles/ops_sparse.dir/src/spmv/spmv_host.cpp.o
[ 44%] Building ASC object CMakeFiles/ops_sparse.dir/src/spmv/spmv_kernel.cpp.o
[ 55%] Linking ASC shared library libops_sparse.so
[ 55%] Built target ops_sparse
[ 77%] Building CXX object test/spmv/CMakeFiles/spmv_test.dir/spmv_test.cpp.o
[ 77%] Building CXX object test/spmm/CMakeFiles/spmm_test.dir/spmm_test.cpp.o
[ 88%] Linking CXX executable spmm_test
[100%] Linking CXX executable spmv_test
[100%] Built target spmv_test
[100%] Built target spmm_test
-- Install configuration: "Debug"
-- Installing: /mnt/workspace/gitCode/cann/ops-sparse/build_out/lib64/libops_sparse.so
-- Up-to-date: /mnt/workspace/gitCode/cann/ops-sparse/build_out/include/cann_ops_sparse.h

========== Running spmm_test ==========
========== SpMM Test ==========
SpMM test PASSED
[PASS] spmm_test

========== Running spmv_test ==========
========== SpMV Test ==========
SpMV test PASSED
[PASS] spmv_test

========================================
Test Summary:
  Passed: 2 - spmm spmv
  Failed: 0 -
========================================
```

### 测试摘要

| 项目     | 结果        |
| -------- | ----------- |
| 状态     | 全部通过    |
| 通过测试 | 2 (spmm, spmv) |
| 失败测试 | 0           |

---

## 单算子测试

### 执行命令

```bash
bash build.sh --run --ops=spmv
```

### 构建输出

```text
BUILD_OPS=spmv, RUN_TEST=ON, ENABLE_PACKAGE=FALSE
-- The C compiler identification is GNU 9.4.0
-- The CXX compiler identification is GNU 9.4.0
-- Detecting C compiler ABI info
-- Detecting C compiler ABI info - done
-- Check for working C compiler: /usr/bin/cc - skipped
-- Detecting C compile features
-- Detecting C compile features - done
-- Detecting CXX compiler ABI info
-- Detecting CXX compiler ABI info - done
-- Check for working CXX compiler: /usr/bin/c++ - skipped
-- Detecting CXX compile features
-- Detecting CXX compile features - done
-- SOC_VERSION=ascend910b3, NPU_ARCH=dav-2201
-- SOC_ARCH_DIRS=
-- System processer: aarch64
-- CMAKE_ASC_COMPILER: /home/developer/Ascend/cann-9.0.0/bin/bisheng
-- ASCEND_CANN_PACKAGE_LINUX_PATH: /home/developer/Ascend/cann-9.0.0/aarch64-linux
-- CMAKE_ASC_LLD_LINKER: /home/developer/Ascend/cann-9.0.0/aarch64-linux/ccec_compiler/bin/ld.lld
-- Configuring done
-- Generating done
-- Build files have been written to: /mnt/workspace/gitCode/cann/ops-sparse/build
[ 14%] Building ASC object CMakeFiles/ops_sparse.dir/src/spmm/spmm_host.cpp.o
[ 42%] Building ASC object CMakeFiles/ops_sparse.dir/src/spmv/spmv_host.cpp.o
[ 42%] Building ASC object CMakeFiles/ops_sparse.dir/src/spmm/spmm_kernel.cpp.o
[ 57%] Building ASC object CMakeFiles/ops_sparse.dir/src/spmv/spmv_kernel.cpp.o
[ 71%] Linking ASC shared library libops_sparse.so
[ 71%] Built target ops_sparse
[ 85%] Building CXX object test/spmv/CMakeFiles/spmv_test.dir/spmv_test.cpp.o
[100%] Linking CXX executable spmv_test
[100%] Built target spmv_test
-- Install configuration: "Debug"
-- Installing: /mnt/workspace/gitCode/cann/ops-sparse/build_out/lib64/libops_sparse.so
-- Up-to-date: /mnt/workspace/gitCode/cann/ops-sparse/build_out/include/cann_ops_sparse.h

========== Running spmv_test ==========
========== SpMV Test ==========
SpMV test PASSED
[PASS] spmv_test

========================================
Test Summary:
  Passed: 1 - spmv
  Failed: 0 -
========================================
```

### 测试摘要

| 项目     | 结果      |
| -------- | --------- |
| 状态     | 通过      |
| 通过测试 | 1 (spmv)  |
| 失败测试 | 0         |

---

## 多算子测试

### 执行命令

```bash
bash build.sh --run --ops=spmv,spmm
```

### 构建输出

```text
BUILD_OPS=spmv,spmm, RUN_TEST=ON, ENABLE_PACKAGE=FALSE
-- The C compiler identification is GNU 9.4.0
-- The CXX compiler identification is GNU 9.4.0
-- Detecting C compiler ABI info
-- Detecting C compiler ABI info - done
-- Check for working C compiler: /usr/bin/cc - skipped
-- Detecting C compile features
-- Detecting C compile features - done
-- Detecting CXX compiler ABI info
-- Detecting CXX compiler ABI info - done
-- Check for working CXX compiler: /usr/bin/c++ - skipped
-- Detecting CXX compile features
-- Detecting CXX compile features - done
-- SOC_VERSION=ascend910b3, NPU_ARCH=dav-2201
-- SOC_ARCH_DIRS=
-- System processer: aarch64
-- CMAKE_ASC_COMPILER: /home/developer/Ascend/cann-9.0.0/bin/bisheng
-- ASCEND_CANN_PACKAGE_LINUX_PATH: /home/developer/Ascend/cann-9.0.0/aarch64-linux
-- CMAKE_ASC_LLD_LINKER: /home/developer/Ascend/cann-9.0.0/aarch64-linux/ccec_compiler/bin/ld.lld
-- Configuring done
-- Generating done
-- Build files have been written to: /mnt/workspace/gitCode/cann/ops-sparse/build
[ 11%] Building ASC object CMakeFiles/ops_sparse.dir/src/spmm/spmm_host.cpp.o
[ 22%] Building ASC object CMakeFiles/ops_sparse.dir/src/spmv/spmv_host.cpp.o
[ 33%] Building ASC object CMakeFiles/ops_sparse.dir/src/spmm/spmm_kernel.cpp.o
[ 44%] Building ASC object CMakeFiles/ops_sparse.dir/src/spmv/spmv_kernel.cpp.o
[ 55%] Linking ASC shared library libops_sparse.so
[ 55%] Built target ops_sparse
[ 66%] Building CXX object test/spmv/CMakeFiles/spmv_test.dir/spmv_test.cpp.o
[ 77%] Building CXX object test/spmm/CMakeFiles/spmm_test.dir/spmm_test.cpp.o
[ 88%] Linking CXX executable spmv_test
[100%] Linking CXX executable spmm_test
[100%] Built target spmv_test
[100%] Built target spmm_test
-- Install configuration: "Debug"
-- Installing: /mnt/workspace/gitCode/cann/ops-sparse/build_out/lib64/libops_sparse.so
-- Up-to-date: /mnt/workspace/gitCode/cann/ops-sparse/build_out/include/cann_ops_sparse.h

========== Running spmv_test ==========
========== SpMV Test ==========
SpMV test PASSED
[PASS] spmv_test

========== Running spmm_test ==========
========== SpMM Test ==========
SpMM test PASSED
[PASS] spmm_test

========================================
Test Summary:
  Passed: 2 - spmv spmm
  Failed: 0 -
========================================
```

### 测试摘要

| 项目     | 结果             |
| -------- | ---------------- |
| 状态     | 全部通过         |
| 通过测试 | 2 (spmv, spmm)   |
| 失败测试 | 0                |

---

## 打包测试

### 执行命令

```bash
bash build.sh --pkg
```

### 构建输出

```text
BUILD_OPS=, RUN_TEST=OFF, ENABLE_PACKAGE=TRUE
-- The C compiler identification is GNU 9.4.0
-- The CXX compiler identification is GNU 9.4.0
-- Detecting C compiler ABI info
-- Detecting C compiler ABI info - done
-- Check for working C compiler: /usr/bin/cc - skipped
-- Detecting C compile features
-- Detecting C compile features - done
-- Detecting CXX compiler ABI info
-- Detecting CXX compiler ABI info - done
-- Check for working CXX compiler: /usr/bin/c++ - skipped
-- Detecting CXX compile features
-- Detecting CXX compile features - done
-- SOC_VERSION=ascend910b3, NPU_ARCH=dav-2201
-- SOC_ARCH_DIRS=
-- System processer: aarch64
-- CMAKE_ASC_COMPILER: /home/developer/Ascend/cann-9.0.0/bin/bisheng
-- ASCEND_CANN_PACKAGE_LINUX_PATH: /home/developer/Ascend/cann-9.0.0/aarch64-linux
-- CMAKE_ASC_LLD_LINKER: /home/developer/Ascend/cann-9.0.0/aarch64-linux/ccec_compiler/bin/ld.lld
-- Downloading makeself from https://gitcode.com/cann-src-third-party/makeself/releases/download/release-2.5.0-patch1.0/makeself-release-2.5.0-patch1.tar.gz
-- CMAKE_INSTALL_PREFIX = /mnt/workspace/gitCode/cann/ops-sparse/build_out
-- CMAKE_SOURCE_DIR = /mnt/workspace/gitCode/cann/ops-sparse
-- CMAKE_BINARY_DIR = /mnt/workspace/gitCode/cann/ops-sparse/build
-- CPACK_PACKAGE_FILE_NAME=cann-910b-ops-sparse-1.0.0_linux-aarch64 (SOC=ascend910b3 -> 910b)
-- CMAKE_INSTALL_PREFIX = /mnt/workspace/gitCode/cann/ops-sparse/build_out
-- Configuring done
-- Generating done
-- Build files have been written to: /mnt/workspace/gitCode/cann/ops-sparse/build
[ 16%] Generating version.info for ops_sparse run package
[ 16%] Built target gen_ops_sparse_version_info
[ 33%] Building ASC object CMakeFiles/ops_sparse.dir/src/spmm/spmm_host.cpp.o
[ 66%] Building ASC object CMakeFiles/ops_sparse.dir/src/spmv/spmv_host.cpp.o
[ 66%] Building ASC object CMakeFiles/ops_sparse.dir/src/spmm/spmm_kernel.cpp.o
[ 83%] Building ASC object CMakeFiles/ops_sparse.dir/src/spmv/spmv_kernel.cpp.o
[100%] Linking ASC shared library libops_sparse.so
[100%] Built target ops_sparse
[ 16%] Generating version.info for ops_sparse run package
[ 16%] Built target gen_ops_sparse_version_info
Consolidate compiler generated dependencies of target ops_sparse
[ 33%] Building ASC object CMakeFiles/ops_sparse.dir/src/spmm/spmm_kernel.cpp.o
[ 50%] Building ASC object CMakeFiles/ops_sparse.dir/src/spmv/spmv_kernel.cpp.o
[ 66%] Linking ASC shared library libops_sparse.so
[100%] Built target ops_sparse
Run CPack packaging tool...
CPack: Create package using External
CPack: Install projects
CPack: - Run preinstall target for: ops_sparse
CPack: - Install project: ops_sparse []
CPack: Create package

About to compress 868 KB of data...
Adding files to archive named "/mnt/workspace/gitCode/cann/ops-sparse/build_out/cann-910b-ops-sparse-1.0.0_linux-aarch64.run"...
./include/cann_ops_sparse.h
./lib64/libops_sparse.so
./share/info/ops_sparse/help.info
./share/info/ops_sparse/install.sh
./share/info/ops_sparse/version.info
CRC: 2295896204
MD5: d2ca0c82150d2aed5e2560d90e84b7de

Self-extractable archive "/mnt/workspace/gitCode/cann/ops-sparse/build_out/cann-910b-ops-sparse-1.0.0_linux-aarch64.run" successfully created.
CPack: - package: /mnt/workspace/gitCode/cann/ops-sparse/build_out/cann-910b-ops-sparse-1.0.0_linux-aarch64.json generated.
```

### 打包摘要

| 项目     | 结果                                                  |
| -------- | ----------------------------------------------------- |
| 状态     | 成功                                                  |
| 包类型   | 自解压安装包                                          |
| 包文件   | `cann-910b-ops-sparse-1.0.0_linux-aarch64.run`        |
| 包大小   | 868 KB                                                |
| MD5      | `d2ca0c82150d2aed5e2560d90e84b7de`                    |
| 包含内容 | `include/cann_ops_sparse.h`                           |
|          | `lib64/libops_sparse.so`                              |
|          | `share/info/ops_sparse/help.info`                     |
|          | `share/info/ops_sparse/install.sh`                    |
|          | `share/info/ops_sparse/version.info`                  |

### 包目录结构

#### 文件列表

```text
Target directory: cann-910b-ops-sparse-1.0.0_linux-aarch64
-rw-r--r-- developer/developer 2986 2026-04-03 15:19 ./include/cann_ops_sparse.h
-rw-r--r-- developer/developer 836480 2026-04-03 15:46 ./lib64/libops_sparse.so
-rw-r--r-- developer/developer    582 2026-04-02 19:17 ./share/info/ops_sparse/help.info
-rwxr-xr-x developer/developer   9819 2026-04-02 19:17 ./share/info/ops_sparse/install.sh
-r--r--r-- developer/developer     48 2026-04-03 15:46 ./share/info/ops_sparse/version.info
```

#### 树形结构

```
cann-910b-ops-sparse-1.0.0_linux-aarch64/
├── include/
│   └── cann_ops_sparse.h
├── lib64/
│   └── libops_sparse.so
└── share/
    └── info/
        └── ops_sparse/
            ├── help.info
            ├── install.sh
            └── version.info
```

---

## arch35 架构验证（ascend950）

### 验证目标

验证 ascend950 SOC 打包流程，确认 arch35 目录下的文件能被正确编译和打包。

### 验证内容

1. **SOC 识别**: ascend950 应被正确识别并映射到 `NPU_ARCH=dav-3510`
2. **架构目录匹配**: `SOC_ARCH_DIRS` 应正确设置为 `arch35`
3. **源文件过滤与添加**:
   - arch35 目录下的源文件应从基础源文件中过滤掉
   - 根据 `SOC_ARCH_DIRS` 重新添加 arch35 特定源文件
4. **编译过程**: arch35 源文件应被 ASC 编译器正确编译
5. **打包结果**: 生成的包名应包含 950 标识，且包含所有必要文件

### 预期结果

| 项目     | 预期值                                              |
| -------- | --------------------------------------------------- |
| 包名     | `cann-950-ops-sparse-1.0.0_linux-x86_64.run`        |
| 源文件   | 4个通用源文件 + 2个 arch35 特定源文件 = 6个总源文件 |
| 库文件   | 包含 arch35 优化代码的 `libops_sparse.so`           |

### 执行命令

```bash
bash build.sh --pkg --soc=ascend950
```

### 构建输出

```text
BUILD_OPS=, RUN_TEST=OFF, ENABLE_PACKAGE=TRUE
-- The C compiler identification is GNU 13.3.0
-- The CXX compiler identification is GNU 13.3.0
-- SOC_VERSION=ascend950, NPU_ARCH=dav-3510
--   -> SOC [ascend950] matched arch35 (DAV-3510)
-- SOC_ARCH_DIRS=arch35
-- [src] Total source files before filtering: /home/chenb/code/fork/ops-sparse/src/spmm/arch35/spmm_kernel_arch35.cpp;/home/chenb/code/fork/ops-sparse/src/spmm/spmm_host.cpp;/home/chenb/code/fork/ops-sparse/src/spmm/spmm_kernel.cpp;/home/chenb/code/fork/ops-sparse/src/spmv/arch35/spmv_kernel_arch35.cpp;/home/chenb/code/fork/ops-sparse/src/spmv/spmv_host.cpp;/home/chenb/code/fork/ops-sparse/src/spmv/spmv_kernel.cpp
-- [src] Filtering out arch35 sources: /home/chenb/code/fork/ops-sparse/src/spmm/arch35/spmm_kernel_arch35.cpp;/home/chenb/code/fork/ops-sparse/src/spmv/arch35/spmv_kernel_arch35.cpp
-- [src] SOC_ARCH_DIRS specified, collecting arch-specific sources: arch35
-- [src] Adding arch35 sources: /home/chenb/code/fork/ops-sparse/src/spmm/arch35/spmm_kernel_arch35.cpp;/home/chenb/code/fork/ops-sparse/src/spmv/arch35/spmv_kernel_arch35.cpp
-- [src] Source file summary: 4 generic + 2 arch-specific = 6 total
-- System processer: x86_64
-- CMAKE_ASC_COMPILER: /home/chenb/Ascend/cann-9.0.0/bin/bisheng
-- ASCEND_CANN_PACKAGE_LINUX_PATH: /home/chenb/Ascend/cann-9.0.0/x86_64-linux
-- Downloading makeself from https://gitcode.com/cann-src-third-party/makeself/...
-- CPACK_PACKAGE_FILE_NAME=cann-950-ops-sparse-1.0.0_linux-x86_64 (SOC=ascend950 -> 950)
-- Configuring done (1.6s)
-- Generating done (0.0s)
-- Build files have been written to: /home/chenb/code/fork/ops-sparse/build

[ 12%] Generating version.info for ops_sparse run package
[ 25%] Building ASC object CMakeFiles/ops_sparse.dir/src/spmm/spmm_host.cpp.o
[ 37%] Building ASC object CMakeFiles/ops_sparse.dir/src/spmm/arch35/spmm_kernel_arch35.cpp.o
[ 62%] Building ASC object CMakeFiles/ops_sparse.dir/src/spmm/spmm_kernel.cpp.o
[ 62%] Building ASC object CMakeFiles/ops_sparse.dir/src/spmv/spmv_kernel.cpp.o
[ 75%] Building ASC object CMakeFiles/ops_sparse.dir/src/spmv/spmv_host.cpp.o
[ 87%] Building ASC object CMakeFiles/ops_sparse.dir/src/spmv/arch35/spmv_kernel_arch35.cpp.o
[100%] Linking ASC shared library libops_sparse.so
[100%] Built target ops_sparse

Run CPack packaging tool...
CPack: Create package using External
CPack: Install projects
CPack: - Run preinstall target for: ops_sparse
CPack: - Install project: ops_sparse []
CPack: Create package

About to compress 2668 KB of data...
Adding files to archive named "/home/chenb/code/fork/ops-sparse/build_out/cann-950-ops-sparse-1.0.0_linux-x86_64.run"...
./include/cann_ops_sparse.h
./lib64/libops_sparse.so
./share/info/ops_sparse/help.info
./share/info/ops_sparse/install.sh
./share/info/ops_sparse/version.info
CRC: 3456284861
MD5: 022327516ca35ed107af8d56a3e0ca10

Self-extractable archive "/home/chenb/code/fork/ops-sparse/build_out/cann-950-ops-sparse-1.0.0_linux-x86_64.run" successfully created.
CPack: - package: /home/chenb/code/fork/ops-sparse/build_out/cann-950-ops-sparse-1.0.0_linux-x86_64.json generated.
```

### 验证结果汇总

| 验证项       | 预期结果                                      | 实际结果                                                         | 状态 |
| ------------ | --------------------------------------------- | ---------------------------------------------------------------- | ---- |
| SOC 识别     | ascend950 -> dav-3510                         | `SOC_VERSION=ascend950, NPU_ARCH=dav-3510`                       | 通过 |
| 架构目录匹配 | arch35                                        | `SOC_ARCH_DIRS=arch35`                                           | 通过 |
| 源文件过滤   | 过滤 arch35 文件                              | `Filtering out arch35 sources: spmm_kernel_arch35.cpp, spmv_kernel_arch35.cpp` | 通过 |
| 源文件添加   | 添加 arch35 文件                              | `Adding arch35 sources: spmm_kernel_arch35.cpp, spmv_kernel_arch35.cpp` | 通过 |
| 源文件统计   | 4通用 + 2特定 = 6                             | `Source file summary: 4 generic + 2 arch-specific = 6 total`     | 通过 |
| arch35 编译  | 编译 arch35 源文件                            | `Building ASC object .../spmm/arch35/spmm_kernel_arch35.cpp.o`     | 通过 |
|              |                                               | `Building ASC object .../spmv/arch35/spmv_kernel_arch35.cpp.o`     | 通过 |
| 包名生成     | `cann-950-ops-sparse-*`                       | `cann-950-ops-sparse-1.0.0_linux-x86_64.run`                     | 通过 |
| 包完整性     | 包含所有必要文件                              | `include/`, `lib64/`, `share/info/ops_sparse/`                   | 通过 |

### 构建摘要

| 项目         | 结果                                          |
| ------------ | --------------------------------------------- |
| 状态         | 成功                                          |
| SOC 版本     | ascend950                                     |
| NPU 架构     | dav-3510 (DAV-3510)                           |
| 架构目录     | arch35                                        |
| 源文件统计   | 4个通用源文件 + 2个 arch35 特定源文件 = 6个总源文件 |
| 编译器       | bisheng (Ascend C)                            |
| 系统架构     | x86_64                                        |

### 打包摘要

| 项目     | 结果                                           |
| -------- | ---------------------------------------------- |
| 状态     | 成功                                           |
| 包类型   | 自解压安装包 (makeself)                        |
| 包文件   | `cann-950-ops-sparse-1.0.0_linux-x86_64.run`   |
| 包大小   | 2668 KB (~2.6 MB)                              |
| CRC      | 3456284861                                     |
| MD5      | `022327516ca35ed107af8d56a3e0ca10`             |
| 包路径   | `/home/chenb/code/fork/ops-sparse/build_out/`  |

### 包目录结构

#### 文件列表

```text
Target directory: cann-950-ops-sparse-1.0.0_linux-x86_64
-rw-r--r-- 2986 2026-04-02 20:37 ./include/cann_ops_sparse.h
-rw-r--r-- 2682312 2026-04-03 17:09 ./lib64/libops_sparse.so
-rw-r--r-- 582 2026-04-02 17:34 ./share/info/ops_sparse/help.info
-rwxr-xr-x 9748 2026-04-03 16:02 ./share/info/ops_sparse/install.sh
-r--r--r-- 48 2026-04-03 17:09 ./share/info/ops_sparse/version.info
```

#### 树形结构

```
cann-950-ops-sparse-1.0.0_linux-x86_64/
├── include/
│   └── cann_ops_sparse.h
├── lib64/
│   └── libops_sparse.so (2.6 MB, 包含 arch35 优化代码)
└── share/
    └── info/
        └── ops_sparse/
            ├── help.info
            ├── install.sh
            └── version.info (Version=1.0.0, custom_opp_compiler_version=9.0.0)
```

### 库文件信息

```bash
$ file libops_sparse.so
```

```text
libops_sparse.so: ELF 64-bit LSB shared object, x86-64, version 1 (SYSV),
                  dynamically linked, with debug_info, not stripped
```

| 属性       | 值                                  |
| ---------- | ----------------------------------- |
| 文件类型   | ELF 64-bit LSB 共享对象             |
| 架构       | x86-64                              |
| 调试信息   | 包含 (with debug_info, not stripped)|
| 大小       | 2,682,312 字节 (~2.6 MB)            |

### 结论

ascend950 SOC 打包流程验证 **全部通过**。

1. SOC 版本正确识别为 ascend950，NPU 架构正确映射为 dav-3510
2. 架构特定目录机制工作正常：arch35 文件被正确过滤并重新添加
3. arch35 源文件被 ASC 编译器成功编译为目标文件
4. 生成的包名正确包含 950 标识 (`cann-950-ops-sparse-1.0.0_linux-x86_64.run`)
5. 包文件完整，包含头文件、库文件和安装脚本
6. 库文件包含 arch35 优化代码，大小约 2.6 MB
