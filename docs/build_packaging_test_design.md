# ops-sparse 编包、构建和测试系统设计文档

## 1. 概述

本文档详细描述 ops-sparse 项目的构建系统、打包机制以及测试框架的设计。ops-sparse 是面向华为昇腾 AI 处理器的高性能稀疏算子库，提供 SpMV（稀疏矩阵-向量乘法）、SpMM（稀疏矩阵-矩阵乘法）等基础稀疏运算。

## 2. 整体架构

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        构建系统架构                                      │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   ┌──────────┐     ┌──────────────┐     ┌──────────────────┐           │
│   │ build.sh │────▶│   CMake      │────▶│  libops_sparse.so │           │
│   │ 构建脚本  │     │ 构建系统      │     │   动态库          │           │
│   └──────────┘     └──────────────┘     └──────────────────┘           │
│          │              │                                               │
│          ▼              ▼                                               │
│   ┌──────────┐     ┌──────────────┐                                    │
│   │  --ops   │     │  测试程序    │                                    │
│   │  --soc   │     │  *_test     │                                    │
│   │  --run   │     └──────────────┘                                    │
│   │  --pkg   │                                                         │
│   └──────────┘                                                         │
│          │                                                             │
│          ▼                                                             │
│   ┌─────────────────────────────────────────────────────────┐          │
│   │                    打包系统                              │          │
│   │  ┌──────────┐  ┌──────────┐  ┌──────────────────────┐   │          │
│   │  │ package  │  │makeself  │  │ cann-950-ops-sparse │   │          │
│   │  │.cmake    │──│.cmake    │──▶│ -1.0.0_linux-x86_64 │   │          │
│   │  │(CPack)   │  │(.run包)   │  │      .run           │   │          │
│   │  └──────────┘  └──────────┘  └──────────────────────┘   │          │
│   └─────────────────────────────────────────────────────────┘          │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## 3. 构建系统设计

### 3.1 构建入口 (build.sh)

`build.sh` 是主要的构建入口脚本，提供以下命令行参数：

| 参数 | 说明 | 示例 |
|------|------|------|
| `--ops=op1,op2` | 指定要构建的算子（逗号分隔） | `--ops=spmv,spmm` |
| `--run` | 构建并运行测试 | `--run` |
| `--soc=SOC` | 指定目标 SOC 版本 | `--soc=ascend950` |
| `--pkg` | 构建并打包成 run 包 | `--pkg` |

**使用示例：**

```bash
# 只编译库
bash build.sh

# 编译指定算子
bash build.sh --ops=spmv
bash build.sh --ops=spmv,spmm

# 编译并运行全部测试
bash build.sh --run

# 编译并运行指定算子测试
bash build.sh --ops=spmv --run

# 编译并打包
bash build.sh --pkg

# 为指定 SOC 打包
bash build.sh --pkg --soc=ascend950
```

**参数约束：**
- `--run` 和 `--pkg` 不能同时使用
- `--run` 未指定 `--ops` 时，默认包含 `test/` 下所有包含 `CMakeLists.txt` 的算子子目录

### 3.2 CMake 构建系统

#### 3.2.1 根 CMakeLists.txt

```cmake
# 项目信息
project(ops_sparse VERSION 1.0.0)

# SOC_VERSION -> NPU_ARCH 映射
# ascend910b/910_93 -> dav-2201
# ascend950         -> dav-3510
# ascend310p        -> dav-2002
```

**关键配置选项：**

| 选项 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `SOC_VERSION` | 字符串 | `ascend910b3` | 目标 SOC 版本 |
| `ENABLE_PACKAGE` | 布尔 | `OFF` | 启用打包功能 |
| `BUILD_TEST` | 布尔 | `OFF` | 构建测试程序 |

#### 3.2.2 SOC 版本映射机制

```cmake
function(get_soc_arch_dirs soc_version arch_dirs)
  # ascend950 系列  -> arch35 (DAV-3510 优化)
  # ascend310p 系列 -> arch20 (DAV-2002 优化)
  # ascend910b 系列 -> 通用实现 (DAV-2201)
endfunction()
```

| SOC 版本 | NPU 架构 | 优化目录 | 说明 |
|----------|----------|----------|------|
| ascend910b* | dav-2201 | 无 | 使用通用实现 |
| ascend910_93* | dav-2201 | 无 | 使用通用实现 |
| ascend950* | dav-3510 | arch35 | 使用 arch35 优化 |
| ascend310p* | dav-2002 | arch20 | 使用 arch20 优化 |

#### 3.2.3 架构特定源码过滤 (src/CMakeLists.txt)

源码选择机制：
1. 收集所有 `.cpp` 文件作为基础源文件
2. 过滤掉架构特定目录中的文件（`arch35`, `arch20`）
3. 根据 `SOC_ARCH_DIRS` 重新添加当前 SOC 对应的架构特定文件

```
src/
├── spmv/
│   ├── spmv_host.cpp          # 通用实现
│   ├── spmv_kernel.cpp        # 通用实现
│   └── arch35/
│       └── spmv_kernel_arch35.cpp  # ascend950 专用优化
└── spmm/
    ├── spmm_host.cpp
    ├── spmm_kernel.cpp
    └── arch35/
        └── spmm_kernel_arch35.cpp
```

### 3.3 支持的 SOC 版本

构建脚本通过前缀匹配识别 SOC 版本：

```bash
SUPPORT_COMPUTE_UNIT_SHORT=(
    "ascend910_93"  # 最长字符串优先匹配
    "ascend910b"
    "ascend950"
    "ascend310p"
)
```

## 4. 打包系统设计

### 4.1 打包流程

当使用 `--pkg` 参数时，系统执行以下流程：

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  build.sh   │───▶│  CMake      │───▶│   CPack     │───▶│  makeself   │
│  --pkg      │    │  ENABLE_    │    │  External   │    │  .run 包    │
│             │    │  PACKAGE=ON │    │  Generator  │    │             │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
```

### 4.2 打包配置 (cmake/package.cmake)

#### 4.2.1 包命名规则

```
cann-{CHIP_SHORT}-ops-sparse-{VERSION}_linux-{ARCH}.run
```

| SOC 版本 | CHIP_SHORT | 示例包名 |
|----------|------------|----------|
| ascend950* | 950 | `cann-950-ops-sparse-1.0.0_linux-x86_64.run` |
| ascend910b* | 910b | `cann-910b-ops-sparse-1.0.0_linux-x86_64.run` |
| ascend910_93* | A3 | `cann-A3-ops-sparse-1.0.0_linux-x86_64.run` |
| ascend310p* | 310p | `cann-310p-ops-sparse-1.0.0_linux-x86_64.run` |

#### 4.2.2 makeself 打包配置

```cmake
# makeself.cmake 关键配置
execute_process(
  COMMAND "${MAKESELF_SCRIPT}" --nocomp
          --help-header share/info/ops_sparse/help.info
          "${CPACK_TEMPORARY_DIRECTORY}" "${RUN_FILE}"
          "${CPACK_PACKAGE_NAME} ${CPACK_PACKAGE_VERSION}"
          share/info/ops_sparse/install.sh
)
```

参数说明：
- `--nocomp`: 不压缩，加快解压速度
- `--help-header`: 帮助信息文件路径
- `startup_script`: 启动脚本为 `install.sh`

### 4.3 Run 包文件结构

生成的 `.run` 包内部结构：

```
cann-{chip}-ops-sparse-{version}_linux-{arch}.run
├── include/
│   └── cann_ops_sparse.h          # 公共头文件
├── lib64/
│   └── libops_sparse.so             # 动态库
└── share/
    └── info/
        └── ops_sparse/
            ├── help.info            # 帮助信息
            ├── install.sh           # 安装/卸载脚本
            └── version.info         # 版本信息
```

## 5. 安装与卸载系统

### 5.1 安装脚本 (scripts/package/install.sh)

#### 5.1.1 安装命令

```bash
# 默认路径安装
./cann-950-ops-sparse-1.0.0_linux-x86_64.run --install

# 指定路径安装
./cann-950-ops-sparse-1.0.0_linux-x86_64.run --install --install-path=/opt/ascend

# 静默安装
./cann-950-ops-sparse-1.0.0_linux-x86_64.run --install --quiet
```

#### 5.1.2 默认安装路径

| 用户类型 | 默认路径 |
|----------|----------|
| root | `/usr/local/Ascend` |
| 普通用户 | `~/Ascend` |

#### 5.1.3 安装后文件布局

```
<install-path>/
└── cann/
    ├── lib64/
    │   └── libops_sparse.so
    ├── include/
    │   └── cann_ops_sparse.h
    └── share/
        └── info/
            └── ops_sparse/
                ├── help.info
                ├── install.sh
                └── version.info
```

#### 5.1.4 卸载功能

```bash
# 卸载
./cann-950-ops-sparse-1.0.0_linux-x86_64.run --uninstall

# 从指定路径卸载
./cann-950-ops-sparse-1.0.0_linux-x86_64.run --uninstall --install-path=/opt/ascend
```

#### 5.1.5 安装脚本特性

1. **路径解析**：自动处理相对路径、软链接，解析为绝对路径
2. **权限管理**：临时授予写权限（`chmod u+w`），安装后恢复
3. **路径校验**：禁止包含空格或特殊字符（`< > | & $ ; \``）的路径
4. **安全检查**：检查 run 包完整性（必须包含 lib64 和 include）

### 5.2 帮助信息 (scripts/package/help.info)

```
--install                 Install ops_sparse libraries (libops_sparse.so, headers)
  --install-path=<path>    Specify installation path
  --quiet                  Quiet mode, skip confirmations
--uninstall               Uninstall ops_sparse from the specified path
  --install-path=<path>    Specify installation path to uninstall
  --quiet                  Quiet mode
--help                    Show this help
```

## 6. 测试系统设计

### 6.1 测试目录结构

```
test/
├── CMakeLists.txt          # 遍历 TEST_NAMES 中的算子
├── spmv/
│   ├── CMakeLists.txt      # 定义 spmv_test 可执行文件
│   └── spmv_test.cpp       # SpMV 测试实现
└── spmm/
    ├── CMakeLists.txt
    └── spmm_test.cpp
```

### 6.2 测试构建配置

测试构建通过 `BUILD_TEST` 和 `TEST_NAMES` 变量控制：

```bash
# 在 build.sh 中转换参数
cmake -DBUILD_TEST=ON -DTEST_NAMES="spmv;spmm" ...
```

### 6.3 测试运行机制

`build.sh` 的测试执行流程：

```bash
# 1. 构建测试可执行文件
cmake --build ${BUILD_DIR} -j

# 2. 逐个运行测试
for op in "${OP_ARRAY[@]}"; do
    TEST_BIN="${BUILD_DIR}/test/${op}/${op}_test"
    "${TEST_BIN}"
    # 记录通过/失败状态
done

# 3. 输出测试汇总
# ========================================
# Test Summary:
#   Passed: 2 - spmm spmv
#   Failed: 0 -
# ========================================
```

### 6.4 测试 CMakeLists.txt 示例

```cmake
# test/spmv/CMakeLists.txt
set(op spmv)
add_executable(${op}_test ${op}_test.cpp)
target_link_libraries(${op}_test PRIVATE
    ops_sparse
    ${ASCEND_HOME_PATH}/lib64/libascendcl.so
)
```

## 7. 关键参数流程图

### 7.1 --ops 参数流程

```
build.sh --ops=spmv,spmm
         │
         ▼
    ┌────────────┐
    │  逗号分隔   │───▶ spmv,spmm
    │  字符串解析 │
    └────────────┘
         │
         ▼
    ┌────────────┐
    │ 转换为     │───▶ spmv;spmm (CMake 列表格式)
    │ CMake 列表 │
    └────────────┘
         │
         ▼
cmake -DTEST_NAMES=spmv;spmm -DBUILD_TEST=ON
         │
         ▼
    test/CMakeLists.txt 遍历 TEST_NAMES
         │
         ▼
    构建 spmv_test, spmm_test
```

### 7.2 --soc 参数流程

```
build.sh --soc=ascend950
         │
         ▼
    ┌────────────┐
    │ SOC 版本   │───▶ ascend950
    │ 校验(前缀) │     匹配 ascend950
    └────────────┘
         │
         ▼
cmake -DSOC_VERSION=ascend950
         │
         ▼
    ┌────────────┐
    │ NPU_ARCH   │───▶ dav-3510
    │ 映射       │
    └────────────┘
         │
         ▼
    ┌────────────┐
    │ SOC_ARCH   │───▶ arch35
    │ _DIRS 确定 │
    └────────────┘
         │
         ▼
    src/CMakeLists.txt 筛选源码
         │
         ▼
    编译 arch35/ 下优化代码
```

### 7.3 --pkg 参数流程

```
build.sh --pkg --soc=ascend950
         │
         ▼
cmake -DENABLE_PACKAGE=ON
         │
         ▼
    cmake/package.cmake
         │
         ├──────▶ 下载 makeself（如需要）
         │
         ├──────▶ 生成 version.info
         │
         ├──────▶ 安装 install.sh, help.info
         │
         └──────▶ 调用 pack() 函数
                      │
                      ▼
                 CPACK External
                 Generator
                      │
                      ▼
                 cmake/makeself.cmake
                      │
                      ▼
                 makeself.sh --nocomp
                      │
                      ▼
              ┌──────────────────┐
              │ cann-950-ops-    │
              │ sparse-1.0.0_    │
              │ linux-x86_64.run │
              └──────────────────┘
```

## 8. 版本信息生成

### 8.1 version.info 内容

由 `scripts/util/gen_version_info.sh` 生成，包含：
- Run 包版本号（`cmake/version.cmake` 中定义）
- 编译器版本信息

### 8.2 版本定义

```cmake
# cmake/version.cmake
set(OPS_SPARSE_RUN_VERSION "1.0.0")
```

## 9. 依赖关系

### 9.1 构建依赖

| 依赖 | 用途 | 来源 |
|------|------|------|
| Ascend/CANN Toolkit | 算子编译和链接 | 环境变量 `ASCEND_HOME_PATH` 或 `ASCEND_TOOLKIT_HOME` |
| CMake 3.16+ | 构建系统 | 系统安装 |
| makeself | 生成 .run 包 | 自动下载到 `build/third_party/` |

### 9.2 运行时依赖

| 依赖 | 用途 |
|------|------|
| libascendcl.so | AscendCL 运行时库 |
| libops_sparse.so | 本库动态库 |

## 10. 输出目录结构

### 10.1 构建输出 (build/)

```
build/
├── CMakeFiles/
├── src/
│   └── CMakeFiles/ops_sparse.dir/
├── test/
│   ├── spmv/
│   │   └── spmv_test
│   └── spmm/
│       └── spmm_test
└── third_party/
    └── makeself/          # 自动下载的 makeself 工具
```

### 10.2 安装输出 (build_out/)

```
build_out/
├── lib64/
│   └── libops_sparse.so
└── include/
    └── cann_ops_sparse.h
```

### 10.3 打包输出 (build_out/*.run)

```
build_out/
└── cann-950-ops-sparse-1.0.0_linux-x86_64.run
```

## 11. 设计要点总结

### 11.1 架构设计亮点

1. **SOC 自适应编译**：通过 `get_soc_arch_dirs()` 函数自动选择最优实现
2. **灵活的算子选择**：`--ops` 参数支持单算子、多算子构建和测试
3. **一体化打包**：集成 makeself 生成自解压 run 包
4. **完善的安装机制**：支持安装、卸载、自定义路径、静默模式

### 11.2 扩展性设计

1. **新增 SOC 版本**：在 `get_soc_arch_dirs()` 中添加映射，在 `SUPPORT_COMPUTE_UNIT_SHORT` 中添加支持
2. **新增算子**：在 `src/` 和 `test/` 下创建对应目录，自动被 CMake 识别
3. **新增架构优化**：创建 `archXX/` 目录，在 `ARCH_SPECIFIC_DIRS` 中注册

### 11.3 安全考虑

1. 安装路径合法性检查（禁止特殊字符）
2. 软链接解析为实际路径
3. 临时权限授予机制（安装后恢复）
4. run 包完整性校验

## 12. 参考文档

- 主 CMakeLists.txt: `/home/chenb/code/fork/ops-sparse/CMakeLists.txt`
- 打包配置: `/home/chenb/code/fork/ops-sparse/cmake/package.cmake`
- 安装脚本: `/home/chenb/code/fork/ops-sparse/scripts/package/install.sh`
- 构建脚本: `/home/chenb/code/fork/ops-sparse/build.sh`
