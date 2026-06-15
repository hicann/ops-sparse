# 项目目录

> 本章罗列的部分目录或文件为可选，请以实际交付件为准。另请注意：
>
> - **算子源码目录**：`src/` 下按算子名组织（如 `src/spmv/`、`src/spmm/`），每个算子目录包含其 host 侧、kernel 侧实现及头文件。
> - **测试目录**：`test/` 下按算子名组织（如 `test/spmv/`、`test/spmm/`），每个算子目录包含对应的测试源码和说明文档，可参考 `test/<算子名>/README.md` 了解算子调用样例。
> - **测试工程**：`test/` 下用例需在配置工程时开启 `BUILD_TEST` 并设置 `TEST_NAMES`（分号分隔的算子测试子目录名）后才会参与构建，详见根目录 [CMakeLists.txt](../../../CMakeLists.txt) 与 [QuickStart](../../../docs/QUICKSTART.md)。
> - 若需补充算子或文档，欢迎参考 [贡献指南](../../../CONTRIBUTING.md)。

项目全量目录层级介绍如下：

```text
├── src                                                # 算子源码目录
│   ├── CMakeLists.txt                                 # src 源文件收集与编译规则
│   ├── common                                         # 初始化、上下文管理等通用代码
│   │   ├── aclsparse_handle_internal.h                # 句柄内部定义
│   │   ├── aclsparse_descr_internal.h                 # 描述符内部定义
│   │   ├── aclsparse_auxiliary.cpp                    # 辅助功能实现
│   │   ├── aclsparse_logger_manager.h                 # 日志管理器头文件
│   │   └── aclsparse_logger_manager.cpp               # 日志管理器实现
│   └── ${operator}                                    # 算子目录（如 spmv、spmm、sddmm）
│       ├── ${operator}_host.cpp                       # Host 侧：参数检查、任务下发等
│       ├── ${operator}_kernel.cpp                     # Device 侧：Ascend C 核函数入口
│       ├── ${operator}_csr_mat.h                      # 可选，CSR 矩阵数据结构头文件
│       ├── ${operator}_csr_mat.cpp                    # 可选，CSR 矩阵处理实现
│       ├── ${operator}.h                              # 算子对外头文件
│       └── ${arch_dir}                                # 可选，面向特定架构的源码（如 arch35），由 SOC 配置决定是否编译
├── include                                            # 对外头文件
│   ├── cann_ops_sparse.h                              # aclsparse API 声明
│   └── cann_ops_sparse_common.h                       # 公共类型与声明
├── docs                                               # 项目文档
│   ├── QUICKSTART.md                                  # 快速入门
│   └── zh
│       ├── api_list.md                                # 接口列表
│       └── install
│           ├── quick_install.md                       # 环境部署
│           └── dir_structure.md                       # 目录结构
├── examples                                           # 算子调用示例代码（预留目录，当前样例位于 test/ 下）
├── scripts                                            # 辅助脚本（打包安装说明、版本信息生成等）
│   ├── package
│   └── util
├── test                                               # 算子级测试（BUILD_TEST=ON 且配置 TEST_NAMES 时参与构建）
│   ├── CMakeLists.txt                                 # 按 TEST_NAMES 批量 add_subdirectory
│   └── ${operator}                                    # 算子目录（如 spmv、spmm）
│       ├── CMakeLists.txt
│       ├── ${operator}_test.cpp                       # 可执行测试源文件
│       └── README.md                                  # 可选，算子说明文档与调用样例
├── CMakeLists.txt                                     # 工程入口：ops_sparse 库、安装规则、可选测试与打包
├── CONTRIBUTING.md                                    # 贡献指南
├── LICENSE                                            # 开源许可证
├── README.md                                          # 项目总览
├── SECURITY.md                                        # 安全声明
├── build.sh                                           # 编译脚本
├── install_deps.sh                                    # 依赖安装脚本
└── ...
```
