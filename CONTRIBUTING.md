# 贡献指南

本项目欢迎广大开发者体验并参与贡献，在参与社区贡献之前。请参见[cann-community](https://gitcode.com/cann/community)了解行为准则，进行CLA协议签署，了解源码仓的贡献流程。

开发者准备本地代码与提交PR时需要重点关注如下几点：

1. 提交PR时，请按照PR模板仔细填写本次PR的业务背景、目的、方案等信息。
2. 若您的修改不是简单的bug修复，而是涉及到新增特性、新增接口、新增配置参数或者修改代码流程等，请务必先通过Issue进行方案讨论，以避免您的代码被拒绝合入。若您不确定本次修改是否可被归为"简单的bug修复"，亦可通过提交Issue进行方案讨论。

开发者贡献场景主要包括：

## 一、贡献新算子

  算子开发贡献流程如下：

  ![算子开发贡献流程](./docs/zh/figures/算子开发贡献流程.png "算子开发贡献流程图")

  如果您有全新的稀疏算子希望基于NPU设计与实现，欢迎在Issue中提出您的想法与设计方案。完整的贡献过程如下：

### 1. 创建Issue需求

新建 `Requirement|需求建议` 类Issue，并阐明新增算子的设计方案。Issue一般需包含以下内容：

- **背景信息**
- **价值/作用**
- **设计方案**

请在提交的Issue中评论`/assign @yourself` 认领该任务。

### 2. 需求评审

Sig组将指派Committer对您提交的Issue进行评审并反馈修改意见。请在完成修改后，于Issue中@对应Committer。

若需求被接纳，[sig成员](https://gitcode.com/cann/community/blob/master/CANN/sigs/ops-linear-algebra/sig-info.yaml)将为您分配合适的算子分类路径，请将贡献算子提交至对应目录。

### 3. PR提交

生态最简算子交付件如下：

```text
src/${operator}/                            # 算子源码目录
├── ${operator}_host.cpp                    # Host侧实现文件
├── ${operator}_kernel.cpp                  # Device侧Kernel实现文件
├── ${operator}.h                           # 算子对外头文件
└── ${arch_dir}/                            # 可选，面向特定架构的实现（如 arch35）
    ├── ${operator}_host.cpp                # 架构专用Host侧实现
    └── ${operator}_kernel.cpp              # 架构专用Device侧Kernel实现

test/${operator}/                           # 算子测试目录
├── CMakeLists.txt                          # 测试编译配置文件
└── ${operator}_test.cpp                    # 算子测试文件
```

PR上库要求：

- 代码交付件：需提供算子Host侧实现、Device侧Kernel实现、算子测试文件，开发过程参考[快速入门](docs/QUICKSTART.md)。
- 文档交付件：算子README文档为必选，需放置在`test/${operator}/README.md`，其余文档可视情况提供。
- 精度要求：新贡献算子需满足精度标准，具体请参见[生态算子开源精度标准](https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/experimental_standard.md)。
- 合规检查：
  - 代码是否符合《[C++ 编程规范](https://gitcode.com/cann/community/blob/master/contributor/coding-standards/C++%20Coding%20standards.md)》
  - 代码是否编译通过（`bash build.sh --ops=${operator} --soc=${soc_version}`）
  - 测试是否全部通过（`bash build.sh --ops=${operator} --soc=${soc_version} --run`）
  - Markdown文档语法是否符合规范
- 接口声明：新增算子需在 `include/cann_ops_sparse.h` 中添加API声明。
- 贡献目录：按sig成员意见提交至 `src/${operator}/` 目录下，可参考已有算子文件放置规则。
- PR提交：通过`git`命令提交目标分支PR，检查PR标题是否清晰、PR描述是否规范（指明更改内容和原因、是否关联对应Issue）、是否签署CLA。

如果您希望贡献项目标准算子，其交付件和开发过程比生态算子复杂，包括多架构支持、CSR矩阵预处理、GTest测试框架等，具体贡献指导参考[附录](#附录)。

### 4. CI门禁

通过评论 `compile` 指令触发开源仓门禁，并依据CI检测结果进行修改，目前CI门禁包含以下检查项：

- 代码编译
- 静态检查（如涉及codecheck误报，请提交给sig成员屏蔽）
- UT测试

门禁通过后，请在关联的Issue中@指派的Committer。

### 5. Committer检视

Committer检视后将反馈检视意见，请根据意见修改，完成后@指派的Committer。

### 6. Maintainer合入

Committer检视通过后，标注 `/lgtm`标签。Maintainer将在1天内进行最终审核，确认无问题后，将标注 `/approve` 标签合入PR。

## 二、算子Bug修复

如果您在本项目中发现了某些算子Bug，希望对其进行修复，欢迎您新建Issue进行反馈和跟踪处理。

您可以按照[提交Issue/处理Issue任务](https://gitcode.com/cann/community#提交Issue处理Issue任务)指引新建 `Bug-Report|缺陷反馈` 类Issue对Bug进行描述，然后在评论框中输入"/assign"或"/assign @yourself"，将该Issue分配给您进行处理。

## 三、算子优化

如果您对本项目中某些算子实现有泛化性增强/性能优化思路，希望着手实现这些优化点，欢迎您对算子进行优化贡献。

您可以按照[提交Issue/处理Issue任务](https://gitcode.com/cann/community#提交Issue处理Issue任务)指引新建 `Requirement|需求建议` 类Issue对优化点进行说明，并提供您的设计方案，然后在评论框中输入"/assign"或"/assign @yourself"，将该Issue分配给您进行跟踪优化。

## 四、文档纠错

如果您在本项目中发现某些算子文档描述错误，欢迎您新建Issue进行反馈和修复。

您可以按照[提交Issue/处理Issue任务](https://gitcode.com/cann/community#提交Issue处理Issue任务)指引新建 `Documentation|文档反馈` 类Issue指出对应文档的问题，然后在评论框中输入"/assign"或"/assign @yourself"，将该Issue分配给您纠正对应文档描述。

## 五、帮助解决他人Issue

如果社区中他人遇到的问题您有合适的解决方法，欢迎您在Issue中发表评论交流，帮助他人解决问题和痛点，共同优化易用性。

如果对应Issue需要进行代码修改，您可以在Issue评论框中输入"/assign"或"/assign @yourself"，将该Issue分配给您，跟踪协助解决问题。

## 附录

项目标准算子交付件如下：

```text
src/${operator}/                            # 算子源码目录
├── ${operator}_host.cpp                    # Host侧：参数校验、任务下发
├── ${operator}_kernel.cpp                  # Device侧：Ascend C核函数实现
├── ${operator}_csr_mat.h                   # 可选，CSR矩阵数据结构头文件
├── ${operator}_csr_mat.cpp                 # 可选，CSR矩阵预处理实现
├── ${operator}.h                           # 算子对外头文件
└── ${arch_dir}/                            # 可选，面向特定架构的实现（如 arch35）
    ├── ${operator}_host.cpp                # 架构专用Host侧实现
    ├── ${operator}_kernel.cpp              # 架构专用Device侧Kernel实现
    ├── ${operator}_csr_mat.h               # 可选，架构专用CSR矩阵数据结构头文件
    └── ${operator}_csr_mat.cpp             # 可选，架构专用CSR矩阵预处理实现

test/${operator}/                           # 算子测试目录
├── CMakeLists.txt                          # 测试编译配置
├── ${operator}_test.cpp                    # 测试主文件
└── README.md                               # 算子说明文档
```

> **说明**：
> - `${operator}` 为算子名，如 `spmv`、`spmm`、`sddmm`。
> - `${arch_dir}` 为架构子目录，如 `arch35`（Ascend 950），由编译时 `--soc` 参数决定。
> - 算子源码目录（`src/`）下**无需**提供 `CMakeLists.txt`，由 `src/CMakeLists.txt` 自动收集。
> - 测试目录（`test/`）下**必须**提供 `CMakeLists.txt`，注册测试可执行文件。
> - 目录结构详细说明参考[目录结构说明](docs/zh/install/dir_structure.md)。

**具体示例**（以 `spmv` 算子为例）：

```text
src/spmv/
├── spmv_host.cpp
├── spmv_kernel.cpp
├── spmv_csr_mat.h
├── spmv_csr_mat.cpp
└── spmv.h

test/spmv/
├── CMakeLists.txt
├── spmv_test.cpp
└── README.md
```

PR上库要求：

除满足[生态最简算子交付件](#3-pr提交)的基本要求外，项目标准算子还需额外满足：

- 代码交付件：需提供多架构支持（如 arch35）、CSR矩阵预处理实现、GTest测试框架等。
- 文档交付件：算子README文档需包含接口说明、参数约束、算法流程、编译测试命令、调用示例等内容。
- 合规检查：代码还需符合标准算子基础编程规范。
