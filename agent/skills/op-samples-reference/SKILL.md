---
name: op-samples-reference
description: cann-samples 仓库参考技能，提供高性能算子样例、端到端调优实践、SIMT 编程模型参考资源。
---

# cann-samples 参考

## 仓库说明

`cann-samples` 是 CANN 官方高性能算子样例仓库，涵盖从基本编程模型到性能调优的全套实践。

## 仓库管理

```bash
# 首次初始化（由 init.sh 自动执行）
git clone https://gitcode.com/cann/cann-samples.git .agent/cann-samples

# 更新到最新
git -C .agent/cann-samples pull --rebase
```

## 仓库结构

```
.agent/cann-samples/
├── Samples/
│   ├── 0_Introduction/        # 基本编程模型和 Tiling 策略
│   │   ├── hello_world/
│   │   ├── vector_add/
│   │   └── ...
│   ├── 1_Features/            # 优化特性实现
│   │   ├── hardware_features/
│   │   │   ├── simt/          # SIMT 编程模型
│   │   │   └── ...
│   │   ├── double_buffer/
│   │   └── ...
│   ├── 2_Performance/         # 性能调优实践
│   │   ├── memory_opt/
│   │   ├── instruction_opt/
│   │   └── ...
│   └── 3_Utilities/           # 辅助工具
└── README.md
```

## 检索策略（按优先级）

1. **目录索引**：先读 `Samples/` 目录结构，定位目标分类
2. **样例 README**：读目标样例的 README.md，了解功能和适用场景
3. **源代码**：读 `.cpp` / `.h` 文件，参考实现细节
4. **性能分析**：读 `2_Performance/` 中的调优实践

## 使用场景

### 架构设计阶段（1.3.A）

查阅以下资源辅助设计选型：
- `Samples/0_Introduction/`：基本编程模型和 Tiling 策略
- `Samples/1_Features/`：可用优化手段，提前规划优化策略
- `Samples/2_Performance/`：同类算子的架构设计和性能优化路径
- **SIMT 算子设计**：若目标算子采用 SIMT 编程模型，**必须参考** `Samples/1_Features/hardware_features/simt/` 中的样例，了解 SIMT 架构下的 Tiling 策略、线程映射和 Kernel 结构设计

**操作方式**：进入 `.agent/cann-samples/Samples/` 对应子目录，阅读 README.md 了解样例概述，再进入具体样例目录查看设计文档和源码。

### 算子开发阶段（2.x.1.A）

查阅以下资源辅助代码实现：
- `Samples/0_Introduction/`：编程模型和 Tiling 实现
- `Samples/1_Features/`：优化特性实现
- **SIMT 算子开发**：若目标算子采用 SIMT 编程模型，**必须参考** `Samples/1_Features/hardware_features/simt/` 中的样例代码，学习 SIMT Kernel 的具体实现方式、线程索引计算、共享内存使用等
- 构建配置：参考样例中的 CMakeLists.txt 了解编译配置和依赖管理方式

**操作方式**：直接阅读 `.agent/cann-samples/Samples/` 下目标样例的源码文件（`.cpp`/`.h`），关注 kernel 实现、tiling 数据结构和 host 侧调用逻辑。

### 性能优化阶段（3.2）

查阅以下资源辅助瓶颈分析：
- `Samples/2_Performance/memory_opt/`：内存访问优化（UB 复用、对齐、合并搬运）
- `Samples/2_Performance/instruction_opt/`：指令级优化（向量化、流水线）
- `Samples/2_Performance/`：系统级优化（多核负载均衡、任务调度）
- 端到端调优路径：`Samples/2_Performance/` 下的 story 类样例通常包含从 baseline 到极限性能的完整调优过程
- 性能分析工具：参考 `Samples/3_Utilities/` 中的工具样例，学习 Profiling 和仿真分析方法

**操作方式**：优先阅读 `Samples/2_Performance/` 下相关 story 目录中的 docs/ 文档和 README.md，理解调优思路，再对照代码实现学习具体优化手法。

## 注意事项

- 样例代码的编码风格可能与 `sparse-ascendc-coding-rules` 不同，参考算法思路即可，不照搬风格
- 部分样例依赖 `third_party/tensor_api` 子模块，若缺失需手动初始化
- 注意检查样例 README 中的 CANN 版本要求，确保与当前环境兼容
- 仓库持续更新，若未找到相关参考，可查看仓库根目录 README.md 的 Latest News 了解最近新增内容
