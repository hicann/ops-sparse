---
name: asc-devkit-reference
description: asc-devkit 仓库参考技能，提供 Ascend C 官方 API 文档、示例代码、Tiling 配置等参考资源。
---

# asc-devkit 参考

## 仓库说明

`asc-devkit` 是 Ascend C 官方开发工具包，包含 API 文档、示例代码、实现参考和 Tiling 配置。

## 仓库管理

```bash
# 首次初始化（由 init.sh 自动执行）
git clone https://gitcode.com/cann/asc-devkit.git .agent/asc-devkit

# 更新到最新
git -C .agent/asc-devkit pull --rebase
```

## 仓库结构

```
.agent/asc-devkit/
├── docs/api/context/          # API 官方文档（Markdown，含配图）
│   ├── figures/               # 文档配图（流水线时序、内存布局、参数示意）
│   ├── Add.md                 # API 文档（可能有多个变体：Add-25.md, Add-35.md）
│   ├── DataCopy.md
│   └── ...
├── examples/                  # 示例代码（按算子分类）
│   ├── add/
│   ├── reduce/
│   └── ...
├── impl/adv_api/tiling/       # 官方 Tiling 参数配置
├── include/ascendc/           # 头文件（类型定义、接口声明）
├── cmake/                     # CMake 构建配置
└── scripts/                   # 辅助脚本
```

## API 变体搜索（强制）

Ascend C 的 API 可能有多个架构变体文件（如 `Add.md`、`Add-25.md`、`Add-35.md`），**必须搜索所有变体**：

```bash
# 搜索某 API 的所有文档变体
ls .agent/asc-devkit/docs/api/context/ | grep -i "^Add"
# 输出：Add.md  Add-25.md  Add-35.md
# 必须全部阅读后再确定使用哪个版本
```

**禁止**只读基础文档（如 `Add.md`）而忽略架构变体，不同变体的参数约束和平台支持可能不同。

## 配图阅读（强制）

API 文档中的配图（`figures/` 目录）常包含文字未明确表达的关键约束：
- **流水线时序图**：MTE2/V/MTE3 的依赖与并行关系
- **内存布局图**：UB 槽位摆放规则、对齐边界
- **参数示意图**：stride/block 在 UB/GM 的几何含义

**必须使用 Read 工具逐张读取配图**，禁止仅看正文文字。

## 检索策略（按优先级）

1. **索引查找**：先读 `docs/api/context/` 目录索引，定位目标 API
2. **API 文档**：读取 API Markdown + 所有变体 + 配图
3. **示例代码**：在 `examples/` 中搜索同类算子
4. **实现参考**：在 `impl/` 中查看底层实现
5. **头文件**：在 `include/ascendc/` 中确认类型定义

## 使用场景

### 架构设计阶段（1.3.A）

查阅以下资源辅助设计决策：
- `docs/api/context/`：API 官方文档，确认功能、参数约束和平台支持
- `examples/`：同类算子的示例代码，参考架构设计
- `impl/adv_api/tiling/`：官方 Tiling 参数配置参考

### 算子开发阶段（2.x.1.A）

查阅以下资源辅助代码实现：
- `docs/api/context/`：API 用法确认、参数签名核对（**搜索所有变体**）
- `examples/`：同类算子的代码实现参考
- `include/ascendc/`：头文件查阅，确认类型定义和接口声明

### 性能优化阶段（3.2）

查阅以下资源辅助瓶颈分析：
- `examples/`：优化实践和性能调优样例
- `impl/`：底层实现参考和优化策略
- `docs/api/context/figures/`：流水线时序图辅助瓶颈定位

## 与其他技能的关系

| 技能 | 职责分工 | 协作方式 |
|------|---------|---------|
| `ascendc-docs-search` | 在线文档搜索（华为昇腾社区） | 本地 asc-devkit 文档不足时，使用 ascendc-docs-search 在线搜索兜底 |
| `ascendc-api-best-practices` | API 使用约束和最佳实践 | 查阅 asc-devkit API 文档后，结合 best-practices 确认使用约束 |
| `op-samples-reference` | cann-samples 高性能样例 | asc-devkit 侧重 API/示例/实现参考，cann-samples 侧重端到端性能调优实践 |

## 注意事项

- API 文档是**权威来源**，与 `ascendc-api-best-practices` 技能互补（后者侧重实践经验）
- 示例代码的编码风格可能与 `sparse-ascendc-coding-rules` 不同，参考算法思路即可，不照搬风格
- 部分示例依赖特定 CANN 版本，注意检查 README 中的版本要求
- API 文档中的参数约束和平台支持信息以官方文档为准，设计方案中引用的 API 必须经过完整验证
