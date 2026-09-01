# SDDMM算子 (arch22)

## 算子概述

SDDMM（Sampled Dense-Dense Matrix Multiplication）实现两个稠密矩阵相乘后按稀疏矩阵模式采样的运算，对标 cuSPARSE `cusparseSDDMM`。

```
C_out = (alpha * op(X) * op(Y) + beta * C) ∘ spy(C)
```

## 支持芯片

- Ascend 910B（架构 arch22 / DAV_2201）

## 接口与参数

公共接口定义、参数说明及调用示例见上层 [../README.md](../README.md)。本文件仅补充 arch22 特有的实现约束和说明。

## 精度标准

采用 MIXED_TOLERANCE 模式：逐元素 `|actual - golden| ≤ max(atol + rtol * |golden|, maxAbsErrLimit)`，整体 `matchedRatio ≥ 0.99`。

| 数据类型 | rtol | atol | maxAbsErrLimit fixedValue |
|----------|------|------|---------------------------|
| FLOAT32  | 2⁻¹⁰ | 2⁻²³ | 1e-2 |
| FLOAT16  | 2⁻⁹  | 2⁻⁹  | 1e-1 |

FP16 路径在 FP32 中完成点积累加，写回前饱和截断到 [−65504, 65504]。

## Ascend 910B (arch22) 实现说明

FP32 主路径按非零元分块的向量化归约组织计算：

1. 每行将 X 对应行载入 UB 一次
2. 至多 `jTile` 个非零元的 Y 列连续暂存为 UB 中 `jTile × kPitch` 矩阵，连续列合并为一次多块 DMA
3. `src0RepStride=0` 广播 X 行做 `Mul`，`WholeReduceSum` 折叠为点积
4. `y = alpha * acc + beta * C` 全部在向量流水完成

以下情形回退到逐非零元标量路径：Y 列非连续、`K > 4088`（hugeK）。`alpha == 0` 时整行退化为向量 `y = beta * C`。

FP16 路径在 FP32 精度下完成所有计算，仅在 GM 读写时进行 half↔float 转换。当 K > 4088 时，FP16 使用逐行分区（不使用 nnz 分区），DotProduct 通过逐块标量累加保证正确性。
