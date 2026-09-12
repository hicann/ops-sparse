# spgemm

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950PR/Ascend 950DT</term>：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：不支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：不支持
<!-- end id3 -->
<!-- npu="310b" id4 -->
- <term>Atlas 200I/500 A2 推理产品</term>：不支持
<!-- end id4 -->
<!-- npu="310p" id5 -->
- <term>Atlas 推理系列产品</term>：不支持
<!-- end id5 -->
<!-- npu="910" id6 -->
- <term>Atlas 训练系列产品</term>：不支持
<!-- end id6 -->

## 功能说明

- **接口功能**：

  `torch.sparse.mm` 用于计算两个二维稀疏矩阵的乘积。导入 `cann_ops_sparse` 后，NPU
  CSR 输入通过 `SparseCsrPrivateUse1`、NPU COO 输入通过 `SparsePrivateUse1` 分发到
  `aten::_sparse_sparse_matmul` 的 SpGEMM 实现。PyTorch 的 CSR 稀疏乘稀疏路径还可能在
  内部使用 `aten::_sparse_addmm`，扩展已为该路径注册受限实现。

  `torch_npu.sparse.mm` 是与 `torch.sparse.mm` 等价的 Python façade，二者进入同一个
  ATen dispatcher。首次执行时，扩展按需 JIT 编译 C++ wrapper，并调用
  `aclsparseSpGEMMWorkEstimation`、`aclsparseSpGEMMEstimateMemory`、
  `aclsparseSpGEMMCompute` 和 `aclsparseSpGEMMCopy` 完成计算。

- **计算公式**：

  对形状分别为 $(M, K)$ 和 $(K, N)$ 的稀疏矩阵 $A$、$B$，计算：

  $$
  C = A B, \qquad C_{ij} = \sum_{k=0}^{K-1} A_{ik}B_{kj}
  $$

  输出 $C$ 的形状为 $(M, N)$。多个中间乘积落到同一坐标时会合并；数值抵消得到的显式
  零仍保留在输出稀疏结构中。

## 函数原型

标准 PyTorch 接口：

```python
torch.sparse.mm(
    mat1,
    mat2,
    reduce="sum"
) -> Tensor
```

等价的 `torch_npu.sparse` façade：

```python
torch_npu.sparse.mm(
    mat1,
    mat2,
    reduce="sum"
) -> Tensor
```

## 参数说明

> **说明：**<br>
>
>- M 表示 `mat1` 的行数，K 表示 `mat1` 的列数和 `mat2` 的行数，N 表示 `mat2` 的列数。
>- nnzA、nnzB 分别表示两个输入稀疏矩阵存储的元素数量。

### torch.sparse.mm / torch_npu.sparse.mm

| 参数名 | 参数类型 | 可选/必选 | 描述 | 数据类型 | 维度(shape) |
|--------|----------|-----------|------|----------|-------------|
| mat1 | Tensor | 必选 | 左侧二维稀疏矩阵，必须位于 NPU。支持 CSR 或已合并的 COO；不支持稠密 Tensor。CSR 的 crow_indices 和 col_indices 支持 int32、int64，COO indices 为 int64。 | float16、bfloat16、float32、complex64 | (M, K) |
| mat2 | Tensor | 必选 | 右侧二维稀疏矩阵，必须与 mat1 位于同一 NPU，且 dtype、layout 与 mat1 一致。支持 CSR 或已合并的 COO；不支持稠密 Tensor。 | 与 mat1 相同 | (K, N) |
| reduce | str | 可选 | 归约方式，默认值为 `"sum"`。当前 NPU SpGEMM 路径仅支持默认的 `"sum"`，不支持 `"mean"`、`"amax"` 或 `"amin"`。 | string | - |

输入 values 无需预先保证连续；wrapper 会在当前 NPU 上生成连续视图。CSR 的 int64 索引会在
调用 ACLSparse 前转换为 int32，因此索引值、shape 和 nnz 必须位于 int32 可表示范围内。

## 返回值说明

### torch.sparse.mm / torch_npu.sparse.mm

| 参数名 | 参数类型 | 可选/必选 | 描述 | 数据类型 | 维度(shape) |
|--------|----------|-----------|------|----------|-------------|
| output | Tensor | 必选 | 新创建的 NPU 稀疏 Tensor，不与输入共用输出 storage。CSR 输入返回 CSR，COO 输入返回已合并的 COO。CSR 输出每行列索引严格升序且无重复坐标；COO 输出为 coalesced。 | 与 mat1、mat2 相同 | (M, N) |

CSR 输出的 crow_indices 和 col_indices 为 int32；COO 输出 indices 为 int64。输出与输入位于
同一 NPU 和当前 stream。

## 约束说明

- 当前接口支持前向计算，不支持 Sparse × Sparse 的反向传播。
- `mat1`、`mat2` 必须是二维 NPU 稀疏 Tensor，位于同一设备，且 value dtype 完全一致。
- 仅支持 CSR × CSR 或 COO × COO，不支持 CSR/COO 混合输入、CSC、BSR、BSC、batch 和广播。
- COO 输入必须已调用 `coalesce()`；未合并的 COO 输入会抛出异常。
- `mat1.shape[1]` 必须等于 `mat2.shape[0]`。
- M、K、N、nnzA、nnzB、输出 nnz 和中间乘积数量均不得超过 `INT32_MAX`，同时受可用
  Device 内存限制。
- ACLSparse 底层仅支持 int32、零基 CSR。输入 CSR 索引可以是 int64，但所有索引值必须能
  安全转换到 int32。
- 当前仅支持非转置矩阵乘，不支持转置或共轭转置参数。
- 支持输入 nnz 为 0 以及乘积 nnz 为 0；显式零作为结构项保留，不会自动裁剪。
- 实数 dtype 不接受带非零虚部的标量。complex64 路径的内部 alpha 支持复数。
- PyTorch 内部 `aten::_sparse_addmm` 路径仅支持 `beta=0`，并要求加数是 nnz 为 0 的 CSR
  Tensor；该限制不改变 `torch.sparse.mm` 的公开签名。
- wrapper 使用当前 NPU stream 并异步提交计算，不在接口末尾主动同步。将输出搬到 CPU、
  读取输出值或释放相关外部资源前，调用方应完成必要的 stream 同步。
- 首次 NPU 调用会触发 JIT 编译。运行前必须能够找到与目标 SOC 匹配的
  `libops_sparse.so`；源码构建时可通过 `OPS_SPARSE_LIB_DIR` 指定其目录。

### 特性参数组

| 特性参数组 | 参数字段名称 |
| :---: | :---: |
| 公共参数组 | mat1、mat2、output |
| 稀疏格式参数组 | layout、crow_indices、col_indices、indices |
| 数据类型参数组 | values dtype |
| 归约参数组 | reduce |
| 设备与执行参数组 | device、stream |

### 基准信息说明

#### 公共参数组

- 入参为空的场景处理：
  - nnz 为 0 的合法 CSR/COO 输入支持计算。
  - 结果没有结构项时，返回 shape 正确、nnz 为 0 的稀疏 Tensor。

| 参数 | 单参数校验 | 存在性校验 | 一致性校验 | 特性交叉校验 |
| --- | --- | --- | --- | --- |
| mat1 | NPU；二维；CSR 或 coalesced COO；value dtype 为 float16、bfloat16、float32 或 complex64 | 必须存在 | 与 mat2 的 device、dtype、layout 一致 | mat1.shape[1] 等于 mat2.shape[0] |
| mat2 | NPU；二维；CSR 或 coalesced COO；value dtype 为 float16、bfloat16、float32 或 complex64 | 必须存在 | 与 mat1 的 device、dtype、layout 一致 | mat2.shape[0] 等于 mat1.shape[1] |
| reduce | 仅支持字符串 `"sum"` | 可选，默认值为 `"sum"` | 无 | 非默认归约不进入当前 SpGEMM 路径 |
| output | CSR 或 coalesced COO；value dtype 与输入相同 | 必须输出 | device 和 layout 跟随输入 | shape 为 (mat1.shape[0], mat2.shape[1]) |

CSR 索引校验：

| 参数 | 数据类型 | shape | 约束 |
| --- | --- | --- | --- |
| crow_indices | int32、int64 | mat1 为 (M+1,)，mat2 为 (K+1,) | 零基、单调非降，末元素等于对应 nnz |
| col_indices | int32、int64 | 分别为 (nnzA,)、(nnzB,) | 每行索引位于合法列范围内 |
| output.crow_indices | int32 | (M+1,) | 零基、单调非降，末元素等于 output nnz |
| output.col_indices | int32 | (output nnz,) | 每行严格升序，无重复坐标 |

COO 索引校验：

| 参数 | 数据类型 | shape | 约束 |
| --- | --- | --- | --- |
| mat1.indices | int64 | (2, nnzA) | 输入必须 coalesced，索引位于 (M, K) 范围内 |
| mat2.indices | int64 | (2, nnzB) | 输入必须 coalesced，索引位于 (K, N) 范围内 |
| output.indices | int64 | (2, output nnz) | 输出为 coalesced COO |

## 确定性计算

默认支持确定性计算。当前 Torch Extension 固定使用 `ACL_SPARSE_SPGEMM_DEFAULT` 算法；相同
有效输入、运行环境和 stream 顺序下，输出稀疏结构顺序确定。浮点结果仍受数据类型精度和
运行环境影响。

## 调用示例

- 标准 PyTorch API 调用：

  ```python
  import torch
  import torch_npu
  import cann_ops_sparse

  device = "npu:0"
  torch.npu.set_device(device)

  matrix = torch.sparse_csr_tensor(
      torch.tensor([0, 2, 3], dtype=torch.int32, device=device),
      torch.tensor([0, 1, 1], dtype=torch.int32, device=device),
      torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32, device=device),
      size=(2, 2),
      device=device,
  )

  output = torch.sparse.mm(matrix, matrix)
  torch.npu.synchronize()

  print(output.crow_indices().cpu())  # tensor([0, 2, 3], dtype=torch.int32)
  print(output.col_indices().cpu())   # tensor([0, 1, 1], dtype=torch.int32)
  print(output.values().cpu())        # tensor([1., 8., 9.])
  ```

- `torch_npu.sparse` façade 调用：

  ```python
  import torch
  import torch_npu
  import cann_ops_sparse

  device = "npu:0"
  torch.npu.set_device(device)

  indices = torch.tensor(
      [[0, 0, 1], [0, 1, 1]], dtype=torch.int64, device=device
  )
  values = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32, device=device)
  matrix = torch.sparse_coo_tensor(indices, values, (2, 2), device=device).coalesce()

  output = torch_npu.sparse.mm(matrix, matrix)
  torch.npu.synchronize()

  print(output.indices().cpu())  # tensor([[0, 0, 1], [0, 1, 1]])
  print(output.values().cpu())   # tensor([1., 8., 9.])
  ```
