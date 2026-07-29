# Scatter算子

> 6.1 文档补全阶段产出。developer-doc 依据算子代码与设计文档编写，面向使用者。

## 算子概述

`aclsparseScatter` 算子用于将稀疏向量 `X` 的非零值散布（scatter）到稠密向量 `Y` 的对应位置，原地更新 `Y`。对 `X` 中每个非零元素执行：

```
Y[X.indices[i]] = X.values[i],  i ∈ [0, X.nnz)
```

该接口为 Generic API（描述符模式），对标 cuSPARSE `cusparseScatter`。语义为纯数据搬运，不涉及任何浮点运算，输出与输入 `X.values` 逐位一致（bitwise exact）。

关键语义说明：

- **idxBase 处理**：当 `X.idxBase = ACL_SPARSE_INDEX_BASE_ONE` 时，`indices` 基于 1，内部访问 `Y` 时减 1 调整；`X.idxBase = ACL_SPARSE_INDEX_BASE_ZERO` 时直接使用。
- **索引排序**：允许 `indices` 未排序，算子不依赖排序假设。
- **索引重复**：当多个 `i` 的 `indices[i]` 指向同一位置时，最终写入值为最后一个完成的写入（last-write-wins），由于多线程并发执行，**行为不确定**，与 cuSPARSE 一致。
- **索引越界**：算子不做运行时索引值校验（host 侧无法读取 device 内存）。调用方需保证 `indices[i] - idxBase ∈ [0, Y.nums)`，越界访问行为未定义，与 cuSPARSE 一致。
- **未写入位置**：`Y` 中未被 `scatter` 覆盖的位置保持原值不变。
- **nnz = 0**：直接返回成功，不启动 kernel，`Y` 保持原值。

包含以下接口：

| 接口名 | 功能简述 |
|--------|---------|
| aclsparseScatter | 将稀疏向量 X 的非零值散布到稠密向量 Y 的对应位置，原地更新 Y |

## 算子执行接口

### aclsparseScatter

#### 产品支持情况

- Ascend 950PR / Ascend 950DT：支持
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品：不支持
- Atlas A2 训练系列产品 / Atlas A2 推理系列产品：不支持

#### 函数原型

```cpp
aclsparseStatus_t aclsparseScatter(aclsparseHandle_t handle, aclsparseConstSpVecDescr_t vecX, aclsparseDnVecDescr_t vecY)
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 |
|--------|----------|---------|------|
| handle | 输入 | aclsparseHandle_t | ops-sparse 库上下文句柄，携带 stream 等信息，不可为 nullptr，须先调用 `aclsparseSetStream` 绑定 stream，Host 内存 |
| vecX | 输入 | aclsparseConstSpVecDescr_t | 稀疏向量描述符（const，只读）。含 size、nnz、indices[nnz]、values[nnz]、idxType、idxBase、valueType；`size <= vecY.nums`，`nnz <= size`；Host(descr) / Device(data) 内存 |
| vecY | 输入/输出 | aclsparseDnVecDescr_t | 稠密向量描述符，其 values[nums] 被 `X` 散布原地写入。`valueType` 必须与 `vecX.valueType` 一致；Host(descr) / Device(data) 内存 |

> vecX 的 `indices` 与 `values` 为 Device 内存指针，由描述符创建时（`aclsparseCreateConstSpVec` / `aclsparseCreateSpVec`）传入；vecY 的 `values` 同理由 `aclsparseCreateDnVec` 传入。

#### 支持数据类型

| 数据类型 | aclDataType | 说明 |
|----------|-------------|------|
| FP32 | ACL_FLOAT | 单精度浮点，valSize = 4 bytes |
| FP16 | ACL_FLOAT16 | 半精度浮点，valSize = 2 bytes |
| BF16 | ACL_BF16 | bfloat16，valSize = 2 bytes |

| 索引类型 | aclsparseIndexType_t | 说明 |
|----------|----------------------|------|
| I32 | ACL_SPARSE_INDEX_32I | 32 位有符号整数 |
| I64 | ACL_SPARSE_INDEX_64I | 64 位有符号整数 |

| 索引基址 | aclsparseIndexBase_t | 说明 |
|----------|----------------------|------|
| ZERO | ACL_SPARSE_INDEX_BASE_ZERO | 0-based 索引，`indices[i]` 直接用作 Y 偏移 |
| ONE | ACL_SPARSE_INDEX_BASE_ONE | 1-based 索引，内部访问 Y 时减 1 |

> `vecX.valueType` 必须与 `vecY.valueType` 一致，否则返回 `ACL_SPARSE_STATUS_NOT_SUPPORTED`。

#### 约束说明

**shape 约束**：

- `vecX.nnz >= 0`、`vecX.size >= 0`、`vecY.nums >= 0`。
- `vecX.size <= vecY.nums`（稀疏向量大小不得超过稠密向量长度），违反返回 `ACL_SPARSE_STATUS_INVALID_VALUE`。
- `vecX.nnz <= vecX.size`（非零元素数不得超过向量大小），违反返回 `ACL_SPARSE_STATUS_INVALID_VALUE`。
- `vecX.nnz` 不得超过 `UINT32_MAX`（受多核切分参数类型限制），违反返回 `ACL_SPARSE_STATUS_NOT_SUPPORTED`。

**dtype 约束**：

- `vecX.valueType` 必须为 `ACL_FLOAT` / `ACL_FLOAT16` / `ACL_BF16` 之一，否则返回 `ACL_SPARSE_STATUS_NOT_SUPPORTED`。
- `vecX.idxType` 必须为 `ACL_SPARSE_INDEX_32I` / `ACL_SPARSE_INDEX_64I` 之一，否则返回 `ACL_SPARSE_STATUS_NOT_SUPPORTED`。
- `vecX.idxBase` 必须为 `ACL_SPARSE_INDEX_BASE_ZERO` / `ACL_SPARSE_INDEX_BASE_ONE` 之一，否则返回 `ACL_SPARSE_STATUS_INVALID_VALUE`。
- `vecX.valueType == vecY.valueType`，否则返回 `ACL_SPARSE_STATUS_NOT_SUPPORTED`。

**指针约束**：

- `handle` 不可为 nullptr，否则返回 `ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR`；且须先调用 `aclsparseSetStream` 绑定 stream，否则返回 `ACL_SPARSE_STATUS_INVALID_VALUE`。
- `vecX`、`vecY` 描述符不可为 nullptr，否则返回 `ACL_SPARSE_STATUS_INVALID_VALUE`。
- `nnz > 0` 时，`vecX.indices`、`vecX.values`、`vecY.values` 均不可为 nullptr，否则返回 `ACL_SPARSE_STATUS_INVALID_VALUE`。
- `nnz == 0` 时，不访问数据指针，快速返回成功。

**对齐约束**：`vecX.indices`、`vecX.values`、`vecY.values` 需对齐到 16 字节（对标 cuSPARSE），由描述符创建侧保证，kernel 侧不校验。

**输入数据契约（调用方前置条件）**：`vecX.indices[i] - idxBase ∈ [0, vecY.nums)`。算子不对索引值做运行时边界校验（host 侧无法读取 device 内存），越界访问行为未定义，由调用方承担后果。与 cuSPARSE 行为一致。

**索引重复**：`indices` 中存在重复值时，多个写入竞争同一位置，最终值取决于线程调度顺序，**行为不确定**（last-write-wins）。如需确定性结果，调用方应保证 `indices` 互异。

**异步执行**：函数异步执行，kernel 入队到 `handle` 关联的 stream 后立即返回，结果可能尚未就绪。调用后需通过 `aclrtSynchronizeStream(stream)` 同步后再读取 `vecY.values`。

**workspace**：本算子无需 workspace，无额外 Device 内存分配。

**平台约束**：Kernel 使用 SIMT（`__simt_vf__` / `asc_vf_call`）线程级并行，目标架构为 arch35 / Ascend950，其他芯片不支持。

#### 返回值

| 返回值 | 含义 |
|--------|------|
| ACL_SPARSE_STATUS_SUCCESS | 执行成功（含 `nnz == 0` 快速返回路径） |
| ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR | `handle` 为 nullptr |
| ACL_SPARSE_STATUS_INVALID_VALUE | `vecX` / `vecY` 为 nullptr；`idxBase` 枚举非法；`vecX.size > vecY.nums`；`vecX.nnz > vecX.size`；`nnz > 0` 时 `indices` / `values` / `vecY.values` 为 nullptr；stream 未绑定 |
| ACL_SPARSE_STATUS_NOT_SUPPORTED | `valueType` 不在 {FP32, FP16, BF16} 范围；`idxType` 不在 {I32, I64} 范围；`vecX.valueType != vecY.valueType`；`nnz > UINT32_MAX` |
| ACL_SPARSE_STATUS_INTERNAL_ERROR | `GetAivCoreCount()` 返回 0（设备异常） |

#### 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../docs/zh/develop/compile_and_run_example.md)。

```cpp
#include <cstdint>
#include <cstdio>
#include <memory>
#include <vector>

#include "acl/acl.h"
#include "cann_ops_sparse.h"

#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)

#define LOG_PRINT(message, ...)         \
    do {                                \
        printf(message, ##__VA_ARGS__); \
    } while (0)

class AclContext {
public:
    explicit AclContext(int32_t deviceId) : deviceId_(deviceId) {}

    ~AclContext()
    {
        if (stream_ != nullptr) {
            aclrtDestroyStream(stream_);
            stream_ = nullptr;
        }
        if (deviceSet_) {
            aclrtResetDevice(deviceId_);
            deviceSet_ = false;
        }
        if (aclInited_) {
            aclFinalize();
            aclInited_ = false;
        }
    }

    int Init()
    {
        auto ret = aclInit(nullptr);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
        aclInited_ = true;

        ret = aclrtSetDevice(deviceId_);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
        deviceSet_ = true;

        ret = aclrtCreateStream(&stream_);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
        return ACL_SUCCESS;
    }

    aclrtStream Stream() const { return stream_; }

private:
    int32_t deviceId_;
    aclrtStream stream_ = nullptr;
    bool aclInited_ = false;
    bool deviceSet_ = false;
};

// Device 内存智能指针：带 aclrtFree deleter，离开作用域自动释放
using DevicePtr = std::unique_ptr<void, aclError (*)(void *)>;

// 描述符智能指针：带 Destroy deleter，离开作用域自动销毁，错误路径不泄露
using ConstSpVecPtr = std::unique_ptr<const aclsparseSpVecDescr,
    aclsparseStatus_t (*)(aclsparseConstSpVecDescr_t)>;
using DnVecPtr = std::unique_ptr<aclsparseDnVecDescr,
    aclsparseStatus_t (*)(aclsparseConstDnVecDescr_t)>;

// 辅助：分配 Device 内存，失败返回空指针
// ACL_MEM_MALLOC_HUGE_FIRST 优先使用大页内存（HugePage），可提升 HBM 读写带宽
static DevicePtr MallocDevice(size_t sizeBytes)
{
    void *rawPtr = nullptr;
    auto ret = aclrtMalloc(&rawPtr, sizeBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret);
              return DevicePtr(nullptr, aclrtFree));
    return DevicePtr(rawPtr, aclrtFree);
}

// 辅助：分配 Device 内存并拷贝 Host 数据，失败返回空指针
static DevicePtr AllocAndCopyDevice(const void *hostPtr, size_t sizeBytes)
{
    DevicePtr devPtr = MallocDevice(sizeBytes);
    CHECK_RET(devPtr != nullptr, return DevicePtr(nullptr, aclrtFree));
    if (hostPtr != nullptr && sizeBytes > 0) {
        auto ret = aclrtMemcpy(devPtr.get(), sizeBytes, hostPtr, sizeBytes, ACL_MEMCPY_HOST_TO_DEVICE);
        CHECK_RET(ret == ACL_SUCCESS,
                  LOG_PRINT("aclrtMemcpy H2D failed. ERROR: %d\n", ret);
                  return DevicePtr(nullptr, aclrtFree));
    }
    return devPtr;
}

int aclsparseScatterTest(AclContext &ctx)
{
    aclrtStream stream = ctx.Stream();

    // 1. 创建 ops-sparse 句柄
    aclsparseHandle_t rawHandle = nullptr;
    auto sparseRet = aclsparseCreate(&rawHandle);
    CHECK_RET(sparseRet == ACL_SPARSE_STATUS_SUCCESS, LOG_PRINT("aclsparseCreate failed: %d\n", sparseRet);
              return sparseRet);
    std::unique_ptr<aclsparseContext, aclsparseStatus_t (*)(aclsparseHandle_t)> handlePtr(rawHandle, aclsparseDestroy);
    sparseRet = aclsparseSetStream(handlePtr.get(), stream);
    CHECK_RET(sparseRet == ACL_SPARSE_STATUS_SUCCESS, LOG_PRINT("aclsparseSetStream failed: %d\n", sparseRet);
              return sparseRet);

    // 2. 准备 Host 端数据
    //    稀疏向量 X: size=8, nnz=4, FP32, I32, 0-based 索引
    //    稠密向量 Y: nums=8, FP32
    //
    //    X.indices = [0, 2, 5, 7]
    //    X.values  = [1.0, 2.0, 3.0, 4.0]
    //    Y 初始值  = [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0]
    //
    //    scatter 后预期 Y:
    //    Y[0]=1.0, Y[2]=2.0, Y[5]=3.0, Y[7]=4.0，其余位置保持 10.0
    //    Y = [1.0, 10.0, 2.0, 10.0, 10.0, 3.0, 10.0, 4.0]
    int64_t size  = 8;
    int64_t nnz   = 4;
    int64_t nums  = 8;

    std::vector<int32_t> hIndices = {0, 2, 5, 7};
    std::vector<float>   hXValues = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float>   hYValues = {10.0f, 10.0f, 10.0f, 10.0f, 10.0f, 10.0f, 10.0f, 10.0f};

    // 3. 拷贝数据到 Device（DevicePtr 自动管理释放，错误路径不泄露）
    DevicePtr dIndices = AllocAndCopyDevice(hIndices.data(), nnz * sizeof(int32_t));
    CHECK_RET(dIndices != nullptr, LOG_PRINT("AllocAndCopyDevice dIndices failed\n");
              return ACL_SPARSE_STATUS_INTERNAL_ERROR);
    DevicePtr dXValues = AllocAndCopyDevice(hXValues.data(), nnz * sizeof(float));
    CHECK_RET(dXValues != nullptr, LOG_PRINT("AllocAndCopyDevice dXValues failed\n");
              return ACL_SPARSE_STATUS_INTERNAL_ERROR);
    DevicePtr dYValues = AllocAndCopyDevice(hYValues.data(), nums * sizeof(float));
    CHECK_RET(dYValues != nullptr, LOG_PRINT("AllocAndCopyDevice dYValues failed\n");
              return ACL_SPARSE_STATUS_INTERNAL_ERROR);

    // 4. 创建稀疏向量描述符 vecX（const，只读）和稠密向量描述符 vecY（输出）
    //    描述符由 ConstSpVecPtr / DnVecPtr RAII 管理，CHECK_RET 提前返回时自动销毁
    aclsparseConstSpVecDescr_t vecX = nullptr;
    sparseRet = aclsparseCreateConstSpVec(&vecX, size, nnz, dIndices.get(), dXValues.get(),
                                          ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_BASE_ZERO, ACL_FLOAT);
    CHECK_RET(sparseRet == ACL_SPARSE_STATUS_SUCCESS, LOG_PRINT("aclsparseCreateConstSpVec failed: %d\n", sparseRet);
              return sparseRet);
    ConstSpVecPtr vecXPtr(vecX, aclsparseDestroySpVec);

    aclsparseDnVecDescr_t vecY = nullptr;
    sparseRet = aclsparseCreateDnVec(&vecY, nums, dYValues.get(), ACL_FLOAT);
    CHECK_RET(sparseRet == ACL_SPARSE_STATUS_SUCCESS, LOG_PRINT("aclsparseCreateDnVec failed: %d\n", sparseRet);
              return sparseRet);
    DnVecPtr vecYPtr(vecY, aclsparseDestroyDnVec);

    // 5. 执行 scatter：Y[X.indices[i]] = X.values[i]
    sparseRet = aclsparseScatter(handlePtr.get(), vecXPtr.get(), vecYPtr.get());
    CHECK_RET(sparseRet == ACL_SPARSE_STATUS_SUCCESS, LOG_PRINT("aclsparseScatter failed: %d\n", sparseRet);
              return sparseRet);

    // 6. 同步等待计算完成（异步语义，调用方负责同步）
    auto aclRet = aclrtSynchronizeStream(stream);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", aclRet);
              return ACL_SPARSE_STATUS_EXECUTION_FAILED);

    // 7. 将结果拷贝回 Host 并打印
    std::vector<float> hResult(nums, 0.0f);
    aclRet = aclrtMemcpy(hResult.data(), nums * sizeof(float), dYValues.get(), nums * sizeof(float),
                         ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy D2H failed. ERROR: %d\n", aclRet);
              return ACL_SPARSE_STATUS_EXECUTION_FAILED);

    LOG_PRINT("\nResult Y after scatter:\n  ");
    for (int64_t i = 0; i < nums; i++) {
        LOG_PRINT("%.1f ", hResult[i]);
    }
    LOG_PRINT("\n");

    // 8. 清理：描述符由 vecXPtr / vecYPtr 自动销毁，句柄由 handlePtr 自动销毁，
    //    Device 内存由 DevicePtr 自动释放
    return ACL_SPARSE_STATUS_SUCCESS;
}

int main()
{
    AclContext ctx(0);
    auto ret = ctx.Init();
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = aclsparseScatterTest(ctx);
    CHECK_RET(ret == ACL_SPARSE_STATUS_SUCCESS, LOG_PRINT("aclsparseScatterTest failed: %d\n", ret); return ret);
    return 0;
}
```

预期输出如下（Ascend950 示例值）：

```
Result Y after scatter:
  1.0 10.0 2.0 10.0 10.0 3.0 10.0 4.0 
```

> 输出展示了 scatter 语义：`indices=[0,2,5,7]` 指定的位置被 `values=[1,2,3,4]` 覆盖，其余位置保持初始值 `10.0` 不变。

#### 对标说明

本算子对标 cuSPARSE `cusparseScatter` 接口，对齐其语义与行为约定：

| 对标维度 | cuSPARSE `cusparseScatter` | 本算子 `aclsparseScatter` |
|----------|----------------------------|---------------------------|
| 接口风格 | Generic API（描述符模式：SpVec + DnVec） | Generic API（描述符模式：SpVec + DnVec） |
| 核心语义 | `Y[X.indices[i]] = X.values[i]` | `Y[X.indices[i]] = X.values[i]` |
| idxBase 处理 | ONE 时内部减 1 | ONE 时内部减 1（统一用 `indices[i] - idxBase`） |
| 索引排序 | 允许未排序 | 允许未排序 |
| 索引重复 | last-write-wins，行为不确定 | last-write-wins，行为不确定 |
| 索引越界 | 不做运行时校验，行为未定义 | 不做运行时校验，行为未定义 |
| nnz = 0 | 无操作 | 快速返回，不启动 kernel |
| 异步执行 | 异步，调用方同步 | 异步，调用方同步 |
| workspace | 无 | 无 |
| 值类型 | FP32 / FP64 / FP16 / BF16 / INT8 / Complex 等 | FP32 / FP16 / BF16（核心子集，FP64/INT8/Complex 暂不实现） |
| 索引类型 | I32 / I64 | I32 / I64 |

**差异说明**：

- **值类型范围**：本算子当前实现核心子集 FP32 / FP16 / BF16（需求文档 §2.4 确认），cuSPARSE 支持的 FP64 / INT8 / Complex64 / Complex128 暂不实现，后续按需扩展。
- **实现架构**：cuSPARSE 基于 CUDA thread 级并行实现；本算子基于 Ascend C SIMT（`__simt_vf__` + grid-stride loop）线程级并行实现，语义等价。
- **精度**：scatter 为纯数据搬运（无计算），本算子输出与 `X.values` 逐位一致（bitwise exact），与 cuSPARSE 行为一致。
