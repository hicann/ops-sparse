# Gather 算子

## 算子概述

Gather 算子实现从稠密向量中按稀疏向量的索引数组收集元素，属于 Generic API 体系下的稀疏向量-稠密向量基本运算。

输出结果 X 满足：

```
X.values[i] = Y[X.indices[i] - idxBase]   for i = 0 .. nnz-1
```

其中 `X` 为稀疏向量（Sparse Vector），包含 `indices`（非零元素在稠密向量中的位置）和 `values`（非零元素的值）；`Y` 为稠密向量（Dense Vector）。调用后 `X.values` 被覆写为从 `Y` 中收集到的值。

该算子不需要 workspace，也不需要 preprocess 阶段，单步调用即可完成计算。

## 算子执行接口

### aclsparseGather

#### 产品支持情况

- Ascend 950PR / Ascend 950DT：支持
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品：不支持
- Atlas A2 训练系列产品 / Atlas A2 推理系列产品：不支持

#### 函数原型

```cpp
aclsparseStatus_t aclsparseGather(
    aclsparseHandle_t handle,
    aclsparseConstDnVecDescr_t vecY,
    aclsparseSpVecDescr_t vecX)
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 | 内存位置 |
|--------|----------|---------|------|---------|
| handle | 输入 | aclsparseHandle_t | ops-sparse 库上下文句柄，携带 stream，须先调用 `aclsparseSetStream` | Host |
| vecY | 输入 | aclsparseConstDnVecDescr_t | 稠密向量 Y 的描述符（数据源），仅读取 values | Host |
| vecX | 输出 | aclsparseSpVecDescr_t | 稀疏向量 X 的描述符：读取 indices 和 idxBase，写入 values | Host |

#### 约束说明

- handle 不可为 nullptr，否则返回 `ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR`
- vecY、vecX 不可为 nullptr，否则返回 `ACL_SPARSE_STATUS_INVALID_VALUE`
- vecX.valueType 与 vecY.valueType 必须一致
- vecX.size（稀疏向量的稠密维度）<= vecY.size（稠密向量的大小）
- vecX.indices 中每个索引值 `X.indices[i] - idxBase` 必须在 `[0, vecY.size)` 范围内（运行时不做边界检查，越界访问将导致未定义行为）
- 不需要 workspace，不需要预处理阶段
- 支持索引乱序（indices 不需要排序）
- 支持 vecX.indices 中存在重复元素

#### 支持的数据类型

| 数据类型 | 枚举值 | 支持 |
|---------|--------|------|
| FP32 | `ACL_FLOAT` | ✅ |
| FP16 | `ACL_FLOAT16` | ✅ |
| BF16 | `ACL_BF16` | ✅ |
| FP64 | `ACL_DOUBLE` | ✅ |

#### 支持的索引类型

| 索引类型 | 枚举值 | 支持 |
|---------|--------|------|
| 32 位有符号整数 | `ACL_SPARSE_INDEX_32I` | ✅ |
| 64 位有符号整数 | `ACL_SPARSE_INDEX_64I` | ✅ |

#### 索引基址

| 基址 | 枚举值 | 说明 |
|------|--------|------|
| 0-based | `ACL_SPARSE_INDEX_BASE_ZERO` | C 兼容（默认） |
| 1-based | `ACL_SPARSE_INDEX_BASE_ONE` | Fortran 兼容 |

索引基址在创建稀疏向量描述符（`aclsparseCreateSpVec`）时指定，Gather 通过 `X.indices[i] - idxBase` 计算实际访问位置。

#### 特性说明

| 特性 | 支持 | 说明 |
|------|------|------|
| 额外 buffer | 不需要 | 无需 workspace 分配 |
| Preprocess | 不需要 | 无预处理阶段 |
| 确定性 | ✅ |每次调用结果 bit-wise 一致 |
| 索引乱序 | ✅ | indices 不要求排序 |
| 异步执行 | ✅ | 调用后需 `aclrtSynchronizeStream` 等待完成 |

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../docs/zh/develop/compile_and_run_example.md)。

```cpp
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

using DevicePtr = std::unique_ptr<void, aclError (*)(void *)>;

static int AllocAndCopyDevice(DevicePtr &devicePtr, const void *hostPtr, size_t sizeBytes)
{
    void *rawPtr = nullptr;
    auto ret = aclrtMalloc(&rawPtr, sizeBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    devicePtr.reset(rawPtr);

    if (hostPtr != nullptr && sizeBytes > 0) {
        ret = aclrtMemcpy(devicePtr.get(), sizeBytes, hostPtr, sizeBytes, ACL_MEMCPY_HOST_TO_DEVICE);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy data to Device failed. ERROR: %d\n", ret); return ret);
    }
    return ACL_SUCCESS;
}

static int AllocDevice(DevicePtr &devicePtr, size_t sizeBytes)
{
    void *rawPtr = nullptr;
    auto ret = aclrtMalloc(&rawPtr, sizeBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    devicePtr.reset(rawPtr);
    return ACL_SUCCESS;
}

int aclsparseGatherTest(AclContext &ctx)
{
    aclrtStream stream = ctx.Stream();

    // 1. 创建 ops-sparse 句柄
    aclsparseHandle_t rawHandle = nullptr;
    auto sparseRet = aclsparseCreate(&rawHandle);
    CHECK_RET(sparseRet == ACL_SPARSE_STATUS_SUCCESS, LOG_PRINT("aclsparseCreate failed: %d\n", sparseRet);
              return sparseRet);
    std::unique_ptr<aclsparseContext, aclsparseStatus_t (*)(aclsparseHandle_t)> handlePtr(rawHandle, aclsparseDestroy);

    sparseRet = aclsparseSetStream(static_cast<aclsparseHandle_t>(handlePtr.get()), stream);
    CHECK_RET(sparseRet == ACL_SPARSE_STATUS_SUCCESS, LOG_PRINT("aclsparseSetStream failed: %d\n", sparseRet);
              return sparseRet);

    // 2. 准备 Host 端数据
    //    稠密向量 Y = {10.0, 20.0, 30.0, 40.0, 50.0}
    //    稀疏向量 X.indices = {0, 2, 4}（0-based），nnz = 3
    //    期望输出 X.values = {10.0, 30.0, 50.0}
    int64_t vecSize = 5;
    int64_t nnz = 3;

    std::vector<float> hY = {10.0f, 20.0f, 30.0f, 40.0f, 50.0f};
    std::vector<int32_t> hIndices = {0, 2, 4};

    // 3. 拷贝数据到 Device
    DevicePtr dY(nullptr, aclrtFree);
    DevicePtr dIndices(nullptr, aclrtFree);
    DevicePtr dXValues(nullptr, aclrtFree);

    auto aclRet = AllocAndCopyDevice(dY, hY.data(), vecSize * sizeof(float));
    CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);
    aclRet = AllocAndCopyDevice(dIndices, hIndices.data(), nnz * sizeof(int32_t));
    CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);
    aclRet = AllocDevice(dXValues, nnz * sizeof(float));
    CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);

    // 4. 创建稠密向量描述符（数据源 Y）
    aclsparseConstDnVecDescr_t dnVecY = nullptr;
    sparseRet = aclsparseCreateConstDnVec(&dnVecY, vecSize, dY.get(), ACL_FLOAT);
    CHECK_RET(sparseRet == ACL_SPARSE_STATUS_SUCCESS, LOG_PRINT("aclsparseCreateConstDnVec failed: %d\n", sparseRet);
              return sparseRet);

    // 5. 创建稀疏向量描述符（输出 X）
    aclsparseSpVecDescr_t spVecX = nullptr;
    sparseRet = aclsparseCreateSpVec(&spVecX, vecSize, nnz, dIndices.get(), dXValues.get(),
                                     ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_BASE_ZERO, ACL_FLOAT);
    CHECK_RET(sparseRet == ACL_SPARSE_STATUS_SUCCESS, LOG_PRINT("aclsparseCreateSpVec failed: %d\n", sparseRet);
              return sparseRet);

    // 6. 调用 Gather
    sparseRet = aclsparseGather(handlePtr.get(), dnVecY, spVecX);
    CHECK_RET(sparseRet == ACL_SPARSE_STATUS_SUCCESS, LOG_PRINT("aclsparseGather failed: %d\n", sparseRet);
              return sparseRet);

    // 7. 同步等待计算完成
    aclRet = aclrtSynchronizeStream(stream);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", aclRet); return aclRet);

    // 8. 将结果拷贝回 Host 并打印
    std::vector<float> hXValues(nnz, 0.0f);
    aclRet = aclrtMemcpy(hXValues.data(), nnz * sizeof(float), dXValues.get(), nnz * sizeof(float),
                         ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("copy X.values to Host failed. ERROR: %d\n", aclRet); return aclRet);

    LOG_PRINT("\nResult:\n");
    LOG_PRINT("  X.values: ");
    for (int64_t i = 0; i < nnz; i++) {
        LOG_PRINT("%.1f ", hXValues[i]);
    }
    LOG_PRINT("\n");

    // 9. 清理描述符
    aclsparseDestroySpVec(spVecX);
    aclsparseDestroyDnVec(dnVecY);

    return ACL_SPARSE_STATUS_SUCCESS;
}

int main()
{
    AclContext ctx(0);
    auto ret = ctx.Init();
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = aclsparseGatherTest(ctx);
    CHECK_RET(ret == ACL_SPARSE_STATUS_SUCCESS, LOG_PRINT("aclsparseGatherTest failed: %d\n", ret); return ret);
    return 0;
}
```

预期输出如下：

```
Result:
  X.values: 10.0 30.0 50.0
```
