# 快速入门：基于ops-sparse仓

## 使用须知

本指南旨在帮助您快速上手CANN和`ops-sparse`算子仓的使用。为方便快速了解算子开发全流程，将以**Spmv**算子为实践对象，其源文件位于`ops-sparse/src/spmv`，具体操作流程如下：

1. **[环境部署](zh/install/quick_install.md)**：完成软件包安装和源码下载，此处不再赘述。快速入门场景下，**推荐WebIDE或Docker环境**，安装操作简单。

   > **说明**：当前WebIDE或Docker环境默认最新商发版CANN包；如需体验master分支最新能力，可手动安装CANN包，注意软件与源码版本配套。

2. **[编译运行](#一编译运行)**：编译算子包并安装，实现快速调用算子。

3. **[算子开发](#二算子开发)**：通过修改现有算子Kernel，体验开发、编译、验证的完整闭环。

4. **[算子调试](#三算子调试)**：掌握算子打印和性能采集方法。

## 一、编译运行

本阶段目的是**快速体验项目标准流程**，验证环境能否成功进行算子源码编译、打包、安装和运行。

### 1. 编译Spmv算子

环境准备好后（注意软件与源码版本配套），进入环境并访问项目源码根目录，编译指定算子。

通用编译命令格式：`bash build.sh --pkg --soc=<芯片版本> --ops=<算子名>`。以Spmv算子为例，编译命令如下：

```bash
bash build.sh --pkg --soc=ascend910b --ops=spmv
```

若提示如下信息，说明编译成功。

```bash
Self-extractable archive "cann-ops-sparse-${cann_version}_linux-${arch}.run" successfully created.
```

编译成功后，run包存放于项目根目录的build_out目录下。

### 2. 安装Spmv算子包

```bash
./build_out/cann-ops-sparse-*linux*.run
```

### 3. 快速验证：运行算子样例

通用的运行命令格式：`bash build.sh --soc=<芯片版本> --ops=<算子名> --run`。

以Spmv为例，其提供了简单算子样例`test/spmv/spmv_test.cpp`，运行该样例验证算子功能是否正常。

```bash
bash build.sh --soc=ascend910b --ops=spmv --run
```

预期输出：打印算子`Spmv`的进度比对结果，表明算子已成功部署并正确执行。

```txt
[Success] test case accuracy is verification passed.
[PASS] spmv_test

========================================
Test Summary:
  Passed: 1 - spmv
  Failed: 0 -
========================================
```

## 二、算子开发

本阶段目的是对已成功运行的Spmv算子尝试**修改核函数代码**。

### 1. 修改Kernel实现

找到Spmv算子的核心kernel实现文件`sparse/src/spmv_kernel.cpp`，尝试修改算子中的process操作：

```cpp
__aicore__ inline void Process()
{
    for (uint64_t i = MAX_TILE_SIZE * id; i < rows; i += MAX_TILE_SIZE * blockNum) {
        uint64_t end = ((i + MAX_TILE_SIZE) > rows) ? rows : (i + MAX_TILE_SIZE);
        GlobalTensor<float> gm;
        // gm.SetGlobalBuffer(y + i, (end - i + 15) / 15 * 15);
        // 新增修改逻辑
        for (uint64_t j = i; j < end; j++) {
            uint32_t rstart = info->ptrs[j];
            uint32_t rend = info->ptrs[j + 1];
            float res = 0.0;
            for (uint32_t l = rstart; l < rend; l++) {
                res += info->values[l] * x[info->idxs[l]];
            }

            y[j] = res;
        }
        DataCacheCleanAndInvalid<float, AscendC::CacheLine::ENTIRE_DATA_CACHE, AscendC::DcciDst::CACHELINE_OUT>(gm);
    }
}
```

### 2. 编译与验证

重复[编译运行](#一编译运行)章节中的步骤：

1. **重新编译**：
    先回到项目根目录，编译命令如下：

    ```bash
    bash build.sh --pkg --soc=ascend910b --ops=spmv
    ```

2. **重新安装**：

    ```bash
    ./build_out/cann-ops-sparse-*linux*.run
    ```

3. **重新验证**：

    ```bash
    bash build.sh --soc=ascend910b --ops=spmv --run
    ```

4. **成功标志**：输出结果精度比对成功。

    ```txt
    [Success] test case accuracy is verification passed.
    [PASS] spmv_test

    ========================================
    Test Summary:
    Passed: 1 - spmv
    Failed: 0 - 
    ========================================
    ```

## 三、算子调试

本阶段以Spmv为例，在算子中添加打印并采集算子性能数据，以便后续问题分析定位。

### 1. 打印

算子如果出现执行失败、精度异常等问题，添加打印进行问题分析和定位。

请在`sparse/src/spmv_kernel.cpp`中进行代码修改。

* **printf**

  该接口支持打印Scalar类型数据，如整数、字符型、布尔型等，详细介绍请参见[《Ascend C API》](https://hiascend.com/document/redirect/CannCommunityAscendCApi)中“算子调测API > printf”。

  ```cpp
    __aicore__ inline void Init(
        GM_ADDR sync, GM_ADDR buffer, GM_ADDR x, GM_ADDR y, uint64_t rows, uint64_t cols, uint64_t nnz)
    {
        info = reinterpret_cast<__gm__ SpmvCsrInfo *>(buffer);
        blockNum = GetBlockNum();
        id = GetBlockIdx();
        this->x = (__gm__ float *)x;
        this->y = (__gm__ float *)y;
        this->rows = rows;
        this->cols = cols;
        this->nnz = nnz;
    }
    // 打印非零元素数量
    AscendC::PRINTF("this->nnz is %llu\n", this->nnz);
  ```

* **DumpTensor**

  该接口支持Dump指定Tensor的内容，同时支持打印自定义附加信息，比如当前行号等，详细介绍请参见[《Ascend C API》](https://hiascend.com/document/redirect/CannCommunityAscendCApi)中“算子调测API > DumpTensor”。

  ```cpp
    GlobalTensor<float> gm;
    gm.SetGlobalBuffer(y + i, (end - i + 15) / 15 * 15);
    for (uint64_t j = i; j < end; j++) {
        uint32_t rstart = info->ptrs[j];
        uint32_t rend = info->ptrs[j + 1];
        float res = 0.0;
        for (uint32_t l = rstart; l < rend; l++) {
            res += info->values[l] * x[info->idxs[l]];
        }

        y[j] = res;
    }
    // 打印gm Tensor信息
    DumpTensor(gm, 0, MAX_TILE_SIZE);
  ```

### 2. 性能采集

当算子功能验证正确后，可通过`msprof`工具采集算子性能数据。

- **生成可执行文件**

    调用Spmv算子的test样例，生成可执行文件（spmv_test），该文件位于项目`ops-sparse/build/test/spmv`目录。

    ```bash
    bash build.sh --soc=ascend910b --ops=spmv
    ```

- **采集性能数据**

    进入Spmv算子可执行文件目录`ops-sparse/build/test/spmv`，执行如下命令：

    ```bash
    msprof --application="./spmv_test"
    ```

采集结果在项目`ops-sparse/build/test/spmv`目录，msprof命令执行完后会自动解析并导出性能数据结果文件，详细内容请参见[msprof](https://www.hiascend.com/document/detail/zh/mindstudio/82RC1/T&ITools/Profiling/atlasprofiling_16_0110.html#ZH-CN_TOPIC_0000002504160251)。
