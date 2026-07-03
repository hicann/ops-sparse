---
name: sparse-ST-develop
description: ops-sparse 稀疏算子测试开发规范，定义测试框架结构、单层 Eigen golden 参考、NPU 封装、验证逻辑等。
---

# Sparse ST 测试开发规范

## 测试标杆体系（单层 Eigen Golden）

使用 **Eigen SparseMatrix** 作为唯一 CPU golden 参考：

```
                      ┌─────────────────────────────┐
                      │     NPU 算子执行             │
                      └──────────┬──────────────────┘
                                 │ NPU 结果
                                 ▼
                      ┌─────────────────────────────┐
                      │       精度验证               │
                      └──────────┬──────────────────┘
                                 │
                      ┌──────────▼──────────────────┐
                      │   Eigen Golden (FP64)        │
                      │   （唯一 CPU 参考基准）       │
                      └─────────────────────────────┘
```

| 实现 | 用途 | 依赖 |
|------|------|------|
| Eigen SparseMatrix<double> | 唯一 CPU golden 参考，与 cuSPARSE 结果对齐 | Eigen3 (header-only) |

### Eigen 作为测试标杆的原因

- **C++ 原生**：与测试代码同语言，无需跨语言调用
- **header-only**：无复杂构建依赖，CANN SDK 通常自带
- **SparseMatrix (CSR)**：直接对应 cuSPARSE 的 CSR 格式，语义对齐
- **社区公认**：广泛使用的科学计算库，精度可复现
- **FP64 精度**：使用 `double` 计算，避免精度损失，作为可靠基准

## 测试框架

ops-sparse 使用**自定义测试框架**（非 GTest），测试代码结构遵循本 SKILL 中定义的规范模板。
公共测试基础设施位于 `test/frame/`，每个算子开发必须优先复用这些公共头文件。

### 公共头文件（`test/frame/`）

| 头文件 | 职责 |
|--------|------|
| `types.h` | `PrecisionMode` 枚举（ABS/REL/MERE_MARE/EXACT/INTEGER）+ `VerifyConfig` 精度配置 + `CaseSummary` 统计 |
| `verify.h` | `Verifier` 精度比对类（策略模式：AbsStrategy / RelStrategy / MereMareStrategy / ExactStrategy / IntegerStrategy） |
| `fill.h` | `SparseFillGenerator` + `CsrMatrix/CooMatrix/CscMatrix` 结构体 + `makeSparseCsr/Coo/Csc/makeDense/ makeDenseFloat/makeDiagCsr/makeEmptyCsr` 快速填充函数 |
| `descriptor_manager.h` | Generic 描述符 RAII 封装：`SpMatManager/DnVecManager/DnMatManager/HandleManager/DeviceBuffer`，自动 Create/Destroy |
| `sparse_test.h` | `AclEnvScope` RAII 环境初始化 + `SparseTestParamBase` 参数结构体基类 + `aclDataTypeOf<T>()` 模板 + `SPARSE_CHECK_RET/SPARSE_LOG` 宏 |
| `csv_loader.h` | `csv_map` / `ReadMap` / `GetCasesFromCsv<ParamType>` / `parseBool/parseInt/parseFloat` 等 CSV 解析器 |

### 强制规则

- 所有新算子测试**必须**使用 `test/frame/` 公共头文件，禁止在算子测试代码中重新定义相同功能
- 精度配置必须使用 `VerifyConfig` + `Verifier`，禁止手写 `Verify()` 函数
- 描述符生命周期必须使用 RAII Manager（Handle/SpMat/DnVec/DnMat），禁止裸指针手动 destroy
- ACL 环境初始化必须使用 `AclEnvScope`，禁止手写 Init/Finalize
- 若新增通用功能，必须先补充到 `test/frame/` 对应头文件，再在算子测试中调用

### 算子级文件（每个算子独立）

| 文件 | 职责 | 位置 |
|------|------|------|
| `{op}_param.h` | 参数结构体，继承 `SparseTestParamBase`，实现 `fillCustom()` + `caseId()` | `test/{op}/` |
| `{op}_golden.h` | CPU golden 参考实现（使用 Eigen SparseMatrix<double> FP64） | `test/{op}/` |
| `{op}_npu_wrapper.h` | NPU 封装（描述符创建/销毁、kernel 调用、D2H 拷贝） | `test/{op}/archXX/` |
| `{op}_test.cpp` | GTest 入口，`::testing::TestWithParam<{Op}Param>` + `TEST_P` | `test/{op}/archXX/` |
| `{op}_test.csv` | CSV 用例表，列名=API 参数名 | `test/{op}/archXX/` |

> 每个算子新增时，必须在 `test/{op}/` 下创建 `{op}_param.h` + `{op}_golden.h`，在 `test/{op}/archXX/` 下创建 `{op}_npu_wrapper.h` + `{op}_test.cpp` + `{op}_test.csv`。

### 算子交付清单

#### 新算子（GTest + CSV 模式，spmv/spmm 除外）

```
test/{op_name}/
├── CMakeLists.txt              # 使用 ops_sparse_add_gtest_tests 宏注册算子
├── README.md                   # 算子测试说明（算子描述 + 测试覆盖情况 + 编译运行）
├── {op_name}_param.h           # 参数结构体，继承 SparseTestParamBase
├── {op_name}_golden.h          # CPU golden 参考（Eigen SparseMatrix<double> FP64）
└── archXX/
    ├── {op_name}_npu_wrapper.h # NPU 封装（描述符创建/销毁、kernel 调用、D2H 拷贝）
    ├── {op_name}_test.cpp      # GTest 测试入口（禁止定义 main 函数）
    └── {op_name}_test.csv      # CSV 用例表（基础列 + 算子自定义列）
```

**新算子交付标准**：

- [ ] `{op_name}_param.h` 参数结构体继承 `SparseTestParamBase`（csv_loader.h），实现 `fillCustom()` + `caseId()`
- [ ] `{op_name}_golden.h` 实现完整，使用 Eigen SparseMatrix<double> FP64 避免精度损失
- [ ] `{op_name}_npu_wrapper.h` 使用 RAII Manager（Handle/SpMat/DnVec/DnMat/DeviceBuffer），禁止裸指针
- [ ] `{op_name}_test.cpp` 使用 `test/frame/` 公共头文件（禁止手写 Verify / Init / descriptor destroy）
- [ ] `{op_name}_test.cpp` 使用 `::testing::TestWithParam<ParamType>` + `TEST_P`，**禁止定义 main 函数**
- [ ] `{op_name}_test.csv` 含基础列（m, n, sparsity, empty_row_prob, seed, expect_result）+ 算子自定义列
- [ ] CMakeLists.txt 通过 `ops_sparse_add_gtest_tests({op_name} ${OPS_SPARSE})` 注册（**不是** `ops_sparse_add_test`）
- [ ] 编译通过：`bash build.sh --ops={op_name}`
- [ ] ST 通过：`./{op_name}_test --gtest_filter=*` 至少一个用例可执行

#### 老算子（仅 spmv / spmm，保持现状）

```
test/{op_name}/
├── CMakeLists.txt              # 使用 ops_sparse_add_test 宏注册算子
├── README.md                   # 算子测试说明
└── archXX/
    └── {op_name}_test.cpp      # 自定义 main + TestRegistry 统计
```

**老算子交付标准**：

- [ ] `TestRegistry` 统计通过/失败用例
- [ ] CMakeLists.txt 通过 `ops_sparse_add_test({op_name} ${OPS_SPARSE})` 注册
- [ ] 编译通过：`bash build.sh --ops={op_name}`
- [ ] ST 通过：`bash build.sh --ops={op_name} --run`

### 测试代码组织（强制，新算子）

新算子测试入口必须使用 GTest + CSV 参数化 + `test/frame/` 框架，典型结构：

```cpp
#include <gtest/gtest.h>
#include "sparse_test.h"
#include "csv_loader.h"
#include "fill.h"
#include "verify.h"
#include "../spmv_param.h"
#include "../spmv_golden.h"
#include "spmv_npu_wrapper.h"

#include "acl/acl.h"
#include "cann_ops_sparse.h"

using namespace sparse_test;

// 1. GTest 测试夹具
class SpMVTest : public ::testing::TestWithParam<SpMVParam> {
public:
    static void SetUpTestSuite() {
        env_ = std::make_unique<AclEnvScope>();
        spHandle_ = std::make_unique<HandleManager>();
        spHandle_->setStream(env_->stream());
    }

    static void TearDownTestSuite() {
        spHandle_.reset();
        env_.reset();
    }

protected:
    inline static std::unique_ptr<AclEnvScope> env_;
    inline static std::unique_ptr<HandleManager> spHandle_;
};

// 2. TEST_P 测试用例
TEST_P(SpMVTest, GenericSuccess) {
    auto p = GetParam();
    PrintCaseInfoString(p);

    // 期望成功的用例
    ASSERT_EQ(p.expect_result, "ACL_SPARSE_STATUS_SUCCESS");

    // 1. 生成 CSR 数据（使用框架 fill.h）
    auto csr = makeSparseCsr(p.m, p.n, p.sparsity, p.seed);
    auto xVec = makeDenseFloat(p.n, -5.0, 10.0, p.seed + 1);
    auto yInit = makeDenseFloat(p.m, -5.0, 10.0, p.seed + 2);

    // 2. Eigen golden 作为唯一比对基准
    auto yGolden = SpMVGolden(csr, xVec, yInit, p.alpha, p.beta, p.transpose);

    // 3. NPU 调用（使用 npu_wrapper.h 封装）
    auto yNpu = SpMVNpuWrapper(*spHandle_, env_->stream(),
                                csr, xVec, yInit,
                                p.alpha, p.beta, p.transpose,
                                p.compute_type);

    // 4. 精度比对（使用框架 verify.h，阈值从 CSV 读取）
    VerifyConfig cfg;
    cfg.SetMode(PrecisionMode::MERE_MARE)
       .SetMERE(p.mere_threshold)
       .SetMARE(p.mare_multiplier * p.mere_threshold);
    EXPECT_TRUE(Verifier::verifyVector(yNpu, yGolden, cfg, p.caseId()));
}

// 3. 参数化实例化（从 CSV 加载用例）
INSTANTIATE_TEST_SUITE_P(
    SpMV,
    SpMVTest,
    ::testing::ValuesIn(GetCasesFromCsv<SpMVParam>("spmv_test.csv")),
    [](const ::testing::TestParamInfo<SpMVParam>& info) {
        return info.param.caseId();
    }
);

// 禁止定义 main 函数（由 test/frame/test_main.cpp 提供）
```

### Null Handle / 异常路径测试（TEST_F 模式）

对于 null handle、invalid descriptor 等异常路径测试，使用 `TEST_F` 而非 `TEST_P`，不下 CSV：

```cpp
class SpMVExceptionTest : public ::testing::Test {
protected:
    void SetUp() override {
        env_ = std::make_unique<AclEnvScope>();
        spHandle_ = std::make_unique<HandleManager>();
        spHandle_->setStream(env_->stream());
    }
    void TearDown() override {
        spHandle_.reset();
        env_.reset();
    }
    std::unique_ptr<AclEnvScope> env_;
    std::unique_ptr<HandleManager> spHandle_;
};

TEST_F(SpMVExceptionTest, NullHandle) {
    // 传入 nullptr handle，期望返回 ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR
    auto ret = aclsparseSpMV(nullptr, ACL_SPARSE_OP_NON_TRANSPOSE,
                              &alpha, matA.cget(), vecX.cget(),
                              &beta, vecY.get(), ACL_FLOAT,
                              ACL_SPARSE_SPMV_ALG_DEFAULT, nullptr);
    EXPECT_EQ(ret, ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR);
}

TEST_F(SpMVExceptionTest, InvalidSpMatDescr) {
    // 传入 nullptr matA，期望返回 ACL_SPARSE_STATUS_INVALID_VALUE
    auto ret = aclsparseSpMV(spHandle_->get(), ACL_SPARSE_OP_NON_TRANSPOSE,
                              &alpha, nullptr, vecX.cget(),
                              &beta, vecY.get(), ACL_FLOAT,
                              ACL_SPARSE_SPMV_ALG_DEFAULT, nullptr);
    EXPECT_EQ(ret, ACL_SPARSE_STATUS_INVALID_VALUE);
}

TEST_F(SpMVExceptionTest, NullDnVec) {
    // 传入 nullptr vecX，期望返回 ACL_SPARSE_STATUS_INVALID_VALUE
    auto ret = aclsparseSpMV(spHandle_->get(), ACL_SPARSE_OP_NON_TRANSPOSE,
                              &alpha, matA.cget(), nullptr,
                              &beta, vecY.get(), ACL_FLOAT,
                              ACL_SPARSE_SPMV_ALG_DEFAULT, nullptr);
    EXPECT_EQ(ret, ACL_SPARSE_STATUS_INVALID_VALUE);
}
```

### 测试代码组织（老算子，仅 spmv / spmm）

老算子保留自定义 main + TestRegistry 统计模式，典型结构：

```cpp
#include "sparse_test.h"
#include "fill.h"
#include "verify.h"
#include "descriptor_manager.h"

#include "acl/acl.h"
#include "cann_ops_sparse.h"

using namespace sparse_test;

static bool TestSpmv(const char* caseName, int64_t m, int64_t n, double sparsity,
                     float alpha, float beta, bool trans, aclrtStream stream) {
    std::cout << "==== " << caseName << " ==== m=" << m << " n=" << n
              << " sp=" << sparsity << " alpha=" << alpha << " beta=" << beta << "\n";

    // 1. 生成 CSR 数据（使用框架 fill.h）
    auto csr = makeSparseCsr(m, n, sparsity, 42);
    auto xVec = makeDenseFloat(n, -5.0, 10.0, 100);
    auto yInit = makeDenseFloat(m, -5.0, 10.0, 101);

    // 2. 自写 CPU golden（老算子保留）
    auto yGolden = SpmvCpu(csr, xVec, yInit, alpha, beta, trans);

    // 4. Device 内存（RAII）
    auto dRowPtr = DeviceBuffer::copyFrom(csr.rowOffsets.data(), csr.rowOffsets.size() * sizeof(int32_t));
    auto dColIdx = DeviceBuffer::copyFrom(csr.colIndices.data(), csr.colIndices.size() * sizeof(int32_t));
    auto dVals   = DeviceBuffer::copyFrom(csr.values.data(),      csr.values.size() * sizeof(float));
    auto dX      = DeviceBuffer::copyFrom(xVec.data(),            xVec.size() * sizeof(float));
    auto dY      = DeviceBuffer::copyFrom(yInit.data(),           yInit.size() * sizeof(float));

    // 5. 描述符（RAII）
    HandleManager handle;
    handle.setStream(stream);
    SpMatManager matA = SpMatManager::createConstCsr(m, n, csr.nnz,
        dRowPtr.get(), dColIdx.get(), dVals.get());
    DnVecManager vecX = DnVecManager::createConst(n, dX.get(), ACL_FLOAT);
    DnVecManager vecY = DnVecManager::create(m, dY.get(), ACL_FLOAT);

    // 6. 算子调用
    auto op = trans ? ACL_SPARSE_OP_TRANSPOSE : ACL_SPARSE_OP_NON_TRANSPOSE;
    auto ret = aclsparseSpMV(handle.get(), op, &alpha, matA.cget(), vecX.cget(),
                             &beta, vecY.get(), ACL_FLOAT,
                             ACL_SPARSE_SPMV_ALG_DEFAULT, nullptr);
    SPARSE_CHECK_RET(ret == ACL_SPARSE_STATUS_SUCCESS, return false);
    SPARSE_CHECK_RET(aclrtSynchronizeStream(stream) == ACL_SUCCESS, return false);

    // 7. D2H
    std::vector<float> yNpu(m);
    dY.copyToHost(yNpu.data(), yNpu.size() * sizeof(float));

    // 8. 精度比对（使用框架 verify.h）
    VerifyConfig cfg;
    cfg.SetMode(PrecisionMode::MERE_MARE).SetMERE(0.0001220703125).SetMARE(10.0);
    return Verifier::verifyVector(yNpu, yGolden, cfg, std::string(caseName));
}

int main() {
    AclEnvScope env;
    TestRegistry reg;

    reg.record("spmv-f32-default", TestSpmv("default-f32", 512, 1024, 0.9, 1.0f, 0.0f, false, env.stream()));
    // ... 更多用例 ...

    reg.printSummary();
    return reg.failCount() == 0 ? 0 : 1;
}
```

## Eigen Golden 参考实现

每个算子必须在 `test/{op_name}/{op_name}_golden.h` 中实现 Eigen golden 参考。以下是 SpMV / SpMM 样例，新增算子时可参照其 API 形状。

### SpMV Eigen Golden（CSR 格式）

```cpp
// test/{op_name}/{op_name}_golden.h
#pragma once

#include <Eigen/Eigen>
#include <Eigen/SparseCore>
#include "../frame/fill.h"

namespace sparse_test {

// SpMV 的 Eigen golden 实现
// 使用 Eigen::SparseMatrix<double> FP64 作为唯一 CPU 参考基准
template <typename T>
std::vector<T> SpMVGolden(
    const CsrMatrix& csr,
    const std::vector<T>& x,
    const std::vector<T>& yInit,
    T alpha, T beta, bool transpose)
{
    using Scalar = double;  // Eigen 用 FP64 避免精度损失

    int rows = static_cast<int>(csr.rows);
    int cols = static_cast<int>(csr.cols);
    int nnz = static_cast<int>(csr.nnz);

    // 1. 构建 Eigen SparseMatrix（CSR = RowMajor）
    Eigen::SparseMatrix<Scalar, Eigen::RowMajor> A(rows, cols);
    std::vector<Eigen::Triplet<Scalar>> triplets;
    triplets.reserve(nnz);
    for (int i = 0; i < rows; i++) {
        for (int j = csr.rowOffsets[i]; j < csr.rowOffsets[i + 1]; j++) {
            triplets.emplace_back(i, csr.colIndices[j], static_cast<Scalar>(csr.values[j]));
        }
    }
    A.setFromTriplets(triplets.begin(), triplets.end());

    // 2. 构建 Eigen Vector（转换为 FP64）
    Eigen::VectorXd xVec(cols), yVec(rows);
    for (int i = 0; i < cols; i++) xVec(i) = static_cast<Scalar>(x[i]);
    for (int i = 0; i < rows; i++) yVec(i) = static_cast<Scalar>(yInit[i]);

    // 3. 稀疏矩阵-向量乘法
    Eigen::VectorXd result;
    if (transpose) {
        result = alpha * (A.transpose() * xVec) + beta * yVec;
    } else {
        result = alpha * (A * xVec) + beta * yVec;
    }

    // 4. 转换回 T 类型
    std::vector<T> output(rows);
    for (int i = 0; i < rows; i++) {
        output[i] = static_cast<T>(result(i));
    }
    return output;
}

}  // namespace sparse_test
```

### SpMM Eigen Golden

```cpp
// test/{op_name}/{op_name}_golden.h (SpMM 部分)
namespace sparse_test {

// SpMM 的 Eigen golden 实现：C = alpha * A * B + beta * C
// A: sparse (m x k, CSR), B: dense (k x n), C: dense (m x n)
template <typename T>
std::vector<T> SpMMGolden(
    const CsrMatrix& A,
    const std::vector<T>& B,
    const std::vector<T>& C_init,
    int k, int n,
    T alpha, T beta,
    bool rowMajorB, bool rowMajorC)
{
    using Scalar = double;
    int m = static_cast<int>(A.rows);

    // 1. 构建 Eigen SparseMatrix
    Eigen::SparseMatrix<Scalar, Eigen::RowMajor> Amat(m, k);
    std::vector<Eigen::Triplet<Scalar>> triplets;
    for (int i = 0; i < m; i++)
        for (int j = A.rowOffsets[i]; j < A.rowOffsets[i + 1]; j++)
            triplets.emplace_back(i, A.colIndices[j], static_cast<Scalar>(A.values[j]));
    Amat.setFromTriplets(triplets.begin(), triplets.end());

    // 2. 构建 Eigen DenseMatrix（转换为 FP64）
    Eigen::MatrixXd Bmat(k, n), Cmat(m, n);
    for (int r = 0; r < k; r++)
        for (int c = 0; c < n; c++)
            Bmat(r, c) = rowMajorB ? B[r * n + c] : B[r + c * k];
    for (int r = 0; r < m; r++)
        for (int c = 0; c < n; c++)
            Cmat(r, c) = rowMajorC ? C_init[r * n + c] : C_init[r + c * m];

    // 3. 稀疏矩阵-稠密矩阵乘法
    Cmat = alpha * (Amat * Bmat) + beta * Cmat;

    // 4. 转换回 T 类型
    std::vector<T> output(m * n);
    for (int r = 0; r < m; r++)
        for (int c = 0; c < n; c++) {
            if (rowMajorC) output[r * n + c] = static_cast<T>(Cmat(r, c));
            else           output[r + c * m] = static_cast<T>(Cmat(r, c));
        }
    return output;
}

}  // namespace sparse_test
```

## 测试用例执行函数（完整流程）

> 该示例已在上方"测试代码组织"章节给出，使用 `test/frame/` 框架（`makeSparseCsr` / `DeviceBuffer` / `SpMatManager` / `Verifier` / `AclEnvScope`）。下方不再重复。

## NPU 封装规范

新算子测试**必须**使用框架 RAII Manager，禁止裸指针手写 `aclrtMalloc / aclrtFree / aclsparseDestroy*`。

### Generic API（使用 RAII）

通过 `test/frame/descriptor_manager.h` 的 `HandleManager/SpMatManager/DnVecManager/DnMatManager/DeviceBuffer` 自动管理生命周期：

```cpp
{
    // Device 内存（自动释放）
    auto dRowPtr = DeviceBuffer::copyFrom(csr.rowOffsets.data(), ...);
    auto dVals   = DeviceBuffer::copyFrom(csr.values.data(), ...);

    // 描述符（自动 Destroy）
    HandleManager handle;
    handle.setStream(stream);
    auto matA = SpMatManager::createConstCsr(m, n, nnz, dRowPtr.get(), ...);
    auto vecX = DnVecManager::createConst(n, dX.get(), ACL_FLOAT);
    auto vecY = DnVecManager::create(m, dY.get(), ACL_FLOAT);

    // 算子调用
    aclsparseSpMV(handle.get(), op, &alpha, matA.cget(), vecX.cget(),
                  &beta, vecY.get(), ACL_FLOAT, ACL_SPARSE_SPMV_ALG_DEFAULT, nullptr);

    // D2H
    std::vector<float> yNpu(m);
    dY.copyToHost(yNpu.data(), ...);
}  // exit scope → 自动 Destroy
```

### Legacy API（使用 DeviceBuffer + HandleManager）

Legacy API 无 SpMatDescr/DnVec/DnMat 描述符，使用 `DeviceBuffer` 管理内存，`HandleManager` 管理 handle：

```cpp
{
    auto dDl = DeviceBuffer::copyFrom(dl.data(), (m - 1) * sizeof(float));
    auto dD  = DeviceBuffer::copyFrom(d.data(),  m * sizeof(float));
    auto dDu = DeviceBuffer::copyFrom(du.data(), (m - 1) * sizeof(float));
    auto dX  = DeviceBuffer::copyFrom(xHost.data(), m * nrhs * sizeof(float));

    HandleManager handle;
    handle.setStream(stream);

    // Legacy 算子可能需要 MatDescr（手动创建/销毁，因 ops-sparse 当前未封装）
    // aclsparseMatDescr_t matDescr;
    // aclsparseCreateMatDescr(&matDescr);
    // ... aclsparseSetMatType / aclsparseSetMatIndexBase ...
    // aclsparseDestroyMatDescr(matDescr);

    aclsparseSgtsv2(handle.get(), m, dDl.get(), dD.get(), dDu.get(), dX.get(), nrhs, m, &bufferSize);
    aclrtSynchronizeStream(stream);

    std::vector<float> xNpu(m * nrhs);
    dX.copyToHost(xNpu.data(), xNpu.size() * sizeof(float));

    VerifyConfig cfg;
    cfg.SetMode(PrecisionMode::MERE_MARE).SetMERE(0.0001220703125).SetMARE(10.0);
    Verifier::verifyVector(xNpu, xGolden, cfg, caseName);
}
```

### Legacy API 注意事项

- **每种精度版本独立测试**：Legacy API 同一算子可能同时提供 S/D 版本，必须分别测试
- **无需 Generic 描述符**：不存在 SpMatDescr 创建/销毁，**但 MatDescr 若接口要求则必须创建（手动管理）**
- **workspace 处理**：部分 Legacy 算子通过输入参数 `pBufferSize` 返回大小（输入输出语义），需注意
- **错误处理**：所有步骤任一步骤失败，RAII 自动清理（DeviceBuffer 析构时释放、HandleManager 析构时 destroy）

## CMakeLists.txt 模板

### 新算子（GTest + CSV 模式，spmv/spmm 除外）

每个新算子的 `test/{op_name}/CMakeLists.txt` 只需一行：

```cmake
ops_sparse_add_gtest_tests({op_name} ${OPS_SPARSE})
```

- `ops_sparse_add_gtest_tests` 宏位于 `cmake/test.cmake`
- 自动链接 GTest + `test/frame/test_main.cpp`（共享 main 入口）
- 自动包含 `test/frame/` 作为 include 路径
- 自动查找 `test/{op_name}/archXX/{op_name}_test.cpp`（按 SOC_ARCH_DIRS 匹配）
- 自动拷贝 `test/{op_name}/archXX/{op_name}_test.csv` 到 build 目录
- 默认启用 Eigen（golden 参考），可通过 `TEST_USE_EIGEN=OFF` 关闭

### 老算子（仅 spmv / spmm，保持现状）

```cmake
ops_sparse_add_test({op_name} ${OPS_SPARSE})
```

- `ops_sparse_add_test` 宏位于 `cmake/test.cmake`
- 自动包含 `test/frame/` 作为 include 路径
- 自动查找 `test/{op_name}/archXX/{op_name}_test.cpp`（按 SOC_ARCH_DIRS 匹配）
- 不链接 GTest，不链接共享 test_main.cpp

## Eigen 精度容差规范

由于 Eigen 使用 FP64 计算，NPU 使用 T 类型，不同 dtype 的容差不同：

| NPU dtype | Eigen golden rtol | Eigen golden atol | 说明 |
|-----------|-------------------|-------------------|------|
| FP32 | 1e-5 | 1e-5 | FP32 标准 |
| FP16 | 1e-3 | 1e-3 | FP16 标准 |
| BF16 | 5e-3 | 5e-3 | BF16 标准 |
| INT32 | 0 (位精确) | 0 | 整数计算 |

## 测试用例设计指南

### L0 用例（基础功能验证）

| Shape | nnz | dtype | 格式 | 说明 |
|-------|-----|-------|------|------|
| 4×4 | 8 | FP32 | CSR | 基本方阵 |
| 8×16 | 24 | FP32 | CSR | 宽矩阵 |
| 16×8 | 24 | FP32 | CSR | 高矩阵 |

### L1 用例（覆盖 + 边界 + 大 shape）

| Shape | nnz | dtype | 格式 | 说明 |
|-------|-----|-------|------|------|
| 2048×2048 | 32768 | FP32 | CSR | 大 shape |
| 1×100 | 10 | FP32 | CSR | 单行 |
| 100×1 | 10 | FP32 | CSR | 单列 |
| 10×10 | 0 | FP32 | CSR | 空矩阵 |
| 64×64 | 128 | FP16 | CSR | 半精度 |
| 64×64 | 128 | FP32 | CSR(T) | 转置 |

## 常见问题排查

| 问题 | 症状 | 解决方案 |
|------|------|---------|
| Eigen header 找不到 | 编译报错 `fatal error: Eigen/Eigen: No such file or directory` | 检查 CMake 是否启用 `TEST_USE_EIGEN=ON`；确认 Eigen3 已安装（`find_package(Eigen3 3.3)`）；若无需 Eigen 可设 `TEST_USE_EIGEN=OFF` |
| CSV 路径错误 | 运行时报错 `ReadMap: cannot open xxx_test.csv` | 确认 CSV 文件已拷贝到 build 目录（`ops_sparse_add_gtest_tests` 宏自动处理）；检查相对路径是否正确 |
| `gtest_main` 冲突 | 链接报错 `multiple definition of main` | 确认 `{op}_test.cpp` 中**没有**定义 `main()` 函数（由 `test/frame/test_main.cpp` 提供） |
| 描述符泄漏 | 运行结束后内存泄漏警告 | 检查所有 `aclsparseCreate*` 是否有配对的 `aclsparseDestroy*`；推荐使用 RAII Manager（`SpMatManager`/`DnVecManager`/`DnMatManager`） |
| RAII 双重释放 | 运行崩溃 `double free or corruption` | 检查是否同时使用了 RAII Manager 和手动 `aclsparseDestroy*`；二者只能选其一 |
| arch 目录不匹配 | CMake 报错 `test arch 'archXX' has no matching src` | 确认 `SOC_VERSION` 与 `SOC_ARCH_DIRS` 匹配；检查 `src/{op}/archXX/` 和 `test/{op}/archXX/` 目录是否存在 |
| fill/golden 不匹配 | 精度验证失败，golden 结果与预期不符 | 检查 `fill.h` 生成的数据格式是否与 `golden.h` 期望的输入格式一致（如 CSR vs COO） |
| 精度 fail | `Verifier::verifyVector` 返回 false | 检查 `VerifyConfig` 的精度模式和阈值是否合理；对比 NPU 输出与 golden 的最大误差位置；参考 `ops-precision-standard` 技能 |
| 稀疏格式不支持 | 算子返回 `ACL_SPARSE_STATUS_NOT_SUPPORTED` | 检查传入的稀疏格式（CSR/COO/CSC）是否被算子支持；参考算子文档或 `cann_ops_sparse.h` 中的格式枚举 |
| null handle `spHandle_` 为空 | 测试崩溃 `Segmentation fault` | 检查 `SetUp()` 中 `spHandle_` 是否成功创建；确认 `HandleManager` 构造未抛异常 |
