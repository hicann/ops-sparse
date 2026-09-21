/**
 * ----------------------------------------------------------------------------------------------------------
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ----------------------------------------------------------------------------------------------------------
 */
#ifndef TEST_SPMMA_COMPRESS_NPU_WRAPPER_H
#define TEST_SPMMA_COMPRESS_NPU_WRAPPER_H
#include "spmma_compress_golden.h"
#include "cann_ops_sparseLt.h"
#include "descriptor_manager.h"
#include "sparse_test.h"
#include "verify.h"
#include <sstream>

namespace sparse_test {
inline void CompressCheck(aclsparseStatus_t status, const char* operation)
{
    if (status != ACL_SPARSE_STATUS_SUCCESS)
        throw std::runtime_error(std::string(operation) + " status=" + std::to_string(status));
}
inline void CompressAclCheck(aclError status, const char* operation)
{
    if (status != ACL_SUCCESS)
        throw std::runtime_error(std::string(operation) + " ACL status=" + std::to_string(status));
}
template <class T, auto Destroy>
struct CompressOwner {
    T value = nullptr;
    CompressOwner() = default;
    ~CompressOwner()
    {
        if (value)
            Destroy(&value);
    }
    CompressOwner(const CompressOwner&) = delete;
    CompressOwner& operator=(const CompressOwner&) = delete;
};
inline aclDataType CompressDtype(const CompressParam& p)
{
    if (p.dtype == "FP32")
        return ACL_FLOAT;
    if (p.dtype == "FP16")
        return ACL_FLOAT16;
    if (p.dtype == "BF16")
        return ACL_BF16;
    if (p.dtype == "INT8") {
        return ACL_INT8;
    }
    throw std::invalid_argument("unknown dtype");
}
class CompressPlan {
public:
    CompressOwner<aclsparseLtHandle_t, aclsparseLtDestroy> handle;
    CompressOwner<aclsparseLtMatDescriptor_t, aclsparseLtMatDescriptorDestroy> a;
    CompressOwner<aclsparseLtMatDescriptor_t, aclsparseLtMatDescriptorDestroy> b;
    CompressOwner<aclsparseLtMatDescriptor_t, aclsparseLtMatDescriptorDestroy> c;
    CompressOwner<aclsparseLtMatDescriptor_t, aclsparseLtMatDescriptorDestroy> d;
    CompressOwner<aclsparseLtMatmulDescriptor_t, aclsparseLtMatmulDescriptorDestroy> md;
    CompressOwner<aclsparseLtMatmulAlgSelection_t, aclsparseLtMatmulAlgSelectionDestroy> alg;
    CompressOwner<aclsparseLtMatmulPlan_t, aclsparseLtMatmulPlanDestroy> plan;

    explicit CompressPlan(const CompressParam& p)
    {
        CompressCheck(aclsparseLtInit(&handle.value), "LtInit");
        InitMatrices(p);
        const auto transpose = p.op == "T" ? ACL_SPARSE_OP_TRANSPOSE : ACL_SPARSE_OP_NON_TRANSPOSE;
        const auto compute = p.dtype == "INT8" ? ACL_SPARSE_COMPUTE_32I : ACL_SPARSE_COMPUTE_32F;
        CompressCheck(aclsparseLtMatmulDescriptorInit(&handle.value, &md.value,
            p.side == "A" ? transpose : ACL_SPARSE_OP_NON_TRANSPOSE,
            p.side == "B" ? transpose : ACL_SPARSE_OP_NON_TRANSPOSE, &a.value, &b.value, &c.value,
            &d.value, compute),
            "MatmulDescriptorInit");
        CompressCheck(
            aclsparseLtMatmulAlgSelectionInit(&handle.value, &alg.value, &md.value, ACL_SPARSE_LT_MATMUL_ALG_DEFAULT),
            "AlgInit");
        CompressCheck(aclsparseLtMatmulPlanInit(&handle.value, &plan.value, &md.value, &alg.value), "PlanInit");
    }

private:
    void SetBatches(aclsparseLtMatDescriptor_t* dst, int32_t batches)
    {
        CompressCheck(
            aclsparseLtMatDescSetAttribute(&handle.value, dst, ACLSPARSELT_MAT_NUM_BATCHES, &batches, sizeof(batches)),
            "SetBatches");
    }

    void InitDense(aclsparseLtMatDescriptor_t* dst, int64_t rows, int64_t cols, int64_t ld, aclsparseOrder_t layout,
        aclDataType dtype, int32_t batches)
    {
        CompressCheck(
            aclsparseLtDenseDescriptorInit(&handle.value, dst, rows, cols, ld, 16, dtype, layout), "DenseInit");
        SetBatches(dst, batches);
    }

    void InitSparse(aclsparseLtMatDescriptor_t* dst, const CompressParam& p)
    {
        const auto order = p.order == "ROW" ? ACL_SPARSE_ORDER_ROW : ACL_SPARSE_ORDER_COL;
        CompressCheck(aclsparseLtStructuredDescriptorInit(&handle.value, dst, p.rows, p.cols, p.ld, 16,
            CompressDtype(p), order, ACL_SPARSE_LT_SPARSITY_50_PERCENT),
            "StructuredInit");
        SetBatches(dst, p.batches);
        CompressCheck(aclsparseLtMatDescSetAttribute(
            &handle.value, dst, ACLSPARSELT_MAT_BATCH_STRIDE, &p.stride, sizeof(p.stride)),
            "SetStride");
    }

    void InitMatrices(const CompressParam& p)
    {
        const int64_t logicalRows = p.op == "N" ? p.rows : p.cols;
        const int64_t logicalCols = p.op == "N" ? p.cols : p.rows;
        const int64_t m = p.side == "A" ? logicalRows : 32;
        const int64_t k = p.side == "A" ? logicalCols : logicalRows;
        const int64_t n = p.side == "A" ? 32 : logicalCols;
        const auto dtype = CompressDtype(p);
        if (p.side == "A") {
            InitSparse(&a.value, p);
            InitDense(&b.value, k, n, k, ACL_SPARSE_ORDER_COL, dtype, p.batches);
        } else {
            InitDense(&a.value, m, k, k, ACL_SPARSE_ORDER_ROW, dtype, p.batches);
            InitSparse(&b.value, p);
        }
        const auto outputType = dtype == ACL_INT8 ? ACL_INT32 : dtype;
        InitDense(&c.value, m, n, n, ACL_SPARSE_ORDER_ROW, outputType, p.batches);
        InitDense(&d.value, m, n, n, ACL_SPARSE_ORDER_ROW, outputType, p.batches);
    }
};

class CompressEvent {
public:
    aclrtEvent value = nullptr;
    CompressEvent()
    {
        CompressAclCheck(aclrtCreateEvent(&value), "CreateEvent");
    }
    ~CompressEvent()
    {
        if (value) {
            aclrtDestroyEvent(value);
        }
    }
    CompressEvent(const CompressEvent&) = delete;
    CompressEvent& operator=(const CompressEvent&) = delete;
};
constexpr size_t COMPRESS_GUARD = 256;
class CompressBuffers {
public:
    std::vector<uint8_t> inputBefore;
    std::vector<uint8_t> outputBefore;
    DeviceBuffer input;
    DeviceBuffer output;
    CompressBuffers(const CompressParam& p, const std::vector<uint8_t>& dense)
        : inputBefore(dense.size() + 2 * COMPRESS_GUARD, 0xa5),
          outputBefore(p.compressedBytes() + 2 * COMPRESS_GUARD, 0x5a)
    {
        std::copy(dense.begin(), dense.end(), inputBefore.begin() + COMPRESS_GUARD);
        input = DeviceBuffer::copyFrom(inputBefore.data(), inputBefore.size());
        output = DeviceBuffer::copyFrom(outputBefore.data(), outputBefore.size());
    }
    void* dense()
    {
        return static_cast<uint8_t*>(input.get()) + COMPRESS_GUARD;
    }
    void* compressed()
    {
        return static_cast<uint8_t*>(output.get()) + COMPRESS_GUARD;
    }
};
inline bool CompressExact(
    const std::vector<uint8_t>& actual, const std::vector<uint8_t>& expected, const std::string& label)
{
    VerifyConfig cfg;
    cfg.mode = PrecisionMode::EXACT;
    // 每个字节一一映射为 [0,255] 内可精确表示的 float，复用精确比较器。
    // 按字节比较原始位模式，保留对 NaN 载荷位和零符号位差异的检查。
    constexpr size_t chunk = 65536;
    if (actual.size() != expected.size())
        return false;
    for (size_t start = 0; start < actual.size(); start += chunk) {
        const size_t end = std::min(start + chunk, actual.size());
        std::vector<float> a(actual.begin() + start, actual.begin() + end);
        std::vector<float> e(expected.begin() + start, expected.begin() + end);
        if (!Verifier::verifyVector(a, e, cfg, label + " byte_offset=" + std::to_string(start)))
            return false;
    }
    return true;
}
inline void CheckCompressResult(const CompressParam& p, CompressBuffers& buffers, const std::vector<uint8_t>& expected)
{
    std::vector<uint8_t> inputAfter(buffers.inputBefore.size());
    std::vector<uint8_t> outputAfter(buffers.outputBefore.size());
    buffers.input.copyToHost(inputAfter.data(), inputAfter.size());
    buffers.output.copyToHost(outputAfter.data(), outputAfter.size());
    if (!CompressExact(inputAfter, buffers.inputBefore, p.name + ": input/padding/guards"))
        throw std::runtime_error("input modified");
    auto guarded = buffers.outputBefore;
    if (p.pattern == "overfull") {
        std::copy(
            outputAfter.begin() + COMPRESS_GUARD, outputAfter.end() - COMPRESS_GUARD, guarded.begin() + COMPRESS_GUARD);
    } else
        std::copy(expected.begin(), expected.end(), guarded.begin() + COMPRESS_GUARD);
    if (!CompressExact(outputAfter, guarded, p.name + ": values/metadata/guards")) {
        const auto pos = std::mismatch(outputAfter.begin(), outputAfter.end(), guarded.begin()).first;
        const size_t offset = std::distance(outputAfter.begin(), pos);
        std::ostringstream error;
        error << "output mismatch allocation_byte=" << offset << " values_bytes=" << p.valuesBytes() << " got=0x"
              << std::hex << unsigned(outputAfter[offset]) << " expected=0x" << unsigned(guarded[offset]);
        throw std::runtime_error(error.str());
    }
}
inline void RunCompressCase(const CompressParam& p, aclrtStream explicitStream)
{
    CompressPlan plan(p);
    size_t compressed = 123;
    size_t workspace = 456;
    CompressCheck(aclsparseLtSpMMACompressedSize(&plan.handle.value, &plan.plan.value, &compressed, &workspace),
        "CompressedSize");
    if (compressed != p.compressedBytes() || workspace != 0)
        throw std::runtime_error("incorrect query sizes");
    const auto dense = MakeCompressInput(p);
    const auto golden = p.pattern == "overfull" ? std::vector<uint8_t>{} : CompressGolden(p, dense);
    CompressBuffers buffers(p, dense);
    aclrtStream stream = p.defaultStream ? nullptr : explicitStream;
    CompressCheck(aclsparseLtSpMMACompress(&plan.handle.value, &plan.plan.value, buffers.dense(), buffers.compressed(),
        p.workspaceSentinel ? reinterpret_cast<void*>(uintptr_t{16}) : nullptr, stream),
        "Compress");
    CompressAclCheck(aclrtSynchronizeStream(stream), "Synchronize");
    CheckCompressResult(p, buffers, golden);
}
} // namespace sparse_test
#endif
