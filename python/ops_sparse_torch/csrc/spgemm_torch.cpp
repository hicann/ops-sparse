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

#include <ATen/ATen.h>
#include <ATen/ops/_sparse_coo_tensor_unsafe.h>
#include <c10/core/Scalar.h>
#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>
#include <torch/library.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>

#include <array>
#include <cstdint>
#include <limits>
#include <string>
#include <utility>

#include "cann_ops_sparse.h"
#include "securec.h"

namespace {

class HandleGuard {
public:
    HandleGuard()
    {
        Check(aclsparseCreate(&value_), "aclsparseCreate");
    }

    ~HandleGuard()
    {
        if (value_ != nullptr) {
            (void)aclsparseDestroy(value_);
        }
    }

    HandleGuard(const HandleGuard &) = delete;
    HandleGuard &operator=(const HandleGuard &) = delete;

    aclsparseHandle_t Get() const
    {
        return value_;
    }

    static void Check(aclsparseStatus_t status, const char *stage)
    {
        TORCH_CHECK(status == ACL_SPARSE_STATUS_SUCCESS, stage,
            " failed with aclsparse status ", static_cast<int>(status));
    }

private:
    aclsparseHandle_t value_ = nullptr;
};

class ConstSpMatGuard {
public:
    ~ConstSpMatGuard()
    {
        if (value_ != nullptr) {
            (void)aclsparseDestroySpMat(value_);
        }
    }

    aclsparseConstSpMatDescr_t *Out()
    {
        return &value_;
    }

    aclsparseConstSpMatDescr_t Get() const
    {
        return value_;
    }

private:
    aclsparseConstSpMatDescr_t value_ = nullptr;
};

class SpMatGuard {
public:
    ~SpMatGuard()
    {
        if (value_ != nullptr) {
            (void)aclsparseDestroySpMat(value_);
        }
    }

    aclsparseSpMatDescr_t *Out()
    {
        return &value_;
    }

    aclsparseSpMatDescr_t Get() const
    {
        return value_;
    }

private:
    aclsparseSpMatDescr_t value_ = nullptr;
};

class SpGemmDescrGuard {
public:
    SpGemmDescrGuard()
    {
        HandleGuard::Check(aclsparseSpGEMMCreateDescr(&value_),
            "aclsparseSpGEMMCreateDescr");
    }

    ~SpGemmDescrGuard()
    {
        if (value_ != nullptr) {
            (void)aclsparseSpGEMMDestroyDescr(value_);
        }
    }

    aclsparseSpGEMMDescr_t Get() const
    {
        return value_;
    }

private:
    aclsparseSpGEMMDescr_t value_ = nullptr;
};

aclDataType ToAclDataType(at::ScalarType type)
{
    switch (type) {
        case at::kHalf:
            return ACL_FLOAT16;
        case at::kBFloat16:
            return ACL_BF16;
        case at::kFloat:
            return ACL_FLOAT;
        case at::kComplexFloat:
            return ACL_COMPLEX64;
        default:
            TORCH_CHECK(false, "SpGEMM supports float16, bfloat16, float32 and complex64, got ", type);
    }
}

struct HostScalar {
    alignas(8) std::array<uint8_t, 8> bytes{};

    const void *Data() const
    {
        return bytes.data();
    }
};

template <typename T>
void StoreScalar(HostScalar &destination, const T &value)
{
    TORCH_CHECK(memcpy_s(destination.bytes.data(), destination.bytes.size(),
        &value, sizeof(value)) == EOK, "failed to store the SpGEMM host scalar");
}

HostScalar MakeScalar(const c10::Scalar &scalar, at::ScalarType type)
{
    HostScalar result{};
    const c10::complex<double> value = scalar.toComplexDouble();
    if (type != at::kComplexFloat) {
        TORCH_CHECK(value.imag() == 0.0, "real SpGEMM dtype cannot use a complex scalar");
    }
    switch (type) {
        case at::kHalf: {
            const c10::Half converted(static_cast<float>(value.real()));
            StoreScalar(result, converted);
            break;
        }
        case at::kBFloat16: {
            const c10::BFloat16 converted(static_cast<float>(value.real()));
            StoreScalar(result, converted);
            break;
        }
        case at::kFloat: {
            const float converted = static_cast<float>(value.real());
            StoreScalar(result, converted);
            break;
        }
        case at::kComplexFloat: {
            const aclsparseComplex converted{
                static_cast<float>(value.real()), static_cast<float>(value.imag())};
            StoreScalar(result, converted);
            break;
        }
        default:
            TORCH_CHECK(false, "unsupported scalar dtype");
    }
    return result;
}

void ValidateCsrOperand(const at::Tensor &tensor, const char *name)
{
    TORCH_CHECK(tensor.layout() == at::kSparseCsr, name, " must use sparse CSR layout");
    TORCH_CHECK(tensor.is_privateuseone(), name, " must be on an Ascend NPU");
    TORCH_CHECK(tensor.dim() == 2, name, " must be a two-dimensional sparse matrix");
    TORCH_CHECK(tensor.size(0) <= std::numeric_limits<int32_t>::max() &&
        tensor.size(1) <= std::numeric_limits<int32_t>::max() &&
        tensor._nnz() <= std::numeric_limits<int32_t>::max(),
        name, " exceeds the int32 CSR size limit");
}

at::Tensor ToCsrOnNpu(const at::Tensor &tensor, const char *name)
{
    if (tensor.layout() == at::kSparseCsr) {
        return tensor;
    }
    TORCH_CHECK(tensor.layout() == at::kSparse, name, " must use COO or CSR sparse layout");
    TORCH_CHECK(tensor.is_privateuseone(), name, " must be on an Ascend NPU");
    TORCH_CHECK(tensor.dim() == 2, name, " must be a two-dimensional sparse matrix");
    TORCH_CHECK(tensor.is_coalesced(), name,
        " must be coalesced before COO-to-CSR conversion on NPU");

    const int64_t rows = tensor.size(0);
    const auto indexOptions = tensor.values().options().dtype(at::kInt);
    at::Tensor indices = tensor._indices();
    at::Tensor row = indices.select(0, 0).to(at::kInt).contiguous();
    at::Tensor col = indices.select(0, 1).to(at::kInt).contiguous();
    at::Tensor counts = at::bincount(row, {}, rows).to(at::kInt);
    at::Tensor crow = at::cat({at::zeros({1}, indexOptions),
        at::cumsum(counts, 0, at::kInt)}, 0);
    return at::sparse_csr_tensor(crow, col, tensor.values().contiguous(),
        {rows, tensor.size(1)}, tensor.values().options().layout(at::kSparseCsr));
}

at::Tensor CsrToCooOnNpu(const at::Tensor &tensor)
{
    const int64_t rows = tensor.size(0);
    const int64_t nnz = tensor._nnz();
    at::Tensor crow = tensor.crow_indices();
    at::Tensor lengths = (crow.slice(0, 1) - crow.slice(0, 0, rows)).to(at::kLong);
    at::Tensor rowIds = at::repeat_interleave(
        at::arange(rows, tensor.values().options().dtype(at::kLong)), lengths, 0, nnz);
    at::Tensor colIds = tensor.col_indices().to(at::kLong);
    at::Tensor indices = at::stack({rowIds, colIds}, 0);
    return at::_sparse_coo_tensor_unsafe(indices, tensor.values(),
        {rows, tensor.size(1)}, tensor.values().options().layout(at::kSparse), true);
}

void ValidateSpGemmInputs(const at::Tensor &mat1, const at::Tensor &mat2)
{
    ValidateCsrOperand(mat1, "mat1");
    ValidateCsrOperand(mat2, "mat2");
    TORCH_CHECK(mat1.device() == mat2.device(), "mat1 and mat2 must be on the same NPU");
    TORCH_CHECK(mat1.scalar_type() == mat2.scalar_type(),
        "mat1 and mat2 must have the same dtype");
    TORCH_CHECK(mat1.size(1) == mat2.size(0),
        "mat1 and mat2 shapes cannot be multiplied (", mat1.sizes(), " and ",
        mat2.sizes(), ")");
}

struct PreparedSpGemm {
    int64_t m = 0;
    int64_t k = 0;
    int64_t n = 0;
    aclDataType aclType = ACL_FLOAT;
    HostScalar alpha;
    HostScalar beta;
    at::Tensor aCrow;
    at::Tensor aCol;
    at::Tensor aValues;
    at::Tensor bCrow;
    at::Tensor bCol;
    at::Tensor bValues;
    at::Tensor cCrow;
};

PreparedSpGemm PrepareSpGemm(
    const at::Tensor &mat1, const at::Tensor &mat2,
    const c10::Scalar &alphaScalar)
{
    PreparedSpGemm prepared;
    prepared.m = mat1.size(0);
    prepared.k = mat1.size(1);
    prepared.n = mat2.size(1);
    const at::ScalarType scalarType = mat1.scalar_type();
    prepared.aclType = ToAclDataType(scalarType);
    prepared.alpha = MakeScalar(alphaScalar, scalarType);
    prepared.beta = MakeScalar(c10::Scalar(0.0), scalarType);
    // PyTorch accepts int64 sparse indices; the public aclsparse contract is int32.
    // Keep the conversion on NPU and make every array contiguous before exposing a pointer.
    prepared.aCrow = mat1.crow_indices().to(at::kInt).contiguous();
    prepared.aCol = mat1.col_indices().to(at::kInt).contiguous();
    prepared.aValues = mat1.values().contiguous();
    prepared.bCrow = mat2.crow_indices().to(at::kInt).contiguous();
    prepared.bCol = mat2.col_indices().to(at::kInt).contiguous();
    prepared.bValues = mat2.values().contiguous();
    prepared.cCrow = at::empty(
        {prepared.m + 1}, prepared.aValues.options().dtype(at::kInt));
    return prepared;
}

void CreateSpGemmDescriptors(
    PreparedSpGemm &prepared, ConstSpMatGuard &matA,
    ConstSpMatGuard &matB, SpMatGuard &matC)
{
    HandleGuard::Check(aclsparseCreateConstCsr(
        matA.Out(), prepared.m, prepared.k, prepared.aValues.numel(),
        prepared.aCrow.const_data_ptr(), prepared.aCol.const_data_ptr(),
        prepared.aValues.const_data_ptr(),
        ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_32I,
        ACL_SPARSE_INDEX_BASE_ZERO, prepared.aclType), "aclsparseCreateConstCsr(matA)");
    HandleGuard::Check(aclsparseCreateConstCsr(
        matB.Out(), prepared.k, prepared.n, prepared.bValues.numel(),
        prepared.bCrow.const_data_ptr(), prepared.bCol.const_data_ptr(),
        prepared.bValues.const_data_ptr(),
        ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_32I,
        ACL_SPARSE_INDEX_BASE_ZERO, prepared.aclType), "aclsparseCreateConstCsr(matB)");
    HandleGuard::Check(aclsparseCreateCsr(
        matC.Out(), prepared.m, prepared.n, 0,
        prepared.cCrow.mutable_data_ptr(), nullptr, nullptr,
        ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_32I,
        ACL_SPARSE_INDEX_BASE_ZERO, prepared.aclType), "aclsparseCreateCsr(matC)");
}

struct SpGemmOutput {
    at::Tensor columns;
    at::Tensor values;
    at::Tensor workBuffer;
    at::Tensor computeBuffer;
};

SpGemmOutput CopySpGemmOutput(
    HandleGuard &handle, PreparedSpGemm &prepared,
    ConstSpMatGuard &matA, ConstSpMatGuard &matB, SpMatGuard &matC,
    SpGemmDescrGuard &descr, aclsparseOperation_t op, aclsparseSpGEMMAlg_t alg)
{
    int64_t rowsC = 0;
    int64_t colsC = 0;
    int64_t nnzC = 0;
    HandleGuard::Check(aclsparseSpMatGetSize(matC.Get(), &rowsC, &colsC, &nnzC),
        "aclsparseSpMatGetSize(matC)");
    TORCH_CHECK(rowsC == prepared.m && colsC == prepared.n && nnzC >= 0 &&
        nnzC <= std::numeric_limits<int32_t>::max(), "invalid SpGEMM output metadata");
    SpGemmOutput output{
        at::empty({nnzC}, prepared.aValues.options().dtype(at::kInt)),
        at::empty({nnzC}, prepared.aValues.options()), {}, {}};
    HandleGuard::Check(aclsparseCsrSetPointers(matC.Get(), prepared.cCrow.mutable_data_ptr(),
        output.columns.mutable_data_ptr(), output.values.mutable_data_ptr()),
        "aclsparseCsrSetPointers(matC)");
    HandleGuard::Check(aclsparseSpGEMMCopy(handle.Get(), op, op,
        prepared.alpha.Data(), matA.Get(), matB.Get(), prepared.beta.Data(), matC.Get(), prepared.aclType,
        alg, descr.Get()), "SpGEMM Copy");
    return output;
}

SpGemmOutput ExecuteSpGemm(
    HandleGuard &handle, PreparedSpGemm &prepared,
    ConstSpMatGuard &matA, ConstSpMatGuard &matB, SpMatGuard &matC)
{
    SpGemmDescrGuard descr;
    constexpr aclsparseOperation_t op = ACL_SPARSE_OP_NON_TRANSPOSE;
    constexpr aclsparseSpGEMMAlg_t alg = ACL_SPARSE_SPGEMM_DEFAULT;
    size_t bufferSize1 = 0;
    HandleGuard::Check(aclsparseSpGEMMWorkEstimation(handle.Get(), op, op,
        prepared.alpha.Data(), matA.Get(), matB.Get(), prepared.beta.Data(), matC.Get(), prepared.aclType,
        alg, descr.Get(), &bufferSize1, nullptr), "SpGEMM WorkEstimation query");
    TORCH_CHECK(bufferSize1 <= static_cast<size_t>(std::numeric_limits<int64_t>::max()),
        "SpGEMM work buffer exceeds PyTorch tensor limits");
    at::Tensor buffer1 = at::empty({static_cast<int64_t>(bufferSize1)},
        prepared.aValues.options().dtype(at::kByte));
    HandleGuard::Check(aclsparseSpGEMMWorkEstimation(handle.Get(), op, op,
        prepared.alpha.Data(), matA.Get(), matB.Get(), prepared.beta.Data(), matC.Get(), prepared.aclType,
        alg, descr.Get(), &bufferSize1, buffer1.mutable_data_ptr()),
        "SpGEMM WorkEstimation execute");
    size_t bufferSize2 = 0;
    size_t bufferSize3 = 0;
    HandleGuard::Check(aclsparseSpGEMMEstimateMemory(handle.Get(), op, op,
        prepared.alpha.Data(), matA.Get(), matB.Get(), prepared.beta.Data(), matC.Get(), prepared.aclType,
        alg, descr.Get(), 1.0F, &bufferSize3, nullptr, &bufferSize2),
        "SpGEMM EstimateMemory");
    TORCH_CHECK(bufferSize2 <= static_cast<size_t>(std::numeric_limits<int64_t>::max()),
        "SpGEMM compute buffer exceeds PyTorch tensor limits");
    at::Tensor buffer2 = at::empty({static_cast<int64_t>(bufferSize2)},
        prepared.aValues.options().dtype(at::kByte));
    HandleGuard::Check(aclsparseSpGEMMCompute(handle.Get(), op, op,
        prepared.alpha.Data(), matA.Get(), matB.Get(), prepared.beta.Data(), matC.Get(), prepared.aclType,
        alg, descr.Get(), &bufferSize2, buffer2.mutable_data_ptr()),
        "SpGEMM Compute");
    SpGemmOutput output = CopySpGemmOutput(handle, prepared, matA, matB, matC, descr, op, alg);
    output.workBuffer = std::move(buffer1);
    output.computeBuffer = std::move(buffer2);
    return output;
}

at::Tensor SpGemmCsr(
    const at::Tensor &mat1Input, const at::Tensor &mat2Input,
    const c10::Scalar &alphaScalar)
{
    ValidateSpGemmInputs(mat1Input, mat2Input);
    PreparedSpGemm prepared = PrepareSpGemm(mat1Input, mat2Input, alphaScalar);
    // stream(true) drains torch_npu's task queue before the direct aclsparse
    // launches and preserves ordering with the tensor conversions above.
    aclrtStream stream = c10_npu::getCurrentNPUStream(mat1Input.device().index()).stream(true);
    HandleGuard handle;
    HandleGuard::Check(aclsparseSetStream(handle.Get(), stream), "aclsparseSetStream");
    HandleGuard::Check(aclsparseSetPointerMode(handle.Get(), ACL_SPARSE_POINTER_MODE_HOST),
        "aclsparseSetPointerMode");
    ConstSpMatGuard matA;
    ConstSpMatGuard matB;
    SpMatGuard matC;
    CreateSpGemmDescriptors(prepared, matA, matB, matC);
    SpGemmOutput output = ExecuteSpGemm(handle, prepared, matA, matB, matC);
    return at::sparse_csr_tensor(prepared.cCrow, output.columns, output.values,
        {prepared.m, prepared.n}, output.values.options().layout(at::kSparseCsr));
}

at::Tensor SparseSparseMatmulCsr(const at::Tensor &mat1, const at::Tensor &mat2)
{
    return SpGemmCsr(mat1, mat2, c10::Scalar(1.0));
}

at::Tensor SparseSparseMatmulCoo(const at::Tensor &mat1, const at::Tensor &mat2)
{
    at::Tensor csr1 = ToCsrOnNpu(mat1, "mat1");
    at::Tensor csr2 = ToCsrOnNpu(mat2, "mat2");
    return CsrToCooOnNpu(SpGemmCsr(csr1, csr2, c10::Scalar(1.0)));
}

at::Tensor SparseAddmmCsr(
    const at::Tensor &self, const at::Tensor &mat1, const at::Tensor &mat2,
    const c10::Scalar &beta, const c10::Scalar &alpha)
{
    const c10::complex<double> betaValue = beta.toComplexDouble();
    TORCH_CHECK(betaValue.real() == 0.0 && betaValue.imag() == 0.0,
        "NPU sparse CSR addmm currently supports beta=0 only");
    TORCH_CHECK(self.layout() == at::kSparseCsr && self._nnz() == 0,
        "NPU sparse CSR addmm currently requires an empty sparse addend");
    return SpGemmCsr(mat1, mat2, alpha);
}

TORCH_LIBRARY_IMPL(aten, SparseCsrPrivateUse1, m)
{
    m.impl("_sparse_sparse_matmul", TORCH_FN(SparseSparseMatmulCsr));
    // torch.sparse.mm(CSR, CSR) in PyTorch 2.7 lowers through
    // _sparse_addmm with an empty CSR addend and beta=0.
    m.impl("_sparse_addmm", TORCH_FN(SparseAddmmCsr));
}

TORCH_LIBRARY_IMPL(aten, SparsePrivateUse1, m)
{
    m.impl("_sparse_sparse_matmul", TORCH_FN(SparseSparseMatmulCoo));
}

}  // namespace
