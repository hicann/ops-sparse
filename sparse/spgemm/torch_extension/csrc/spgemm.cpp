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

#include "aclsparse_common.h"

#include <ATen/ops/_sparse_coo_tensor_unsafe.h>
#include <c10/core/Scalar.h>
#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>
#include <pybind11/complex.h>
#include <torch/extension.h>

#include <array>
#include <complex>
#include <cstdint>
#include <limits>
#include <string>
#include <utility>

#include "securec.h"

namespace {

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
        ACLSPARSE_CHECK(aclsparseSpGEMMCreateDescr(&value_));
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
    ACLSPARSE_CHECK(aclsparseCreateConstCsr(
        matA.Out(), prepared.m, prepared.k, prepared.aValues.numel(),
        prepared.aCrow.const_data_ptr(), prepared.aCol.const_data_ptr(),
        prepared.aValues.const_data_ptr(),
        ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_32I,
        ACL_SPARSE_INDEX_BASE_ZERO, prepared.aclType));
    ACLSPARSE_CHECK(aclsparseCreateConstCsr(
        matB.Out(), prepared.k, prepared.n, prepared.bValues.numel(),
        prepared.bCrow.const_data_ptr(), prepared.bCol.const_data_ptr(),
        prepared.bValues.const_data_ptr(),
        ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_32I,
        ACL_SPARSE_INDEX_BASE_ZERO, prepared.aclType));
    ACLSPARSE_CHECK(aclsparseCreateCsr(
        matC.Out(), prepared.m, prepared.n, 0,
        prepared.cCrow.mutable_data_ptr(), nullptr, nullptr,
        ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_32I,
        ACL_SPARSE_INDEX_BASE_ZERO, prepared.aclType));
}

struct SpGemmOutput {
    at::Tensor columns;
    at::Tensor values;
};

SpGemmOutput CopySpGemmOutput(
    aclsparseHandle_t handle, PreparedSpGemm &prepared,
    ConstSpMatGuard &matA, ConstSpMatGuard &matB, SpMatGuard &matC,
    SpGemmDescrGuard &descr, aclsparseOperation_t op, aclsparseSpGEMMAlg_t alg)
{
    int64_t rowsC = 0;
    int64_t colsC = 0;
    int64_t nnzC = 0;
    ACLSPARSE_CHECK(aclsparseSpMatGetSize(matC.Get(), &rowsC, &colsC, &nnzC));
    TORCH_CHECK(rowsC == prepared.m && colsC == prepared.n && nnzC >= 0 &&
        nnzC <= std::numeric_limits<int32_t>::max(), "invalid SpGEMM output metadata");
    SpGemmOutput output{
        at::empty({nnzC}, prepared.aValues.options().dtype(at::kInt)),
        at::empty({nnzC}, prepared.aValues.options())};
    ACLSPARSE_CHECK(aclsparseCsrSetPointers(
        matC.Get(), prepared.cCrow.mutable_data_ptr(), output.columns.mutable_data_ptr(),
        output.values.mutable_data_ptr()));
    ACLSPARSE_CHECK(aclsparseSpGEMMCopy(handle, op, op,
        prepared.alpha.Data(), matA.Get(), matB.Get(), prepared.beta.Data(), matC.Get(), prepared.aclType,
        alg, descr.Get()));
    return output;
}

SpGemmOutput ExecuteSpGemm(
    cann_ops_sparse::AclSparseContext &context, PreparedSpGemm &prepared,
    ConstSpMatGuard &matA, ConstSpMatGuard &matB, SpMatGuard &matC)
{
    SpGemmDescrGuard descr;
    constexpr aclsparseOperation_t op = ACL_SPARSE_OP_NON_TRANSPOSE;
    constexpr aclsparseSpGEMMAlg_t alg = ACL_SPARSE_SPGEMM_DEFAULT;
    size_t bufferSize1 = 0;
    const aclsparseHandle_t handle = context.handle();
    ACLSPARSE_CHECK(aclsparseSpGEMMWorkEstimation(handle, op, op,
        prepared.alpha.Data(), matA.Get(), matB.Get(), prepared.beta.Data(), matC.Get(), prepared.aclType,
        alg, descr.Get(), &bufferSize1, nullptr));
    TORCH_CHECK(bufferSize1 <= static_cast<size_t>(std::numeric_limits<int64_t>::max()),
        "SpGEMM work buffer exceeds PyTorch tensor limits");
    cann_ops_sparse::AclSparseWorkspace buffer1(context, prepared.aValues, bufferSize1);
    ACLSPARSE_CHECK(aclsparseSpGEMMWorkEstimation(handle, op, op,
        prepared.alpha.Data(), matA.Get(), matB.Get(), prepared.beta.Data(), matC.Get(), prepared.aclType,
        alg, descr.Get(), &bufferSize1, buffer1.data()));
    size_t bufferSize2 = 0;
    size_t bufferSize3 = 0;
    ACLSPARSE_CHECK(aclsparseSpGEMMEstimateMemory(handle, op, op,
        prepared.alpha.Data(), matA.Get(), matB.Get(), prepared.beta.Data(), matC.Get(), prepared.aclType,
        alg, descr.Get(), 1.0F, &bufferSize3, nullptr, &bufferSize2));
    TORCH_CHECK(bufferSize2 <= static_cast<size_t>(std::numeric_limits<int64_t>::max()),
        "SpGEMM compute buffer exceeds PyTorch tensor limits");
    cann_ops_sparse::AclSparseWorkspace buffer2(context, prepared.aValues, bufferSize2);
    ACLSPARSE_CHECK(aclsparseSpGEMMCompute(handle, op, op,
        prepared.alpha.Data(), matA.Get(), matB.Get(), prepared.beta.Data(), matC.Get(), prepared.aclType,
        alg, descr.Get(), &bufferSize2, buffer2.data()));
    return CopySpGemmOutput(handle, prepared, matA, matB, matC, descr, op, alg);
}

at::Tensor SpGemmCsr(
    const at::Tensor &mat1Input, const at::Tensor &mat2Input,
    const c10::Scalar &alphaScalar)
{
    ValidateSpGemmInputs(mat1Input, mat2Input);
    PreparedSpGemm prepared = PrepareSpGemm(mat1Input, mat2Input, alphaScalar);
    // PrepareSpGemm may enqueue index conversions in torch_npu's task queue.
    // Drain it before launching ACLSparse directly on the underlying stream.
    (void)c10_npu::getCurrentNPUStream(mat1Input.device().index()).stream(true);
    return cann_ops_sparse::RunAclSparse(mat1Input, [&](cann_ops_sparse::AclSparseContext &context) {
        ConstSpMatGuard matA;
        ConstSpMatGuard matB;
        SpMatGuard matC;
        CreateSpGemmDescriptors(prepared, matA, matB, matC);
        SpGemmOutput output = ExecuteSpGemm(context, prepared, matA, matB, matC);
        at::Tensor result = at::sparse_csr_tensor(prepared.cCrow, output.columns, output.values,
            {prepared.m, prepared.n}, output.values.options().layout(at::kSparseCsr));
        context.RecordTensors(prepared.aCrow, prepared.aCol, prepared.aValues,
            prepared.bCrow, prepared.bCol, prepared.bValues, prepared.cCrow,
            // A SparseCsrTensorImpl has no standalone storage.  Its three
            // component tensors above own all device allocations used by
            // result, so recording them is both sufficient and required.
            output.columns, output.values);
        return result;
    });
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

}  // namespace

at::Tensor SparseAddmmCsrPy(
    const at::Tensor &self, const at::Tensor &mat1, const at::Tensor &mat2,
    const std::complex<double> &beta, const std::complex<double> &alpha)
{
    return SparseAddmmCsr(self, mat1, mat2,
        c10::Scalar(c10::complex<double>(beta.real(), beta.imag())),
        c10::Scalar(c10::complex<double>(alpha.real(), alpha.imag())));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("sparse_sparse_matmul_csr", &SparseSparseMatmulCsr);
    m.def("sparse_sparse_matmul_coo", &SparseSparseMatmulCoo);
    m.def("sparse_addmm_csr", &SparseAddmmCsrPy);
}
