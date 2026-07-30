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

/*!
 * \file gather_kernel.cpp
 * \brief gather kernel 实现（SIMT，仅 arch35 可用）。
 *
 * 两层结构：
 *   Layer 1: __simt_vf__ 计算函数（线程级并行，grid-stride loop）
 *   Layer 2: __global__ + _kernel_do（直接 asc_vf_call，无 dispatcher 类）
 *
 * Gather:  X.values[i] = Y[X.indices[i] - idxBase]  for i = 0 .. nnz-1.
 */

#include <cstdint>
#include "acl/acl_base_rt.h"
#include "kernel_operator.h"
#include "simt_api/asc_simt.h"
#include "gather_kernel.h"
#include "cann_ops_sparse.h"

// ===========================================================================
// Layer 1: SIMT VF 计算函数
// ===========================================================================
template <typename ValT, typename IdxT, aclsparseIndexBase_t idxBase>
__simt_vf__ __aicore__ __launch_bounds__(kGatherMaxThreadsPerBlock) inline void GatherSimtCompute(
    __gm__ const IdxT* indices, __gm__ const ValT* yValues, __gm__ ValT* xValues, int64_t nnz)
{
    int64_t globalTid = threadIdx.x + blockIdx.x * blockDim.x;
    int64_t gridStride = blockDim.x * gridDim.x;

    for (int64_t i = globalTid; i < nnz; i += gridStride) {
        int64_t pos = indices[i] - idxBase;
        xValues[i] = yValues[pos];
    }
}

// ===========================================================================
// Layer 2: __global__ kernel
// ===========================================================================
template <typename ValT, typename IdxT, aclsparseIndexBase_t idxBase>
__global__ __aicore__ void gather_kernel(
    GM_ADDR gmIndices, GM_ADDR gmYValues, GM_ADDR gmXValues, const GatherTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    asc_vf_call<GatherSimtCompute<ValT, IdxT, idxBase>>(
        dim3{kGatherMaxThreadsPerBlock}, (__gm__ const IdxT*)gmIndices, (__gm__ const ValT*)gmYValues,
        (__gm__ ValT*)gmXValues, tiling.nnz);
}

// since c++20
template <typename T>
struct type_identity {
    using type = T;
};

// ===========================================================================
// Layer 3: kernel_do 启动器
// ===========================================================================
extern "C" void gather_kernel_do(
    GM_ADDR indices, GM_ADDR yValues, GM_ADDR xValues, const GatherTilingData& tiling, aclrtStream stream)
{
    // c++17 lambda 不能写模板参数，只能这样传编译期值
    using base0_t = std::integral_constant<aclsparseIndexBase_t, ACL_SPARSE_INDEX_BASE_ZERO>;
    using base1_t = std::integral_constant<aclsparseIndexBase_t, ACL_SPARSE_INDEX_BASE_ONE>;
    // arm 平台不支持在 host 创建 bf16 变量，因此使用 type_identity 封装
    auto match_case = [&tiling, &stream, &indices, &yValues, &xValues](
                          aclDataType aclFloat_t, auto float_v, aclsparseIndexType_t aclIndex_t, auto int_v,
                          auto idxBase_v) -> void {
        using float_t = typename decltype(float_v)::type;
        using int_t = decltype(int_v);
        using idxBase_t = decltype(idxBase_v);
        if (tiling.valType == aclFloat_t && tiling.idxType == aclIndex_t && tiling.idxBase == idxBase_t::value) {
            gather_kernel<float_t, int_t, idxBase_t::value>
                <<<tiling.numBlocks, nullptr, stream>>>(indices, yValues, xValues, tiling);
        }
    };
    match_case(ACL_FLOAT, type_identity<float>{}, ACL_SPARSE_INDEX_32I, int32_t{}, base0_t{});
    match_case(ACL_FLOAT, type_identity<float>{}, ACL_SPARSE_INDEX_32I, int32_t{}, base1_t{});
    match_case(ACL_FLOAT, type_identity<float>{}, ACL_SPARSE_INDEX_64I, int64_t{}, base0_t{});
    match_case(ACL_FLOAT, type_identity<float>{}, ACL_SPARSE_INDEX_64I, int64_t{}, base1_t{});
    match_case(ACL_FLOAT16, type_identity<__fp16>{}, ACL_SPARSE_INDEX_32I, int32_t{}, base0_t{});
    match_case(ACL_FLOAT16, type_identity<__fp16>{}, ACL_SPARSE_INDEX_32I, int32_t{}, base1_t{});
    match_case(ACL_FLOAT16, type_identity<__fp16>{}, ACL_SPARSE_INDEX_64I, int64_t{}, base0_t{});
    match_case(ACL_FLOAT16, type_identity<__fp16>{}, ACL_SPARSE_INDEX_64I, int64_t{}, base1_t{});
    // 疑似编译器 bug: arm 平台 host 侧函数中无法出现 __bf16 类型，即使不实例化任何值
    // 由于 gather 算子只进行拷贝，没有浮点数数值计算，这里用 __fp16 替代
    match_case(ACL_BF16, type_identity<__fp16>{}, ACL_SPARSE_INDEX_32I, int32_t{}, base0_t{});
    match_case(ACL_BF16, type_identity<__fp16>{}, ACL_SPARSE_INDEX_32I, int32_t{}, base1_t{});
    match_case(ACL_BF16, type_identity<__fp16>{}, ACL_SPARSE_INDEX_64I, int64_t{}, base0_t{});
    match_case(ACL_BF16, type_identity<__fp16>{}, ACL_SPARSE_INDEX_64I, int64_t{}, base1_t{});
    match_case(ACL_DOUBLE, type_identity<double>{}, ACL_SPARSE_INDEX_32I, int32_t{}, base0_t{});
    match_case(ACL_DOUBLE, type_identity<double>{}, ACL_SPARSE_INDEX_32I, int32_t{}, base1_t{});
    match_case(ACL_DOUBLE, type_identity<double>{}, ACL_SPARSE_INDEX_64I, int64_t{}, base0_t{});
    match_case(ACL_DOUBLE, type_identity<double>{}, ACL_SPARSE_INDEX_64I, int64_t{}, base1_t{});
}
