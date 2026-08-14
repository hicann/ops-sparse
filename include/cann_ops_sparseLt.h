/**
 * ----------------------------------------------------------------------------------------------------------
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ----------------------------------------------------------------------------------------------------------
 */

#ifndef CANN_OPS_SPARSELT_H_
#define CANN_OPS_SPARSELT_H_

#include "cann_ops_sparse.h"

// aclsparseLtHandle_t: opaque handle for the aclsparseLt library context.
struct aclsparseLtContext;
typedef struct aclsparseLtContext* aclsparseLtHandle_t;

/* ========== Descriptor Types ========== */

// aclsparseLtMatDescriptor_t: opaque pointer for a matrix descriptor (dense / structured).
struct aclsparseLtMatDescriptor;
typedef struct aclsparseLtMatDescriptor* aclsparseLtMatDescriptor_t;

// aclsparseLtMatmulDescriptor_t: opaque pointer for a matmul descriptor.
struct aclsparseLtMatmulDescriptor;
typedef struct aclsparseLtMatmulDescriptor* aclsparseLtMatmulDescriptor_t;

// Structured sparsity mode.
typedef enum aclsparseLtSparsity_t {
    ACL_SPARSE_LT_SPARSITY_50_PERCENT = 0   // 2:4 structured sparsity
} aclsparseLtSparsity_t;

// Compute precision
typedef enum aclsparseComputeType_t {
    ACL_SPARSE_COMPUTE_16F = 0,
    ACL_SPARSE_COMPUTE_32F,
    ACL_SPARSE_COMPUTE_32I
} aclsparseComputeType_t;

#ifdef __cplusplus
extern "C" {
#endif

/* ========== Library Management Functions ========== */

/**
 * @brief 初始化 aclsparseLt 库句柄。
 *
 * 在主机端分配轻量级硬件资源，创建 aclsparseLt 库上下文。
 * 必须在调用任何其他 aclsparseLt 函数之前调用。
 * 库上下文绑定到当前 NPU 设备，多设备使用需为每个设备创建独立 handle。
 *
 * @param handle OUT, HOST, aclsparseLt 库句柄输出参数。调用前 *handle 必须为 nullptr。
 * @return ACL_SPARSE_STATUS_SUCCESS 成功
 *         ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR handle 指针为空
 *         ACL_SPARSE_STATUS_INVALID_VALUE *handle 非空（防止覆盖已有句柄导致泄漏）
 *         ACL_SPARSE_STATUS_ALLOC_FAILED 内存分配失败
 */
aclsparseStatus_t aclsparseLtInit(aclsparseLtHandle_t* handle);

/**
 * @brief 释放 aclsparseLt 库句柄占用的全部资源。
 *
 * 与特定 handle 关联的最后一次调用。调用后该 handle 不可再使用。
 *
 * @param handle IN, HOST, 指向要销毁的 aclsparseLt 库句柄的指针。
 *               若 handle 指针本身为 nullptr 则返回 HANDLE_IS_NULLPTR；
 *               若 handle 指向的句柄值为 nullptr 则视为空操作，返回 SUCCESS。
 * @return ACL_SPARSE_STATUS_SUCCESS 成功
 *         ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR handle 指针为空
 */
aclsparseStatus_t aclsparseLtDestroy(const aclsparseLtHandle_t* handle);

/**
 * @brief 返回状态码对应的枚举名字符串。
 *
 * @param status IN, HOST, 要转换的状态码。
 * @return const char* 指向枚举名字符串的指针（如 "ACL_SPARSE_STATUS_SUCCESS"）。
 *         未识别的状态码返回 "unrecognized error code"。
 */
const char* aclsparseLtGetErrorName(aclsparseStatus_t status);

/**
 * @brief 返回状态码对应的描述性字符串。
 *
 * @param status IN, HOST, 要转换的状态码。
 * @return const char* 指向描述性字符串的指针。
 *         未识别的状态码返回 "unrecognized error code"。
 */
const char* aclsparseLtGetErrorString(aclsparseStatus_t status);

/* ========== Matrix Descriptor Management ========== */

/**
 * @brief 初始化稠密矩阵描述符。
 *
 * 在主机端分配并填充矩阵描述符结构体，记录稠密矩阵的形状、布局、数据类型等元信息。
 *
 * @param handle     IN,  HOST, aclsparseLt 库句柄，不可为 nullptr。
 * @param matDescr   OUT, HOST, 矩阵描述符输出。调用前 *matDescr 须为 nullptr。
 * @param rows       IN,  HOST, 行数（须 > 0 且为对应 valueType 对齐倍数的整数倍）。
 * @param cols       IN,  HOST, 列数（须 > 0 且为对应 valueType 对齐倍数的整数倍）。
 * @param ld         IN,  HOST, leading dimension。COL 序须 >= rows，ROW 序须 >= cols；
 *                   须为对应 valueType 对齐倍数的整数倍。
 * @param alignment  IN,  HOST, 内存对齐字节数，须为 16 的倍数且非 0。
 * @param valueType  IN,  HOST, 矩阵数据存储类型。
 * @param order      IN,  HOST, 内存布局（ACL_SPARSE_ORDER_ROW / ACL_SPARSE_ORDER_COL）。
 * @return ACL_SPARSE_STATUS_SUCCESS 成功
 *         ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR handle 指针为空
 *         ACL_SPARSE_STATUS_INVALID_VALUE matDescr 为空、*matDescr 非空、维度/对齐/布局非法
 *         ACL_SPARSE_STATUS_NOT_SUPPORTED valueType 不在支持列表内
 *         ACL_SPARSE_STATUS_ALLOC_FAILED 内存分配失败
 */
aclsparseStatus_t aclsparseLtDenseDescriptorInit(
    const aclsparseLtHandle_t*       handle,
    aclsparseLtMatDescriptor_t*      matDescr,
    int64_t                          rows,
    int64_t                          cols,
    int64_t                          ld,
    uint32_t                         alignment,
    aclDataType                      valueType,
    aclsparseOrder_t                 order);

/**
 * @brief 初始化结构化稀疏矩阵描述符（2:4 结构化稀疏）。
 *
 * 与 aclsparseLtDenseDescriptorInit 类似，但描述符标记为 structured，
 * 对齐倍数采用 structured 档位，并记录稀疏模式。
 *
 * @param handle     IN,  HOST, aclsparseLt 库句柄，不可为 nullptr。
 * @param matDescr   OUT, HOST, 矩阵描述符输出。调用前 *matDescr 须为 nullptr。
 * @param rows       IN,  HOST, 行数（须 > 0 且为 structured 档位对齐倍数的整数倍）。
 * @param cols       IN,  HOST, 列数（须 > 0 且为 structured 档位对齐倍数的整数倍）。
 * @param ld         IN,  HOST, leading dimension。COL 序须 >= rows，ROW 序须 >= cols；
 *                   须为 structured 档位对齐倍数的整数倍。
 * @param alignment  IN,  HOST, 内存对齐字节数，须为 16 的倍数且非 0。
 * @param valueType  IN,  HOST, 矩阵数据存储类型。
 * @param order      IN,  HOST, 内存布局（ACL_SPARSE_ORDER_ROW / ACL_SPARSE_ORDER_COL）。
 * @param sparsity   IN,  HOST, 稀疏模式（当前仅支持 ACL_SPARSE_LT_SPARSITY_50_PERCENT）。
 * @return ACL_SPARSE_STATUS_SUCCESS 成功
 *         ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR handle 指针为空
 *         ACL_SPARSE_STATUS_INVALID_VALUE matDescr 为空、*matDescr 非空、维度/对齐/布局/sparsity 非法
 *         ACL_SPARSE_STATUS_NOT_SUPPORTED valueType 不在支持列表内
 *         ACL_SPARSE_STATUS_ALLOC_FAILED 内存分配失败
 */
aclsparseStatus_t aclsparseLtStructuredDescriptorInit(
    const aclsparseLtHandle_t*       handle,
    aclsparseLtMatDescriptor_t*      matDescr,
    int64_t                          rows,
    int64_t                          cols,
    int64_t                          ld,
    uint32_t                         alignment,
    aclDataType                      valueType,
    aclsparseOrder_t                 order,
    aclsparseLtSparsity_t            sparsity);

/**
 * @brief 销毁矩阵描述符，释放其占用的主机内存。
 *
 * @param matDescr INOUT, HOST, 指向要销毁的矩阵描述符的指针。
 *                 若指针本身为 nullptr 则返回 HANDLE_IS_NULLPTR；
 *                 若 *matDescr 为 nullptr 则视为空操作，返回 SUCCESS。
 *                 销毁成功后 *matDescr 被置为 nullptr。
 * @return ACL_SPARSE_STATUS_SUCCESS 成功
 *         ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR matDescr 指针为空
 */
aclsparseStatus_t aclsparseLtMatDescriptorDestroy(aclsparseLtMatDescriptor_t* matDescr);

/* ========== Matmul Descriptor Management ========== */

/**
 * @brief 初始化矩阵乘法描述符。
 *
 * 在主机端分配并填充 matmul 描述符结构体，记录 matmul 运算的完整元信息：
 * 操作类型（opA/opB）、四个矩阵描述符引用（matA/matB/matC/matD）、计算精度。
 * matA/matB/matC/matD 为非所有权引用，Destroy 时不会销毁它们，用户须自行释放。
 *
 * 约束：
 * - matA 与 matB 中有且仅有一个为 structured 描述符。
 * - matC 与 matD 须具有相同的 ld 和 order。
 * - INT8/FP8/FP4 类型下（以 matA.valueType 为准），opA/opB 须满足与 orderA/orderB 的组合约束。
 * - matC/matD 的 rows、cols 均 <= 2097120。
 *
 * @param handle       IN,  HOST, aclsparseLt 库句柄，不可为 nullptr。
 * @param matmulDescr  OUT, HOST, matmul 描述符输出。调用前 *matmulDescr 须为 nullptr。
 * @param opA          IN,  HOST, 作用于矩阵 A 的操作（NON_TRANSPOSE / TRANSPOSE）。
 * @param opB          IN,  HOST, 作用于矩阵 B 的操作。
 * @param matA         IN,  HOST, 矩阵 A 描述符（structured 或 dense），不可为 nullptr。
 * @param matB         IN,  HOST, 矩阵 B 描述符（structured 或 dense），不可为 nullptr。
 * @param matC         IN,  HOST, 矩阵 C 描述符（dense），不可为 nullptr。
 * @param matD         IN,  HOST, 矩阵 D 描述符（dense），不可为 nullptr。
 * @param computeType  IN,  HOST, 计算精度（16F / 32F / 32I）。
 * @return ACL_SPARSE_STATUS_SUCCESS 成功
 *         ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR handle 指针为空
 *         ACL_SPARSE_STATUS_INVALID_VALUE matmulDescr 为空、*matmulDescr 非空、mat 描述符为空、
 *                                         opA/opB 非法、结构化位置/C/D 一致性/操作布局组合/维度约束违反
 *         ACL_SPARSE_STATUS_NOT_SUPPORTED computeType 不在支持列表内
 *         ACL_SPARSE_STATUS_ALLOC_FAILED 内存分配失败
 */
aclsparseStatus_t aclsparseLtMatmulDescriptorInit(
    const aclsparseLtHandle_t*             handle,
    aclsparseLtMatmulDescriptor_t*         matmulDescr,
    aclsparseOperation_t                   opA,
    aclsparseOperation_t                   opB,
    const aclsparseLtMatDescriptor_t*      matA,
    const aclsparseLtMatDescriptor_t*      matB,
    const aclsparseLtMatDescriptor_t*      matC,
    const aclsparseLtMatDescriptor_t*      matD,
    aclsparseComputeType_t                 computeType);

/**
 * @brief 销毁 matmul 描述符，释放其占用的主机内存。
 *
 * 注意：本接口不销毁 matA/matB/matC/matD 描述符（非所有权引用），
 * 用户须自行调用 aclsparseLtMatDescriptorDestroy 释放它们。
 *
 * @param matmulDescr INOUT, HOST, 指向要销毁的 matmul 描述符的指针。
 *                   若指针本身为 nullptr 则返回 HANDLE_IS_NULLPTR；
 *                   若 *matmulDescr 为 nullptr 则视为空操作，返回 SUCCESS。
 *                   销毁成功后 *matmulDescr 被置为 nullptr。
 * @return ACL_SPARSE_STATUS_SUCCESS 成功
 *         ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR matmulDescr 指针为空
 */
aclsparseStatus_t aclsparseLtMatmulDescriptorDestroy(aclsparseLtMatmulDescriptor_t* matmulDescr);

#ifdef __cplusplus
}
#endif

#endif
