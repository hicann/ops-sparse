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

#ifndef CANN_OPS_SPARSE_H_
#define CANN_OPS_SPARSE_H_

#include <unistd.h>
#include <stdint.h>
#include <acl/acl_base_rt.h>

#ifdef __cplusplus
extern "C" {
#endif
// This type indicates which operations is applied to the related input (e.g. sparse matrix, or vector).
typedef enum aclsparseOperation_t {
    ACL_SPARSE_OP_NON_TRANSPOSE = 0,   // The non-transpose operation is selected.
    ACL_SPARSE_OP_TRANSPOSE,           // The transpose operation is selected.
    ACL_SPARSE_OP_CONJUGATE_TRANSPOSE  // The conjugate transpose operation is selected.
} aclsparseOperation_t;

struct aclsparseContext;
struct aclsparseSpMatDescr;
struct aclsparseDnVecDescr;
struct aclsparseDnMatDescr;

typedef struct aclsparseContext* aclsparseHandle_t;
typedef struct aclsparseSpMatDescr* aclsparseSpMatDescr_t;
typedef struct aclsparseDnVecDescr* aclsparseDnVecDescr_t;
typedef struct aclsparseDnMatDescr* aclsparseDnMatDescr_t;

typedef struct aclsparseSpMatDescr const* aclsparseConstSpMatDescr_t;
typedef struct aclsparseDnVecDescr const* aclsparseConstDnVecDescr_t;
typedef struct aclsparseDnMatDescr const* aclsparseConstDnMatDescr_t;

// Dense matrix data layout (used by B / C of SpMM).
typedef enum aclsparseOrder_t {
    ACL_SPARSE_ORDER_ROW = 0,   // Row-major.
    ACL_SPARSE_ORDER_COL = 1,   // Column-major.
} aclsparseOrder_t;

// Algorithm enum for SpMM.
typedef enum aclsparseSpMMAlg_t {
    ACL_SPARSE_SPMM_ALG_DEFAULT = 0,
    // Recommended entry for any sparse matrix format. V1 routes CSR to the same
    // SIMT/AIV implementation as ACL_SPARSE_SPMM_CSR_ALG1.
    ACL_SPARSE_SPMM_CSR_ALG1,
    // Explicit CSR algorithm 1. V1 uses the same SIMT/AIV implementation as
    // ACL_SPARSE_SPMM_ALG_DEFAULT.
    ACL_SPARSE_SPMM_CSR_FP32_HIGH_PRECISION_ALG,
    // fp32 high-precision algorithm. Same SIMT path as the default, but the fp32
    // accumulation uses Kahan compensated summation to suppress rounding /
    // cancellation error on long rows. Trades throughput for accuracy and is only
    // meaningful for fp32 (silently ignored for fp16 / int8). The other algorithms
    // remain the high-performance choice with standard (lower) fp32 precision.
    // ACL_SPARSE_SPMM_COO_ALG1,  // Reserved: COO support (V2).
} aclsparseSpMMAlg_t;

// This data type represents the status returned by the library functions and it can have the following values:
typedef enum aclsparseStatus_t {
    ACL_SPARSE_STATUS_SUCCESS = 0,
    // The operation completed successfully
    ACL_SPARSE_STATUS_NOT_INITIALIZED,
    // The aclSPARSE library was not initialized. This is usually caused by the
    // lack of a prior call, an error in the Ascend Runtime API called by the
    // aclSPARSE routine, or an error in the hardware setup To correct: call
    // aclsparseCreate() prior to the function call; and check that the
    // hardware, an appropriate version of the driver, and the aclSPARSE library
    // are correctly installed The error also applies to generic APIs (aclSPARSE
    // Generic APIs) for indicating a matrix/vector descriptor not initialized
    ACL_SPARSE_STATUS_ALLOC_FAILED,
    // Resource allocation failed inside the aclSPARSE library. This is usually
    // caused by a device memory allocation (aclrtMalloc()) or by a host memory
    // allocation failure To correct: prior to the function call, deallocate
    // previously allocated memory as much as possible
    ACL_SPARSE_STATUS_INVALID_VALUE,
    // An unsupported value or parameter was passed to the function (a negative
    // vector size, for example) To correct: ensure that all the parameters being
    // passed have valid values
    ACL_SPARSE_STATUS_ARCH_MISMATCH,
    // The function requires a feature absent from the device architecture
    // To correct: compile and run the application on a device with appropriate
    // compute capability
    ACL_SPARSE_STATUS_EXECUTION_FAILED,
    // The NPU program failed to execute. This is often caused by a launch
    // failure of the kernel on the NPU, which can be caused by multiple
    // reasons To correct: check that the hardware, an appropriate version of
    // the driver, and the aclSPARSE library are correctly installed
    ACL_SPARSE_STATUS_INTERNAL_ERROR,
    // An internal aclSPARSE operation failed
    // To correct: check that the hardware, an appropriate version of the driver,
    // and the aclSPARSE library are correctly installed. Also, check that the
    // memory passed as a parameter to the routine is not being deallocated prior
    // to the routine completion
    ACL_SPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED,
    // The matrix type is not supported by this function. This is
    // usually caused by passing an invalid matrix descriptor to the
    // function To correct: check that the fields in
    // aclsparseSpMatDescr_t descrA were set correctly
    ACL_SPARSE_STATUS_NOT_SUPPORTED,
    // The operation or data type combination is currently not supported by the
    // function
    ACL_SPARSE_STATUS_INSUFFICIENT_RESOURCES,
    // The resources for the computation, such as NPU global or shared
    // memory, are not sufficient to complete the operation. The error can
    // also indicate that the current computation mode (e.g. bit size of
    // sparse matrix indices) does not allow to handle the given input
    ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR
    // The handle is a null pointer
} aclsparseStatus_t;

typedef enum aclsparseSpMVAlg_t {
    ACL_SPARSE_SPMV_ALG_DEFAULT,
    // Default algorithm for any sparse matrix format.
    ACL_SPARSE_SPMV_COO_ALG1,
    // Default algorithm for COO sparse matrix format. May produce slightly different results
    // during different runs with the same input parameters.
    ACL_SPARSE_SPMV_COO_ALG2,
    // Provides deterministic (bit-wise) results for each run. If opA !=
    // ACL_SPARSE_OPERATION_NON_TRANSPOSE, it is identical to ACL_SPARSE_SPMV_COO_ALG1.
    ACL_SPARSE_SPMV_CSR_ALG1,
    // Default algorithm for CSR/CSC sparse matrix format. May produce slightly different
    // results during different runs with the same input parameters.
    ACL_SPARSE_SPMV_CSR_ALG2,
    // Provides deterministic (bit-wise) results for each run. If opA !=
    // ACL_SPARSE_OPERATION_NON_TRANSPOSE, it is identical to ACL_SPARSE_SPMV_CSR_ALG1.
    ACL_SPARSE_SPMV_SELL_ALG1,
    // Default algorithm for Sliced Ellpack sparse matrix format. Provides deterministic
    // (bit-wise) results for each run.
} aclsparseSpMVAlg_t;

// This type indicates the index type for representing the sparse matrix indices.
typedef enum aclsparseIndexType_t {
    ACL_SPARSE_INDEX_32I = 0,  // 32-bit unsigned integer [0, 2^31 - 1]（当前 SpMV/SpMM 已实现）
    ACL_SPARSE_INDEX_64I       // 64-bit unsigned integer [0, 2^63 - 1]（暂未支持，CreateCsr 将返回 NOT_SUPPORTED）
} aclsparseIndexType_t;

// This type indicates if the base of the matrix indices is zero or one.
typedef enum aclsparseIndexBase_t {
    ACL_SPARSE_INDEX_BASE_ZERO = 0,  // The base index is zero (C compatibility).
    ACL_SPARSE_INDEX_BASE_ONE        // The base index is one (Fortran compatibility).
} aclsparseIndexBase_t;

// This type indicates the format of the sparse matrix.
typedef enum aclsparseFormat_t {
    ACL_SPARSE_FORMAT_COO = 0,
    // The matrix is stored in Coordinate (COO) format organized in Structure of Arrays (SoA) layout
    ACL_SPARSE_FORMAT_CSR,
    // The matrix is stored in Compressed Sparse Row (CSR) format
    ACL_SPARSE_FORMAT_CSC,
    // The matrix is stored in Compressed Sparse Column (CSC) format
    ACL_SPARSE_FORMAT_BLOCKED_ELL,
    // The matrix is stored in Blocked-Ellpack (Blocked-ELL) format
    ACL_SPARSE_FORMAT_SLICED_ELL,
    // The matrix is stored in Sliced-Ellpack (Sliced-ELL) format
    ACL_SPARSE_FORMAT_BSR
    // The matrix is stored in Block Sparse Row (BSR) format
} aclsparseFormat_t;

/**
 * @brief 创建一个稠密向量。
 *
 * @param dnVecDescr IN/OUT, HOST, 稀疏向量描述符。
 * @param size IN, HOST, 向量的大小。
 * @param values IN, DEVICE, 向量的值。
 * @param valueType IN, HOST, 值的数据类型。
 * @return aclsparseStatus_t 返回稀疏操作的状态。
 */
aclsparseStatus_t aclsparseCreateDnVec(aclsparseDnVecDescr_t *dnVecDescr, int64_t size, void *values, aclDataType valueType);

/**
 * @brief 创建只读(const)稠密向量描述符。数据指针为 const。
 */
aclsparseStatus_t aclsparseCreateConstDnVec(aclsparseConstDnVecDescr_t *dnVecDescr, int64_t size,
    const void *values, aclDataType valueType);

/**
 * @brief 销毁稀疏向量描述符。
 *
 * @param dnVecDescr IN, HOST, 要销毁的稀疏向量描述符。
 * @return aclsparseStatus_t 返回执行状态，成功返回ACL_SPARSE_STATUS_SUCCESS。
 */
aclsparseStatus_t aclsparseDestroyDnVec(aclsparseConstDnVecDescr_t dnVecDescr);

/**
 * @brief 创建稀疏矩阵的CSR格式。
 *
 * 该函数用于创建一个稀疏矩阵的CSR（Compressed Sparse Row）格式。
 * CSR格式通过三个数组来表示稀疏矩阵：行偏移量数组、列索引数组和非零元素数组。
 *
 * @param spMatDescr IN/OUT, HOST, 稀疏矩阵描述符。
 *                      用于描述稀疏矩阵的属性和数据。
 * @param rows IN, HOST, 矩阵的行数。
 * @param cols IN, HOST, 矩阵的列数。
 * @param nnz IN, HOST, 矩阵的非零元素个数。
 * @param csrRowOffsets IN, DEVICE, 指向CSR格式行偏移量数组的指针。
 *                         该数组的长度为rows + 1，用于表示每行的非零元素起始位置。
 * @param csrColInd IN, DEVICE, 指向CSR格式列索引数组的指针。
 *                       该数组的长度为nnz，用于表示每个非零元素的列索引。
 * @param csrValues IN, DEVICE, 指向CSR格式非零元素数组的指针。
 *                     该数组的长度为nnz，用于存储矩阵的非零元素值。
 * @param csrRowOffsetsType IN, HOST, 行偏移量数组的数据类型。
 * @param csrColIndType IN, HOST, 列索引数组的数据类型。
 * @param idxBase IN, HOST, 索引的基值，可以是0或1。
 *                        0表示索引从0开始，1表示索引从1开始。
 * @param valueType IN, HOST, 非零元素的数据类型。
 * @return aclsparseStatus_t 返回执行状态，成功返回ACL_SPARSE_STATUS_SUCCESS。
 */
aclsparseStatus_t aclsparseCreateCsr(aclsparseSpMatDescr_t *spMatDescr, int64_t rows, int64_t cols, int64_t nnz,
    void *csrRowOffsets, void *csrColInd, void *csrValues, aclsparseIndexType_t csrRowOffsetsType,
    aclsparseIndexType_t csrColIndType, aclsparseIndexBase_t idxBase, aclDataType valueType);

/**
 * @brief 创建只读(const)CSR 稀疏矩阵描述符。
 */
aclsparseStatus_t aclsparseCreateConstCsr(aclsparseConstSpMatDescr_t *spMatDescr, int64_t rows, int64_t cols, int64_t nnz,
    const void *csrRowOffsets, const void *csrColInd, const void *csrValues, aclsparseIndexType_t csrRowOffsetsType,
    aclsparseIndexType_t csrColIndType, aclsparseIndexBase_t idxBase, aclDataType valueType);

/**
 * @brief 创建一个CSC（Compressed Sparse Column）稀疏矩阵。
 *
 * @note 当前版本暂未支持：SpMV / SpMM 仅实现 CSR 路径，调用本接口返回 ACL_SPARSE_STATUS_NOT_SUPPORTED。
 *
 * @param spMatDescr IN, HOST, 稀疏矩阵描述符，用于存储稀疏矩阵的元数据。
 * @param rows IN, HOST, 矩阵的行数。
 * @param cols IN, HOST, 矩阵的列数。
 * @param nnz IN, HOST, 矩阵中非零元素的数量。
 * @param cscColOffsets IN, DEVICE, 指向列偏移数组的指针，用于存储每列的起始位置。
 * @param cscRowInd IN, DEVICE, 指向行索引数组的指针，用于存储非零元素的行索引。
 * @param cscValues IN, DEVICE, 指向非零元素值数组的指针。
 * @param cscColOffsetsType IN, HOST, 列偏移数组的数据类型。
 * @param cscRowIndType IN, HOST, 行索引数组的数据类型。
 * @param idxBase IN, HOST, 索引的基础值，通常为0或1。
 * @param valueType IN, HOST, 非零元素值的数据类型。
 * @return aclsparseStatus_t 返回执行状态，成功则为ACL_SPARSE_STATUS_SUCCESS。
 */
aclsparseStatus_t aclsparseCreateCsc(aclsparseSpMatDescr_t *spMatDescr, int64_t rows, int64_t cols, int64_t nnz,
    void *cscColOffsets, void *cscRowInd, void *cscValues, aclsparseIndexType_t cscColOffsetsType,
    aclsparseIndexType_t cscRowIndType, aclsparseIndexBase_t idxBase, aclDataType valueType);

/**
 * @brief 创建只读(const)CSC 稀疏矩阵描述符。
 *
 * @note 当前版本暂未支持，调用返回 ACL_SPARSE_STATUS_NOT_SUPPORTED。
 */
aclsparseStatus_t aclsparseCreateConstCsc(aclsparseConstSpMatDescr_t *spMatDescr, int64_t rows, int64_t cols, int64_t nnz,
    const void *cscColOffsets, const void *cscRowInd, const void *cscValues, aclsparseIndexType_t cscColOffsetsType,
    aclsparseIndexType_t cscRowIndType, aclsparseIndexBase_t idxBase, aclDataType valueType);

/**
 * @brief 销毁稀疏矩阵对象。
 *
 * @param spMatDescr IN, HOST, 稀疏矩阵描述符，用于标识要销毁的稀疏矩阵对象。
 * @return aclsparseStatus_t 返回执行状态，若成功返回ACL_SPARSE_STATUS_SUCCESS。
 */
aclsparseStatus_t aclsparseDestroySpMat(aclsparseConstSpMatDescr_t spMatDescr);

/**
 * @brief 获取稀疏矩阵向量乘法（SpMV）所需的缓冲区大小
 *
 * @param handle IN, HOST, aclsparse 句柄
 * @param opA IN, HOST, 稀疏矩阵操作类型
 * @param alpha IN, HOST/DEVICE, 标量 alpha 指针
 * @param matA IN, HOST, 稀疏矩阵描述符
 * @param vecX IN, HOST, 稠密向量 x 描述符
 * @param beta IN, HOST/DEVICE, 标量 beta 指针
 * @param vecY IN, HOST, 稠密向量 y 描述符
 * @param computeType IN, HOST, 计算精度类型
 * @param alg IN, HOST, SpMV 算法类型
 * @param bufferSize OUT, HOST, 输出所需缓冲区大小
 * @return aclsparseStatus_t 状态码
 */
aclsparseStatus_t aclsparseSpMVGetBufferSize(aclsparseHandle_t handle, aclsparseOperation_t opA, const void *alpha,
                                                aclsparseConstSpMatDescr_t matA, aclsparseConstDnVecDescr_t vecX, const void *beta, aclsparseDnVecDescr_t vecY, aclDataType computeType,
                                                aclsparseSpMVAlg_t alg, size_t *bufferSize);

/**
 * @brief 对稀疏矩阵进行预处理，以便在后续的 SpMV 计算中使用。
 *
 * 可选调用；对同一 sparsity pattern 多次执行 SpMV 时可加速后续计算。
 * 调用后将 buffer 标记为 matA 的 active buffer，后续 SpMV 可复用预处理结果。
 *
 * @param handle IN, HOST, aclsparse 句柄
 * @param opA IN, HOST, 稀疏矩阵操作类型
 * @param alpha IN, HOST/DEVICE, 标量 alpha 指针
 * @param matA IN, HOST, 稀疏矩阵描述符
 * @param vecX IN, HOST, 稠密向量 x 描述符
 * @param beta IN, HOST/DEVICE, 标量 beta 指针
 * @param vecY IN, HOST, 稠密向量 y 描述符
 * @param computeType IN, HOST, 计算精度类型
 * @param alg IN, HOST, SpMV 算法类型
 * @param externalBuffer IN, DEVICE, 工作缓冲区
 * @return aclsparseStatus_t 状态码
 */
aclsparseStatus_t aclsparseSpMVPreprocess(aclsparseHandle_t handle, aclsparseOperation_t opA, const void *alpha,
    aclsparseConstSpMatDescr_t matA, aclsparseConstDnVecDescr_t vecX, const void *beta, aclsparseDnVecDescr_t vecY,
    aclDataType computeType, aclsparseSpMVAlg_t alg, void *externalBuffer);

/**
 * @brief 稀疏矩阵向量乘法（SpMV）计算入口
 *
 * @param handle IN, HOST, aclsparse 句柄
 * @param opA IN, HOST, 稀疏矩阵操作类型
 * @param alpha IN, HOST/DEVICE, 标量 alpha 指针
 * @param matA IN, HOST, 稀疏矩阵描述符
 * @param vecX IN, HOST, 稠密向量 x 描述符
 * @param beta IN, HOST/DEVICE, 标量 beta 指针
 * @param vecY IN, HOST, 稠密向量 y 描述符
 * @param computeType IN, HOST, 计算精度类型
 * @param alg IN, HOST, SpMV 算法类型
 * @param externalBuffer IN, DEVICE, 工作缓冲区
 * @return aclsparseStatus_t 状态码
 */
aclsparseStatus_t aclsparseSpMV(aclsparseHandle_t handle, aclsparseOperation_t opA, const void *alpha,
                                aclsparseConstSpMatDescr_t matA, aclsparseConstDnVecDescr_t vecX, const void *beta, aclsparseDnVecDescr_t vecY, aclDataType computeType,
                                aclsparseSpMVAlg_t alg, void *externalBuffer);

/**
 * @brief 创建稠密矩阵描述符。
 *
 * 用于描述 SpMM 中的 B / C 稠密矩阵。
 *
 * @param dnMatDescr IN/OUT, HOST，输出的稠密矩阵描述符。
 * @param rows       IN, HOST，矩阵的行数。
 * @param cols       IN, HOST，矩阵的列数。
 * @param ld         IN, HOST，leading dimension。行主序时需 >= cols；列主序时需 >= rows。
 * @param values     IN, DEVICE，矩阵数据指针。
 * @param valueType  IN, HOST，元素数据类型。支持 ACL_FLOAT / ACL_FLOAT16 / ACL_INT8
 *                   （用于 B 矩阵）；C 矩阵在 INT8 路径下为 ACL_INT32。
 * @param order      IN, HOST，布局：ACL_SPARSE_ORDER_ROW / ACL_SPARSE_ORDER_COL。
 * @return aclsparseStatus_t
 */
aclsparseStatus_t aclsparseCreateDnMat(aclsparseDnMatDescr_t *dnMatDescr,
    int64_t rows, int64_t cols, int64_t ld, void *values,
    aclDataType valueType, aclsparseOrder_t order);

/**
 * @brief 创建只读(const)稠密矩阵描述符。
 */
aclsparseStatus_t aclsparseCreateConstDnMat(aclsparseConstDnMatDescr_t *dnMatDescr,
    int64_t rows, int64_t cols, int64_t ld, const void *values,
    aclDataType valueType, aclsparseOrder_t order);

/**
 * @brief 销毁稠密矩阵描述符。
 */
aclsparseStatus_t aclsparseDestroyDnMat(aclsparseConstDnMatDescr_t dnMatDescr);

/**
 * @brief 计算 SpMM 所需 workspace 字节数。
 */
aclsparseStatus_t aclsparseSpMMGetBufferSize(
    aclsparseHandle_t handle, aclsparseOperation_t opA, aclsparseOperation_t opB,
    const void *alpha, aclsparseConstSpMatDescr_t matA, aclsparseConstDnMatDescr_t matB,
    const void *beta, aclsparseDnMatDescr_t matC, aclDataType computeType,
    aclsparseSpMMAlg_t alg, size_t *size);

/**
 * @brief 对稀疏矩阵进行预处理，加速后续 SpMM 计算。
 */
aclsparseStatus_t aclsparseSpMMPreprocess(
    aclsparseHandle_t handle, aclsparseOperation_t opA, aclsparseOperation_t opB,
    const void *alpha, aclsparseConstSpMatDescr_t matA, aclsparseConstDnMatDescr_t matB,
    const void *beta, aclsparseDnMatDescr_t matC, aclDataType computeType,
    aclsparseSpMMAlg_t alg, void *buffer);

/**
 * @brief 稀疏矩阵-稠密矩阵乘法：C = alpha * op(A) * op(B) + beta * C
 */
aclsparseStatus_t aclsparseSpMM(
    aclsparseHandle_t handle, aclsparseOperation_t opA, aclsparseOperation_t opB,
    const void *alpha, aclsparseConstSpMatDescr_t matA, aclsparseConstDnMatDescr_t matB,
    const void *beta, aclsparseDnMatDescr_t matC, aclDataType computeType,
    aclsparseSpMMAlg_t alg, void *buffer);

/**
 * @brief 创建 ops-sparse handle
 *
 * 在堆上分配一个 ops-sparse handle，并通过 @p handle 输出给调用者。
 *
 * @param handle 输出参数，用于接收创建的 handle 指针。调用前 *handle
 *               必须为 nullptr，否则视为已存在以防止内存泄漏。
 * @return ACL_SPARSE_STATUS_SUCCESS 成功
 *         ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR handle 指针为空
 *         ACL_SPARSE_STATUS_INVALID_VALUE *handle 非空
 *         ACL_SPARSE_STATUS_ALLOC_FAILED 内存分配失败
 */
aclsparseStatus_t aclsparseCreate(aclsparseHandle_t *handle);

/**
 * @brief 销毁 ops-sparse handle
 *
 * 释放由 aclsparseCreate 创建的 handle 所占有的全部资源。
 *
 * @param handle 要销毁的 handle
 * @return ACL_SPARSE_STATUS_SUCCESS 成功
 *         ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR handle 为空
 */
aclsparseStatus_t aclsparseDestroy(aclsparseHandle_t handle);

/**
 * @brief 设置 handle 的 stream
 * @param handle aclsparse handle（void*）
 * @param stream 要设置的 stream，nullptr 表示使用默认 stream
 * @return ACL_SPARSE_STATUS_SUCCESS 成功
 *         ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR handle 为空
 */
aclsparseStatus_t aclsparseSetStream(aclsparseHandle_t handle, aclrtStream stream);

/**
 * @brief 获取 handle 的 stream
 * @param handle aclsparse handle（void*）
 * @param stream 输出参数，返回当前 stream
 * @return ACL_SPARSE_STATUS_SUCCESS 成功
 *         ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR handle 为空
 *         ACL_SPARSE_STATUS_INVALID_VALUE stream 输出参数为空
 */
aclsparseStatus_t aclsparseGetStream(aclsparseHandle_t handle, aclrtStream *stream);

/**
 * @brief 获取 ops-sparse 版本号
 *
 * 版本号编码为 MAJOR * 10000 + MINOR * 100 + PATCH，如 10000 表示 1.0.0。
 *
 * @param handle handle指针，可为NULL
 * @param version 输出参数，接收版本号
 * @return ACL_SPARSE_STATUS_SUCCESS 成功
 *         ACL_SPARSE_STATUS_INVALID_VALUE 参数无效
 */
aclsparseStatus_t aclsparseGetVersion(aclsparseHandle_t handle, int *version);

#ifdef __cplusplus
}
#endif

#endif