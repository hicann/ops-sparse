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

#ifndef SPMV_HOST_H
#define SPMV_HOST_H

#include <unistd.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif
// This type indicates which operations is applied to the related input (e.g. sparse matrix, or vector).
typedef enum AclSparseOp {
    ACL_SPARSE_OP_NON_TRANSPOSE = 0,   // The non-transpose operation is selected.
    ACL_SPARSE_OP_TRANSPOSE,           // The transpose operation is selected.
    ACL_SPARSE_OP_CONJUGATE_TRANSPOSE  // The conjugate transpose operation is selected.
} AclSparseOp;

typedef void* AclSparseSpMatDesc;
typedef void* AclSparseDnVecDesc;
typedef void* AclSparseHandler;

// This data type represents the status returned by the library functions and it can have the following values:
typedef enum AclSparseStatus {
    ACL_SPARSE_STATUS_SUCCESS = 0,
    // The operation completed successfully
    ACL_SPARSE_STATUS_NOT_INITIALIZED,
    // The aclSPARSE library was not initialized. This is usually caused by the
    // lack of a prior call, an error in the Ascend Runtime API called by the
    // aclSPARSE routine, or an error in the hardware setup To correct: call
    // aclSparseCreate() prior to the function call; and check that the
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
    // aclSparseMatDescr_t descrA were set correctly
    ACL_SPARSE_STATUS_NOT_SUPPORTED,
    // The operation or data type combination is currently not supported by the
    // function
    ACL_SPARSE_STATUS_INSUFFICIENT_RESOURCES
    // The resources for the computation, such as NPU global or shared
    // memory, are not sufficient to complete the operation. The error can
    // also indicate that the current computation mode (e.g. bit size of
    // sparse matrix indices) does not allow to handle the given input
} AclSparseStatus;

// The type is an enumerant to specify the data precision. It is used when the data reference does not carry the type
// itself (e.g void *).
typedef enum AclDataType {
    ACL_R_32F = 0,  // real as a float
    ACL_R_64F,      // real as a double
    ACL_R_16F,      // real as a half
    ACL_R_8I,       // real as a signed 8-bit int
    ACL_C_32F,      // complex as a pair of float numbers
    ACL_C_64F,      // complex as a pair of double numbers
    ACL_C_16F,      // complex as a pair of half numbers
    ACL_C_8I,       // complex as a pair of signed 8-bit int numbers
    ACL_R_8U,       // real as a unsigned 8-bit int
    ACL_C_8U,       // complex as a pair of unsigned 8-bit int numbers
    ACL_R_32I,      // real as a signed 32-bit int
    ACL_C_32I,      // complex as a pair of signed 32-bit int numbers
    ACL_R_32U,      // real as a unsigned 32-bit int
    ACL_C_32U,      // complex as a pair of unsigned 32-bit int numbers
    ACL_R_16BF,     // real as a BF16
    ACL_C_16BF,     // complex as a pair of BF16 numbers
    ACL_R_4I,       // real as a signed 4-bit int
    ACL_C_4I,       // complex as a pair of signed 4-bit int numbers
    ACL_R_4U,       // real as a unsigned 4-bit int
    ACL_C_4U,       // complex as a pair of unsigned 4-bit int numbers
    ACL_R_16I,      // real as a signed 16-bit int
    ACL_C_16I,      // complex as a pair of signed 16-bit int numbers
    ACL_R_16U,      // real as a unsigned 16-bit int
    ACL_C_16U,      // complex as a pair of unsigned 16-bit int numbers
    ACL_R_64I,      // real as a signed 64-bit int
    ACL_C_64I,      // complex as a pair of signed 64-bit int numbers
    ACL_R_64U,      // real as a unsigned 64-bit int
    ACL_C_64U       // complex as a pair of unsigned 64-bit int numbers
} AclDataType;

typedef enum AclSparseSpmvAlg {
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
} AclSparseSpmvAlg;

// This type indicates the index type for representing the sparse matrix indices.
typedef enum AclSparseIndexType {
    ACL_SPARSE_INDEX_32I = 0,  // 32-bit unsigned integer [0, 2^31 - 1]
    ACL_SPARSE_INDEX_64I       // 64-bit unsigned integer [0, 2^63 - 1]
} AclSparseIndexType;

// This type indicates if the base of the matrix indices is zero or one.
typedef enum AclSparseIndexBase {
    ACL_SPARSE_INDEX_BASE_ZERO = 0,  // The base index is zero (C compatibility).
    ACL_SPARSE_INDEX_BASE_ONE        // The base index is one (Fortran compatibility).
} AclSparseIndexBase;

// This type indicates the format of the sparse matrix.
typedef enum AclSparseFormat {
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
} AclSparseFormat;

/**
 * @brief 创建稀疏矩阵处理器。
 * @param handle IN/OUT, HOST，返回创建的稀疏矩阵处理器句柄。
 * @return AclSparseStatus，表示函数执行状态。
 */
AclSparseStatus aclSparseCreate(AclSparseHandler *handle);

/**
 * @brief 销毁稀疏矩阵处理器。
 * @param handle IN, HOST, 稀疏矩阵处理器句柄。
 * @return 返回销毁操作的状态。
 */
AclSparseStatus aclSparseDestroy(AclSparseHandler handle);

/**
 * @brief 创建一个稠密向量。
 *
 * @param dnVecDescr IN/OUT, HOST, 稀疏向量描述符。
 * @param size IN, HOST, 向量的大小。
 * @param values IN, DEVICE, 向量的值。
 * @param valueType IN, HOST, 值的数据类型。
 * @return AclSparseStatus 返回稀疏操作的状态。
 */
AclSparseStatus aclSparseCreateDnVec(AclSparseDnVecDesc *dnVecDescr, int64_t size, void *values, AclDataType valueType);

/**
 * @brief 销毁稀疏向量描述符。
 *
 * @param dnVecDescr IN, HOST, 要销毁的稀疏向量描述符。
 * @return AclSparseStatus 返回执行状态，成功返回ACL_SPARSE_STATUS_SUCCESS。
 */
AclSparseStatus aclSparseDestroyDnVec(AclSparseDnVecDesc dnVecDescr);

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
 * @return AclSparseStatus 返回执行状态，成功返回ACL_SPARSE_STATUS_SUCCESS。
 */
AclSparseStatus aclSparseCreateCsr(AclSparseSpMatDesc *spMatDescr, int64_t rows, int64_t cols, int64_t nnz,
    void *csrRowOffsets, void *csrColInd, void *csrValues, AclSparseIndexType csrRowOffsetsType,
    AclSparseIndexType csrColIndType, AclSparseIndexBase idxBase, AclDataType valueType);

/**
 * @brief 创建一个CSC（Compressed Sparse Column）稀疏矩阵。
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
 * @return AclSparseStatus 返回执行状态，成功则为ACL_SPARSE_STATUS_SUCCESS。
 */
AclSparseStatus aclSparseCreateCsc(AclSparseSpMatDesc *spMatDescr, int64_t rows, int64_t cols, int64_t nnz,
    void *cscColOffsets, void *cscRowInd, void *cscValues, AclSparseIndexType cscColOffsetsType,
    AclSparseIndexType cscRowIndType, AclSparseIndexBase idxBase, AclDataType valueType);

/**
 * @brief 销毁稀疏矩阵对象。
 *
 * @param spMatDescr IN, HOST, 稀疏矩阵描述符，用于标识要销毁的稀疏矩阵对象。
 * @return AclSparseStatus 返回执行状态，若成功返回ACL_SPARSE_STATUS_SUCCESS。
 */
AclSparseStatus aclSparseDestroySpMat(AclSparseSpMatDesc spMatDescr);

/**
 * @brief 获取稀疏矩阵与向量相乘（SpMV）操作所需的缓冲区大小。
 *
 * @param handle IN, HOST, AclSparseHandler类型，稀疏计算句柄。
 * @param op IN, HOST, AclSparseOp类型，稀疏操作类型。
 * @param alpha IN, HOST/DEVICE, 指向常量的指针，表示SpMV操作中的alpha参数。
 * @param mat IN, HOST, AclSparseSpMatDesc类型，稀疏矩阵描述符。
 * @param x IN, HOST, AclSparseDnVecDesc类型，输入向量x的描述符。
 * @param beta IN, HOST/DEVICE, 指向常量的指针，表示SpMV操作中的beta参数。
 * @param y IN, HOST, AclSparseDnVecDesc类型，输出向量y的描述符。
 * @param computeType IN, HOST, AclDataType类型，计算的数据类型。
 * @param alg IN, HOST, AclSparseSpmvAlg类型，SpMV算法类型。
 * @param size IN, HOST, 指向size_t的指针，用于返回所需的缓冲区大小。
 * @return AclSparseStatus类型，表示函数执行状态。
 */
AclSparseStatus aclSparseSpmvGetBufferSize(AclSparseHandler handle, AclSparseOp op, const void *alpha,
    AclSparseSpMatDesc mat, AclSparseDnVecDesc x, const void *beta, AclSparseDnVecDesc y, AclDataType computeType,
    AclSparseSpmvAlg alg, size_t *size);

/**
 * @brief 对稀疏矩阵进行预处理，以便在后续的SpMV计算中使用。
 *
 * @param handle IN, HOST, AclSparseHandler类型的句柄，用于管理稀疏计算资源。
 * @param op IN, HOST, 稀疏操作类型，定义了稀疏矩阵的计算方式。
 * @param alpha IN, HOST/DEVICE, 指向常量alpha的指针，用于缩放稀疏矩阵。
 * @param mat IN, HOST 稀疏矩阵的描述符，包含了矩阵的属性和数据。
 * @param x IN, HOST 密集向量x的描述符，作为SpMV计算的输入。
 * @param beta IN, HOST/DEVICE, 指向常量beta的指针，用于缩放结果向量。
 * @param y IN, HOST, 密集向量y的描述符，作为SpMV计算的输出。
 * @param computeType IN, HOST, 计算的数据类型，定义了计算过程中使用的数据精度。
 * @param alg IN, HOST, SpMV计算的算法类型，定义了计算的具体实现方式。
 * @param buffer IN, DEVICE, 用于存储预处理结果的缓冲区指针。
 *
 * @return AclSparseStatus 返回预处理操作的状态，表示操作是否成功。
 */
AclSparseStatus aclSparseSpmvPreprocess(AclSparseHandler handle, AclSparseOp op, const void *alpha,
    AclSparseSpMatDesc mat, AclSparseDnVecDesc x, const void *beta, AclSparseDnVecDesc y, AclDataType computeType,
    AclSparseSpmvAlg alg, void *buffer);

/**
 * @brief 稀疏矩阵向量乘法函数
 *
 * @param handle IN, HOST, 稀疏矩阵处理器句柄
 * @param op IN, HOST, 稀疏矩阵操作符
 * @param alpha IN, HOST/DEVICE, 标量alpha
 * @param mat IN, HOST, 稀疏矩阵描述符
 * @param x IN, HOST, 密集向量x描述符
 * @param beta IN, HOST/DEVICE, 标量beta
 * @param y IN, HOST, 密集向量y描述符
 * @param computeType IN, HOST, 计算类型
 * @param alg IN, HOST, 稀疏矩阵向量乘法算法
 * @param buffer IN, DEVICE, 工作缓冲区指针
 * @return AclSparseStatus 返回状态
 */
AclSparseStatus aclSparseSpmv(AclSparseHandler handle, AclSparseOp op, const void *alpha, AclSparseSpMatDesc mat,
    AclSparseDnVecDesc x, const void *beta, AclSparseDnVecDesc y, AclDataType computeType, AclSparseSpmvAlg alg,
    void *buffer);

AclSparseStatus aclSparseSpmvShowWorkSpace(AclSparseHandler handle, void *buffer);

#ifdef __cplusplus
}
#endif

#endif