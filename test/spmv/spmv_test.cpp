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

#include <acl/acl.h>           // aclrtMalloc, aclrtMemcpy, etc.
#include <cann_ops_sparse.h>  // aclSparseSpMV
#include <stdio.h>             // printf
#include <stdlib.h>            // EXIT_FAILURE
#include <string.h>
#include <time.h>
#include <iostream>
#include <new>                 // std::bad_alloc

#define CHECK_ACL(x)                                                                        \
    do {                                                                                    \
        aclError __ret = x;                                                                 \
        if (__ret != ACL_ERROR_NONE) {                                                      \
            std::cerr << __FILE__ << ":" << __LINE__ << " aclError:" << __ret << std::endl; \
        }                                                                                   \
    } while (0);

#define CHECK_ACL_SPARSE(x)                                                                       \
    do {                                                                                          \
        AclSparseStatus __ret = x;                                                                \
        if (__ret != ACL_SPARSE_STATUS_SUCCESS) {                                                 \
            std::cerr << __FILE__ << ":" << __LINE__ << " aclSparseError:" << __ret << std::endl; \
        }                                                                                         \
    } while (0);

// 在线生成测试用例
int Init(int32_t deviceId, aclrtStream *stream)
{
    CHECK_ACL(aclInit(nullptr));
    CHECK_ACL(aclrtSetDevice(deviceId));
    CHECK_ACL(aclrtCreateStream(stream));

    return 0;
}

int Deinit(int32_t deviceId, aclrtStream stream)
{
    CHECK_ACL(aclrtDestroyStream(stream));
    CHECK_ACL(aclrtResetDevice(deviceId));
    CHECK_ACL(aclFinalize());
    return 0;
}

void spmv_csr_cpu(uint64_t rows, uint64_t cols, uint64_t nnz, int *ptrs, int *idxs, float *vals, float *x, float *y)
{
    for (uint64_t i = 0; i < rows; i++) {
        float res = 0.0;
        uint64_t start = ptrs[i];
        uint64_t end = ptrs[i + 1];
        for (uint64_t j = start; j < end; j++) {
            res += vals[j] * x[idxs[j]];
        }
        y[i] = res;
    }
}

// 生成随机 CSR 矩阵
void generate_random_csr(uint64_t rows, uint64_t cols, uint64_t nnz, int *csrOffsets, int *colsIndices, float *values)
{
    csrOffsets[0] = 0;
    uint64_t current_nnz = 0;
    
    // 为每一行生成随机的非零元素
    for (uint64_t i = 0; i < rows; i++) {
        uint64_t row_nnz = nnz / rows;
        if (i < nnz % rows) {
            row_nnz++;
        }
        
        bool *col_used = nullptr;
        if (cols > 0) {
            try {
                col_used = new bool[cols]();
            } catch (const std::bad_alloc& e) {
                std::cerr << "Memory allocation failed: " << e.what() << std::endl;
                // 清理已分配的资源
                delete[] col_used;
                return;
            }
        }
        
        for (uint64_t j = 0; j < row_nnz; j++) {
            if (cols == 0) break; // 避免除0错误
            int col;
            do {
                col = rand() % cols;
            } while (col_used[col]);
            col_used[col] = true;
            colsIndices[current_nnz] = col;
            values[current_nnz] = 0.12345 + 0.000001 * current_nnz;
            current_nnz++;
        }
        if (col_used != nullptr) {
            delete[] col_used;
        }
        
        csrOffsets[i + 1] = current_nnz;
    }
}

int main(void)
{
    clock_t start, end;

    const uint64_t A_num_rows = 1000;
    const uint64_t A_num_cols = 1000;
    const uint64_t A_nnz = 5000;  // 稀疏度约为 0.5%
    int32_t deviceId = 0;
    aclrtStream stream = 0;
    Init(deviceId, &stream);

    int *hA_columns;
    int *hA_csrOffsets;
    float *hA_values;
    float *hX;
    float *hY;

    // csr的三个参数
    hA_csrOffsets = (int *)malloc((A_num_rows + 1) * sizeof(int));
    hA_columns = (int *)malloc(A_nnz * sizeof(int));
    hA_values = (float *)malloc(A_nnz * sizeof(float));
    // 列向量
    hX = (float *)malloc(A_num_cols * sizeof(float));  // b vector
    for (uint64_t i = 0; i < A_num_cols; i++) {
        hX[i] = 1.2345 + i * 0.00000001;
    }

    hY = (float *)malloc(A_num_rows * sizeof(float));             // result vector
    float *hYBase = (float *)malloc(A_num_rows * sizeof(float));  // result vector

    float alpha = 1.0f;
    float beta = 0.0f;

    printf("======start test spmv!======\n");
    
    srand(time(NULL));
    generate_random_csr(A_num_rows, A_num_cols, A_nnz, hA_csrOffsets, hA_columns, hA_values);
    
    printf("Generated random CSR matrix with %lu rows, %lu cols, %lu non-zero elements\n", A_num_rows, A_num_cols, A_nnz);
    
    // 计算 CPU 参考结果
    spmv_csr_cpu(A_num_rows, A_num_cols, A_nnz, hA_csrOffsets, hA_columns, hA_values, hX, hYBase);

    start = clock();

    //--------------------------------------------------------------------------
    // Device memory management
    int *dA_csrOffsets, *dA_columns;
    float *dA_values, *dX, *dY;
    CHECK_ACL(aclrtMalloc((void **)&dA_csrOffsets, (A_num_rows + 1) * sizeof(int), ACL_MEM_MALLOC_HUGE_FIRST))
    CHECK_ACL(aclrtMalloc((void **)&dA_columns, A_nnz * sizeof(int), ACL_MEM_MALLOC_HUGE_FIRST))
    CHECK_ACL(aclrtMalloc((void **)&dA_values, A_nnz * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST))
    CHECK_ACL(aclrtMalloc((void **)&dX, A_num_cols * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST))
    CHECK_ACL(aclrtMalloc((void **)&dY, A_num_rows * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST))

    end = clock();
    double time_aclrtMalloc = (double)(end - start) / CLOCKS_PER_SEC;

    CHECK_ACL(aclrtMemcpy(dA_csrOffsets,
        (A_num_rows + 1) * sizeof(int),
        hA_csrOffsets,
        (A_num_rows + 1) * sizeof(int),
        ACL_MEMCPY_HOST_TO_DEVICE))
    CHECK_ACL(aclrtMemcpy(dA_columns, A_nnz * sizeof(int), hA_columns, A_nnz * sizeof(int), ACL_MEMCPY_HOST_TO_DEVICE))
    CHECK_ACL(
        aclrtMemcpy(dA_values, A_nnz * sizeof(float), hA_values, A_nnz * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE))

    start = clock();
    CHECK_ACL(aclrtMemcpy(dX,
        A_num_cols * sizeof(float),
        hX,
        A_num_cols * sizeof(float),
        ACL_MEMCPY_HOST_TO_DEVICE))
    end = clock();
    double time_copyXDataToGPU = (double)(end - start) / CLOCKS_PER_SEC;

    CHECK_ACL(aclrtMemcpy(dY,
        A_num_rows * sizeof(float),
        hY,
        A_num_rows * sizeof(float),
        ACL_MEMCPY_HOST_TO_DEVICE))
    //--------------------------------------------------------------------------
    // aclSparse APIs
    AclSparseHandler handle = NULL;
    AclSparseSpMatDesc matA;
    AclSparseDnVecDesc vecX, vecY;
    void *dBuffer = NULL;
    size_t bufferSize = 0;
    CHECK_ACL_SPARSE(aclSparseCreate(&handle))
    for (uint64_t i = 0; i < 10; i++) {
        printf("%f ", hA_values[i]);
    }
    printf(" ... ");
    for (uint64_t i = A_nnz - 20; i < A_nnz - 10; i++) {
        printf("%f ", hA_values[i]);
    }

    printf("\r\n");
    printf("dA_values = %p\r\n", dA_values);

    start = clock();
    // Create sparse matrix A in CSR format
    CHECK_ACL_SPARSE(aclSparseCreateCsr(&matA,
        A_num_rows,
        A_num_cols,
        A_nnz,
        dA_csrOffsets,
        dA_columns,
        dA_values,
        ACL_SPARSE_INDEX_32I,
        ACL_SPARSE_INDEX_32I,
        ACL_SPARSE_INDEX_BASE_ZERO,
        ACL_FLOAT))
    end = clock();
    double time_aclSparseCreateCsr = (double)(end - start) / CLOCKS_PER_SEC;
    printf("Time for aclSparseCreateCsr: %f sec\n", time_aclSparseCreateCsr);

    start = clock();
    // Create dense vector X
    CHECK_ACL_SPARSE(aclSparseCreateDnVec(&vecX, A_num_cols, dX, ACL_FLOAT))
    // Create dense vector y
    CHECK_ACL_SPARSE(aclSparseCreateDnVec(&vecY, A_num_rows, dY, ACL_FLOAT))
    end = clock();
    double time_aclSparseCreateDnVec = (double)(end - start) / CLOCKS_PER_SEC;
    printf("Time for aclSparseCreateDnVec: %f sec\n", time_aclSparseCreateDnVec);

    // allocate an external buffer if needed
    CHECK_ACL_SPARSE(aclSparseSpmvGetBufferSize(handle,
        ACL_SPARSE_OP_NON_TRANSPOSE,
        &alpha,
        matA,
        vecX,
        &beta,
        vecY,
        ACL_FLOAT,
        ACL_SPARSE_SPMV_ALG_DEFAULT,
        &bufferSize))
    CHECK_ACL(aclrtMalloc(&dBuffer, bufferSize, ACL_MEM_MALLOC_HUGE_FIRST))

    start = clock();

    CHECK_ACL_SPARSE(aclSparseSpmvPreprocess(handle,
        ACL_SPARSE_OP_NON_TRANSPOSE,
        &alpha,
        matA,
        vecX,
        &beta,
        vecY,
        ACL_FLOAT,
        ACL_SPARSE_SPMV_ALG_DEFAULT,
        dBuffer))

    end = clock();

    double time_aclSparseSpMVPre = (double)(end - start) / CLOCKS_PER_SEC;

    start = clock();

    // execute SpMV
    CHECK_ACL_SPARSE(aclSparseSpmv(handle,
        ACL_SPARSE_OP_NON_TRANSPOSE,
        &alpha,
        matA,
        vecX,
        &beta,
        vecY,
        ACL_FLOAT,
        ACL_SPARSE_SPMV_ALG_DEFAULT,
        dBuffer))

    end = clock();

    double time_aclSparseSpMV = (double)(end - start) / CLOCKS_PER_SEC;

    // destroy matrix/vector descriptors
    CHECK_ACL_SPARSE(aclSparseDestroySpMat(matA))
    CHECK_ACL_SPARSE(aclSparseDestroyDnVec(vecX))
    CHECK_ACL_SPARSE(aclSparseDestroyDnVec(vecY))
    CHECK_ACL_SPARSE(aclSparseDestroy(handle))

    start = clock();
    //--------------------------------------------------------------------------
    // device result copy back
    CHECK_ACL(aclrtMemcpy(hY, A_num_rows * sizeof(float), dY, A_num_rows * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST))

    end = clock();
    double time_copyResultToCPU = (double)(end - start) / CLOCKS_PER_SEC;

    for (uint64_t i = 0; i < A_num_rows; i++) {
        if (abs(hY[i] - hYBase[i]) > 0.001) {
            printf("%lu expect %f , actual %f\r\n", i, hYBase[i], hY[i]);
            printf("test failed\r\n");
            return 0;
        }
    }
    printf("[Success] test case accuracy is verification passed.\r\n");

    start = clock();

    //--------------------------------------------------------------------------
    // device memory deallocation
    CHECK_ACL(aclrtFree(dBuffer))
    CHECK_ACL(aclrtFree(dA_csrOffsets))
    CHECK_ACL(aclrtFree(dA_columns))
    CHECK_ACL(aclrtFree(dA_values))
    CHECK_ACL(aclrtFree(dX))
    CHECK_ACL(aclrtFree(dY))

    free(hA_columns);  // 释放内存
    free(hA_csrOffsets);
    free(hA_values);
    free(hX);
    free(hY);
    free(hYBase);

    hA_columns = NULL;  // 避免悬垂指针（dangling pointer）
    hA_csrOffsets = NULL;
    hA_values = NULL;
    hX = NULL;
    hY = NULL;
    hYBase = NULL;

    end = clock();
    double time_freeGPUMemory = (double)(end - start) / CLOCKS_PER_SEC;

    Deinit(deviceId, stream);
    return EXIT_SUCCESS;
}