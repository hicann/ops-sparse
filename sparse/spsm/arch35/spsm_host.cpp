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

// 1. C 标准库
#include <cstdint>

// 2. C++ 标准库
#include <algorithm>
#include <new>
#include <vector>

// 3. ACL/CANN 头文件
// 注: acl/acl.h 由 cann_ops_sparse.h 间接引入, 不在此显式 include
// 注: tiling/platform/platform_ascendc.h 由 aclsparse_host_utils.h 间接引入, 不在此显式 include
#include "log/log.h"

// 4. 算子本地头文件
#include "cann_ops_sparse.h"
#include "aclsparse_host_utils.h"
#include "aclsparse_descr_internal.h"
#include "spsm.h"
#include "spsm_kernel.h"

namespace {

// ============================================================================
// 动态获取核数 (禁止硬编码)
// ============================================================================
// 获取失败时兜底为 1 (保守下限): Solve kernel 依赖 SyncAll (需所有启动核都到达),
// blockDim 大于实际核数会卡死, 故兜底值取 1 而非固定核数。
constexpr uint32_t kSpsmBlockDimFallback = 1u;

uint32_t GetSpsmBlockDim()
{
    // 走公共 GetAivCoreCount() (禁止绕过公共封装直接调 platform)
    // 不做进程级缓存: 多卡场景下不同卡核数可能不同, 每次调用按当前设备取值
    const uint32_t aiv = GetAivCoreCount();
    if (aiv > 0u) {
        return aiv;
    }
    OP_LOGW("aclsparseSpSM", "GetAivCoreCount returned 0, fallback to %u cores",
            kSpsmBlockDimFallback);
    return kSpsmBlockDimFallback;
}

// ============================================================================
// 参数校验
// ============================================================================

// PointerMode=DEVICE 未实现 (alpha 标量 host 侧直接解引用), 校验拦截。
// 仅支持 HOST 模式, DEVICE 模式返回 NOT_SUPPORTED (用 OP_LOGE 记录失败分支)。
// aclsparseSpSMBufferSize 与 ExtractSpsmInputs 共用, 避免重复实现。
static aclsparseStatus_t CheckSpsmPointerModeHost(aclsparseHandle_t handle)
{
    aclsparsePointerMode_t pointerMode = ACL_SPARSE_POINTER_MODE_HOST;
    aclsparseStatus_t pmSt = aclsparseGetPointerMode(handle, &pointerMode);
    if (pmSt != ACL_SPARSE_STATUS_SUCCESS) {
        OP_LOGE("aclsparseSpSM", "aclsparseGetPointerMode failed, st=%d", static_cast<int>(pmSt));
        return pmSt;
    }
    if (pointerMode == ACL_SPARSE_POINTER_MODE_DEVICE) {
        OP_LOGE("aclsparseSpSM", "PointerMode=DEVICE not supported (only HOST), use aclsparseSetPointerMode(HOST)");
        return ACL_SPARSE_STATUS_NOT_SUPPORTED;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

static aclsparseStatus_t ValidateSpsmOperation(aclsparseOperation_t opA)
{
    if (opA != ACL_SPARSE_OP_NON_TRANSPOSE && opA != ACL_SPARSE_OP_TRANSPOSE) {
        return ACL_SPARSE_STATUS_NOT_SUPPORTED;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

static aclsparseStatus_t ValidateSpsmOpB(aclsparseOperation_t opB)
{
    if (opB != ACL_SPARSE_OP_NON_TRANSPOSE) {
        return ACL_SPARSE_STATUS_NOT_SUPPORTED;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

static bool SpsmDimensionsMatch(const aclsparseSpMatDescr *matA,
                                const aclsparseDnMatDescr *matB,
                                const aclsparseDnMatDescr *matC)
{
    if (matA->rows != matA->cols) {
        return false;
    }
    if (matA->rows != static_cast<uint64_t>(matB->rows) ||
        matA->rows != static_cast<uint64_t>(matC->rows)) {
        return false;
    }
    return matB->cols == matC->cols;
}

static aclsparseStatus_t ValidateSpsmFormat(const aclsparseSpMatDescr *matA)
{
    if (matA->format != ACL_SPARSE_FORMAT_CSR) {
        OP_LOGE("aclsparseSpSM", "unsupported format=%d (only CSR supported)", static_cast<int>(matA->format));
        return ACL_SPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED;
    }
    aclsparseStatus_t idxSt = AclsparseValidateSupportedCsrIndexTypes(matA->ptrType, matA->IdxType);
    if (idxSt != ACL_SPARSE_STATUS_SUCCESS) {
        OP_LOGE("aclsparseSpSM", "unsupported CSR index types (ptrType=%d, idxType=%d)",
                static_cast<int>(matA->ptrType), static_cast<int>(matA->IdxType));
        return idxSt;
    }
    if (matA->baseType != ACL_SPARSE_INDEX_BASE_ZERO &&
        matA->baseType != ACL_SPARSE_INDEX_BASE_ONE) {
        OP_LOGE("aclsparseSpSM", "unsupported baseType=%d (only ZERO/ONE supported)",
                static_cast<int>(matA->baseType));
        return ACL_SPARSE_STATUS_NOT_SUPPORTED;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

static aclsparseStatus_t ValidateSpsmDtype(const aclsparseSpMatDescr *matA,
                                            const aclsparseDnMatDescr *matB,
                                            const aclsparseDnMatDescr *matC,
                                            aclDataType computeType)
{
    // 只支持 FP32
    if (matA->valueType != ACL_FLOAT || matB->valueType != ACL_FLOAT ||
        matC->valueType != ACL_FLOAT || computeType != ACL_FLOAT) {
        OP_LOGE("aclsparseSpSM", "unsupported dtype combo (matA=%d, matB=%d, matC=%d, compute=%d), only FP32",
                static_cast<int>(matA->valueType), static_cast<int>(matB->valueType),
                static_cast<int>(matC->valueType), static_cast<int>(computeType));
        return ACL_SPARSE_STATUS_NOT_SUPPORTED;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

static aclsparseStatus_t ValidateSpsmDiagAndFill(const aclsparseSpMatDescr *matA)
{
    // v2: UNIT + NON_UNIT 均支持
    aclsparseDiagType_t diagType = matA->diagType;
    if (diagType != ACL_SPARSE_DIAG_TYPE_UNIT &&
        diagType != ACL_SPARSE_DIAG_TYPE_NON_UNIT) {
        OP_LOGE("aclsparseSpSM", "unsupported diagType=%d (UNIT/NON_UNIT supported)",
                static_cast<int>(diagType));
        return ACL_SPARSE_STATUS_NOT_SUPPORTED;
    }
    aclsparseFillMode_t fillMode = matA->fillMode;
    if (fillMode != ACL_SPARSE_FILL_MODE_LOWER &&
        fillMode != ACL_SPARSE_FILL_MODE_UPPER) {
        OP_LOGE("aclsparseSpSM", "invalid fillMode=%d (only LOWER/UPPER supported)",
                static_cast<int>(fillMode));
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

static aclsparseStatus_t ValidateSpsmShapeAndCapacity(const aclsparseSpMatDescr *matA,
                                                       const aclsparseDnMatDescr *matB,
                                                       const aclsparseDnMatDescr *matC)
{
    if (!SpsmDimensionsMatch(matA, matB, matC)) {
        OP_LOGE("aclsparseSpSM", "dimension mismatch: A=%llux%llu, B=%lldx%lld, C=%lldx%lld",
                static_cast<unsigned long long>(matA->rows), static_cast<unsigned long long>(matA->cols),
                static_cast<long long>(matB->rows), static_cast<long long>(matB->cols),
                static_cast<long long>(matC->rows), static_cast<long long>(matC->cols));
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (matA->rows > static_cast<uint64_t>(INT32_MAX) ||
        matC->cols > static_cast<int64_t>(INT32_MAX) ||
        matA->nnz > static_cast<uint64_t>(INT32_MAX)) {
        OP_LOGE("aclsparseSpSM", "dimension exceeds INT32_MAX (rows=%llu, cols=%lld, nnz=%llu)",
                static_cast<unsigned long long>(matA->rows), static_cast<long long>(matC->cols),
                static_cast<unsigned long long>(matA->nnz));
        return ACL_SPARSE_STATUS_NOT_SUPPORTED;
    }
    {
        int32_t m = static_cast<int32_t>(matA->rows);
        int32_t n = static_cast<int32_t>(matC->cols);
        if (m > 0 && n > 0 && static_cast<int64_t>(m) > INT64_MAX / static_cast<int64_t>(n) / 4) {
            OP_LOGE("aclsparseSpSM", "m*n too large for workspace (m=%d, n=%d)", m, n);
            return ACL_SPARSE_STATUS_NOT_SUPPORTED;
        }
    }
    // ld 校验按 order 区分: ROW→ld>=cols, COL→ld>=rows (参考 aclsparseCreateDnMat 契约)
    {
        int64_t bReq = (matB->order == ACL_SPARSE_ORDER_COL) ? matB->rows : matB->cols;
        if (matB->ld < 0 || matB->ld < bReq) {
            OP_LOGE("aclsparseSpSM", "matB ld(%lld) < required(%lld) (order=%d)",
                    static_cast<long long>(matB->ld), static_cast<long long>(bReq),
                    static_cast<int>(matB->order));
            return ACL_SPARSE_STATUS_INVALID_VALUE;
        }
    }
    {
        int64_t cReq = (matC->order == ACL_SPARSE_ORDER_COL) ? matC->rows : matC->cols;
        if (matC->ld < 0 || matC->ld < cReq) {
            OP_LOGE("aclsparseSpSM", "matC ld(%lld) < required(%lld) (order=%d)",
                    static_cast<long long>(matC->ld), static_cast<long long>(cReq),
                    static_cast<int>(matC->order));
            return ACL_SPARSE_STATUS_INVALID_VALUE;
        }
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

aclsparseStatus_t ValidateSpsmInputs(const aclsparseSpMatDescr *matA,
                                     const aclsparseDnMatDescr *matB,
                                     const aclsparseDnMatDescr *matC,
                                     aclsparseOperation_t opA,
                                     aclsparseOperation_t opB,
                                     aclDataType computeType,
                                     aclsparseSpSMAlg_t alg)
{
    if (matA == nullptr || matB == nullptr || matC == nullptr) {
        OP_LOGE("aclsparseSpSM", "matA/B/C is nullptr (matA=%p, matB=%p, matC=%p)", matA, matB, matC);
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    aclsparseStatus_t st = ValidateSpsmOperation(opA);
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        OP_LOGE("aclsparseSpSM", "unsupported opA=%d (only N/T supported)", static_cast<int>(opA));
        return st;
    }
    st = ValidateSpsmOpB(opB);
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        OP_LOGE("aclsparseSpSM", "unsupported opB=%d (only NON_TRANSPOSE supported)",
                static_cast<int>(opB));
        return st;
    }
    st = ValidateSpsmFormat(matA);
    if (st != ACL_SPARSE_STATUS_SUCCESS) { return st; }
    st = ValidateSpsmDtype(matA, matB, matC, computeType);
    if (st != ACL_SPARSE_STATUS_SUCCESS) { return st; }
    st = ValidateSpsmDiagAndFill(matA);
    if (st != ACL_SPARSE_STATUS_SUCCESS) { return st; }
    if (alg != ACL_SPARSE_SPSM_ALG_DEFAULT) {
        OP_LOGE("aclsparseSpSM", "unsupported alg=%d (only DEFAULT supported)", static_cast<int>(alg));
        return ACL_SPARSE_STATUS_NOT_SUPPORTED;
    }
    return ValidateSpsmShapeAndCapacity(matA, matB, matC);
}

// ============================================================================
// workspace 偏移计算
// ============================================================================
struct SpsmWsOffsets {
    int64_t levelRowPtrOff;
    int64_t levelRowIdxOff;
    int64_t diagValOff;
    int64_t transRowOffOff;
    int64_t transColIndOff;
    int64_t transValOff;
    int64_t denseBufOff;
    int64_t totalBytes;
};

// needDenseBuf: 仅 orderB==COL || orderC==COL 时分配 denseBuf 区 (避免无谓 m*n*4 字节)。
// 未分配时 denseBufOff=0 (kernel 仅在 orderB/orderC==COL 路径访问 denseBufGm_, 该路径不触发)。
static SpsmWsOffsets ComputeSpsmWsOffsets(int64_t m, int64_t n, int64_t nnz, bool needDenseBuf)
{
    SpsmWsOffsets o;
    // by-value 传递不落盘 GM, levelRowPtrOff 直接从 header 起算
    o.levelRowPtrOff = SPSM_WS_HEADER_BYTES;
    o.levelRowIdxOff = spsm_align_up(o.levelRowPtrOff + static_cast<int64_t>(sizeof(int32_t)) * (m + 1), SPSM_WS_ALIGN);
    o.diagValOff     = spsm_align_up(o.levelRowIdxOff + static_cast<int64_t>(sizeof(int32_t)) * m, SPSM_WS_ALIGN);
    // transRowOffOff 直接接 diagValOff
    o.transRowOffOff = spsm_align_up(o.diagValOff + static_cast<int64_t>(sizeof(float)) * m, SPSM_WS_ALIGN);
    o.transColIndOff = spsm_align_up(o.transRowOffOff + static_cast<int64_t>(sizeof(int32_t)) * (m + 1), SPSM_WS_ALIGN);
    o.transValOff    = spsm_align_up(o.transColIndOff + static_cast<int64_t>(sizeof(int32_t)) * nnz, SPSM_WS_ALIGN);
    if (needDenseBuf) {
        o.denseBufOff = spsm_align_up(o.transValOff + static_cast<int64_t>(sizeof(float)) * nnz, SPSM_WS_ALIGN);
        o.totalBytes  = spsm_align_up(o.denseBufOff + static_cast<int64_t>(sizeof(float)) * m * n, SPSM_WS_ALIGN);
    } else {
        // denseBuf 未使用 (orderB==ROW && orderC==ROW): 不分配该区。
        o.denseBufOff = 0;
        o.totalBytes  = spsm_align_up(o.transValOff + static_cast<int64_t>(sizeof(float)) * nnz, SPSM_WS_ALIGN);
    }
    return o;
}

// ============================================================================
// 辅助推导
// ============================================================================
static int32_t SpsmSwapFillMode(int32_t fillMode)
{
    return (fillMode == SPSM_FILL_LOWER) ? SPSM_FILL_UPPER : SPSM_FILL_LOWER;
}

static int32_t SpsmDeriveEffectiveFillMode(int32_t needTranspose, int32_t fillMode)
{
    return needTranspose ? SpsmSwapFillMode(fillMode) : fillMode;
}

static int32_t SpsmDeriveSolveDir(int32_t effectiveFillMode)
{
    return (effectiveFillMode == SPSM_FILL_LOWER) ? SPSM_SOLVE_FORWARD : SPSM_SOLVE_BACKWARD;
}

// ============================================================================
// maxRowLen 计算: 已内联到 PrepareAnalysisInputs (opA=N 路径 D2H rowOff + host 扫描)
// ============================================================================

// ============================================================================
// indexBase 有效值 (ONE 在内部归一化为 ZERO)
//
// 信任描述符策略 (对齐 cuSPARSE 语义):
//   - descriptorBase=0 (ZERO): effectiveBase=0, kernel 不减.
//   - descriptorBase=1 (ONE): effectiveBase=1, kernel 减 1 归一化为 0-based.
// 直接在 PrepareNormalInputs 内联赋值 (effectiveIndexBase = descBase), 无需独立函数。
// ============================================================================

// ============================================================================
// kChunkSize 计算: UB 容量推算
// ============================================================================

// UB 系统预留 (与 ComputeKChunkSize / ValidateMaxRowLenUbCapacity 共享)
constexpr int64_t kSpsmUbReserved = 8 * 1024;
// diagBuf 固定占用 (kernel InitBuffer(diagBuf_, kUbSlotBytes), kUbSlotBytes=32)
constexpr int64_t kSpsmDiagBufBytes = 32;

// maxRowLen UB 容量校验: colIndQue_/valsQue_ 各 2 份 (kBufferNum=2),
// 共 4*maxRowLen*4B. 当 maxRowLen 过大导致 colIndQue_/valsQue_ 占用超出 UB 可用空间时,
// kernel 侧 InitBuffer 会溢出, 此处提前拒绝.
static aclsparseStatus_t ValidateMaxRowLenUbCapacity(int32_t maxRowLen)
{
    const int64_t ubSize = static_cast<int64_t>(GetUbSize());
    if (ubSize <= kSpsmUbReserved + kSpsmDiagBufBytes) {
        return ACL_SPARSE_STATUS_SUCCESS;
    }
    int64_t maxSafeRowLen = (ubSize - kSpsmUbReserved - kSpsmDiagBufBytes) /
                            (4 * static_cast<int64_t>(sizeof(float)));
    if (static_cast<int64_t>(maxRowLen) > maxSafeRowLen) {
        OP_LOGE("aclsparseSpSM", "maxRowLen=%d exceeds UB safe limit=%lld (colIndQue_/valsQue_ overflow)",
                maxRowLen, static_cast<long long>(maxSafeRowLen));
        return ACL_SPARSE_STATUS_NOT_SUPPORTED;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

static int32_t ComputeKChunkSize(int32_t maxRowLen, int32_t n, int32_t L)
{
    // UB 容量: 走公共 GetUbSize() 动态获取 (禁止硬编码, 同文件已复用 GetAivCoreCount)。
    // 系统预留 8KB (UB_RESERVED), 获取失败(返回 0)时兜底返回 kChunkSize=1。
    const int64_t ubSize = static_cast<int64_t>(GetUbSize());
    if (ubSize <= kSpsmUbReserved + kSpsmDiagBufBytes) { return 1; }
    // Solve kernel UB 占用 (kBufferNum=2, 三个 TQue 统一双缓冲):
    //   colIndQue_ + valsQue_ = 4 * maxRowLenAlign * 4B (各 2 份)
    //   inQue_ = 2 * kChunkAlign * 4B (双缓冲, init/reduction 交替使用两份物理 buffer)
    //   accBuf_ + tmpBuf_ = 2 * kChunkAlign * 4B
    //   diagBuf_ = 32B (固定)
    //   levelRowPtrBuf_ = align32((L+1)*4B) (预加载 levelRowPtr, 避免裸 GM 标量读取)
    int64_t levelRowPtrBufBytes = ((static_cast<int64_t>(L) + 1) * sizeof(int32_t) + 31) / 32 * 32;
    int64_t fixedCost = kSpsmUbReserved + 4 * static_cast<int64_t>(maxRowLen) * sizeof(float) +
                        kSpsmDiagBufBytes + levelRowPtrBufBytes;
    int64_t ubAvail = ubSize - fixedCost;
    if (ubAvail < 32) { return 1; }
    int64_t kChunkRaw = ubAvail / (4 * sizeof(float));
    int64_t kChunk = std::min(kChunkRaw, static_cast<int64_t>(n));
    // 对齐到 8 (32B / sizeof(float))
    kChunk = (kChunk / 8) * 8;
    if (kChunk < 1) { kChunk = 1; }
    return static_cast<int32_t>(kChunk);
}

// buffer 大小一致性校验: 若 BufferSize 阶段已缓存大小, 则 Analysis 阶段重算结果须不超,
// 防止用户在 BufferSize 与 Analysis 之间更换 matA/matB/matC 导致 buffer 分配不足.
static aclsparseStatus_t ValidateBufferSizeConsistency(const aclsparseSpSMDescr *spsmDescrInner,
                                                         int64_t analysisTotalBytes)
{
    if (spsmDescrInner->cachedBufferSize > 0 &&
        analysisTotalBytes > spsmDescrInner->cachedBufferSize) {
        OP_LOGE("aclsparseSpSM", "buffer size mismatch (Analysis needs=%lld, BufferSize cached=%lld)",
                static_cast<long long>(analysisTotalBytes),
                static_cast<long long>(spsmDescrInner->cachedBufferSize));
        return ACL_SPARSE_STATUS_INSUFFICIENT_RESOURCES;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

// levelRowPtrBuf UB 容量校验: kernel 预加载 levelRowPtr[L+1] 到 UB,
// L 过大时 levelRowPtrBuf 占用超 UB, kernel InitBuffer 会失败, 此处提前拒绝.
static aclsparseStatus_t ValidateLevelRowPtrUbCapacity(int32_t L)
{
    int64_t ubSize = static_cast<int64_t>(GetUbSize());
    int64_t levelRowPtrBufBytes = ((static_cast<int64_t>(L) + 1) * sizeof(int32_t) + 31) / 32 * 32;
    if (ubSize > 0 && levelRowPtrBufBytes + kSpsmUbReserved + kSpsmDiagBufBytes >= ubSize) {
        OP_LOGE("aclsparseSpSM", "L=%d too large, levelRowPtrBuf=%lld exceeds UB",
                L, static_cast<long long>(levelRowPtrBufBytes));
        return ACL_SPARSE_STATUS_NOT_SUPPORTED;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

// ============================================================================
// CSR→CSC 转置 (host 侧)
//
// 输入: A 的 CSR (rowOff[m+1], colInd[nnz], values[nnz])
// 输出: A^T 的 CSR (transRowOff[m+1], transColInd[nnz], transValues[nnz])
//
// 实现: D2H → CPU 转置 → H2D。Analysis 一次性开销, 不影响 Solve 性能。
// ============================================================================

// D2H: 拷贝 CSR 三数组 (rowOff, colInd, values) 从 device 到 host.
static aclsparseStatus_t Csr2CscD2H(const void *csrRowOffsets, const void *csrColInd,
                                     const void *csrValues,
                                     std::vector<int32_t>& hRowOff,
                                     std::vector<int32_t>& hColInd,
                                     std::vector<float>& hValues,
                                     int32_t m, int64_t nnz)
{
    aclError aclRet = aclrtMemcpy(hRowOff.data(), sizeof(int32_t) * (static_cast<size_t>(m) + 1),
                                  csrRowOffsets, sizeof(int32_t) * (static_cast<size_t>(m) + 1),
                                  ACL_MEMCPY_DEVICE_TO_HOST);
    if (aclRet != ACL_ERROR_NONE) {
        OP_LOGE("aclsparseSpSM", "Csr2Csc: aclrtMemcpy rowOff D2H failed, ret=%d", static_cast<int>(aclRet));
        return ACL_SPARSE_STATUS_EXECUTION_FAILED;
    }
    aclRet = aclrtMemcpy(hColInd.data(), sizeof(int32_t) * nnz,
                         csrColInd, sizeof(int32_t) * nnz,
                         ACL_MEMCPY_DEVICE_TO_HOST);
    if (aclRet != ACL_ERROR_NONE) {
        OP_LOGE("aclsparseSpSM", "Csr2Csc: aclrtMemcpy colInd D2H failed, ret=%d", static_cast<int>(aclRet));
        return ACL_SPARSE_STATUS_EXECUTION_FAILED;
    }
    aclRet = aclrtMemcpy(hValues.data(), sizeof(float) * nnz,
                         csrValues, sizeof(float) * nnz,
                         ACL_MEMCPY_DEVICE_TO_HOST);
    if (aclRet != ACL_ERROR_NONE) {
        OP_LOGE("aclsparseSpSM", "Csr2Csc: aclrtMemcpy values D2H failed, ret=%d", static_cast<int>(aclRet));
        return ACL_SPARSE_STATUS_EXECUTION_FAILED;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

// indexBase=ONE 的 colInd 归一化 (ONE-based → ZERO-based).
// 信任描述符: ONE 描述符就归一化 (减 indexBase), ZERO 描述符就不归一化.
// 不再扫描 colInd 猜测数据真实 base, 不再对 UNIT 做特殊处理 (对齐 cuSPARSE 语义,
// 与 opA=N 路径 PrepareNormalInputs 的信任描述符策略一致).
static void Csr2CscNormalizeColInd(std::vector<int32_t>& hColInd, int64_t nnz,
                                     int32_t indexBase)
{
    if (indexBase == 0) { return; }
    for (int64_t k = 0; k < nnz; k++) {
        hColInd[static_cast<size_t>(k)] -= indexBase;
    }
}

// CPU 转置: CSR → CSC (= A^T 的 CSR).
// Pass 1: 统计每列 nnz → 前缀和 → 散列写入. 同时计算转置后最大行长.
static void Csr2CscTranspose(const std::vector<int32_t>& hRowOff,
                              const std::vector<int32_t>& hColInd,
                              const std::vector<float>& hValues,
                              std::vector<int32_t>& hTransRowOff,
                              std::vector<int32_t>& hTransColInd,
                              std::vector<float>& hTransValues,
                              int32_t m, int64_t nnz, int32_t *transMaxRowLen)
{
    // Pass 1: 统计每列 nnz
    for (int64_t k = 0; k < nnz; k++) {
        int32_t col = hColInd[static_cast<size_t>(k)];
        if (col >= 0 && col < m) {
            hTransRowOff[static_cast<size_t>(col)]++;
        }
    }
    // Pass 2: 前缀和 → transRowOff[0..m]
    int32_t sum = 0;
    for (int32_t i = 0; i < m; i++) {
        int32_t cnt = hTransRowOff[static_cast<size_t>(i)];
        hTransRowOff[static_cast<size_t>(i)] = sum;
        sum += cnt;
    }
    hTransRowOff[static_cast<size_t>(m)] = sum;

    // Pass 3: 散列写入
    std::vector<int32_t> writePos(hTransRowOff);  // 写指针副本
    for (int32_t i = 0; i < m; i++) {
        int32_t s = hRowOff[static_cast<size_t>(i)];
        int32_t e = hRowOff[static_cast<size_t>(i) + 1];
        for (int32_t p = s; p < e; p++) {
            int32_t col = hColInd[static_cast<size_t>(p)];
            if (col >= 0 && col < m) {
                int32_t pos = writePos[static_cast<size_t>(col)];
                hTransColInd[static_cast<size_t>(pos)] = i;  // A^T 的 colInd = A 的 rowInd
                hTransValues[static_cast<size_t>(pos)] = hValues[static_cast<size_t>(p)];
                writePos[static_cast<size_t>(col)] = pos + 1;
            }
        }
    }

    // 计算转置后 CSR 的最大行长 (kernel UB buffer 分配依据).
    // T 模式下 kernel 使用转置 CSR, maxRowLen 必须基于转置结果, 否则 UB 溢出.
    if (transMaxRowLen != nullptr) {
        int32_t tMaxLen = 0;
        for (int32_t i = 0; i < m; i++) {
            int32_t len = hTransRowOff[static_cast<size_t>(i) + 1] -
                          hTransRowOff[static_cast<size_t>(i)];
            if (len > tMaxLen) { tMaxLen = len; }
        }
        if (tMaxLen < 1) { tMaxLen = 1; }
        *transMaxRowLen = tMaxLen;
    }
}

// H2D: 拷贝转置后 CSR 三数组从 host 到 device workspace.
static aclsparseStatus_t Csr2CscH2D(void *workspace, const SpsmWsOffsets &off,
                                     const std::vector<int32_t>& hTransRowOff,
                                     const std::vector<int32_t>& hTransColInd,
                                     const std::vector<float>& hTransValues,
                                     int32_t m, int64_t nnz)
{
    uint8_t *ws = static_cast<uint8_t *>(workspace);
    aclError aclRet = aclrtMemcpy(ws + off.transRowOffOff, sizeof(int32_t) * (static_cast<size_t>(m) + 1),
                                  hTransRowOff.data(), sizeof(int32_t) * (static_cast<size_t>(m) + 1),
                                  ACL_MEMCPY_HOST_TO_DEVICE);
    if (aclRet != ACL_ERROR_NONE) {
        OP_LOGE("aclsparseSpSM", "Csr2Csc: aclrtMemcpy transRowOff H2D failed, ret=%d", static_cast<int>(aclRet));
        return ACL_SPARSE_STATUS_EXECUTION_FAILED;
    }
    aclRet = aclrtMemcpy(ws + off.transColIndOff, sizeof(int32_t) * nnz,
                         hTransColInd.data(), sizeof(int32_t) * nnz,
                         ACL_MEMCPY_HOST_TO_DEVICE);
    if (aclRet != ACL_ERROR_NONE) {
        OP_LOGE("aclsparseSpSM", "Csr2Csc: aclrtMemcpy transColInd H2D failed, ret=%d", static_cast<int>(aclRet));
        return ACL_SPARSE_STATUS_EXECUTION_FAILED;
    }
    aclRet = aclrtMemcpy(ws + off.transValOff, sizeof(float) * nnz,
                         hTransValues.data(), sizeof(float) * nnz,
                         ACL_MEMCPY_HOST_TO_DEVICE);
    if (aclRet != ACL_ERROR_NONE) {
        OP_LOGE("aclsparseSpSM", "Csr2Csc: aclrtMemcpy transValues H2D failed, ret=%d", static_cast<int>(aclRet));
        return ACL_SPARSE_STATUS_EXECUTION_FAILED;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

static aclsparseStatus_t SpsmCsr2CscHost(const void *csrRowOffsets,
                                           const void *csrColInd,
                                           const void *csrValues,
                                           void *workspace,
                                           const SpsmWsOffsets &off,
                                           int32_t m, int64_t nnz, int32_t indexBase,
                                           int32_t *transMaxRowLen,
                                           std::vector<int32_t>* outTransRowOff,
                                           std::vector<int32_t>* outTransColInd,
                                           std::vector<float>* outTransValues)
{
    if (transMaxRowLen != nullptr) { *transMaxRowLen = 1; }
    if (m <= 0 || nnz <= 0) {
        // 构造空转置 CSR (rowOff 全零), 供 host level scheduling 使用
        if (outTransRowOff != nullptr) {
            outTransRowOff->assign(static_cast<size_t>(m) + 1, 0);
        }
        if (outTransColInd != nullptr) { outTransColInd->clear(); }
        if (outTransValues != nullptr) { outTransValues->clear(); }
        return ACL_SPARSE_STATUS_SUCCESS;
    }

    std::vector<int32_t> hRowOff(static_cast<size_t>(m) + 1);
    std::vector<int32_t> hColInd(static_cast<size_t>(nnz));
    std::vector<float> hValues(static_cast<size_t>(nnz));

    aclsparseStatus_t st = Csr2CscD2H(csrRowOffsets, csrColInd, csrValues,
                                       hRowOff, hColInd, hValues, m, nnz);
    if (st != ACL_SPARSE_STATUS_SUCCESS) { return st; }

    Csr2CscNormalizeColInd(hColInd, nnz, indexBase);

    std::vector<int32_t> hTransRowOff(static_cast<size_t>(m) + 1, 0);
    std::vector<int32_t> hTransColInd(static_cast<size_t>(nnz));
    std::vector<float> hTransValues(static_cast<size_t>(nnz));

    Csr2CscTranspose(hRowOff, hColInd, hValues,
                     hTransRowOff, hTransColInd, hTransValues,
                     m, nnz, transMaxRowLen);

    st = Csr2CscH2D(workspace, off, hTransRowOff, hTransColInd, hTransValues, m, nnz);
    if (st != ACL_SPARSE_STATUS_SUCCESS) { return st; }

    if (outTransRowOff != nullptr) { *outTransRowOff = std::move(hTransRowOff); }
    if (outTransColInd != nullptr) { *outTransColInd = std::move(hTransColInd); }
    if (outTransValues != nullptr) { *outTransValues = std::move(hTransValues); }
    return ACL_SPARSE_STATUS_SUCCESS;
}

// ============================================================================
// host level scheduling (Analysis 移至 host CPU)
//
// 以下两个函数是 kernel 版 SpsmAnalysisAIV::ComputeRowLevels +
// BuildLevelBuckets + WritebackAnalysisResults 的逐字段等价复刻。
// ============================================================================

// 处理单个 (i, p) 列索引条目: 更新 maxLevel / hDiagVal / singular.
// 从 SpsmHostComputeRowLevels 内层循环抽出, 控制函数嵌套深度.
static void SpsmProcessRowEntry(int32_t i, int32_t p, int32_t m, int32_t indexBase,
                                  bool isNonUnit, int32_t effectiveFillMode,
                                  const std::vector<int32_t>& hColInd,
                                  const std::vector<float>& hValues,
                                  std::vector<float>& hDiagVal,
                                  const std::vector<int32_t>& hLevelBuf,
                                  int32_t& maxLevel, bool& singular)
{
    int32_t j = hColInd[static_cast<size_t>(p)] - indexBase;
    // 非法 colInd: 标记奇异 (Analysis 阶段拒绝), 避免层级调度被静默破坏
    if (j < 0 || j >= m) {
        singular = true;
        return;
    }
    if (j == i) {
        if (isNonUnit) {
            float dv = hValues[static_cast<size_t>(p)];
            hDiagVal[static_cast<size_t>(i)] = dv;
            if (dv == 0.0f) { singular = true; }
        }
        return;
    }
    bool isDep = (effectiveFillMode == SPSM_FILL_LOWER) ? (j < i) : (j > i);
    if (isDep) {
        int32_t depLevel = hLevelBuf[static_cast<size_t>(j)];
        if (depLevel + 1 > maxLevel) {
            maxLevel = depLevel + 1;
        }
    }
}

// host 版 ComputeRowLevels (与 SpsmAnalysisAIV::ComputeRowLevels 等价)
// 输入: hRowOff[m+1], hColInd[nnz], hValues[nnz], m, effectiveFillMode, solveDir, diagType, indexBase
// 输出: hLevelBuf[m], hDiagVal[m](NON_UNIT), singular
static void SpsmHostComputeRowLevels(
    const std::vector<int32_t>& hRowOff,
    const std::vector<int32_t>& hColInd,
    const std::vector<float>& hValues,
    int32_t m, int32_t effectiveFillMode, int32_t solveDir,
    int32_t diagType, int32_t indexBase,
    std::vector<int32_t>& hLevelBuf,
    std::vector<float>& hDiagVal,
    bool& singular)
{
    hLevelBuf.assign(static_cast<size_t>(m), 0);
    hDiagVal.assign(static_cast<size_t>(m), 0.0f);
    singular = false;
    if (m <= 0) {
        return;
    }

    const bool isNonUnit = (diagType == SPSM_DIAG_NON_UNIT);
    const int32_t start = (solveDir == SPSM_SOLVE_FORWARD) ? 0 : (m - 1);
    const int32_t end   = (solveDir == SPSM_SOLVE_FORWARD) ? m : -1;
    const int32_t step  = (solveDir == SPSM_SOLVE_FORWARD) ? 1 : -1;

    for (int32_t i = start; i != end; i += step) {
        int32_t s = hRowOff[static_cast<size_t>(i)];
        int32_t e = hRowOff[static_cast<size_t>(i) + 1];
        int32_t len = e - s;
        if (len < 0) { len = 0; }

        int32_t maxLevel = 0;
        for (int32_t p = s; p < e; p++) {
            SpsmProcessRowEntry(i, p, m, indexBase, isNonUnit, effectiveFillMode,
                                hColInd, hValues, hDiagVal, hLevelBuf, maxLevel, singular);
        }

        // NON_UNIT: 未找到对角线元素 → 隐式零对角元 → 奇异
        if (isNonUnit && hDiagVal[static_cast<size_t>(i)] == 0.0f) {
            singular = true;
        }

        hLevelBuf[static_cast<size_t>(i)] = maxLevel;
    }
}

// host 版 BuildLevelBuckets (与 SpsmAnalysisAIV::BuildLevelBuckets 等价)
// 输入: hLevelBuf[m], m, L
// 输出: levelRowPtr[L+1], levelRowIdx[m]
static void SpsmHostBuildLevelBuckets(
    const std::vector<int32_t>& hLevelBuf, int32_t m, int32_t L,
    std::vector<int32_t>& levelRowPtr,
    std::vector<int32_t>& levelRowIdx)
{
    levelRowPtr.assign(static_cast<size_t>(L) + 1, 0);
    levelRowIdx.assign(static_cast<size_t>(m), 0);
    if (m <= 0 || L <= 0) {
        return;
    }

    // Pass 1: 计数每 level 行数
    std::vector<int32_t> hCountBuf(static_cast<size_t>(L) + 1, 0);
    for (int32_t i = 0; i < m; i++) {
        int32_t lv = hLevelBuf[static_cast<size_t>(i)];
        hCountBuf[static_cast<size_t>(lv)]++;
    }

    // Pass 2: 前缀和 → levelRowPtr
    // levelRowPtr[0] = 0, levelRowPtr[l+1] = levelRowPtr[l] + countBuf[l]
    levelRowPtr[0] = 0;
    for (int32_t l = 0; l < L; l++) {
        levelRowPtr[static_cast<size_t>(l) + 1] =
            levelRowPtr[static_cast<size_t>(l)] + hCountBuf[static_cast<size_t>(l)];
    }

    // Pass 3: 复制 levelRowPtr 到 writePos, 按 level 散列写入 levelRowIdx
    std::vector<int32_t> writePos = levelRowPtr;
    for (int32_t i = 0; i < m; i++) {
        int32_t lv = hLevelBuf[static_cast<size_t>(i)];
        int32_t pos = writePos[static_cast<size_t>(lv)];
        levelRowIdx[static_cast<size_t>(pos)] = i;
        writePos[static_cast<size_t>(lv)] = pos + 1;
    }
}

// ============================================================================
// host CSR 容器: 承载 host level scheduling 所需的有效 CSR (opA=T 为转置 CSR,
// opA=N 为原始 CSR), 供 Analysis 在 host CPU 计算时使用。
// ============================================================================
struct SpsmHostCsr {
    std::vector<int32_t> rowOff;
    std::vector<int32_t> colInd;
    std::vector<float> values;
    int32_t indexBase = 0;
};

// ============================================================================
// Tiling 构建 + 缓存
// ============================================================================

// 填充 SpsmTilingData 字段 (纯字段赋值, 无副作用).
// L 由 host level scheduling 计算后传入。
static void FillSpsmTilingDataFields(SpsmTilingData &td, aclsparseSpMatDescr *matAInner,
                                       aclsparseDnMatDescr *matBInner, aclsparseDnMatDescr *matCInner,
                                       const void *alpha,
                                       int32_t maxRowLen, int32_t kChunkSize, int32_t effectiveIndexBase,
                                       int32_t needTranspose, int32_t effectiveFillMode,
                                       int32_t diagType, int32_t L,
                                       const SpsmWsOffsets &off)
{
    const int32_t m = static_cast<int32_t>(matAInner->rows);
    const int32_t n = static_cast<int32_t>(matCInner->cols);

    td.m = m;
    td.n = n;
    td.ldb = static_cast<int32_t>(matBInner->ld);
    td.ldc = static_cast<int32_t>(matCInner->ld);
    td.orderB = (matBInner->order == ACL_SPARSE_ORDER_COL) ? SPSM_ORDER_COL : SPSM_ORDER_ROW;
    td.orderC = (matCInner->order == ACL_SPARSE_ORDER_COL) ? SPSM_ORDER_COL : SPSM_ORDER_ROW;
    td.needTranspose = needTranspose;
    td.effectiveFillMode = effectiveFillMode;
    td.diagType = diagType;
    td.L = L;  // host level scheduling 计算填充
    td.kChunkSize = kChunkSize;
    td.maxRowLen = maxRowLen;
    td.alpha_host = *static_cast<const float *>(alpha);
    // indexBase: kernel 读取 colInd 时减去此值 (ONE 归一化为 ZERO).
    // opA=T: host Csr2Csc 已按描述符归一化, 此处恒为 0 (避免 kernel 二次减).
    // opA=N: 信任描述符 (effectiveIndexBase 直接回传 descriptorBase).
    td.indexBase = effectiveIndexBase;
    td.levelRowPtrOff = off.levelRowPtrOff;
    td.levelRowIdxOff = off.levelRowIdxOff;
    td.diagValOff = off.diagValOff;
    td.transRowOffOff = off.transRowOffOff;
    td.transColIndOff = off.transColIndOff;
    td.transValOff = off.transValOff;
    td.denseBufOff = off.denseBufOff;
}

// 构造 SpsmTilingData 并缓存到 spsmDescrInner->cachedTiling,
// Solve 时刷新 alpha 后 by-value 传入 kernel。
static aclsparseStatus_t SpsmBuildTilingFields(
    aclsparseSpMatDescr *matAInner, aclsparseDnMatDescr *matBInner,
    aclsparseDnMatDescr *matCInner, aclsparseOperation_t opA,
    aclsparseOperation_t opB,
    const void *alpha, aclsparseSpSMDescr *spsmDescrInner,
    const SpsmWsOffsets &off, int32_t maxRowLen, int32_t kChunkSize,
    int32_t effectiveIndexBase, int32_t L)
{
    const int32_t needTranspose = (opA == ACL_SPARSE_OP_TRANSPOSE) ? 1 : 0;
    const int32_t fillMode = (matAInner->fillMode == ACL_SPARSE_FILL_MODE_UPPER) ? SPSM_FILL_UPPER : SPSM_FILL_LOWER;
    const int32_t effectiveFillMode = SpsmDeriveEffectiveFillMode(needTranspose, fillMode);
    const int32_t diagType = (matAInner->diagType == ACL_SPARSE_DIAG_TYPE_UNIT) ? SPSM_DIAG_UNIT : SPSM_DIAG_NON_UNIT;

    SpsmTilingData td{};
    FillSpsmTilingDataFields(td, matAInner, matBInner, matCInner, alpha,
                              maxRowLen, kChunkSize, effectiveIndexBase,
                              needTranspose, effectiveFillMode,
                              diagType, L, off);

    // 缓存 TilingData 到 spsmDescr (Solve 时刷新 alpha 后 by-value 传入 kernel)
    spsmDescrInner->cachedTiling = td;

    // 缓存元数据到 spsmDescr (跨阶段一致性校验用)
    spsmDescrInner->opA = opA;
    spsmDescrInner->opB = opB;
    spsmDescrInner->needTranspose = needTranspose;
    // 缓存 Analysis 阶段 matA 的 (rows, cols, nnz, format), 跨阶段一致性校验用
    spsmDescrInner->matARows = matAInner->rows;
    spsmDescrInner->matACols = matAInner->cols;
    spsmDescrInner->matANnz = matAInner->nnz;
    spsmDescrInner->matAFormat = matAInner->format;
    // 缓存 Analysis 阶段 B/C 的 (rows, cols, order, ld), 跨阶段一致性校验用
    spsmDescrInner->matBRows = matBInner->rows;
    spsmDescrInner->matBCols = matBInner->cols;
    spsmDescrInner->matBLd = matBInner->ld;
    spsmDescrInner->matBOrder = matBInner->order;
    spsmDescrInner->matCRows = matCInner->rows;
    spsmDescrInner->matCCols = matCInner->cols;
    spsmDescrInner->matCLd = matCInner->ld;
    spsmDescrInner->matCOrder = matCInner->order;
    return ACL_SPARSE_STATUS_SUCCESS;
}

} // namespace

// ============================================================================
// SpSM 描述符管理
// ============================================================================
aclsparseStatus_t aclsparseSpSMCreateDescr(aclsparseSpSMDescr_t *spsmDescr)
{
    if (spsmDescr == nullptr) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    auto *inner = new (std::nothrow) aclsparseSpSMDescr();
    if (inner == nullptr) {
        return ACL_SPARSE_STATUS_ALLOC_FAILED;
    }
    *spsmDescr = inner;
    return ACL_SPARSE_STATUS_SUCCESS;
}

aclsparseStatus_t aclsparseSpSMDestroyDescr(aclsparseSpSMDescr_t spsmDescr)
{
    if (spsmDescr == nullptr) {
        return ACL_SPARSE_STATUS_SUCCESS;
    }
    delete spsmDescr;
    return ACL_SPARSE_STATUS_SUCCESS;
}

// ============================================================================
// 阶段 1: BufferSize
// ============================================================================
aclsparseStatus_t aclsparseSpSMBufferSize(
    aclsparseHandle_t handle, aclsparseOperation_t opA, aclsparseOperation_t opB,
    const void *alpha, aclsparseConstSpMatDescr_t matA,
    aclsparseConstDnMatDescr_t matB, aclsparseDnMatDescr_t matC,
    aclDataType computeType, aclsparseSpSMAlg_t alg,
    aclsparseSpSMDescr_t spsmDescr, size_t *bufferSize)
{
    if (handle == nullptr) {
        OP_LOGE("aclsparseSpSM", "handle is nullptr");
        return ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR;
    }
    if (bufferSize == nullptr) {
        OP_LOGE("aclsparseSpSM", "bufferSize is nullptr");
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (spsmDescr == nullptr) {
        OP_LOGE("aclsparseSpSM", "spsmDescr is nullptr");
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (alpha == nullptr) {
        OP_LOGE("aclsparseSpSM", "alpha is nullptr");
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    // PointerMode=DEVICE 未实现 (alpha 标量 host 侧直接解引用), 校验拦截。
    // 必须在解引用 alpha 之前拦截 (仅支持 HOST 模式)。
    aclsparseStatus_t pmSt = CheckSpsmPointerModeHost(handle);
    if (pmSt != ACL_SPARSE_STATUS_SUCCESS) {
        return pmSt;
    }
    aclsparseSpMatDescr *matAInner = SpsmToMatInner(matA);
    aclsparseDnMatDescr *matBInner = SpsmToDnMatInner(matB);
    aclsparseDnMatDescr *matCInner = matC;
    aclsparseStatus_t st = ValidateSpsmInputs(matAInner, matBInner, matCInner,
                                              opA, opB, computeType, alg);
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        return st;
    }
    const int64_t m = static_cast<int64_t>(matAInner->rows);
    const int64_t nnz = static_cast<int64_t>(matAInner->nnz);
    const int64_t n = static_cast<int64_t>(matCInner->cols);
    OP_LOGI("aclsparseSpSM", "BufferSize: opA=%d, opB=%d, m=%llu, n=%lld, nnz=%llu",
            static_cast<int>(opA), static_cast<int>(opB),
            static_cast<unsigned long long>(m), static_cast<long long>(n),
            static_cast<unsigned long long>(nnz));
    // denseBuf 仅 orderB==COL || orderC==COL 时分配
    const bool needDenseBuf = (matBInner->order == ACL_SPARSE_ORDER_COL) ||
                              (matCInner->order == ACL_SPARSE_ORDER_COL);
    SpsmWsOffsets off = ComputeSpsmWsOffsets(m, n, nnz, needDenseBuf);
    *bufferSize = static_cast<size_t>(off.totalBytes);
    // 缓存 BufferSize 算出的大小, Analysis 阶段校验一致性
    spsmDescr->cachedBufferSize = off.totalBytes;
    return ACL_SPARSE_STATUS_SUCCESS;
}

// ============================================================================
// 阶段 2: Analysis (host CPU 计算 level scheduling)
// ============================================================================

// opA=N 路径: D2H colInd + values (nnz>0 时). 抽出以控制 PrepareNormalInputs 行数.
static aclsparseStatus_t D2HColIndValues(aclsparseSpMatDescr *matAInner, int64_t nnz,
                                           std::vector<int32_t>& hColInd,
                                           std::vector<float>& hValues)
{
    if (nnz <= 0) {
        return ACL_SPARSE_STATUS_SUCCESS;
    }
    hColInd.resize(static_cast<size_t>(nnz));
    hValues.resize(static_cast<size_t>(nnz));
    aclError aclRet = aclrtMemcpy(hColInd.data(), sizeof(int32_t) * nnz,
                                   matAInner->idxs, sizeof(int32_t) * nnz,
                                   ACL_MEMCPY_DEVICE_TO_HOST);
    if (aclRet != ACL_ERROR_NONE) {
        OP_LOGE("aclsparseSpSM", "aclrtMemcpy colInd D2H failed, ret=%d", static_cast<int>(aclRet));
        return ACL_SPARSE_STATUS_EXECUTION_FAILED;
    }
    aclRet = aclrtMemcpy(hValues.data(), sizeof(float) * nnz,
                          matAInner->values, sizeof(float) * nnz,
                          ACL_MEMCPY_DEVICE_TO_HOST);
    if (aclRet != ACL_ERROR_NONE) {
        OP_LOGE("aclsparseSpSM", "aclrtMemcpy values D2H failed, ret=%d", static_cast<int>(aclRet));
        return ACL_SPARSE_STATUS_EXECUTION_FAILED;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

// opA=T 路径: CSR→CSC 转置 (host 侧) + 计算转置后 maxRowLen + 产出 host 转置 CSR.
static aclsparseStatus_t PrepareTransposeInputs(aclsparseSpMatDescr *matAInner,
                                                  void *buffer, const SpsmWsOffsets &off,
                                                  int32_t m, int64_t nnz,
                                                  int32_t &maxRowLen,
                                                  int32_t &effectiveIndexBase,
                                                  SpsmHostCsr &hostCsr)
{
    int32_t idxBase = (matAInner->baseType == ACL_SPARSE_INDEX_BASE_ONE) ? 1 : 0;
    // SpsmCsr2CscHost 同时产出 host 转置 CSR, 供 host level scheduling 使用
    aclsparseStatus_t st = SpsmCsr2CscHost(matAInner->ptrs, matAInner->idxs, matAInner->values,
                                             buffer, off, m, nnz, idxBase, &maxRowLen,
                                             &hostCsr.rowOff, &hostCsr.colInd, &hostCsr.values);
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        return st;
    }
    // Csr2Csc 已将 colInd 归一化为 0-based, kernel 读转置 CSR, 不需再减.
    hostCsr.indexBase = 0;
    effectiveIndexBase = 0;
    return ACL_SPARSE_STATUS_SUCCESS;
}

// opA=N 路径: D2H rowOff + colInd + values, 计算 maxRowLen + indexBase + host CSR.
static aclsparseStatus_t PrepareNormalInputs(aclsparseSpMatDescr *matAInner,
                                               int32_t m, int64_t nnz,
                                               int32_t &maxRowLen,
                                               int32_t &effectiveIndexBase,
                                               SpsmHostCsr &hostCsr)
{
    if (m <= 0) {
        return ACL_SPARSE_STATUS_SUCCESS;
    }
    std::vector<int32_t> hRowOff(static_cast<size_t>(m) + 1);
    aclError aclRet = aclrtMemcpy(hRowOff.data(), sizeof(int32_t) * (static_cast<size_t>(m) + 1),
                                   matAInner->ptrs, sizeof(int32_t) * (static_cast<size_t>(m) + 1),
                                   ACL_MEMCPY_DEVICE_TO_HOST);
    if (aclRet != ACL_ERROR_NONE) {
        OP_LOGE("aclsparseSpSM", "aclrtMemcpy rowOff D2H failed, ret=%d", static_cast<int>(aclRet));
        return ACL_SPARSE_STATUS_EXECUTION_FAILED;
    }
    std::vector<int32_t> hColInd;
    std::vector<float> hValues;
    aclsparseStatus_t st = D2HColIndValues(matAInner, nnz, hColInd, hValues);
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        return st;
    }
    // 计算 maxRowLen
    int32_t maxLen = 0;
    for (int32_t i = 0; i < m; i++) {
        int32_t len = hRowOff[static_cast<size_t>(i) + 1] - hRowOff[static_cast<size_t>(i)];
        if (len > maxLen) { maxLen = len; }
    }
    if (maxLen < 1) { maxLen = 1; }
    maxRowLen = maxLen;
    // opA=N: 信任描述符, 设置有效 indexBase (ONE 在内部归一化为 ZERO).
    int32_t descBase = (matAInner->baseType == ACL_SPARSE_INDEX_BASE_ONE) ? 1 : 0;
    effectiveIndexBase = descBase;
    hostCsr.rowOff = std::move(hRowOff);
    hostCsr.colInd = std::move(hColInd);
    hostCsr.values = std::move(hValues);
    hostCsr.indexBase = effectiveIndexBase;
    return ACL_SPARSE_STATUS_SUCCESS;
}

// 准备 Analysis 输入 (转置/maxRowLen/indexBase + host CSR).
// 同时产出 host CSR (有效 CSR), 供 host level scheduling 使用。
// opA=T: D2H → normalize → transpose → H2D, hostCsr = 转置 CSR (indexBase=0)
// opA=N: D2H rowOff + colInd + values, hostCsr = 原始 CSR (indexBase=effectiveIndexBase)
// kChunkSize 不在此计算 (需 L, 移至 aclsparseSpSMAnalysis 中 L 确定后计算)
static aclsparseStatus_t PrepareAnalysisInputs(aclsparseSpMatDescr *matAInner,
                                                  aclsparseOperation_t opA,
                                                  void *buffer, const SpsmWsOffsets &off,
                                                  int32_t m, int64_t nnz,
                                                  int32_t &maxRowLen,
                                                  int32_t &effectiveIndexBase,
                                                  SpsmHostCsr &hostCsr)
{
    // needTranspose 用 opA 局部判断, 不能用 spsmDescrInner->needTranspose
    // (后者在 SpsmBuildTilingFields 中才设置). T 模式下 kernel 使用转置 CSR,
    // maxRowLen 必须基于转置结果, 否则 UB buffer 可能溢出.
    const bool needTranspose = (opA == ACL_SPARSE_OP_TRANSPOSE);
    maxRowLen = 1;
    // effectiveIndexBase: kernel 读取 colInd 时减去此值.
    // opA=T: host Csr2Csc 已按描述符归一化为 0-based, kernel 不需再减 -> 恒为 0.
    // opA=N: 信任描述符 (ONE 减 1, ZERO 不减), 对齐 cuSPARSE 语义.
    effectiveIndexBase = 0;
    aclsparseStatus_t st;
    if (needTranspose) {
        st = PrepareTransposeInputs(matAInner, buffer, off, m, nnz,
                                      maxRowLen, effectiveIndexBase, hostCsr);
    } else {
        st = PrepareNormalInputs(matAInner, m, nnz, maxRowLen, effectiveIndexBase, hostCsr);
    }
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        return st;
    }

    // UB 容量校验: maxRowLen 过大时 colIndQue_/valsQue_ 会溢出 UB, 提前拒绝
    st = ValidateMaxRowLenUbCapacity(maxRowLen);
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        return st;
    }

    return ACL_SPARSE_STATUS_SUCCESS;
}

// H2D 写回: levelRowPtr[L+1], levelRowIdx[m], diagVal[m](NON_UNIT).
// 用同步 aclrtMemcpy (不传 stream, host 阻塞至完成), 不调用 aclrtSynchronizeStream.
static aclsparseStatus_t WritebackLevelData(void *buffer, const SpsmWsOffsets &off,
                                              const std::vector<int32_t>& levelRowPtr,
                                              const std::vector<int32_t>& levelRowIdx,
                                              const std::vector<float>& hDiagVal,
                                              int32_t m, int32_t L, int32_t diagType)
{
    uint8_t *ws = static_cast<uint8_t *>(buffer);
    if (m > 0 && L > 0) {
        aclError aclRet = aclrtMemcpy(ws + off.levelRowPtrOff,
                                       sizeof(int32_t) * (static_cast<size_t>(L) + 1),
                                       levelRowPtr.data(),
                                       sizeof(int32_t) * (static_cast<size_t>(L) + 1),
                                       ACL_MEMCPY_HOST_TO_DEVICE);
        if (aclRet != ACL_ERROR_NONE) {
            OP_LOGE("aclsparseSpSM", "aclrtMemcpy levelRowPtr H2D failed, ret=%d", static_cast<int>(aclRet));
            return ACL_SPARSE_STATUS_EXECUTION_FAILED;
        }
        aclRet = aclrtMemcpy(ws + off.levelRowIdxOff,
                              sizeof(int32_t) * static_cast<size_t>(m),
                              levelRowIdx.data(),
                              sizeof(int32_t) * static_cast<size_t>(m),
                              ACL_MEMCPY_HOST_TO_DEVICE);
        if (aclRet != ACL_ERROR_NONE) {
            OP_LOGE("aclsparseSpSM", "aclrtMemcpy levelRowIdx H2D failed, ret=%d", static_cast<int>(aclRet));
            return ACL_SPARSE_STATUS_EXECUTION_FAILED;
        }
    }
    if (diagType == SPSM_DIAG_NON_UNIT && m > 0) {
        aclError aclRet = aclrtMemcpy(ws + off.diagValOff,
                                       sizeof(float) * static_cast<size_t>(m),
                                       hDiagVal.data(),
                                       sizeof(float) * static_cast<size_t>(m),
                                       ACL_MEMCPY_HOST_TO_DEVICE);
        if (aclRet != ACL_ERROR_NONE) {
            OP_LOGE("aclsparseSpSM", "aclrtMemcpy diagVal H2D failed, ret=%d", static_cast<int>(aclRet));
            return ACL_SPARSE_STATUS_EXECUTION_FAILED;
        }
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

// host level scheduling + H2D 写回 + 奇异检测。
// 替代原 LaunchAnalysisKernelAndWait (含 aclrtSynchronizeStream)。
// 全程 host CPU 计算 + 同步 aclrtMemcpy (不传 stream), 不调用 aclrtSynchronizeStream。
// 返回 L (通过引用), singular 时返回 NOT_SUPPORTED。
static aclsparseStatus_t SpsmHostLevelScheduling(aclsparseSpMatDescr *matAInner,
                                                     aclsparseSpSMDescr *spsmDescrInner,
                                                     void *buffer, const SpsmWsOffsets &off,
                                                     const SpsmHostCsr &hostCsr,
                                                     int32_t m, int32_t &L)
{
    // 推导 effectiveFillMode/solveDir/diagType (与 SpsmBuildTilingFields 一致)
    const int32_t needTranspose = (spsmDescrInner->needTranspose != 0) ? 1 : 0;
    const int32_t fillMode = (matAInner->fillMode == ACL_SPARSE_FILL_MODE_UPPER) ? SPSM_FILL_UPPER : SPSM_FILL_LOWER;
    const int32_t effectiveFillMode = SpsmDeriveEffectiveFillMode(needTranspose, fillMode);
    const int32_t solveDir = SpsmDeriveSolveDir(effectiveFillMode);
    const int32_t diagType = (matAInner->diagType == ACL_SPARSE_DIAG_TYPE_UNIT) ? SPSM_DIAG_UNIT : SPSM_DIAG_NON_UNIT;

    // 1) host ComputeRowLevels
    std::vector<int32_t> hLevelBuf;
    std::vector<float> hDiagVal;
    bool singular = false;
    SpsmHostComputeRowLevels(hostCsr.rowOff, hostCsr.colInd, hostCsr.values,
                              m, effectiveFillMode, solveDir, diagType, hostCsr.indexBase,
                              hLevelBuf, hDiagVal, singular);
    if (singular) {
        OP_LOGE("aclsparseSpSM", "singular matrix detected (zero diagonal)");
        spsmDescrInner->analyzed = false;
        return ACL_SPARSE_STATUS_NOT_SUPPORTED;
    }

    // 2) L = max(hLevelBuf) + 1 (与 kernel ComputeMaxLevelPlusOne 等价)
    //    m<=0 时 L=0 (与 kernel Process 早返回行为一致)
    L = 0;
    if (m > 0) {
        for (int32_t i = 0; i < m; i++) {
            if (hLevelBuf[static_cast<size_t>(i)] > L) {
                L = hLevelBuf[static_cast<size_t>(i)];
            }
        }
        L += 1;
    }

    // 3) host BuildLevelBuckets
    std::vector<int32_t> levelRowPtr;
    std::vector<int32_t> levelRowIdx;
    SpsmHostBuildLevelBuckets(hLevelBuf, m, L, levelRowPtr, levelRowIdx);

    // 4) H2D 写回 (抽出至 WritebackLevelData)
    aclsparseStatus_t st = WritebackLevelData(buffer, off, levelRowPtr, levelRowIdx,
                                                hDiagVal, m, L, diagType);
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        return st;
    }

    spsmDescrInner->L = L;
    return ACL_SPARSE_STATUS_SUCCESS;
}

// Analysis 收尾 (日志 + 绑定 buffer + 标记 analyzed).
static aclsparseStatus_t FinalizeAnalysis(aclsparseSpSMDescr *spsmDescrInner,
                                           void *buffer, int32_t m, int32_t n,
                                           int32_t L, int32_t kChunkSize, int32_t maxRowLen)
{
    OP_LOGD("aclsparseSpSM", "analysis done: m=%d, n=%d, L=%d, kChunk=%d, maxRowLen=%d",
            m, n, L, kChunkSize, maxRowLen);

    // 绑定 active buffer + 标记 analyzed
    // active buffer 仅在 spsmDescr 内部维护, 不写回输入 matA (matA 形参为 const, 避免破坏 const 契约)
    spsmDescrInner->buffer = buffer;
    spsmDescrInner->analyzed = true;
    return ACL_SPARSE_STATUS_SUCCESS;
}

// 公共前置校验 + 内部描述符提取结果 (aclsparseSpSMAnalysis / aclsparseSpSM 共用).
// 返回结构体以消除两函数开头重复的 "声明4指针 + 调用 + 检查返回值" 代码块.
namespace {
struct SpsmExtractedInputs {
    aclsparseStatus_t status = ACL_SPARSE_STATUS_SUCCESS;
    aclsparseSpSMDescr *spsmDescrInner = nullptr;
    aclsparseSpMatDescr *matAInner = nullptr;
    aclsparseDnMatDescr *matBInner = nullptr;
    aclsparseDnMatDescr *matCInner = nullptr;
};

// 公共前置校验 + 内部描述符提取 (aclsparseSpSMAnalysis / aclsparseSpSM 共用).
// 消除两函数重复的 handle/spsmDescr/alpha nullptr 校验 + inner 指针提取 + ValidateSpsmInputs.
SpsmExtractedInputs ExtractSpsmInputs(
    aclsparseHandle_t handle, const void *alpha,
    aclsparseConstSpMatDescr_t matA, aclsparseConstDnMatDescr_t matB,
    aclsparseDnMatDescr_t matC, aclsparseOperation_t opA,
    aclsparseOperation_t opB, aclDataType computeType, aclsparseSpSMAlg_t alg,
    aclsparseSpSMDescr_t spsmDescr)
{
    SpsmExtractedInputs r{};
    if (handle == nullptr) {
        OP_LOGE("aclsparseSpSM", "handle is nullptr");
        r.status = ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR;
        return r;
    }
    if (spsmDescr == nullptr) {
        OP_LOGE("aclsparseSpSM", "spsmDescr is nullptr");
        r.status = ACL_SPARSE_STATUS_INVALID_VALUE;
        return r;
    }
    if (alpha == nullptr) {
        OP_LOGE("aclsparseSpSM", "alpha is nullptr");
        r.status = ACL_SPARSE_STATUS_INVALID_VALUE;
        return r;
    }
    // PointerMode=DEVICE 未实现 (alpha 标量 host 侧直接解引用), 校验拦截。
    // 必须在解引用 alpha 之前拦截 (仅支持 HOST 模式)。
    aclsparseStatus_t pmSt = CheckSpsmPointerModeHost(handle);
    if (pmSt != ACL_SPARSE_STATUS_SUCCESS) {
        r.status = pmSt;
        return r;
    }
    r.spsmDescrInner = spsmDescr;
    r.matAInner = SpsmToMatInner(matA);
    r.matBInner = SpsmToDnMatInner(matB);
    r.matCInner = matC;
    r.status = ValidateSpsmInputs(r.matAInner, r.matBInner, r.matCInner, opA, opB, computeType, alg);
    return r;
}
} // namespace

aclsparseStatus_t aclsparseSpSMAnalysis(
    aclsparseHandle_t handle, aclsparseOperation_t opA, aclsparseOperation_t opB,
    const void *alpha, aclsparseConstSpMatDescr_t matA,
    aclsparseConstDnMatDescr_t matB, aclsparseDnMatDescr_t matC,
    aclDataType computeType, aclsparseSpSMAlg_t alg,
    aclsparseSpSMDescr_t spsmDescr, void *buffer)
{
    auto in = ExtractSpsmInputs(handle, alpha, matA, matB, matC, opA, opB, computeType, alg, spsmDescr);
    if (in.status != ACL_SPARSE_STATUS_SUCCESS) {
        return in.status;
    }
    if (buffer == nullptr) {
        OP_LOGE("aclsparseSpSM", "buffer is nullptr");
        return ACL_SPARSE_STATUS_INSUFFICIENT_RESOURCES;
    }

    const int32_t m = static_cast<int32_t>(in.matAInner->rows);
    const int64_t nnz = static_cast<int64_t>(in.matAInner->nnz);
    const int32_t n = static_cast<int32_t>(in.matCInner->cols);
    OP_LOGI("aclsparseSpSM", "Analysis: opA=%d, opB=%d, m=%llu, n=%lld, nnz=%llu",
            static_cast<int>(opA), static_cast<int>(opB),
            static_cast<unsigned long long>(m), static_cast<long long>(n),
            static_cast<unsigned long long>(nnz));
    // denseBuf 仅 orderB==COL || orderC==COL 时分配
    const bool needDenseBuf = (in.matBInner->order == ACL_SPARSE_ORDER_COL) ||
                              (in.matCInner->order == ACL_SPARSE_ORDER_COL);
    SpsmWsOffsets off = ComputeSpsmWsOffsets(m, n, nnz, needDenseBuf);

    // 校验 buffer 大小: 若 BufferSize 阶段已缓存大小, 则 Analysis 阶段重算结果须一致
    // (防止用户在 BufferSize 与 Analysis 之间更换 matA/matB/matC 导致 buffer 分配不足)
    aclsparseStatus_t st = ValidateBufferSizeConsistency(in.spsmDescrInner, off.totalBytes);
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        return st;
    }

    // 提前推导 needTranspose 并缓存到 spsmDescrInner (SpsmHostLevelScheduling 依赖)
    in.spsmDescrInner->needTranspose = (opA == ACL_SPARSE_OP_TRANSPOSE) ? 1 : 0;

    // 1) 准备 Analysis 输入 (转置/maxRowLen/indexBase + host CSR)
    int32_t maxRowLen = 1;
    int32_t effectiveIndexBase = 0;
    SpsmHostCsr hostCsr;
    st = PrepareAnalysisInputs(in.matAInner, opA, buffer, off, m, nnz,
                               maxRowLen, effectiveIndexBase, hostCsr);
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        return st;
    }

    // 2) host level scheduling (全程 host CPU, 无 kernel, 无 stream 同步)
    int32_t L = 0;
    st = SpsmHostLevelScheduling(in.matAInner, in.spsmDescrInner, buffer, off, hostCsr, m, L);
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        return st;
    }

    // 校验 levelRowPtrBuf UB 容量 (kernel 预加载 levelRowPtr[L+1] 到 UB):
    // L 过大时 levelRowPtrBuf 占用超 UB, kernel InitBuffer 会失败, 此处提前拒绝
    st = ValidateLevelRowPtrUbCapacity(L);
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        return st;
    }

    // 3) 计算 kChunkSize (需 L, 用于 levelRowPtrBuf UB 占用计算)
    int32_t kChunkSize = ComputeKChunkSize(maxRowLen, n, L);

    // 4) 构造 + 缓存 TilingData (by-value 缓存)
    st = SpsmBuildTilingFields(in.matAInner, in.matBInner, in.matCInner, opA, opB,
                                alpha, in.spsmDescrInner, off, maxRowLen, kChunkSize,
                                effectiveIndexBase, L);
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        return st;
    }

    // 5) 收尾 (日志 + 绑定 buffer + 标记 analyzed)
    return FinalizeAnalysis(in.spsmDescrInner, buffer, m, n, L, kChunkSize, maxRowLen);
}

// 校验 matB 的 rows/cols/ld/order 与 Analysis 阶段缓存一致
static aclsparseStatus_t ValidateMatBFields(const aclsparseDnMatDescr *matBInner,
                                              const aclsparseSpSMDescr *spsmDescrInner)
{
    if (matBInner->rows != spsmDescrInner->matBRows ||
        matBInner->cols != spsmDescrInner->matBCols ||
        matBInner->ld != spsmDescrInner->matBLd ||
        matBInner->order != spsmDescrInner->matBOrder) {
        OP_LOGE("aclsparseSpSM", "matB mismatch (Solve rows=%llu cols=%lld ld=%lld order=%d, "
                "Analysis rows=%llu cols=%lld ld=%lld order=%d)",
                static_cast<unsigned long long>(matBInner->rows),
                static_cast<long long>(matBInner->cols),
                static_cast<long long>(matBInner->ld),
                static_cast<int>(matBInner->order),
                static_cast<unsigned long long>(spsmDescrInner->matBRows),
                static_cast<long long>(spsmDescrInner->matBCols),
                static_cast<long long>(spsmDescrInner->matBLd),
                static_cast<int>(spsmDescrInner->matBOrder));
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

// 校验 matC 的 rows/cols/ld/order 与 Analysis 阶段缓存一致
static aclsparseStatus_t ValidateMatCFields(const aclsparseDnMatDescr *matCInner,
                                              const aclsparseSpSMDescr *spsmDescrInner)
{
    if (matCInner->rows != spsmDescrInner->matCRows ||
        matCInner->cols != spsmDescrInner->matCCols ||
        matCInner->ld != spsmDescrInner->matCLd ||
        matCInner->order != spsmDescrInner->matCOrder) {
        OP_LOGE("aclsparseSpSM", "matC mismatch (Solve rows=%llu cols=%lld ld=%lld order=%d, "
                "Analysis rows=%llu cols=%lld ld=%lld order=%d)",
                static_cast<unsigned long long>(matCInner->rows),
                static_cast<long long>(matCInner->cols),
                static_cast<long long>(matCInner->ld),
                static_cast<int>(matCInner->order),
                static_cast<unsigned long long>(spsmDescrInner->matCRows),
                static_cast<long long>(spsmDescrInner->matCCols),
                static_cast<long long>(spsmDescrInner->matCLd),
                static_cast<int>(spsmDescrInner->matCOrder));
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

// 校验 matA 的 rows/cols/nnz/format 与 Analysis 阶段缓存一致
static aclsparseStatus_t ValidateMatAFields(const aclsparseSpMatDescr *matAInner,
                                              const aclsparseSpSMDescr *spsmDescrInner)
{
    if (matAInner->rows != spsmDescrInner->matARows ||
        matAInner->cols != spsmDescrInner->matACols ||
        matAInner->nnz != spsmDescrInner->matANnz ||
        matAInner->format != spsmDescrInner->matAFormat) {
        OP_LOGE("aclsparseSpSM", "matA mismatch (Solve rows=%llu cols=%llu nnz=%llu format=%d, "
                "Analysis rows=%llu cols=%llu nnz=%llu format=%d)",
                static_cast<unsigned long long>(matAInner->rows),
                static_cast<unsigned long long>(matAInner->cols),
                static_cast<unsigned long long>(matAInner->nnz),
                static_cast<int>(matAInner->format),
                static_cast<unsigned long long>(spsmDescrInner->matARows),
                static_cast<unsigned long long>(spsmDescrInner->matACols),
                static_cast<unsigned long long>(spsmDescrInner->matANnz),
                static_cast<int>(spsmDescrInner->matAFormat));
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

// 跨阶段一致性校验: analyzed 标记 + opA/opB 与 Analysis 阶段一致 +
// matA/matB/matC 的关键字段与 Analysis 阶段一致 + buffer 绑定.
static aclsparseStatus_t ValidateSpsmConsistency(aclsparseSpSMDescr *spsmDescrInner,
                                                    aclsparseOperation_t opA,
                                                    aclsparseOperation_t opB,
                                                    aclsparseSpMatDescr *matAInner,
                                                    aclsparseDnMatDescr *matBInner,
                                                    aclsparseDnMatDescr *matCInner,
                                                    void *&buffer)
{
    if (!spsmDescrInner->analyzed) {
        OP_LOGE("aclsparseSpSM", "Solve called before Analysis (not initialized)");
        return ACL_SPARSE_STATUS_NOT_INITIALIZED;
    }
    if (spsmDescrInner->opA != opA) {
        OP_LOGE("aclsparseSpSM", "opA mismatch (Solve opA=%d, Analysis opA=%d)",
                static_cast<int>(opA), static_cast<int>(spsmDescrInner->opA));
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (spsmDescrInner->opB != opB) {
        OP_LOGE("aclsparseSpSM", "opB mismatch (Solve opB=%d, Analysis opB=%d)",
                static_cast<int>(opB), static_cast<int>(spsmDescrInner->opB));
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    aclsparseStatus_t st = ValidateMatAFields(matAInner, spsmDescrInner);
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        return st;
    }
    st = ValidateMatBFields(matBInner, spsmDescrInner);
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        return st;
    }
    st = ValidateMatCFields(matCInner, spsmDescrInner);
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        return st;
    }
    buffer = spsmDescrInner->buffer;
    if (buffer == nullptr) {
        OP_LOGE("aclsparseSpSM", "buffer is nullptr");
        return ACL_SPARSE_STATUS_INSUFFICIENT_RESOURCES;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

// Solve 阶段 launch 序列 (全异步, 同 stream 顺序)
//   1) transpose-in:  orderB=COL 时 B(COL) -> denseBuf(ROW)  [SIMT kernel]
//   2) solve:         by-value tiling, 仅 ROW 路径              [MemBase kernel]
//   3) transpose-out: orderC=COL 时 denseBuf(ROW) -> C(COL)    [SIMT kernel]
// 同 stream 上 kernel 按序执行, 无需跨 kernel SyncAll。
static aclsparseStatus_t LaunchSpsmSolve(aclsparseSpMatDescr *matAInner,
                                           aclsparseDnMatDescr *matBInner,
                                           aclsparseDnMatDescr *matCInner,
                                           aclsparseSpSMDescr *spsmDescrInner,
                                           const void *alpha, void *buffer,
                                           aclrtStream stream)
{
    const uint32_t blockDim = GetSpsmBlockDim();
    // 从 cachedTiling 取出, 刷新 alpha 后 by-value 传入 kernel
    SpsmTilingData td = spsmDescrInner->cachedTiling;
    td.alpha_host = *static_cast<const float *>(alpha);

    const int32_t m = td.m;
    const int32_t n = td.n;
    void *denseBuf = static_cast<uint8_t *>(buffer) + td.denseBufOff;

    OP_LOGI("aclsparseSpSM", "launching spsm_solve kernel: blockDim=%u, L=%d, orderB=%d, orderC=%d",
            blockDim, spsmDescrInner->L, td.orderB, td.orderC);

    // 1) transpose-in: orderB=COL 时 B(COL) -> denseBuf(ROW)
    if (td.orderB == SPSM_ORDER_COL) {
        spsm_transpose_kernel_do(reinterpret_cast<GM_ADDR>(matBInner->values),
                                 reinterpret_cast<GM_ADDR>(denseBuf),
                                 m, n, td.ldb,
                                 /*COL->ROW*/ 0, blockDim, stream);
    }

    // 2) solve (by-value tiling, 异步, 不做 stream 同步 — 硬性约束)
    spsm_solve_kernel_do(reinterpret_cast<GM_ADDR>(matAInner->ptrs),
                         reinterpret_cast<GM_ADDR>(matAInner->idxs),
                         reinterpret_cast<GM_ADDR>(matAInner->values),
                         reinterpret_cast<GM_ADDR>(matBInner->values),
                         reinterpret_cast<GM_ADDR>(matCInner->values),
                         reinterpret_cast<GM_ADDR>(buffer),
                         td, blockDim, stream);

    // 3) transpose-out: orderC=COL 时 denseBuf(ROW) -> C(COL)
    if (td.orderC == SPSM_ORDER_COL) {
        spsm_transpose_kernel_do(reinterpret_cast<GM_ADDR>(denseBuf),
                                 reinterpret_cast<GM_ADDR>(matCInner->values),
                                 m, n, td.ldc,
                                 /*ROW->COL*/ 1, blockDim, stream);
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

// ============================================================================
// 阶段 3: Solve (异步, 禁止 aclrtSynchronizeStream)
// TilingData 从 cachedTiling 取出, 刷新 alpha 后 by-value 传入 kernel
// ============================================================================
aclsparseStatus_t aclsparseSpSM(
    aclsparseHandle_t handle, aclsparseOperation_t opA, aclsparseOperation_t opB,
    const void *alpha, aclsparseConstSpMatDescr_t matA,
    aclsparseConstDnMatDescr_t matB, aclsparseDnMatDescr_t matC,
    aclDataType computeType, aclsparseSpSMAlg_t alg,
    aclsparseSpSMDescr_t spsmDescr)
{
    auto in = ExtractSpsmInputs(handle, alpha, matA, matB, matC, opA, opB, computeType, alg, spsmDescr);
    if (in.status != ACL_SPARSE_STATUS_SUCCESS) {
        return in.status;
    }

    OP_LOGI("aclsparseSpSM", "Solve: opA=%d, opB=%d, m=%llu, n=%lld, nnz=%llu",
            static_cast<int>(opA), static_cast<int>(opB),
            static_cast<unsigned long long>(in.matAInner->rows), static_cast<long long>(in.matCInner->cols),
            static_cast<unsigned long long>(in.matAInner->nnz));

    // 跨阶段一致性校验
    void *buffer = nullptr;
    aclsparseStatus_t st = ValidateSpsmConsistency(in.spsmDescrInner, opA, opB,
                                                      in.matAInner, in.matBInner, in.matCInner, buffer);
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        return st;
    }

    aclrtStream stream = nullptr;
    st = aclsparseGetStream(handle, &stream);
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        OP_LOGE("aclsparseSpSM", "aclsparseGetStream failed, st=%d", static_cast<int>(st));
        return st;
    }

    return LaunchSpsmSolve(in.matAInner, in.matBInner, in.matCInner, in.spsmDescrInner,
                            alpha, buffer, stream);
}
