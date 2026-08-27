/**
 * ----------------------------------------------------------------------------------------------------------
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ----------------------------------------------------------------------------------------------------------
 */

// SpGEMM SIMT kernel for Ascend 950PR (dav-3510)。
//
// 两阶段：符号阶段（结构）+ 数值阶段（值）。
// 使用 SIMT 编程模型：__aicore__ SPMD 外层 + asc_vf_call / __simt_vf__ 内层。
// 支持 fp32 / fp16 / bf16；fp16/bf16 在 fp32 中累加。
//
// β≠0 路径：host 侧 memset valuesC 后再跑数值 kernel → β·C_in = 0
// FP16/BF16 路径：在 __simt_vf__ 中手动 IEEE754 位转换（uint16↔float），
//   因 SIMT __simt_vf__ 的 half 类型 GM 读写不可靠，读为 uint16 手动转换可修复。
// n>64 路径：GM-backed 每 block bitmap（符号）+ float 累加器（数值），
//   替代 local 数组溢出路径。
//
// 仅 arch35（__NPU_ARCH__==3510）；fp32 要求 bit-wise 确定性。

#include <stdint.h>
#include <type_traits>

#include "kernel_operator.h"
#include "simt_api/asc_simt.h"
#include "simt_api/common_functions.h"
#include "spgemm.h"

#if defined(__NPU_ARCH__) && (__NPU_ARCH__ != 3510)
#error "SpGEMM SIMT: this TU is only for dav-3510 / Ascend 950PR (__NPU_ARCH__==3510)."
#endif

using namespace AscendC;

namespace {

constexpr uint32_t kMaxSimtThreadsPerBlock = 512u;
constexpr int32_t kMaxDenseCols = 64;      /* threshold for multi-threaded local path */
constexpr int32_t kMaxLocalAccumN = 256;   /* threshold for single-threaded local path */
/* Tile width for n > kSpgemmTileN path. Each tile uses local accBuf[128] (512B)
 * + maskBuf[16] (16B) = 528B, fitting in register file without spill.
 * Aligned with spgemm.h kSpgemmTileN. */
constexpr int32_t kTileN = 128;

/* IEEE 754 手动位转换工具
 *
 * 问题：__simt_vf__ 中 static_cast<half>(float) 和 static_cast<float>(half)
 *   在 arch35 SIMT 向量核上产生 NaN/垃圾值。half 类型的 GM 加载/存储
 *   和类型转换指令在 bisheng 编译器的 __simt_vf__ 上下文中不可靠。
 *
 * 方案：将 half/bf16 作为 uint16 从 GM 读写，用手动 IEEE 754 位操作
 *   转换为/自 float。仅使用 uint32/float 操作，SIMT 可靠支持。
 */

static __simt_callee__ __aicore__ inline void HalfToFloat(uint16_t h, float &out) {
    uint32_t sign = (static_cast<uint32_t>(h) >> 15) & 1u;
    uint32_t exp  = (static_cast<uint32_t>(h) >> 10) & 0x1Fu;
    uint32_t mant = static_cast<uint32_t>(h) & 0x3FFu;

    uint32_t f;
    if (exp == 0u) {
        if (mant == 0u) {
            f = sign << 31;                                    /* ±0 */
        } else {
            /* Denormalized half → normalize */
            int32_t e = -1;
            uint32_t m = mant;
            do { ++e; m <<= 1; } while ((m & 0x400u) == 0u);
            f = (sign << 31) | ((static_cast<uint32_t>(127 - 15 - e)) << 23) | ((m & 0x3FFu) << 13);
        }
    } else if (exp == 0x1Fu) {
        f = (sign << 31) | (0xFFu << 23) | (mant << 13);      /* Inf/NaN */
    } else {
        f = (sign << 31) | ((exp + 127u - 15u) << 23) | (mant << 13);
    }

    float result;
    __builtin_memcpy(&out, &f, sizeof(float));
}

static __simt_callee__ __aicore__ inline void FloatToHalf(float val, uint16_t &out) {
    /* Saturate to FP16 range before conversion */
    constexpr float kFp16Max = 65504.0f;
    constexpr float kFp16Min = -65504.0f;
    if (val > kFp16Max) val = kFp16Max;
    if (val < kFp16Min) val = kFp16Min;

    uint32_t bits;
    __builtin_memcpy(&bits, &val, sizeof(float));

    uint32_t sign = (bits >> 31) & 1u;
    int32_t  exp  = static_cast<int32_t>((bits >> 23) & 0xFFu);
    uint32_t mant = bits & 0x7FFFFFu;

    uint16_t h;
    if (exp == 0xFF) {
        /* Inf/NaN */
        out = static_cast<uint16_t>((sign << 15) | 0x7C00u | (mant ? 1u : 0u));
    } else {
        int32_t newExp = exp - 127 + 15;
        if (newExp >= 0x1F) {
            out = static_cast<uint16_t>((sign << 15) | 0x7C00u);    /* Overflow → Inf */
        } else if (newExp <= 0) {
            if (newExp < -10) {
                out = static_cast<uint16_t>(sign << 15);             /* Underflow → 0 */
            } else {
                /* Denormalized half */
                uint32_t m = mant | 0x800000u;
                int32_t shift = 14 - newExp;
                out = static_cast<uint16_t>((sign << 15) | (m >> shift));
            }
        } else {
            /* Normalized — round to nearest even */
            uint32_t roundingBias = 0x1000u + ((mant >> 13) & 1u);
            uint32_t rounded = (mant + roundingBias) >> 13;
            if (rounded > 0x3FFu) {  /* mantissa overflow → exponent increment */
                rounded = 0u;
                newExp += 1;
            }
            if (newExp >= 0x1F) {
                out = static_cast<uint16_t>((sign << 15) | 0x7C00u);    /* Overflow → Inf */
            } else {
                out = static_cast<uint16_t>((sign << 15) | (static_cast<uint32_t>(newExp) << 10) | rounded);
            }
        }
    }
}

static __simt_callee__ __aicore__ inline void Bf16ToFloat(uint16_t b, float &out) {
    /* BF16 is the upper 16 bits of FP32 — just extend with zero lower bits */
    uint32_t f = static_cast<uint32_t>(b) << 16;
    float result;
    __builtin_memcpy(&out, &f, sizeof(float));
}

static __simt_callee__ __aicore__ inline void FloatToBf16(float val, uint16_t &out) {
    uint32_t bits;
    __builtin_memcpy(&bits, &val, sizeof(float));
    /* Round to nearest even: add 0x7FFF + (lsb of upper half) */
    uint32_t lsb = (bits >> 16) & 1u;
    uint32_t roundingBias = 0x7FFFu + lsb;
    uint32_t rounded = bits + roundingBias;
    out = static_cast<uint16_t>(rounded >> 16);
}

/* Read a value from GM as float — __simt_callee__ allows non-void return. */
template<typename ValT>
static __simt_callee__ __aicore__ inline float ReadGmAsFloat(
    __gm__ const ValT *gmBuf, int32_t idx)
{
    if constexpr (std::is_same<ValT, float>::value) {
        return gmBuf[idx];
    } else {
        __gm__ const uint16_t *u16Buf =
            reinterpret_cast<__gm__ const uint16_t *>(gmBuf);
        if constexpr (std::is_same<ValT, half>::value) {
            float out; HalfToFloat(u16Buf[idx], out); return out;
        } else {
            float out; Bf16ToFloat(u16Buf[idx], out); return out;
        }
    }
}

/* Write a float value to GM as target dtype. */
template<typename ValT>
static __simt_callee__ __aicore__ inline void WriteFloatToGm(
    __gm__ ValT *gmBuf, int32_t idx, float val)
{
    if constexpr (std::is_same<ValT, float>::value) {
        gmBuf[idx] = val;
    } else {
        __gm__ uint16_t *u16Buf =
            reinterpret_cast<__gm__ uint16_t *>(gmBuf);
        uint16_t bits = 0;
        if constexpr (std::is_same<ValT, half>::value) {
            FloatToHalf(val, bits);
        } else {
            FloatToBf16(val, bits);
        }
        u16Buf[idx] = bits;
    }
}

/* Read existing GM value as float (for β·C_in path). */
template<typename ValT>
static __simt_callee__ __aicore__ inline float ReadGmValAsFloat(
    __gm__ ValT *gmBuf, int32_t idx)
{
    return ReadGmAsFloat<ValT>(gmBuf, idx);
}

/* ---- Common helpers shared across kernel classes ---- */

/* Get row bounds from binEdge for current block. */
static __aicore__ inline void SpgemmGetRowBounds(
    int32_t blockDim, __gm__ const int32_t *binEdge,
    int32_t &rowStart, int32_t &rowEnd)
{
    int32_t outerId = static_cast<int32_t>(GetBlockIdx());
    rowStart = 0;
    rowEnd = 0;
    if (outerId >= 0 && outerId < blockDim) {
        rowStart = binEdge[outerId];
        rowEnd = binEdge[outerId + 1];
    }
}

/* Compute SIMT thread count for non-tiled kernels. */
static __aicore__ inline uint32_t SpgemmComputeSimtThreads(
    int32_t numRows, int32_t rowBinNum, int32_t n, bool forceSingleThread)
{
    if (forceSingleThread) return 1u;
    if (n > kMaxDenseCols) return 1u;  /* n > 64: single thread to avoid register pressure */

    uint32_t simtThreads = 1u;
    const uint64_t totalWork = static_cast<uint64_t>(numRows);
    if (totalWork > 0u && rowBinNum > 0) {
        simtThreads = static_cast<uint32_t>(
            (totalWork + static_cast<uint64_t>(rowBinNum) - 1u) /
            static_cast<uint64_t>(rowBinNum));
        if (simtThreads > kMaxSimtThreadsPerBlock) {
            simtThreads = kMaxSimtThreadsPerBlock;
        }
    }
    if (simtThreads < kSpgemmWarpSize) simtThreads = kSpgemmWarpSize;
    return simtThreads;
}

/* Write accumulated result to GM with β·C_in blending. */
template<typename ValT>
static __simt_callee__ __aicore__ inline void SpgemmWriteResult(
    __gm__ ValT *valuesC, int32_t outIdx,
    float alpha, float accumVal,
    float beta, bool betaZero)
{
    float result = alpha * accumVal;
    if (!betaZero) {
        float cInVal = ReadGmValAsFloat<ValT>(valuesC, outIdx);
        result += beta * cInVal;
    }
    WriteFloatToGm<ValT>(valuesC, outIdx, result);
}

/* 从 tiling 读取 alpha/beta（支持 host 指针或 device 指针）。 */
static __aicore__ inline void SpgemmReadAlphaBeta(
    const SpgemmNumericTilingData &tiling,
    float &alpha, float &beta)
{
    if (tiling.alphaPtr != 0) {
        alpha = *reinterpret_cast<__gm__ const float *>(tiling.alphaPtr);
    } else {
        alpha = tiling.alphaHost;
    }
    if (tiling.betaPtr != 0) {
        beta = *reinterpret_cast<__gm__ const float *>(tiling.betaPtr);
    } else {
        beta = tiling.betaHost;
    }
}

/* ============================================================================
 * ---- Symbolic phase SIMT compute (per-thread) ----
 * ============================================================================ */
template<bool UseGmBitmap>
__simt_vf__ __aicore__ __launch_bounds__(kMaxSimtThreadsPerBlock) inline void SpgemmSymbolicSimtCompute(
    __gm__ const int32_t *aRowPtr,
    __gm__ const int32_t *aColInd,
    __gm__ const int32_t *bRowPtr,
    __gm__ const int32_t *bColInd,
    __gm__ int32_t *nnzPerRow,
    __gm__ int32_t *reorder,
    __gm__ uint8_t *wsBase,
    int64_t gmAccumOffset,
    int32_t n,
    int32_t rowStart,
    int32_t rowEnd,
    int32_t blockId)
{
    const int32_t numRows = rowEnd - rowStart;
    if (numRows <= 0) return;

    const uint32_t threadNum = blockDim.x;
    const uint32_t tid = threadIdx.x;

    if constexpr (UseGmBitmap) {
        /* n > 64 时使用 GM-backed 每 block bitmap。
         * Only tid==0 processes rows sequentially (one bitmap per block).
         * Other threads must still participate in SIMT dispatch (no early return
         * before asc_vf_call returns). */
        if (tid != 0u) return;

        int32_t bitmapBytes = (n + 7) / 8;
        int64_t blockStride = static_cast<int64_t>(n) * static_cast<int64_t>(sizeof(float));
        __gm__ uint8_t *bitmap = wsBase + gmAccumOffset +
            static_cast<int64_t>(blockId) * blockStride;

        for (int32_t r = 0; r < numRows; ++r) {
            int32_t logicalRow = rowStart + r;
            int32_t row = reorder[logicalRow];
            int32_t aStart = aRowPtr[row];
            int32_t aEnd   = aRowPtr[row + 1];

            /* Zero bitmap */
            for (int32_t i = 0; i < bitmapBytes; ++i) bitmap[i] = 0;

            /* Set bits for each column encountered */
            for (int32_t p = aStart; p < aEnd; ++p) {
                int32_t k = aColInd[p];
                int32_t bStart = bRowPtr[k];
                int32_t bEnd   = bRowPtr[k + 1];
                for (int32_t q = bStart; q < bEnd; ++q) {
                    int32_t j = bColInd[q];
                    if (j >= 0 && j < n) {
                        bitmap[j >> 3] |= static_cast<uint8_t>(1u << (j & 7));
                    }
                }
            }

            /* Count set bits (popcount) */
            int32_t colCount = 0;
            for (int32_t i = 0; i < bitmapBytes; ++i) {
                uint8_t b = bitmap[i];
                while (b != 0) {
                    colCount += (b & 1u);
                    b >>= 1;
                }
            }
            nnzPerRow[row] = colCount;
        }
    } else {
        /* n ≤ 64: use local bitmask (original path, correct and fast) */
        for (int32_t r = static_cast<int32_t>(tid); r < numRows; r += static_cast<int32_t>(threadNum)) {
            int32_t logicalRow = rowStart + r;
            int32_t row = reorder[logicalRow];
            int32_t aStart = aRowPtr[row];
            int32_t aEnd   = aRowPtr[row + 1];
            int32_t colCount = 0;

            uint8_t maskBuf[kMaxLocalAccumN / 8];
            for (int32_t i = 0; i < (n + 7) / 8; ++i) maskBuf[i] = 0;

            for (int32_t p = aStart; p < aEnd; ++p) {
                int32_t k = aColInd[p];
                int32_t bStart = bRowPtr[k];
                int32_t bEnd   = bRowPtr[k + 1];
                for (int32_t q = bStart; q < bEnd; ++q) {
                    int32_t j = bColInd[q];
                    if (j >= 0 && j < n) {
                        int32_t byteIdx = j >> 3;
                        uint8_t bitMask = static_cast<uint8_t>(1u << (j & 7));
                        if ((maskBuf[byteIdx] & bitMask) == 0u) {
                            maskBuf[byteIdx] |= bitMask;
                            ++colCount;
                        }
                    }
                }
            }
            nnzPerRow[row] = colCount;
        }
    }
}

/* ---- 符号阶段 SIMT compute：n > 128 的分块路径 ----
 *
 * local 路径（n ≤ 256）的 maskBuf[32] 在寄存器中，但数值阶段的 accBuf[256]
 * 会导致溢出。GM-backed 路径（n > 256）有 GM 缓存一致性问题。
 *
 * 方案：将 n 按 kTileN（128）列分块。每块使用 maskBuf[16]（16 字节），
 * 遍历 A×B 一次，只统计落在该块范围内的列。无溢出风险。
 *
 * 确保符号和数值阶段使用相同的分块遍历，消除列集不一致问题。
 */
__simt_vf__ __aicore__ __launch_bounds__(kMaxSimtThreadsPerBlock) inline void SpgemmSymbolicSimtComputeTiled(
    __gm__ const int32_t *aRowPtr,
    __gm__ const int32_t *aColInd,
    __gm__ const int32_t *bRowPtr,
    __gm__ const int32_t *bColInd,
    __gm__ int32_t *nnzPerRow,
    __gm__ int32_t *reorder,
    int32_t n,
    int32_t rowStart,
    int32_t rowEnd)
{
    const int32_t numRows = rowEnd - rowStart;
    if (numRows <= 0) return;

    const uint32_t threadNum = blockDim.x;
    const uint32_t tid = threadIdx.x;

    /* Single-thread for tile path (n > 64 → register pressure concern) */
    if (tid != 0u) return;

    const int32_t numTiles = (n + kTileN - 1) / kTileN;

    for (int32_t r = 0; r < numRows; ++r) {
        int32_t logicalRow = rowStart + r;
        int32_t row = reorder[logicalRow];
        int32_t aStart = aRowPtr[row];
        int32_t aEnd   = aRowPtr[row + 1];
        int32_t colCount = 0;

        /* Flush GM cache before reading B data for this row.
         * Ensures symbolic and numeric phases see consistent B.colInd. */
        asc_threadfence();

        for (int32_t tileIdx = 0; tileIdx < numTiles; ++tileIdx) {
            int32_t tileStart = tileIdx * kTileN;
            int32_t tileEnd = tileStart + kTileN;
            if (tileEnd > n) tileEnd = n;
            int32_t tileLen = tileEnd - tileStart;
            int32_t maskBytes = (tileLen + 7) / 8;

            /* Local bitmask — max 16 bytes for tile=128, always in register file */
            uint8_t maskBuf[kTileN / 8];
            for (int32_t i = 0; i < maskBytes; ++i) maskBuf[i] = 0;

            for (int32_t p = aStart; p < aEnd; ++p) {
                int32_t k = aColInd[p];
                int32_t bStart = bRowPtr[k];
                int32_t bEnd   = bRowPtr[k + 1];
                for (int32_t q = bStart; q < bEnd; ++q) {
                    int32_t j = bColInd[q];
                    if (j >= tileStart && j < tileEnd) {
                        int32_t localJ = j - tileStart;
                        maskBuf[localJ >> 3] |= static_cast<uint8_t>(1u << (localJ & 7));
                    }
                }
            }

            /* Count set bits (popcount) for this tile */
            for (int32_t i = 0; i < maskBytes; ++i) {
                uint8_t b = maskBuf[i];
                while (b != 0) {
                    colCount += (b & 1u);
                    b >>= 1;
                }
            }
        }
        nnzPerRow[row] = colCount;
    }
}

/* ============================================================================
 * ---- Numeric phase SIMT compute (per-thread) ----
 * ============================================================================ */
template<typename ValT, bool UseGmAccum>
__simt_vf__ __aicore__ __launch_bounds__(kMaxSimtThreadsPerBlock) inline void SpgemmNumericSimtCompute(
    __gm__ const int32_t *aRowPtr,
    __gm__ const int32_t *aColInd,
    __gm__ const ValT *aValues,
    __gm__ const int32_t *bRowPtr,
    __gm__ const int32_t *bColInd,
    __gm__ const ValT *bValues,
    __gm__ const int32_t *rowPtrC,
    __gm__ int32_t *colIndC,
    __gm__ ValT *valuesC,
    __gm__ int32_t *reorder,
    __gm__ int32_t *nnzPerRow,
    __gm__ uint8_t *wsBase,
    int64_t numAccumOffset,
    int32_t n,
    float alpha,
    float beta,
    int32_t rowStart,
    int32_t rowEnd,
    int32_t blockId)
{
    const int32_t numRows = rowEnd - rowStart;
    if (numRows <= 0) return;

    const uint32_t threadNum = blockDim.x;
    const uint32_t tid = threadIdx.x;
    const bool betaZero = (beta == 0.0f);

    if constexpr (UseGmAccum) {
        /* n > 64 时使用 GM 累加器：每个 block 一块 GM 内存存放 dense float 数组。
         * 仅 tid==0 顺序处理各行。maskBuf 是小的 local 数组（最多 128 字节），
         * 放在寄存器中；accum 是较大的 GM 数组。两者分离避免寄存器溢出。
         *
         * Per-block layout: [float accum[n]] (GM)
         * Per-block stride = n * sizeof(float) */
        if (tid != 0u) return;

        int64_t blockStride = static_cast<int64_t>(n) * static_cast<int64_t>(sizeof(float));
        __gm__ float *accum = reinterpret_cast<__gm__ float *>(
            wsBase + numAccumOffset + static_cast<int64_t>(blockId) * blockStride);

        /* Local bitmask — max 128 bytes for n=1024, fits in register file */
        constexpr int32_t kMaxMaskBytes = 128;

        for (int32_t r = 0; r < numRows; ++r) {
            int32_t logicalRow = rowStart + r;
            int32_t row = reorder[logicalRow];
            int32_t aStart = aRowPtr[row];
            int32_t aEnd   = aRowPtr[row + 1];
            int32_t cStart = rowPtrC[row];

            int32_t maskBytes = (n + 7) / 8;
            if (maskBytes > kMaxMaskBytes) maskBytes = kMaxMaskBytes;

            /* Zero accumulator (GM) + local maskBuf (register) */
            uint8_t maskBuf[kMaxMaskBytes];
            for (int32_t j = 0; j < n; ++j) accum[j] = 0.0f;
            for (int32_t i = 0; i < maskBytes; ++i) maskBuf[i] = 0;

            /* Accumulate products + set mask bits — same logic as n≤64 path */
            for (int32_t p = aStart; p < aEnd; ++p) {
                int32_t k = aColInd[p];
                float aVal = ReadGmAsFloat<ValT>(aValues, p);
                int32_t bStart = bRowPtr[k];
                int32_t bEnd   = bRowPtr[k + 1];
                for (int32_t q = bStart; q < bEnd; ++q) {
                    int32_t j = bColInd[q];
                    if (j >= 0 && j < n) {
                        float bVal = ReadGmAsFloat<ValT>(bValues, q);
                        accum[j] += aVal * bVal;
                        maskBuf[j >> 3] |= static_cast<uint8_t>(1u << (j & 7));
                    }
                }
            }

            /* Output sorted by column — use local maskBuf (NOT accum!=0) */
            int32_t outIdx = cStart;
            for (int32_t j = 0; j < n; ++j) {
                if (maskBuf[j >> 3] & static_cast<uint8_t>(1u << (j & 7))) {
                    colIndC[outIdx] = j;
                    SpgemmWriteResult<ValT>(valuesC, outIdx, alpha, accum[j], beta, betaZero);
                    ++outIdx;
                }
            }
        }
    } else {
        /* n ≤ 64: local dense accumulator (original path, correct and fast) */
        for (int32_t r = static_cast<int32_t>(tid); r < numRows; r += static_cast<int32_t>(threadNum)) {
            int32_t logicalRow = rowStart + r;
            int32_t row = reorder[logicalRow];
            int32_t aStart = aRowPtr[row];
            int32_t aEnd   = aRowPtr[row + 1];
            int32_t cStart = rowPtrC[row];

            float accBuf[kMaxLocalAccumN];
            uint8_t maskBuf[kMaxLocalAccumN / 8];
            for (int32_t i = 0; i < n; ++i) accBuf[i] = 0.0f;
            for (int32_t i = 0; i < (n + 7) / 8; ++i) maskBuf[i] = 0;

            for (int32_t p = aStart; p < aEnd; ++p) {
                int32_t k = aColInd[p];
                float aVal = ReadGmAsFloat<ValT>(aValues, p);
                int32_t bStart = bRowPtr[k];
                int32_t bEnd   = bRowPtr[k + 1];
                for (int32_t q = bStart; q < bEnd; ++q) {
                    int32_t j = bColInd[q];
                    if (j >= 0 && j < n) {
                        float bVal = ReadGmAsFloat<ValT>(bValues, q);
                        accBuf[j] += aVal * bVal;
                        maskBuf[j >> 3] |= static_cast<uint8_t>(1u << (j & 7));
                    }
                }
            }

            int32_t outIdx = cStart;
            for (int32_t j = 0; j < n; ++j) {
                int32_t byteIdx = j >> 3;
                uint8_t bitMask = static_cast<uint8_t>(1u << (j & 7));
                if (maskBuf[byteIdx] & bitMask) {
                    colIndC[outIdx] = j;
                    SpgemmWriteResult<ValT>(valuesC, outIdx, alpha, accBuf[j], beta, betaZero);
                    ++outIdx;
                }
            }
        }
    }
}

/* ============================================================================
 * ---- Numeric phase SIMT compute: tile-based path for n > 128 ----
 *
 * Same tiling strategy as SpgemmSymbolicSimtComputeTiled.
 * Each tile uses local accBuf[kTileN] (512B) + maskBuf[kTileN/8] (16B) = 528B,
 * which fits in the SIMT register file without spill.
 *
 * For each tile, traverse A.row × B.row and accumulate only columns within
 * the tile range [tileStart, tileEnd). Output sorted by column within each
 * tile, advancing outIdx sequentially across tiles.
 *
 * This ensures the numeric phase's column set exactly matches the symbolic
 * phase's column set (same tile boundaries, same traversal order), because
 * both phases use the same local-sized maskBuf with no GM spill interference.
 * ============================================================================ */
template<typename ValT>
__simt_vf__ __aicore__ __launch_bounds__(kMaxSimtThreadsPerBlock) inline void SpgemmNumericSimtComputeTiled(
    __gm__ const int32_t *aRowPtr,
    __gm__ const int32_t *aColInd,
    __gm__ const ValT *aValues,
    __gm__ const int32_t *bRowPtr,
    __gm__ const int32_t *bColInd,
    __gm__ const ValT *bValues,
    __gm__ const int32_t *rowPtrC,
    __gm__ int32_t *colIndC,
    __gm__ ValT *valuesC,
    __gm__ int32_t *reorder,
    int32_t n,
    float alpha,
    float beta,
    int32_t rowStart,
    int32_t rowEnd)
{
    const int32_t numRows = rowEnd - rowStart;
    if (numRows <= 0) return;

    const uint32_t tid = threadIdx.x;
    const bool betaZero = (beta == 0.0f);

    /* Single-thread for tile path */
    if (tid != 0u) return;

    const int32_t numTiles = (n + kTileN - 1) / kTileN;

    for (int32_t r = 0; r < numRows; ++r) {
        int32_t logicalRow = rowStart + r;
        int32_t row = reorder[logicalRow];
        int32_t aStart = aRowPtr[row];
        int32_t aEnd   = aRowPtr[row + 1];
        int32_t cStart = rowPtrC[row];

        int32_t outIdx = cStart;

        /* Flush GM cache before reading B data for this row.
         * Ensures symbolic and numeric phases see consistent B.colInd. */
        asc_threadfence();

        for (int32_t tileIdx = 0; tileIdx < numTiles; ++tileIdx) {
            int32_t tileStart = tileIdx * kTileN;
            int32_t tileEnd = tileStart + kTileN;
            if (tileEnd > n) tileEnd = n;
            int32_t tileLen = tileEnd - tileStart;
            int32_t maskBytes = (tileLen + 7) / 8;

            /* Local dense accumulator + bitmask — 528B total, no spill */
            float accBuf[kTileN];
            uint8_t maskBuf[kTileN / 8];
            for (int32_t i = 0; i < tileLen; ++i) accBuf[i] = 0.0f;
            for (int32_t i = 0; i < maskBytes; ++i) maskBuf[i] = 0;

            /* Accumulate products for columns in [tileStart, tileEnd) */
            for (int32_t p = aStart; p < aEnd; ++p) {
                int32_t k = aColInd[p];
                float aVal = ReadGmAsFloat<ValT>(aValues, p);
                int32_t bStart = bRowPtr[k];
                int32_t bEnd   = bRowPtr[k + 1];
                for (int32_t q = bStart; q < bEnd; ++q) {
                    int32_t j = bColInd[q];
                    if (j >= tileStart && j < tileEnd) {
                        int32_t localJ = j - tileStart;
                        float bVal = ReadGmAsFloat<ValT>(bValues, q);
                        accBuf[localJ] += aVal * bVal;
                        maskBuf[localJ >> 3] |= static_cast<uint8_t>(1u << (localJ & 7));
                    }
                }
            }

            /* Output sorted by column within this tile.
             * "宁多不漏": output all maskBuf-marked columns, including
             * those where accum cancelled to 0. */
            for (int32_t localJ = 0; localJ < tileLen; ++localJ) {
                if (maskBuf[localJ >> 3] & static_cast<uint8_t>(1u << (localJ & 7))) {
                    int32_t j = tileStart + localJ;
                    colIndC[outIdx] = j;
                    SpgemmWriteResult<ValT>(valuesC, outIdx, alpha, accBuf[localJ], beta, betaZero);
                    ++outIdx;
                }
            }
        }
    }
}

template<bool UseGmBitmap>
class KernelSpgemmSymbolic {
public:
    __aicore__ inline KernelSpgemmSymbolic() {}

    __aicore__ inline void Init(
        GM_ADDR aRowPtrGM, GM_ADDR aColIndGM,
        GM_ADDR bRowPtrGM, GM_ADDR bColIndGM,
        GM_ADDR nnzPerRowGM, GM_ADDR workspaceGM,
        const SpgemmSymbolicTilingData &tiling)
    {
        tiling_ = tiling;
        aRowPtr_  = reinterpret_cast<__gm__ const int32_t *>(aRowPtrGM);
        aColInd_  = reinterpret_cast<__gm__ const int32_t *>(aColIndGM);
        bRowPtr_  = reinterpret_cast<__gm__ const int32_t *>(bRowPtrGM);
        bColInd_  = reinterpret_cast<__gm__ const int32_t *>(bColIndGM);
        nnzPerRow_ = reinterpret_cast<__gm__ int32_t *>(nnzPerRowGM);
        wsBase_   = reinterpret_cast<__gm__ uint8_t *>(workspaceGM);
        reorder_  = reinterpret_cast<__gm__ int32_t *>(wsBase_ + tiling_.reorderOffset);
        binEdge_  = reinterpret_cast<__gm__ int32_t *>(wsBase_ + tiling_.binEdgeOffset);
    }

    __aicore__ inline void Process()
    {
        int32_t rowBinNum = tiling_.blockDim;
        int32_t rowStart = 0, rowEnd = 0;
        SpgemmGetRowBounds(rowBinNum, binEdge_, rowStart, rowEnd);

        /* 不能对空 bin 提前 return（950PR 上有死锁风险），每个核都必须到达 asc_vf_call。 */
        const int32_t numRows = rowEnd - rowStart;
        uint32_t simtThreads = SpgemmComputeSimtThreads(
            numRows, rowBinNum, tiling_.n, /*forceSingleThread=*/UseGmBitmap);
        int32_t outerId = static_cast<int32_t>(GetBlockIdx());

        asc_vf_call<SpgemmSymbolicSimtCompute<UseGmBitmap>>(
            dim3{simtThreads},
            aRowPtr_, aColInd_, bRowPtr_, bColInd_, nnzPerRow_,
            reorder_, wsBase_, tiling_.symBitmapOffset,
            tiling_.n, rowStart, rowEnd, outerId);
    }

private:
    __gm__ const int32_t *aRowPtr_{nullptr};
    __gm__ const int32_t *aColInd_{nullptr};
    __gm__ const int32_t *bRowPtr_{nullptr};
    __gm__ const int32_t *bColInd_{nullptr};
    __gm__ int32_t *nnzPerRow_{nullptr};
    __gm__ uint8_t *wsBase_{nullptr};
    __gm__ int32_t *reorder_{nullptr};
    __gm__ int32_t *binEdge_{nullptr};
    SpgemmSymbolicTilingData tiling_{};
};

/* Tile-based symbolic kernel for n > 128.
 * Calls SpgemmSymbolicSimtComputeTiled which splits n into tiles of kTileN. */
class KernelSpgemmSymbolicTiled {
public:
    __aicore__ inline KernelSpgemmSymbolicTiled() {}

    __aicore__ inline void Init(
        GM_ADDR aRowPtrGM, GM_ADDR aColIndGM,
        GM_ADDR bRowPtrGM, GM_ADDR bColIndGM,
        GM_ADDR nnzPerRowGM, GM_ADDR workspaceGM,
        const SpgemmSymbolicTilingData &tiling)
    {
        tiling_ = tiling;
        aRowPtr_  = reinterpret_cast<__gm__ const int32_t *>(aRowPtrGM);
        aColInd_  = reinterpret_cast<__gm__ const int32_t *>(aColIndGM);
        bRowPtr_  = reinterpret_cast<__gm__ const int32_t *>(bRowPtrGM);
        bColInd_  = reinterpret_cast<__gm__ const int32_t *>(bColIndGM);
        nnzPerRow_ = reinterpret_cast<__gm__ int32_t *>(nnzPerRowGM);
        wsBase_   = reinterpret_cast<__gm__ uint8_t *>(workspaceGM);
        reorder_  = reinterpret_cast<__gm__ int32_t *>(wsBase_ + tiling_.reorderOffset);
        binEdge_  = reinterpret_cast<__gm__ int32_t *>(wsBase_ + tiling_.binEdgeOffset);
    }

    __aicore__ inline void Process()
    {
        int32_t rowBinNum = tiling_.blockDim;
        int32_t rowStart = 0, rowEnd = 0;
        SpgemmGetRowBounds(rowBinNum, binEdge_, rowStart, rowEnd);

        /* 每个核都必须到达 asc_vf_call，否则死锁。 */
        asc_vf_call<SpgemmSymbolicSimtComputeTiled>(
            dim3{1u},
            aRowPtr_, aColInd_, bRowPtr_, bColInd_, nnzPerRow_,
            reorder_, tiling_.n, rowStart, rowEnd);
    }

private:
    __gm__ const int32_t *aRowPtr_{nullptr};
    __gm__ const int32_t *aColInd_{nullptr};
    __gm__ const int32_t *bRowPtr_{nullptr};
    __gm__ const int32_t *bColInd_{nullptr};
    __gm__ int32_t *nnzPerRow_{nullptr};
    __gm__ uint8_t *wsBase_{nullptr};
    __gm__ int32_t *reorder_{nullptr};
    __gm__ int32_t *binEdge_{nullptr};
    SpgemmSymbolicTilingData tiling_{};
};

template<typename ValT, bool UseGmAccum>
class KernelSpgemmNumeric {
public:
    __aicore__ inline KernelSpgemmNumeric() {}

    __aicore__ inline void Init(
        GM_ADDR aRowPtrGM, GM_ADDR aColIndGM, GM_ADDR aValuesGM,
        GM_ADDR bRowPtrGM, GM_ADDR bColIndGM, GM_ADDR bValuesGM,
        GM_ADDR rowPtrCGM, GM_ADDR colIndCGM, GM_ADDR valuesCGM,
        GM_ADDR nnzPerRowGM, GM_ADDR workspaceGM,
        const SpgemmNumericTilingData &tiling)
    {
        tiling_ = tiling;
        aRowPtr_  = reinterpret_cast<__gm__ const int32_t *>(aRowPtrGM);
        aColInd_  = reinterpret_cast<__gm__ const int32_t *>(aColIndGM);
        aValues_  = reinterpret_cast<__gm__ const ValT *>(aValuesGM);
        bRowPtr_  = reinterpret_cast<__gm__ const int32_t *>(bRowPtrGM);
        bColInd_  = reinterpret_cast<__gm__ const int32_t *>(bColIndGM);
        bValues_  = reinterpret_cast<__gm__ const ValT *>(bValuesGM);
        rowPtrC_  = reinterpret_cast<__gm__ const int32_t *>(rowPtrCGM);
        colIndC_  = reinterpret_cast<__gm__ int32_t *>(colIndCGM);
        valuesC_  = reinterpret_cast<__gm__ ValT *>(valuesCGM);
        nnzPerRow_ = reinterpret_cast<__gm__ int32_t *>(nnzPerRowGM);
        wsBase_   = reinterpret_cast<__gm__ uint8_t *>(workspaceGM);
        reorder_  = reinterpret_cast<__gm__ int32_t *>(wsBase_ + tiling_.reorderOffset);
        binEdge_  = reinterpret_cast<__gm__ int32_t *>(wsBase_ + tiling_.binEdgeOffset);

        SpgemmReadAlphaBeta(tiling_, alpha_, beta_);
    }

    __aicore__ inline void Process()
    {
        int32_t rowBinNum = tiling_.blockDim;
        int32_t rowStart = 0, rowEnd = 0;
        SpgemmGetRowBounds(rowBinNum, binEdge_, rowStart, rowEnd);

        /* 每个核都必须到达 asc_vf_call，否则死锁。 */
        const int32_t numRows = rowEnd - rowStart;
        uint32_t simtThreads = SpgemmComputeSimtThreads(
            numRows, rowBinNum, tiling_.n, /*forceSingleThread=*/UseGmAccum);
        int32_t outerId = static_cast<int32_t>(GetBlockIdx());

        asc_vf_call<SpgemmNumericSimtCompute<ValT, UseGmAccum>>(
            dim3{simtThreads},
            aRowPtr_, aColInd_, aValues_,
            bRowPtr_, bColInd_, bValues_,
            rowPtrC_, colIndC_, valuesC_,
            reorder_, nnzPerRow_,
            wsBase_, tiling_.numAccumOffset,
            tiling_.n, alpha_, beta_, rowStart, rowEnd, outerId);
    }

private:
    __gm__ const int32_t *aRowPtr_{nullptr};
    __gm__ const int32_t *aColInd_{nullptr};
    __gm__ const ValT *aValues_{nullptr};
    __gm__ const int32_t *bRowPtr_{nullptr};
    __gm__ const int32_t *bColInd_{nullptr};
    __gm__ const ValT *bValues_{nullptr};
    __gm__ const int32_t *rowPtrC_{nullptr};
    __gm__ int32_t *colIndC_{nullptr};
    __gm__ ValT *valuesC_{nullptr};
    __gm__ int32_t *nnzPerRow_{nullptr};
    __gm__ uint8_t *wsBase_{nullptr};
    __gm__ int32_t *reorder_{nullptr};
    __gm__ int32_t *binEdge_{nullptr};
    SpgemmNumericTilingData tiling_{};
    float alpha_{0.0f};
    float beta_{0.0f};
};

/* Tile-based numeric kernel for n > 128.
 * Calls SpgemmNumericSimtComputeTiled which splits n into tiles of kTileN.
 * Each tile uses accBuf[128] (512B) + maskBuf[16] (16B) = 528B, no spill. */
template<typename ValT>
class KernelSpgemmNumericTiled {
public:
    __aicore__ inline KernelSpgemmNumericTiled() {}

    __aicore__ inline void Init(
        GM_ADDR aRowPtrGM, GM_ADDR aColIndGM, GM_ADDR aValuesGM,
        GM_ADDR bRowPtrGM, GM_ADDR bColIndGM, GM_ADDR bValuesGM,
        GM_ADDR rowPtrCGM, GM_ADDR colIndCGM, GM_ADDR valuesCGM,
        GM_ADDR nnzPerRowGM, GM_ADDR workspaceGM,
        const SpgemmNumericTilingData &tiling)
    {
        tiling_ = tiling;
        aRowPtr_  = reinterpret_cast<__gm__ const int32_t *>(aRowPtrGM);
        aColInd_  = reinterpret_cast<__gm__ const int32_t *>(aColIndGM);
        aValues_  = reinterpret_cast<__gm__ const ValT *>(aValuesGM);
        bRowPtr_  = reinterpret_cast<__gm__ const int32_t *>(bRowPtrGM);
        bColInd_  = reinterpret_cast<__gm__ const int32_t *>(bColIndGM);
        bValues_  = reinterpret_cast<__gm__ const ValT *>(bValuesGM);
        rowPtrC_  = reinterpret_cast<__gm__ const int32_t *>(rowPtrCGM);
        colIndC_  = reinterpret_cast<__gm__ int32_t *>(colIndCGM);
        valuesC_  = reinterpret_cast<__gm__ ValT *>(valuesCGM);
        wsBase_   = reinterpret_cast<__gm__ uint8_t *>(workspaceGM);
        reorder_  = reinterpret_cast<__gm__ int32_t *>(wsBase_ + tiling_.reorderOffset);
        binEdge_  = reinterpret_cast<__gm__ int32_t *>(wsBase_ + tiling_.binEdgeOffset);

        SpgemmReadAlphaBeta(tiling_, alpha_, beta_);
    }

    __aicore__ inline void Process()
    {
        int32_t rowBinNum = tiling_.blockDim;
        int32_t rowStart = 0, rowEnd = 0;
        SpgemmGetRowBounds(rowBinNum, binEdge_, rowStart, rowEnd);

        /* 每个核都必须到达 asc_vf_call，否则死锁。 */
        asc_vf_call<SpgemmNumericSimtComputeTiled<ValT>>(
            dim3{1u},
            aRowPtr_, aColInd_, aValues_,
            bRowPtr_, bColInd_, bValues_,
            rowPtrC_, colIndC_, valuesC_,
            reorder_, tiling_.n, alpha_, beta_, rowStart, rowEnd);
    }

private:
    __gm__ const int32_t *aRowPtr_{nullptr};
    __gm__ const int32_t *aColInd_{nullptr};
    __gm__ const ValT *aValues_{nullptr};
    __gm__ const int32_t *bRowPtr_{nullptr};
    __gm__ const int32_t *bColInd_{nullptr};
    __gm__ const ValT *bValues_{nullptr};
    __gm__ const int32_t *rowPtrC_{nullptr};
    __gm__ int32_t *colIndC_{nullptr};
    __gm__ ValT *valuesC_{nullptr};
    __gm__ uint8_t *wsBase_{nullptr};
    __gm__ int32_t *reorder_{nullptr};
    __gm__ int32_t *binEdge_{nullptr};
    SpgemmNumericTilingData tiling_{};
    float alpha_{0.0f};
    float beta_{0.0f};
};

} // namespace

/* ---- GM tiling loaders ---- */
__aicore__ inline SpgemmSymbolicTilingData LoadSymbolicTiling(GM_ADDR tilingGM)
{
    __gm__ const SpgemmSymbolicTilingData *p =
        reinterpret_cast<__gm__ const SpgemmSymbolicTilingData *>(tilingGM);
    SpgemmSymbolicTilingData t;
    t.m = p->m; t.n = p->n; t.blockDim = p->blockDim;
    t.baseA = p->baseA; t.baseB = p->baseB; t.baseC = p->baseC;
    t.reorderOffset = p->reorderOffset; t.binEdgeOffset = p->binEdgeOffset;
    t.symBitmapOffset = p->symBitmapOffset;
    return t;
}

__aicore__ inline SpgemmPrefixSumTilingData LoadPrefixSumTiling(GM_ADDR tilingGM)
{
    __gm__ const SpgemmPrefixSumTilingData *p =
        reinterpret_cast<__gm__ const SpgemmPrefixSumTilingData *>(tilingGM);
    SpgemmPrefixSumTilingData t;
    t.m = p->m; t.baseC = p->baseC;
    return t;
}

__aicore__ inline SpgemmNumericTilingData LoadNumericTiling(GM_ADDR tilingGM)
{
    __gm__ const SpgemmNumericTilingData *p =
        reinterpret_cast<__gm__ const SpgemmNumericTilingData *>(tilingGM);
    SpgemmNumericTilingData t;
    t.m = p->m; t.n = p->n; t.blockDim = p->blockDim;
    t.baseA = p->baseA; t.baseB = p->baseB; t.baseC = p->baseC;
    t.reorderOffset = p->reorderOffset; t.binEdgeOffset = p->binEdgeOffset;
    t.rowPtrCOffset = p->rowPtrCOffset; t.nnzPerRowOffset = p->nnzPerRowOffset;
    t.accumOffset = p->accumOffset; t.numAccumOffset = p->numAccumOffset;
    t.symBitmapOffset = p->symBitmapOffset;
    t.alphaHost = p->alphaHost; t.betaHost = p->betaHost;
    t.alphaPtr = p->alphaPtr; t.betaPtr = p->betaPtr;
    return t;
}

__aicore__ inline SpgemmFillTilingData LoadFillTiling(GM_ADDR tilingGM)
{
    __gm__ const SpgemmFillTilingData *p =
        reinterpret_cast<__gm__ const SpgemmFillTilingData *>(tilingGM);
    SpgemmFillTilingData t;
    t.count = p->count; t.value = p->value;
    return t;
}

/* ===== Kernel entry points ===== */

extern "C" __global__ __aicore__ void spgemm_symbolic_kernel(
    GM_ADDR aRowPtrGM, GM_ADDR aColIndGM,
    GM_ADDR bRowPtrGM, GM_ADDR bColIndGM,
    GM_ADDR nnzPerRowGM, GM_ADDR workspaceGM,
    GM_ADDR tilingGM)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    SpgemmSymbolicTilingData tiling = LoadSymbolicTiling(tilingGM);
    if (tiling.n <= kTileN) {
        /* n ≤ 128: local bitmask path, multi-threaded when n ≤ 64 */
        KernelSpgemmSymbolic<false> op;
        op.Init(aRowPtrGM, aColIndGM, bRowPtrGM, bColIndGM, nnzPerRowGM, workspaceGM, tiling);
        op.Process();
    } else {
        /* n > 128: tile-based path, split n into ≤128 tiles.
         * Each tile uses maskBuf[16] (16B), no register spill.
         * Replaces previous GM-backed path that had cache coherence issues. */
        KernelSpgemmSymbolicTiled op;
        op.Init(aRowPtrGM, aColIndGM, bRowPtrGM, bColIndGM, nnzPerRowGM, workspaceGM, tiling);
        op.Process();
    }
}

extern "C" __global__ __aicore__ void spgemm_prefixsum_kernel(
    GM_ADDR nnzPerRowGM, GM_ADDR rowPtrCGM, GM_ADDR nnzCGM,
    GM_ADDR tilingGM)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if (GetBlockIdx() != 0) return;

    SpgemmPrefixSumTilingData tiling = LoadPrefixSumTiling(tilingGM);
    __gm__ const int32_t *nnzPerRow = reinterpret_cast<__gm__ const int32_t *>(nnzPerRowGM);
    __gm__ int32_t *rowPtrC = reinterpret_cast<__gm__ int32_t *>(rowPtrCGM);
    __gm__ int32_t *nnzCOut = reinterpret_cast<__gm__ int32_t *>(nnzCGM);

    int32_t prefix = tiling.baseC;
    rowPtrC[0] = prefix;
    for (int32_t i = 0; i < tiling.m; ++i) {
        prefix += nnzPerRow[i];
        rowPtrC[i + 1] = prefix;
    }
    nnzCOut[0] = prefix - tiling.baseC;
}

extern "C" __global__ __aicore__ void spgemm_numeric_kernel_fp32(
    GM_ADDR aRowPtr, GM_ADDR aColInd, GM_ADDR aValues,
    GM_ADDR bRowPtr, GM_ADDR bColInd, GM_ADDR bValues,
    GM_ADDR rowPtrC, GM_ADDR colIndC, GM_ADDR valuesC,
    GM_ADDR nnzPerRow, GM_ADDR workspaceGM,
    GM_ADDR tilingGM)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    SpgemmNumericTilingData tiling = LoadNumericTiling(tilingGM);
    if (tiling.n <= kTileN) {
        /* n ≤ 128: local dense accumulator path */
        KernelSpgemmNumeric<float, false> op;
        op.Init(aRowPtr, aColInd, aValues, bRowPtr, bColInd, bValues,
                rowPtrC, colIndC, valuesC, nnzPerRow, workspaceGM, tiling);
        op.Process();
    } else {
        /* n > 128: tile-based path, each tile uses accBuf[128]+maskBuf[16]=528B */
        KernelSpgemmNumericTiled<float> op;
        op.Init(aRowPtr, aColInd, aValues, bRowPtr, bColInd, bValues,
                rowPtrC, colIndC, valuesC, nnzPerRow, workspaceGM, tiling);
        op.Process();
    }
}

extern "C" __global__ __aicore__ void spgemm_numeric_kernel_fp16(
    GM_ADDR aRowPtr, GM_ADDR aColInd, GM_ADDR aValues,
    GM_ADDR bRowPtr, GM_ADDR bColInd, GM_ADDR bValues,
    GM_ADDR rowPtrC, GM_ADDR colIndC, GM_ADDR valuesC,
    GM_ADDR nnzPerRow, GM_ADDR workspaceGM,
    GM_ADDR tilingGM)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    SpgemmNumericTilingData tiling = LoadNumericTiling(tilingGM);
    if (tiling.n <= kTileN) {
        KernelSpgemmNumeric<half, false> op;
        op.Init(aRowPtr, aColInd, aValues, bRowPtr, bColInd, bValues,
                rowPtrC, colIndC, valuesC, nnzPerRow, workspaceGM, tiling);
        op.Process();
    } else {
        KernelSpgemmNumericTiled<half> op;
        op.Init(aRowPtr, aColInd, aValues, bRowPtr, bColInd, bValues,
                rowPtrC, colIndC, valuesC, nnzPerRow, workspaceGM, tiling);
        op.Process();
    }
}

extern "C" __global__ __aicore__ void spgemm_numeric_kernel_bf16(
    GM_ADDR aRowPtr, GM_ADDR aColInd, GM_ADDR aValues,
    GM_ADDR bRowPtr, GM_ADDR bColInd, GM_ADDR bValues,
    GM_ADDR rowPtrC, GM_ADDR colIndC, GM_ADDR valuesC,
    GM_ADDR nnzPerRow, GM_ADDR workspaceGM,
    GM_ADDR tilingGM)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    SpgemmNumericTilingData tiling = LoadNumericTiling(tilingGM);
    if (tiling.n <= kTileN) {
        KernelSpgemmNumeric<bfloat16_t, false> op;
        op.Init(aRowPtr, aColInd, aValues, bRowPtr, bColInd, bValues,
                rowPtrC, colIndC, valuesC, nnzPerRow, workspaceGM, tiling);
        op.Process();
    } else {
        KernelSpgemmNumericTiled<bfloat16_t> op;
        op.Init(aRowPtr, aColInd, aValues, bRowPtr, bColInd, bValues,
                rowPtrC, colIndC, valuesC, nnzPerRow, workspaceGM, tiling);
        op.Process();
    }
}

extern "C" __global__ __aicore__ void spgemm_fill_kernel(
    GM_ADDR dstGM, GM_ADDR tilingGM)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if (GetBlockIdx() != 0) return;
    SpgemmFillTilingData tiling = LoadFillTiling(tilingGM);
    __gm__ int32_t *dst = reinterpret_cast<__gm__ int32_t *>(dstGM);
    for (int32_t i = 0; i < tiling.count; ++i) {
        dst[i] = tiling.value;
    }
}

/* ===== Host-side launch dispatchers ===== */

void spgemm_symbolic_kernel_do(
    GM_ADDR aRowPtr, GM_ADDR aColInd,
    GM_ADDR bRowPtr, GM_ADDR bColInd,
    GM_ADDR nnzPerRow, GM_ADDR workspaceGM,
    GM_ADDR tilingGM,
    uint32_t numBlocks, void *stream)
{
    spgemm_symbolic_kernel<<<numBlocks, nullptr, stream>>>(
        aRowPtr, aColInd, bRowPtr, bColInd, nnzPerRow, workspaceGM, tilingGM);
}

void spgemm_prefixsum_kernel_do(
    GM_ADDR nnzPerRow, GM_ADDR rowPtrC, GM_ADDR nnzCDev,
    GM_ADDR tilingGM, void *stream)
{
    spgemm_prefixsum_kernel<<<1, nullptr, stream>>>(
        nnzPerRow, rowPtrC, nnzCDev, tilingGM);
}

void spgemm_numeric_kernel_do(
    GM_ADDR aRowPtr, GM_ADDR aColInd, GM_ADDR aValues,
    GM_ADDR bRowPtr, GM_ADDR bColInd, GM_ADDR bValues,
    GM_ADDR rowPtrC, GM_ADDR colIndC, GM_ADDR valuesC,
    GM_ADDR nnzPerRow, GM_ADDR workspaceGM,
    GM_ADDR tilingGM,
    int32_t dataType, uint32_t numBlocks, void *stream)
{
    if (dataType == SPGEMM_DTYPE_FP32) {
        spgemm_numeric_kernel_fp32<<<numBlocks, nullptr, stream>>>(
            aRowPtr, aColInd, aValues, bRowPtr, bColInd, bValues,
            rowPtrC, colIndC, valuesC, nnzPerRow, workspaceGM, tilingGM);
    } else if (dataType == SPGEMM_DTYPE_FP16) {
        spgemm_numeric_kernel_fp16<<<numBlocks, nullptr, stream>>>(
            aRowPtr, aColInd, aValues, bRowPtr, bColInd, bValues,
            rowPtrC, colIndC, valuesC, nnzPerRow, workspaceGM, tilingGM);
    } else if (dataType == SPGEMM_DTYPE_BF16) {
        spgemm_numeric_kernel_bf16<<<numBlocks, nullptr, stream>>>(
            aRowPtr, aColInd, aValues, bRowPtr, bColInd, bValues,
            rowPtrC, colIndC, valuesC, nnzPerRow, workspaceGM, tilingGM);
    }
}

void spgemm_fill_kernel_do(
    GM_ADDR dst, GM_ADDR tilingGM, void *stream)
{
    spgemm_fill_kernel<<<1, nullptr, stream>>>(dst, tilingGM);
}
