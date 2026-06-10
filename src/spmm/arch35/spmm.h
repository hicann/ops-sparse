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

#ifndef SPMM_H
#define SPMM_H

#include <stdint.h>
#include <unistd.h>
#include <acl/acl_base_rt.h>

#ifndef __gm__
#define __gm__
#endif

#ifndef SPMM_ROUNDUP
#define SPMM_ROUNDUP(a, b) (((a) + (b) - 1) / (b) * (b))
#endif

// TilingKey bit layout (8 active bits, see 2.3.3 of the design doc):
//   bit 0      : algo        (0=ALG_DEFAULT, 1=CSR_ALG1)
//   bit 1      : opB         (0=N, 1=T)
//   bits 2-3   : order_pair  (0=RR, 1=RC, 2=CR, 3=CC)
//   bit 4      : beta_zero   (1 => skip C-old read)
//   bits 5-7   : tile_combo  (0..7, indexes preset (m_tile,n_tile,k_pack) table)
//   bits 8-63  : reserved
#define SPMM_TK_BIT_ALGO        0
#define SPMM_TK_BIT_OPB         1
#define SPMM_TK_BIT_ORDER_PAIR  2
#define SPMM_TK_BIT_BETA_ZERO   4
#define SPMM_TK_BIT_TILE_COMBO  5

#define SPMM_TK_GET_ALGO(k)        ((uint32_t)(((k) >> SPMM_TK_BIT_ALGO)       & 0x1u))
#define SPMM_TK_GET_OPB(k)         ((uint32_t)(((k) >> SPMM_TK_BIT_OPB)        & 0x1u))
#define SPMM_TK_GET_ORDER_PAIR(k)  ((uint32_t)(((k) >> SPMM_TK_BIT_ORDER_PAIR) & 0x3u))
#define SPMM_TK_GET_BETA_ZERO(k)   ((uint32_t)(((k) >> SPMM_TK_BIT_BETA_ZERO)  & 0x1u))
#define SPMM_TK_GET_TILE_COMBO(k)  ((uint32_t)(((k) >> SPMM_TK_BIT_TILE_COMBO) & 0x7u))

#define SPMM_TK_PACK(algo, opB, order_pair, beta_zero, tile_combo)              \
    ((uint64_t)(((uint64_t)((algo)       & 0x1u) << SPMM_TK_BIT_ALGO)       |   \
                ((uint64_t)((opB)        & 0x1u) << SPMM_TK_BIT_OPB)        |   \
                ((uint64_t)((order_pair) & 0x3u) << SPMM_TK_BIT_ORDER_PAIR) |   \
                ((uint64_t)((beta_zero)  & 0x1u) << SPMM_TK_BIT_BETA_ZERO)  |   \
                ((uint64_t)((tile_combo) & 0x7u) << SPMM_TK_BIT_TILE_COMBO)))

// Algorithm tags decoded out of TilingKey.
#define SPMM_ALGO_DEFAULT  0
#define SPMM_ALGO_CSR_ALG1 1

// Order pair tags (orderB, orderC).
#define SPMM_ORDER_RR 0
#define SPMM_ORDER_RC 1
#define SPMM_ORDER_CR 2
#define SPMM_ORDER_CC 3

// Meta flag bits (workspace header word, decoupled from TilingKey).
#define SPMM_META_PREPROCESS_DONE 0x1u

// Data type encoding for kernel dispatch.
#define SPMM_DTYPE_FP32 0
#define SPMM_DTYPE_FP16 1
#define SPMM_DTYPE_INT8 2

static inline int32_t SpmmDataTypeFromAcl(aclDataType valueType) {
    if (valueType == ACL_FLOAT)       return SPMM_DTYPE_FP32;
    if (valueType == ACL_FLOAT16)     return SPMM_DTYPE_FP16;
    if (valueType == ACL_INT8)        return SPMM_DTYPE_INT8;
    return SPMM_DTYPE_FP32;
}

// SpmmTilingData is laid out at the start of workspace, right after a 64-byte
// header. It is read once by the kernel via GM->scratch.
typedef struct SpmmTilingData {
    int32_t m;
    int32_t n;
    int32_t k;
    int32_t nnz;

    int32_t ldb;
    int32_t ldc;

    int32_t m_tile;
    int32_t n_tile;
    int32_t k_pack;

    int32_t row_bin_num;
    int32_t reorder_offset;     // byte offset to reorder table in workspace
    int32_t bin_edge_offset;    // byte offset to row-bin edge table in workspace

    int32_t meta_flag;          // bit0: preprocess_done
    int32_t blockDim;

    int32_t data_type;          // SPMM_DTYPE_FP32 / FP16 / INT8

    int32_t high_precision;     // 0=standard, 1=fp32 Kahan (CSR_FP32_HIGH_PRECISION_ALG)

    int32_t opB;                // 0 = NON_TRANSPOSE, 1 = TRANSPOSE
    int32_t order_pair;         // SPMM_ORDER_RR / RC / CR / CC

    float alpha_host;
    float beta_host;
} SpmmTilingData;

// Workspace layout (all aligned to 64B):
//   [ 0      , tilingOff       )  : 64B header (reserved / padding)
//   [ tilingOff   , reorderOff )  : SpmmTilingData
//   [ reorderOff  , binEdgeOff )  : int32 reorder[m]    (logical row -> original row)
//   [ binEdgeOff  , endOff     )  : int32 bin_edge[bin_num + 1]
#define SPMM_WS_HEADER_BYTES     64
#define SPMM_WS_ALIGN            64

static inline int64_t spmm_align_up(int64_t v, int64_t a) {
    if (a <= 0) {
        return v;
    }
    return ((v + a - 1) / a) * a;
}

static inline int64_t spmm_workspace_size(int64_t m, int64_t blockDim) {
    int64_t tilingOff   = SPMM_WS_HEADER_BYTES;
    int64_t reorderOff  = spmm_align_up(tilingOff + (int64_t)sizeof(SpmmTilingData), SPMM_WS_ALIGN);
    int64_t binEdgeOff  = spmm_align_up(reorderOff + (int64_t)sizeof(int32_t) * m, SPMM_WS_ALIGN);
    int64_t endOff      = spmm_align_up(binEdgeOff + (int64_t)sizeof(int32_t) * (blockDim + 1), SPMM_WS_ALIGN);
    return endOff;
}

#endif // SPMM_H
