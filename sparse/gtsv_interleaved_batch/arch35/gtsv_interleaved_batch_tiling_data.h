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
 * \file gtsv_interleaved_batch_tiling_data.h
 * \brief aclsparseSgtsvInterleavedBatch TilingData 结构体定义（Host / Kernel 共用）。
 *        SIMT 模型：tiling 仅需 m 和 batchCount。
 */

#ifndef GTSV_INTERLEAVED_BATCH_TILING_DATA_H_
#define GTSV_INTERLEAVED_BATCH_TILING_DATA_H_

#include <cstdint>

struct GtsvInterleavedBatchTilingData {
    int32_t m;
    int32_t batchCount;
};

#endif // GTSV_INTERLEAVED_BATCH_TILING_DATA_H_
