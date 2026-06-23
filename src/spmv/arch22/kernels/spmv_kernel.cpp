/**
 * ----------------------------------------------------------------------------------------------------------
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under
 * the terms and conditions of CANN Open Software License Agreement Version 2.0
 * (the "License"). Please refer to the License for details. You may not use
 * this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
 * AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
 * FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
 * for the full text of the License.
 * ----------------------------------------------------------------------------------------------------------
 */

// 统一 host 分发器 —— 根据 SPMV_* 类型 ID 派发到各自的 __global__ kernel
// 各 kernel 定义在 spmv_kernel_*type*.cpp 中

#include "spmv_kernel.h"
#include "../spmv.h"

extern "C" void spmv_kernel_do(GM_ADDR csrRowPtr, GM_ADDR csrColInd,
							   GM_ADDR csrVal, GM_ADDR xVec, GM_ADDR yVec,
							   SpmvTilingData tiling, const void *alpha,
							   const void *beta, int32_t computeType,
							   int32_t valType, int32_t outType, bool trans,
							   uint32_t numBlocks, void *stream) {
#define SPMV_LAUNCH(CompT, ValT, OutT)                                         \
	spmv_kernel_##CompT##_##ValT##_##OutT<<<numBlocks, nullptr, stream>>>(     \
		csrRowPtr, csrColInd, csrVal, xVec, yVec,                              \
		tiling.totalRowsNum, tiling.totalColNum, a, b, trans)

	if (computeType == SPMV_COMPUTE_I32) {
		int32_t a = *static_cast<const int32_t *>(alpha);
		int32_t b = *static_cast<const int32_t *>(beta);
		SPMV_LAUNCH(int32_t, int32_t, int32_t);
	}
	else {
		float a = *static_cast<const float *>(alpha);
		float b = *static_cast<const float *>(beta);
		if (valType == SPMV_VAL_HALF && outType == SPMV_OUT_HALF)
			SPMV_LAUNCH(float, half, half);
		else if (valType == SPMV_VAL_HALF)
			SPMV_LAUNCH(float, half, float);
		else if (valType == SPMV_VAL_BF16 && outType == SPMV_OUT_BF16)
			SPMV_LAUNCH(float, bfloat16_t, bfloat16_t);
		else if (valType == SPMV_VAL_BF16)
			SPMV_LAUNCH(float, bfloat16_t, float);
		else if (valType == SPMV_VAL_I32)
			SPMV_LAUNCH(float, int32_t, float);
		else
			SPMV_LAUNCH(float, float, float);
	}

#undef SPMV_LAUNCH
}
