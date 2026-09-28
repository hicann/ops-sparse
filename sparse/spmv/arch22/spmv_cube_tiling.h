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

#ifndef SPMV_CUBE_TILING_H
#define SPMV_CUBE_TILING_H

#include <cstddef>
#include <cstdint>

#include "kernels/tiling_spmv_cube.h"

// -------------------------------------------------------------------
// cube 版 SpMV 的 host 侧 tiling 计算。
//
// 同时被 aclsparseSpMVGetBufferSize（查询 workspace 大小）与 aclsparseSpMV
// （准备 tiling 并启动 kernel）使用，保证两者对 workspace 布局的认知一致。
// 纯 host 代码，不依赖 AscendC。
// -------------------------------------------------------------------
namespace spmv_cube {

/// cube 分块边长。fp32 的 fractal 为 M/N=16、K=8，fp16 为 M/N=16、K=16，
/// 故须为 16 的倍数；constants.h 只提供 32/64/128 的下三角常量矩阵。
constexpr uint32_t kCubeTileLen = 128;

constexpr inline uint32_t CeilDivU32(uint32_t value, uint32_t divisor) {
	return divisor == 0 ? 0 : (value + divisor - 1) / divisor;
}

constexpr inline uint32_t AlignUpU32(uint32_t value, uint32_t alignment) {
	return CeilDivU32(value, alignment) * alignment;
}

/**
 * @brief cube 版 kernel 的 workspace 布局（见 run_spmv_cube）：
 *   [0, productsBytes)             -> csr gather 的乘积 (T = half / float)
 *   [productsBytes, totalBytes)    -> 块内推测扫描结果
 *   (OutputT = CubeOutType_t<T>，half / float 输入均为 float)
 */
struct CubeWorkspaceLayout {
	/// nnz 向矩阵块 (tileLen^2) 对齐后的长度
	uint32_t paddedNnz;
	/// gather 乘积的元素字节数（half 输入为 2，fp32 输入为 4）
	size_t valElemSize;
	/// gather 乘积区字节数
	size_t productsBytes;
	/// workspace 总字节数
	size_t totalBytes;
};

/// paddedNnz 以 uint32_t 传给 kernel，nnz 过大时对齐会回绕。
constexpr inline bool IsCubeNnzSupported(uint32_t nnz) {
	return nnz <= 0xFFFFFFFFu - (kCubeTileLen * kCubeTileLen - 1);
}

/**
 * @brief 计算 cube 版 kernel 所需的 workspace 布局。
 *
 * @return nnz 过大（对齐后回绕 uint32_t）时返回 false。
 */
inline bool ComputeCubeWorkspaceLayout(uint32_t nnz, bool isHalfInput,
									   CubeWorkspaceLayout *layout) {
	if (!IsCubeNnzSupported(nnz)) {
		return false;
	}

	const uint32_t alignSize = kCubeTileLen * kCubeTileLen;
	layout->paddedNnz = AlignUpU32(nnz, alignSize);
	layout->valElemSize = isHalfInput ? sizeof(uint16_t) : sizeof(float);
	layout->productsBytes = static_cast<size_t>(layout->paddedNnz) * layout->valElemSize;
	layout->totalBytes =
		layout->productsBytes + static_cast<size_t>(layout->paddedNnz) * sizeof(float);
	return true;
}

/**
 * @brief 计算 cube 版 kernel 的 tiling 与 launch 的 block 数。
 *
 * host 侧只负责切分：按 nnz 均分每个 AI Core 组负责的非零元区间，无需回读 rowPtr。
 * 各块的起始段（行）偏移由 kernel 侧对 indptr 做二分查找（fused searchsorted）
 * 自行推导。
 *
 * @param numAivBlocks IN, AIV 核数（cube kernel 为 MIX(1 AIC : 2 AIV) 形态，
 *                     launch 的 block 数按 AIC 计）。
 * @return 参数不受支持（空矩阵 / nnz 过大）时返回 false，调用方应回退到向量版 kernel。
 */
inline bool MakeCubeTiling(uint32_t rows, uint32_t cols, uint32_t nnz,
						   float alpha, float beta, uint32_t numAivBlocks,
						   tcuscan::SpMVCubeTiling *tiling, uint32_t *aicBlocks) {
	if (rows == 0 || cols == 0 || nnz == 0) {
		// 空矩阵退化为 y = beta * y，交给向量版 kernel 处理。
		return false;
	}

	if (!IsCubeNnzSupported(nnz)) {
		return false;
	}

	const uint32_t blocks = numAivBlocks / 2 > 0 ? numAivBlocks / 2 : 1;

	// 每块负责的非零元数：必须是矩阵块 (tileLen^2) 的整数倍——段求和按矩阵块
	// 回滚块内推测扫描的结果。
	//
	// 该值还须不小于最长行，否则会出现某块内不含任何段结尾。这里无需回读 rowPtr
	// 求最长行：aclsparseSpMV 在派发前已用 UB 容量上界（ComputeMaxRowLength，
	// 约 1e4 个元素）校验过最长行，而 nnz >= 1 时 blockLen >= tileLen^2 = 16384
	// 已大于该上界，故约束自动成立。若日后调小 kCubeTileLen（使 tileLen^2 降到
	// UB 上界以下），此处须重新引入最长行下界。
	const uint32_t alignSize = kCubeTileLen * kCubeTileLen;
	const uint32_t blockLen = AlignUpU32(CeilDivU32(nnz, blocks), alignSize);

	*tiling = tcuscan::SpMVCubeTiling{nnz, rows, cols, kCubeTileLen, blockLen, alpha, beta};
	*aicBlocks = blocks;
	return true;
}

} // namespace spmv_cube

#endif // SPMV_CUBE_TILING_H
