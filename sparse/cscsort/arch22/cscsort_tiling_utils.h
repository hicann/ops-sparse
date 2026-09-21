/**
 * ----------------------------------------------------------------------------------------------------------
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ----------------------------------------------------------------------------------------------------------
 */

#ifndef CSCSORT_TILING_UTILS_H_
#define CSCSORT_TILING_UTILS_H_

#include <algorithm>
#include <cstdint>

namespace CscsortTiling {

// arch22（DAV-2201）的 AscendC Sort 仅支持 half/float 键且纯降序：热路径用
// float 合成键 + 硬件 Sort 全排（Sort/Extract repeatTime ≤ 255，每 repeat 32
// 元素 ⇒ 单次全排上限 8160）。合成键须为 float32 精确整数：行域位宽由 kernel
// 按段内实测最大行号自适应，超宽段 kernel 内回退标量归并。sortTmpSize 为 Sort
// 树 tmp 段（8B/元素，按 Align32(runSize) 计，dav_c220 实现同款口径）。

// A2 DataCopyPad（arch35 注释同款）blockLen 上限 2^21 - 1 字节。
inline constexpr uint32_t kDataCopyPadMaxBytes = (1U << 21U) - 1U;
inline constexpr uint32_t kDataCopyPadMaxElems = kDataCopyPadMaxBytes / sizeof(int32_t);
// run 段字节预算（按 padded=Align32(runSize) 长度）：
//   4B/e × 8 段：keyIn, pIn, keyOut, pOut（兜底归并输出复用）, keyF(f32),
//                iota, posTerm, dstIdx
//   8B/e dstKey（2N float 交织）+ 8B/e 排序树 tmp = 48B/e，取整按 52。
inline constexpr uint32_t kRunBytesPerElem = 52U;
inline constexpr uint64_t kReservedBytes = 8192U;
// Sort/Extract repeatTime ≤ 255，每 repeat 32 元素。
inline constexpr uint32_t kMaxFullSortElems = 255U * 32U;
inline constexpr uint32_t kSortTmpBytesPerElem = 8U;

inline uint32_t AlignTo32Elems(uint32_t count) { return (count + 31U) & ~31U; }

inline bool FindMaxRunSize(uint64_t ubSize, uint32_t &runSize, uint32_t &sortTmpSize)
{
    uint64_t usable = (ubSize > kReservedBytes) ? ubSize - kReservedBytes : 0U;
    uint32_t bestRun = static_cast<uint32_t>(
        std::min<uint64_t>(usable / kRunBytesPerElem,
                           std::min<uint64_t>(kDataCopyPadMaxElems, kMaxFullSortElems)));
    if (bestRun < 32U) {
        return false;
    }
    runSize = bestRun;
    sortTmpSize = AlignTo32Elems(bestRun) * kSortTmpBytesPerElem;
    return true;
}

}  // namespace CscsortTiling

#endif  // CSCSORT_TILING_UTILS_H_
