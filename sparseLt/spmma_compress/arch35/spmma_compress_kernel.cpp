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

#include <cstdint>

#include "kernel_operator.h"
#include "spmma_compress_kernel.h"

using namespace AscendC;

namespace {
constexpr uint32_t kVectorBytes = 256;
constexpr uint32_t kPositionBits = 2;
constexpr uint32_t kMetadataBits = 4;
constexpr uint16_t kFirstPairCode = 0x4; // 2:4 的位置 (0, 1)，也是 1:2 的位置 0 编码。
constexpr uint16_t kSecondFp32Code = 0xe; // 1:2 的位置 1 编码。

// U 为对应元素宽度的无符号整数，选值和搬运均保留原始位模式。
// 每个 lane 对应一个分组，保留值与位置码使用相同的 lane 顺序。
template <typename U>
struct GroupSelection {
    Reg::RegTensor<U> firstValue;
    Reg::RegTensor<U> secondValue;
    Reg::RegTensor<U> positionCode;
};

// 两组的四位位置码合成一个字节，再按 U 的宽度合并相邻字节。
template <typename U>
__simd_callee__ inline void StoreMetadata(Reg::RegTensor<U>& positionCodes, __ubuf__ uint8_t* metadata,
    uint32_t groupCount)
{
    Reg::RegTensor<U> even, odd, shifted, packed;
    auto mask = Reg::CreateMask<U, Reg::MaskPattern::ALL>();
    Reg::DeInterleave<U>(even, odd, positionCodes, positionCodes);
    Reg::ShiftLefts<U>(shifted, odd, static_cast<int16_t>(kMetadataBits), mask);
    Reg::Or<U>(packed, even, shifted, mask);
    if constexpr (sizeof(U) >= 2) {
        Reg::DeInterleave<U>(even, odd, packed, packed);
        Reg::ShiftLefts<U>(shifted, odd, static_cast<int16_t>(8), mask);
        Reg::Or<U>(packed, even, shifted, mask);
    }
    if constexpr (sizeof(U) == 4) {
        Reg::DeInterleave<U>(even, odd, packed, packed);
        Reg::ShiftLefts<U>(shifted, odd, static_cast<int16_t>(16), mask);
        Reg::Or<U>(packed, even, shifted, mask);
    }
    // 按 U 存储时可能多写最后一个字内的字节；它们仅落入 UB 预留区，CopyOut 不搬出。
    Reg::Store<U>(reinterpret_cast<__ubuf__ U*>(metadata), packed,
        (groupCount / 2 + sizeof(U) - 1) / sizeof(U));
}

// 未选的两个位置均为零时，候选位置对才能保留该组的全部非零值。
template <typename U, uint16_t FirstPosition, uint16_t SecondPosition>
__simd_callee__ inline void TrySelectPair(Reg::RegTensor<U>& excludedNonzero0,
    Reg::RegTensor<U>& excludedNonzero1, Reg::RegTensor<U>& firstValue,
    Reg::RegTensor<U>& secondValue, GroupSelection<U>& selected)
{
    static_assert(FirstPosition < SecondPosition && SecondPosition < 4, "invalid position pair");
    constexpr uint16_t positionCode = FirstPosition | (SecondPosition << kPositionBits);
    Reg::RegTensor<U> excludedNonzero, code;
    Reg::MaskReg validPair;
    auto mask = Reg::CreateMask<U, Reg::MaskPattern::ALL>();
    Reg::Or<U>(excludedNonzero, excludedNonzero0, excludedNonzero1, mask);
    Reg::Compares<U, CMPMODE::EQ>(validPair, excludedNonzero, static_cast<U>(0), mask);
    Reg::Duplicate<U>(code, static_cast<U>(positionCode), mask);
    Reg::Select<U>(selected.firstValue, firstValue, selected.firstValue, validPair);
    Reg::Select<U>(selected.secondValue, secondValue, selected.secondValue, validPair);
    Reg::Select<U>(selected.positionCode, code, selected.positionCode, validPair);
}

// FP32 的 1:2 分组：位置 1 非零时保留它，否则保留位置 0（含全零组）。
template <typename U>
__simd_callee__ inline void SelectOneAndStore(Reg::RegTensor<U>& value0, Reg::RegTensor<U>& value1,
    __ubuf__ U* values, __ubuf__ uint8_t* metadata, uint32_t groupCount)
{
    auto mask = Reg::CreateMask<U, Reg::MaskPattern::ALL>();
    Reg::RegTensor<U> magnitudeMask, nonzero1, selectedValue, firstCode, secondCode, positionCode;
    Reg::MaskReg takeSecond;
    // 符号位仅在判零时屏蔽；选值仍复制原始位模式。
    Reg::Duplicate<U>(magnitudeMask, static_cast<U>(0x7fffffffU), mask);
    Reg::And<U>(nonzero1, value1, magnitudeMask, mask);
    Reg::Compares<U, CMPMODE::NE>(takeSecond, nonzero1, static_cast<U>(0), mask);
    Reg::Select<U>(selectedValue, value1, value0, takeSecond);
    Reg::Duplicate<U>(firstCode, static_cast<U>(kFirstPairCode), mask);
    Reg::Duplicate<U>(secondCode, static_cast<U>(kSecondFp32Code), mask);
    Reg::Select<U>(positionCode, secondCode, firstCode, takeSecond);
    Reg::Store<U>(values, selectedValue, groupCount);
    StoreMetadata<U>(positionCode, metadata, groupCount);
}

// FP16/BF16/INT8 的 2:4 分组，两个保留值按原位置升序写出。
template <typename U>
__simd_callee__ inline void SelectTwoAndStore(Reg::RegTensor<U>& value0, Reg::RegTensor<U>& value1,
    Reg::RegTensor<U>& value2, Reg::RegTensor<U>& value3, __ubuf__ U* values,
    __ubuf__ uint8_t* metadata, uint32_t groupCount)
{
    constexpr uint32_t lanes = kVectorBytes / sizeof(U);
    constexpr U nonzeroBits = sizeof(U) == 2 ? static_cast<U>(0x7fffU) : static_cast<U>(0xffU);
    auto mask = Reg::CreateMask<U, Reg::MaskPattern::ALL>();
    Reg::RegTensor<U> magnitudeMask, nonzero0, nonzero1, nonzero2, nonzero3;
    GroupSelection<U> selected;
    // 浮点正负零都按零处理；INT8 使用完整字节判断。
    Reg::Duplicate<U>(magnitudeMask, nonzeroBits, mask);
    Reg::And<U>(nonzero0, value0, magnitudeMask, mask);
    Reg::And<U>(nonzero1, value1, magnitudeMask, mask);
    Reg::And<U>(nonzero2, value2, magnitudeMask, mask);
    Reg::And<U>(nonzero3, value3, magnitudeMask, mask);
    Reg::Duplicate<U>(selected.firstValue, static_cast<U>(0), mask);
    Reg::Duplicate<U>(selected.secondValue, static_cast<U>(0), mask);
    Reg::Duplicate<U>(selected.positionCode, static_cast<U>(kFirstPairCode), mask);

    // 较小的位置对后处理并覆盖前值，使不足额组优先补选靠前的零位置。
    TrySelectPair<U, 2, 3>(nonzero0, nonzero1, value2, value3, selected);
    TrySelectPair<U, 1, 3>(nonzero0, nonzero2, value1, value3, selected);
    TrySelectPair<U, 1, 2>(nonzero0, nonzero3, value1, value2, selected);
    TrySelectPair<U, 0, 3>(nonzero1, nonzero2, value0, value3, selected);
    TrySelectPair<U, 0, 2>(nonzero1, nonzero3, value0, value2, selected);
    TrySelectPair<U, 0, 1>(nonzero2, nonzero3, value0, value1, selected);

    Reg::RegTensor<U> low, high;
    Reg::Interleave<U>(low, high, selected.firstValue, selected.secondValue);
    const uint32_t valueCount = groupCount * 2;
    Reg::Store<U>(values, low, valueCount < lanes ? valueCount : lanes);
    if (valueCount > lanes) {
        Reg::Store<U>(values + lanes, high, valueCount - lanes);
    }
    StoreMetadata<U>(selected.positionCode, metadata, groupCount);
}

template <typename U>
__simd_vf__ inline void ClearInput(__ubuf__ U* input, uint32_t elementCount)
{
    constexpr uint32_t lanes = kVectorBytes / sizeof(U);
    auto mask = Reg::CreateMask<U, Reg::MaskPattern::ALL>();
    Reg::RegTensor<U> zero;
    Reg::Duplicate<U>(zero, static_cast<U>(0), mask);
    for (uint32_t offset = 0; offset < elementCount; offset += lanes) {
        Reg::StoreAlign<U>(input + offset, zero, mask);
    }
}

// 连续元素拆为组内位置向量：每个 lane 对应一组的同一位置。
template <typename U>
__simd_vf__ inline void CompressRow(__ubuf__ U* input, __ubuf__ U* values,
    __ubuf__ uint8_t* metadata, uint32_t validWidth)
{
    constexpr uint32_t lanes = kVectorBytes / sizeof(U);
    constexpr uint32_t groupSize = sizeof(U) == 4 ? 2 : 4;
    for (uint32_t offset = 0; offset < validWidth; offset += lanes) {
        const uint32_t validCount = validWidth - offset < lanes ? validWidth - offset : lanes;
        Reg::RegTensor<U> source, even, odd, value0, value1, value2, value3;
        Reg::LoadAlign<U>(source, input + offset);
        if constexpr (groupSize == 2) {
            Reg::DeInterleave<U>(value0, value1, source, source);
            SelectOneAndStore<U>(value0, value1, values + offset / 2,
                metadata + offset / (2 * groupSize), validCount / groupSize);
        } else {
            Reg::DeInterleave<U>(even, odd, source, source);
            Reg::DeInterleave<U>(value0, value2, even, even);
            Reg::DeInterleave<U>(value1, value3, odd, odd);
            SelectTwoAndStore<U>(value0, value1, value2, value3, values + offset / 2,
                metadata + offset / (2 * groupSize), validCount / groupSize);
        }
    }
}

// 相邻物理行分别提供组内位置；同一 lane 的两个或四个值构成一个分组。
template <typename U>
__simd_vf__ inline void CompressColumns(__ubuf__ U* input, __ubuf__ U* values,
    __ubuf__ uint8_t* metadata, uint32_t rowStrideElements, uint32_t validWidth)
{
    constexpr uint32_t lanes = kVectorBytes / sizeof(U);
    constexpr uint32_t groupSize = sizeof(U) == 4 ? 2 : 4;
    for (uint32_t offset = 0; offset < validWidth; offset += lanes) {
        const uint32_t validCount = validWidth - offset < lanes ? validWidth - offset : lanes;
        Reg::RegTensor<U> value0, value1, value2, value3;
        Reg::LoadAlign<U>(value0, input + offset);
        Reg::LoadAlign<U>(value1, input + rowStrideElements + offset);
        if constexpr (groupSize == 2) {
            SelectOneAndStore<U>(value0, value1, values + offset * (groupSize / 2),
                metadata + offset / 2, validCount);
        } else {
            Reg::LoadAlign<U>(value2, input + 2 * rowStrideElements + offset);
            Reg::LoadAlign<U>(value3, input + 3 * rowStrideElements + offset);
            SelectTwoAndStore<U>(value0, value1, value2, value3, values + offset * (groupSize / 2),
                metadata + offset / 2, validCount);
        }
    }
}

template <HardEvent Event>
__aicore__ inline void SynchronizePipeline()
{
    SetFlag<Event>(EVENT_ID0);
    WaitFlag<Event>(EVENT_ID0);
}

// 坐标与有效尺寸均按物理存储解释；末尾 tile 的有效尺寸可能小于分配尺寸。
struct TilePosition {
    uint64_t batch;
    uint64_t rowOffset;
    uint64_t colOffset;
    uint32_t validRows;
    uint32_t validCols;
};

template <typename U>
class SpmmaCompressKernel {
public:
    __aicore__ inline void Init(GM_ADDR dense, GM_ADDR compressed,
        const SpMmaCompressTilingData& tiling, TPipe& pipe);
    __aicore__ inline void Process();

private:
    __aicore__ inline TilePosition GetTilePosition(uint64_t tileId);
    __aicore__ inline void CopyIn(const TilePosition& tile);
    __aicore__ inline void Compute(const TilePosition& tile);
    __aicore__ inline void CopyOut(const TilePosition& tile);

    SpMmaCompressTilingData tilingData_;

    // GM 输出先存放所有 values，再存放所有 metadata。
    GlobalTensor<U> inputGm_;
    GlobalTensor<U> valuesGm_;
    GlobalTensor<uint8_t> metadataGm_;

    // UB 按行使用对齐后的 pitch，搬运时仅处理 tile 的有效区域。
    TBuf<TPosition::VECCALC> inputBuffer_;
    TBuf<TPosition::VECCALC> valuesBuffer_;
    TBuf<TPosition::VECCALC> metadataBuffer_;
    LocalTensor<U> inputLocal_;
    LocalTensor<U> valuesLocal_;
    LocalTensor<uint8_t> metadataLocal_;
};

template <typename U>
__aicore__ inline void SpmmaCompressKernel<U>::Init(GM_ADDR dense, GM_ADDR compressed,
    const SpMmaCompressTilingData& tiling, TPipe& pipe)
{
    tilingData_ = tiling;
    inputGm_.SetGlobalBuffer(reinterpret_cast<__gm__ U*>(dense), tilingData_.inputBytes / sizeof(U));
    valuesGm_.SetGlobalBuffer(reinterpret_cast<__gm__ U*>(compressed), tilingData_.valuesBytes / sizeof(U));
    metadataGm_.SetGlobalBuffer(compressed + tilingData_.valuesBytes, tilingData_.metadataBytes);
    pipe.InitBuffer(inputBuffer_, tilingData_.inputBufferBytes);
    pipe.InitBuffer(valuesBuffer_, tilingData_.valueBufferBytes);
    pipe.InitBuffer(metadataBuffer_, tilingData_.metadataBufferBytes);
    inputLocal_ = inputBuffer_.Get<U>();
    valuesLocal_ = valuesBuffer_.Get<U>();
    metadataLocal_ = metadataBuffer_.Get<uint8_t>();
}

template <typename U>
__aicore__ inline void SpmmaCompressKernel<U>::Process()
{
    // 各核以 blockDim 为步长领取 tile，避免不同核写入同一输出分组。
    for (uint64_t tileId = GetBlockIdx(); tileId < tilingData_.totalTiles; tileId += GetBlockNum()) {
        const auto tile = GetTilePosition(tileId);
        CopyIn(tile);
        Compute(tile);
        CopyOut(tile);
    }
}

template <typename U>
__aicore__ inline TilePosition SpmmaCompressKernel<U>::GetTilePosition(uint64_t tileId)
{
    const uint64_t tilesPerBatch = tilingData_.tilesPerRowBand * tilingData_.rowBandsPerBatch;
    const uint64_t tileInBatch = tileId % tilesPerBatch;
    TilePosition tile{};
    tile.batch = tileId / tilesPerBatch;
    tile.rowOffset = tileInBatch / tilingData_.tilesPerRowBand * tilingData_.tileRows;
    tile.colOffset = tileInBatch % tilingData_.tilesPerRowBand * tilingData_.tileCols;
    tile.validRows = static_cast<uint32_t>(tilingData_.rows - tile.rowOffset < tilingData_.tileRows ?
        tilingData_.rows - tile.rowOffset : tilingData_.tileRows);
    tile.validCols = static_cast<uint32_t>(tilingData_.cols - tile.colOffset < tilingData_.tileCols ?
        tilingData_.cols - tile.colOffset : tilingData_.tileCols);
    return tile;
}

template <typename U>
__aicore__ inline void SpmmaCompressKernel<U>::CopyIn(const TilePosition& tile)
{
    auto* input = reinterpret_cast<__ubuf__ U*>(inputLocal_.GetPhyAddr());
    asc_vf_call<ClearInput<U>>(input, tile.validRows * tilingData_.inputPitch / sizeof(U));
    // 清零完成后才能 DMA 覆写；该依赖也保护上一 tile 的向量读取。
    SynchronizePipeline<HardEvent::V_MTE2>();
    const DataCopyExtParams copyParams{1, static_cast<uint32_t>(tile.validCols * sizeof(U)), 0, 0, 0};
    const DataCopyPadExtParams<U> padParams{true, 0, 0, static_cast<U>(0)};
    for (uint32_t row = 0; row < tile.validRows; ++row) {
        const uint64_t inputOffset = tile.batch * tilingData_.batchStride +
            (tile.rowOffset + row) * tilingData_.ld + tile.colOffset;
        DataCopyPad(inputLocal_[row * tilingData_.inputPitch / sizeof(U)],
            inputGm_[inputOffset], copyParams, padParams);
    }
    // 搬入完成后再做完整向量加载；行尾填充保留前面的清零结果。
    SynchronizePipeline<HardEvent::MTE2_V>();
}

template <typename U>
__aicore__ inline void SpmmaCompressKernel<U>::Compute(const TilePosition& tile)
{
    auto* input = reinterpret_cast<__ubuf__ U*>(inputLocal_.GetPhyAddr());
    auto* values = reinterpret_cast<__ubuf__ U*>(valuesLocal_.GetPhyAddr());
    auto* metadata = reinterpret_cast<__ubuf__ uint8_t*>(metadataLocal_.GetPhyAddr());
    const uint32_t rowStep = tilingData_.alongRow ? 1 : tilingData_.groupSize;
    for (uint32_t row = 0; row < tile.validRows; row += rowStep) {
        const uint32_t fragment = row / rowStep;
        auto* groupInput = input + row * tilingData_.inputPitch / sizeof(U);
        auto* groupValues = values + fragment * tilingData_.valuePitch / sizeof(U);
        auto* groupMetadata = metadata + fragment * tilingData_.metadataPitch;
        if (tilingData_.alongRow) {
            asc_vf_call<CompressRow<U>>(groupInput, groupValues, groupMetadata, tile.validCols);
        } else {
            asc_vf_call<CompressColumns<U>>(groupInput, groupValues, groupMetadata,
                tilingData_.inputPitch / sizeof(U), tile.validCols);
        }
    }
    // 向量结果就绪后，MTE3 才能将 values 和 metadata 搬出。
    SynchronizePipeline<HardEvent::V_MTE3>();
}

template <typename U>
__aicore__ inline void SpmmaCompressKernel<U>::CopyOut(const TilePosition& tile)
{
    const uint32_t rowStep = tilingData_.alongRow ? 1 : tilingData_.groupSize;
    const uint32_t groupCount = tilingData_.alongRow ? tile.validCols / tilingData_.groupSize : tile.validCols;
    const uint32_t valuesPerGroup = tilingData_.groupSize / 2;
    const DataCopyExtParams valueCopy{1, static_cast<uint32_t>(groupCount * valuesPerGroup * sizeof(U)), 0, 0, 0};
    const DataCopyExtParams metadataCopy{1, groupCount / 2, 0, 0, 0};
    for (uint32_t row = 0; row < tile.validRows; row += rowStep) {
        const uint64_t batchGroupOffset = tile.batch * (tilingData_.rows * tilingData_.cols / tilingData_.groupSize);
        const uint64_t groupOffset = batchGroupOffset + (tilingData_.alongRow ?
            (tile.rowOffset + row) * (tilingData_.cols / tilingData_.groupSize) +
                tile.colOffset / tilingData_.groupSize :
            (tile.rowOffset + row) / tilingData_.groupSize * tilingData_.cols + tile.colOffset);
        const uint32_t fragment = row / rowStep;
        DataCopyPad(valuesGm_[groupOffset * valuesPerGroup],
            valuesLocal_[fragment * tilingData_.valuePitch / sizeof(U)], valueCopy);
        DataCopyPad(metadataGm_[groupOffset / 2], metadataLocal_[fragment * tilingData_.metadataPitch], metadataCopy);
    }
    // 输出搬完后，下一 tile 才能复用 values 和 metadata 的 UB 空间。
    SynchronizePipeline<HardEvent::MTE3_V>();
}
} // namespace

extern "C" __global__ __aicore__ void spmma_compress_u32(GM_ADDR dense, GM_ADDR compressed,
    SpMmaCompressTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe;
    SpmmaCompressKernel<uint32_t> kernel;
    kernel.Init(dense, compressed, tiling, pipe);
    kernel.Process();
}

extern "C" __global__ __aicore__ void spmma_compress_u16(GM_ADDR dense, GM_ADDR compressed,
    SpMmaCompressTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe;
    SpmmaCompressKernel<uint16_t> kernel;
    kernel.Init(dense, compressed, tiling, pipe);
    kernel.Process();
}

extern "C" __global__ __aicore__ void spmma_compress_u8(GM_ADDR dense, GM_ADDR compressed,
    SpMmaCompressTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe;
    SpmmaCompressKernel<uint8_t> kernel;
    kernel.Init(dense, compressed, tiling, pipe);
    kernel.Process();
}

extern "C" void spmma_compress_kernel_do(GM_ADDR dense, GM_ADDR compressed,
    uint32_t blockDim, const SpMmaCompressTilingData& tiling, void* stream)
{
    if (tiling.dataBytes == 4) {
        spmma_compress_u32<<<blockDim, nullptr, stream>>>(dense, compressed, tiling);
    } else if (tiling.dataBytes == 2) {
        spmma_compress_u16<<<blockDim, nullptr, stream>>>(dense, compressed, tiling);
    } else {
        spmma_compress_u8<<<blockDim, nullptr, stream>>>(dense, compressed, tiling);
    }
}
