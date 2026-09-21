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
#ifndef TEST_SPMMA_COMPRESS_GOLDEN_H
#define TEST_SPMMA_COMPRESS_GOLDEN_H
#include "spmma_compress_param.h"
#include <algorithm>
#include <array>
#include <set>
#include <vector>

namespace sparse_test {
// 使用描述符坐标计算源地址，不依赖 kernel 的分块布局。
inline size_t SourceByte(const CompressParam& p, int b, int64_t row, int64_t col)
{
    const int64_t element = p.order == "ROW" ? row * p.ld + col : col * p.ld + row;
    return (b * p.stride + element) * p.bytes();
}
inline std::array<size_t, 4> GroupSources(const CompressParam& p, size_t index)
{
    const size_t groupsPerBatch = static_cast<size_t>(p.rows) * p.cols / p.group();
    const int batch = static_cast<int>(index / groupsPerBatch);
    const size_t local = index % groupsPerBatch;
    std::array<size_t, 4> src{};
    for (int pos = 0; pos < p.group(); ++pos) {
        int64_t major;
        int64_t minor;
        if (p.alongRow()) {
            major = local / (p.physicalCols() / p.group());
            minor = local % (p.physicalCols() / p.group()) * p.group() + pos;
        } else {
            major = local / p.physicalCols() * p.group() + pos;
            minor = local % p.physicalCols();
        }
        src[pos] = p.order == "ROW" ? SourceByte(p, batch, major, minor) : SourceByte(p, batch, minor, major);
    }
    return src;
}
inline bool RawNonzero(const uint8_t* value, size_t bytes, bool integer)
{
    for (size_t i = 0; i < bytes; ++i) {
        const uint8_t byte = (!integer && i + 1 == bytes) ? value[i] % 128 : value[i];
        if (byte != 0)
            return true;
    }
    return false;
}
inline uint8_t PositionCode(const std::set<int>& selected, int group)
{
    if (group == 2)
        return *selected.begin() == 0 ? 4 : 14;
    static const std::array<std::array<int, 3>, 6> table{
        {{{0, 1, 4}}, {{0, 2, 8}}, {{1, 2, 9}}, {{0, 3, 12}}, {{1, 3, 13}}, {{2, 3, 14}}}};
    const int first = *selected.begin();
    const int second = *std::next(selected.begin());
    for (const auto& entry : table)
        if (entry[0] == first && entry[1] == second)
            return entry[2];
    throw std::invalid_argument("invalid selected positions");
}
inline std::vector<uint8_t> CompressGolden(const CompressParam& p, const std::vector<uint8_t>& dense)
{
    if (dense.size() != p.spanBytes())
        throw std::invalid_argument("dense span mismatch");
    std::vector<uint8_t> result(p.compressedBytes(), 0);
    const size_t totalGroups = p.elements() / p.group();
    size_t dst = 0;
    for (size_t g = 0; g < totalGroups; ++g) {
        const auto src = GroupSources(p, g);
        std::set<int> selected;
        for (int t = 0; t < p.group(); ++t)
            if (RawNonzero(&dense.at(src[t]), p.bytes(), p.dtype == "INT8"))
                selected.insert(t);
        if (selected.size() > static_cast<size_t>(p.group() / 2))
            throw std::invalid_argument("input violates structured sparsity");
        for (int t = 0; selected.size() < static_cast<size_t>(p.group() / 2); ++t)
            selected.insert(t);
        for (int t : selected) {
            std::copy_n(dense.begin() + src[t], p.bytes(), result.begin() + dst);
            dst += p.bytes();
        }
        const uint8_t code = PositionCode(selected, p.group());
        result[p.valuesBytes() + g / 2] += code * (g % 2 == 0 ? 1 : 16);
    }
    return result;
}
inline std::vector<uint32_t> SpecialBits(const std::string& dtype)
{
    if (dtype == "FP32")
        return {0x3f800001, 0xbf801fff, 0x7f800000, 0xff800000, 0x7fc12345, 0xffc54321, 0x7f812345, 0x00000001,
            0x80000001, 0x007fffff, 0x00800000};
    if (dtype == "FP16")
        return {0x3c01, 0xbc03, 0x7c00, 0xfc00, 0x7e55, 0xfe33, 0x7d01, 0x0001, 0x8001, 0x03ff, 0x0400};
    if (dtype == "BF16")
        return {0x3f81, 0xbf83, 0x7f80, 0xff80, 0x7fc5, 0xffe3, 0x7f81, 0x0001, 0x8001, 0x007f, 0x0080};
    return {0x80, 0xff, 0x01, 0x7f};
}
inline void PutBits(std::vector<uint8_t>& data, size_t offset, size_t bytes, uint32_t value)
{
    for (size_t i = 0; i < bytes; ++i)
        data.at(offset + i) = static_cast<uint8_t>(value >> (8 * i));
}
inline std::vector<uint8_t> MakeCompressInput(const CompressParam& p)
{
    std::vector<uint8_t> dense(p.spanBytes(), 0xa7); // 行尾和批次间隙保留填充值，用于检查越界修改。
    const auto bits = SpecialBits(p.dtype);
    const size_t groups = static_cast<size_t>(p.rows) * p.cols / p.group();
    const int inputBatches = p.stride == 0 ? 1 : p.batches;
    static const int pairs[6][2] = {{0, 1}, {0, 2}, {0, 3}, {1, 2}, {1, 3}, {2, 3}};
    for (int b = 0; b < inputBatches; ++b)
        for (size_t g = 0; g < groups; ++g) {
            const auto src = GroupSources(p, b * groups + g);
            for (int pos = 0; pos < p.group(); ++pos) {
                const uint32_t zero = p.dtype != "INT8" && (pos % 2 || p.pattern == "negative_zero")
                                          ? (uint32_t{1} << (p.bytes() * 8 - 1))
                                          : 0;
                PutBits(dense, src[pos], p.bytes(), zero);
            }
            if (p.pattern == "zero" || p.pattern == "negative_zero")
                continue;
            const size_t cycle = g % (p.group() == 2 ? 3 : 11);
            std::vector<int> occupied;
            if (p.pattern == "overfull")
                for (int pos = 0; pos < p.group(); ++pos)
                    occupied.push_back(pos);
            else if (p.group() == 2) {
                if (cycle < 2)
                    occupied.push_back(cycle);
            } else if (cycle < 6)
                occupied = {pairs[cycle][0], pairs[cycle][1]};
            else if (cycle < 10)
                occupied = {static_cast<int>(cycle - 6)};
            for (int pos : occupied)
                PutBits(dense, src[pos], p.bytes(), bits[(g + pos + b * 3) % bits.size()]);
        }
    return dense;
}
} // namespace sparse_test
#endif
