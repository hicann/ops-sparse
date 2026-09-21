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
#ifndef TEST_SPMMA_COMPRESS_PARAM_H
#define TEST_SPMMA_COMPRESS_PARAM_H
#include "csv_loader.h"
#include <charconv>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <stdexcept>

namespace sparse_test {
inline std::invalid_argument CompressConfigError(const csv_map& row, const std::string& field)
{
    const auto name = row.find("case_name");
    return std::invalid_argument(
        "compression case " + (name == row.end() ? std::string("<unnamed>") : name->second) + ": invalid " + field);
}

inline const std::string& CompressRequired(const csv_map& row, const std::string& field)
{
    const auto item = row.find(field);
    if (item == row.end() || item->second.empty()) {
        throw CompressConfigError(row, field);
    }
    return item->second;
}

inline std::string CompressChoice(
    const csv_map& row, const std::string& field, std::initializer_list<const char*> choices)
{
    const auto& value = CompressRequired(row, field);
    if (std::find(choices.begin(), choices.end(), value) == choices.end()) {
        throw CompressConfigError(row, field);
    }
    return value;
}

inline int64_t CompressInteger(const csv_map& row, const std::string& field)
{
    const auto& value = CompressRequired(row, field);
    int64_t result = 0;
    const auto parsed = std::from_chars(value.data(), value.data() + value.size(), result);
    if (parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size()) {
        throw CompressConfigError(row, field);
    }
    return result;
}

inline int CompressNonnegativeInt(const csv_map& row, const std::string& field)
{
    const int64_t value = CompressInteger(row, field);
    if (value < 0 || value > std::numeric_limits<int>::max()) {
        throw CompressConfigError(row, field);
    }
    return static_cast<int>(value);
}

inline bool CompressBoolean(const csv_map& row, const std::string& field)
{
    const auto value = CompressChoice(row, field, {"0", "1", "false", "true", "no", "yes"});
    return value == "1" || value == "true" || value == "yes";
}

struct CompressParam : SparseTestParamBase {
    std::string name;
    std::string dtype = "FP32";
    std::string order = "ROW";
    std::string op = "N";
    std::string side = "A";
    std::string pattern = "mixed";
    int64_t rows = 32;
    int64_t cols = 64;
    int64_t ld = 64;
    int64_t stride = 0;
    int batches = 1;
    int level = 0;
    bool defaultStream = false;
    bool workspaceSentinel = false;
    void fillCustom(const csv_map& r) override
    {
        name = CompressRequired(r, "case_name");
        dtype = CompressChoice(r, "dtype", {"FP32", "FP16", "BF16", "INT8"});
        order = CompressChoice(r, "order", {"ROW", "COL"});
        op = CompressChoice(r, "op", {"N", "T"});
        side = CompressChoice(r, "side", {"A", "B"});
        pattern = CompressChoice(r, "pattern", {"mixed", "special", "zero", "negative_zero", "overfull"});
        rows = CompressInteger(r, "rows");
        cols = CompressInteger(r, "cols");
        ld = CompressInteger(r, "ld");
        stride = CompressInteger(r, "stride");
        batches = CompressNonnegativeInt(r, "batches");
        level = CompressNonnegativeInt(r, "level");
        defaultStream = CompressBoolean(r, "default_stream");
        workspaceSentinel = CompressBoolean(r, "workspace_sentinel");
        ValidateStorage(r);
    }
    std::string caseId() const override
    {
        return name;
    }
    size_t bytes() const
    {
        if (dtype == "FP32")
            return 4;
        if (dtype == "FP16" || dtype == "BF16")
            return 2;
        if (dtype == "INT8")
            return 1;
        throw std::invalid_argument("unknown dtype");
    }
    int group() const
    {
        return dtype == "FP32" ? 2 : 4;
    }
    int64_t physicalRows() const
    {
        return order == "ROW" ? rows : cols;
    }
    int64_t physicalCols() const
    {
        return order == "ROW" ? cols : rows;
    }
    bool alongRow() const
    {
        return (op == "N") == (order == "ROW");
    }
    size_t spanBytes() const
    {
        return ((batches - 1) * stride + (physicalRows() - 1) * ld + physicalCols()) * bytes();
    }
    size_t elements() const
    {
        return static_cast<size_t>(rows) * cols * batches;
    }
    size_t valuesBytes() const
    {
        return elements() / 2 * bytes();
    }
    size_t compressedBytes() const
    {
        return valuesBytes() + elements() / group() / 2;
    }

private:
    void ValidateShape(const csv_map& r) const
    {
        const int64_t alignment = dtype == "FP32" ? 8 : dtype == "INT8" ? 32 : 16;
        if (rows <= 0 || cols <= 0 || batches <= 0 || stride < 0 || rows > std::numeric_limits<int32_t>::max() ||
            cols > std::numeric_limits<int32_t>::max()) {
            throw CompressConfigError(r, "shape or batch");
        }
        if (ld < physicalCols() || rows % alignment != 0 || cols % alignment != 0 || ld % alignment != 0) {
            throw CompressConfigError(r, "shape alignment or ld");
        }
    }

    void ValidateStorage(const csv_map& r) const
    {
        ValidateShape(r);
        const size_t dataBytes = bytes();
        if (dataBytes == 0) {
            throw CompressConfigError(r, "dtype width");
        }
        const uint64_t maxElements = static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) / dataBytes;
        const uint64_t rowCount = static_cast<uint64_t>(physicalRows());
        const uint64_t width = static_cast<uint64_t>(physicalCols());
        const uint64_t rowGaps = rowCount - 1;
        if (rowGaps == 0 || static_cast<uint64_t>(ld) > (maxElements - width) / rowGaps) {
            throw CompressConfigError(r, "input span overflow");
        }
        const uint64_t span = rowGaps * static_cast<uint64_t>(ld) + width;
        if (stride != 0 && static_cast<uint64_t>(stride) < span) {
            throw CompressConfigError(r, "stride");
        }
        const uint64_t extraBatches = static_cast<uint64_t>(batches - 1);
        if (extraBatches != 0 && static_cast<uint64_t>(stride) > (maxElements - span) / extraBatches) {
            throw CompressConfigError(r, "batch span overflow");
        }
        const uint64_t matrixElements = static_cast<uint64_t>(rows) * static_cast<uint64_t>(cols);
        if (batches <= 0 || matrixElements > maxElements / static_cast<uint64_t>(batches)) {
            throw CompressConfigError(r, "output size overflow");
        }
    }
};
inline void PrintTo(const CompressParam& p, std::ostream* os)
{
    *os << p.name;
}
} // namespace sparse_test
#endif
