// ----------------------------------------------------------------------------------------------------------
// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software; you can redistribute it and/or modify it under the terms of conditions of
// CANN Open Software License Agreement Version 2 (the "License").
// You may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License is
// distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and limitations under the License.
// ----------------------------------------------------------------------------------------------------------
// DenseToSparse arch22 test reference layer: task-book value encoding
// (70/20/10 distribution, +/-0/INF/NAN coverage), dense/case generation and
// multi-format golden construction (CSR/CSC/COO/Blocked-ELL, ROW/COL,
// base 0/1).
#ifndef DENSE_TO_SPARSE_ARCH22_REF_H
#define DENSE_TO_SPARSE_ARCH22_REF_H

#include <acl/acl.h>
#include "cann_ops_sparse.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

// Secure memory helpers (securec): the golden layer writes fixed-width
// scalar encodings, so dst capacity always equals the copy count.
#include "securec.h"

namespace densetosparse {

inline void CheckedMemcpy(void *dst, const void *src, size_t count)
{
    // Fixed-width scalar encodings: the destination buffer always holds
    // exactly `count` bytes, so its capacity equals the copy size.
    const size_t dstCapacity = count;
    if (memcpy_s(dst, dstCapacity, src, count) != EOK) {
        throw std::runtime_error("memcpy_s failed in the ref layer");
    }
}

inline void CheckedMemset(void *dst, int value, size_t count)
{
    const size_t dstCapacity = count;
    if (memset_s(dst, dstCapacity, value, count) != EOK) {
        throw std::runtime_error("memset_s failed in the ref layer");
    }
}




const char *FmtName(aclsparseFormat_t f)
{
    switch (f) {
    case ACL_SPARSE_FORMAT_CSR: return "csr";
    case ACL_SPARSE_FORMAT_CSC: return "csc";
    case ACL_SPARSE_FORMAT_COO: return "coo";
    case ACL_SPARSE_FORMAT_BLOCKED_ELL: return "bell";
    default: return "?";
    }
}

const char *DtName(aclDataType t)
{
    switch (t) {
    case ACL_INT8: return "int8";
    case ACL_FLOAT16: return "fp16";
    case ACL_BF16: return "bf16";
    case ACL_FLOAT: return "fp32";
    case ACL_COMPLEX64: return "c64";
    default: return "?";
    }
}

int DTypeBytes(aclDataType t)
{
    if (t == ACL_INT8) return 1;
    if (t == ACL_FLOAT16 || t == ACL_BF16) return 2;
    if (t == ACL_FLOAT) return 4;
    return 8;
}

// ===================== 任务书 §3.5 值分布生成（70/20/10） =====================
// 普通values按 70% uniform[-1,1] / 20% N(μ=0,σ=1) / 10% 特殊值池生成：
//   特殊值池 = {+0, -0, +Inf, -Inf, qNaN, 最大有限值, 最小正规数, 最小次正规数}
// int8 以 {-1,0,1} 均匀 / 截断正态 / {0, INT8_MIN, INT8_MAX} 边界对应
// （整型无 Inf/NaN/±0 之分）；complex64 实部、虚部独立调用同一生成器。
// 判定不受影响：golden 按位读取 Dense64 实际字节，任何合法位模式均可。
uint32_t DrawF32Bits(std::mt19937 &g)
{
    static std::uniform_real_distribution<float> cat(0.0f, 1.0f);
    static std::uniform_real_distribution<float> uni(-1.0f, 1.0f);
    static std::normal_distribution<float> nrm(0.0f, 1.0f);
    static const uint32_t sp[] = {0x00000000u, 0x80000000u, 0x7F800000u,
                                  0xFF800000u, 0x7FC00000u, 0x7F7FFFFFu,
                                  0x00800000u, 0x00000001u};
    const float u = cat(g);
    uint32_t b = 0;
    if (u < 0.70f) {
        const float v = uni(g);
        CheckedMemcpy(&b, &v, 4);
    } else if (u < 0.90f) {
        const float v = nrm(g);
        CheckedMemcpy(&b, &v, 4);
    } else {
        size_t k = static_cast<size_t>(cat(g) * 8.0f);
        if (k > 7) k = 7;
        b = sp[k];
    }
    return b;
}

// f32 -> f16 简化转换（保留符号/±0/±Inf/NaN/次正规语义；golden 按位对齐，
// 舍入精度无影响）。
uint16_t F32ToF16Bits(uint32_t f)
{
    const uint32_t sign = (f >> 16) & 0x8000u;
    const int32_t e = static_cast<int32_t>((f >> 23) & 0xFFu);
    const uint32_t m = f & 0x7FFFFFu;
    if (e == 0xFF) {
        return static_cast<uint16_t>(sign | 0x7C00u | (m ? 0x200u : 0u));
    }
    const int32_t re = e - 127 + 15;
    if (re >= 0x1F) {
        return static_cast<uint16_t>(sign | 0x7C00u);
    }
    if (re <= 0) {
        if (re < -10) {
            return static_cast<uint16_t>(sign);
        }
        return static_cast<uint16_t>(sign | ((m | 0x800000u) >> (14 - re)));
    }
    return static_cast<uint16_t>(sign | (static_cast<uint32_t>(re) << 10) |
                                 (m >> 13));
}

void FillTaskValue(aclDataType dtype, std::mt19937 &g, void *out)
{
    if (dtype == ACL_INT8) {
        static std::uniform_real_distribution<float> cat(0.0f, 1.0f);
        static std::uniform_real_distribution<float> uni(-1.0f, 1.0f);
        static std::normal_distribution<float> nrm(0.0f, 1.0f);
        static const int sp[] = {0, 0, -128, 127};
        const float u = cat(g);
        int v;
        if (u < 0.70f) {
            v = static_cast<int>(std::lround(uni(g)));
        } else if (u < 0.90f) {
            v = static_cast<int>(std::lround(nrm(g)));
        } else {
            size_t k = static_cast<size_t>(cat(g) * 4.0f);
            if (k > 3) k = 3;
            v = sp[k];
        }
        *static_cast<int8_t *>(out) = static_cast<int8_t>(v);
        return;
    }
    if (dtype == ACL_FLOAT) {
        const uint32_t b = DrawF32Bits(g);
        CheckedMemcpy(out, &b, 4);
        return;
    }
    if (dtype == ACL_COMPLEX64) {
        const uint32_t re = DrawF32Bits(g);  // 实/虚部独立生成
        const uint32_t im = DrawF32Bits(g);
        CheckedMemcpy(out, &re, 4);
        CheckedMemcpy(static_cast<uint8_t *>(out) + 4, &im, 4);
        return;
    }
    const uint32_t f = DrawF32Bits(g);
    if (dtype == ACL_BF16) {
        const uint16_t h = static_cast<uint16_t>(f >> 16);  // 截断舍入
        CheckedMemcpy(out, &h, 2);
    } else {
        const uint16_t h = F32ToF16Bits(f);
        CheckedMemcpy(out, &h, 2);
    }
}

// Special-value bit patterns (1001=+Inf, 1002=-Inf, 1003=NaN, 1004=-0,
// odd=finite nonzero) encoded per dtype.
inline uint32_t SpecialBits16(uint32_t baseExp, uint64_t code)
{
    if (code == 1001) return (baseExp + 1) << 7;   // +Inf
    if (code == 1002) return 0x8000 | ((baseExp + 1) << 7); // -Inf
    if (code == 1003) return 0x7E00;               // NaN
    if (code == 1004) return 0x8000;               // -0
    if (code & 1) return 0x3C00 | static_cast<uint16_t>(code & 0xF);
    return 0;
}

void FillValue(aclDataType dtype, uint64_t code, void *out)
{
    CheckedMemset(out, 0, DTypeBytes(dtype));
    switch (dtype) {
    case ACL_INT8:
        *static_cast<int8_t *>(out) = static_cast<int8_t>(code & 0x7F);
        break;
    case ACL_FLOAT16: {
        const uint16_t bits = static_cast<uint16_t>(
            SpecialBits16(15, code));
        CheckedMemcpy(out, &bits, 2);
        break;
    }
    case ACL_BF16: {
        const uint16_t bits = static_cast<uint16_t>(
            SpecialBits16(15, code));
        CheckedMemcpy(out, &bits, 2);
        break;
    }
    case ACL_FLOAT: {
        uint32_t bits = 0;
        if (code == 1001) bits = 0x7F800000;
        else if (code == 1002) bits = 0xFF800000;
        else if (code == 1003) bits = 0x7FC00000;
        else if (code == 1004) bits = 0x80000000;
        else if (code & 1) {
            float v = 1.0f + (code & 0xF);
            CheckedMemcpy(&bits, &v, 4);
        }
        CheckedMemcpy(out, &bits, 4);
        break;
    }
    case ACL_COMPLEX64: {
        uint32_t re = 0, im = 0;
        if (code == 1001) { re = 0x7F800000; }
        else if (code == 1002) { im = 0x7FC00000; }
        else if (code == 1003) { im = 0x80000000; }
        else if (code & 1) { re = 0x3F800000; }
        else if (code != 0) { im = 0x40000000; }
        CheckedMemcpy(out, &re, 4);
        CheckedMemcpy(static_cast<uint8_t *>(out) + 4, &im, 4);
        break;
    }
    default: break;
    }
}

bool IsNonzeroBits(aclDataType dtype, const uint8_t *raw)
{
    switch (dtype) {
    case ACL_INT8: return raw[0] != 0;
    case ACL_FLOAT16:
    case ACL_BF16: {
        uint16_t bits;
        CheckedMemcpy(&bits, raw, 2);
        return (bits & 0x7FFF) != 0;
    }
    case ACL_FLOAT: {
        uint32_t bits;
        CheckedMemcpy(&bits, raw, 4);
        return (bits & 0x7FFFFFFF) != 0;
    }
    case ACL_COMPLEX64: {
        uint32_t re, im;
        CheckedMemcpy(&re, raw, 4);
        CheckedMemcpy(&im, raw + 4, 4);
        return ((re & 0x7FFFFFFF) | (im & 0x7FFFFFFF)) != 0;
    }
    default: return false;
    }
}

struct CaseCfg {
    aclsparseFormat_t format;
    aclDataType dtype;
    aclsparseIndexBase_t base;
    aclsparseOrder_t order;
    int rows;
    int cols;
    int ldPad;
    int bellB;
    int ellCols;
    unsigned seed;
    bool zeroFill = false;
    int nnzPerRow = -1;      // >=0: exact per-row nonzero count
    int denseHeadRows = -1;  // >=0: first N rows fully dense (mixed case)
};

struct Dense64 {
    std::vector<uint8_t> bytes;
    int64_t ld;
    int elemBytes;
    aclsparseOrder_t order;
    const uint8_t *At(int r, int c) const
    {
        const int64_t major = order == ACL_SPARSE_ORDER_ROW ? r : c;
        const int64_t minor = order == ACL_SPARSE_ORDER_ROW ? c : r;
        return bytes.data() + (major * ld + minor) * elemBytes;
    }
};

Dense64 MakeDense(const CaseCfg &cfg, std::mt19937 &rng)
{
    Dense64 d;
    d.elemBytes = DTypeBytes(cfg.dtype);
    d.order = cfg.order;
    d.ld = (cfg.order == ACL_SPARSE_ORDER_ROW ? cfg.cols : cfg.rows) +
           cfg.ldPad;
    if (d.ld < 1) {
        d.ld = 1;
    }
    const int64_t majorDim =
        cfg.order == ACL_SPARSE_ORDER_ROW ? cfg.rows : cfg.cols;
    d.bytes.resize(static_cast<size_t>(majorDim) * d.ld * d.elemBytes, 0);
    std::uniform_int_distribution<int> coin(0, 3);
    if (cfg.nnzPerRow >= 0) {
        // Exact per-row density with distinct positions.
        std::vector<int> perm(cfg.cols);
        for (int r = 0; r < cfg.rows; ++r) {
            for (int i = 0; i < cfg.cols; ++i) {
                perm[i] = i;
            }
            std::shuffle(perm.begin(), perm.end(), rng);
            const int n = std::min(cfg.nnzPerRow, cfg.cols);
            for (int i = 0; i < n; ++i) {
                FillValue(cfg.dtype, 1 + ((r * 31 + perm[i] * 17) & 0xF),
                          const_cast<uint8_t *>(d.At(r, perm[i])));
            }
        }
        return d;
    }
    for (int r = 0; r < cfg.rows; ++r) {
        for (int c = 0; c < cfg.cols; ++c) {
            const bool denseHead = cfg.denseHeadRows >= 0 &&
                                   r < cfg.denseHeadRows;
            const int roll = denseHead ? 1 : (cfg.zeroFill ? 0 : coin(rng));
            if (roll == 0) {
                continue;  // 结构零位置（不生成值）
            }
            // 任务书 §3.5：70% uniform[-1,1] / 20% N(0,1) / 10% 特殊值
            // （±0、±Inf、NaN、边界/最大有限、最小正规、最小次正规）。
            FillTaskValue(cfg.dtype, rng,
                          const_cast<uint8_t *>(d.At(r, c)));
        }
    }
    return d;
}

struct Golden {
    std::vector<int32_t> offsets;
    std::vector<int32_t> rows, cols;
    std::vector<uint8_t> values;
    int64_t nnz = 0;
    int majorDim = 0;
};

// One pass over the dense matrix in golden scan order: values, COO row/
// col pairs and the per-major counts (CSR/CSC).
void CollectGoldenNonzeros(const CaseCfg &cfg, const Dense64 &d, Golden &g,
    std::vector<int> &cnt)
{
    const bool isCsc = cfg.format == ACL_SPARSE_FORMAT_CSC;
    const bool isCoo = cfg.format == ACL_SPARSE_FORMAT_COO;
    for (int outer = 0; outer < (isCsc ? cfg.cols : cfg.rows); ++outer) {
        for (int inner = 0; inner < (isCsc ? cfg.rows : cfg.cols); ++inner) {
            const int r = isCsc ? inner : outer;
            const int c = isCsc ? outer : inner;
            if (!IsNonzeroBits(cfg.dtype, d.At(r, c))) continue;
            g.values.insert(g.values.end(), d.At(r, c),
                            d.At(r, c) + d.elemBytes);
            if (isCoo) {
                g.rows.push_back(r);
                g.cols.push_back(c);
            } else if (isCsc) {
                g.cols.push_back(r);
                ++cnt[c + 1];
            } else {
                g.rows.push_back(c);
                ++cnt[r + 1];
            }
        }
    }
}

Golden MakeGolden(const CaseCfg &cfg, const Dense64 &d)
{
    Golden g;
    const bool isCsc = cfg.format == ACL_SPARSE_FORMAT_CSC;
    const bool isCoo = cfg.format == ACL_SPARSE_FORMAT_COO;
    g.majorDim = isCsc ? cfg.cols : cfg.rows;
    std::vector<int> cnt;
    if (!isCoo) {
        cnt.assign(g.majorDim + 1, 0);
        g.offsets.assign(g.majorDim + 1, 0);
    }
    CollectGoldenNonzeros(cfg, d, g, cnt);
    g.nnz = isCoo ? g.rows.size() : g.values.size() / d.elemBytes;
    if (!isCoo) {
        for (int i = 1; i <= g.majorDim; ++i) cnt[i] += cnt[i - 1];
        for (int i = 0; i <= g.majorDim; ++i)
            g.offsets[i] = cnt[i] + static_cast<int32_t>(cfg.base);
    }
    if (isCoo) {
        for (auto &v : g.rows) v += static_cast<int32_t>(cfg.base);
        for (auto &v : g.cols) v += static_cast<int32_t>(cfg.base);
    } else {
        for (auto &v : (isCsc ? g.cols : g.rows))
            v += static_cast<int32_t>(cfg.base);
    }
    return g;
}

} // namespace densetosparse

using densetosparse::CaseCfg;
using densetosparse::CheckedMemcpy;
using densetosparse::CheckedMemset;
using densetosparse::DtName;
using densetosparse::FmtName;
using densetosparse::DTypeBytes;
using densetosparse::Dense64;
using densetosparse::DrawF32Bits;
using densetosparse::FillTaskValue;
using densetosparse::FillValue;
using densetosparse::Golden;
using densetosparse::IsNonzeroBits;
using densetosparse::MakeDense;
using densetosparse::MakeGolden;

#endif  // DENSE_TO_SPARSE_ARCH22_REF_H
