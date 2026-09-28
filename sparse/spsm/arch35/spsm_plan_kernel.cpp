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

#include "kernel_operator.h"
#include "spsm_plan_kernel.h"
#include "simt_api/asc_simt.h"
#include "simt_api/device_functions.h"

namespace
{
constexpr uint32_t THREADS = 256;
struct Key
{
    int32_t row, col, slot;
};
struct Number
{
    float real, imag;
};

struct Wide
{
    float hi, lo;
};

struct WideNumber
{
    Wide real, imag;
};

// Error-free transforms keep the rounding discarded by each FP32 operation.
// Do not compile these with reassociation/fast-math enabled.
__simt_callee__ inline Wide AddWide(Wide a, Wide b)
{
    float s = a.hi + b.hi;
    if (!isfinite(s))
    {
        return { s, 0.0f };
    }
    float v = s - a.hi;
    float e = ((a.hi - (s - v)) + (b.hi - v)) + (a.lo + b.lo);
    float h = s + e;
    return { h, e - (h - s) };
}

__simt_callee__ inline Wide ProductWide(float a, Wide b)
{
    float p = a * b.hi;
    if (!isfinite(p))
    {
        return { p, 0.0f };
    }
    float e = fmaf(a, b.hi, -p) + a * b.lo;
    float h = p + e;
    return { h, e - (h - p) };
}

// Accumulate product and addition residuals without renormalizing every term.
// Normalize once at the row boundary before division or publishing a result.
__simt_callee__ inline Wide AccumulateProduct(Wide sum, float a, Wide b)
{
    float p = a * b.hi;
    float s = sum.hi + p;
    if (!isfinite(s))
    {
        return { s, 0 };
    }
    float v = s - sum.hi;
    float error = (sum.hi - (s - v)) + (p - v);
    float productError = fmaf(a, b.hi, -p) + a * b.lo;
    return { s, (sum.lo + productError) + error };
}

__simt_callee__ inline Key ReadKey(__gm__ Key* keys, int64_t k)
{
    return { keys[k].row, keys[k].col, keys[k].slot };
}

__simt_callee__ inline void WriteKey(__gm__ Key* keys, int64_t k, Key value)
{
    keys[k].row = value.row;
    keys[k].col = value.col;
    keys[k].slot = value.slot;
}

__simt_callee__ inline bool Less(Key a, Key b)
{
    return a.row < b.row || (a.row == b.row && (a.col < b.col || (a.col == b.col && a.slot < b.slot)));
}

__simt_callee__ inline Number Read(__gm__ float* p, int64_t i, int32_t components)
{
    return { p[i * components], components == 2 ? p[i * components + 1] : 0.0f };
}

__simt_callee__ inline void Write(__gm__ float* p, int64_t i, int32_t components, Number v)
{
    p[i * components] = v.real;
    if (components == 2)
    {
        p[i * components + 1] = v.imag;
    }
}

__simt_callee__ inline Number Mul(Number a, Number b)
{
    return { a.real * b.real - a.imag * b.imag, a.real * b.imag + a.imag * b.real };
}

// Rare finite-range path: halve the numerator before summation. Halve a
// large denominator too, so neither Smith sum overflows before the division.
__simt_callee__ inline Number DivideLargeFinite(Number a, Number b, bool largeDenominator)
{
    a.real *= 0.5f;
    a.imag *= 0.5f;
    if (largeDenominator)
    {
        b.real *= 0.5f;
        b.imag *= 0.5f;
    }
    Number result { };
    if (fabsf(b.real) >= fabsf(b.imag))
    {
        float ratio = b.imag / b.real;
        float denominator = b.real + b.imag * ratio;
        result = { (a.real + a.imag * ratio) / denominator, (a.imag - a.real * ratio) / denominator };
    }
    else
    {
        float ratio = b.real / b.imag;
        float denominator = b.imag + b.real * ratio;
        result = { (a.real * ratio + a.imag) / denominator, (a.imag * ratio - a.real) / denominator };
    }
    if (!largeDenominator)
    {
        result.real *= 2.0f;
        result.imag *= 2.0f;
    }
    return result;
}

// Scaled complex division avoids squaring a large/small diagonal.
__simt_callee__ inline Number Divide(Number a, Number b, int32_t components)
{
    if (components == 1)
    {
        return { a.real / b.real, 0.0f };
    }
    // Avoid inf * 0 in the scaled formula for axis-aligned diagonals.
    // This also preserves each component of a non-finite RHS for UNIT.
    if (b.imag == 0.0f)
    {
        return { a.real / b.real, a.imag / b.real };
    }
    if (b.real == 0.0f)
    {
        return { a.imag / b.imag, -a.real / b.imag };
    }
    float ar = b.real < 0 ? -b.real : b.real;
    float ai = b.imag < 0 ? -b.imag : b.imag;
    constexpr float HALF_MAX_FLOAT = 0x1.fffffep+126f;
    bool largeDenominator = ar > HALF_MAX_FLOAT || ai > HALF_MAX_FLOAT;
    if ((largeDenominator || fabsf(a.real) > HALF_MAX_FLOAT || fabsf(a.imag) > HALF_MAX_FLOAT) && isfinite(a.real)
        && isfinite(a.imag) && isfinite(b.real) && isfinite(b.imag))
    {
        return DivideLargeFinite(a, b, largeDenominator);
    }
    if (ar >= ai)
    {
        float ratio = b.imag / b.real;
        float denominator = b.real + b.imag * ratio;
        return { (a.real + a.imag * ratio) / denominator, (a.imag - a.real * ratio) / denominator };
    }
    float ratio = b.real / b.imag;
    float denominator = b.imag + b.real * ratio;
    return { (a.real * ratio + a.imag) / denominator, (a.imag * ratio - a.real) / denominator };
}

__simt_callee__ inline WideNumber MultiplyWide(Number a, WideNumber b, int32_t components)
{
    Wide re = ProductWide(a.real, b.real);
    if (components == 1)
    {
        return { re, { 0, 0 } };
    }
    re = AddWide(re, ProductWide(-a.imag, b.imag));
    Wide im = AddWide(ProductWide(a.real, b.imag), ProductWide(a.imag, b.real));
    return { re, im };
}

__simt_callee__ inline WideNumber SubtractWideProduct(
    WideNumber sum, Number a, Number b, Number tail, int32_t components)
{
    sum.real = AccumulateProduct(sum.real, -a.real, { b.real, tail.real });
    if (components == 2)
    {
        sum.real = AccumulateProduct(sum.real, a.imag, { b.imag, tail.imag });
        sum.imag = AccumulateProduct(sum.imag, -a.real, { b.imag, tail.imag });
        sum.imag = AccumulateProduct(sum.imag, -a.imag, { b.real, tail.real });
    }
    return sum;
}

__simt_callee__ inline Wide DivideWideReal(Wide a, float b)
{
    a = AddWide(a, { 0, 0 });
    if (b == 1.0f)
    {
        return a;
    }
    float q = a.hi / b;
    if (!isfinite(q) || !isfinite(b))
    {
        return { q, 0 };
    }
    float residual = fmaf(-q, b, a.hi) + a.lo;
    return AddWide({ q, 0 }, { residual / b, 0 });
}

// Refine the stable FP32 quotient using the compensated residual. Axis-aligned
// and non-finite inputs retain the existing scaled division semantics.
__simt_callee__ inline WideNumber DivideWide(WideNumber a, Number b, int32_t components)
{
    if (components == 1 || b.imag == 0.0f)
    {
        return { DivideWideReal(a.real, b.real), components == 2 ? DivideWideReal(a.imag, b.real) : Wide { 0, 0 } };
    }
    if (b.real == 0.0f)
    {
        return { DivideWideReal(a.imag, b.imag), DivideWideReal({ -a.real.hi, -a.real.lo }, b.imag) };
    }
    a = { AddWide(a.real, { 0, 0 }), AddWide(a.imag, { 0, 0 }) };
    Number q = Divide({ a.real.hi, a.imag.hi }, b, components);
    WideNumber result { { q.real, 0 }, { q.imag, 0 } };
    if (!isfinite(q.real) || !isfinite(q.imag) || !isfinite(b.real) || !isfinite(b.imag))
    {
        return result;
    }
    WideNumber product = MultiplyWide({ -b.real, -b.imag }, result, components);
    Wide re = AddWide(a.real, product.real), im = AddWide(a.imag, product.imag);
    if (!isfinite(re.hi) || !isfinite(im.hi))
    {
        return result;
    }
    Number correction = Divide({ re.hi + re.lo, im.hi + im.lo }, b, components);
    return { AddWide(result.real, { correction.real, 0 }), AddWide(result.imag, { correction.imag, 0 }) };
}

__simt_vf__ __aicore__ __launch_bounds__(1) inline void CheckPointers(
    __gm__ int32_t* ptr, __gm__ int32_t* summary, int32_t m, int32_t nnz, int32_t format, int32_t base)
{
    for (int32_t i = 0; i < 16; ++i)
    {
        summary[i] = 0;
    }
    if (format != 2)
    {
        if (ptr[0] != base || static_cast<int64_t>(ptr[m]) != static_cast<int64_t>(nnz) + base)
        {
            summary[0] = 1;
        }
        for (int32_t row = 0; row < m; ++row)
        {
            int64_t start = static_cast<int64_t>(ptr[row]) - base;
            int64_t end = static_cast<int64_t>(ptr[row + 1]) - base;
            if (start < 0 || end < start || end > nnz)
            {
                summary[0] = 1;
            }
        }
    }
}

__simt_callee__ inline int32_t FindRow(__gm__ int32_t* ptr, int32_t m, int32_t slot, int32_t base)
{
    int32_t lo = 0, hi = m;
    while (lo < hi)
    {
        int32_t mid = lo + (hi - lo) / 2;
        if (static_cast<int64_t>(ptr[mid + 1]) - base <= slot)
        {
            lo = mid + 1;
        }
        else
        {
            hi = mid;
        }
    }
    return lo;
}

__simt_vf__ __aicore__ __launch_bounds__(THREADS) inline void Decode(__gm__ int32_t* ptr, __gm__ int32_t* idx,
    __gm__ uint8_t* ws, int64_t p_key0, int32_t p_nnz, int32_t p_format, int32_t p_base, int32_t p_m, int32_t p_opA,
    uint32_t block, uint32_t blocks)
{
    auto keys = reinterpret_cast<__gm__ Key*>(ws + p_key0);
    auto status = reinterpret_cast<__gm__ int32_t*>(ws);
    if (status[0] == 0)
    {
        for (int64_t k = block * THREADS + AscendC::Simt::GetThreadIdx(); k < p_nnz; k += blocks * THREADS)
        {
            int64_t a = p_format == 2 ? static_cast<int64_t>(ptr[k]) - p_base : FindRow(ptr, p_m, k, p_base);
            int64_t b = static_cast<int64_t>(idx[k]) - p_base;
            bool valid = a >= 0 && a < p_m && b >= 0 && b < p_m;
            bool swap = (p_format == 1) != (p_opA != 0);
            WriteKey(keys, k,
                { valid ? static_cast<int32_t>(swap ? b : a) : -1, valid ? static_cast<int32_t>(swap ? a : b) : -1,
                    static_cast<int32_t>(k) });
        }
    }
}

// Each entry obtains a unique destination using its rank in the opposite run.
// The original slot breaks ties, so duplicate coordinates have a fixed order.
__simt_vf__ __aicore__ __launch_bounds__(THREADS) inline void Merge(
    __gm__ uint8_t* ws, int64_t p_key0, int64_t p_key1, int32_t p_nnz, int64_t width, uint32_t block, uint32_t blocks)
{
    auto src = reinterpret_cast<__gm__ Key*>(ws + p_key0);
    auto dst = reinterpret_cast<__gm__ Key*>(ws + p_key1);
    if (reinterpret_cast<__gm__ int32_t*>(ws)[0] == 0)
    {
        for (int64_t k = block * THREADS + AscendC::Simt::GetThreadIdx(); k < p_nnz; k += blocks * THREADS)
        {
            int64_t start = k / (2 * width) * (2 * width);
            int64_t middle = start + width < p_nnz ? start + width : p_nnz;
            int64_t end = start + 2 * width < p_nnz ? start + 2 * width : p_nnz;
            bool left = k < middle;
            int64_t other = left ? middle : start;
            int64_t lo = other, hi = left ? end : middle;
            Key key = ReadKey(src, k);
            while (lo < hi)
            {
                int64_t mid = lo + (hi - lo) / 2;
                if (Less(ReadKey(src, mid), key))
                {
                    lo = mid + 1;
                }
                else
                {
                    hi = mid;
                }
            }
            WriteKey(dst, start + (k - (left ? start : middle)) + lo - other, key);
        }
    }
}

__simt_vf__ __aicore__ __launch_bounds__(THREADS) inline void Canonical(__gm__ float* values, __gm__ uint8_t* ws,
    int64_t p_key0, int64_t p_row, int64_t p_col, int64_t p_perm, int64_t p_val, int32_t p_nnz, int32_t p_components,
    int32_t p_opA, int32_t p_m, uint32_t block, uint32_t blocks)
{
    auto summary = reinterpret_cast<__gm__ int32_t*>(ws);
    auto keys = reinterpret_cast<__gm__ Key*>(ws + p_key0);
    auto row = reinterpret_cast<__gm__ int32_t*>(ws + p_row);
    auto col = reinterpret_cast<__gm__ int32_t*>(ws + p_col);
    auto perm = reinterpret_cast<__gm__ int32_t*>(ws + p_perm);
    auto val = reinterpret_cast<__gm__ float*>(ws + p_val);
    // CHECK failure is immutable until the next CHECK. No thread sets status
    // here; ANALYZE checks the first sorted key after this kernel completes.
    if (summary[0] == 0)
    {
        for (int64_t k = block * THREADS + AscendC::Simt::GetThreadIdx(); k < p_nnz; k += blocks * THREADS)
        {
            Key key = ReadKey(keys, k);
            col[k] = key.col;
            perm[k] = key.slot;
            Number v = Read(values, key.slot, p_components);
            if (p_opA == 2)
            {
                v.imag = -v.imag;
            }
            Write(val, k, p_components, v);
        }
        for (int64_t r = block * THREADS + AscendC::Simt::GetThreadIdx(); r <= p_m; r += blocks * THREADS)
        {
            int32_t lo = 0, hi = p_nnz;
            while (lo < hi)
            {
                int32_t mid = lo + (hi - lo) / 2;
                if (keys[mid].row < r)
                {
                    lo = mid + 1;
                }
                else
                {
                    hi = mid;
                }
            }
            row[r] = lo;
        }
    }
}

__simt_callee__ inline bool Dependency(int32_t row, int32_t col, int32_t upper)
{
    return upper ? col > row : col < row;
}

__simt_callee__ inline Number Diagonal(__gm__ int32_t* row, __gm__ int32_t* col, __gm__ int32_t* perm,
    __gm__ float* val, int32_t r, int32_t components, int32_t unit, int32_t opA, bool original)
{
    Wide real { 0.0f, 0.0f };
    Wide imag { 0.0f, 0.0f };
    if (unit)
    {
        return { 1.0f, 0.0f };
    }
    // Canonical columns are sorted, including duplicates. A diagonal lookup
    // must not scan all off-diagonal entries on every numeric update.
    int32_t lo = row[r], hi = row[r + 1];
    while (lo < hi)
    {
        int32_t mid = lo + (hi - lo) / 2;
        if (col[mid] < r)
        {
            lo = mid + 1;
        }
        else
        {
            hi = mid;
        }
    }
    // Retain small duplicate values across cancellation before testing for zero.
    for (int32_t k = lo; k < row[r + 1] && col[k] == r; ++k)
    {
        Number v = Read(val, original ? perm[k] : k, components);
        real = AddWide(real, { v.real, 0.0f });
        imag = AddWide(imag, { v.imag, 0.0f });
    }
    return { real.hi, original && opA == 2 ? -imag.hi : imag.hi };
}

__simt_callee__ inline void MakeLevelBuckets(__gm__ int32_t* levelPtr, __gm__ int32_t* levelIdx, __gm__ int32_t* level,
    __gm__ int32_t* counts, int32_t p_m, int32_t levels)
{
    levelPtr[0] = 0;
    for (int32_t l = 0; l < levels; ++l)
    {
        levelPtr[l + 1] = levelPtr[l] + counts[l];
        counts[l] = levelPtr[l];
    }
    for (int32_t r = 0; r < p_m; ++r)
    {
        levelIdx[counts[level[r]]++] = r;
    }
}

struct AnalysisStats
{
    int32_t levels;
    int32_t maxRowLength;
    int32_t maxDependencyDistance;
};

__simt_callee__ inline AnalysisStats AnalyzeRows(__gm__ int32_t* row, __gm__ int32_t* col, __gm__ float* val,
    __gm__ float* diag, __gm__ int32_t* level, __gm__ int32_t* counts, __gm__ int32_t* summary, int32_t p_m,
    int32_t p_upper, int32_t p_components, int32_t p_unit, int32_t p_opA)
{
    AnalysisStats stats { 0, 0, 0 };
    for (int32_t step = 0; step < p_m; ++step)
    {
        int32_t r = p_upper ? p_m - 1 - step : step;
        int32_t rowLevel = 0;
        for (int32_t k = row[r]; k < row[r + 1]; ++k)
        {
            int32_t c = col[k];
            if (Dependency(r, c, p_upper))
            {
                rowLevel = level[c] + 1 > rowLevel ? level[c] + 1 : rowLevel;
                int32_t distance = p_upper ? c - r : r - c;
                stats.maxDependencyDistance
                    = distance > stats.maxDependencyDistance ? distance : stats.maxDependencyDistance;
            }
        }
        level[r] = rowLevel;
        counts[rowLevel] += 1;
        stats.levels = rowLevel + 1 > stats.levels ? rowLevel + 1 : stats.levels;
        int32_t rowLength = row[r + 1] - row[r];
        stats.maxRowLength = rowLength > stats.maxRowLength ? rowLength : stats.maxRowLength;
        Number diagonal = Diagonal(row, col, nullptr, val, r, p_components, p_unit, p_opA, false);
        if (diagonal.real == 0.0f && diagonal.imag == 0.0f)
        {
            summary[0] = 2;
        }
        Write(diag, r, p_components, diagonal);
    }
    return stats;
}

// Triangular dependencies have a known order. This linear device pass has no
// inter-core spin locks and requires no host arrays or matrix-sized transfers.
__simt_vf__ __aicore__ __launch_bounds__(1) inline void Analyze(__gm__ uint8_t* ws, int64_t p_key0, int32_t p_nnz,
    int64_t p_row, int64_t p_col, int64_t p_perm, int64_t p_val, int64_t p_diag, int64_t p_rowLevel, int64_t p_scratch,
    int64_t p_levelPtr, int64_t p_levelIdx, int32_t p_m, int32_t p_upper, int32_t p_components, int32_t p_unit,
    int32_t p_opA)
{
    auto summary = reinterpret_cast<__gm__ int32_t*>(ws);
    auto keys = reinterpret_cast<__gm__ Key*>(ws + p_key0);
    if (p_nnz > 0 && summary[0] == 0 && keys[0].row < 0)
    {
        summary[0] = 1;
    }
    if (summary[0] != 0)
    {
        return;
    }
    auto row = reinterpret_cast<__gm__ int32_t*>(ws + p_row);
    auto col = reinterpret_cast<__gm__ int32_t*>(ws + p_col);
    auto level = reinterpret_cast<__gm__ int32_t*>(ws + p_rowLevel);
    auto counts = reinterpret_cast<__gm__ int32_t*>(ws + p_scratch);
    for (int32_t r = 0; r <= p_m; ++r)
    {
        counts[r] = 0;
    }
    AnalysisStats stats = AnalyzeRows(row, col, reinterpret_cast<__gm__ float*>(ws + p_val),
        reinterpret_cast<__gm__ float*>(ws + p_diag), level, counts, summary, p_m, p_upper, p_components, p_unit,
        p_opA);
    MakeLevelBuckets(reinterpret_cast<__gm__ int32_t*>(ws + p_levelPtr),
        reinterpret_cast<__gm__ int32_t*>(ws + p_levelIdx), level, counts, p_m, stats.levels);
    summary[1] = stats.levels;
    summary[2] = stats.maxRowLength;
    summary[3] = stats.maxDependencyDistance;
}

__simt_callee__ inline void DenseScatter(__gm__ float* dense, __gm__ float* scratch, int32_t p_components,
    int32_t p_m, int32_t p_n, int32_t p_orderC, int64_t p_ldc, uint32_t block, uint32_t blocks)
{
    for (int64_t k = block * THREADS + AscendC::Simt::GetThreadIdx(); k < static_cast<int64_t>(p_m) * p_n;
        k += blocks * THREADS)
    {
        int64_t r = k / p_n, c = k % p_n;
        int64_t dst = p_orderC ? c * p_ldc + r : r * p_ldc + c;
        Write(dense, dst, p_components, Read(scratch, k, p_components));
    }
}

__simt_callee__ inline void DenseSnapshot(__gm__ float* dense, __gm__ uint8_t* ws, __gm__ float* scratch,
    Number alpha, int32_t p_components, int32_t p_m, int32_t p_n, int32_t p_opB, int32_t p_orderB, int64_t p_ldb,
    int64_t p_low, int64_t p_rowLevel, int64_t p_diag, bool roots, uint32_t block, uint32_t blocks, bool raw)
{
    for (int64_t k = block * THREADS + AscendC::Simt::GetThreadIdx(); k < static_cast<int64_t>(p_m) * p_n;
        k += blocks * THREADS)
    {
        int64_t r = k / p_n, c = k % p_n;
        int64_t br = p_opB ? c : r, bc = p_opB ? r : c;
        int64_t src = p_orderB ? bc * p_ldb + br : br * p_ldb + bc;
        Number value = Read(dense, src, p_components);
        if (p_opB == 2)
        {
            value.imag = -value.imag;
        }
        if (raw)
        {
            Write(scratch, k, p_components, value);
        }
        else if (p_low)
        {
            WideNumber scaled = MultiplyWide(alpha, { { value.real, 0 }, { value.imag, 0 } }, p_components);
            if (roots && reinterpret_cast<__gm__ int32_t*>(ws + p_rowLevel)[r] == 0)
            {
                scaled = DivideWide(
                    scaled, Read(reinterpret_cast<__gm__ float*>(ws + p_diag), r, p_components), p_components);
            }
            Write(scratch, k, p_components, { scaled.real.hi, scaled.imag.hi });
            Write(reinterpret_cast<__gm__ float*>(ws + p_low), k, p_components, { scaled.real.lo, scaled.imag.lo });
        }
        else
        {
            Number scaled = Mul(alpha, value);
            if (roots && reinterpret_cast<__gm__ int32_t*>(ws + p_rowLevel)[r] == 0)
            {
                scaled = Divide(
                    scaled, Read(reinterpret_cast<__gm__ float*>(ws + p_diag), r, p_components), p_components);
            }
            Write(scratch, k, p_components, scaled);
        }
    }
}

__simt_vf__ __aicore__ __launch_bounds__(THREADS) inline void DenseCopy(__gm__ float* dense, __gm__ float* alpha,
    __gm__ uint8_t* ws, int64_t p_scratch, float p_alphaReal, float p_alphaImag, int32_t p_deviceAlpha,
    int32_t p_components, int32_t p_m, int32_t p_n, int32_t p_orderC, int64_t p_ldc, int32_t p_opB, int32_t p_orderB,
    int64_t p_ldb, int64_t p_low, int64_t p_rowLevel, int64_t p_diag, bool roots, bool output, uint32_t block,
    uint32_t blocks, bool raw)
{
    auto scratch = reinterpret_cast<__gm__ float*>(ws + p_scratch);
    if (output)
    {
        DenseScatter(dense, scratch, p_components, p_m, p_n, p_orderC, p_ldc, block, blocks);
        return;
    }
    Number scale = p_deviceAlpha ? Read(alpha, 0, p_components) : Number { p_alphaReal, p_alphaImag };
    DenseSnapshot(dense, ws, scratch, scale, p_components, p_m, p_n, p_opB, p_orderB, p_ldb, p_low, p_rowLevel,
        p_diag, roots, block, blocks, raw);
}

// Other columns retain the raw B snapshot until their turn, including when
// B and C overlap. Only this panel's rounding-tail storage is initialized.
__simt_vf__ __aicore__ __launch_bounds__(THREADS) inline void PreparePanel(__gm__ uint8_t* ws, __gm__ float* alpha,
    int64_t p_scratch, int64_t p_low, int64_t p_rowLevel, int64_t p_diag, int32_t p_m, int32_t p_n, int32_t p_lowWidth,
    int32_t begin, int32_t columns, int32_t p_components, int32_t p_deviceAlpha, float p_alphaReal, float p_alphaImag,
    bool roots, uint32_t block, uint32_t blocks)
{
    auto x = reinterpret_cast<__gm__ float*>(ws + p_scratch);
    auto low = reinterpret_cast<__gm__ float*>(ws + p_low);
    Number a = p_deviceAlpha ? Read(alpha, 0, p_components) : Number { p_alphaReal, p_alphaImag };
    for (int64_t k = block * THREADS + AscendC::Simt::GetThreadIdx(); k < static_cast<int64_t>(p_m) * columns;
        k += blocks * THREADS)
    {
        int64_t r = k / columns, c = k % columns, dst = r * p_n + begin + c;
        Number b = Read(x, dst, p_components);
        WideNumber scaled = MultiplyWide(a, { { b.real, 0 }, { b.imag, 0 } }, p_components);
        if (roots && reinterpret_cast<__gm__ int32_t*>(ws + p_rowLevel)[r] == 0)
        {
            scaled
                = DivideWide(scaled, Read(reinterpret_cast<__gm__ float*>(ws + p_diag), r, p_components), p_components);
        }
        Write(x, dst, p_components, { scaled.real.hi, scaled.imag.hi });
        Write(low, r * p_lowWidth + c, p_components, { scaled.real.lo, scaled.imag.lo });
    }
}

// Each lane owns a complete RHS, including all its dependencies. The path
// handles arbitrarily long rows/deep graphs without UB-sized rows or barriers.
template <int p_components, bool UNIT>
__simt_callee__ inline void SolveWideRow(__gm__ int32_t* row, __gm__ int32_t* col, __gm__ float* val,
    __gm__ float* diag, __gm__ float* x, __gm__ float* low, int32_t r, int64_t rhs, int64_t local, int32_t p_n,
    int32_t p_upper, int32_t p_lowWidth, int32_t p_lowRows, int32_t lowRow, bool window, Number scale)
{
    Number accum = Read(x, static_cast<int64_t>(r) * p_n + rhs, p_components);
    WideNumber initial;
    if (window)
    {
        initial = MultiplyWide(scale, { { accum.real, 0 }, { accum.imag, 0 } }, p_components);
    }
    else
    {
        Number tail = Read(low, static_cast<int64_t>(r) * p_lowWidth + local, p_components);
        initial = { { accum.real, tail.real }, { accum.imag, tail.imag } };
    }
    WideNumber sum = initial;
    for (int32_t k = row[r]; k < row[r + 1]; ++k)
    {
        int32_t c = col[k];
        if (Dependency(r, c, p_upper))
        {
            Number a = Read(val, k, p_components);
            Number b = Read(x, static_cast<int64_t>(c) * p_n + rhs, p_components);
            int32_t lowCol = window ? c & (p_lowRows - 1) : c;
            Number bl = Read(low, static_cast<int64_t>(lowCol) * p_lowWidth + local, p_components);
            sum = SubtractWideProduct(sum, a, b, bl, p_components);
        }
    }
    Number diagonal = UNIT ? Number { 1, 0 } : Read(diag, r, p_components);
    WideNumber result = DivideWide(sum, diagonal, p_components);
    Write(x, static_cast<int64_t>(r) * p_n + rhs, p_components, { result.real.hi, result.imag.hi });
    Write(low, static_cast<int64_t>(lowRow) * p_lowWidth + local, p_components,
        { result.real.lo, result.imag.lo });
}

template <int p_components>
__simt_callee__ inline void SolveBasicRow(__gm__ int32_t* row, __gm__ int32_t* col, __gm__ float* val,
    __gm__ float* diag, __gm__ float* x, int32_t r, int64_t rhs, int32_t p_n, int32_t p_upper)
{
    Number accum = Read(x, static_cast<int64_t>(r) * p_n + rhs, p_components);
    for (int32_t k = row[r]; k < row[r + 1]; ++k)
    {
        int32_t c = col[k];
        if (Dependency(r, c, p_upper))
        {
            Number product
                = Mul(Read(val, k, p_components), Read(x, static_cast<int64_t>(c) * p_n + rhs, p_components));
            accum.real -= product.real;
            accum.imag -= product.imag;
        }
    }
    Write(x, static_cast<int64_t>(r) * p_n + rhs, p_components,
        Divide(accum, Read(diag, r, p_components), p_components));
}

template <int p_components, bool UNIT>
__simt_vf__ __aicore__ __launch_bounds__(THREADS) inline void Solve(__gm__ uint8_t* ws, int64_t p_row, int64_t p_col,
    int64_t p_val, int64_t p_diag, int64_t p_scratch, int32_t p_n, int32_t p_m, int32_t p_upper, int64_t p_low,
    int64_t p_levelPtr, int64_t p_levelIdx, uint32_t block, uint32_t blocks, int32_t p_lowWidth, int32_t begin,
    int32_t columns, int32_t p_lowRows, __gm__ float* alpha, int32_t p_deviceAlpha, float p_alphaReal,
    float p_alphaImag)
{
    auto row = reinterpret_cast<__gm__ int32_t*>(ws + p_row);
    auto col = reinterpret_cast<__gm__ int32_t*>(ws + p_col);
    auto val = reinterpret_cast<__gm__ float*>(ws + p_val);
    auto diag = reinterpret_cast<__gm__ float*>(ws + p_diag);
    auto x = reinterpret_cast<__gm__ float*>(ws + p_scratch);
    auto levelPtr = reinterpret_cast<__gm__ int32_t*>(ws + p_levelPtr);
    auto levelIdx = reinterpret_cast<__gm__ int32_t*>(ws + p_levelIdx);
    bool window = p_lowRows < p_m;
    Number scale = p_deviceAlpha ? Read(alpha, 0, p_components) : Number { p_alphaReal, p_alphaImag };
    for (int64_t local = block + AscendC::Simt::GetThreadIdx() * blocks; local < columns; local += blocks * THREADS)
    {
        int64_t rhs = begin + local;
        // SNAPSHOT_ROOTS already solved all independent rows in parallel.
        // Remaining level buckets provide a valid topological order even when
        // row numbers are far apart, so the serial path need not visit roots.
        for (int32_t step = window ? 0 : levelPtr[1]; step < p_m; ++step)
        {
            int32_t r = window ? (p_upper ? p_m - 1 - step : step) : levelIdx[step];
            int32_t lowRow = window ? r & (p_lowRows - 1) : r;
            if (p_low)
            {
                auto low = reinterpret_cast<__gm__ float*>(ws + p_low);
                SolveWideRow<p_components, UNIT>(row, col, val, diag, x, low, r, rhs, local, p_n, p_upper,
                    p_lowWidth, p_lowRows, lowRow, window, scale);
            }
            else
            {
                SolveBasicRow<p_components>(row, col, val, diag, x, r, rhs, p_n, p_upper);
            }
        }
    }
}

template <int COMPONENTS, bool UNIT>
__aicore__ inline void RunSerial(__gm__ uint8_t* workspace, __gm__ float* alpha, const SpsmPlanTiling& t, int32_t begin,
    int32_t columns, uint32_t block, uint32_t blocks)
{
    asc_vf_call<Solve<COMPONENTS, UNIT>>(dim3 { THREADS, 1, 1 }, workspace, t.row, t.col, t.val, t.diag, t.scratch, t.n,
        t.m, t.upper, t.low, t.levelPtr, t.levelIdx, block, blocks, t.lowWidth, begin, columns, t.lowRows, alpha,
        t.deviceAlpha, t.alphaReal, t.alphaImag);
}

// One subgroup cooperates on a sparse dot product. Rows in the same level are
// independent; the enclosing kernel establishes a device-wide level barrier.
template <int GROUP>
__simt_callee__ inline WideNumber AccumulateLevelRow(__gm__ int32_t* row, __gm__ int32_t* col,
    __gm__ float* val, __gm__ float* x, __gm__ float* low, int32_t r, int32_t rhs, int32_t local, int32_t p_n,
    int32_t p_upper, int32_t p_components, int32_t p_lowWidth, uint32_t lane, bool compensated)
{
    WideNumber sum { { 0, 0 }, { 0, 0 } };
    for (int64_t k = static_cast<int64_t>(row[r]) + lane; k < row[r + 1]; k += GROUP)
    {
        int32_t c = col[k];
        if (Dependency(r, c, p_upper))
        {
            Number a = Read(val, k, p_components);
            Number b = Read(x, static_cast<int64_t>(c) * p_n + rhs, p_components);
            if (compensated)
            {
                Number bl = Read(low, static_cast<int64_t>(c) * p_lowWidth + local, p_components);
                sum = SubtractWideProduct(sum, a, b, bl, p_components);
            }
            else
            {
                Number product = Mul(a, b);
                sum.real.hi -= product.real;
                sum.imag.hi -= product.imag;
            }
        }
    }
    return sum;
}

template <int GROUP>
__simt_callee__ inline WideNumber ReduceLevelRow(
    __gm__ uint8_t* ws, int64_t p_reduce, uint32_t block, uint32_t lane, WideNumber sum, bool compensated)
{
    constexpr int WARP = GROUP < 32 ? GROUP : 32;
    for (int offset = WARP / 2; offset > 0; offset /= 2)
    {
        WideNumber other { { asc_shfl_down(sum.real.hi, offset, WARP), asc_shfl_down(sum.real.lo, offset, WARP) },
            { asc_shfl_down(sum.imag.hi, offset, WARP), asc_shfl_down(sum.imag.lo, offset, WARP) } };
        if (compensated)
        {
            sum.real = AddWide(sum.real, other.real);
            sum.imag = AddWide(sum.imag, other.imag);
        }
        else
        {
            sum.real.hi += other.real.hi;
            sum.imag.hi += other.imag.hi;
        }
    }
    if constexpr (GROUP == THREADS)
    {
        auto partial = reinterpret_cast<__gm__ float*>(ws + p_reduce) + block * 32;
        if (lane % 32 == 0)
        {
            int slot = (lane / 32) * 4;
            partial[slot] = sum.real.hi;
            partial[slot + 1] = sum.real.lo;
            partial[slot + 2] = sum.imag.hi;
            partial[slot + 3] = sum.imag.lo;
        }
        asc_threadfence_block();
        asc_syncthreads();
        if (lane == 0)
        {
            for (int warp = 1; warp < THREADS / 32; ++warp)
            {
                if (compensated)
                {
                    sum.real = AddWide(sum.real, { partial[warp * 4], partial[warp * 4 + 1] });
                    sum.imag = AddWide(sum.imag, { partial[warp * 4 + 2], partial[warp * 4 + 3] });
                }
                else
                {
                    sum.real.hi += partial[warp * 4];
                    sum.imag.hi += partial[warp * 4 + 2];
                }
            }
        }
    }
    return sum;
}

__simt_callee__ inline void WriteLevelResult(__gm__ float* x, __gm__ float* low, __gm__ float* diag,
    WideNumber sum, int32_t r, int32_t rhs, int32_t local, int32_t p_n, int32_t p_components, int32_t p_lowWidth,
    bool compensated)
{
    int64_t dst = static_cast<int64_t>(r) * p_n + rhs;
    Number input = Read(x, dst, p_components);
    if (compensated)
    {
        Number tail = Read(low, static_cast<int64_t>(r) * p_lowWidth + local, p_components);
        sum.real = AddWide(sum.real, { input.real, tail.real });
        sum.imag = AddWide(sum.imag, { input.imag, tail.imag });
        WideNumber result = DivideWide({ sum.real, sum.imag }, Read(diag, r, p_components), p_components);
        Write(x, dst, p_components, { result.real.hi, result.imag.hi });
        Write(low, static_cast<int64_t>(r) * p_lowWidth + local, p_components,
            { result.real.lo, result.imag.lo });
    }
    else
    {
        Write(x, dst, p_components,
            Divide({ input.real + sum.real.hi, input.imag + sum.imag.hi }, Read(diag, r, p_components), p_components));
    }
}

template <int GROUP>
__simt_vf__ __aicore__ __launch_bounds__(THREADS) inline void SolveLevel(__gm__ uint8_t* ws, int64_t p_row,
    int64_t p_col, int64_t p_val, int64_t p_diag, int64_t p_scratch, int64_t p_low, int64_t p_levelPtr,
    int64_t p_levelIdx, int32_t p_n, int32_t p_upper, int32_t p_components, int32_t level, int64_t p_reduce,
    uint32_t block, uint32_t blocks, int32_t p_lowWidth, int32_t rhsBegin, int32_t columns)
{
    auto row = reinterpret_cast<__gm__ int32_t*>(ws + p_row);
    auto col = reinterpret_cast<__gm__ int32_t*>(ws + p_col);
    auto val = reinterpret_cast<__gm__ float*>(ws + p_val);
    auto diag = reinterpret_cast<__gm__ float*>(ws + p_diag);
    auto x = reinterpret_cast<__gm__ float*>(ws + p_scratch);
    auto low = reinterpret_cast<__gm__ float*>(ws + p_low);
    auto levelPtr = reinterpret_cast<__gm__ int32_t*>(ws + p_levelPtr);
    auto levelIdx = reinterpret_cast<__gm__ int32_t*>(ws + p_levelIdx);
    uint32_t tid = AscendC::Simt::GetThreadIdx(), lane = tid % GROUP;
    if (level > 0 && tid == 0)
    {
        asc_dcci_entire(ws);
    }
    asc_syncthreads();
    int32_t begin = levelPtr[level], end = levelPtr[level + 1];
    bool compensated = p_low != 0;
    for (int64_t item = (block * THREADS + tid) / GROUP; item < static_cast<int64_t>(end - begin) * columns;
        item += blocks * (THREADS / GROUP))
    {
        int32_t r = levelIdx[begin + item / columns], local = item % columns, rhs = rhsBegin + local;
        WideNumber sum = AccumulateLevelRow<GROUP>(
            row, col, val, x, low, r, rhs, local, p_n, p_upper, p_components, p_lowWidth, lane, compensated);
        sum = ReduceLevelRow<GROUP>(ws, p_reduce, block, lane, sum, compensated);
        if (lane == 0)
        {
            WriteLevelResult(x, low, diag, sum, r, rhs, local, p_n, p_components, p_lowWidth, compensated);
        }
        if constexpr (GROUP == THREADS)
        {
            asc_syncthreads();
        }
    }
    asc_threadfence();
}

// A dependency-free matrix needs only pointwise division. Fuse the dense
// layout/alpha operations with the solve, retaining a snapshot for aliasing.
__simt_vf__ __aicore__ __launch_bounds__(THREADS) inline void SolveDiagonal(__gm__ float* input, __gm__ float* output,
    __gm__ float* alpha, __gm__ uint8_t* ws, int64_t p_diag, int64_t p_scratch, int64_t p_low, int32_t p_m, int32_t p_n,
    int32_t p_components, int32_t p_opB, int32_t p_orderB, int32_t p_orderC, int64_t p_ldb, int64_t p_ldc,
    int32_t p_deviceAlpha, float p_alphaReal, float p_alphaImag, bool snapshot, uint32_t block, uint32_t blocks,
    bool raw)
{
    auto diag = reinterpret_cast<__gm__ float*>(ws + p_diag);
    auto scratch = reinterpret_cast<__gm__ float*>(ws + p_scratch);
    Number a = p_deviceAlpha ? Read(alpha, 0, p_components) : Number { p_alphaReal, p_alphaImag };
    for (int64_t k = block * THREADS + AscendC::Simt::GetThreadIdx(); k < static_cast<int64_t>(p_m) * p_n;
        k += blocks * THREADS)
    {
        int64_t r = k / p_n, c = k % p_n;
        WideNumber scaled;
        if (snapshot)
        {
            Number b = Read(scratch, k, p_components);
            if (raw)
            {
                scaled = MultiplyWide(a, { { b.real, 0 }, { b.imag, 0 } }, p_components);
            }
            else
            {
                Number tail = Read(reinterpret_cast<__gm__ float*>(ws + p_low), k, p_components);
                scaled = { { b.real, tail.real }, { b.imag, tail.imag } };
            }
        }
        else
        {
            int64_t br = p_opB ? c : r, bc = p_opB ? r : c;
            Number b = Read(input, p_orderB ? bc * p_ldb + br : br * p_ldb + bc, p_components);
            if (p_opB == 2)
            {
                b.imag = -b.imag;
            }
            scaled = MultiplyWide(a, { { b.real, 0 }, { b.imag, 0 } }, p_components);
        }
        WideNumber result = DivideWide(scaled, Read(diag, r, p_components), p_components);
        Write(output, p_orderC ? c * p_ldc + r : r * p_ldc + c, p_components, { result.real.hi, result.imag.hi });
    }
}

// Validate all candidate diagonals before modifying any active plan values.
__simt_vf__ __aicore__ __launch_bounds__(THREADS) inline void CheckUpdate(__gm__ float* values, __gm__ uint8_t* ws,
    int64_t p_row, int64_t p_col, int64_t p_perm, int64_t p_candidateDiag, int32_t p_m, int32_t p_components,
    int32_t p_unit, int32_t p_opA, bool general)
{
    auto summary = reinterpret_cast<__gm__ int32_t*>(ws);
    auto row = reinterpret_cast<__gm__ int32_t*>(ws + p_row);
    auto col = reinterpret_cast<__gm__ int32_t*>(ws + p_col);
    auto perm = reinterpret_cast<__gm__ int32_t*>(ws + p_perm);
    auto candidate = reinterpret_cast<__gm__ float*>(ws + p_candidateDiag);
    uint32_t lane = AscendC::Simt::GetThreadIdx();
    if (lane == 0)
    {
        summary[0] = 0;
    }
    asc_syncthreads();
    for (int64_t r = lane; r < p_m; r += THREADS)
    {
        Number d = general ? Diagonal(row, col, perm, values, r, p_components, p_unit, p_opA, true)
                           : Read(values, r, p_components);
        if (!general && p_opA == 2)
        {
            d.imag = -d.imag;
        }
        if (p_unit)
        {
            d = { 1.0f, 0.0f };
        }
        if (d.real == 0.0f && d.imag == 0.0f)
        {
            atomicMax(summary, 2);
        }
        Write(candidate, r, p_components, d);
    }
}

__simt_vf__ __aicore__ __launch_bounds__(THREADS) inline void CommitUpdate(__gm__ float* values, __gm__ uint8_t* ws,
    int64_t p_candidateDiag, int64_t p_diag, int64_t p_val, int64_t p_perm, int32_t p_m, int32_t p_components,
    int32_t p_nnz, int32_t p_opA, int32_t p_unit, bool general, uint32_t block, uint32_t blocks)
{
    auto candidate = reinterpret_cast<__gm__ float*>(ws + p_candidateDiag);
    auto diag = reinterpret_cast<__gm__ float*>(ws + p_diag);
    auto val = reinterpret_cast<__gm__ float*>(ws + p_val);
    auto perm = reinterpret_cast<__gm__ int32_t*>(ws + p_perm);
    for (int64_t r = block * THREADS + AscendC::Simt::GetThreadIdx(); r < p_m; r += blocks * THREADS)
    {
        Write(diag, r, p_components, p_unit ? Number { 1, 0 } : Read(candidate, r, p_components));
    }
    if (general)
    {
        for (int64_t k = block * THREADS + AscendC::Simt::GetThreadIdx(); k < p_nnz; k += blocks * THREADS)
        {
            Number v = Read(values, perm[k], p_components);
            if (p_opA == 2)
            {
                v.imag = -v.imag;
            }
            Write(val, k, p_components, v);
        }
    }
}

__aicore__ inline void RunAnalysisPhase(SpsmPhase phase, __gm__ int32_t* ptr, __gm__ int32_t* idx,
    __gm__ float* values, __gm__ uint8_t* workspace, const SpsmPlanTiling& t, int64_t width, uint32_t block,
    uint32_t blocks)
{
    switch (phase)
    {
    case SpsmPhase::CHECK:
        asc_vf_call<CheckPointers>(dim3 { 1, 1, 1 }, ptr, reinterpret_cast<__gm__ int32_t*>(workspace), t.m, t.nnz,
            t.format, t.base);
        break;
    case SpsmPhase::DECODE:
        asc_vf_call<Decode>(dim3 { THREADS, 1, 1 }, ptr, idx, workspace, t.key0, t.nnz, t.format, t.base, t.m, t.opA,
            block, blocks);
        break;
    case SpsmPhase::MERGE:
        asc_vf_call<Merge>(dim3 { THREADS, 1, 1 }, workspace, t.key0, t.key1, t.nnz, width, block, blocks);
        break;
    case SpsmPhase::CANONICAL:
        asc_vf_call<Canonical>(dim3 { THREADS, 1, 1 }, values, workspace, t.key0, t.row, t.col, t.perm, t.val,
            t.nnz, t.components, t.opA, t.m, block, blocks);
        break;
    case SpsmPhase::ANALYZE:
        asc_vf_call<Analyze>(dim3 { 1, 1, 1 }, workspace, t.key0, t.nnz, t.row, t.col, t.perm, t.val, t.diag,
            t.rowLevel, t.scratch, t.levelPtr, t.levelIdx, t.m, t.upper, t.components, t.unit, t.opA);
        break;
    default:
        break;
    }
}

__aicore__ inline void RunSerialPhase(__gm__ uint8_t* workspace, __gm__ float* alpha, const SpsmPlanTiling& t,
    int64_t width, int32_t columns, uint32_t block, uint32_t blocks)
{
    if (t.components == 1 && t.unit)
    {
        RunSerial<1, true>(workspace, alpha, t, width, columns, block, blocks);
    }
    else if (t.components == 1)
    {
        RunSerial<1, false>(workspace, alpha, t, width, columns, block, blocks);
    }
    else if (t.unit)
    {
        RunSerial<2, true>(workspace, alpha, t, width, columns, block, blocks);
    }
    else
    {
        RunSerial<2, false>(workspace, alpha, t, width, columns, block, blocks);
    }
}

__aicore__ inline void RunLevelPhases(__gm__ uint8_t* workspace, const SpsmPlanTiling& t, int64_t width,
    int32_t columns, uint32_t block, uint32_t blocks)
{
    for (int32_t level = 0; level < t.levels; ++level)
    {
        if (level == 0)
        {
            asc_vf_call<SolveLevel<1>>(dim3 { THREADS, 1, 1 }, workspace, t.row, t.col, t.val, t.diag, t.scratch,
                t.low, t.levelPtr, t.levelIdx, t.n, t.upper, t.components, level, t.reduce, block, blocks, t.lowWidth,
                static_cast<int32_t>(width), columns);
        }
        else if (t.maxRowLen > 512 && columns <= 64)
        {
            asc_vf_call<SolveLevel<THREADS>>(dim3 { THREADS, 1, 1 }, workspace, t.row, t.col, t.val, t.diag,
                t.scratch, t.low, t.levelPtr, t.levelIdx, t.n, t.upper, t.components, level, t.reduce, block, blocks,
                t.lowWidth, static_cast<int32_t>(width), columns);
        }
        else
        {
            asc_vf_call<SolveLevel<16>>(dim3 { THREADS, 1, 1 }, workspace, t.row, t.col, t.val, t.diag, t.scratch,
                t.low, t.levelPtr, t.levelIdx, t.n, t.upper, t.components, level, t.reduce, block, blocks, t.lowWidth,
                static_cast<int32_t>(width), columns);
        }
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::SyncAll();
    }
}

__aicore__ inline void RunSolvePhase(SpsmPhase phase, __gm__ float* values, __gm__ float* dense,
    __gm__ float* alpha, __gm__ uint8_t* workspace, const SpsmPlanTiling& t, int64_t width, int32_t columns,
    uint32_t block, uint32_t blocks)
{
    switch (phase)
    {
    case SpsmPhase::SNAPSHOT:
    case SpsmPhase::SNAPSHOT_ROOTS:
    case SpsmPhase::SCATTER:
        asc_vf_call<DenseCopy>(dim3 { THREADS, 1, 1 }, dense, alpha, workspace, t.scratch, t.alphaReal, t.alphaImag,
            t.deviceAlpha, t.components, t.m, t.n, t.orderC, t.ldc, t.opB, t.orderB, t.ldb, t.low, t.rowLevel, t.diag,
            phase == SpsmPhase::SNAPSHOT_ROOTS, phase == SpsmPhase::SCATTER, block, blocks,
            t.lowWidth < t.n || t.lowRows < t.m);
        break;
    case SpsmPhase::PREPARE_PANEL:
    case SpsmPhase::PREPARE_PANEL_ROOTS:
        asc_vf_call<PreparePanel>(dim3 { THREADS, 1, 1 }, workspace, alpha, t.scratch, t.low, t.rowLevel, t.diag, t.m,
            t.n, t.lowWidth, static_cast<int32_t>(width), columns, t.components, t.deviceAlpha, t.alphaReal, t.alphaImag,
            phase == SpsmPhase::PREPARE_PANEL_ROOTS, block, blocks);
        break;
    case SpsmPhase::SOLVE:
        RunSerialPhase(workspace, alpha, t, width, columns, block, blocks);
        break;
    case SpsmPhase::SOLVE_LEVELS:
        RunLevelPhases(workspace, t, width, columns, block, blocks);
        break;
    case SpsmPhase::SOLVE_DIAGONAL:
        asc_vf_call<SolveDiagonal>(dim3 { THREADS, 1, 1 }, dense, values, alpha, workspace, t.diag, t.scratch, t.low,
            t.m, t.n, t.components, t.opB, t.orderB, t.orderC, t.ldb, t.ldc, t.deviceAlpha, t.alphaReal, t.alphaImag,
            width != 0, block, blocks, t.lowWidth < t.n);
        break;
    default:
        break;
    }
}

__aicore__ inline void RunUpdatePhase(SpsmPhase phase, __gm__ float* values, __gm__ uint8_t* workspace,
    const SpsmPlanTiling& t, uint32_t block, uint32_t blocks)
{
    if (phase == SpsmPhase::CHECK_GENERAL || phase == SpsmPhase::CHECK_DIAGONAL)
    {
        asc_vf_call<CheckUpdate>(dim3 { THREADS, 1, 1 }, values, workspace, t.row, t.col, t.perm, t.candidateDiag,
            t.m, t.components, t.unit, t.opA, phase == SpsmPhase::CHECK_GENERAL);
    }
    else
    {
        asc_vf_call<CommitUpdate>(dim3 { THREADS, 1, 1 }, values, workspace, t.candidateDiag, t.diag, t.val, t.perm,
            t.m, t.components, t.nnz, t.opA, t.unit, phase == SpsmPhase::COMMIT_GENERAL, block, blocks);
    }
}
} // namespace

extern "C" __global__ __aicore__ void spsm_plan_kernel(GM_ADDR ptr, GM_ADDR idx, GM_ADDR values, GM_ADDR dense,
    GM_ADDR alpha, GM_ADDR workspace, SpsmPlanTiling t, SpsmPhase phase, int64_t width)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    auto p = reinterpret_cast<__gm__ int32_t*>(ptr);
    auto i = reinterpret_cast<__gm__ int32_t*>(idx);
    auto v = reinterpret_cast<__gm__ float*>(values);
    auto d = reinterpret_cast<__gm__ float*>(dense);
    auto a = reinterpret_cast<__gm__ float*>(alpha);
    uint32_t block = AscendC::GetBlockIdx(), blocks = AscendC::GetBlockNum();
    int32_t columns = static_cast<int32_t>(t.n - width) < t.lowWidth ? static_cast<int32_t>(t.n - width) : t.lowWidth;
    if (phase <= SpsmPhase::ANALYZE)
    {
        RunAnalysisPhase(phase, p, i, v, workspace, t, width, block, blocks);
    }
    else if (phase <= SpsmPhase::SCATTER)
    {
        RunSolvePhase(phase, v, d, a, workspace, t, width, columns, block, blocks);
    }
    else
    {
        RunUpdatePhase(phase, v, workspace, t, block, blocks);
    }
}

void spsm_plan_kernel_do(GM_ADDR ptr, GM_ADDR idx, GM_ADDR values, GM_ADDR dense, GM_ADDR alpha, GM_ADDR workspace,
    const SpsmPlanTiling& tiling, SpsmPhase phase, int64_t width, uint32_t blocks, void* stream)
{
    spsm_plan_kernel<<<blocks, nullptr, stream>>>(ptr, idx, values, dense, alpha, workspace, tiling, phase, width);
}
