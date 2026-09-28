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
#pragma once

#include "kernel_operator.h"
#include "tcuscan_utils.h"

namespace tcuscan {

/**
 * @brief Performs the CSR gather operation as described in SEGMV algorithm in
 * [1].
 *
 * For each nonzero i, computes `z[i] = values[i] * x[cols[i]]`, i.e. gathers
 * the matching x entry for every CSR nonzero and scales it by that nonzero's
 * value.
 *
 * [1] Segmented Operations for Sparse Matrix Computation on Vector
 * Multiprocessors: https://dl.acm.org/doi/10.5555/865221.
 *
 * @tparam T Input data type.
 */
template <typename T>
class KernelCSRGather {
  constexpr static uint32_t BUFFER_NUM = 2;

 public:
  /**
   * @brief Class constructor.
   *
   * @param [in] values_in_len Length of the input values vector.
   * @param [in] x_in_len Length of the input x vector.
   * @param [in] tile_len Length of the tile processed in a single iteration.
   * @param [in] x_tile_elems_max Maximum number of `x` elements held in Unified
   * Buffer at once. Determines the fast-path threshold and the chunk size used
   * for larger `x` vectors.
   */

  __aicore__ inline KernelCSRGather(uint32_t values_in_len, uint32_t x_in_len,
                                    uint32_t tile_len,
                                    uint32_t x_tile_elems_max)
      : vec_core_num_(AscendC::GetBlockNum() * AscendC::GetTaskRation()),
        values_in_len_(values_in_len),
        x_in_len_(x_in_len),
        tile_len_(tile_len),
        num_tiles_(tcuscan::scalar::CeilDiv(values_in_len, tile_len_)),
        max_num_tiles_per_block_(tcuscan::scalar::CeilDiv(num_tiles_, vec_core_num_)),
        single_x_load_(x_in_len <= x_tile_elems_max),
        x_buf_elems_(single_x_load_ ? x_in_len : x_tile_elems_max),
        num_x_chunks_(tcuscan::scalar::CeilDiv(x_in_len, x_buf_elems_)) {}

  /**
   * @brief Initialize global and local memory structures.
   *
   * @param [in] values_in  Pointer to input values vector.
   * @param [in] cols_in Pointer to input column indices vector.
   * @param [in] x_in Pointer to input x vector.
   * @param [in] z_out Point to output z vector.
   */
  __aicore__ inline void Init(GM_ADDR values_in, GM_ADDR cols_in, GM_ADDR x_in,
                              GM_ADDR z_out) {
    // CSR Matrix (values, columns, row_ptr)
    global_values_.SetGlobalBuffer((__gm__ T*)values_in, values_in_len_);
    // Values are read once and not reused: bypass L2
    global_values_.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);

    global_cols_.SetGlobalBuffer((__gm__ uint32_t*)cols_in, values_in_len_);
    // Column indices are read once and not reused, bypass L2
    global_cols_.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);

    global_x_.SetGlobalBuffer((__gm__ T*)x_in, x_in_len_);
    global_z_.SetGlobalBuffer((__gm__ T*)z_out, values_in_len_);

    pipe.InitBuffer(values_q_, BUFFER_NUM, tile_len_ * sizeof(T));
    pipe.InitBuffer(cols_q_, BUFFER_NUM, tile_len_ * sizeof(uint32_t));
    // Only `x_buf_elems_` (<= x_tile_elems_max) elements of `x` are ever
    // resident,
    // so arbitrarily large `x` vectors are supported.
    pipe.InitBuffer(x_q_, 1, x_buf_elems_ * sizeof(T));
    pipe.InitBuffer(output_q_, BUFFER_NUM, tile_len_ * sizeof(T));

    pipe.InitBuffer(tbuf_, tile_len_ * sizeof(uint32_t));

    if (!single_x_load_) {
      // Scratch buffers required only by the chunked path.
      pipe.InitBuffer(clamp_buf_, tile_len_ * sizeof(int32_t));
      pipe.InitBuffer(gather_buf_, tile_len_ * sizeof(T));
      pipe.InitBuffer(sel_buf_, tile_len_ * sizeof(uint8_t));
    }
  }

  /**
   * @brief Run the kernel.
   */
  __aicore__ inline void Process() {
    uint32_t gm_offset = AscendC::GetBlockIdx() * tile_len_ * max_num_tiles_per_block_;
    const uint32_t num_tiles_to_process = tcuscan::scalar::GetWorkDistribution(
        values_in_len_, tile_len_, vec_core_num_);

    if (num_tiles_to_process == 0) {
      return;
    }

    if (single_x_load_) {
      // Fast path: the whole `x` vector fits in Unified Buffer, load it once.
      tcuscan::copy::CopyGmToVec(x_q_, global_x_, x_in_len_);
      x_lt_ = x_q_.DeQue<T>();
    }

    for (uint32_t tile_idx = 0; tile_idx < num_tiles_to_process; tile_idx++) {
      const bool is_full_tile = gm_offset + tile_len_ <= values_in_len_;
      const uint32_t num_elems =
          is_full_tile ? tile_len_ : values_in_len_ - gm_offset;

      // CopyIN
      tcuscan::copy::CopyGmToVec(values_q_, global_values_[gm_offset], num_elems);
      tcuscan::copy::CopyGmToVec(cols_q_, global_cols_[gm_offset], num_elems);

      // Compute
      if (single_x_load_) {
        ProcessTile(num_elems);
      } else {
        ProcessTileChunked(num_elems);
      }

      // CopyOut
      tcuscan::copy::CopyVecToGm(global_z_[gm_offset], output_q_, num_elems);

      gm_offset += tile_len_;
    }

    if (single_x_load_) {
      x_q_.FreeTensor<T>(x_lt_);
    }
  }

  /**
   * @brief Process tile assumes that `values_q_` is populated with a tile
   * and `x_lt_` is also populated with the whole input.
   *
   *    @param num_elems Number of elements to process
   */
  __aicore__ inline void ProcessTile(uint32_t num_elems) {
    AscendC::LocalTensor<uint32_t> cols_lt = cols_q_.DeQue<uint32_t>();
    AscendC::LocalTensor<T> z_lt = output_q_.AllocTensor<T>();

    AscendC::LocalTensor<uint32_t> cols_uint32_t = tbuf_.Get<uint32_t>();

    AscendC::ShiftLeft<uint32_t>(cols_uint32_t, cols_lt, sizeof(T) / 2,
                        cols_lt.GetSize());
    cols_q_.FreeTensor<uint32_t>(cols_lt);
    AscendC::Gather(z_lt, x_lt_, cols_uint32_t, (uint32_t)0, num_elems);
    AscendC::LocalTensor<T> values_lt = values_q_.DeQue<T>();
    AscendC::Mul(z_lt, z_lt, values_lt, num_elems);
    values_q_.FreeTensor<T>(values_lt);

    output_q_.EnQue<T>(z_lt);
  }

  /**
   * @brief Process a tile when the input `x` vector does not fit in Unified
   * Buffer.
   *
   * The output accumulator is built by scanning `x` in chunks. For every chunk
   * `[base, base + len)` the column indices are translated to chunk-local
   * indices; indices outside the chunk are clamped to a valid position (so the
   * gather never reads out of bounds) and their gathered values are then zeroed
   * via `Select`. Because each column index belongs to exactly one chunk, the
   * per-chunk contributions sum up to the correct gathered value.
   *
   * @param [in] num_elems Number of elements to process.
   */
  __aicore__ inline void ProcessTileChunked(uint32_t num_elems) {
    AscendC::LocalTensor<uint32_t> cols_lt = cols_q_.DeQue<uint32_t>();
    AscendC::LocalTensor<T> values_lt = values_q_.DeQue<T>();
    AscendC::LocalTensor<T> z_lt = output_q_.AllocTensor<T>();

    const AscendC::LocalTensor<int32_t> local_idx_lt = tbuf_.Get<int32_t>();
    const AscendC::LocalTensor<int32_t> clamp_lt = clamp_buf_.Get<int32_t>();
    const AscendC::LocalTensor<uint8_t> sel_lt = sel_buf_.Get<uint8_t>();
    const AscendC::LocalTensor<T> gather_lt = gather_buf_.Get<T>();
    const AscendC::LocalTensor<int32_t> cols_i32 = cols_lt.ReinterpretCast<int32_t>();

    // Vector ops run over the whole tile: lanes past `num_elems` hold stale
    // column indices but are clamped to a valid gather position below, so they
    // are harmless and never contribute to the written output.
    AscendC::Duplicate<T>(z_lt, static_cast<T>(0), tile_len_);

    for (uint32_t chunk = 0; chunk < num_x_chunks_; chunk++) {
      const uint32_t base = chunk * x_buf_elems_;
      const uint32_t len =
          x_in_len_ - base < x_buf_elems_ ? x_in_len_ - base : x_buf_elems_;

      tcuscan::copy::CopyGmToVec(x_q_, global_x_[base], len);
      AscendC::LocalTensor<T> x_chunk_lt = x_q_.DeQue<T>();

      // Chunk-local index of every column: `cols[i] - base`.
      AscendC::Adds(local_idx_lt, cols_i32, -static_cast<int32_t>(base), tile_len_);
      // Clamp to [0, len - 1] so the gather address is always in bounds.
      AscendC::Maxs(clamp_lt, local_idx_lt, 0, tile_len_);
      AscendC::Mins(clamp_lt, clamp_lt, static_cast<int32_t>(len - 1), tile_len_);
      // A column belongs to this chunk iff clamping did not change its index.
      AscendC::Compare(sel_lt, local_idx_lt, clamp_lt, AscendC::CMPMODE::EQ, tile_len_);
      // Turn the element index into a byte offset expected by `Gather`.
      AscendC::Muls(clamp_lt, clamp_lt, static_cast<int32_t>(sizeof(T)), tile_len_);

      AscendC::Gather(gather_lt, x_chunk_lt,
                      clamp_lt.ReinterpretCast<uint32_t>(), (uint32_t)0,
                      tile_len_);
      x_q_.FreeTensor<T>(x_chunk_lt);

      // Zero the lanes whose column index is not in this chunk, then
      // accumulate. Exactly one chunk keeps a non-zero value per lane.
      ZeroWhereNotSelected(gather_lt, sel_lt, tile_len_);
      AscendC::Add(z_lt, z_lt, gather_lt, tile_len_);
    }
    cols_q_.FreeTensor<uint32_t>(cols_lt);

    AscendC::Mul(z_lt, z_lt, values_lt, num_elems);
    values_q_.FreeTensor<T>(values_lt);

    output_q_.EnQue<T>(z_lt);
  }

 private:
  /**
   * @brief Set the elements of `data` whose `sel` bit is 0 to zero, in place.
   *
   * `Select` only operates on `half`/`float`; for `int16_t` the data is
   * reinterpreted as `half` since the selection is a bitwise copy and a zeroed
   * `half` shares the bit pattern of a zeroed `int16_t`.
   *
   * @param [in,out] data Values to mask.
   * @param [in] sel Bit mask; a set bit keeps the value, a cleared bit zeroes
   * it.
   * @param [in] count Number of elements to process.
   */
  __aicore__ inline void ZeroWhereNotSelected(const AscendC::LocalTensor<T>& data,
                                              const AscendC::LocalTensor<uint8_t>& sel,
                                              uint32_t count) {
    if constexpr (std::is_same_v<T, int16_t>) {
      const AscendC::LocalTensor<half> data_h = data.template ReinterpretCast<half>();
      AscendC::Select(data_h, sel, data_h, static_cast<half>(0),
             AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE, count);
    } else {
      AscendC::Select(data, sel, data, static_cast<T>(0),
             AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE, count);
    }
  }

  AscendC::TPipe pipe;
  AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> values_q_;
  AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> cols_q_;
  AscendC::TQue<AscendC::QuePosition::VECIN, 1> x_q_;
  AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> output_q_;

  AscendC::TBuf<AscendC::QuePosition::VECCALC> tbuf_;
  // Scratch buffers used only by the chunked path (see `ProcessTileChunked`).
  AscendC::TBuf<AscendC::QuePosition::VECCALC> clamp_buf_;
  AscendC::TBuf<AscendC::QuePosition::VECCALC> gather_buf_;
  AscendC::TBuf<AscendC::QuePosition::VECCALC> sel_buf_;

  AscendC::GlobalTensor<T> global_values_;
  AscendC::GlobalTensor<uint32_t> global_cols_;
  AscendC::GlobalTensor<uint32_t> global_rows_;
  AscendC::GlobalTensor<T> global_x_;
  AscendC::GlobalTensor<T> global_z_;

  AscendC::LocalTensor<T> x_lt_;

  const uint32_t vec_core_num_;
  const uint32_t values_in_len_;
  const uint32_t x_in_len_;
  const uint32_t tile_len_;
  const uint32_t num_tiles_;
  const uint32_t max_num_tiles_per_block_;
  // True when the whole `x` vector fits in Unified Buffer (fast path).
  const bool single_x_load_;
  // Number of `x` elements resident in Unified Buffer at once.
  const uint32_t x_buf_elems_;
  // Number of chunks `x` is split into (1 on the fast path).
  const uint32_t num_x_chunks_;
};

/**
 * @brief Run the `csr_gather` kernel.
 *
 * @tparam T Input datatype. Supports `half`, `int16_t` and `float32`.
 * @tparam ForceMixMode Indicates if kernel should schedule dummy cube
 * operations to make sure it runs in mix mode. Can be safely set to `false`
 * when running inside another mix mode kernel.
 *
 * @param [in] values_in Pointer to input vector.
 * @param [in] cols_in Pointer to column indices input vector.
 * @param [in] x_in Pointer to x input vector.
 * @param [in] z_out Pointer to output vector.
 * @param [in] values_in_len Length of the input values vector.
 * @param [in] x_in_len Length of the input x vector.
 * @param [in] tile_len Length of the tile processed in a single iteration.
 * @param [in] x_tile_elems_max Maximum number of `x` elements held in Unified
 * Buffer at once. Determines the fast-path threshold and the chunk size used
 * for larger `x` vectors.
 */
template <typename T, bool ForceMixMode = true>
__aicore__ inline void run_csr_gather(GM_ADDR values_in, GM_ADDR cols_in,
                                      GM_ADDR x_in, GM_ADDR z_out,
                                      uint32_t values_in_len, uint32_t x_in_len,
                                      uint32_t tile_len,
                                      uint32_t x_tile_elems_max = 65536) {
  if constexpr (ForceMixMode) {
    tcuscan::exec_mode::EnableCubeCores();
  }

  // If input dtype is fp32, half the chunk size of x.
  if constexpr (sizeof(T) == 4) {
    x_tile_elems_max /= 2;
  }

  if ASCEND_IS_AIV {
    static_assert(std::is_same_v<T, half> || std::is_same_v<T, int16_t> ||
                      std::is_same_v<T, float>,
                  "[csr_gather] Unsupported input dtype");
    KernelCSRGather<T> op(values_in_len, x_in_len, tile_len, x_tile_elems_max);
    op.Init(values_in, cols_in, x_in, z_out);
    op.Process();
  }
}

}  // namespace tcuscan
