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
 * @brief Performs a "row-wise" inclusive scan on an input vector.
 *
 * The algorithm splits the input vector into chunks ("rows") of size
 * \f$K = \textit{matmul_k_size}\f$ and performs local inclusive scans on all
 * the chunks separately. Local scan is performed by transforming the problem
 * into matrix multiplication. Matrix \f$B\f$ is an upper triangular matrix of
 * size \f$K \times K\f$ with ones on the main diagonal and above. Matrix
 * \f$A\f$ is created from the input vector by reshaping it into a matrix with
 * \f$K\f$ dimension equal to \f$K\f$ and \f$M\f$ dimension equal to
 * \f$\textit{matmul_m_size}\f$. Then matrix \f$A\f$ is splitted into \f$M
 * \times K\f$ tiles that are multiplied by the same matrix \f$B\f$. The tiles
 * are equally distributed into different cube cores.
 *
 * The algorithm assumes that the length of the input vector is divisible by
 * \f$M K\f$.
 *
 * The algorithm is a first step in the full-scan calculation.
 *
 * The algorithm supports following data type configurations:
 *   - inputs: `int8_t`; output: `int32_t`;
 *   - inputs: `uint8_t`; output: `uint32_t`;
 *   - inputs: `half`; output: `float`.
 *
 * @tparam InputT Data type of the input vector.
 * @tparam SyncAfter If true, synchronize vector units with cube unit after each
 * matrix tile.
 */
template <typename InputT, bool SyncAfter = false>
class KernelRowScan {
  using OutputT = tcuscan::cube_unit::CubeOutType_t<InputT>;

 public:
  /**
   * @brief Class constructor.
   *
   * @param [in] matmul_k_size Size of the K dimension of A matrix.
   * @param [in] matmul_m_size Size of the M dimension of A matrix.
   * @param [in] vec_len Number of elements in an input vector.
   */
  __aicore__ inline KernelRowScan(uint32_t matmul_k_size,
                                  uint32_t matmul_m_size, uint32_t vec_len)
      : block_num_(AscendC::GetBlockNum()),
        matmul_k_size_(matmul_k_size),
        matmul_m_size_(matmul_m_size),
        tile_size_(matmul_k_size * matmul_m_size),
        vec_len_(vec_len),
        num_tiles_(tcuscan::scalar::FloorDiv(vec_len, tile_size_)),
        max_num_tiles_per_block_(
            tcuscan::scalar::CeilDiv(num_tiles_, block_num_)) {
    static_assert(
        tcuscan::cube_unit::IsCubeSupported<InputT>,
        "Unsupported input Cube dtype. Please use half, float, or int8_t.");
  }

  /**
   * @brief Initialize global and local memory structures.
   *
   * @param [in] input Pointer to input vector in global memory.
   * @param [in] b Pointer to upper triangular matrix filled with ones in
   * global memory.
   * @param [in] output Pointer to output vector in global memory.
   */
  __aicore__ inline void Init(GM_ADDR input, GM_ADDR b, GM_ADDR output) {
    global_A_.SetGlobalBuffer((__gm__ InputT*)input, vec_len_);
    global_B_.SetGlobalBuffer((__gm__ InputT*)b);
    global_C_.SetGlobalBuffer((__gm__ OutputT*)output, vec_len_);

    pipe.InitBuffer(a1_q_, 1, a_cube_tile_size_ * sizeof(InputT));
    pipe.InitBuffer(a2_q_, 1, a_cube_tile_size_ * sizeof(InputT));
    pipe.InitBuffer(b1_q_, 1, b_cube_tile_size_ * sizeof(InputT));
    pipe.InitBuffer(b2_q_, 1, b_cube_tile_size_ * sizeof(InputT));

    pipe.InitBuffer(co1_q_, 1, c_cube_tile_size_ * sizeof(OutputT));
  }

  /**
   * @brief Run the kernel - process all tiles.
   */
  __aicore__ inline void Process() {
    // Load the B matrix only once from global memory
    const uint32_t num_tiles_to_process =
        tcuscan::scalar::GetWorkDistribution(vec_len_, tile_size_, block_num_);

    if (num_tiles_to_process == 0) {
      return;
    }

    LoadBToL0();
    for (uint32_t idx = 0; idx < num_tiles_to_process; ++idx) {
      CubeIter(idx);
      if constexpr (SyncAfter) {
        tcuscan::sync::SyncGroup<tcuscan::sync::GroupSyncDirection::FULL>();
      }
    }
    tcuscan::queue::FreeFromQ<InputT>(b2_q_);
  }

 private:
  __aicore__ inline void CubeIter(uint32_t iter_idx) {
    tcuscan::copy::CopyGmToL1A(
        a1_q_,
        global_A_[AscendC::GetBlockIdx() * a_cube_tile_size_ *
                      max_num_tiles_per_block_ +
                  iter_idx * a_cube_tile_size_],
        m_blocks_, k_blocks_);
    LoadA1ToA2();
    tcuscan::cube_unit::Multiply<InputT, false /* accumulate_c */,
                                 true /* free_a*/, false /* free_b */>(
        a2_q_, b2_q_, co1_q_, matmul_m_size_, N_, K_);
    tcuscan::copy::CopyCL0ToGlobal(
        global_C_[AscendC::GetBlockIdx() * c_cube_tile_size_ *
                      max_num_tiles_per_block_ +
                  iter_idx * c_cube_tile_size_],
        co1_q_, matmul_m_size_, N_);
  }

  __aicore__ inline void LoadBToL0() {
    tcuscan::copy::CopyTransposedGmToL0B(b2_q_, b1_q_, global_B_, k_blocks_,
                                         n_blocks_);
  }

  __aicore__ inline void LoadA1ToA2() {
    tcuscan::copy::CopyL1ToL0A<InputT, true>(a2_q_, a1_q_, m_blocks_,
                                             k_blocks_);
  }

  AscendC::TPipe pipe;

  AscendC::TQue<AscendC::QuePosition::A1, 1> a1_q_;
  AscendC::TQue<AscendC::QuePosition::A2, 1> a2_q_;
  AscendC::TQue<AscendC::QuePosition::B1, 1> b1_q_;
  AscendC::TQue<AscendC::QuePosition::B2, 1> b2_q_;

  AscendC::TQue<AscendC::QuePosition::CO1, 1> co1_q_;

  AscendC::GlobalTensor<InputT> global_A_, global_B_;
  AscendC::GlobalTensor<OutputT> global_C_;

  const uint32_t block_num_;
  const uint32_t matmul_k_size_;
  const uint32_t matmul_m_size_;
  const uint32_t tile_size_;
  const uint32_t vec_len_;
  const uint32_t num_tiles_;
  const uint32_t max_num_tiles_per_block_;

  const uint32_t K_ = matmul_k_size_;
  const uint32_t N_ = matmul_k_size_;

  const uint32_t n_blocks_ = N_ / 16;
  const uint32_t k_blocks_ = sizeof(InputT) * K_ / 32;
  const uint32_t m_blocks_ = matmul_m_size_ / 16;

  const uint32_t a_cube_tile_size_ = matmul_m_size_ * K_;
  const uint32_t b_cube_tile_size_ = K_ * N_;
  const uint32_t c_cube_tile_size_ = matmul_m_size_ * N_;
};

}  // namespace tcuscan
