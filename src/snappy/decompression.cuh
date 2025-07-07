/*
 * Copyright (c) 2018, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
// MIT License
//
// Modifications Copyright (C) 2023-2024 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#pragma once

#include "device_functions.hiph"
#include "snappy/types.h"
#include "snappy/symbol.hiph"
#include "snappy/decompression_state.hiph"
#include "snappy/decompression_prefetch.hiph"
#include "snappy/decompression_decode.hiph"
#include "snappy/decompression_process.hiph"

namespace hipcomp
{
  namespace snappy
  {
    constexpr unsigned DECOMP_WARPS_PER_BLOCK = 3; // 3 warps per stream, 1 stream per block

    template <int warpsize,
              int BATCH_SIZE,
              int LOG2_BATCH_COUNT,
              int LOG2_PREFETCH_SIZE,
              int PREFETCH_SECTORS,
              int LITERAL_SECTORS,
              int SNAPPY_MAX_STREAM_SIZE,
              int PREFETCH_SLEEP_NS,
              int DECODE_SLEEP_NS,
              int PROCESS_SLEEP_NS,
              bool LOG_CYCLECOUNT>
    class Decompressor {
    private:
      static constexpr int BATCH_COUNT = (1 << LOG2_BATCH_COUNT);
      static constexpr int PREFETCH_SIZE = (1 << LOG2_PREFETCH_SIZE); // 4KB, in 32B chunks
      using UNSNAP_STATE_S = unsnap_state_s<LZ77Symbol, BATCH_SIZE, BATCH_COUNT, PREFETCH_SIZE>;
                                                                     //: TODO: amd: does it make sense to tune this for AMD to have the same amount of chunks?
      /**
       * \brief Parses the size of the uncompressed block.
       *
       * Parses the uncompressed size that is encoded as varint at the beginning of the Snappy compressed block.
       * Varints are used to encode integers of arbitrary precision. They are Little-Endian byte sequences where
       * the 7 least-significant bits per per byte encode the integer while the most-significant bit
       * encodes if the byte sequence continues (1) or not (0).
       * Note that 0b10000000=0x80 and 0b01111111=0x7f.
       *
       * \note While varints can be used to encode integers of variable precision,
       *       here we assume that the maximum size of the integer can be 32 bits, i.e. at max 5 bytes are read.
       *
       * \param[inout] s decompression state
       * \param[inout] cur current read position/cursor in the compressed input stream.
       */
      static __device__ inline uint32_t decode_uncompressed_size(
        int *error,           //: inout
        const uint8_t *& cur, //: inout
        const uint8_t *end    //: in
      ) {
        uint32_t uncompressed_size = *cur++;
        if (uncompressed_size > 0x7f)
        {
          uint32_t c = (cur < end) ? *cur++ : 0;
          uncompressed_size = (uncompressed_size & 0x7f) | (c << 7);
          if (uncompressed_size >= (0x80 << 7))
          {
            c = (cur < end) ? *cur++ : 0;
            uncompressed_size = (uncompressed_size & ((0x7f << 7) | 0x7f)) | (c << 14);
            if (uncompressed_size >= (0x80 << 14))
            {
              c = (cur < end) ? *cur++ : 0;
              uncompressed_size =
                  (uncompressed_size & ((0x7f << 14) | (0x7f << 7) | 0x7f)) | (c << 21);
              if (uncompressed_size >= (0x80 << 21))
              {
                c = (cur < end) ? *cur++ : 0;
                if (c < 0x8)
                  uncompressed_size =
                      (uncompressed_size & ((0x7f << 21) | (0x7f << 14) | (0x7f << 7) | 0x7f)) |
                      (c << 28);
                else
                  *error = -1;
              }
            }
          }
        }
        return uncompressed_size;
      }

      template <typename DECODER, typename PREFETCHER, typename PROCESSOR>
      static __device__ inline void decompress(
          const uint8_t *const __restrict__ device_in_ptr,
          const uint64_t device_in_bytes,
          uint8_t *const __restrict__ device_out_ptr,
          const uint64_t device_out_available_bytes,
          hipcompStatus_t *const __restrict__ outputs,
          uint64_t *__restrict__ device_out_bytes
      ) {
        __shared__ __align__(16) UNSNAP_STATE_S state_g;

        int t = threadIdx.x;
        UNSNAP_STATE_S *s = &state_g;

        if (!t)
        {
          s->in.srcDevice = device_in_ptr;
          s->in.srcSize = device_in_bytes;
          s->in.dstDevice = device_out_ptr;
          s->in.dstSize = device_out_available_bytes;
        }
        if (t < BATCH_COUNT)
        {
          s->q.batch_len[t] = 0;
        }
        __syncthreads();
        if (!t)
        {
          const uint8_t *cur = reinterpret_cast<const uint8_t *>(s->in.srcDevice);
          const uint8_t *end = cur + s->in.srcSize;
          s->error = 0;
          if constexpr (LOG_CYCLECOUNT) {
            s->tstart = clock();
          }
          if (cur < end)
          {
            // Read uncompressed size (varint), limited to 32-bit
            uint32_t uncompressed_size = decode_uncompressed_size(&s->error, cur, end);
            s->uncompressed_size = uncompressed_size;
            s->bytes_left = uncompressed_size;
            s->base = cur;
            s->end = end;
            if (s->in.dstSize == 0)
              s->in.dstSize = uncompressed_size;
            if ((cur >= end && uncompressed_size != 0) || (uncompressed_size > s->in.dstSize))
            {
              s->error = -1;
            }
          }
          else
          {
            s->error = -1;
          }
          s->q.prefetch_end = 0;
          s->q.prefetch_wrpos = 0;
          s->q.prefetch_rdpos = 0;
        }
        __syncthreads();
        if (!s->error)
        {
          if (t < warpsize)
          {
            // WARP0: decode lengths and offsets, i.e. the LZ77 symbols, for WARP2
            DECODER::apply(s, t);
          }
          else if (t < 2 * warpsize)
          {
            // WARP1: prefetch byte stream for WARP0
            PREFETCHER::apply(s, t & (warpsize - 1));
          }
          else if (t < 3 * warpsize)
          {
            // WARP2: process the LZ77 symbols and write the decoded data to the output buffer
            PROCESSOR::apply(s, t & (warpsize - 1));
          }
          __syncthreads();
        }
        if (!t)
        {
          if (device_out_bytes)
            *device_out_bytes = s->uncompressed_size - s->bytes_left;
          if (outputs)
            *outputs = s->error ? hipcompErrorCannotDecompress : hipcompSuccess;
        }
      }

  public:
      static __device__ inline void apply(
        const uint8_t *const __restrict__ device_in_ptr,
        const uint64_t device_in_bytes,
        uint8_t *const __restrict__ device_out_ptr,
        const uint64_t device_out_available_bytes,
        hipcompStatus_t *const __restrict__ outputs,
        uint64_t *__restrict__ device_out_bytes)
      {
        decompress<
          DecodeSymbols<
            warpsize,
            UNSNAP_STATE_S,
            TryDecodeStringOf2To3ByteSymbols<warpsize, UNSNAP_STATE_S>,
            TryDecodeStringOf2To5ByteSymbols<warpsize, UNSNAP_STATE_S>,
            DECODE_SLEEP_NS
          >,
          PrefetchByteStream<warpsize, UNSNAP_STATE_S, PREFETCH_SECTORS, PREFETCH_SLEEP_NS>,
          ProcessSymbols<warpsize, UNSNAP_STATE_S, LITERAL_SECTORS, PROCESS_SLEEP_NS>
        >(device_in_ptr,device_in_bytes,device_out_ptr,device_out_available_bytes,outputs,device_out_bytes);
      }
  };

    /**
     * \brief Snappy decompression device function
     **/
    template <int warpsize,
              int BATCH_SIZE=warpsize,
              int LOG2_BATCH_COUNT = 2, //> The actual BATCH_COUNT is then calculated as (1 << LOG2_BATCH_COUNT). Example: Default value 2 results in BATCH_COUNT 4.
              int LOG2_PREFETCH_SIZE = 12, //> The actual PREFETCH_SIZE is then calculated as (1 << LOG2_PREFETCH_SIZE). Example: Default value 12 results in PREFETCH_SIZE 4096 (bytes).
              int PREFETCH_SECTORS = 8, //> How many loads in flight when prefetching
              int LITERAL_SECTORS = 4,
              int SNAPPY_MAX_STREAM_SIZE = 0x7fff'ffff,
              int PREFETCH_SLEEP_NS = 1600,
              int DECODE_SLEEP_NS = 50,
              int PROCESS_SLEEP_NS = 100,
              bool LOG_CYCLECOUNT = false //> Log the cycle count
              >
    __device__ inline void do_unsnap(
        const uint8_t *const __restrict__ device_in_ptr,
        const uint64_t device_in_bytes,
        uint8_t *const __restrict__ device_out_ptr,
        const uint64_t device_out_available_bytes,
        hipcompStatus_t *const __restrict__ outputs,
        uint64_t *__restrict__ device_out_bytes)
    {
      Decompressor<warpsize,
                  BATCH_SIZE,
                  LOG2_BATCH_COUNT,
                  LOG2_PREFETCH_SIZE,
                  PREFETCH_SECTORS,
                  LITERAL_SECTORS,
                  SNAPPY_MAX_STREAM_SIZE,
                  PREFETCH_SLEEP_NS,
                  DECODE_SLEEP_NS,
                  PROCESS_SLEEP_NS,
                  LOG_CYCLECOUNT
      >::apply(
        device_in_ptr,
        device_in_bytes,
        device_out_ptr,
        device_out_available_bytes,
        outputs,
        device_out_bytes
      );
    }
  } // snappy namespace
} // hipcomp namespace
