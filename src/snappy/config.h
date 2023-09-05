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
// Modifications Copyright (C) 2023 Advanced Micro Devices, Inc. All rights reserved.

#ifndef SNAPPY_CONFIG_HIPH
#define SNAPPY_CONFIG_HIPH

#include "device_types.h"

namespace hipcomp
{
  namespace snappy
  {

    //////////////
    // COMPRESSION
    //////////////

    constexpr unsigned HASH_BITS = 12;
    //: results in 4096 hash map entries a 2 Byte

    // TBD: Tentatively limits to 2-byte codes to prevent long copy search followed by long literal
    // encoding
    constexpr unsigned MAX_LITERAL_LENGTH = 256;

    constexpr unsigned MAX_COPY_LENGTH = 64;      // Syntax limit
    constexpr unsigned MAX_COPY_DISTANCE = 32768; // Matches encoder limit as described in snappy format description

    constexpr unsigned COMP_THREADS_PER_BLOCK = 2 * warpsize; // 2 warps per stream, 1 stream per block

    ////////////////
    // DECOMPRESSION
    ////////////////

    constexpr unsigned DECOMP_THREADS_PER_BLOCK = 3 * warpsize; // 3 warps per stream, 1 stream per block

#if defined(__HIP_PLATFORM_HCC__) || defined(__HIP_PLATFORM_AMD__)
    constexpr unsigned PREFETCH_SLEEP_NS = 1600;
    constexpr unsigned DECODE_SLEEP_NS = 50;
    constexpr unsigned PROCESS_SLEEP_NS = 100;
#else
    constexpr unsigned PREFETCH_SLEEP_NS = 1600;
    constexpr unsigned DECODE_SLEEP_NS = 50;
    constexpr unsigned PROCESS_SLEEP_NS = 100;
#endif

    // Not supporting streams longer than this (not what snappy is intended for)
    constexpr unsigned SNAPPY_MAX_STREAM_SIZE = 0x7fffffff;

#ifdef __HIP_PLATFORM_AMD__
    //: results in BATCH_SIZE 64
    constexpr unsigned LOG2_BATCH_SIZE = 6;
#else
    //: results in BATCH_SIZE 32
    constexpr unsigned LOG2_BATCH_SIZE = 5;
#endif
    constexpr unsigned BATCH_SIZE = (1 << LOG2_BATCH_SIZE);
    constexpr unsigned LOG2_BATCH_COUNT = 2;
    constexpr unsigned BATCH_COUNT = (1 << LOG2_BATCH_COUNT);
    constexpr unsigned LOG2_PREFETCH_SIZE = 12;
    constexpr unsigned PREFETCH_SIZE = (1 << LOG2_PREFETCH_SIZE); // 4KB, in 32B chunks
                                                                  //: TODO: amd: does it make sense to tune this for AMD to have the same amount of chunks?
    constexpr unsigned PREFETCH_SECTORS = 8;                      // How many loads in flight when prefetching
    constexpr unsigned LITERAL_SECTORS = 4;                       // How many loads in flight when processing the literal

    constexpr unsigned LOG_CYCLECOUNT = 0;
  } // namespace snappy
} // namespace hipcomp

#endif // SNAPPY_CONFIG_HIPH