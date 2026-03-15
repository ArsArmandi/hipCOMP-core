// MIT License
//
// Modifications Copyright (C) 2023-2024 Advanced Micro Devices, Inc. All rights
// reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#pragma once

// CompressArgs, CommonHeader, FormatType are in the global namespace
#include "hipcomp_common_deps/hlif_shared_types.hpp"

#include "hipcomp.h"
#include <hip/hip_runtime.h>
#include <stdint.h>

namespace gdeflate {
namespace hlif {

uint32_t batchedGdeflateCompMaxBlockOccupancy(int device_id);

uint32_t batchedGdeflateDecompMaxBlockOccupancy(int device_id);

void gdeflateHlifBatchCompress(const CompressArgs &compress_args,
                               uint32_t max_comp_ctas, hipStream_t stream);

void gdeflateHlifBatchDecompress(const uint8_t *comp_data_buffer,
                                 uint8_t *decomp_buffer,
                                 size_t uncomp_chunk_size, uint32_t *ix_chunk,
                                 size_t num_chunks,
                                 const size_t *comp_chunk_offsets,
                                 const size_t *comp_chunk_sizes,
                                 uint32_t max_decomp_ctas, hipStream_t stream,
                                 hipcompStatus_t *output_status);

} // namespace hlif
} // namespace gdeflate
