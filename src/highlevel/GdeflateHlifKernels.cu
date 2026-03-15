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

#include "GdeflateHlifKernels.h"
#include "HipUtils.h"
#include "gdeflate.h"
#include "hipcomp/gdeflate.h"
#include "lowlevel/gdeflateKernels.h"

#include <hip/hip_runtime.h>

// ---------------------------------------------------------------------------
// Small helper kernels (global namespace, no HIP namespace conflicts)
// ---------------------------------------------------------------------------

__global__ static void
gdeflate_buildPtrArrayFromOffsets(const void **d_ptrs, const uint8_t *base,
                                  const size_t *offsets, size_t n) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    d_ptrs[i] = base + offsets[i];
}

__global__ static void gdeflate_buildUniformPtrArray(void **d_ptrs,
                                                     uint8_t *base,
                                                     size_t stride, size_t n) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    d_ptrs[i] = base + i * stride;
}

__global__ static void gdeflate_fillConstSizeArray(size_t *d_sizes, size_t val,
                                                   size_t n) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    d_sizes[i] = val;
}

// Fill sizes with min(chunk_size, total_size - i*chunk_size) so the last
// (possibly partial) chunk gets its actual byte count rather than chunk_size.
__global__ static void gdeflate_fillActualSizes(size_t *d_sizes,
                                                size_t chunk_size,
                                                size_t total_size, size_t n) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    size_t offset = i * chunk_size;
    d_sizes[i] = (offset + chunk_size <= total_size) ? chunk_size
                                                      : (total_size - offset);
  }
}

__global__ static void gdeflate_fillUniformOffsets(size_t *d_offsets,
                                                   size_t stride, size_t n) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    d_offsets[i] = i * stride;
}

// Sum sizes[] into *out  (single-thread; batch counts are small).
__global__ static void gdeflate_sumSizes(size_t *out, const size_t *sizes,
                                         size_t n) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    size_t total = 0;
    for (size_t i = 0; i < n; i++)
      total += sizes[i];
    *out = total;
  }
}

// Fill the CommonHeader for GDeflate.
__global__ static void gdeflate_fillHeader(CommonHeader *hdr,
                                           size_t decomp_data_size,
                                           size_t num_chunks,
                                           size_t uncomp_chunk_size,
                                           uint32_t comp_data_offset) {
  if (threadIdx.x != 0 || blockIdx.x != 0)
    return;
  hdr->magic_number = 0;
  hdr->major_version = 2;
  hdr->minor_version = 2;
  hdr->format = GDeflate; // global-namespace FormatType enum
  hdr->comp_data_size = 0; // set later by gdeflate_sumSizes
  hdr->decomp_data_size = decomp_data_size;
  hdr->num_chunks = num_chunks;
  hdr->include_chunk_starts = true;
  hdr->full_comp_buffer_checksum = 0;
  hdr->decomp_buffer_checksum = 0;
  hdr->include_per_chunk_comp_buffer_checksums = false;
  hdr->include_per_chunk_decomp_buffer_checksums = false;
  hdr->uncomp_chunk_size = uncomp_chunk_size;
  hdr->comp_data_offset = comp_data_offset;
}

namespace gdeflate {
namespace hlif {

// ---------------------------------------------------------------------------
// Block occupancy: 2× SM count is a reasonable upper bound.
// ---------------------------------------------------------------------------
uint32_t batchedGdeflateCompMaxBlockOccupancy(int device_id) {
  hipDeviceProp_t prop =
      hipcomp::HipUtils::device_properties(device_id);
  return static_cast<uint32_t>(prop.multiProcessorCount * 2);
}

uint32_t batchedGdeflateDecompMaxBlockOccupancy(int device_id) {
  hipDeviceProp_t prop =
      hipcomp::HipUtils::device_properties(device_id);
  return static_cast<uint32_t>(prop.multiProcessorCount * 2);
}

// ---------------------------------------------------------------------------
// Batch compression
//
// Writes each chunk directly into comp_buffer at stride max_comp_chunk_size,
// which exactly fits the space reserved by calculate_max_compressed_output_size.
// ---------------------------------------------------------------------------
void gdeflateHlifBatchCompress(const CompressArgs &compress_args,
                               uint32_t /*max_comp_ctas*/,
                               hipStream_t stream) {
  const size_t n = compress_args.num_chunks;
  const size_t uc = compress_args.uncomp_chunk_size;
  const size_t mc = compress_args.max_comp_chunk_size;

  if (n == 0)
    return;

  const void **d_in_ptrs = nullptr;
  void **d_out_ptrs = nullptr;
  size_t *d_in_sizes = nullptr;

  hipcomp::HipUtils::check(hipMalloc(&d_in_ptrs, n * sizeof(void *)));
  hipcomp::HipUtils::check(hipMalloc(&d_out_ptrs, n * sizeof(void *)));
  hipcomp::HipUtils::check(hipMalloc(&d_in_sizes, n * sizeof(size_t)));

  const int threads = 256;
  const int blocks = static_cast<int>((n + threads - 1) / threads);

  gdeflate_buildUniformPtrArray<<<blocks, threads, 0, stream>>>(
      (void **)d_in_ptrs,
      const_cast<uint8_t *>(compress_args.decomp_buffer), uc, n);

  // Use actual chunk sizes: the last chunk may be smaller than uc.
  gdeflate_fillActualSizes<<<blocks, threads, 0, stream>>>(
      d_in_sizes, uc, compress_args.decomp_buffer_size, n);

  gdeflate_buildUniformPtrArray<<<blocks, threads, 0, stream>>>(
      d_out_ptrs, compress_args.comp_buffer, mc, n);

  gdeflate_fillUniformOffsets<<<blocks, threads, 0, stream>>>(
      compress_args.comp_chunk_offsets, mc, n);

  size_t temp_bytes = 0;
  gdeflate::compressGetTempSize(n, uc, &temp_bytes, gdeflate::HIGH_THROUGHPUT);
  void *d_temp = nullptr;
  hipcomp::HipUtils::check(hipMalloc(&d_temp, temp_bytes > 0 ? temp_bytes : 1));

  // gdeflate compression is CPU-side; sync before calling.
  hipcomp::HipUtils::check(hipStreamSynchronize(stream));

  hipcompBatchedGdeflateOpts_t opts = {0};
  hipcompBatchedGdeflateCompressAsync(
      (const void *const *)d_in_ptrs, (const size_t *)d_in_sizes, uc, n,
      d_temp, temp_bytes, (void *const *)d_out_ptrs,
      compress_args.comp_chunk_sizes, opts, stream);

  // Write total compressed size into the header's comp_data_size field.
  gdeflate_sumSizes<<<1, 1, 0, stream>>>(compress_args.ix_output,
                                         compress_args.comp_chunk_sizes, n);

  const uint32_t comp_data_offset = static_cast<uint32_t>(
      (uintptr_t)compress_args.comp_buffer -
      (uintptr_t)compress_args.common_header);
  gdeflate_fillHeader<<<1, 1, 0, stream>>>(compress_args.common_header,
                                           compress_args.decomp_buffer_size, n,
                                           uc, comp_data_offset);

  hipcomp::HipUtils::check(hipFree(d_temp));
  hipcomp::HipUtils::check(hipFree(d_in_sizes));
  hipcomp::HipUtils::check(hipFree(d_out_ptrs));
  hipcomp::HipUtils::check(hipFree(d_in_ptrs));
}

// ---------------------------------------------------------------------------
// Batch decompression
// ---------------------------------------------------------------------------
void gdeflateHlifBatchDecompress(const uint8_t *comp_data_buffer,
                                 uint8_t *decomp_buffer,
                                 size_t uncomp_chunk_size,
                                 uint32_t * /*ix_chunk*/, size_t num_chunks,
                                 const size_t *comp_chunk_offsets,
                                 const size_t *comp_chunk_sizes,
                                 uint32_t /*max_decomp_ctas*/,
                                 hipStream_t stream,
                                 hipcompStatus_t *output_status,
                                 size_t total_decomp_size) {
  if (num_chunks == 0)
    return;

  const void **d_comp_ptrs = nullptr;
  void **d_decomp_ptrs = nullptr;
  size_t *d_decomp_sizes = nullptr;

  hipcomp::HipUtils::check(hipMalloc(&d_comp_ptrs, num_chunks * sizeof(void *)));
  hipcomp::HipUtils::check(hipMalloc(&d_decomp_ptrs, num_chunks * sizeof(void *)));
  hipcomp::HipUtils::check(
      hipMalloc(&d_decomp_sizes, num_chunks * sizeof(size_t)));

  const int threads = 256;
  const int blocks = static_cast<int>((num_chunks + threads - 1) / threads);

  gdeflate_buildPtrArrayFromOffsets<<<blocks, threads, 0, stream>>>(
      d_comp_ptrs, comp_data_buffer, comp_chunk_offsets, num_chunks);

  gdeflate_buildUniformPtrArray<<<blocks, threads, 0, stream>>>(
      d_decomp_ptrs, decomp_buffer, uncomp_chunk_size, num_chunks);

  // Compute per-chunk uncompressed sizes from the known total: the last chunk
  // may be smaller than uncomp_chunk_size. Avoid getDecompressSizeAsync which
  // reads per-chunk sizes from compressed-stream headers — that kernel returns
  // 0 on gfx1030 (RDNA2) in the current libgdeflate build.
  gdeflate_fillActualSizes<<<blocks, threads, 0, stream>>>(
      d_decomp_sizes, uncomp_chunk_size, total_decomp_size, num_chunks);

  // Sync before calling into the gdeflate library: it may launch kernels on
  // an internal stream, so our pointer and size arrays must be fully ready.
  hipcomp::HipUtils::check(hipStreamSynchronize(stream));

  size_t temp_bytes = 0;
  gdeflate::decompressGetTempSize(num_chunks, uncomp_chunk_size, &temp_bytes);
  void *d_temp = nullptr;
  hipcomp::HipUtils::check(hipMalloc(&d_temp, temp_bytes > 0 ? temp_bytes : 1));

  // Reuse output_status buffer as gdeflate status storage (same size guaranteed
  // by static_assert in gdeflateKernels.cu).
  auto *d_gdeflate_status =
      reinterpret_cast<gdeflate::gdeflateStatus_t *>(output_status);

  gdeflate::decompressAsync(
      (const void *const *)d_comp_ptrs, comp_chunk_sizes,
      (const size_t *)d_decomp_sizes,
      nullptr,     // actual sizes not required
      0,           // unused parameter
      num_chunks, d_temp, temp_bytes, (void *const *)d_decomp_ptrs,
      d_gdeflate_status, stream);

  // Use hipDeviceSynchronize rather than hipStreamSynchronize: the gdeflate
  // library may enqueue GPU kernels on an internal stream rather than the
  // caller's stream, so we must wait for all pending device work.
  hipcomp::HipUtils::check(hipDeviceSynchronize());

  if (output_status)
    hipcomp::convertGdeflateOutputStatuses(output_status, num_chunks, stream);

  hipcomp::HipUtils::check(hipFree(d_temp));
  hipcomp::HipUtils::check(hipFree(d_decomp_sizes));
  hipcomp::HipUtils::check(hipFree(d_decomp_ptrs));
  hipcomp::HipUtils::check(hipFree(d_comp_ptrs));
}

} // namespace hlif
} // namespace gdeflate
