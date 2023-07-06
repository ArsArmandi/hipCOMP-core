/*
 * Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
// Modifications Copyright (C) 2023 Advanced Micro Devices, Inc. All rights reserved.

#define CATCH_CONFIG_MAIN

#include "hipcomp/cascaded.h"

#include "../../../tests/catch.hpp"
#include "../CascadedCommon.h"
#include "../CascadedMetadata.h"
#include "../CascadedMetadataOnGPU.h"
#include "common.h"

#include "hip_runtime.h"

#include <cstdlib>

#ifndef HIP_RT_CALL
#define HIP_RT_CALL(call)                                                     \
  {                                                                            \
    hipError_t hipStatus = call;                                             \
    if (hipSuccess != hipStatus) {                                           \
      fprintf(                                                                 \
          stderr,                                                              \
          "ERROR: HIP RT call \"%s\" in line %d of file %s failed with %s "   \
          "(%d).\n",                                                           \
          #call,                                                               \
          __LINE__,                                                            \
          __FILE__,                                                            \
          hipGetErrorString(hipStatus),                                      \
          hipStatus);                                                         \
      abort();                                                                 \
    }                                                                          \
  }
#endif

using namespace hipcomp;
using namespace hipcomp::highlevel;

/******************************************************************************
 * HELPER FUNCTIONS ***********************************************************
 *****************************************************************************/

namespace
{

template <typename T>
__global__ void toGPU(
    T* const output,
    T const* const input,
    size_t const num,
    hipStream_t stream)
{
  HIP_RT_CALL(hipMemcpyAsync(
      output, input, num * sizeof(T), hipMemcpyHostToDevice, stream));
}

template <typename T>
__global__ void fromGPU(
    T* const output,
    T const* const input,
    size_t const num,
    hipStream_t stream)
{
  HIP_RT_CALL(hipMemcpyAsync(
      output, input, num * sizeof(T), hipMemcpyDeviceToHost, stream));
}

} // namespace

/******************************************************************************
 * UNIT TEST ******************************************************************
 *****************************************************************************/

TEST_CASE("Metadata-fcns", "[small]")
{
  // Create test header
  hipcompCascadedFormatOpts format_opts;
  format_opts.num_RLEs = 1;
  format_opts.num_deltas = 0;
  format_opts.use_bp = 1;

  CascadedMetadata meta_in(
      format_opts,
      HIPCOMP_TYPE_INT,
      sizeof(CascadedMetadata),
      sizeof(CascadedMetadata));

  meta_in.setHeader(0, {9, 0, 0});
  meta_in.setHeader(1, {37, 5, 3});
  meta_in.setHeader(2, {49, 2, 5});
  meta_in.setDataOffset(0, 10);
  meta_in.setDataOffset(1, 38);
  meta_in.setDataOffset(2, 58);

  short version_num = 1;

  hipStream_t stream;
  hipStreamCreate(&stream);

  // get size of serialized metadata
  size_t serialized_metadata_bytes
      = CascadedMetadataOnGPU::getSerializedSizeOf(meta_in);

  // set serialized metadata
  void* d_meta;
  HIP_RT_CALL(hipMalloc(
      (void**)&d_meta, serialized_metadata_bytes)); // version number + metadata

  // Copy to GPU
  CascadedMetadataOnGPU gpuMetadata(d_meta, serialized_metadata_bytes);
  gpuMetadata.copyToGPU(meta_in, 0);

  void* meta_out;
  hipcompStatus_t err = hipcompSuccess;

  // Get temp and output sizes from metadata
  size_t temp_bytes;
  size_t out_bytes;
  size_t metadata_bytes;
  err = hipcompCascadedDecompressConfigure(d_meta, serialized_metadata_bytes, &meta_out, &metadata_bytes, &temp_bytes, &out_bytes, stream);
  REQUIRE(err == hipcompSuccess);

  CHECK(
      (static_cast<CascadedMetadata*>(meta_out))->getNumRLEs()
      == meta_in.getNumRLEs());

  CHECK(
      (static_cast<CascadedMetadata*>(meta_out))->getNumDeltas()
      == meta_in.getNumDeltas());
  CHECK(
      (static_cast<CascadedMetadata*>(meta_out))->useBitPacking()
      == meta_in.useBitPacking());
  CHECK(
      (static_cast<CascadedMetadata*>(meta_out))->getCompressedSize()
      == meta_in.getCompressedSize());
  CHECK(
      (static_cast<CascadedMetadata*>(meta_out))->getUncompressedSize()
      == meta_in.getUncompressedSize());
  CHECK(
      (static_cast<CascadedMetadata*>(meta_out))->getValueType()
      == meta_in.getValueType());
  REQUIRE(
      (static_cast<CascadedMetadata*>(meta_out))->getNumInputs()
      == meta_in.getNumInputs());
  for (size_t i = 0; i < meta_in.getNumInputs(); ++i) {
    CHECK(
        (static_cast<CascadedMetadata*>(meta_out))->getHeader(i).length
        == meta_in.getHeader(i).length);
    CHECK(
        (static_cast<CascadedMetadata*>(meta_out))->getHeader(i).minValue.i32
        == meta_in.getHeader(i).minValue.i32);
    CHECK(
        (static_cast<CascadedMetadata*>(meta_out))->getHeader(i).numBits
        == meta_in.getHeader(i).numBits);
    CHECK(
        (static_cast<CascadedMetadata*>(meta_out))->getDataOffset(i)
        == meta_in.getDataOffset(i));
  }

  CHECK(temp_bytes == 4096);
  CHECK(out_bytes == sizeof(CascadedMetadata));

//  HIP_RT_CALL(hipFree(d_meta));
  hipcompCascadedDestroyMetadata(meta_out);

}