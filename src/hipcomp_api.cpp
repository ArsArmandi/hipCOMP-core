/*
 * Copyright (c) 2017-2020, NVIDIA CORPORATION. All rights reserved.
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

#include "Check.h"
#include "highlevel/BitcompMetadata.h"
#include "highlevel/CascadedMetadata.h"
#include "highlevel/LZ4Metadata.h"
#include "highlevel/Metadata.h"

#include "hipcomp.h"
#include "hipcomp/cascaded.h"
#include "hipcomp/lz4.h"
#include "hipcomp/bitcomp.h"

#include <cstdlib>
#include <cstring>
#include <iostream>

using namespace hipcomp;
using namespace hipcomp::highlevel;

#define DEPRECATED_FUNC(replacement)                                           \
  do {                                                                         \
    static bool has_warned = false;                                            \
    if (!has_warned) {                                                         \
      warn_deprecated(__FUNCTION__, replacement);                              \
      has_warned = true;                                                       \
    }                                                                          \
  } while (false)

namespace
{

void warn_deprecated(
    const std::string& name, const std::string& replacement = "")
{
  const char* const level = std::getenv("HIPCOMP_WARN");
  if (level == nullptr || strcmp(level, "SILENT") != 0) {
    std::cerr << "WARNING: The function '" << name
              << "' is deprecated and "
                 "will be removed in future releases. ";
    if (!replacement.empty()) {
      std::cerr << "Please use '" << replacement << "' instead. ";
    }
    std::cerr << "To suppress this message, define the environment variable "
                 "'HIPCOMP_WARN=SILENT'."
              << std::endl;
  }
}

} // namespace

hipcompStatus_t hipcompDecompressGetMetadata(
    const void* const in_ptr,
    const size_t in_bytes,
    void** const metadata_ptr,
    cudaStream_t stream)
{
  DEPRECATED_FUNC("hipcomp*DecompressConfigure()");

  cudaStreamSynchronize(stream);
  if (hipcompLZ4IsData(in_ptr, in_bytes, stream)) {
    size_t metadata_bytes, temp_bytes, uncompressed_bytes;
    return hipcompLZ4DecompressConfigure(
        in_ptr,
        in_bytes,
        metadata_ptr,
        &metadata_bytes,
        &temp_bytes,
        &uncompressed_bytes,
        stream);
  }
#ifdef ENABLE_BITCOMP
  else if (hipcompIsBitcompData(in_ptr, in_bytes)) {
    size_t temp_bytes;
    size_t out_bytes;
    size_t metadata_bytes;
    return hipcompBitcompDecompressConfigure(
        in_ptr,
        in_bytes,
        metadata_ptr,
        &metadata_bytes,
        &temp_bytes,
        &out_bytes,
        stream);
  }
#endif
  else {
    size_t temp_bytes;
    size_t out_bytes;
    size_t metadata_bytes;

    return hipcompCascadedDecompressConfigure(
               in_ptr, 
               in_bytes, 
               metadata_ptr, 
               &metadata_bytes, 
               &temp_bytes, 
               &out_bytes, 
               stream);
  }
}

void hipcompDecompressDestroyMetadata(void* const metadata_ptr)
{
  DEPRECATED_FUNC("hipcomp*DestroyMetadata()");
#ifdef ENABLE_BITCOMP
  const Metadata* const metadata = static_cast<const Metadata*>(metadata_ptr);
#endif
  if (hipcompLZ4IsMetadata(metadata_ptr)) {
    hipcompLZ4DestroyMetadata(metadata_ptr);
  }
#ifdef ENABLE_BITCOMP
  else if (metadata->getCompressionType() == BitcompMetadata::COMPRESSION_ID) {
    hipcompBitcompDestroyMetadata(metadata_ptr);
  }
#endif
  else {
    hipcompCascadedDestroyMetadata(metadata_ptr);
  }
}

hipcompStatus_t hipcompDecompressGetTempSize(
    const void* const metadata_ptr, size_t* const temp_bytes)
{
  DEPRECATED_FUNC("hipcomp*DecompressConfigure()");

  const Metadata* const metadata = static_cast<const Metadata*>(metadata_ptr);
  if (hipcompLZ4IsMetadata(metadata_ptr)) {
    try {
      size_t metadata_bytes = sizeof(LZ4Metadata);
      size_t uncompressed_bytes;
      CHECK_API_CALL(hipcompLZ4DecompressConfigure(
          nullptr,
          metadata_bytes,
          (void**)&metadata_ptr,
          &metadata_bytes,
          temp_bytes,
          &uncompressed_bytes,
          0));
    } catch (const std::exception& e) {
      return Check::exception_to_error(e, "hipcompDecompressGetTempSize()");
    }
    return hipcompSuccess;
  }
  else if (metadata->getCompressionType() == BitcompMetadata::COMPRESSION_ID) {
    *temp_bytes = 0;
#ifdef ENABLE_BITCOMP
    return hipcompSuccess;
#else
    return hipcompErrorNotSupported;
#endif
  }
  else {
    *temp_bytes = static_cast<const CascadedMetadata*>(metadata_ptr)->getTempBytes();
    return hipcompSuccess;

  }
}

hipcompStatus_t hipcompDecompressGetOutputSize(
    const void* const metadata_ptr, size_t* const output_bytes)
{
  DEPRECATED_FUNC("hipcomp*DecompressConfigure()");

  if (metadata_ptr == nullptr) {
    std::cerr << "Cannot get the output size from a null metadata."
              << std::endl;
    return hipcompErrorInvalidValue;
  }
  if (output_bytes == nullptr) {
    std::cerr << "Cannot write the output size to a null location."
              << std::endl;
    return hipcompErrorInvalidValue;
  }

  const Metadata* const metadata = static_cast<const Metadata*>(metadata_ptr);
  *output_bytes = metadata->getUncompressedSize();

  return hipcompSuccess;
}

hipcompStatus_t hipcompDecompressGetType(
    const void* const metadata_ptr, hipcompType_t* const type)
{
  DEPRECATED_FUNC(""); // no replacement

  if (metadata_ptr == nullptr) {
    std::cerr << "Cannot get the type from a null metadata." << std::endl;
    return hipcompErrorInvalidValue;
  }
  if (type == nullptr) {
    std::cerr << "Cannot write the typeto a null location." << std::endl;
    return hipcompErrorInvalidValue;
  }

  if (hipcompLZ4IsMetadata(metadata_ptr)) {
    // LZ4 always operates on bytes
    *type = HIPCOMP_TYPE_CHAR;
  }
  else {
    const Metadata* const metadata = static_cast<const Metadata*>(metadata_ptr);
    *type = metadata->getValueType();
  }

  return hipcompSuccess;
}

hipcompStatus_t hipcompDecompressAsync(
    const void* const in_ptr,
    const size_t in_bytes,
    void* const temp_ptr,
    const size_t temp_bytes,
    void* const metadata_ptr,
    void* const out_ptr,
    const size_t out_bytes,
    cudaStream_t stream)
{
  DEPRECATED_FUNC("hipcomp*DecompressAsync()");

  const Metadata* const metadata = static_cast<const Metadata*>(metadata_ptr);
  if (hipcompLZ4IsMetadata(metadata_ptr)) {
    return hipcompLZ4DecompressAsync(
        in_ptr,
        in_bytes,
        metadata_ptr,
        0,
        temp_ptr,
        temp_bytes,
        out_ptr,
        out_bytes,
        stream);
  }
  else if (metadata->getCompressionType() == BitcompMetadata::COMPRESSION_ID) {
#ifdef ENABLE_BITCOMP
    const size_t metadata_bytes = sizeof(BitcompMetadata);
    return hipcompBitcompDecompressAsync(
        in_ptr,
        in_bytes,
        metadata_ptr,
        metadata_bytes,
        temp_ptr,
        temp_bytes,
        out_ptr,
        out_bytes,
        stream);
#else
    return hipcompErrorNotSupported;
#endif
  } else {
    size_t metadata_bytes = 0;
    return hipcompCascadedDecompressAsync(
        in_ptr,
        in_bytes,
        metadata_ptr,
        metadata_bytes,
        temp_ptr,
        temp_bytes,
        out_ptr,
        out_bytes,
        stream);
  }
}