/*
 * Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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

#include "BitcompMetadata.h"
#include "common.h"
#include "hipcomp.hpp"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <stdexcept>
#include <string>

#include <iostream>

#ifdef ENABLE_BITCOMP

#include <bitcomp.h>

namespace hipcomp
{
namespace highlevel
{

BitcompMetadata::BitcompMetadata(
    const void* const memPtr, size_t compressedBytes) :
    Metadata(HIPCOMP_TYPE_UCHAR, 0, compressedBytes, COMPRESSION_ID),
    plan(0)
{
  size_t uncompressedBytes = 0;
  bitcompDataType_t t;
  if (bitcompCreatePlanFromCompressedData(&plan, memPtr) != BITCOMP_SUCCESS
      || bitcompGetUncompressedSizeFromHandle(plan, &uncompressedBytes) != BITCOMP_SUCCESS
      || bitcompGetDataTypeFromHandle(plan, &t) != BITCOMP_SUCCESS) {
    throw HipCompException(
        hipcompErrorInternal, "BitcompMetadata: plan creation error");
  }
  hipcompType_t dataType;
  switch (t) {
  case BITCOMP_UNSIGNED_8BIT:
    dataType = HIPCOMP_TYPE_UCHAR;
    break;
  case BITCOMP_SIGNED_8BIT:
    dataType = HIPCOMP_TYPE_CHAR;
    break;
  case BITCOMP_UNSIGNED_16BIT:
    dataType = HIPCOMP_TYPE_USHORT;
    break;
  case BITCOMP_SIGNED_16BIT:
    dataType = HIPCOMP_TYPE_SHORT;
    break;
  case BITCOMP_UNSIGNED_32BIT:
    dataType = HIPCOMP_TYPE_UINT;
    break;
  case BITCOMP_SIGNED_32BIT:
    dataType = HIPCOMP_TYPE_INT;
    break;
  case BITCOMP_UNSIGNED_64BIT:
    dataType = HIPCOMP_TYPE_ULONGLONG;
    break;
  case BITCOMP_SIGNED_64BIT:
    dataType = HIPCOMP_TYPE_LONGLONG;
    break;
  default:
    throw HipCompException(
        hipcompErrorNotSupported, "BitcompMetadata: unsupported data type");
  }
  this->setUncompressedSize(uncompressedBytes);
  this->setValueType(dataType);
}

} // namespace highlevel
} // namespace hipcomp

#endif // ENABLE_BITCOMP