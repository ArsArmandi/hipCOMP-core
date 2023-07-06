/*
 * Copyright (c) 2018-2020, NVIDIA CORPORATION. All rights reserved.
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

#pragma once

#include "CascadedMetadata.h"

#include <cassert>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace hipcomp
{
namespace highlevel
{

// Opaque structures
typedef int hipcompHandle_t;
typedef int hipcompConfig_t;
struct hipcompIntConfig_t;

/* Simple compress handle */
struct hipcompCompressHandle_t
{
  int numRLEs = 0;
  int numDeltas = 0;
  int bitPacking = 0;
  int sysmem = 0; // where the compressed data resides
  int duplicate = 1;

  hipcompType_t dataInType;
  hipcompType_t dataRunType;
  size_t comp_size = 0;
  size_t all_comp_size = 0;
};

typedef int hipcompConfig_t;

hipcompIntConfig_t* createConfig(const CascadedMetadata* metadata);

void destroyConfig(hipcompIntConfig_t* config);

/* Adds an RLE stage (A,B,C):(1,2,3) -> (A,B,B,C,C,C). Takes two inputs and
 * produces one output.
 *
 * Input.  valsId is the buffer id of the input values, runsId is the buffer id
 * of the input runs. If the id < numInputs specified in hipcompDecompressLaunch
 * then the data is taken from user-provided inputHdrs/inputData in
 * hipcompDecompressLaunch. In this case the type and packing options must be
 * specified. If valPacking=0 then the input data is treated as of valType
 * type. If valPacking=1 then each value is stored as an unsigned integer of
 * numBits bits and represents an offset to minValue of inputType type. If
 * the id >= numInputs then the data is taken from the output of one of the
 * previous decompression layers.
 *
 * Output.  outputId should be equal or larger than numInputs. If the output id
 * is the same as outputId in 'config' then this is the
 * final decompression stage. maxOutputSize specifies the maximum decompressed
 * chunk size (number of elements) for this stage. */
hipcompStatus_t hipcompConfigAddRLE_BP(
    hipcompIntConfig_t* const config,
    int outputId,
    size_t maxOutputSize,
    int valId,
    hipcompType_t valType,
    int valPacking,
    int runId,
    hipcompType_t runType,
    int runPacking);

/* Adds a Delta stage (A,B,C) -> (A,A+B,A+B+C). Takes one input and produces
 * one output.
 *
 * Input.  valsId is the buffer id of the input values, runsId is the buffer id
 * of the input runs. If the id < numInputs specified in hipcompDecompressLaunch
 * then the data is taken from user-provided inputHdrs/inputData in
 * hipcompDecompressLaunch. In this case the type and packing options must be
 * specified. If valPacking=0 then the input data is treated as of valType
 * type. If valPacking=1 then each value is stored as an unsigned integer of
 * numBits bits and represents an offset to minValue of inputType type. If
 * the id >= numInputs then the data is taken from the output of one of the
 * previous decompression layers.
 *
 * Output.  outputId should be equal or larger than numInputs. If the output id
 * is the same as outputId specified in 'config' then this is the
 * final decompression stage. maxOutputSize specifies the maximum decompressed
 * chunk size (number of elements) for this stage. */
hipcompStatus_t hipcompConfigAddDelta_BP(
    hipcompIntConfig_t* const config,
    int outputId,
    size_t maxOutputSize,
    int valId,
    hipcompType_t valType,
    int valPacking);

/* Adds an RLE stage (A,B,C):(1,2,3) -> (A,B,B,C,C,C). Takes two inputs and
 * produces one output.
 *
 * Input.  valsId is the buffer id of the input values, runsId is the buffer id
 * of the input runs. If the id < numInputs specified in hipcompDecompressLaunch
 * then the data is taken from user-provided inputHdrs/inputData in
 * hipcompDecompressLaunch. In this case the type option must be specified. If
 * the id >= numInputs then the data is taken from the output of one of the
 * previous decompression layers.
 *
 * Output.  outputId should be equal or larger than numInputs. If the output id
 * is the same as outputId specified in 'config' then this is the
 * final decompression stage. maxOutputSize specifies the maximum decompressed
 * chunk size (number of elements) for this stage. */
hipcompStatus_t hipcompConfigAddRLE(
    hipcompIntConfig_t* const config,
    int outputId,
    size_t maxOutputSize,
    int valId,
    hipcompType_t valType,
    int runId,
    hipcompType_t runType);

/* Adds a Delta stage (A,B,C) -> (A,A+B,A+B+C). Takes one input and produces
 * one output.
 *
 * Input.  valsId is the buffer id of the input values, runsId is the buffer id
 * of the input runs. If the id < numInputs specified in hipcompDecompressLaunch
 * then the data is taken from user-provided inputHdrs/inputData in
 * hipcompDecompressLaunch. In this case the type option must be specified. If
 * the id >= numInputs then the data is taken from the output of one of the
 * previous decompression layers.
 *
 * Output.  outputId should be equal or larger than numInputs. If the output id
 * is the same as outputId specified in 'config' then this is the
 * final decompression stage. maxOutputSize specifies the maximum decompressed
 * chunk size (number of elements) for this stage. */
hipcompStatus_t hipcompConfigAddDelta(
    hipcompIntConfig_t* const config,
    int outputId,
    size_t maxOutputSize,
    int valId,
    hipcompType_t valType);
/* Adds a Byte-packing stage. Takes one input and produces one output.
 *
 * Input.  valsId is the buffer id of the input values. If the id < numInputs
 * specified in hipcompDecompressLaunch then the data is taken from user-provided
 * inputHdrs/inputData in hipcompDecompressLaunch. In this case the type option
 * must be specified. Each value is stored as an unsigned integer of
 * numBits bits and represents an offset to minValue of inputType type. If
 * the id >= numInputs then the data is taken from the output of one of the
 * previous decompression layers.
 *
 * Output.  outputId should be equal or larger than numInputs. If the output id
 * is the same as outputId specified in 'config' then this is the
 * final decompression stage. maxOutputSize specifies the maximum decompressed
 * chunk size (number of elements) for this stage. */
hipcompStatus_t hipcompConfigAddBP(
    hipcompIntConfig_t* const config,
    int outputId,
    size_t maxOutputSize,
    int valId,
    hipcompType_t valType);

/* Creates a decompression handle assigned to the specified cascaded scheme
 * described by the config.  Assumes that the workspaceStorage is pre-allocated
 * and accessible by the GPU, otherwise flags an error.
 * Multiple handles can be created but each handle takes resources.
 * */
hipcompStatus_t hipcompCreateHandleAsync(
    hipcompHandle_t* handle,
    hipcompIntConfig_t* const config,
    void* workspaceStorage,
    size_t workspaceBytes,
    cudaStream_t stream);

/* Reconfigures the workspace. This will try to adjust the allocation policy to
 * fit the specified memory budget of workspaceBytes. On success the handle
 * will release the previous temporary storage and use the new memory space,
 * otherwise hipErrorNotSupported will be returned and no changes to the
 * workspace will be made. */
hipcompStatus_t hipcompSetWorkspace(
    hipcompHandle_t handle, void* workspaceStorage, size_t workspaceBytes);

/* Gets the current workspace size in bytes. */
hipcompStatus_t
hipcompGetWorkspaceSize(hipcompHandle_t handle, size_t* workspaceBytes);

/* Changes the stream used by the handle. */
hipcompStatus_t hipcompSetStream(hipcompHandle_t handle, cudaStream_t streamId);

/* Gets the current stream assigned to the handle. */
hipcompStatus_t hipcompGetStream(hipcompHandle_t handle, cudaStream_t* streamId);

/* Sets the output length of a particular node. This is helpful when the node is
 * the output node of a RLE layer in a multi-GPU system. With this method, the
 * cudaStreamSynchronize() in hipcompDecompressLaunch() can be eliminated which
 * preserves the concurrency. */
hipcompStatus_t
hipcompSetNodeLength(hipcompHandle_t handle, int nodeId, size_t output_length);

/* Usage.  Submits a decompression task to the GPU asynchronously and returns
 * the task ID. In practice this would pipeline memory copies and kernels into
 * the assigned CUDA stream(s). The function uses no additional memory and is
 * very lightweight. The main usage pattern is to subdivide your data into
 * relatively large chunks and submit one decompression task per chunk. We must
 * support non-pinned data and this might require implementing a staging
 * pipeline inside the library. We do not want to expose CUDA streams or events
 * to the user externally since we might be able to do some efficient
 * scheduling internally by querying the status of streams or using some other
 * custom heuristics. However, the user is free to record events before and
 * after the decompress launch function to create more complex dependencies.
 *
 * Input.  The input headers and data arrays are processed according to the
 * compression DAG specified by the handle's config. Each inputHdrs/inputData
 * pair represent some input data such as RLE values, RLE runs, dictionary
 * values, or any other input depending on the configuration plan.
 * The corresponding inputData buffer should contain length
 * number of elements of type valType. If valPacking=1 then
 * hipcompPackedHeader_t<valType> should be used. The corresponding inputData
 * buffer should contain length number of elements of numBits bits each, the
 * values will be treated as unsigned offsets to the shared minValue of type
 * valType.  Both inputHdrs and inputData must be accessible by the GPU, and
 * hostHdrs is a copy of inputHdrs that is accessible to the CPU.
 * numOutputElements The number of elements that will be output.
 *
 * Output.  Decompressed data will be stored in outputData and the final number
 * of uncompressed elements will be written to outputSize. Note that the type
 * of output values is specified in 'config'. */
hipcompStatus_t hipcompDecompressLaunch(
    hipcompHandle_t handle,
    size_t numUncompressedElements,
    void* outputData,
    size_t outputSize,
    const void** inputData,
    const void** hostHdrs);

/* Releases all memory associated with the decompression handle. */
hipcompStatus_t hipcompDestroyHandle(hipcompHandle_t handle);

inline hipcompType_t selectRunsType(const size_t length)
{
  if (length <= std::numeric_limits<uint8_t>::max()) {
    return HIPCOMP_TYPE_UCHAR;
  } else if (length <= std::numeric_limits<uint16_t>::max()) {
    return HIPCOMP_TYPE_USHORT;
  } else if (length <= std::numeric_limits<uint32_t>::max()) {
    return HIPCOMP_TYPE_UINT;
  } else {
    return HIPCOMP_TYPE_ULONGLONG;
  }
}

} // namespace highlevel
} // namespace hipcomp