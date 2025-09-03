/*
 * Copyright (c) 2019-2020, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
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

#include <string>
#include <iostream>

#include "hip/hip_runtime.h"
#include "rocm-core/rocm_version.h"

#define ROCM_VERSION (ROCM_VERSION_MAJOR * 10'000'000 + ROCM_VERSION_MINOR * 100'000 + ROCM_VERSION_PATCH)

// NOTE(HIP/AMD): No feature check for warp sync builtins available (not yet used)
#define HIPCOMP_HIP_HAS_WARP_SYNC_BUILTINS (ROCM_VERSION >= 60020000)

// This macro dynamically obtains the warp size and adjust itself for host and device
// compilation.
// Attention: when used inside of a device function it is necessary to guard this macro
// within #ifdef __HIP_DEVICE_COMPILE__ guards to avoid calling a host function from
// a device function
// NOTE: If you are on the device, you can pass any value for DeviceNum.
// NOTE: On the host, if you provide a DeviceNum < 0, the macro will try to autodetect the
//       number of the current device.
#ifdef __HIP_DEVICE_COMPILE__
#define HIPCOMP_OBTAIN_DYNAMIC_WARPSIZE(DeviceNum) int __WS = warpSize;
#else
#define HIPCOMP_OBTAIN_DYNAMIC_WARPSIZE(DeviceNum) \
    int __current_device_id = DeviceNum; \
    if ( __current_device_id < 0 && hipGetDevice(&__current_device_id) != hipSuccess ) { \
      std::cerr << "error: no AMD device found" << std::endl; \
      __builtin_trap(); \
    } \
    int __WS = -1; \
    if ( hipDeviceGetAttribute(&__WS, hipDeviceAttributeWarpSize, __current_device_id) != hipSuccess ) { \
      std::cerr << "error: could not obtain warp size" << std::endl; \
      __builtin_trap(); \
    } \
    (void)__current_device_id;
#endif

// A macro that takes arbitrary code and executes it while setting the warp size
// to the provided name as a constexpr.
// USAGE: HIPCOMP_EXECUTE_WARPSIZE_DEPENDENT_CODE(<device-num-expr>, <statements>)
// NOTE: The macro currently assumes that code is always run on the current
//       device.
#define HIPCOMP_EXECUTE_WARPSIZE_DEPENDENT_CODE(DeviceNum, ...) \
    do { \
      HIPCOMP_OBTAIN_DYNAMIC_WARPSIZE(DeviceNum); \
      if(__WS == 64){ \
        constexpr int HIPCOMP_WARPSIZE = 64; \
        __VA_ARGS__ \
      } \
      else if(__WS == 32){ \
        constexpr int HIPCOMP_WARPSIZE = 32; \
        __VA_ARGS__ \
      } \
      else{ \
        __builtin_trap(); \
      } \
    } while(0);

namespace hipcomp
{

enum CopyDirection {
  HOST_TO_DEVICE = hipMemcpyHostToDevice,
  DEVICE_TO_HOST = hipMemcpyDeviceToHost,
  DEVICE_TO_DEVICE = hipMemcpyDeviceToDevice
};

class HipUtils
{
public:
  /**
   * @brief Convert hip errors into exceptions. Will throw an exception
   * unless `err == hipSuccess`.
   *
   * @param err The error.
   * @param msg The message to attach to the exception.
   */
  static void check(const hipError_t err, const std::string& msg = "");

  static void sync(hipStream_t stream);

  static void check_last_error(const std::string& msg = "");

  /**
  * \brief Get the given device's properties.
  */
  static hipDeviceProp_t device_properties(int device_id);

  /**
   * @brief Perform checked asynchronous memcpy.
   *
   * @tparam T The data type.
   * @param dst The destination address.
   * @param src The source address.
   * @param count The number of elements to copy.
   * @param kind The direction of the copy.
   * @param stream THe stream to operate on.
   */
  template <typename T>
  static void copy_async(
      T* const dst,
      const T* const src,
      const size_t count,
      const CopyDirection kind,
      hipStream_t stream)
  {
    check(
        hipMemcpyAsync(dst, src, sizeof(T) * count,
          static_cast<hipMemcpyKind>(kind), stream),
        "HipUtils::copy_async(dst, src, count, kind, stream)");
  }

  /**
   * @brief Perform a synchronous memcpy.
   *
   * @tparam T The data type.
   * @param dst The destination address.
   * @param src The source address.
   * @param count The number of elements to copy.
   * @param kind The direction of the copy.
   */
  template <typename T>
  static void copy(
      T* const dst,
      const T* const src,
      const size_t count,
      const CopyDirection kind)
  {
    check(
        hipMemcpy(dst, src, sizeof(T) * count, static_cast<hipMemcpyKind>(kind)),
        "HipUtils::copy(dst, src, count, kind)");
  }

  static bool is_device_pointer(const void* ptr);

  template <typename T>
  static T* device_pointer(T* const ptr)
  {
    return reinterpret_cast<T*>(void_device_pointer(ptr));
  }

private:
  static const void* void_device_pointer(const void* ptr);
  static void* void_device_pointer(void* ptr);
};

} // namespace hipcomp
