// Copyright (c) 2023 Advanced Micro Devices, Inc.
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

#ifndef SRC_SNAPPY_GPU_DEFS_H
#define SRC_SNAPPY_GPU_DEFS_H

/** 
 * This file's definition are influenced by the following preprocessor controls 
 * 
 * HIP:
 * * __HIP_PLATFORM_HCC__ / __HIP_PLATFORM_AMD__: Set by CMake for lang HIP. If defined, it means we are building for AMD GPUs.
 * 
 * HIP/AMD:
 *   * SNAPPY_COMPRESS_GROUPSIZE64: If defined, use a group size (or, logical warpsize) of 64 for Snappy's compressor stage. 
 *     Otherwise use a group size of 32, i.e. the upper 32 bits are masked out.
 *   * SNAPPY_PREFETCH_GROUPSIZE64: If defined, use a group size (or, logical warpsize) of 64 for Snappy's compressed byte prefetcher stage (decompression). 
 *     Otherwise use a group size of 32, i.e. the upper 32 bits are masked out.
 *   * SNAPPY_DECODE_GROUPSIZE64: If defined, use a group size (or, logical warpsize) of 64 for Snappy's LZ77 symbol decoder stage (decompression). 
 *     Otherwise use a group size of 32, i.e. the upper 32 bits are masked out.
 *   * SNAPPY_PROCESS_GROUPSIZE64: If defined, use a group size (or, logical warpsize) of 64 for Snappy's LZ77 symbol process stage (decompression). 
 *     Otherwise use a group size of 32, i.e. the upper 32 bits are masked out.
 * 
 * HIP/NVIDIA:
 *   * __CUDA_ARCH__: If >= 700, __nanosleep is used instead of clock()
 *   * __CUDACC_VER_MAJOR__: Set by nvcc. If >= 9, warp-level communication builtins with _sync suffix are used.
*/

#if (__CUDACC_VER_MAJOR__ >= 9)
#  define INDEPENDENT_THREAD_SCHEDULING
#endif
#ifdef INDEPENDENT_THREAD_SCHEDULING
template <typename T> 
__device__ inline T SHFL10(T v) {
  return __shfl_sync(~0, v, 0);
}
//#  define SHFL10(v)        __shfl_sync(~0, (v), 0)
#  define SHFL1(v, t)      __shfl_sync(~0, (v), (t))
#  define SHFL1_XOR(v, m)  __shfl_xor_sync(~0, (v), (m))
#  define SYNCWARP()      __syncwarp()
#  define BALLOT1(v)       __ballot_sync(~0, (v))
#else
template <typename T> 
__device__ inline T SHFL10(T v) {
  return __shfl(v, 0);
}
// #  define SHFL10(v)        __shfl((v), 0)
#  define SHFL1(v, t)      __shfl((v), (t))
#  define SHFL1_XOR(v, m)  __shfl_xor((v), (m))
#  define SYNCWARP()
#  define BALLOT1(v)       __ballot((v))
#endif

#if (__CUDA_ARCH__ >= 700)
#  define NANOSLEEP(d)  __nanosleep((d))
#else
//: includes the __HIP_PLATFORM_AMD__ case
#  define NANOSLEEP(d)  clock()
#endif

// Snappy GPU types
#if defined(__HIP_PLATFORM_HCC__) || defined(__HIP_PLATFORM_AMD__)
constexpr int32_t warpsize = 64;
constexpr uint32_t uwarpsize = 64u;
typedef uint64_t warp_mask_t;
typedef int64_t signed_warp_mask_t;


#define CLZ(x) __clzll(x)
//#define FFS(x) __ffsll(static_cast<unsigned long long int>(x))
#define POPC(x) __popcll(x) 
__device__ inline int FFS(long long int x) {
  return __ffsll(static_cast<unsigned long long int>(x));
}

#define PREFETCH_SLEEP_NS 1600
#define DECODE_SLEEP_NS 50
#define PROCESS_SLEEP_NS 100
#else
constexpr int32_t warpsize = 32;
constexpr uint32_t uwarpsize = 32u;
typedef uint32_t warp_mask_t;
typedef int32_t signed_warp_mask_t;

#define CLZ(x) __clz(x)
// #define FFS(x) __ffs(x)
__device__ inline int FFS(int x) {
  return __ffs(x);
}
#define POPC(x) __popc(x)

#define PREFETCH_SLEEP_NS 1600
#define DECODE_SLEEP_NS 50
#define PROCESS_SLEEP_NS 100
#endif

//: Quantities for bit shift operations
constexpr warp_mask_t LANE_MASK_ZERO = 0;
constexpr warp_mask_t LANE_MASK_ONE  = 1;
constexpr warp_mask_t LANE_MASK_TWO  = 2;

namespace hipcomp {
  namespace snappy {
    /**
     * Ballot (warp vote function) with full mask (all bits are "1", i.e. all lanes participate) but a number of higher bits
     * might be cutoff depending on teh RET_MASK_TYPE template parameter.
     * 
     * \param[in] t the warp lane id, i.e. threadIdx.x % WARPSIZE
     * \param[in] predicate the predicate to use for active lanes. Active lanes fit into the RET_MASK_TYPE.
     * \param[in] predicate_excluded the predicate to use for excluded lanes. Excluded lanes do not fit into RET_MASK_TYPE,
     *            i.e. their respective contribution is cutoff. Currently, unused by the specializations but optimizations might
     *            require to decouple the grou size/logical warp size from the returned mask type (RET_MASK_TYPE).
     * \note All of a warp's threads must be participate in this operation on AMD architectures as well as 
     *       NVIDIA Pascal and older NVIDIA architectures.
     */
    template <typename RET_MASK_TYPE,unsigned WARPSIZE>
    __device__ inline RET_MASK_TYPE ballot1(uint32_t t, int predicate,int predicate_excluded = 0);

    // ballot1 specializations
    #if defined(__HIP_PLATFORM_HCC__) || defined(__HIP_PLATFORM_AMD__)
    template <>
    __device__ inline uint64_t ballot1<uint64_t,64u>(uint32_t t,int predicate,int predicate_excluded) {
      return BALLOT1(predicate);
    }

    template <>
    __device__ inline uint32_t ballot1<uint32_t,64u>(uint32_t t,int predicate,int predicate_excluded) {
      uint64_t mask_64 = BALLOT1(predicate);
      return *reinterpret_cast<uint32_t*>(&mask_64);
    }
    #else
    template <>
    __device__ inline uint32_t ballot1<uint32_t,32u>(uint32_t t,int predicate,int predicate_excluded) {
      return BALLOT1(predicate);
    }
    #endif

    template <unsigned GROUPSIZE,unsigned WARPSIZE>
    class WarpReduce {
    public:
      template <typename T>
      static __device__ inline T prefix_sum(uint32_t t, T thread_value);
      template <typename T>
      static __device__ inline T sum(uint32_t t, T thread_value);
    };

    // Warp reduction helpers
    #if defined(__HIP_PLATFORM_HCC__) || defined(__HIP_PLATFORM_AMD__)
    template <typename T> inline __device__ T WarpReduceSum2(T acc)     { return acc + SHFL1_XOR(acc, 1); }
    template <typename T> inline __device__ T WarpReduceSum4(T acc)     { acc = WarpReduceSum2(acc); return acc + SHFL1_XOR(acc, 2); }
    template <typename T> inline __device__ T WarpReduceSum8(T acc)     { acc = WarpReduceSum4(acc); return acc + SHFL1_XOR(acc, 4); }
    template <typename T> inline __device__ T WarpReduceSum16(T acc)    { acc = WarpReduceSum8(acc); return acc + SHFL1_XOR(acc, 8); }
    template <typename T> inline __device__ T WarpReduceSum32(T acc)    { acc = WarpReduceSum16(acc); return acc + SHFL1_XOR(acc, 16); }
    template <typename T> inline __device__ T WarpReduceSum64(T acc)    { acc = WarpReduceSum32(acc); return acc + SHFL1_XOR(acc, 32); }

    template <typename T> inline __device__ T WarpReducePos2(T pos, uint32_t t) { T tmp = SHFL1(pos, t & 0x3e); pos += (t & 1) ? tmp : 0; return pos; }
    template <typename T> inline __device__ T WarpReducePos4(T pos, uint32_t t) { T tmp; pos = WarpReducePos2(pos, t); tmp = SHFL1(pos, (t & 0x3c) | 1); pos += (t & 2) ? tmp : 0; return pos; }
    template <typename T> inline __device__ T WarpReducePos8(T pos, uint32_t t) { T tmp; pos = WarpReducePos4(pos, t); tmp = SHFL1(pos, (t & 0x38) | 3); pos += (t & 4) ? tmp : 0; return pos; }
    template <typename T> inline __device__ T WarpReducePos16(T pos, uint32_t t) { T tmp; pos = WarpReducePos8(pos, t); tmp = SHFL1(pos, (t & 0x30) | 7); pos += (t & 8) ? tmp : 0; return pos; }
    template <typename T> inline __device__ T WarpReducePos32(T pos, uint32_t t) { T tmp; pos = WarpReducePos16(pos, t); tmp = SHFL1(pos, (t & 0x20) | 15); pos += (t & 16) ? tmp : 0; return pos; }
    template <typename T> inline __device__ T WarpReducePos64(T pos, uint32_t t) { T tmp; pos = WarpReducePos32(pos, t); tmp = SHFL1(pos, 0x1f); pos += (t & 32) ? tmp : 0; return pos; }
    //#define WarpReduceSum(pos) WarpReduceSum64(pos)
    //#define WarpReducePos(pos,t) WarpReducePos64(pos,t)

    template <>
    class WarpReduce<32u,64u> {
    public:
      template <typename T>
      static __device__ inline T prefix_sum(uint32_t t, T thread_value) {
        return WarpReducePos32(thread_value, t);
      }
      template <typename T>
      static __device__ inline T sum(uint32_t t, T thread_value) {
        return WarpReduceSum32(thread_value);
      }
    };

    template <>
    class WarpReduce<64u,64u> {
    public:
      template <typename T>
      static __device__ inline T prefix_sum(uint32_t t, T thread_value) {
        return WarpReducePos64(thread_value, t);
      }
      template <typename T>
      static __device__ inline T sum(uint32_t t, T thread_value) {
        return WarpReduceSum64(thread_value);
      }
    };
    #else
    template <typename T> inline __device__ T WarpReduceSum2(T acc)     { return acc + SHFL1_XOR(acc, 1); }
    template <typename T> inline __device__ T WarpReduceSum4(T acc)     { acc = WarpReduceSum2(acc); return acc + SHFL1_XOR(acc, 2); }
    template <typename T> inline __device__ T WarpReduceSum8(T acc)     { acc = WarpReduceSum4(acc); return acc + SHFL1_XOR(acc, 4); }
    template <typename T> inline __device__ T WarpReduceSum16(T acc)    { acc = WarpReduceSum8(acc); return acc + SHFL1_XOR(acc, 8); }
    template <typename T> inline __device__ T WarpReduceSum32(T acc)    { acc = WarpReduceSum16(acc); return acc + SHFL1_XOR(acc, 16); }

    template <typename T> inline __device__ T WarpReducePos2(T pos, uint32_t t) { T tmp = SHFL1(pos, t & 0x1e); pos += (t & 1) ? tmp : 0; return pos; }
    template <typename T> inline __device__ T WarpReducePos4(T pos, uint32_t t) { T tmp; pos = WarpReducePos2(pos, t); tmp = SHFL1(pos, (t & 0x1c) | 1); pos += (t & 2) ? tmp : 0; return pos; }
    template <typename T> inline __device__ T WarpReducePos8(T pos, uint32_t t) { T tmp; pos = WarpReducePos4(pos, t); tmp = SHFL1(pos, (t & 0x18) | 3); pos += (t & 4) ? tmp : 0; return pos; }
    template <typename T> inline __device__ T WarpReducePos16(T pos, uint32_t t) { T tmp; pos = WarpReducePos8(pos, t); tmp = SHFL1(pos, (t & 0x10) | 7); pos += (t & 8) ? tmp : 0; return pos; }
    template <typename T> inline __device__ T WarpReducePos32(T pos, uint32_t t) { T tmp; pos = WarpReducePos16(pos, t); tmp = SHFL1(pos, 0xf); pos += (t & 16) ? tmp : 0; return pos; }

    template <>
    class WarpReduce<32u,32u> {
    public:
      template <typename T>
      static __device__ inline T prefix_sum(uint32_t t, T thread_value) {
        return WarpReducePos32(thread_value, t);
      }
      template <typename T>
      static __device__ inline T sum(uint32_t t, T thread_value) {
        return WarpReduceSum32(thread_value);
      }
    };
    #endif
  } // namespace snappy
} // namespace hipcomp

# endif //  SRC_SNAPPY_GPU_DEFS_H