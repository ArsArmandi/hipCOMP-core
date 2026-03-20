/*
 * libgdeflate — HIP GPU GDeflate compress/decompress kernels
 * Apache 2.0 — see LICENSE
 *
 * Public API matching what hipCOMP-core expects via find_package(gdeflate).
 */
#pragma once

#include <cstddef>
#include <hip/hip_runtime.h>

namespace gdeflate {

// ---- Enumerations -------------------------------------------------------

enum gdeflate_compression_algo {
    HIGH_THROUGHPUT  = 0,  ///< Speed-optimised (LZ77 + Huffman)
    HIGH_COMPRESSION = 1,  ///< Ratio-optimised  (LZ77 + Huffman, deeper search)
    ENTROPY_ONLY     = 2,  ///< No LZ77, Huffman entropy coding only
};

/// Per-chunk decompression status written to device memory by the GPU kernel.
/// sizeof(gdeflateStatus_t) MUST equal sizeof(int) == 4 (checked by hipCOMP-core).
enum gdeflateStatus_t : int {
    gdeflateSuccess = 0,
    gdeflateError   = 1,
};
static_assert(sizeof(gdeflateStatus_t) == sizeof(int),
              "gdeflateStatus_t must be 4 bytes");

// ---- Decompression API --------------------------------------------------

/**
 * Query the scratch-buffer size needed by decompressAsync.
 *
 * @param num_chunks            Number of compressed chunks in the batch.
 * @param max_uncomp_chunk_size Maximum uncompressed size of any single chunk.
 * @param temp_bytes            [out] Required scratch buffer size in bytes.
 */
void decompressGetTempSize(size_t num_chunks,
                           size_t max_uncomp_chunk_size,
                           size_t* temp_bytes);

/**
 * Read uncompressed sizes from compressed GDeflate stream headers (async).
 *
 * Launches a GPU kernel that reads each TileStream header and writes the
 * uncompressed size to d_uncomp_bytes.
 */
void getDecompressSizeAsync(const void* const*  d_comp_ptrs,
                            const size_t*       d_comp_bytes,
                            size_t*             d_uncomp_bytes,
                            size_t              batch_size,
                            hipStream_t         stream);

/**
 * Decompress a batch of GDeflate-compressed chunks on the GPU (async).
 *
 * @param d_comp_ptrs           Device array of pointers to compressed data.
 * @param d_comp_bytes          Device array of compressed sizes.
 * @param d_uncomp_bytes        Device array of expected uncompressed sizes.
 * @param d_actual_uncomp_bytes Device array where actual sizes are written.
 * @param                       (unused, present for API compatibility)
 * @param batch_size            Number of chunks.
 * @param d_temp                Scratch buffer (size from decompressGetTempSize).
 * @param temp_bytes            Size of scratch buffer.
 * @param d_out_ptrs            Device array of output buffer pointers.
 * @param d_statuses            Device array of per-chunk status codes.
 * @param stream                HIP stream to enqueue work on.
 */
void decompressAsync(const void* const*   d_comp_ptrs,
                     const size_t*        d_comp_bytes,
                     const size_t*        d_uncomp_bytes,
                     size_t*              d_actual_uncomp_bytes,
                     size_t               /*unused*/,
                     size_t               batch_size,
                     void*                d_temp,
                     size_t               temp_bytes,
                     void* const*         d_out_ptrs,
                     gdeflateStatus_t*    d_statuses,
                     hipStream_t          stream);

// ---- Compression API ----------------------------------------------------

/**
 * Query the scratch-buffer size needed by compressAsync.
 */
void compressGetTempSize(size_t                    batch_size,
                         size_t                    max_chunk_size,
                         size_t*                   temp_bytes,
                         gdeflate_compression_algo algo);

/**
 * Query the maximum compressed output size for a single chunk.
 */
void compressGetMaxOutputChunkSize(size_t  max_chunk_size,
                                   size_t* max_out_bytes);

/**
 * Compress a batch of chunks to GDeflate format.
 *
 * Compression is performed CPU-side (D→H copy, compress, H→D copy) because
 * a GPU GDeflate encoder requires specialised hardware support that is not yet
 * available.  The function blocks the calling CPU thread until all chunks are
 * compressed and the results are written back to device memory.
 */
void compressAsync(const void* const*        d_in_ptrs,
                   const size_t*             d_in_bytes,
                   size_t                    max_chunk_size,
                   size_t                    batch_size,
                   void*                     d_temp,
                   size_t                    temp_bytes,
                   void* const*              d_out_ptrs,
                   size_t*                   d_out_bytes,
                   gdeflate_compression_algo algo,
                   hipStream_t               stream);

} // namespace gdeflate
