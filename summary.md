# hipCOMP-core — Build Summary

## How to build

A devcontainer is provided in `.devcontainer/`. It uses the ROCm 6.4 Ubuntu 22.04 image and mounts the repo at `/workspace`.

### Prerequisites

1. Copy the pre-built gdeflate install tree into the repo root:

   ```bash
   cp -r /path/to/libgdeflate/build/install/ hipCOMP-core/gdeflate
   ```

   The tree must have the shape:
   ```
   gdeflate/
     include/gdeflate.h
     lib/libgdeflate.so -> libgdeflate.so.1
     lib/libgdeflate.so.1 -> libgdeflate.so.1.0.0
     lib/libgdeflate.so.1.0.0
     lib/cmake/gdeflate/…
   ```

2. Start the devcontainer (VS Code *Reopen in Container*, or manually):

   ```bash
   docker compose -f .devcontainer/docker-compose.yml up -d
   ```

### Build command (inside the container)

```bash
bash /workspace/.devcontainer/build.sh
```

This runs:

```bash
mkdir -p /workspace/build && cd /workspace/build

cmake /workspace \
  -DCMAKE_PREFIX_PATH=/opt/rocm \
  -Dgdeflate_ROOT=/workspace/gdeflate

make -j$(nproc) install

# Install gdeflate runtime library so the system linker finds libhipcomp.so
cp /workspace/gdeflate/lib/libgdeflate.so.1.0.0 /usr/local/lib/
ln -sf /usr/local/lib/libgdeflate.so.1.0.0 /usr/local/lib/libgdeflate.so.1
ln -sf /usr/local/lib/libgdeflate.so.1       /usr/local/lib/libgdeflate.so
ldconfig
```

`make install` places the library and headers under `/usr/local` by default.

---

## Source changes required to compile

### 1. `src/lowlevel/gdeflateKernels.h` — wrong HIP include

```diff
-#include <hip.h>
+#include <hip/hip_runtime.h>
```

`<hip.h>` does not exist in the ROCm include tree; the correct header is `<hip/hip_runtime.h>`.

---

### 2. `src/hipcomp_common_deps/hlif_shared.cuh` — `tiled_partition<64>` fails on RDNA2

The HLIF framework dispatches on warp size at runtime (32 or 64). It instantiates kernel templates for **both** sizes. On RDNA2 / gfx1030, the hardware wavefront is 32, so the ROCm cooperative-groups header rejects `tiled_partition<64>` with a static assertion at compile time.

Fix: wrap each of the three `tiled_partition<warpsize>` call sites with a compile-time guard using `__AMDGCN_WAVEFRONT_SIZE__` (defined by the AMD device compiler) and `if constexpr`, so the 64-wide instantiation is discarded for architectures whose native wavefront is 32:

```cpp
// Before (three identical sites):
} else {
    HlifCompressBatch<chunks_per_block>(
        ..., cg::tiled_partition<warpsize>(cta_group));
}

// After:
} else {
#if defined(__AMDGCN_WAVEFRONT_SIZE__)
    if constexpr (warpsize <= __AMDGCN_WAVEFRONT_SIZE__) {
        HlifCompressBatch<chunks_per_block>(
            ..., cg::tiled_partition<warpsize>(cta_group));
    }
#else
    HlifCompressBatch<chunks_per_block>(
        ..., cg::tiled_partition<warpsize>(cta_group));
#endif
}
```

The same pattern was applied to both `HlifCompressBatchKernel` overloads and the `HlifDecompressBatch` helper. At runtime on gfx1030, `hipDeviceGetAttribute` returns WS=32, so the host always selects the 32-wide kernel path; the 64-wide kernel is compiled but never launched.

---

### 3. `src/highlevel/GdeflateHlifKernels.h` and `GdeflateHlifKernels.cu` — missing files

The high-level gdeflate manager (`GdeflateBatchManager`) references `gdeflate::hlif::*` functions from `GdeflateHlifKernels.h`, but neither the header nor its implementation existed in the repo.

**`GdeflateHlifKernels.h`** declares four functions in the `gdeflate::hlif` namespace:

```cpp
uint32_t batchedGdeflateCompMaxBlockOccupancy(int device_id);
uint32_t batchedGdeflateDecompMaxBlockOccupancy(int device_id);
void gdeflateHlifBatchCompress(const CompressArgs &, uint32_t max_ctas, hipStream_t);
void gdeflateHlifBatchDecompress(const uint8_t *comp_data_buffer,
                                 uint8_t *decomp_buffer, size_t uncomp_chunk_size,
                                 uint32_t *ix_chunk, size_t num_chunks,
                                 const size_t *comp_chunk_offsets,
                                 const size_t *comp_chunk_sizes,
                                 uint32_t max_ctas, hipStream_t,
                                 hipcompStatus_t *output_status,
                                 size_t total_decomp_size);
```

**`GdeflateHlifKernels.cu`** implements them by bridging the existing low-level `hipcompBatchedGdeflate*Async` API (which calls through to `libgdeflate`):

- Small GPU helper kernels build the pointer arrays that `libgdeflate` expects from the flat buffer + offset layout used by the HLIF framework.
- Compression writes each chunk directly into `comp_buffer` at stride `max_comp_chunk_size`, which exactly fills the space pre-allocated by `calculate_max_compressed_output_size`. The common header is filled by a single-thread kernel.
- Decompression builds pointer arrays from `comp_chunk_offsets` and calls `gdeflate::decompressAsync`, then converts the gdeflate status codes to `hipcompStatus_t` via the existing `convertGdeflateOutputStatuses` kernel.
- Block occupancy is estimated as `2 × SM count`.

---

### 4. `GdeflateHlifKernels.cu` — wrong chunk sizes in compression and decompression

Two related bugs were found and fixed after running `test_gdeflate` and `test_gdeflate_batch_c_api`.

#### 4a. Compression: last chunk always sent as full size

`gdeflateHlifBatchCompress` was passing `uncomp_chunk_size` as the actual input size for every chunk, including the last one which may be smaller. This caused the compressor to read past the real data.

Fix: added `gdeflate_fillActualSizes`, a GPU kernel that fills each slot with `min(chunk_size, total_size - i * chunk_size)`, and used it instead of the constant-fill kernel:

```cpp
gdeflate_fillActualSizes<<<blocks, threads, 0, stream>>>(
    d_in_sizes, uncomp_chunk_size, compress_args.decomp_buffer_size, n);
```

#### 4b. Decompression: `getDecompressSizeAsync` returns 0 on gfx1030 — fixed in libgdeflate

`gdeflateHlifBatchDecompress` calls `gdeflate::getDecompressSizeAsync` to read per-chunk uncompressed sizes from the compressed-stream headers. On gfx1030 (RDNA2), that GPU kernel was silently returning 0 for every chunk, so `decompressAsync` wrote 0 bytes — producing an all-zeros result.

Root cause (fixed in `libgdeflate`): `GDeflateTileStream` struct methods were missing `__host__ __device__` qualifiers. Calling host-only methods from device code is undefined behaviour; the AMD compiler silently emitted incorrect code, reading `numTiles = 0` and thus returning `decomp_size = 0`.

Fix: added `__host__ __device__` to all `GDeflateTileStream` methods in `libgdeflate/src/gdeflate_internal.h` (libgdeflate commit `f9a2477`). No changes to hipCOMP-core required for this bug.

---

### 5. `src/highlevel/BatchManager.hpp` — compressed output size underestimated

`calculate_max_compressed_output_size` stored per-chunk compressed sizes as `uint32_t` but the on-disk layout uses `size_t`. The allocated buffer was too small for large chunk counts, causing silent memory corruption.

Fix: changed `sizeof(uint32_t)` → `sizeof(size_t)` for `chunk_sizes_size`, and added a conservative `alignment_padding = alignof(size_t) - 1` to account for the `roundUpToAlignment` padding added before the metadata arrays:

```cpp
const size_t chunk_sizes_size = sizeof(size_t) * comp_config.num_chunks;
const size_t alignment_padding = alignof(size_t) - 1;

return sizeof(CommonHeader) + sizeof(FormatSpecHeader) +
       alignment_padding + chunk_offsets_size + chunk_sizes_size +
       checksum_size + comp_buffer_size;
```

---

### 6. `libgdeflate` — GPU aperture violation for non-4-byte-aligned output pointers

`gdeflate_decompress_kernel` uses `atomicOr` to write output bytes, which requires 4-byte aligned target addresses. `StoreByte` and `ReadOutputByte` computed alignment relative to the `out` pointer offset (`off & 3`), implicitly assuming `out` itself is 4-byte aligned. When the HLIF decompressor passes `decomp_buffer + i * uncomp_chunk_size` and `uncomp_chunk_size % 4 != 0` (e.g., 32769), pointers for chunk i ≥ 1 are misaligned, triggering `HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION`. The tile-zeroing loop had the same problem using raw 32-bit stores to a potentially misaligned address.

LZ4 and all other formats in this repo use non-4-aligned chunk strides without any special handling — the alignment requirement was a gdeflate-specific bug.

Fix is in **`libgdeflate/src/gdeflate_decompress.hip`** (no hipCOMP-core changes needed):

```cpp
// Before (assumes out is 4-byte aligned):
uint32_t mod4 = off & 3u;
atomicOr((uint32_t*)(out + off - mod4), ...);

// After (absolute address alignment):
uintptr_t addr = (uintptr_t)(out + off);
uint32_t  mod4 = (uint32_t)(addr & 3u);
atomicOr((uint32_t*)(addr - mod4), ...);
```

The zeroing loop is changed from word-granular (`*(uint32_t*)... = 0`) to byte-granular (`output[i] = 0`) so it is correct for any pointer alignment.

After this fix all 7 `test_gdeflate` cases and all 6 `test_gdeflate_batch_c_api` cases pass on gfx1030.

---

## Git workflow

The source changes are committed on branch `fix/rocm-gfx1030-build` in a public fork at `https://github.com/ArsArmandi/hipCOMP-core`, with `ROCm/hipCOMP-core` set as the `upstream` remote.

### Remotes

```
origin    https://github.com/ArsArmandi/hipCOMP-core  (fork)
upstream  https://github.com/ROCm/hipCOMP-core         (original)
```

### Rebasing when upstream updates

```bash
git fetch upstream
git rebase upstream/release/rocmds-25.10
git push --force-with-lease origin fix/rocm-gfx1030-build
```

### Opening a PR to upstream

On GitHub: `ArsArmandi/hipCOMP-core` → **Contribute → Open pull request** → base: `ROCm/hipCOMP-core:release/rocmds-25.10`.

### Files NOT included in the PR branch

`.devcontainer/`, `gdeflate/`, and `summary.md` are personal build infrastructure and are not committed to the fixes branch.
