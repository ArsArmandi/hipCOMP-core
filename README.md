<!-- MIT License
  --
  -- Copyright (c) 2023 Advanced Micro Devices, Inc.
  --
  -- Permission is hereby granted, free of charge, to any person obtaining a copy
  -- of this software and associated documentation files (the "Software"), to deal
  -- in the Software without restriction, including without limitation the rights
  -- to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
  -- copies of the Software, and to permit persons to whom the Software is
  -- furnished to do so, subject to the following conditions:
  --
  -- The above copyright notice and this permission notice shall be included in all
  -- copies or substantial portions of the Software.
  --
  -- THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
  -- IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
  -- FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
  -- AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
  -- LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
  -- OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
  -- SOFTWARE.
  -->

# hipCOMP-CORE

hipCOMP CORE is a library for fast lossless compression/decompression on the GPU which contains the algorithm implementation.

The code is based on ``nvCOMP`` release branch [``branch-2.1``](https://github.com/NVIDIA/nvcomp/tree/branch-2.1).

## Build From Source

### HIP/AMD

> **NOTE:** If you experience compiler errors related to ``cooperative_groups`` with ROCm versions ``<=6.0.X``, additionally specify the ``CMake`` build option `-D CG_WORKAROUND=1`.

```bash
cd hipcomp-core/
mkdir build/
cd build/
CMAKE_PREFIX_PATH=/opt/rocm/lib/cmake cmake ../
# To build with tests, append `-D BUILD_TESTS=1`:
# CMAKE_PREFIX_PATH=/opt/rocm/lib/cmake cmake ../ -D BUILD_TESTS=1
make
```

### HIP/CUDA

Like HIP/AMD but with additional `-D CUDA_BACKEND=1` option:

```bash
cd hipcomp-core/
mkdir build/
cd build/
CMAKE_PREFIX_PATH=/opt/rocm/lib/cmake cmake ../ -D CUDA_BACKEND=1
# To build with tests, append `-D BUILD_TESTS=1`:
# CMAKE_PREFIX_PATH=/opt/rocm/lib/cmake cmake ../ -D BUILD_TESTS=1 -D CUDA_BACKEND=1
make
```

#### Debugging

To create debug builds append the following option:

```bash
-D CMAKE_BUILD_TYPE=Debug
```

### Run tests

After completing the build, run:

```bash
cd hipcomp-core/
cd build/
make test
```

Tips:

* Select a particular GPU by setting the environment variable `HIP_VISIBLE_DEVICES=<id>` (or `CUDA_VISIBLE_DEVICES=<id>` with CUDA backend) before running ``make test``.

<!-- REMOVE BELOW BEFORE RELEASE -->

> **NOTE: REMOVE BELOW BEFORE RELEASE!**

#### Legal Requirements:
Always include the appropriate copyright and MIT X11 notice (see below)
* at the top of the AMD developed files, and
* in a LICENSE text file in the top level directory.

#### Standard ongoing code readiness and release obligations:
* Follow the Developer Guidelines here: http://confluence.amd.com/pages/viewpage.action?pageId=52793191
