# hipCOMP Licensing

LICENSE FILE:

* [LICENSE](LICENSE)

## Files Subject to the BSD 3-Clause License

This project contains work derived from NVIDIA nvCOMP v2.1 that is licensed under the 3-Clause BSD License:

* LICENSE FILE: [NVCOMP_2_2_LICENSE](NVCOMP_2_2_LICENSE)
* HOMEPAGE: https://github.com/NVIDIA/nvcomp/tree/branch-2.1
* FILES:
  * CMakeLists.txt
  * scripts/build_dev_debug.sh
  * scripts/build_dev_release.sh
  * src/CMakeLists.txt
  * src/RunLengthEncodeGPU.hip.cpp
  * src/RunLengthEncodeGPU.h
  * src/SnappyBatch.cpp
  * src/SnappyKernels.h
  * src/TempSpaceBroker.cpp
  * src/TempSpaceBroker.h
  * src/highlevel/CascadedSelector.cpp
  * src/highlevel/CascadedSelector.h
  * src/highlevel/CascadedSelectorKernels.hip.cpp
  * src/highlevel/CascadedSelectorKernels.h
  * src/highlevel/HighLevelLZ4API.cpp
  * src/highlevel/LZ4Compressor.hip.cpp
  * src/highlevel/LZ4Compressor.h
  * src/highlevel/LZ4Decompressor.hip.cpp
  * src/highlevel/LZ4Decompressor.h
  * src/highlevel/LZ4Metadata.cpp
  * src/highlevel/LZ4Metadata.h
  * src/highlevel/LZ4MetadataOnGPU.hip.cpp
  * src/highlevel/LZ4MetadataOnGPU.h
  * src/highlevel/Metadata.cpp
  * src/highlevel/Metadata.h
  * src/highlevel/MutableLZ4MetadataOnGPU.cpp
  * src/highlevel/MutableLZ4MetadataOnGPU.h
  * src/highlevel/test/CMakeLists.txt
  * src/highlevel/test/CascadedAuto_test.cpp
  * src/highlevel/test/CascadedMetadata_test.cpp
  * src/highlevel/test/CascadedSelector_test.cpp
  * src/highlevel/test/DecompressHelpers_test.cpp
  * src/lowlevel/BitcompBatch.hip.cpp
  * src/lowlevel/CascadedBatch.hip.cpp
  * src/lowlevel/LZ4Batch.cpp
  * src/lowlevel/LZ4CompressionKernels.hip.cpp
  * src/lowlevel/LZ4CompressionKernels.h
  * src/lowlevel/gdeflateBatch.cpp
  * src/lowlevel/gdeflateKernels.hip.cpp
  * src/lowlevel/gdeflateKernels.h
  * src/lowlevel/test/CMakeLists.txt
  * src/nvcomp_api.cpp
  * src/nvcomp_cub.hip.cpph
  * src/test/BitPackGPU_test.cpp
  * src/test/CMakeLists.txt
  * src/test/HipUtils_test.cpp
  * src/test/DeltaGPU_test.cpp
  * src/test/RunLengthEncodeGPU_test.cpp
  * src/test/SnappyLargeTokens_test.cpp
  * src/test/TempSpaceBroker_test.cpp
  * src/type_macros.h
  * src/unpack.h
  * tests/CMakeLists.txt
  * tests/test_batch_c_api.h
  * tests/test_bitcomp.cpp
  * tests/test_bitcomp_batch.cpp
  * tests/test_bitcomp_batch_c_api.c
  * tests/test_cascaded_batch.cpp
  * tests/test_cascaded_c_api.c
  * tests/test_cascaded_cpp_api.cpp
  * tests/test_cascaded_selector.cpp
  * tests/test_cascadedbatch_c_api.c
  * tests/test_common.h
  * tests/test_decompress_c_api.c
  * tests/test_gdeflate_batch_c_api.c
  * tests/test_lz4.cpp
  * tests/test_lz4batch_c_api.c
  * tests/test_random.cpp
  * tests/test_random_auto.cpp
  * tests/test_random_lz4.cpp
  * tests/test_snappy_batch_c_api.c

## Files Subject to the Apache 2.0 License

This project contains work derived from NVIDIA nvCOMP v2.1 that is licensed under the Apache 2.0 license:

* LICENSE TEXT:

  ```
  Copyright (c) 2019, NVIDIA CORPORATION.
  Licensed under the Apache License, Version 2.0 (the "License");
  you may not use this file except in compliance with the License.
  You may obtain a copy of the License at
      http://www.apache.org/licenses/LICENSE-2.0
  Unless required by applicable law or agreed to in writing, software
  distributed under the License is distributed on an "AS IS" BASIS,
  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  See the License for the specific language governing permissions and
  limitations under the License.
  ```

* HOMEPAGE: https://github.com/NVIDIA/nvcomp/tree/branch-2.1

* FILES:
  * cmake/nvcomp-config.cmake.in
  * src/SnappyBlockUtils.hip.cpph
  * src/SnappyKernels.hip.cpp

## Files Subject to the Boost Software Lincese, Version 1.0

This project contains work derived that is licensed under the Boost Software License, Version 1.0, license:

* LICENSE TEXT:

  ```
  Copyright (c) 2018 Two Blue Cubes Ltd. All rights reserved.
 
  Distributed under the Boost Software License, Version 1.0. (See accompanying
  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
  ```

* FILES:
  
  * tests/catch.hpp

## Other Files

All other files are subject to this project's [LICENSE](LICENSE).