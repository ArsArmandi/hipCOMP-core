# hipCOMP Licensing

LICENSE FILE:

* [LICENSE](LICENSE)

## Files Subject to the BSD 3-Clause License

This project contains work derived from NVIDIA nvCOMP v2.2 that is licensed under the 3-Clause BSD License:

* NVCOMP-2.2 LICENSE FILE: [NVCOMP_2_2_LICENSE](NVCOMP_2_2_LICENSE)
* HOMEPAGE: https://github.com/NVIDIA/nvcomp/tree/branch-2.2
* FILES (Copyright years may differ per file):
  * CMakeLists.txt
  * include/hipcomp/ans.h
  * include/hipcomp/ans.hpp
  * include/hipcomp/gdeflate.hpp
  * include/hipcomp/hipcompManager.hpp
  * include/hipcomp/hipcompManagerFactory.hpp
  * include/hipcomp/shared_types.h
  * include/hipcomp/snappy.hpp
  * scripts/build_dev_debug.sh
  * scripts/build_dev_release.sh
  * src/CMakeLists.txt
  * src/CascadedKernels.hiph
  * src/LZ4Kernels.hiph
  * src/LZ4Types.h
  * src/RunLengthEncodeGPU.h
  * src/RunLengthEncodeGPU.hip
  * src/SnappyBatch.cpp
  * src/SnappyKernels.h
  * src/snappy/types.h
  * src/TempSpaceBroker.cpp
  * src/TempSpaceBroker.h
  * src/highlevel/ANSManager.cpp
  * src/highlevel/ANSManager.hpp
  * src/highlevel/BatchManager.hpp
  * src/highlevel/BitcompManager.hip
  * src/highlevel/BitcompManager.hpp
  * src/highlevel/CascadedHlifKernels.h
  * src/highlevel/CascadedHlifKernels.hip
  * src/highlevel/CascadedManager.cpp
  * src/highlevel/CascadedManager.hpp
  * src/highlevel/CompressionConfigs.cpp
  * src/highlevel/CompressionConfigs.hpp
  * src/highlevel/GdeflateBatchManager.hpp
  * src/highlevel/GdeflateManager.cpp
  * src/highlevel/LZ4HlifKernels.h
  * src/highlevel/LZ4HlifKernels.hip
  * src/highlevel/LZ4Manager.cpp
  * src/highlevel/LZ4Manager.hpp
  * src/highlevel/ManagerBase.hpp
  * src/highlevel/PinnedPtrs.hpp
  * src/highlevel/SnappyHlifKernels.h
  * src/highlevel/SnappyHlifKernels.hip
  * src/highlevel/SnappyManager.cpp
  * src/highlevel/SnappyManager.hpp
  * src/highlevel/hipcompManagerFactory.cpp
  * src/highlevel/test/CMakeLists.txt
  * src/highlevel/test/PinnedPtrPool_test.cpp
  * src/hipcomp_common_deps/hlif_shared.hiph
  * src/hipcomp_common_deps/hlif_shared_types.hpp
  * src/lowlevel/BitcompBatch.hip
  * src/lowlevel/CascadedBatch.hip
  * src/lowlevel/LZ4Batch.cpp
  * src/lowlevel/LZ4CompressionKernels.h
  * src/lowlevel/LZ4CompressionKernels.hip
  * src/lowlevel/SnappyBatchKernels.hip
  * src/lowlevel/ansBatch.cpp
  * src/lowlevel/gdeflateBatch.cpp
  * src/lowlevel/gdeflateKernels.h
  * src/lowlevel/gdeflateKernels.hip
  * src/lowlevel/test/CMakeLists.txt
  * src/nvcomp_api.cpp
  * src/nvcomp_cub.hiph
  * src/test/BitPackGPU_test.cpp
  * src/test/CMakeLists.txt
  * src/test/DeltaGPU_test.cpp
  * src/test/HipUtils_test.cpp
  * src/test/RunLengthEncodeGPU_test.cpp
  * src/test/SnappyLargeTokens_test.cpp
  * src/test/TempSpaceBroker_test.cpp
  * src/type_macros.h
  * src/unpack.h
  * tests/CMakeLists.txt
  * tests/test_ans_batch_c_api.c
  * tests/test_batch_c_api.h
  * tests/test_bitcomp.cpp
  * tests/test_bitcomp_batch.cpp
  * tests/test_bitcomp_batch_c_api.c
  * tests/test_cascaded.cpp
  * tests/test_cascaded_batch.cpp
  * tests/test_cascadedbatch_c_api.c
  * tests/test_common.h
  * tests/test_gdeflate.cpp
  * tests/test_gdeflate_batch_c_api.c
  * tests/test_lz4.cpp
  * tests/test_lz4batch_c_api.c
  * tests/test_random_lz4.cpp
  * tests/test_snappy_batch_c_api.c

## Files Subject to the Apache 2.0 License

This project contains work derived from NVIDIA nvCOMP v2.2 that is licensed under the Apache 2.0 license:

* LICENSE TEXT (YEAR differs per file):

  ```
  Copyright (c) <YEAR>, NVIDIA CORPORATION.

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

* HOMEPAGE: https://github.com/NVIDIA/nvcomp/tree/branch-2.2

* FILES:
  * cmake/hipcomp-config.cmake.in
  * src/SnappyBlockUtils.hiph
  * src/SnappyKernels.hip

## Files Subject to the Boost Software Lincese, Version 1.0

This project contains work derived that is licensed under the Boost Software License, Version 1.0, license:

* LICENSE TEXT:

  ```
  Copyright (c) 2022 Two Blue Cubes Ltd. All rights reserved.
  
  Distributed under the Boost Software License, Version 1.0. (See accompanying
  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
  ```

* FILES:
  
  * tests/catch.hpp

## Other Files

All other files are subject to this project's [LICENSE](LICENSE).