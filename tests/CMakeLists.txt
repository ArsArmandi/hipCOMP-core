# Copyright (c) 2017-2020, NVIDIA CORPORATION. All rights reserved.
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Modifications Copyright (C) 2023 Advanced Micro Devices, Inc. All rights reserved.
file(GLOB EXAMPLE_SOURCES *.cpp *.c)

if(NOT gdeflate_FOUND)
list(FILTER EXAMPLE_SOURCES EXCLUDE REGEX "test_gdeflate_batch_c_api.c")
list(FILTER EXAMPLE_SOURCES EXCLUDE REGEX "test_gdeflate.cpp")
endif()
if(NOT ans_FOUND)
list(FILTER EXAMPLE_SOURCES EXCLUDE REGEX "test_ans_batch_c_api.c")
endif()

if(NOT bitcomp_FOUND)
list(FILTER EXAMPLE_SOURCES EXCLUDE REGEX "test_bitcomp.cpp")
endif()

function(add_test_file EXAMPLE_SOURCE)
  get_filename_component(BARE_NAME "${EXAMPLE_SOURCE}" NAME_WE)
  add_executable(${BARE_NAME} ${EXAMPLE_SOURCE})
  if (CUDA_BACKEND)
  if (NOT MSVC)
    target_link_libraries(${BARE_NAME} PRIVATE hipcomp CUDA::cudart)
  else()
    # Including CUDA::cudart on Windows resulted in multiple definition
    # linker errors for gdeflate and bitcomp tests, since
    # cudart_static.lib is already included.
    target_link_libraries(${BARE_NAME} PRIVATE hipcomp)
  endif()
  else (CUDA_BACKEND)
    target_link_libraries(${BARE_NAME} PRIVATE hipcomp hip::host)
  endif (CUDA_BACKEND)
  add_test(NAME ${BARE_NAME} COMMAND ${BARE_NAME})
endfunction()

foreach(EXAMPLE_SOURCE ${EXAMPLE_SOURCES})
  get_filename_component(BARE_NAME "${EXAMPLE_NAME}" NAME_WE)
  add_test_file(${EXAMPLE_SOURCE} ${EXAMPLE_NAME})
endforeach(EXAMPLE_SOURCE ${EXAMPLE_SOURCES})
