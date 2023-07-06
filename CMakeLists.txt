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
cmake_minimum_required(VERSION 3.18)
project(nvcomp LANGUAGES C CUDA CXX)

option(BUILD_TESTS "Build unit and end-to-end tests." OFF)
option(BUILD_BENCHMARKS "Build benchmarks." OFF)
option(BUILD_EXAMPLES "Build examples." OFF)
option(BUILD_STATIC "Build a static library." OFF)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CXX_EXTENSIONS OFF)

list( APPEND CMAKE_MODULE_PATH
  ${CMAKE_SOURCE_DIR}/cmake
)

find_package(CUDAToolkit REQUIRED)

if ("${CMAKE_CXX_COMPILER_ID}" MATCHES "GNU" OR
    "${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang" OR
    "${CMAKE_CXX_COMPILER_ID}" MATCHES "Intel")

    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -g")
endif()

# Using '-G' can cause the register count to overflow for some kernels
# and cause launch failures
# set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG};-G")
set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -g")

if(CMAKE_BUILD_TYPE STREQUAL "DEBUG" OR CMAKE_BUILD_TYPE STREQUAL "Debug")
  message("Debug build.")
else()
  message("Release build.")
  # this is required, as CMake sends CMAKE_CXX_FLAGS_DEBUG to nvcc
  # if no build type is set (but not to g++ for some reason).
  set(CMAKE_BUILD_TYPE "Release")
endif()

set(HIPCOMP_FIND_QUIET "QUIET")
if(DEFINED DEVEL AND NOT DEVEL EQUAL 0)
  unset(HIPCOMP_FIND_QUIET )
  if("${CMAKE_CXX_COMPILER_ID}" MATCHES "GNU")
    # g++ warnings
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Weffc++")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wextra")
    # this doesn't work with old style CMakeLists and cuda
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wpedantic")

    # g++ warnings
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Werror")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wextra")
    # this doesn't work with old style CMakeLists and cuda
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wpedantic")
  endif()

  
  if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL "11")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --Werror=all-warnings")
  endif()

  # TODO: add clang warnings
endif()

  if (MSVC)
    add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:/W3>")
  endif()

if(NOT DEFINED CUB_DIR AND DEFINED ENV{CUB_DIR})
  set(CUB_DIR ENV{CUB_DIR})
endif()

message(STATUS "Finding HIPCUB")
if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS "11" AND NOT DEFINED CUB_DIR)
  message(FATAL_ERROR "When using a CUDA prior to CUDA 11, you must specify "
      "the path of hipcub via the 'HIPCUB_DIR' variable (e.g., "
      "'cmake -DHIPCUB_DIR=...' or 'export HIPCUB_DIR=...'). You can download hipcub "
      "from 'https://github.com/ROCmSoftwarePlatform/hipCUB'.")
endif()
find_path(HIPCUB_HEADER NAMES "hipcub/hipcub.hpp" PATHS "${HIPCUB_DIR}" "${CUDAToolkit_INCLUDE_DIRS}")
if(NOT CUB_HEADER)
  message(FATAL_ERROR "Unable to locate 'hipcub/hipcub.hpp'. Please specify a path "
      "to hipcub via 'HIPCUB_DIR'")
endif()

if(NOT DEFINED HIPCOMP_EXTS_ROOT AND DEFINED ENV{HIPCOMP_EXTS_ROOT})
  set(HIPCOMP_EXTS_ROOT "$ENV{HIPCOMP_EXTS_ROOT}")
endif()

if( DEFINED HIPCOMP_EXTS_ROOT )
  unset( HIPCOMP_FIND_QUIET )
  set( ENV{gdeflate_ROOT} ${HIPCOMP_EXTS_ROOT} )
  set( ENV{bitcomp_ROOT} ${HIPCOMP_EXTS_ROOT} )
endif()

find_package(bitcomp ${HIPCOMP_FIND_QUIET})
if (bitcomp_FOUND)
  message (STATUS "Found bitcomp")
  add_definitions(-DENABLE_BITCOMP)
endif()

find_package(gdeflate ${HIPCOMP_FIND_QUIET})
if (gdeflate_FOUND)
  message (STATUS "Found gdeflate")
  add_definitions(-DENABLE_GDEFLATE)

  # Find ZLIB for CPU compression
  find_package(ZLIB)
endif()

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
file(COPY include/ DESTINATION ${CMAKE_BINARY_DIR}/include)

# determine version
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/include/hipcomp.h" MAIN_HEADER)
string(REGEX MATCH "#define HIPCOMP_MAJOR_VERSION ([0-9]+)" _ "${MAIN_HEADER}")
set(VER_MAJOR "${CMAKE_MATCH_1}")
string(REGEX MATCH "#define HIPCOMP_MINOR_VERSION ([0-9]+)" _ "${MAIN_HEADER}")
set(VER_MINOR "${CMAKE_MATCH_1}")
string(REGEX MATCH "#define HIPCOMP_PATCH_VERSION ([0-9]+)" _ "${MAIN_HEADER}")
set(VER_PATCH "${CMAKE_MATCH_1}")

set(HIPCOMP_VERSION "${VER_MAJOR}.${VER_MINOR}.${VER_PATCH}")

message("Build hipCOMP version ${HIPCOMP_VERSION}")

if (BUILD_TESTS)
  enable_testing()
endif()

add_subdirectory(src)

if (BUILD_TESTS)
  add_subdirectory(tests)
endif()
if (BUILD_BENCHMARKS)
  add_subdirectory(benchmarks)
endif()
if (BUILD_EXAMPLES)
  add_subdirectory(examples)
endif()


if (UNIX)
  include(GNUInstallDirs)
endif()

set(INSTALL_CONFIGDIR ${CMAKE_INSTALL_LIBDIR}/cmake/hipcomp)


#------
# Build directory exporting
include(CMakePackageConfigHelpers)
configure_package_config_file("${CMAKE_CURRENT_SOURCE_DIR}/cmake/hipcomp-config.cmake.in"
                              "${CMAKE_CURRENT_BINARY_DIR}/hipcomp-config.cmake"
                              INSTALL_DESTINATION "${CMAKE_CURRENT_BINARY_DIR}")
export(TARGETS hipcomp
       FILE "${CMAKE_CURRENT_BINARY_DIR}/hipcomp-targets.cmake"
       )


#------
# Install directory setup
configure_package_config_file("${CMAKE_CURRENT_SOURCE_DIR}/cmake/hipcomp-config.cmake.in"
                              "${CMAKE_CURRENT_BINARY_DIR}/cmake/hipcomp-config.cmake"
                              INSTALL_DESTINATION "${INSTALL_CONFIGDIR}")
write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/cmake/hipcompConfigVersion.cmake
    VERSION "${HIPCOMP_VERSION}"
    COMPATIBILITY SameMinorVersion)

install(TARGETS hipcomp EXPORT hipcomp-exports DESTINATION lib)
install(EXPORT hipcomp-exports
        FILE          hipcomp-targets.cmake
        NAMESPACE     hipcomp::
        DESTINATION   "${INSTALL_CONFIGDIR}")


install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/hipcomp/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/hipcomp)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/include/hipcomp.hpp
              ${CMAKE_CURRENT_SOURCE_DIR}/include/hipcomp.h
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/cmake/hipcomp-config.cmake"
              "${CMAKE_CURRENT_BINARY_DIR}/cmake/hipcompConfigVersion.cmake"
        DESTINATION "${INSTALL_CONFIGDIR}")