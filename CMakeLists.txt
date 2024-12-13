# CMAKE_CUDA_ARCHITECTURES <native> requires CMake 3.24.
cmake_minimum_required(VERSION 3.24)

project(
   CoScheduler
   VERSION 0.0.1
   LANGUAGES CXX)

# Require a C++20 compiler for C++.
set(CMAKE_CXX_STANDARD
    20
    CACHE STRING "The C++ standard to use")
set(CMAKE_CXX_STANDARD_REQUIRED
    ON
    CACHE BOOL "Whether the C++ standard is required")
set(CMAKE_CXX_EXTENSIONS OFF)

# Require a C++17 support for CUDA.
set(CMAKE_CUDA_STANDARD
    20
    CACHE STRING "The (CUDA) C++ standard to use")
# Set the architecture to build code for.
set(CMAKE_CUDA_ARCHITECTURES
    native
    CACHE STRING "CUDA architectures to build device code for")

# Enable the CUDA language, now that relevant variables have been set.
enable_language(CUDA)

# Set up the built-in externals.
add_subdirectory(externals)

# Find the necessary packages.
find_package(TBB REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(traccc REQUIRED)

file(
   GLOB
   all_files
   CONFIGURE_DEPENDS
   "src/*.h"
   "src/*.cpp"
   "src/*.cuh"
   "src/*.cu")

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
   set(TargetName "CoScheduler_debug")
else()
   set(TargetName "CoScheduler")
endif()

add_executable(${TargetName} ${all_files})
set(CMAKE_CXX_FLAGS_DEBUG "-Wall -Wextra -g -O0 -fno-omit-frame-pointer")
set(CMAKE_CUDA_FLAGS_DEBUG "-Wall -Wextra -g -O0 -fno-omit-frame-pointer")

find_library(
   PTRACK_LIB_BACKTRACE
   NAMES "backtrace"
   PATH_SUFFIXES "pixeltrack-standalone/external/libbacktrace/lib")

find_library(
   PTRACK_LIB_CUDACORE
   NAMES "CUDACore"
   PATH_SUFFIXES "pixeltrack-standalone/lib/cuda")

find_path(
   PTRACK_INCLUDE_CUDACORE
   NAMES "CUDACore"
   PATH_SUFFIXES "pixeltrack-standalone/src/cuda")

target_include_directories(${TargetName} PUBLIC ${PTRACK_INCLUDE_CUDACORE})

target_link_libraries(
   ${TargetName}
   PUBLIC TBB::tbb
          CUDA::cudart
          traccc::core
          traccc::io
          traccc::cuda
          traccc::device_common
          vecmem::cuda
          ${PTRACK_LIB_CUDACORE}
          ${PTRACK_LIB_BACKTRACE})
