# CMAKE_CUDA_ARCHITECTURES <native> requires CMake 3.24.
cmake_minimum_required(VERSION 3.24)

# Set the architecture to build code for.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES
      native
      CACHE STRING "CUDA architectures to build device code for")
endif()

project(
  CoScheduler
  VERSION 0.0.1
  LANGUAGES CXX CUDA)

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
    17
    CACHE STRING "The (CUDA) C++ standard to use")

find_package(TBB REQUIRED)
find_package(CUDAToolkit REQUIRED)

file(
  GLOB
  all_files
  CONFIGURE_DEPENDS
  "src/*.h"
  "src/*.cpp"
  "src/*.cuh"
  "src/*.cu")

add_executable(CoScheduler ${all_files})
set(CMAKE_CXX_FLAGS_DEBUG "-Wall -Wextra -g -O0 -fno-omit-frame-pointer -fsanitize=address")
set(CMAKE_CUDA_FLAGS_DEBUG "-Wall -Wextra -g -O0 -fno-omit-frame-pointer")
target_link_libraries(CoScheduler PRIVATE TBB::tbb CUDA::cudart)
