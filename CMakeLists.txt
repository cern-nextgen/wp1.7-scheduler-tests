# CMAKE_CUDA_ARCHITECTURES <native> requires CMake 3.24.
cmake_minimum_required(VERSION 3.24)

project(
  CoScheduler
  VERSION 0.0.1
  LANGUAGES CXX)

option(BUILD_TESTS "Build tests" ON)
option(BUILD_TRACCC "Build traccc" ON)
option(BUILD_PATATRACK "Build Patatrack" OFF)

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
#
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# Set up the built-in externals.
add_subdirectory(externals)

find_package(TBB REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(Boost REQUIRED)
if(BUILD_PATATRACK)
  find_package(Patatrack REQUIRED)
endif()
if(BUILD_TRACCC)
  find_package(traccc REQUIRED)
endif()

set(CMAKE_CXX_FLAGS_DEBUG "-Wall -Wextra -g -O0 -fno-omit-frame-pointer")
set(CMAKE_CUDA_FLAGS_DEBUG "-Wall -Wextra -g -O0 -fno-omit-frame-pointer")

# Add the libraries.

file(
  GLOB
  all_files
  CONFIGURE_DEPENDS
  "src/*.h"
  "src/*.cpp"
  "src/*.cuh"
  "src/*.cu")

add_library(CoScheduler SHARED ${all_files})
target_include_directories(
  CoScheduler PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
                     $<INSTALL_INTERFACE:include>)

target_link_libraries(CoScheduler PUBLIC TBB::tbb CUDA::cudart Boost::headers)

if(BUILD_PATATRACK)
  # target_link_libraries(CoScheduler PUBLIC Patatrack::CUDACore
  # Patatrack::Backtrace)
endif()

add_subdirectory(components)

# Add the executables.

add_executable(schedule_simple bin/schedule_simple.cpp)
target_link_libraries(schedule_simple PRIVATE CoScheduler components_simple)

if(BUILD_TRACCC)
  add_executable(schedule_traccc bin/schedule_traccc.cpp)
  target_link_libraries(schedule_traccc PRIVATE CoScheduler components_traccc)
endif()

# Add the tests.
enable_testing()
if(BUILD_TESTS)
  add_subdirectory(tests)
endif()
