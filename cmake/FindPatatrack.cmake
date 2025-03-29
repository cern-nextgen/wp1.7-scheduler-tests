#[=======================================================================[.rst:
FindPatatrack
-------

Finds selected components of standalone Patatrack pixel tracking.

Imported Targets
^^^^^^^^^^^^^^^^

This module provides the following imported targets, if found:

``PATATRACK::backtrace``
  The backtrace library
``Patatrack::CUDACore``
  The CUDACore library


Result Variables
^^^^^^^^^^^^^^^^

This will define the following variables:

``PATATRACK_FOUND``
  True if the system has the Patatrack library.
``PATATRACK_INCLUDE_DIRS``
  Include directories needed to use Patatrack.
``PATATRACK_LIBRARIES``
  Libraries needed to link to Patatrack.
``PATATRACK_BACKTRACE_LIBRARY``
  Libraries needed to link to Patatrack::backtrace.
``PATATRACK_CUDACORE_LIBRARY``
  Libraries needed to link to Patatrack::CUDACore.
``PATATRACK_CUDACORE_INCLUDE_DIR``
  Libraries needed to use Patatrack::CUDACore.

#]=======================================================================]

find_library(
  PATATRACK_BACKTRACE_LIBRARY
  NAMES "backtrace"
  PATH_SUFFIXES "external/libbacktrace/lib")

find_library(
  PATATRACK_CUDACORE_LIBRARY
  NAMES "CUDACore"
  PATH_SUFFIXES "lib/cuda")

find_path(
  PATATRACK_CUDACORE_INCLUDE_DIR
  NAMES "CUDACore"
  PATH_SUFFIXES "src/cuda")

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
  Patatrack HANDLE_COMPONENTS
  FOUND_VAR PATATRACK_FOUND
  REQUIRED_VARS PATATRACK_CUDACORE_LIBRARY PATATRACK_BACKTRACE_LIBRARY
                PATATRACK_CUDACORE_INCLUDE_DIR)

set(PATATRACK_LIBRARIES ${PATATRACK_BACKTRACE_LIBRARY}
                        ${PATATRACK_CUDACORE_LIBRARY})
set(PATATRACK_INCLUDE_DIRS ${PATATRACK_CUDACORE_INCLUDE_DIR})
add_library(Patatrack::CUDACore UNKNOWN IMPORTED)
set_target_properties(
  Patatrack::CUDACore
  PROPERTIES IMPORTED_LOCATION "${PATATRACK_CUDACORE_LIBRARY}"
             INTERFACE_INCLUDE_DIRECTORIES "${PATATRACK_CUDACORE_INCLUDE_DIR}")
add_library(Patatrack::Backtrace UNKNOWN IMPORTED)
set_target_properties(
  Patatrack::Backtrace PROPERTIES IMPORTED_LOCATION
                                  "${PATATRACK_BACKTRACE_LIBRARY}")

foreach(comp IN LISTS PATATRACK_FIND_COMPONENTS)
  mark_as_advanced(PATATRACK_${comp}_LIBRARY)
endforeach()
mark_as_advanced(PATATRACK_CUDACORE_INCLUDE_DIR)
mark_as_advanced(PATATRACK_INCLUDE_DIR)
