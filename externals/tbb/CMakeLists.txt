
# CMake include(s).
cmake_minimum_required( VERSION 3.24 )
include( FetchContent )

# Tell the user what's happening.
message( STATUS "Building oneTBB as part of the project" )

# Declare where to get TBB from.
set( SCHED_TBB_SOURCE
   "URL;https://github.com/oneapi-src/oneTBB/archive/refs/tags/v2021.13.0.tar.gz;URL_MD5;f287cd007240a838286ff13e7deaee12"
   CACHE STRING "Source for TBB, when built as part of this project" )
mark_as_advanced( SCHED_TBB_SOURCE )
FetchContent_Declare( TBB ${SCHED_TBB_SOURCE} OVERRIDE_FIND_PACKAGE )

# Options used in the build of TBB.
set( TBB_TEST FALSE CACHE BOOL "Turn off the TBB tests" )
set( TBB_STRICT FALSE CACHE BOOL "Do not throw errors on compiler warnings" )

# Get it into the current directory.
FetchContent_MakeAvailable( TBB )
