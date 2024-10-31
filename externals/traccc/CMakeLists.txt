
# CMake include(s).
cmake_minimum_required( VERSION 3.24 )
include( FetchContent )

# Tell the user what's happening.
message( STATUS "Building traccc as part of the project" )

# Declare where to get traccc from.
set( SCHED_TRACCC_SOURCE
   "URL;https://github.com/acts-project/traccc/archive/refs/tags/v0.17.0.tar.gz;URL_MD5;fdfa678dc60005555edd0b9575873fa3"
   CACHE STRING "Source for traccc, when built as part of this project" )
mark_as_advanced( SCHED_TRACCC_SOURCE )
FetchContent_Declare( traccc ${SCHED_TRACCC_SOURCE} OVERRIDE_FIND_PACKAGE )

# Options used in the build of traccc.
set( TRACCC_BUILD_TESTING FALSE CACHE BOOL
   "Turn off the build of the traccc unit tests" )
set( TRACCC_BUILD_BENCHMARKS FALSE CACHE BOOL
   "Turn off the build of the traccc benchmarks" )
set( TRACCC_BUILD_EXAMPLES FALSE CACHE BOOL
   "Turn off the build of the traccc examples" )
set( TRACCC_USE_ROOT FALSE CACHE BOOL
   "Don't use ROOT as part of the traccc build" )
set( TRACCC_SETUP_TBB FALSE CACHE BOOL
   "Don't set up TBB as part of the traccc build" )
set( TRACCC_BUILD_CUDA TRUE CACHE BOOL
   "Turn on the build of traccc::cuda" )

# Get it into the current directory.
FetchContent_MakeAvailable( traccc )
