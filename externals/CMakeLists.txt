
# Set up all necessary external(s).
add_subdirectory(tbb)
if(BUILD_TRACCC)
  add_subdirectory(traccc)
endif()
