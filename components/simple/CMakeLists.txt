file(
  GLOB
  all_files
  CONFIGURE_DEPENDS
  "*.h"
  "*.cpp"
  "*.cuh"
  "*.cu")
add_library(components_simple OBJECT ${all_files})
target_include_directories(
  components_simple PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
                $<INSTALL_INTERFACE:include>)
target_link_libraries(components_simple PUBLIC CoScheduler)
