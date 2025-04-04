file(
  GLOB
  all_files
  CONFIGURE_DEPENDS
  "*.h"
  "*.cpp"
  "*.cuh"
  "*.cu")
add_library(components_traccc OBJECT ${all_files})
target_include_directories(
  components_traccc PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
                           $<INSTALL_INTERFACE:include>)
target_link_libraries(
  components_traccc PUBLIC CoScheduler traccc::core traccc::io traccc::cuda
                           traccc::device_common vecmem::cuda)
