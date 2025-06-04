add_executable(TestEventStore TestEventStore.cpp)
target_link_libraries(TestEventStore PRIVATE CoScheduler)
add_test(NAME TestEventStore COMMAND TestEventStore)

## TBB tests
add_executable(TestTBBTaskSuspension TestTBBTaskSuspension.cpp)
target_link_libraries(TestTBBTaskSuspension PUBLIC TBB::tbb CUDA::cudart) 
