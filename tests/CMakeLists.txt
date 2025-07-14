add_executable(TestEventStore TestEventStore.cpp)
target_link_libraries(TestEventStore PRIVATE CoScheduler)
add_test(NAME TestEventStore COMMAND TestEventStore)

## TBB tests
add_executable(TestTBBTaskSuspension TestTBBTaskSuspension.cpp)
target_link_libraries(TestTBBTaskSuspension PUBLIC TBB::tbb CUDA::cudart) 


## CUDA test
add_executable(CUDAGraphTest CUDAGraphs.cu)
target_link_libraries(CUDAGraphTest PRIVATE CUDA::cudart TBB::tbb) 
add_test(NAME CUDAGraphTest COMMAND CUDAGraphTest)