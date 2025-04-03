enable_testing()

add_executable(TestEventStore TestEventStore.cpp)
target_link_libraries(TestEventStore PRIVATE CoScheduler)
add_test(NAME TestEventStore COMMAND TestEventStore)
