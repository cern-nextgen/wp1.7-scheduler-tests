When configuring CMake, one should pass -DCMAKE_PREFIX_PATH=<path_to_patatrack>.
Patatrack does not have CMake support and should be installed locally.
Before running executable, make sure input files are available: export TRACCC_TEST_DATA_DIR=<path_to_traccc_data>.

In the main.cpp file, I have commented out test algorithms FirstAlgorithm, SecondAlgorithm, and ThirdAlgorithm.
They are used to verify that CUDA callbacks and algorithm dependencies are managed correctly.
Uncomment and run them to test the scheduler.

Sequential version of traccc algorithm is working properly but crashing with CUDA. It may or may not crash
depending on whether it is run in debug or release mode.

The promise type of the coroutine is parameterized with respect to co_yield and co_return parameters. Promise<void, void>
means there is no co_yield and empty co_return. We use Promise<StatusCode, StatusCode>.

There is TracccAlgorithm and TracccAlgs files. TracccAlgs is supposed to be a replacement for TracccAlgorithm but
using EventStore. At the moment, EventStore only records unique pointer objects. However, some types are not
copyable or they are expensive to copy. Therefore, another record function should be added that stores raw pointer objects.
In particular, it happens inside execute function in TracccAlgs.cpp.

Algorithm dependencies are set up using protected member functions addDependency and addProduct of AlgorithmBase base class.

