# WP 1.7 Scheduler tests

Experiments with asynchronous scheduler for event-processing framework using oneTBB and C++20 coroutines.

## Getting started

Requirements:
- CMake 3.24 or higher
- C++20 compatible compiler
- CUDA 12.4 or higher
- [Patatrack standalone version](https://github.com/cms-patatrack/pixeltrack-standalone)
- Boost 1.78 or higher

The project will automatically fetch and build other dependencies:
- [traccc](https://github.com/acts-project/traccc)
- oneTBB

To build the project, clone the repository and run the following commands:

```sh
Patatrack_ROOT=<path_to_pixeltrack_standalone> cmake -S . -B build
cmake --build build
```

Then run the executables:

```sh
build/bin/schedule_simple
```

or

```sh
TRACCC_TEST_DATA_DIR=<path_to_traccc_data> build/bin/schedule_traccc
```

