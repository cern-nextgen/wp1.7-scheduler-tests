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

The simple test:
```sh
build/bin/schedule_simple
```

or the traccc test:

The data should be downloaded (once) in the build tree by running `build/_deps/traccc-src/data/traccc_data_get_files.sh`
This will download the data in-place in the script's directory. Then run the test:

```sh
TRACCC_TEST_DATA_DIR=build/_deps/traccc-src/data/ ./build/bin/schedule_traccc
```

The download instructions for the traccc data are [in the traccc readme](https://github.com/acts-project/traccc?tab=readme-ov-file#getting-started)

```sh
git clone https://github.com/acts-project/traccc.git
./traccc/data/traccc_data_get_files.sh
```

# Traccc 101

Traccc is a framework agnostic algorithm.
Gaudi is the framework (services, algorithms tools, orchestrator), based on TBB
Athena is a framework based on Gaudi, but re-implements parts of it.

## Gaudi modules

States see wp1.7-scheduler-tests/src/AlgExecState.hpp (suspened is a new addition)

### Dependencies

- Each algo defined data dependency (like in CMS)
- Sequencers are algos or modules that can have sub-algo or sub-modules that express additional non-data dependency
  - Sometimes data dependency is not fully expressed
  - Sequencers also express filtering


Here only data dependency. In Gaudi statuses are different depending we wait for data or non data dependency.