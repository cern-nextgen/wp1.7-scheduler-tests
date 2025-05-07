# Development notes

Sequential version of traccc algorithm is working properly but crashing with CUDA. It may or may not crash
depending on whether it is run in debug or release mode.

The promise type of the coroutine is parametrized with respect to co_yield and co_return parameters. Promise<void, void>
means there is no co_yield and empty co_return. We use Promise<StatusCode, StatusCode>.

There is TracccAlgorithm and TracccAlgs files. TracccAlgs is supposed to be a replacement for TracccAlgorithm but
using EventStore. At the moment, EventStore only records unique pointer objects. However, some types are not
copyable or they are expensive to copy. Therefore, another record function should be added that stores raw pointer objects.
In particular, it happens inside execute function in TracccAlgs.cpp.

Algorithm dependencies are set up using protected member functions addDependency and addProduct of AlgorithmBase base class.

## Building patatrack standalone

Get [pixeltrack-standalone](https://github.com/cms-patatrack/pixeltrack-standalone) project:

```sh
git clone git@github.com:cms-patatrack/pixeltrack-standalone.git
cd pixeltrack-standalone
```

Patatrack is using Makefile. Edit following lines to point to local packages:

```makefile
CUDA_BASE := /usr/local/cuda
TBB_BASE      := $(ONEAPI_BASE)/tbb/latest
TBB_LIBDIR    := $(TBB_BASE)/lib
BOOST_BASE := /usr
```

Then, build:

```sh
make cuda
```

## Building on AlmaLinux8
 AlmaLinux 8 has an old default compiler, so we want to use a newer one. GCC 14 is easily available in the package `gcc-toolset-14`. In turn, the host compiler should be indicated to `nvcc`.

Also the default Boost version (1.68) is too old and wee need the newer boost1.78-devel package, which is not in the include path. (This is now reflected in the CMake file as well).

 The CMake comfiguration hence becomes:
 ```bash
 Patatrack_ROOT=/home/cano/NGT/pixeltrack-standalone/ CXX=/opt/rh/gcc-toolset-14/root/usr/bin/g++ CUDACXX="/usr/local/cuda-12.9/bin/nvcc -ccbin ${CXX}" cmake -S wp1.7-scheduler-tests/ -B build/wp1.7-scheduler-test/ -DBoost_INCLUDE_DIR=/usr/include/boost1.78/
 ```
