#pragma once


#ifndef NDEBUG


#include <cstdlib>
#include <iostream>

#include "Assert.hpp"


#define ASSERT_CUDA(cudaCall)                                                        \
   do {                                                                              \
      cudaError_t error = cudaCall;                                                  \
      if(error != cudaSuccess) {                                                     \
         std::cerr << cudaGetErrorString(error) << std::endl                         \
                   << detail::trace_err(__FILE__, __func__, __LINE__) << (#cudaCall) \
                   << " failed." << std::endl;                                       \
         std::abort();                                                               \
      }                                                                              \
   } while(0)


#else


#define ASSERT_CUDA(condition) ((void)0)


#endif

