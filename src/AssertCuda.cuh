#pragma once


#ifndef NDEBUG


#include <cassert>
#include <iostream>


#define ASSERT_CUDA(cudaCall)                            \
   do {                                                  \
      cudaError_t error = cudaCall;                      \
      if(error != cudaSuccess) {                         \
         std::cerr << cudaGetErrorString(error) << '\n'; \
         assert(error == cudaSuccess);                   \
      }                                                  \
   } while(0)


#else


#define ASSERT_CUDA(condition) ((void)0)


#endif

