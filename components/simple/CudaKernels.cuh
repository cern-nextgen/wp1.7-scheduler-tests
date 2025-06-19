#pragma once


#include <cuda_runtime_api.h>


void launchTestKernel1(cudaStream_t stream);
void launchTestKernel2(cudaStream_t stream);
void launchTestKernel3(cudaStream_t stream);
void launchTestKernel4(cudaStream_t stream);
void launchTestKernel5(cudaStream_t stream);

void* kernel1Address();
void* kernel2Address();
void* kernel3Address();
void* kernel4Address();
void* kernel5Address();

// Macro to assert that CUDA calls do not fail
#define CUDA_ASSERT(call) \
    do { \
        cudaError_t err = (call); \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": " \
                      << cudaGetErrorString(err) << std::endl; \
            std::exit(EXIT_FAILURE); \
        } \
    } while (0)
