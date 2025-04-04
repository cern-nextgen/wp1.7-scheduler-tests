#pragma once


#include <cuda_runtime_api.h>


void launchTestKernel1(cudaStream_t stream);
void launchTestKernel2(cudaStream_t stream);
void launchTestKernel3(cudaStream_t stream);
void launchTestKernel4(cudaStream_t stream);
void launchTestKernel5(cudaStream_t stream);
