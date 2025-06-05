#include <cstdio>
#include <ctime>
#include <iostream>
#include <cstdint>

#include "CudaKernels.cuh"


static __device__ void nanospin(uint64_t ns) {
   uint64_t start = std::clock();
   constexpr uint64_t cps = 1'400'000'000; // Assuming 1.4 GHz clock rate
   uint64_t end = start + ((ns * cps) / 1'000'000'000);
   while(std::clock() < end);
}


static __global__ void testKernel1() {
   nanospin(20'000); // Spin for 20 microseconds
}


static __global__ void testKernel2() {
   nanospin(25'000);
}


static __global__ void testKernel3() {
   nanospin(30'000);
}


static __global__ void testKernel4() {
   nanospin(35'000);
}


static __global__ void testKernel5() {
   nanospin(40'000);
}


void launchTestKernel1(cudaStream_t stream) {
   std::cout << __func__ << std::endl;
   testKernel1<<<2, 2, 0, stream>>>();
}


void launchTestKernel2(cudaStream_t stream) {
   std::cout << __func__ << std::endl;
   testKernel2<<<2, 2, 0, stream>>>();
}


void launchTestKernel3(cudaStream_t stream) {
   std::cout << __func__ << std::endl;
   testKernel3<<<2, 2, 0, stream>>>();
}


void launchTestKernel4(cudaStream_t stream) {
   std::cout << __func__ << std::endl;
   testKernel4<<<2, 2, 0, stream>>>();
}


void launchTestKernel5(cudaStream_t stream) {
   std::cout << __func__ << std::endl;
   testKernel5<<<2, 2, 0, stream>>>();
}
