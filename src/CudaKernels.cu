#include <cstdio>
#include <ctime>
#include <iostream>

#include "CudaKernels.cuh"


static __device__ void run_until_modulo(int mod) {
   while(true) {
      auto current = std::clock();
      if(current % mod == 0) {
         return;
      }
   }
}


static __global__ void testKernel1() {
   run_until_modulo(71);
}


static __global__ void testKernel2() {
   run_until_modulo(72);
}


static __global__ void testKernel3() {
   run_until_modulo(73);
}


static __global__ void testKernel4() {
   run_until_modulo(74);
}


static __global__ void testKernel5() {
   run_until_modulo(75);
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
