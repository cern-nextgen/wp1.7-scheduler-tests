#include <cstdlib>
#include <iostream>

#include "CUDACore/ProductBase.h"
#include "FirstAlgorithm.hpp"
#include "Scheduler.hpp"
#include "SecondAlgorithm.hpp"
#include "StatusCode.hpp"
#include "ThirdAlgorithm.hpp"
#include "TracccAlgorithm.hpp"
#include "TracccCudaAlgorithm.hpp"

int main() {
   // Create the scheduler.
   Scheduler scheduler(100, 4, 4);

   // Create the algorithms.
   //   FirstAlgorithm firstAlgorithm;
   //   SecondAlgorithm secondAlgorithm;
   //   ThirdAlgorithm thirdAlgorithm;
   TracccCudaAlgorithm tracccAlgorithm(10);

   // Add the algorithms to the scheduler.
//   scheduler.addAlgorithm(firstAlgorithm);
//   scheduler.addAlgorithm(secondAlgorithm);
//   scheduler.addAlgorithm(thirdAlgorithm);
   scheduler.addAlgorithm(tracccAlgorithm);

   // Run the scheduler.
   std::cout << (scheduler.run().what()) << std::endl;
   cms::cuda::ProductBase p{};
   return EXIT_SUCCESS;
}
