#include <cstdlib>
#include <iostream>

#include "FirstAlgorithm.hpp"
#include "Scheduler.hpp"
#include "SecondAlgorithm.hpp"
#include "StatusCode.hpp"
#include "TracccAlgorithm.hpp"
#include "TracccCudaAlgorithm.hpp"
#include "ThirdAlgorithm.hpp"

#include "CUDACore/ProductBase.h"

int main() {
   // Create the scheduler.
   Scheduler scheduler(1, 1, 1);

   // Create the algorithms.
   FirstAlgorithm firstAlgorithm;
   SecondAlgorithm secondAlgorithm;
   ThirdAlgorithm thirdAlgorithm;
   TracccCudaAlgorithm tracccAlgorithm;

   // Add the algorithms to the scheduler.
//   scheduler.addAlgorithm(firstAlgorithm);
//   scheduler.addAlgorithm(secondAlgorithm);
   scheduler.addAlgorithm(tracccAlgorithm);

   // Run the scheduler.
   std::cout << (scheduler.run().what()) << std::endl;
   cms::cuda::ProductBase p{};
   return EXIT_SUCCESS;
}
