#include <cstdlib>
#include <iostream>

#include "Scheduler.hpp"
#include "StatusCode.hpp"
#include "TracccAlgorithm.hpp"
#include "TracccAlgs.hpp"
#include "TracccCudaAlgorithm.hpp"

int main() {
   // Create the scheduler.
   Scheduler scheduler(1000, 4, 4);

   // Create the algorithms.
   TracccCellsAlgorithm cellsAlg(10);
   TracccComputeAlgorithm computeAlg(10);

   // Add the algorithms to the scheduler.
   scheduler.addAlgorithm(cellsAlg);
   scheduler.addAlgorithm(computeAlg);

   // Run the scheduler.
   auto w = scheduler.run().what();
   std::cout << w << std::endl;
   return EXIT_SUCCESS;
}
