#include <cstdlib>
#include <iostream>

#include "FirstAlgorithm.hpp"
#include "Scheduler.hpp"
#include "SecondAlgorithm.hpp"
#include "StatusCode.hpp"
#include "TracccAlgorithm.hpp"
#include "ThirdAlgorithm.hpp"


int main() {
   // Create the scheduler.
   Scheduler scheduler(45, 4, 4);

   // Create the algorithms.
   FirstAlgorithm firstAlgorithm;
   SecondAlgorithm secondAlgorithm;
   ThirdAlgorithm thirdAlgorithm;
   TracccAlgorithm tracccAlgorithm;

   // Add the algorithms to the scheduler.
//   scheduler.addAlgorithm(firstAlgorithm);
//   scheduler.addAlgorithm(secondAlgorithm);
   scheduler.addAlgorithm(tracccAlgorithm);

   // Run the scheduler.
   std::cout << (scheduler.run().what()) << std::endl;
   return EXIT_SUCCESS;
}
