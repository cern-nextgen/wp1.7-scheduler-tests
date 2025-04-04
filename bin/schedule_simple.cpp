#include <cstdlib>
#include <iostream>

#include "Scheduler.hpp"
#include "StatusCode.hpp"
#include "FirstAlgorithm.hpp"
#include "SecondAlgorithm.hpp"
#include "ThirdAlgorithm.hpp"


int main() {
  // Create the scheduler.
  Scheduler scheduler(1000, 4, 4);

  // Create the algorithms.
  FirstAlgorithm firstAlgorithm;
  SecondAlgorithm secondAlgorithm;
  ThirdAlgorithm thirdAlgorithm;

  // Add the algorithms to the scheduler.
  scheduler.addAlgorithm(firstAlgorithm);
  scheduler.addAlgorithm(secondAlgorithm);
  scheduler.addAlgorithm(thirdAlgorithm);

  // Run the scheduler.
  std::cout << (scheduler.run().what()) << std::endl;
  return EXIT_SUCCESS;
}
