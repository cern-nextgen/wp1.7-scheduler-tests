#include "ThirdAlgorithm.hpp"

#include <iostream>

#include "CudaKernels.cuh"
#include "EventContext.hpp"
#include "Scheduler.hpp"


StatusCode ThirdAlgorithm::initialize() {
   std::cout << "ThirdAlgorithm::initialize" << std::endl;
   return StatusCode::SUCCESS;
}


AlgorithmBase::AlgCoInterface ThirdAlgorithm::execute(EventContext ctx) const {
   std::cout << "ThirdAlgorithm::execute part1, ctx.eventNumber = " << ctx.eventNumber
             << ", ctx.slotNumber = " << ctx.slotNumber << std::endl;
   ctx.scheduler->setCudaSlotState(ctx.slotNumber, ctx.algNumber, false);
   launchTestKernel5(ctx.stream);
   cudaLaunchHostFunc(ctx.stream, notifyScheduler, new EventContext{ctx});
   cudaStreamSynchronize(ctx.stream);
   co_return StatusCode::SUCCESS;
}


StatusCode ThirdAlgorithm::finalize() {
   std::cout << "ThirdAlgorithm::finalize" << std::endl;
   return StatusCode::SUCCESS;
}


const std::vector<std::string>& ThirdAlgorithm::dependencies() const {
   return s_dependencies;
}


const std::vector<std::string>& ThirdAlgorithm::products() const {
   return s_products;
}
