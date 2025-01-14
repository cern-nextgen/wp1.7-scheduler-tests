#include "ThirdAlgorithm.hpp"

#include <iostream>

#include "CudaKernels.cuh"
#include "EventContext.hpp"
#include "MemberFunctionName.hpp"
#include "Scheduler.hpp"


StatusCode ThirdAlgorithm::initialize() {
   std::cout << MEMBER_FUNCTION_NAME(ThirdAlgorithm) << std::endl;
   return StatusCode::SUCCESS;
}


AlgorithmBase::AlgCoInterface ThirdAlgorithm::execute(EventContext ctx) const {
   std::cout << MEMBER_FUNCTION_NAME(ThirdAlgorithm) + " part1, " << ctx.info() << std::endl;
   ctx.scheduler->setCudaSlotState(ctx.slotNumber, ctx.algNumber, false);
   launchTestKernel5(ctx.stream);
   cudaLaunchHostFunc(ctx.stream, notifyScheduler, new EventContext{ctx});
   cudaStreamSynchronize(ctx.stream);
   co_return StatusCode::SUCCESS;
}


StatusCode ThirdAlgorithm::finalize() {
   std::cout << MEMBER_FUNCTION_NAME(ThirdAlgorithm) << std::endl;
   return StatusCode::SUCCESS;
}


const std::vector<std::string>& ThirdAlgorithm::dependencies() const {
   return s_dependencies;
}


const std::vector<std::string>& ThirdAlgorithm::products() const {
   return s_products;
}
