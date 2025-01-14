#include "SecondAlgorithm.hpp"

#include <iostream>

#include "CudaKernels.cuh"
#include "EventContext.hpp"
#include "MemberFunctionName.hpp"
#include "Scheduler.hpp"


StatusCode SecondAlgorithm::initialize() {
   std::cout << MEMBER_FUNCTION_NAME(SecondAlgorithm) << std::endl;
   return StatusCode::SUCCESS;
}


AlgorithmBase::AlgCoInterface SecondAlgorithm::execute(EventContext ctx) const {
   std::cout << MEMBER_FUNCTION_NAME(SecondAlgorithm) + " part1, " << ctx.info() << std::endl;
   ctx.scheduler->setCudaSlotState(ctx.slotNumber, ctx.algNumber, false);
   launchTestKernel3(ctx.stream);
   cudaLaunchHostFunc(ctx.stream, notifyScheduler, new EventContext{ctx});
   co_yield StatusCode::SUCCESS;

   std::cout << MEMBER_FUNCTION_NAME(SecondAlgorithm) + " part2, " << ctx.info() << std::endl;
   ctx.scheduler->setCudaSlotState(ctx.slotNumber, ctx.algNumber, false);
   launchTestKernel4(ctx.stream);
   cudaLaunchHostFunc(ctx.stream, notifyScheduler, new EventContext{ctx});
   cudaStreamSynchronize(ctx.stream);
   co_return StatusCode::SUCCESS;
}


StatusCode SecondAlgorithm::finalize() {
   std::cout << MEMBER_FUNCTION_NAME(SecondAlgorithm) << std::endl;
   return StatusCode::SUCCESS;
}


const std::vector<std::string>& SecondAlgorithm::dependencies() const {
   return s_dependencies;
}


const std::vector<std::string>& SecondAlgorithm::products() const {
   return s_products;
}
