#include "FirstAlgorithm.hpp"

#include <atomic>
#include <iostream>

#include "CudaKernels.cuh"
#include "EventContext.hpp"
#include "MemberFunctionName.hpp"
#include "Scheduler.hpp"


StatusCode FirstAlgorithm::initialize() {
   std::cout << MEMBER_FUNCTION_NAME(FirstAlgorithm) << std::endl;
   return StatusCode::SUCCESS;
}


AlgorithmBase::AlgCoInterface FirstAlgorithm::execute(EventContext ctx) const {
   static std::atomic_int count = 0;
   std::cout << MEMBER_FUNCTION_NAME(FirstAlgorithm) + " part1, " << ctx.info() << std::endl;

   if(++count == 50) {
      StatusCode status{StatusCode::FAILURE, "FirstAlgorithm execute failed"};
      status.appendMsg("context event number: " + std::to_string(ctx.eventNumber));
      status.appendMsg("context slot number: " + std::to_string(ctx.slotNumber));
      co_return status;
   } else {
      ctx.scheduler->setCudaSlotState(ctx.slotNumber, ctx.algNumber, false);
      launchTestKernel1(ctx.stream);
      cudaLaunchHostFunc(ctx.stream, notifyScheduler, new EventContext{ctx});
      co_yield StatusCode::SUCCESS;
   }

   std::cout << MEMBER_FUNCTION_NAME(FirstAlgorithm) + " part2, " << ctx.info() << std::endl;
   ctx.scheduler->setCudaSlotState(ctx.slotNumber, ctx.algNumber, false);
   launchTestKernel2(ctx.stream);
   cudaLaunchHostFunc(ctx.stream, notifyScheduler, new EventContext{ctx});
   cudaStreamSynchronize(ctx.stream);
   co_return StatusCode::SUCCESS;
}


StatusCode FirstAlgorithm::finalize() {
   std::cout << MEMBER_FUNCTION_NAME(FirstAlgorithm) << std::endl;
   return StatusCode::SUCCESS;
}


const std::vector<std::string>& FirstAlgorithm::dependencies() const {
   return s_dependencies;
}


const std::vector<std::string>& FirstAlgorithm::products() const {
   return s_products;
}
