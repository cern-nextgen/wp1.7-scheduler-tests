#include "FirstAlgorithm.hpp"

#include <atomic>
#include <iostream>
#include <memory>

#include "CudaKernels.cuh"
#include "EventContext.hpp"
#include "EventStore.hpp"
#include "MemberFunctionName.hpp"
#include "Scheduler.hpp"


StatusCode FirstAlgorithm::initialize() {
   SC_CHECK(AlgorithmBase::addProduct<int>("Object1"));
   SC_CHECK(AlgorithmBase::addProduct<int>("Object2"));
   std::cout << MEMBER_FUNCTION_NAME(FirstAlgorithm) << std::endl;
   return StatusCode::SUCCESS;
}


AlgorithmBase::AlgCoInterface FirstAlgorithm::execute(EventContext ctx) const {
   auto output1 = std::make_unique<int>(-1);
   SC_CHECK_YIELD(EventStoreRegistry::of(ctx).record(std::move(output1), AlgorithmBase::products()[0]));
   auto output2 = std::make_unique<int>(-1);
   SC_CHECK_YIELD(EventStoreRegistry::of(ctx).record(std::move(output2), AlgorithmBase::products()[1]));

   static std::atomic_int count = 0;
   std::cout << MEMBER_FUNCTION_NAME(FirstAlgorithm) + " part1, " << ctx.info() << std::endl;

   if(++count == 50) {
      StatusCode status{StatusCode::FAILURE, "FirstAlgorithm execute failed"};
      status.appendMsg("context event number: " + std::to_string(ctx.eventNumber));
      status.appendMsg("context slot number: " + std::to_string(ctx.slotNumber));
      co_return status;
   } else {
      ctx.scheduler->setCudaSlotState(ctx.slotNumber, 0, false);
      launchTestKernel1(ctx.stream);
      cudaLaunchHostFunc(ctx.stream, notifyScheduler, new Notification{ctx, 0});
      co_yield StatusCode::SUCCESS;
   }

   std::cout << MEMBER_FUNCTION_NAME(FirstAlgorithm) + " part2, " << ctx.info() << std::endl;
   ctx.scheduler->setCudaSlotState(ctx.slotNumber, 0, false);
   launchTestKernel2(ctx.stream);
   cudaLaunchHostFunc(ctx.stream, notifyScheduler, new Notification{ctx, 0});
   cudaStreamSynchronize(ctx.stream);
   co_return StatusCode::SUCCESS;
}


StatusCode FirstAlgorithm::finalize() {
   std::cout << MEMBER_FUNCTION_NAME(FirstAlgorithm) << std::endl;
   return StatusCode::SUCCESS;
}
