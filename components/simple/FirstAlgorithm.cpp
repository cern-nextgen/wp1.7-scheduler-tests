#include "FirstAlgorithm.hpp"

#include <atomic>
#include <iostream>
#include <memory>

#include "CudaKernels.cuh"
#include "EventContext.hpp"
#include "EventStore.hpp"
#include "MemberFunctionName.hpp"
#include "Scheduler.hpp"
#include "../../tests/NVTXUtils.hpp"
using WP17Scheduler::NVTXUtils::nvtxcolor;


StatusCode FirstAlgorithm::initialize() {
   nvtx3::scoped_range range{MEMBER_FUNCTION_NAME(FirstAlgorithm)};
   SC_CHECK(AlgorithmBase::addProduct<int>("Object1"));
   SC_CHECK(AlgorithmBase::addProduct<int>("Object2"));
   std::cout << MEMBER_FUNCTION_NAME(FirstAlgorithm) << std::endl;
   return StatusCode::SUCCESS;
}


AlgorithmBase::AlgCoInterface FirstAlgorithm::execute(EventContext ctx) const {
   nvtx3::unique_range range1{MEMBER_FUNCTION_NAME(FirstAlgorithm) + " (1)", nvtxcolor(ctx.eventNumber), nvtx3::payload{ctx.eventNumber}};
   auto output1 = std::make_unique<int>(-1);
   SC_CHECK_YIELD(EventStoreRegistry::of(ctx).record(std::move(output1), AlgorithmBase::products()[0]));
   auto output2 = std::make_unique<int>(-1);
   SC_CHECK_YIELD(EventStoreRegistry::of(ctx).record(std::move(output2), AlgorithmBase::products()[1]));

   static std::atomic_int count = 0;
   std::cout << MEMBER_FUNCTION_NAME(FirstAlgorithm) + " part1, " << ctx.info() << std::endl;

   if(++count == 50) {
      auto r1 = std::move(range1);
      StatusCode status{StatusCode::FAILURE, "FirstAlgorithm execute failed"};
      status.appendMsg("context event number: " + std::to_string(ctx.eventNumber));
      status.appendMsg("context slot number: " + std::to_string(ctx.slotNumber));
      co_return status;
   } else {
      auto r1 = std::move(range1);
      ctx.scheduler->setCudaSlotState(ctx.slotNumber, 0, false);
      launchTestKernel1(ctx.stream);
      cudaLaunchHostFunc(ctx.stream, notifyScheduler, new Notification{ctx, 0});
      co_yield StatusCode::SUCCESS;
   }

   nvtx3::scoped_range range2{MEMBER_FUNCTION_NAME(FirstAlgorithm) + " (2)", nvtxcolor(ctx.eventNumber), nvtx3::payload{ctx.eventNumber}};
   std::cout << MEMBER_FUNCTION_NAME(FirstAlgorithm) + " part2, " << ctx.info() << std::endl;
   ctx.scheduler->setCudaSlotState(ctx.slotNumber, 0, false);
   launchTestKernel2(ctx.stream);
   cudaLaunchHostFunc(ctx.stream, notifyScheduler, new Notification{ctx, 0});
   cudaStreamSynchronize(ctx.stream);
   co_return StatusCode::SUCCESS;
}


StatusCode FirstAlgorithm::finalize() {
   nvtx3::scoped_range range{MEMBER_FUNCTION_NAME(FirstAlgorithm)};
   std::cout << MEMBER_FUNCTION_NAME(FirstAlgorithm) << std::endl;
   return StatusCode::SUCCESS;
}
