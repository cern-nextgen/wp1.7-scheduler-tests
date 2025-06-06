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

FirstAlgorithm::FirstAlgorithm(bool errorEnabled, int errorEventId)
    : m_errorEnabled(errorEnabled), m_errorEventId(errorEventId) {}

StatusCode FirstAlgorithm::initialize() {
   nvtx3::scoped_range range{MEMBER_FUNCTION_NAME(FirstAlgorithm)};
   SC_CHECK(AlgorithmBase::addProduct<int>("Object1"));
   SC_CHECK(AlgorithmBase::addProduct<int>("Object2"));
   std::cout << MEMBER_FUNCTION_NAME(FirstAlgorithm) << std::endl;
   return StatusCode::SUCCESS;
}

AlgorithmBase::AlgCoInterface FirstAlgorithm::execute(EventContext ctx) const {
   std::cout << MEMBER_FUNCTION_NAME(FirstAlgorithm) + " part1 start, " << ctx.info() << " tid=" << gettid() << std::endl;
   auto range1 = std::make_unique<nvtx3::unique_range>(MEMBER_FUNCTION_NAME(FirstAlgorithm) + " part1, " + ctx.info(), nvtxcolor(ctx.eventNumber), nvtx3::payload{gettid()});
   auto output1 = std::make_unique<int>(-1);
   SC_CHECK_YIELD(EventStoreRegistry::of(ctx).record(std::move(output1), AlgorithmBase::products()[0]));
   auto output2 = std::make_unique<int>(-1);
   SC_CHECK_YIELD(EventStoreRegistry::of(ctx).record(std::move(output2), AlgorithmBase::products()[1]));

   // Inject error if enabled
   if(m_errorEnabled && ctx.eventNumber == m_errorEventId) {
      StatusCode status{StatusCode::FAILURE, "FirstAlgorithm execute failed"};
      status.appendMsg("context event number: " + std::to_string(ctx.eventNumber));
      status.appendMsg("context slot number: " + std::to_string(ctx.slotNumber));
      range1.reset();
      co_return status;
   }
   
   ctx.scheduler->setCudaSlotState(ctx.slotNumber, 0, false);
   launchTestKernel1(ctx.stream);
   cudaLaunchHostFunc(ctx.stream, notifyScheduler, new Notification{ctx, 0});
   auto r1 = std::move(range1);
   range1.reset();
   std::cout << MEMBER_FUNCTION_NAME(FirstAlgorithm) + " part1 end, " << ctx.info() << " tid=" << gettid() << std::endl;
   co_yield StatusCode::SUCCESS;

   auto range2 = std::make_unique<nvtx3::unique_range>(MEMBER_FUNCTION_NAME(FirstAlgorithm) + " part2, " + ctx.info(), nvtxcolor(ctx.eventNumber), nvtx3::payload{gettid()});
   std::cout << MEMBER_FUNCTION_NAME(FirstAlgorithm) + " part2, " << ctx.info() << std::endl;
   ctx.scheduler->setCudaSlotState(ctx.slotNumber, 0, false);
   launchTestKernel2(ctx.stream);
   cudaLaunchHostFunc(ctx.stream, notifyScheduler, new Notification{ctx, 0});
   range2.reset();
   // Suspend the coroutine until the kernel is finished.
   co_yield StatusCode::SUCCESS;
   auto range3 = std::make_unique<nvtx3::unique_range>(MEMBER_FUNCTION_NAME(FirstAlgorithm) + " conclusion, " + ctx.info(), nvtxcolor(ctx.eventNumber), nvtx3::payload{gettid()});
   co_return StatusCode::SUCCESS;
}

StatusCode FirstAlgorithm::finalize() {
   nvtx3::scoped_range range{MEMBER_FUNCTION_NAME(FirstAlgorithm)};
   std::cout << MEMBER_FUNCTION_NAME(FirstAlgorithm) << std::endl;
   return StatusCode::SUCCESS;
}
