#include "SecondAlgorithm.hpp"

#include <iostream>

#include "CudaKernels.cuh"
#include "EventContext.hpp"
#include "EventStore.hpp"
#include "MemberFunctionName.hpp"
#include "Scheduler.hpp"
#include "../../tests/NVTXUtils.hpp"
using WP17Scheduler::NVTXUtils::nvtxcolor;


StatusCode SecondAlgorithm::initialize() {
   nvtx3::scoped_range range{MEMBER_FUNCTION_NAME(SecondAlgorithm)};
   SC_CHECK(AlgorithmBase::addDependency<int>("Object1"));
   SC_CHECK(AlgorithmBase::addProduct<int>("Object3"));
   std::cout << MEMBER_FUNCTION_NAME(SecondAlgorithm) << std::endl;
   return StatusCode::SUCCESS;
}


AlgorithmBase::AlgCoInterface SecondAlgorithm::execute(EventContext ctx) const {
   nvtx3::unique_range range{MEMBER_FUNCTION_NAME(SecondAlgorithm) + " part1" + ctx.info(), nvtxcolor(ctx.eventNumber), nvtx3::payload{ctx.eventNumber}};
   const int* input = nullptr;
   SC_CHECK_YIELD(EventStoreRegistry::of(ctx).retrieve(input, AlgorithmBase::dependencies()[0]));
   auto output = std::make_unique<int>(-1);
   SC_CHECK_YIELD(EventStoreRegistry::of(ctx).record(std::move(output), AlgorithmBase::products()[0]));

   std::cout << MEMBER_FUNCTION_NAME(SecondAlgorithm) + " part1, " << ctx.info() << std::endl;
   ctx.scheduler->setCudaSlotState(ctx.slotNumber, 1, false);
   launchTestKernel3(ctx.stream);
   cudaLaunchHostFunc(ctx.stream, notifyScheduler, new Notification{ctx, 1});
   { auto r = std::move(range); } // End range
   co_yield StatusCode::SUCCESS;

   nvtx3::unique_range range2{MEMBER_FUNCTION_NAME(SecondAlgorithm) + " part2" + ctx.info(), nvtxcolor(ctx.eventNumber), nvtx3::payload{ctx.eventNumber}};
   std::cout << MEMBER_FUNCTION_NAME(SecondAlgorithm) + " part2, " << ctx.info() << std::endl;
   ctx.scheduler->setCudaSlotState(ctx.slotNumber, 1, false);
   launchTestKernel4(ctx.stream);
   cudaLaunchHostFunc(ctx.stream, notifyScheduler, new Notification{ctx, 1});
   // Suspend the coroutine until the kernel is finished.
   { auto r2 = std::move(range2); } // End range
   co_yield StatusCode::SUCCESS;
   auto range3 = nvtx3::scoped_range{MEMBER_FUNCTION_NAME(SecondAlgorithm) + " conclusion" + ctx.info(), nvtxcolor(ctx.eventNumber), nvtx3::payload{ctx.eventNumber}};
   co_return StatusCode::SUCCESS;
}


StatusCode SecondAlgorithm::finalize() {
   nvtx3::scoped_range range{MEMBER_FUNCTION_NAME(SecondAlgorithm)};
   std::cout << MEMBER_FUNCTION_NAME(SecondAlgorithm) << std::endl;
   return StatusCode::SUCCESS;
}
