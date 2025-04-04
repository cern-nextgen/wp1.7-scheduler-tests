#include "SecondAlgorithm.hpp"

#include <iostream>

#include "CudaKernels.cuh"
#include "EventContext.hpp"
#include "EventStore.hpp"
#include "MemberFunctionName.hpp"
#include "Scheduler.hpp"


StatusCode SecondAlgorithm::initialize() {
   SC_CHECK(AlgorithmBase::addDependency<int>("Object1"));
   SC_CHECK(AlgorithmBase::addProduct<int>("Object3"));
   std::cout << MEMBER_FUNCTION_NAME(SecondAlgorithm) << std::endl;
   return StatusCode::SUCCESS;
}


AlgorithmBase::AlgCoInterface SecondAlgorithm::execute(EventContext ctx) const {
   const int* input = nullptr;
   SC_CHECK_YIELD(eventStoreOf(ctx).retrieve(input, AlgorithmBase::dependencies()[0]));
   auto output = std::make_unique<int>(-1);
   SC_CHECK_YIELD(eventStoreOf(ctx).record(std::move(output), AlgorithmBase::products()[0]));

   std::cout << MEMBER_FUNCTION_NAME(SecondAlgorithm) + " part1, " << ctx.info() << std::endl;
   ctx.scheduler->setCudaSlotState(ctx.slotNumber, 1, false);
   launchTestKernel3(ctx.stream);
   cudaLaunchHostFunc(ctx.stream, notifyScheduler, new Notification{ctx, 1});
   co_yield StatusCode::SUCCESS;

   std::cout << MEMBER_FUNCTION_NAME(SecondAlgorithm) + " part2, " << ctx.info() << std::endl;
   ctx.scheduler->setCudaSlotState(ctx.slotNumber, 1, false);
   launchTestKernel4(ctx.stream);
   cudaLaunchHostFunc(ctx.stream, notifyScheduler, new Notification{ctx, 1});
   cudaStreamSynchronize(ctx.stream);
   co_return StatusCode::SUCCESS;
}


StatusCode SecondAlgorithm::finalize() {
   std::cout << MEMBER_FUNCTION_NAME(SecondAlgorithm) << std::endl;
   return StatusCode::SUCCESS;
}
