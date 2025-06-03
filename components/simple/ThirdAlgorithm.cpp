#include "ThirdAlgorithm.hpp"

#include <iostream>

#include "CudaKernels.cuh"
#include "EventContext.hpp"
#include "EventStore.hpp"
#include "MemberFunctionName.hpp"
#include "Scheduler.hpp"


StatusCode ThirdAlgorithm::initialize() {
   SC_CHECK(AlgorithmBase::addDependency<int>("Object2"));
   SC_CHECK(AlgorithmBase::addProduct<int>("Object4"));
   std::cout << MEMBER_FUNCTION_NAME(ThirdAlgorithm) << std::endl;
   return StatusCode::SUCCESS;
}


AlgorithmBase::AlgCoInterface ThirdAlgorithm::execute(EventContext ctx) const {
   const int* input = nullptr;
   SC_CHECK_YIELD(EventStoreRegistry::of(ctx).retrieve(input, AlgorithmBase::dependencies()[0]));
   auto output = std::make_unique<int>(-1);
   SC_CHECK_YIELD(EventStoreRegistry::of(ctx).record(std::move(output), AlgorithmBase::products()[0]));

   std::cout << MEMBER_FUNCTION_NAME(ThirdAlgorithm) + " part1, " << ctx.info() << std::endl;
   ctx.scheduler->setCudaSlotState(ctx.slotNumber, 2, false);
   launchTestKernel5(ctx.stream);
   cudaLaunchHostFunc(ctx.stream, notifyScheduler, new Notification{ctx, 2});
   cudaStreamSynchronize(ctx.stream);
   co_return StatusCode::SUCCESS;
}


StatusCode ThirdAlgorithm::finalize() {
   std::cout << MEMBER_FUNCTION_NAME(ThirdAlgorithm) << std::endl;
   return StatusCode::SUCCESS;
}
