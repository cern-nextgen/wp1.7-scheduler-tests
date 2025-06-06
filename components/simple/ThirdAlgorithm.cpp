#include "ThirdAlgorithm.hpp"

#include <iostream>

#include "CudaKernels.cuh"
#include "EventContext.hpp"
#include "EventStore.hpp"
#include "MemberFunctionName.hpp"
#include "Scheduler.hpp"
#include "../../tests/NVTXUtils.hpp"
using WP17Scheduler::NVTXUtils::nvtxcolor;

ThirdAlgorithm::ThirdAlgorithm(bool verbose)
    : m_verbose(verbose) {}

StatusCode ThirdAlgorithm::initialize() {
    nvtx3::scoped_range range{MEMBER_FUNCTION_NAME(ThirdAlgorithm)};
    SC_CHECK(AlgorithmBase::addDependency<int>("Object2"));
    SC_CHECK(AlgorithmBase::addProduct<int>("Object4"));
    if (m_verbose) {
        std::cout << MEMBER_FUNCTION_NAME(ThirdAlgorithm) << std::endl;
    }
    return StatusCode::SUCCESS;
}

AlgorithmBase::AlgCoInterface ThirdAlgorithm::execute(EventContext ctx) const {
    if (m_verbose) {
        std::cout << MEMBER_FUNCTION_NAME(ThirdAlgorithm) + " part1 start, " << ctx.info() << std::endl;
    }
    nvtx3::unique_range range{MEMBER_FUNCTION_NAME(ThirdAlgorithm) + " part1" + ctx.info(), nvtxcolor(ctx.eventNumber), nvtx3::payload{ctx.eventNumber}};
    const int* input = nullptr;
    SC_CHECK_YIELD(EventStoreRegistry::of(ctx).retrieve(input, AlgorithmBase::dependencies()[0]));
    auto output = std::make_unique<int>(-1);
    SC_CHECK_YIELD(EventStoreRegistry::of(ctx).record(std::move(output), AlgorithmBase::products()[0]));

    if (m_verbose) {
        std::cout << MEMBER_FUNCTION_NAME(ThirdAlgorithm) + " part1, " << ctx.info() << std::endl;
    }
    ctx.scheduler->setCudaSlotState(ctx.slotNumber, 2, false);
    launchTestKernel5(ctx.stream);
    cudaLaunchHostFunc(ctx.stream, notifyScheduler, new Notification{ctx, 2});
    { auto r = std::move(range); } // End range
    co_yield StatusCode::SUCCESS;

    if (m_verbose) {
        std::cout << MEMBER_FUNCTION_NAME(ThirdAlgorithm) + " conclusion, " << ctx.info() << std::endl;
    }
    auto range2 = nvtx3::scoped_range{MEMBER_FUNCTION_NAME(ThirdAlgorithm) + " conclusion" + ctx.info(), nvtxcolor(ctx.eventNumber), nvtx3::payload{ctx.eventNumber}};
    co_return StatusCode::SUCCESS;
}

StatusCode ThirdAlgorithm::finalize() {
    nvtx3::scoped_range range{MEMBER_FUNCTION_NAME(ThirdAlgorithm)};
    if (m_verbose) {
        std::cout << MEMBER_FUNCTION_NAME(ThirdAlgorithm) << std::endl;
    }
    return StatusCode::SUCCESS;
}
