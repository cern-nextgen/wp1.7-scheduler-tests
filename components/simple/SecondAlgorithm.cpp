#include "SecondAlgorithm.hpp"

#include <iostream>

#include "CudaKernels.cuh"
#include "EventContext.hpp"
#include "EventStore.hpp"
#include "MemberFunctionName.hpp"
#include "Scheduler.hpp"
#include "../../tests/NVTXUtils.hpp"
using WP17Scheduler::NVTXUtils::nvtxcolor;

SecondAlgorithm::SecondAlgorithm(bool verbose)
    : m_verbose(verbose) {}

StatusCode SecondAlgorithm::initialize() {
    nvtx3::scoped_range range{MEMBER_FUNCTION_NAME(SecondAlgorithm)};
    SC_CHECK(AlgorithmBase::addDependency<int>("Object1"));
    SC_CHECK(AlgorithmBase::addProduct<int>("Object3"));
    if (m_verbose) {
        std::cout << MEMBER_FUNCTION_NAME(SecondAlgorithm) << std::endl;
    }
    return StatusCode::SUCCESS;
}

AlgorithmBase::AlgCoInterface SecondAlgorithm::execute(EventContext ctx) const {
    if (m_verbose) {
        std::cout << MEMBER_FUNCTION_NAME(SecondAlgorithm) + " part1 start, " << ctx.info() << std::endl;
    }
    nvtx3::unique_range range{MEMBER_FUNCTION_NAME(SecondAlgorithm) + " part1" + ctx.info(), nvtxcolor(ctx.eventNumber), nvtx3::payload{ctx.eventNumber}};
    const int* input = nullptr;
    SC_CHECK_YIELD(EventStoreRegistry::of(ctx).retrieve(input, AlgorithmBase::dependencies()[0]));
    auto output = std::make_unique<int>(-1);
    SC_CHECK_YIELD(EventStoreRegistry::of(ctx).record(std::move(output), AlgorithmBase::products()[0]));

    if (m_verbose) {
        std::cout << MEMBER_FUNCTION_NAME(SecondAlgorithm) + " part1, " << ctx.info() << std::endl;
    }
    ctx.scheduler->setCudaSlotState(ctx.slotNumber, 1, false);
    launchTestKernel3(ctx.stream);
    cudaLaunchHostFunc(ctx.stream, notifyScheduler, new Notification{ctx, 1});
    { auto r = std::move(range); } // End range
    co_yield StatusCode::SUCCESS;

    if (m_verbose) {
        std::cout << MEMBER_FUNCTION_NAME(SecondAlgorithm) + " part2 start, " << ctx.info() << std::endl;
    }
    nvtx3::unique_range range2{MEMBER_FUNCTION_NAME(SecondAlgorithm) + " part2" + ctx.info(), nvtxcolor(ctx.eventNumber), nvtx3::payload{ctx.eventNumber}};
    ctx.scheduler->setCudaSlotState(ctx.slotNumber, 1, false);
    launchTestKernel4(ctx.stream);
    cudaLaunchHostFunc(ctx.stream, notifyScheduler, new Notification{ctx, 1});
    { auto r2 = std::move(range2); } // End range
    co_yield StatusCode::SUCCESS;

    if (m_verbose) {
        std::cout << MEMBER_FUNCTION_NAME(SecondAlgorithm) + " conclusion, " << ctx.info() << std::endl;
    }
    auto range3 = nvtx3::scoped_range{MEMBER_FUNCTION_NAME(SecondAlgorithm) + " conclusion" + ctx.info(), nvtxcolor(ctx.eventNumber), nvtx3::payload{ctx.eventNumber}};
    co_return StatusCode::SUCCESS;
}

StatusCode SecondAlgorithm::finalize() {
    nvtx3::scoped_range range{MEMBER_FUNCTION_NAME(SecondAlgorithm)};
    if (m_verbose) {
        std::cout << MEMBER_FUNCTION_NAME(SecondAlgorithm) << std::endl;
    }
    return StatusCode::SUCCESS;
}
