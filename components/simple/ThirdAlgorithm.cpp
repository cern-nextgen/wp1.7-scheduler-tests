#include "ThirdAlgorithm.hpp"

#include <iostream>
#include <future>

#include "CudaKernels.cuh"
#include "EventContext.hpp"
#include "EventStore.hpp"
#include "MemberFunctionName.hpp"
#include "Scheduler.hpp"
#include "CUDAThread.hpp"
#include "CUDAMutex.hpp"
#include "CUDAThreadLocalStream.hpp"
#include "CUDAThreadLocalContext.hpp"
#include "../../tests/NVTXUtils.hpp"
using WP17Scheduler::NVTXUtils::nvtxcolor;

#include <cuda_runtime.h>

// --- ThirdAlgorithmGraph Implementation ---
ThirdAlgorithmGraph::ThirdAlgorithmGraph() {
    std::lock_guard<std::mutex> lock(m_graphMutex);
    CUDA_ASSERT(cudaGraphCreate(&m_graph, 0));
    m_kernel5Params.func = kernel5Address();
    m_kernel5Params.gridDim = dim3(1);
    m_kernel5Params.blockDim = dim3(1);
    m_kernel5Params.sharedMemBytes = 0;
    m_kernel5Params.kernelParams = nullptr;
    m_kernel5Params.extra = nullptr;
    CUDA_ASSERT(cudaGraphAddKernelNode(&m_kernel5Node, m_graph, nullptr, 0, &m_kernel5Params));
    m_hostFunctionParams.fn = notifyScheduler;
    m_hostFunctionParams.userData = nullptr;
    CUDA_ASSERT(cudaGraphAddHostNode(&m_HostFunctionNode, m_graph, &m_kernel5Node, 1, &m_hostFunctionParams));
    CUDA_ASSERT(cudaGraphInstantiate(&m_graphExec, m_graph, nullptr, nullptr, 0));
}
ThirdAlgorithmGraph::~ThirdAlgorithmGraph() {
    std::lock_guard<std::mutex> lock(m_graphMutex);
    if (m_graphExec) cudaGraphExecDestroy(m_graphExec);
    if (m_graph) cudaGraphDestroy(m_graph);
}

void ThirdAlgorithmGraph::launchGraph(cudaStream_t stream, Notification* notification) {
    std::lock_guard<std::mutex> lock(m_graphMutex);
    // Only update host function userData for this launch
    m_hostFunctionParams.userData = notification;
    CUDA_ASSERT(cudaGraphExecHostNodeSetParams(m_graphExec, m_HostFunctionNode, &m_hostFunctionParams));
    CUDA_ASSERT(cudaGraphLaunch(m_graphExec, stream));
}

void ThirdAlgorithmGraph::launchGraphDelegated(cudaStream_t stream, Notification* notification) {
    std::lock_guard<std::mutex> lock(m_graphMutex);
    std::promise<void> promise;
    std::future<void> future = promise.get_future();

    // Only update host function userData for this launch
    m_hostFunctionParams.userData = notification;
    CUDA_ASSERT(cudaGraphExecHostNodeSetParams(m_graphExec, m_HostFunctionNode, &m_hostFunctionParams));

    CUDAThread::post([&, this]() {
        CUDA_ASSERT(cudaGraphLaunch(m_graphExec, stream));
        promise.set_value();
    });

    future.get();
}

// --- ThirdAlgorithm Implementation ---
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

AlgorithmBase::AlgCoInterface ThirdAlgorithm::executeStraightMutexed(EventContext ctx) const {
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
    auto cudaLock = CUDAMutex::lock();
    launchTestKernel5(ctx.stream);
    cudaLaunchHostFunc(ctx.stream, notifyScheduler, new Notification{ctx, 2});
    cudaLock.unlock();
    { auto r = std::move(range); } // End range
    co_yield StatusCode::SUCCESS;

    if (m_verbose) {
        std::cout << MEMBER_FUNCTION_NAME(ThirdAlgorithm) + " conclusion, " << ctx.info() << std::endl;
    }
    auto range2 = nvtx3::scoped_range{MEMBER_FUNCTION_NAME(ThirdAlgorithm) + " conclusion" + ctx.info(), nvtxcolor(ctx.eventNumber), nvtx3::payload{ctx.eventNumber}};
    co_return StatusCode::SUCCESS;
}

AlgorithmBase::AlgCoInterface ThirdAlgorithm::executeStraightThreadLocalStreams(EventContext ctx) const {
    auto stream = CUDAThreadLocalStream::get();
    if (m_verbose) {
        std::cout << MEMBER_FUNCTION_NAME(ThirdAlgorithm) + " part1 start, " << ctx.info() << std::endl;
    }
    nvtx3::unique_range range{MEMBER_FUNCTION_NAME(ThirdAlgorithm) + " part1" + ctx.info() + " stream=" + std::to_string((uint64_t)stream), nvtxcolor(ctx.eventNumber), nvtx3::payload{ctx.eventNumber}};
    const int* input = nullptr;
    SC_CHECK_YIELD(EventStoreRegistry::of(ctx).retrieve(input, AlgorithmBase::dependencies()[0]));
    auto output = std::make_unique<int>(-1);
    SC_CHECK_YIELD(EventStoreRegistry::of(ctx).record(std::move(output), AlgorithmBase::products()[0]));

    if (m_verbose) {
        std::cout << MEMBER_FUNCTION_NAME(ThirdAlgorithm) + " part1, " << ctx.info() << std::endl;
    }
    ctx.scheduler->setCudaSlotState(ctx.slotNumber, 2, false);
    launchTestKernel5(stream);
    cudaLaunchHostFunc(stream, notifyScheduler, new Notification{ctx, 2});
    { auto r = std::move(range); } // End range
    co_yield StatusCode::SUCCESS;

    if (m_verbose) {
        std::cout << MEMBER_FUNCTION_NAME(ThirdAlgorithm) + " conclusion, " << ctx.info() << std::endl;
    }
    auto range2 = nvtx3::scoped_range{MEMBER_FUNCTION_NAME(ThirdAlgorithm) + " conclusion" + ctx.info(), nvtxcolor(ctx.eventNumber), nvtx3::payload{ctx.eventNumber}};
    co_return StatusCode::SUCCESS;
}

AlgorithmBase::AlgCoInterface ThirdAlgorithm::executeStraightThreadLocalContext(EventContext ctx) const {
    CUDAThreadLocalContext::check();
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

AlgorithmBase::AlgCoInterface ThirdAlgorithm::executeGraph(EventContext ctx) const {
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
    m_graphImpl.launchGraph(ctx.stream, new Notification{ctx, 2});
    { auto r = std::move(range); } // End range
    co_yield StatusCode::SUCCESS;

    if (m_verbose) {
        std::cout << MEMBER_FUNCTION_NAME(ThirdAlgorithm) + " conclusion, " << ctx.info() << std::endl;
    }
    auto range2 = nvtx3::scoped_range{MEMBER_FUNCTION_NAME(ThirdAlgorithm) + " conclusion" + ctx.info(), nvtxcolor(ctx.eventNumber), nvtx3::payload{ctx.eventNumber}};
    co_return StatusCode::SUCCESS;
}

AlgorithmBase::AlgCoInterface ThirdAlgorithm::executeGraphFullyDelegated(EventContext ctx) const {
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
    auto * notif = new Notification{ctx, 2};
    CUDAThread::post([this, ctx, notif]() {
        m_graphImpl.launchGraph(ctx.stream, notif);
    });
    { auto r = std::move(range); } // End range
    co_yield StatusCode::SUCCESS;

    if (m_verbose) {
        std::cout << MEMBER_FUNCTION_NAME(ThirdAlgorithm) + " conclusion, " << ctx.info() << std::endl;
    }
    auto range2 = nvtx3::scoped_range{MEMBER_FUNCTION_NAME(ThirdAlgorithm) + " conclusion" + ctx.info(), nvtxcolor(ctx.eventNumber), nvtx3::payload{ctx.eventNumber}};
    co_return StatusCode::SUCCESS;
}

AlgorithmBase::AlgCoInterface ThirdAlgorithm::executeStraightDelegated(EventContext ctx) const {
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
    auto * notif = new Notification{ctx, 2};
    CUDAThread::post([ctx, notif]() {
        launchTestKernel5(ctx.stream);
        cudaLaunchHostFunc(ctx.stream, notifyScheduler, notif);
    });
    { auto r = std::move(range); } // End range
    co_yield StatusCode::SUCCESS;

    if (m_verbose) {
        std::cout << MEMBER_FUNCTION_NAME(ThirdAlgorithm) + " conclusion, " << ctx.info() << std::endl;
    }
    auto range2 = nvtx3::scoped_range{MEMBER_FUNCTION_NAME(ThirdAlgorithm) + " conclusion" + ctx.info(), nvtxcolor(ctx.eventNumber), nvtx3::payload{ctx.eventNumber}};
    co_return StatusCode::SUCCESS;
}

AlgorithmBase::AlgCoInterface ThirdAlgorithm::executeCachedGraph(EventContext ctx) const {
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
    m_graphContainer.launchGraph(ctx.stream, new Notification{ctx, 2});
    { auto r = std::move(range); } // End range
    co_yield StatusCode::SUCCESS;

    if (m_verbose) {
        std::cout << MEMBER_FUNCTION_NAME(ThirdAlgorithm) + " conclusion, " << ctx.info() << std::endl;
    }
    auto range2 = nvtx3::scoped_range{MEMBER_FUNCTION_NAME(ThirdAlgorithm) + " conclusion" + ctx.info(), nvtxcolor(ctx.eventNumber), nvtx3::payload{ctx.eventNumber}};
    co_return StatusCode::SUCCESS;
}

AlgorithmBase::AlgCoInterface ThirdAlgorithm::executeCachedGraphDelegated(EventContext ctx) const {
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
    m_graphContainer.launchGraphDelegated(ctx.stream, new Notification{ctx, 2});
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
