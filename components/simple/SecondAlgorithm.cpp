#include "SecondAlgorithm.hpp"

#include <iostream>
#include <mutex>

#include "CudaKernels.cuh"
#include "EventContext.hpp"
#include "EventStore.hpp"
#include "MemberFunctionName.hpp"
#include "Scheduler.hpp"
#include "CUDAThread.hpp"
#include "../../tests/NVTXUtils.hpp"
using WP17Scheduler::NVTXUtils::nvtxcolor;

#include <cuda_runtime.h>

// --- SecondAlgorithmGraph Implementation ---
SecondAlgorithmGraph::SecondAlgorithmGraph() {
    std::lock_guard<std::mutex> lock(m_graphMutex);

    CUDA_ASSERT(cudaGraphCreate(&m_graph, 0));

    m_kernel3Params.func = kernel3Address();
    m_kernel3Params.gridDim = dim3(1);
    m_kernel3Params.blockDim = dim3(1);
    m_kernel3Params.sharedMemBytes = 0;
    m_kernel3Params.kernelParams = nullptr;
    m_kernel3Params.extra = nullptr;
    CUDA_ASSERT(cudaGraphAddKernelNode(&m_kernel3Node, m_graph, nullptr, 0, &m_kernel3Params));
    
    m_kernel4Params.func = kernel4Address();
    m_kernel4Params.gridDim = dim3(1);
    m_kernel4Params.blockDim = dim3(1);
    m_kernel4Params.sharedMemBytes = 0;
    m_kernel4Params.kernelParams = nullptr;
    m_kernel4Params.extra = nullptr;
    CUDA_ASSERT(cudaGraphAddKernelNode(&m_kernel4Node, m_graph, &m_kernel3Node, 1, &m_kernel4Params));
    
    m_hostFunctionParams.fn = notifyScheduler;
    m_hostFunctionParams.userData = nullptr;
    CUDA_ASSERT(cudaGraphAddHostNode(&m_HostFunctionNode, m_graph, &m_kernel4Node, 1, &m_hostFunctionParams));
    
    CUDA_ASSERT(cudaGraphInstantiate(&m_graphExec, m_graph, nullptr, nullptr, 0));
}

SecondAlgorithmGraph::~SecondAlgorithmGraph() {
    std::lock_guard<std::mutex> lock(m_graphMutex);
    if (m_graphExec) cudaGraphExecDestroy(m_graphExec);
    if (m_graph) cudaGraphDestroy(m_graph);
}

void SecondAlgorithmGraph::launchGraph(cudaStream_t stream, Notification* notification) {
    std::lock_guard<std::mutex> lock(m_graphMutex);

    // Only update host function userData for this launch
    m_hostFunctionParams.userData = notification;
    CUDA_ASSERT(cudaGraphExecHostNodeSetParams(m_graphExec, m_HostFunctionNode, &m_hostFunctionParams));

    CUDA_ASSERT(cudaGraphLaunch(m_graphExec, stream));
}

// --- SecondAlgorithm Implementation ---
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

AlgorithmBase::AlgCoInterface SecondAlgorithm::executeStraight(EventContext ctx) const {
    if (m_verbose) {
        std::cout << MEMBER_FUNCTION_NAME(SecondAlgorithm) + " part1 start, " << ctx.info() << std::endl;
    }
    auto range1 = std::make_unique<nvtx3::unique_range>(MEMBER_FUNCTION_NAME(SecondAlgorithm) + " part1" + ctx.info(), nvtxcolor(ctx.eventNumber), nvtx3::payload{ctx.eventNumber});
    const int* input = nullptr;
    SC_CHECK_YIELD(EventStoreRegistry::of(ctx).retrieve(input, AlgorithmBase::dependencies()[0]));
    auto output = std::make_unique<int>(-1);
    SC_CHECK_YIELD(EventStoreRegistry::of(ctx).record(std::move(output), AlgorithmBase::products()[0]));

    if (m_verbose) {
        std::cout << MEMBER_FUNCTION_NAME(SecondAlgorithm) + " part1, " << ctx.info() << std::endl;
    }
    ctx.scheduler->setCudaSlotState(ctx.slotNumber, 1, false);
    launchTestKernel3(ctx.stream);
    launchTestKernel4(ctx.stream);
    cudaLaunchHostFunc(ctx.stream, notifyScheduler, new Notification{ctx, 1});
    range1.reset(); // End range
    co_yield StatusCode::SUCCESS;

    if (m_verbose) {
        std::cout << MEMBER_FUNCTION_NAME(SecondAlgorithm) + " conclusion, " << ctx.info() << std::endl;
    }
    auto range2 = nvtx3::scoped_range{MEMBER_FUNCTION_NAME(SecondAlgorithm) + " conclusion" + ctx.info(), nvtxcolor(ctx.eventNumber), nvtx3::payload{ctx.eventNumber}};
    co_return StatusCode::SUCCESS;
}

AlgorithmBase::AlgCoInterface SecondAlgorithm::executeStraightDelegated(EventContext ctx) const {
    if (m_verbose) {
        std::cout << MEMBER_FUNCTION_NAME(SecondAlgorithm) + " part1 start, " << ctx.info() << std::endl;
    }
    auto range1 = std::make_unique<nvtx3::unique_range>(MEMBER_FUNCTION_NAME(SecondAlgorithm) + " part1" + ctx.info(), nvtxcolor(ctx.eventNumber), nvtx3::payload{ctx.eventNumber});
    const int* input = nullptr;
    SC_CHECK_YIELD(EventStoreRegistry::of(ctx).retrieve(input, AlgorithmBase::dependencies()[0]));
    auto output = std::make_unique<int>(-1);
    SC_CHECK_YIELD(EventStoreRegistry::of(ctx).record(std::move(output), AlgorithmBase::products()[0]));

    if (m_verbose) {
        std::cout << MEMBER_FUNCTION_NAME(SecondAlgorithm) + " part1, " << ctx.info() << std::endl;
    }
    ctx.scheduler->setCudaSlotState(ctx.slotNumber, 1, false);
    // Launch kernels in a single thread to avoid performance drop
    auto * notif = new Notification{ctx, 1};
    CUDAThread::post([ctx, notif]() {
        launchTestKernel3(ctx.stream);
        launchTestKernel4(ctx.stream);
        cudaLaunchHostFunc(ctx.stream, notifyScheduler, notif);
    });
    range1.reset(); // End range
    co_yield StatusCode::SUCCESS;

    if (m_verbose) {
        std::cout << MEMBER_FUNCTION_NAME(SecondAlgorithm) + " conclusion, " << ctx.info() << std::endl;
    }
    auto range2 = nvtx3::scoped_range{MEMBER_FUNCTION_NAME(SecondAlgorithm) + " conclusion" + ctx.info(), nvtxcolor(ctx.eventNumber), nvtx3::payload{ctx.eventNumber}};
    co_return StatusCode::SUCCESS;
}

AlgorithmBase::AlgCoInterface SecondAlgorithm::executeGraph(EventContext ctx) const {
    if (m_verbose) {
        std::cout << MEMBER_FUNCTION_NAME(SecondAlgorithm) + " [graph] part1 start, " << ctx.info() << std::endl;
    }
    nvtx3::unique_range range{MEMBER_FUNCTION_NAME(SecondAlgorithm) + " [graph] part1" + ctx.info(), nvtxcolor(ctx.eventNumber), nvtx3::payload{ctx.eventNumber}};
    const int* input = nullptr;
    SC_CHECK_YIELD(EventStoreRegistry::of(ctx).retrieve(input, AlgorithmBase::dependencies()[0]));
    auto output = std::make_unique<int>(-1);
    SC_CHECK_YIELD(EventStoreRegistry::of(ctx).record(std::move(output), AlgorithmBase::products()[0]));

    if (m_verbose) {
        std::cout << MEMBER_FUNCTION_NAME(SecondAlgorithm) + " [graph] launching CUDA graph, " << ctx.info() << std::endl;
    }
    ctx.scheduler->setCudaSlotState(ctx.slotNumber, 1, false);

    // Allocate a notification for the host node
    Notification* notification = new Notification{ctx, 1};
    m_graphImpl.launchGraph(ctx.stream, notification);

    { auto r = std::move(range); } // End range
    co_yield StatusCode::SUCCESS;

    if (m_verbose) {
        std::cout << MEMBER_FUNCTION_NAME(SecondAlgorithm) + " [graph] conclusion, " << ctx.info() << std::endl;
    }
    auto range3 = nvtx3::scoped_range{MEMBER_FUNCTION_NAME(SecondAlgorithm) + " [graph] conclusion" + ctx.info(), nvtxcolor(ctx.eventNumber), nvtx3::payload{ctx.eventNumber}};
    co_return StatusCode::SUCCESS;
}

AlgorithmBase::AlgCoInterface SecondAlgorithm::executeCachedGraph(EventContext ctx) const {
    if (m_verbose) {
        std::cout << MEMBER_FUNCTION_NAME(SecondAlgorithm) + " [graph] part1 start, " << ctx.info() << std::endl;
    }
    nvtx3::unique_range range{MEMBER_FUNCTION_NAME(SecondAlgorithm) + " [graph] part1" + ctx.info(), nvtxcolor(ctx.eventNumber), nvtx3::payload{ctx.eventNumber}};
    const int* input = nullptr;
    SC_CHECK_YIELD(EventStoreRegistry::of(ctx).retrieve(input, AlgorithmBase::dependencies()[0]));
    auto output = std::make_unique<int>(-1);
    SC_CHECK_YIELD(EventStoreRegistry::of(ctx).record(std::move(output), AlgorithmBase::products()[0]));

    if (m_verbose) {
        std::cout << MEMBER_FUNCTION_NAME(SecondAlgorithm) + " [graph] launching CUDA graph, " << ctx.info() << std::endl;
    }
    ctx.scheduler->setCudaSlotState(ctx.slotNumber, 1, false);

    // Allocate a notification for the host node
    Notification* notification = new Notification{ctx, 1};
    m_graphContainer.launchGraph(ctx.stream, notification);

    { auto r = std::move(range); } // End range
    co_yield StatusCode::SUCCESS;

    if (m_verbose) {
        std::cout << MEMBER_FUNCTION_NAME(SecondAlgorithm) + " [graph] conclusion, " << ctx.info() << std::endl;
    }
    auto range3 = nvtx3::scoped_range{MEMBER_FUNCTION_NAME(SecondAlgorithm) + " [graph] conclusion" + ctx.info(), nvtxcolor(ctx.eventNumber), nvtx3::payload{ctx.eventNumber}};
    co_return StatusCode::SUCCESS;
}

StatusCode SecondAlgorithm::finalize() {
    nvtx3::scoped_range range{MEMBER_FUNCTION_NAME(SecondAlgorithm)};
    if (m_verbose) {
        std::cout << MEMBER_FUNCTION_NAME(SecondAlgorithm) << std::endl;
    }
    return StatusCode::SUCCESS;
}
