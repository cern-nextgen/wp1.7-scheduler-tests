#include "FirstAlgorithm.hpp"

#include <iostream>
#include <memory>

#include "CudaKernels.cuh"
#include "EventContext.hpp"
#include "EventStore.hpp"
#include "MemberFunctionName.hpp"
#include "Scheduler.hpp"
#include "../../tests/NVTXUtils.hpp"
using WP17Scheduler::NVTXUtils::nvtxcolor;

#include <cuda_runtime.h>

// --- FirstAlgorithmGraph Implementation ---

FirstAlgorithmGraph::FirstAlgorithmGraph() {
    std::lock_guard<std::mutex> lock(m_graphMutex);

    CUDA_ASSERT(cudaGraphCreate(&m_graph, 0));

    m_kernel1Params.func = kernel1Address();
    m_kernel1Params.gridDim = dim3(2);
    m_kernel1Params.blockDim = dim3(2);
    m_kernel1Params.sharedMemBytes = 0;
    m_kernel1Params.kernelParams = nullptr;
    m_kernel1Params.extra = nullptr;
    CUDA_ASSERT(cudaGraphAddKernelNode(&m_kernel1Node, m_graph, nullptr, 0, &m_kernel1Params));
    m_kernel2Params.func = kernel2Address();
    m_kernel2Params.gridDim = dim3(2);
    m_kernel2Params.blockDim = dim3(2); 
    m_kernel2Params.sharedMemBytes = 0;
    m_kernel2Params.kernelParams = nullptr;
    m_kernel2Params.extra = nullptr;
    CUDA_ASSERT(cudaGraphAddKernelNode(&m_kernel2Node, m_graph, &m_kernel1Node, 1, &m_kernel2Params));
    m_hostFunctionParams.fn = notifyScheduler;
    m_hostFunctionParams.userData = nullptr;
    CUDA_ASSERT(cudaGraphAddHostNode(&m_HostFunctionNode, m_graph, &m_kernel2Node, 1, &m_hostFunctionParams));
    CUDA_ASSERT(cudaGraphInstantiate(&m_graphExec, m_graph, nullptr, nullptr, 0));
}

FirstAlgorithmGraph::~FirstAlgorithmGraph() {
    std::lock_guard<std::mutex> lock(m_graphMutex);
    if (m_graphExec) cudaGraphExecDestroy(m_graphExec);
    if (m_graph) cudaGraphDestroy(m_graph);
}

void FirstAlgorithmGraph::launchGraph(cudaStream_t stream, Notification* notification) {
    std::lock_guard<std::mutex> lock(m_graphMutex);

    // Only update host function userData for this launch
    m_hostFunctionParams.userData = notification;
    CUDA_ASSERT(cudaGraphExecHostNodeSetParams(m_graphExec, m_HostFunctionNode, &m_hostFunctionParams));

    CUDA_ASSERT(cudaGraphLaunch(m_graphExec, stream));
}

// --- FirstAlgorithm Implementation ---

FirstAlgorithm::FirstAlgorithm(bool errorEnabled, int errorEventId, bool verbose)
    : m_errorEnabled(errorEnabled), m_errorEventId(errorEventId), m_verbose(verbose) {}

StatusCode FirstAlgorithm::initialize() {
    nvtx3::scoped_range range{MEMBER_FUNCTION_NAME(FirstAlgorithm)};
    SC_CHECK(AlgorithmBase::addProduct<int>("Object1"));
    SC_CHECK(AlgorithmBase::addProduct<int>("Object2"));
    if (m_verbose) {
        std::cout << MEMBER_FUNCTION_NAME(FirstAlgorithm) << std::endl;
    }
    return StatusCode::SUCCESS;
}

AlgorithmBase::AlgCoInterface FirstAlgorithm::execute(EventContext ctx) const {
    if (m_verbose) {
        std::cout << MEMBER_FUNCTION_NAME(FirstAlgorithm) + " part1 start, " << ctx.info() << " tid=" << gettid() << std::endl;
    }
    auto range1 = std::make_unique<nvtx3::unique_range>(MEMBER_FUNCTION_NAME(FirstAlgorithm) + " part1, " + ctx.info(), nvtxcolor(ctx.eventNumber), nvtx3::payload{gettid()});
    auto output1 = std::make_unique<int>(-1);
    SC_CHECK_YIELD(EventStoreRegistry::of(ctx).record(std::move(output1), AlgorithmBase::products()[0]));
    auto output2 = std::make_unique<int>(-1);
    SC_CHECK_YIELD(EventStoreRegistry::of(ctx).record(std::move(output2), AlgorithmBase::products()[1]));

    // Inject error if enabled
    if (m_errorEnabled && ctx.eventNumber == m_errorEventId) {
        StatusCode status{StatusCode::FAILURE, "FirstAlgorithm execute failed"};
        status.appendMsg("context event number: " + std::to_string(ctx.eventNumber));
        status.appendMsg("context slot number: " + std::to_string(ctx.slotNumber));
        range1.reset();
        co_return status;
    }

    ctx.scheduler->setCudaSlotState(ctx.slotNumber, 0, false);
    launchTestKernel1(ctx.stream);
    cudaLaunchHostFunc(ctx.stream, notifyScheduler, new Notification{ctx, 0});
    if (m_verbose) {
        std::cout << MEMBER_FUNCTION_NAME(FirstAlgorithm) + " part1 end, " << ctx.info() << " tid=" << gettid() << std::endl;
    }
    range1.reset();
    co_yield StatusCode::SUCCESS;

    auto range2 = std::make_unique<nvtx3::unique_range>(MEMBER_FUNCTION_NAME(FirstAlgorithm) + " part2, " + ctx.info(), nvtxcolor(ctx.eventNumber), nvtx3::payload{gettid()});
    if (m_verbose) {
        std::cout << MEMBER_FUNCTION_NAME(FirstAlgorithm) + " part2, " << ctx.info() << std::endl;
    }
    ctx.scheduler->setCudaSlotState(ctx.slotNumber, 0, false);
    launchTestKernel2(ctx.stream);
    cudaLaunchHostFunc(ctx.stream, notifyScheduler, new Notification{ctx, 0});
    range2.reset();
    co_yield StatusCode::SUCCESS;

    auto range3 = std::make_unique<nvtx3::unique_range>(MEMBER_FUNCTION_NAME(FirstAlgorithm) + " conclusion, " + ctx.info(), nvtxcolor(ctx.eventNumber), nvtx3::payload{gettid()});
    co_return StatusCode::SUCCESS;
}

AlgorithmBase::AlgCoInterface FirstAlgorithm::executeStraight(EventContext ctx) const {
    if (m_verbose) {
        std::cout << MEMBER_FUNCTION_NAME(FirstAlgorithm) + " part1 start, " << ctx.info() << " tid=" << gettid() << std::endl;
    }
    auto range1 = std::make_unique<nvtx3::unique_range>(MEMBER_FUNCTION_NAME(FirstAlgorithm) + " part1, " + ctx.info(), nvtxcolor(ctx.eventNumber), nvtx3::payload{gettid()});
    auto output1 = std::make_unique<int>(-1);
    SC_CHECK_YIELD(EventStoreRegistry::of(ctx).record(std::move(output1), AlgorithmBase::products()[0]));
    auto output2 = std::make_unique<int>(-1);
    SC_CHECK_YIELD(EventStoreRegistry::of(ctx).record(std::move(output2), AlgorithmBase::products()[1]));

    // Inject error if enabled
    if (m_errorEnabled && ctx.eventNumber == m_errorEventId) {
        StatusCode status{StatusCode::FAILURE, "FirstAlgorithm execute failed"};
        status.appendMsg("context event number: " + std::to_string(ctx.eventNumber));
        status.appendMsg("context slot number: " + std::to_string(ctx.slotNumber));
        range1.reset();
        co_return status;
    }

    ctx.scheduler->setCudaSlotState(ctx.slotNumber, 0, false);
    launchTestKernel1(ctx.stream);
    launchTestKernel2(ctx.stream);
    cudaLaunchHostFunc(ctx.stream, notifyScheduler, new Notification{ctx, 0});
    range1.reset();
    co_yield StatusCode::SUCCESS;

    auto range3 = std::make_unique<nvtx3::unique_range>(MEMBER_FUNCTION_NAME(FirstAlgorithm) + " conclusion, " + ctx.info(), nvtxcolor(ctx.eventNumber), nvtx3::payload{gettid()});
    co_return StatusCode::SUCCESS;
}

AlgorithmBase::AlgCoInterface FirstAlgorithm::executeGraph(EventContext ctx) const {
    if (m_verbose) {
        std::cout << MEMBER_FUNCTION_NAME(FirstAlgorithm) + " part1 start, " << ctx.info() << " tid=" << gettid() << std::endl;
    }
    auto range1 = std::make_unique<nvtx3::unique_range>(MEMBER_FUNCTION_NAME(FirstAlgorithm) + " part1, " + ctx.info(), nvtxcolor(ctx.eventNumber), nvtx3::payload{gettid()});
    auto output1 = std::make_unique<int>(-1);
    SC_CHECK_YIELD(EventStoreRegistry::of(ctx).record(std::move(output1), AlgorithmBase::products()[0]));
    auto output2 = std::make_unique<int>(-1);
    SC_CHECK_YIELD(EventStoreRegistry::of(ctx).record(std::move(output2), AlgorithmBase::products()[1]));

    // Inject error if enabled
    if (m_errorEnabled && ctx.eventNumber == m_errorEventId) {
        StatusCode status{StatusCode::FAILURE, "FirstAlgorithm execute failed"};
        status.appendMsg("context event number: " + std::to_string(ctx.eventNumber));
        status.appendMsg("context slot number: " + std::to_string(ctx.slotNumber));
        range1.reset();
        co_return status;
    }

    ctx.scheduler->setCudaSlotState(ctx.slotNumber, 0, false);
    m_graphImpl.launchGraph(ctx.stream, new Notification{ctx, 0});
    range1.reset();
    if (m_verbose) {
        std::cout << MEMBER_FUNCTION_NAME(FirstAlgorithm) + " part1 end, " << ctx.info() << " tid=" << gettid() << std::endl;
    }
    co_yield StatusCode::SUCCESS;

    auto range2 = std::make_unique<nvtx3::unique_range>(MEMBER_FUNCTION_NAME(FirstAlgorithm) + " conclusion, " + ctx.info(), nvtxcolor(ctx.eventNumber), nvtx3::payload{gettid()});
    co_return StatusCode::SUCCESS;
}

StatusCode FirstAlgorithm::finalize() {
    nvtx3::scoped_range range{MEMBER_FUNCTION_NAME(FirstAlgorithm)};
    if (m_verbose) {
        std::cout << MEMBER_FUNCTION_NAME(FirstAlgorithm) << std::endl;
    }
    return StatusCode::SUCCESS;
}
