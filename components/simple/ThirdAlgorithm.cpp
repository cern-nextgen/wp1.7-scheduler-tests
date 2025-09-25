#include "ThirdAlgorithm.hpp"

#include <iostream>
#include <future>

#include "CudaKernels.cuh"
#include "EventStore.hpp"
#include "MemberFunctionName.hpp"
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
    m_hostFunctionParams.fn = AlgorithmContext::newScheduleResumeCallback;
    m_hostFunctionParams.userData = nullptr;
    CUDA_ASSERT(cudaGraphAddHostNode(&m_HostFunctionNode, m_graph, &m_kernel5Node, 1, &m_hostFunctionParams));
    CUDA_ASSERT(cudaGraphInstantiate(&m_graphExec, m_graph, nullptr, nullptr, 0));
}
ThirdAlgorithmGraph::~ThirdAlgorithmGraph() {
    std::lock_guard<std::mutex> lock(m_graphMutex);
    if (m_graphExec) cudaGraphExecDestroy(m_graphExec);
    if (m_graph) cudaGraphDestroy(m_graph);
}

void ThirdAlgorithmGraph::launchGraph(cudaStream_t stream, AlgorithmContext* context) {
    std::lock_guard<std::mutex> lock(m_graphMutex);
    // Only update host function userData for this launch
    m_hostFunctionParams.userData = context;
    CUDA_ASSERT(cudaGraphExecHostNodeSetParams(m_graphExec, m_HostFunctionNode, &m_hostFunctionParams));
    CUDA_ASSERT(cudaGraphLaunch(m_graphExec, stream));
}

void ThirdAlgorithmGraph::launchGraphDelegated(cudaStream_t stream, AlgorithmContext* context) {
    std::lock_guard<std::mutex> lock(m_graphMutex);
    std::promise<void> promise;
    std::future<void> future = promise.get_future();

    // Only update host function userData for this launch
    m_hostFunctionParams.userData = context;
    CUDA_ASSERT(cudaGraphExecHostNodeSetParams(m_graphExec, m_HostFunctionNode, &m_hostFunctionParams));

    CUDAThread::post([&, this]() {
        CUDA_ASSERT(cudaGraphLaunch(m_graphExec, stream));
        promise.set_value();
    });

    future.get();
}

// --- ThirdAlgorithm Implementation ---
ThirdAlgorithm::ThirdAlgorithm(bool verbose)
    : m_verbose(verbose) {
      std::ignore = addDependency<int>("Object2");
      std::ignore = addProduct<int>("Object4");
    }

StatusCode ThirdAlgorithm::initialize() {
    nvtx3::scoped_range range{MEMBER_FUNCTION_NAME(ThirdAlgorithm)};
    if (m_verbose) {
        std::cout << MEMBER_FUNCTION_NAME(ThirdAlgorithm) << std::endl;
    }
    return StatusCode::SUCCESS;
}

AlgorithmBase::AlgCoInterface ThirdAlgorithm::execute(AlgorithmContext ctx) const {
    if (m_verbose) {
        std::cout << MEMBER_FUNCTION_NAME(ThirdAlgorithm) + " part1 start, " << ctx.info() << std::endl;
    }
    nvtx3::unique_range range{MEMBER_FUNCTION_NAME(ThirdAlgorithm) + " part1" + ctx.info(), nvtxcolor(ctx.eventNumber), nvtx3::payload{ctx.eventNumber}};
    const int* input = nullptr;
    SC_CHECK_YIELD(ctx.eventStore.retrieve(input, dependencies()[0]));
    auto output = std::make_unique<int>(-1);
    SC_CHECK_YIELD(ctx.eventStore.record(std::move(output), products()[0]));

    if (m_verbose) {
        std::cout << MEMBER_FUNCTION_NAME(ThirdAlgorithm) + " part1, " << ctx.info() << std::endl;
    }
    launchTestKernel5(ctx.stream);
    cudaLaunchHostFunc(ctx.stream, AlgorithmContext::newScheduleResumeCallback, new AlgorithmContext{ctx});
    { auto r = std::move(range); } // End range
    co_yield StatusCode::SUCCESS;

    if (m_verbose) {
        std::cout << MEMBER_FUNCTION_NAME(ThirdAlgorithm) + " conclusion, " << ctx.info() << std::endl;
    }
    auto range2 = nvtx3::scoped_range{MEMBER_FUNCTION_NAME(ThirdAlgorithm) + " conclusion" + ctx.info(), nvtxcolor(ctx.eventNumber), nvtx3::payload{ctx.eventNumber}};
    co_return StatusCode::SUCCESS;
}

AlgorithmBase::AlgCoInterface ThirdAlgorithm::executeStraightMutexed(AlgorithmContext ctx) const {
    if (m_verbose) {
        std::cout << MEMBER_FUNCTION_NAME(ThirdAlgorithm) + " part1 start, " << ctx.info() << std::endl;
    }
    nvtx3::unique_range range{MEMBER_FUNCTION_NAME(ThirdAlgorithm) + " part1" + ctx.info(), nvtxcolor(ctx.eventNumber), nvtx3::payload{ctx.eventNumber}};
    const int* input = nullptr;
    SC_CHECK_YIELD(ctx.eventStore.retrieve(input, dependencies()[0]));
    auto output = std::make_unique<int>(-1);
    SC_CHECK_YIELD(ctx.eventStore.record(std::move(output), products()[0]));

    if (m_verbose) {
        std::cout << MEMBER_FUNCTION_NAME(ThirdAlgorithm) + " part1, " << ctx.info() << std::endl;
    }
    auto cudaLock = CUDAMutex::lock();
    launchTestKernel5(ctx.stream);
    cudaLaunchHostFunc(ctx.stream, AlgorithmContext::newScheduleResumeCallback, new AlgorithmContext{ctx});
    cudaLock.unlock();
    { auto r = std::move(range); } // End range
    co_yield StatusCode::SUCCESS;

    if (m_verbose) {
        std::cout << MEMBER_FUNCTION_NAME(ThirdAlgorithm) + " conclusion, " << ctx.info() << std::endl;
    }
    auto range2 = nvtx3::scoped_range{MEMBER_FUNCTION_NAME(ThirdAlgorithm) + " conclusion" + ctx.info(), nvtxcolor(ctx.eventNumber), nvtx3::payload{ctx.eventNumber}};
    co_return StatusCode::SUCCESS;
}

AlgorithmBase::AlgCoInterface ThirdAlgorithm::executeStraightThreadLocalStreams(AlgorithmContext ctx) const {
    auto stream = CUDAThreadLocalStream::get();
    if (m_verbose) {
        std::cout << MEMBER_FUNCTION_NAME(ThirdAlgorithm) + " part1 start, " << ctx.info() << std::endl;
    }
    nvtx3::unique_range range{MEMBER_FUNCTION_NAME(ThirdAlgorithm) + " part1" + ctx.info() + " stream=" + std::to_string((uint64_t)stream), nvtxcolor(ctx.eventNumber), nvtx3::payload{ctx.eventNumber}};
    const int* input = nullptr;
    SC_CHECK_YIELD(ctx.eventStore.retrieve(input, dependencies()[0]));
    auto output = std::make_unique<int>(-1);
    SC_CHECK_YIELD(ctx.eventStore.record(std::move(output), products()[0]));

    if (m_verbose) {
        std::cout << MEMBER_FUNCTION_NAME(ThirdAlgorithm) + " part1, " << ctx.info() << std::endl;
    }
    launchTestKernel5(stream);
    cudaLaunchHostFunc(stream, AlgorithmContext::newScheduleResumeCallback, new AlgorithmContext{ctx});
    { auto r = std::move(range); } // End range
    co_yield StatusCode::SUCCESS;

    if (m_verbose) {
        std::cout << MEMBER_FUNCTION_NAME(ThirdAlgorithm) + " conclusion, " << ctx.info() << std::endl;
    }
    auto range2 = nvtx3::scoped_range{MEMBER_FUNCTION_NAME(ThirdAlgorithm) + " conclusion" + ctx.info(), nvtxcolor(ctx.eventNumber), nvtx3::payload{ctx.eventNumber}};
    co_return StatusCode::SUCCESS;
}

AlgorithmBase::AlgCoInterface ThirdAlgorithm::executeStraightThreadLocalContext(AlgorithmContext ctx) const {
    CUDAThreadLocalContext::check();
    if (m_verbose) {
        std::cout << MEMBER_FUNCTION_NAME(ThirdAlgorithm) + " part1 start, " << ctx.info() << std::endl;
    }
    nvtx3::unique_range range{MEMBER_FUNCTION_NAME(ThirdAlgorithm) + " part1" + ctx.info(), nvtxcolor(ctx.eventNumber), nvtx3::payload{ctx.eventNumber}};
    const int* input = nullptr;
    SC_CHECK_YIELD(ctx.eventStore.retrieve(input, dependencies()[0]));
    auto output = std::make_unique<int>(-1);
    SC_CHECK_YIELD(ctx.eventStore.record(std::move(output), products()[0]));

    if (m_verbose) {
        std::cout << MEMBER_FUNCTION_NAME(ThirdAlgorithm) + " part1, " << ctx.info() << std::endl;
    }
    launchTestKernel5(ctx.stream);
    cudaLaunchHostFunc(ctx.stream, AlgorithmContext::newScheduleResumeCallback, new AlgorithmContext{ctx});
    { auto r = std::move(range); } // End range
    co_yield StatusCode::SUCCESS;

    if (m_verbose) {
        std::cout << MEMBER_FUNCTION_NAME(ThirdAlgorithm) + " conclusion, " << ctx.info() << std::endl;
    }
    auto range2 = nvtx3::scoped_range{MEMBER_FUNCTION_NAME(ThirdAlgorithm) + " conclusion" + ctx.info(), nvtxcolor(ctx.eventNumber), nvtx3::payload{ctx.eventNumber}};
    co_return StatusCode::SUCCESS;
}

AlgorithmBase::AlgCoInterface ThirdAlgorithm::executeGraph(AlgorithmContext ctx) const {
    if (m_verbose) {
        std::cout << MEMBER_FUNCTION_NAME(ThirdAlgorithm) + " part1 start, " << ctx.info() << std::endl;
    }
    nvtx3::unique_range range{MEMBER_FUNCTION_NAME(ThirdAlgorithm) + " part1" + ctx.info(), nvtxcolor(ctx.eventNumber), nvtx3::payload{ctx.eventNumber}};
    const int* input = nullptr;
    SC_CHECK_YIELD(ctx.eventStore.retrieve(input, dependencies()[0]));
    auto output = std::make_unique<int>(-1);
    SC_CHECK_YIELD(ctx.eventStore.record(std::move(output), products()[0]));

    if (m_verbose) {
        std::cout << MEMBER_FUNCTION_NAME(ThirdAlgorithm) + " part1, " << ctx.info() << std::endl;
    }
    m_graphImpl.launchGraph(ctx.stream, new AlgorithmContext{ctx});
    { auto r = std::move(range); } // End range
    co_yield StatusCode::SUCCESS;

    if (m_verbose) {
        std::cout << MEMBER_FUNCTION_NAME(ThirdAlgorithm) + " conclusion, " << ctx.info() << std::endl;
    }
    auto range2 = nvtx3::scoped_range{MEMBER_FUNCTION_NAME(ThirdAlgorithm) + " conclusion" + ctx.info(), nvtxcolor(ctx.eventNumber), nvtx3::payload{ctx.eventNumber}};
    co_return StatusCode::SUCCESS;
}

AlgorithmBase::AlgCoInterface ThirdAlgorithm::executeGraphFullyDelegated(AlgorithmContext ctx) const {
    if (m_verbose) {
        std::cout << MEMBER_FUNCTION_NAME(ThirdAlgorithm) + " part1 start, " << ctx.info() << std::endl;
    }
    nvtx3::unique_range range{MEMBER_FUNCTION_NAME(ThirdAlgorithm) + " part1" + ctx.info(), nvtxcolor(ctx.eventNumber), nvtx3::payload{ctx.eventNumber}};
    const int* input = nullptr;
    SC_CHECK_YIELD(ctx.eventStore.retrieve(input, dependencies()[0]));
    auto output = std::make_unique<int>(-1);
    SC_CHECK_YIELD(ctx.eventStore.record(std::move(output), products()[0]));

    if (m_verbose) {
        std::cout << MEMBER_FUNCTION_NAME(ThirdAlgorithm) + " part1, " << ctx.info() << std::endl;
    }
    auto * notif = new AlgorithmContext{ctx};
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

AlgorithmBase::AlgCoInterface ThirdAlgorithm::executeStraightDelegated(AlgorithmContext ctx) const {
    if (m_verbose) {
        std::cout << MEMBER_FUNCTION_NAME(ThirdAlgorithm) + " part1 start, " << ctx.info() << std::endl;
    }
    nvtx3::unique_range range{MEMBER_FUNCTION_NAME(ThirdAlgorithm) + " part1" + ctx.info(), nvtxcolor(ctx.eventNumber), nvtx3::payload{ctx.eventNumber}};
    const int* input = nullptr;
    SC_CHECK_YIELD(ctx.eventStore.retrieve(input, dependencies()[0]));
    auto output = std::make_unique<int>(-1);
    SC_CHECK_YIELD(ctx.eventStore.record(std::move(output), products()[0]));

    if (m_verbose) {
        std::cout << MEMBER_FUNCTION_NAME(ThirdAlgorithm) + " part1, " << ctx.info() << std::endl;
    }
    auto * notif = new AlgorithmContext{ctx};
    CUDAThread::post([ctx, notif]() {
        launchTestKernel5(ctx.stream);
        cudaLaunchHostFunc(ctx.stream, AlgorithmContext::newScheduleResumeCallback, notif);
    });
    { auto r = std::move(range); } // End range
    co_yield StatusCode::SUCCESS;

    if (m_verbose) {
        std::cout << MEMBER_FUNCTION_NAME(ThirdAlgorithm) + " conclusion, " << ctx.info() << std::endl;
    }
    auto range2 = nvtx3::scoped_range{MEMBER_FUNCTION_NAME(ThirdAlgorithm) + " conclusion" + ctx.info(), nvtxcolor(ctx.eventNumber), nvtx3::payload{ctx.eventNumber}};
    co_return StatusCode::SUCCESS;
}

AlgorithmBase::AlgCoInterface ThirdAlgorithm::executeCachedGraph(AlgorithmContext ctx) const {
    if (m_verbose) {
        std::cout << MEMBER_FUNCTION_NAME(ThirdAlgorithm) + " part1 start, " << ctx.info() << std::endl;
    }
    nvtx3::unique_range range{MEMBER_FUNCTION_NAME(ThirdAlgorithm) + " part1" + ctx.info(), nvtxcolor(ctx.eventNumber), nvtx3::payload{ctx.eventNumber}};
    const int* input = nullptr;
    SC_CHECK_YIELD(ctx.eventStore.retrieve(input, dependencies()[0]));
    auto output = std::make_unique<int>(-1);
    SC_CHECK_YIELD(ctx.eventStore.record(std::move(output), products()[0]));

    if (m_verbose) {
        std::cout << MEMBER_FUNCTION_NAME(ThirdAlgorithm) + " part1, " << ctx.info() << std::endl;
    }
    m_graphContainer.launchGraph(ctx.stream, new AlgorithmContext{ctx});
    { auto r = std::move(range); } // End range
    co_yield StatusCode::SUCCESS;

    if (m_verbose) {
        std::cout << MEMBER_FUNCTION_NAME(ThirdAlgorithm) + " conclusion, " << ctx.info() << std::endl;
    }
    auto range2 = nvtx3::scoped_range{MEMBER_FUNCTION_NAME(ThirdAlgorithm) + " conclusion" + ctx.info(), nvtxcolor(ctx.eventNumber), nvtx3::payload{ctx.eventNumber}};
    co_return StatusCode::SUCCESS;
}

AlgorithmBase::AlgCoInterface ThirdAlgorithm::executeCachedGraphDelegated(AlgorithmContext ctx) const {
    if (m_verbose) {
        std::cout << MEMBER_FUNCTION_NAME(ThirdAlgorithm) + " part1 start, " << ctx.info() << std::endl;
    }
    nvtx3::unique_range range{MEMBER_FUNCTION_NAME(ThirdAlgorithm) + " part1" + ctx.info(), nvtxcolor(ctx.eventNumber), nvtx3::payload{ctx.eventNumber}};
    const int* input = nullptr;
    SC_CHECK_YIELD(ctx.eventStore.retrieve(input, dependencies()[0]));
    auto output = std::make_unique<int>(-1);
    SC_CHECK_YIELD(ctx.eventStore.record(std::move(output), products()[0]));

    if (m_verbose) {
        std::cout << MEMBER_FUNCTION_NAME(ThirdAlgorithm) + " part1, " << ctx.info() << std::endl;
    }
    m_graphContainer.launchGraphDelegated(ctx.stream, new AlgorithmContext{ctx});
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
