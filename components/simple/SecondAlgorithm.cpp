#include "SecondAlgorithm.hpp"

#include <iostream>
#include <mutex>
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
    
    m_hostFunctionParams.fn = AlgorithmContext::newScheduleResumeCallback;
    m_hostFunctionParams.userData = nullptr;
    CUDA_ASSERT(cudaGraphAddHostNode(&m_HostFunctionNode, m_graph, &m_kernel4Node, 1, &m_hostFunctionParams));
    
    CUDA_ASSERT(cudaGraphInstantiate(&m_graphExec, m_graph, nullptr, nullptr, 0));
}

SecondAlgorithmGraph::~SecondAlgorithmGraph() {
    std::lock_guard<std::mutex> lock(m_graphMutex);
    if (m_graphExec) cudaGraphExecDestroy(m_graphExec);
    if (m_graph) cudaGraphDestroy(m_graph);
}

void SecondAlgorithmGraph::launchGraph(cudaStream_t stream, AlgorithmContext* notification) {
    std::lock_guard<std::mutex> lock(m_graphMutex);

    // Only update host function userData for this launch
    m_hostFunctionParams.userData = notification;
    CUDA_ASSERT(cudaGraphExecHostNodeSetParams(m_graphExec, m_HostFunctionNode, &m_hostFunctionParams));

    CUDA_ASSERT(cudaGraphLaunch(m_graphExec, stream));
}

void SecondAlgorithmGraph::launchGraphDelegated(cudaStream_t stream, AlgorithmContext* notification) {
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

// --- SecondAlgorithm Implementation ---
SecondAlgorithm::SecondAlgorithm(bool verbose)
    : m_verbose(verbose) {
       // TODO: returen to initialize after changing algo API
      std::ignore = addDependency<int>("Object1");
      std::ignore = addProduct<int>("Object3");
    }

StatusCode SecondAlgorithm::initialize() {
    nvtx3::scoped_range range{MEMBER_FUNCTION_NAME(SecondAlgorithm)};
    if (m_verbose) {
        std::cout << MEMBER_FUNCTION_NAME(SecondAlgorithm) << std::endl;
    }
    return StatusCode::SUCCESS;
}

AlgorithmBase::AlgCoInterface SecondAlgorithm::execute(AlgorithmContext ctx) const {
    if (m_verbose) {
        std::cout << MEMBER_FUNCTION_NAME(SecondAlgorithm) + " part1 start, " << ctx.info() << std::endl;
    }
    nvtx3::unique_range range{MEMBER_FUNCTION_NAME(SecondAlgorithm) + " part1" + ctx.info(), nvtxcolor(ctx.eventNumber), nvtx3::payload{ctx.eventNumber}};
    const int* input = nullptr;
    SC_CHECK_YIELD(ctx.eventStore.retrieve(input, dependencies()[0]));
    auto output = std::make_unique<int>(-1);
    SC_CHECK_YIELD(ctx.eventStore.record(std::move(output), products()[0]));

    if (m_verbose) {
        std::cout << MEMBER_FUNCTION_NAME(SecondAlgorithm) + " part1, " << ctx.info() << std::endl;
    }
    launchTestKernel3(ctx.stream);
    cudaLaunchHostFunc(ctx.stream, AlgorithmContext::newScheduleResumeCallback, new AlgorithmContext{ctx});
    { auto r = std::move(range); } // End range
    co_yield StatusCode::SUCCESS;

    if (m_verbose) {
        std::cout << MEMBER_FUNCTION_NAME(SecondAlgorithm) + " part2 start, " << ctx.info() << std::endl;
    }
    nvtx3::unique_range range2{MEMBER_FUNCTION_NAME(SecondAlgorithm) + " part2" + ctx.info(), nvtxcolor(ctx.eventNumber), nvtx3::payload{ctx.eventNumber}};
    launchTestKernel4(ctx.stream);
    cudaLaunchHostFunc(ctx.stream, AlgorithmContext::newScheduleResumeCallback, new AlgorithmContext{ctx});
    { auto r2 = std::move(range2); } // End range
    co_yield StatusCode::SUCCESS;

    if (m_verbose) {
        std::cout << MEMBER_FUNCTION_NAME(SecondAlgorithm) + " conclusion, " << ctx.info() << std::endl;
    }
    auto range3 = nvtx3::scoped_range{MEMBER_FUNCTION_NAME(SecondAlgorithm) + " conclusion" + ctx.info(), nvtxcolor(ctx.eventNumber), nvtx3::payload{ctx.eventNumber}};
    co_return StatusCode::SUCCESS;
}

AlgorithmBase::AlgCoInterface SecondAlgorithm::executeStraight(AlgorithmContext ctx) const {
    if (m_verbose) {
        std::cout << MEMBER_FUNCTION_NAME(SecondAlgorithm) + " part1 start, " << ctx.info() << std::endl;
    }
    auto range1 = std::make_unique<nvtx3::unique_range>(MEMBER_FUNCTION_NAME(SecondAlgorithm) + " part1" + ctx.info(), nvtxcolor(ctx.eventNumber), nvtx3::payload{ctx.eventNumber});
    const int* input = nullptr;
    SC_CHECK_YIELD(ctx.eventStore.retrieve(input, dependencies()[0]));
    auto output = std::make_unique<int>(-1);
    SC_CHECK_YIELD(ctx.eventStore.record(std::move(output), products()[0]));

    if (m_verbose) {
        std::cout << MEMBER_FUNCTION_NAME(SecondAlgorithm) + " part1, " << ctx.info() << std::endl;
    }
    launchTestKernel3(ctx.stream);
    launchTestKernel4(ctx.stream);
    cudaLaunchHostFunc(ctx.stream, AlgorithmContext::newScheduleResumeCallback, new AlgorithmContext{ctx});
    range1.reset(); // End range
    co_yield StatusCode::SUCCESS;

    if (m_verbose) {
        std::cout << MEMBER_FUNCTION_NAME(SecondAlgorithm) + " conclusion, " << ctx.info() << std::endl;
    }
    auto range2 = nvtx3::scoped_range{MEMBER_FUNCTION_NAME(SecondAlgorithm) + " conclusion" + ctx.info(), nvtxcolor(ctx.eventNumber), nvtx3::payload{ctx.eventNumber}};
    co_return StatusCode::SUCCESS;
}

AlgorithmBase::AlgCoInterface SecondAlgorithm::executeStraightDelegated(AlgorithmContext ctx) const {
    if (m_verbose) {
        std::cout << MEMBER_FUNCTION_NAME(SecondAlgorithm) + " part1 start, " << ctx.info() << std::endl;
    }
    auto range1 = std::make_unique<nvtx3::unique_range>(MEMBER_FUNCTION_NAME(SecondAlgorithm) + " part1" + ctx.info(), nvtxcolor(ctx.eventNumber), nvtx3::payload{ctx.eventNumber});
    const int* input = nullptr;
    SC_CHECK_YIELD(ctx.eventStore.retrieve(input, dependencies()[0]));
    auto output = std::make_unique<int>(-1);
    SC_CHECK_YIELD(ctx.eventStore.record(std::move(output), products()[0]));

    if (m_verbose) {
        std::cout << MEMBER_FUNCTION_NAME(SecondAlgorithm) + " part1, " << ctx.info() << std::endl;
    }
    // Launch kernels in a single thread to avoid performance drop
    auto * notif = new AlgorithmContext{ctx};
    CUDAThread::post([ctx, notif]() {
        launchTestKernel3(ctx.stream);
        launchTestKernel4(ctx.stream);
        cudaLaunchHostFunc(ctx.stream, AlgorithmContext::newScheduleResumeCallback, notif);
    });
    range1.reset(); // End range
    co_yield StatusCode::SUCCESS;

    if (m_verbose) {
        std::cout << MEMBER_FUNCTION_NAME(SecondAlgorithm) + " conclusion, " << ctx.info() << std::endl;
    }
    auto range2 = nvtx3::scoped_range{MEMBER_FUNCTION_NAME(SecondAlgorithm) + " conclusion" + ctx.info(), nvtxcolor(ctx.eventNumber), nvtx3::payload{ctx.eventNumber}};
    co_return StatusCode::SUCCESS;
}

AlgorithmBase::AlgCoInterface SecondAlgorithm::executeStraightMutexed(AlgorithmContext ctx) const {
    if (m_verbose) {
        std::cout << MEMBER_FUNCTION_NAME(SecondAlgorithm) + " part1 start, " << ctx.info() << std::endl;
    }
    auto range1 = std::make_unique<nvtx3::unique_range>(MEMBER_FUNCTION_NAME(SecondAlgorithm) + " part1" + ctx.info(), nvtxcolor(ctx.eventNumber), nvtx3::payload{ctx.eventNumber});
    const int* input = nullptr;
    SC_CHECK_YIELD(ctx.eventStore.retrieve(input, dependencies()[0]));
    auto output = std::make_unique<int>(-1);
    SC_CHECK_YIELD(ctx.eventStore.record(std::move(output), products()[0]));

    if (m_verbose) {
        std::cout << MEMBER_FUNCTION_NAME(SecondAlgorithm) + " part1, " << ctx.info() << std::endl;
    }
    auto cudaLock = CUDAMutex::lock();
    launchTestKernel3(ctx.stream);
    launchTestKernel4(ctx.stream);
    cudaLaunchHostFunc(ctx.stream, AlgorithmContext::newScheduleResumeCallback, new AlgorithmContext{ctx});
    cudaLock.unlock();
    range1.reset(); // End range
    co_yield StatusCode::SUCCESS;

    if (m_verbose) {
        std::cout << MEMBER_FUNCTION_NAME(SecondAlgorithm) + " conclusion, " << ctx.info() << std::endl;
    }
    auto range2 = nvtx3::scoped_range{MEMBER_FUNCTION_NAME(SecondAlgorithm) + " conclusion" + ctx.info(), nvtxcolor(ctx.eventNumber), nvtx3::payload{ctx.eventNumber}};
    co_return StatusCode::SUCCESS;
}

AlgorithmBase::AlgCoInterface SecondAlgorithm::executeStraightThreadLocalStreams(AlgorithmContext ctx) const {
    auto stream = CUDAThreadLocalStream::get();
    if (m_verbose) {
        std::cout << MEMBER_FUNCTION_NAME(SecondAlgorithm) + " part1 start, " << ctx.info() << std::endl;
    }
    auto range1 = std::make_unique<nvtx3::unique_range>(MEMBER_FUNCTION_NAME(SecondAlgorithm) + " part1" + ctx.info() + " stream=" + std::to_string((uint64_t)stream), nvtxcolor(ctx.eventNumber), nvtx3::payload{ctx.eventNumber});
    const int* input = nullptr;
    SC_CHECK_YIELD(ctx.eventStore.retrieve(input, dependencies()[0]));
    auto output = std::make_unique<int>(-1);
    SC_CHECK_YIELD(ctx.eventStore.record(std::move(output), products()[0]));

    if (m_verbose) {
        std::cout << MEMBER_FUNCTION_NAME(SecondAlgorithm) + " part1, " << ctx.info() << std::endl;
    }
    launchTestKernel3(stream);
    launchTestKernel4(stream);
    cudaLaunchHostFunc(stream, AlgorithmContext::newScheduleResumeCallback, new AlgorithmContext{ctx});
    range1.reset(); // End range
    co_yield StatusCode::SUCCESS;

    if (m_verbose) {
        std::cout << MEMBER_FUNCTION_NAME(SecondAlgorithm) + " conclusion, " << ctx.info() << std::endl;
    }
    auto range2 = nvtx3::scoped_range{MEMBER_FUNCTION_NAME(SecondAlgorithm) + " conclusion" + ctx.info(), nvtxcolor(ctx.eventNumber), nvtx3::payload{ctx.eventNumber}};
    co_return StatusCode::SUCCESS;
}

AlgorithmBase::AlgCoInterface SecondAlgorithm::executeStraightThreadLocalContext(AlgorithmContext ctx) const {
    CUDAThreadLocalContext::check();
    if (m_verbose) {
        std::cout << MEMBER_FUNCTION_NAME(SecondAlgorithm) + " part1 start, " << ctx.info() << std::endl;
    }
    auto range1 = std::make_unique<nvtx3::unique_range>(MEMBER_FUNCTION_NAME(SecondAlgorithm) + " part1" + ctx.info(), nvtxcolor(ctx.eventNumber), nvtx3::payload{ctx.eventNumber});
    const int* input = nullptr;
    SC_CHECK_YIELD(ctx.eventStore.retrieve(input, dependencies()[0]));
    auto output = std::make_unique<int>(-1);
    SC_CHECK_YIELD(ctx.eventStore.record(std::move(output), products()[0]));

    if (m_verbose) {
        std::cout << MEMBER_FUNCTION_NAME(SecondAlgorithm) + " part1, " << ctx.info() << std::endl;
    }
    launchTestKernel3(ctx.stream);
    launchTestKernel4(ctx.stream);
    cudaLaunchHostFunc(ctx.stream, AlgorithmContext::newScheduleResumeCallback, new AlgorithmContext{ctx});
    range1.reset(); // End range
    co_yield StatusCode::SUCCESS;

    if (m_verbose) {
        std::cout << MEMBER_FUNCTION_NAME(SecondAlgorithm) + " conclusion, " << ctx.info() << std::endl;
    }
    auto range2 = nvtx3::scoped_range{MEMBER_FUNCTION_NAME(SecondAlgorithm) + " conclusion" + ctx.info(), nvtxcolor(ctx.eventNumber), nvtx3::payload{ctx.eventNumber}};
    co_return StatusCode::SUCCESS;
}

AlgorithmBase::AlgCoInterface SecondAlgorithm::executeGraph(AlgorithmContext ctx) const {
    if (m_verbose) {
        std::cout << MEMBER_FUNCTION_NAME(SecondAlgorithm) + " [graph] part1 start, " << ctx.info() << std::endl;
    }
    nvtx3::unique_range range{MEMBER_FUNCTION_NAME(SecondAlgorithm) + " [graph] part1" + ctx.info(), nvtxcolor(ctx.eventNumber), nvtx3::payload{ctx.eventNumber}};
    const int* input = nullptr;
    SC_CHECK_YIELD(ctx.eventStore.retrieve(input, dependencies()[0]));
    auto output = std::make_unique<int>(-1);
    SC_CHECK_YIELD(ctx.eventStore.record(std::move(output), products()[0]));

    if (m_verbose) {
        std::cout << MEMBER_FUNCTION_NAME(SecondAlgorithm) + " [graph] launching CUDA graph, " << ctx.info() << std::endl;
    }

    // Allocate a notification for the host node
    auto* notification = new AlgorithmContext{ctx};
    m_graphImpl.launchGraph(ctx.stream, notification);

    { auto r = std::move(range); } // End range
    co_yield StatusCode::SUCCESS;

    if (m_verbose) {
        std::cout << MEMBER_FUNCTION_NAME(SecondAlgorithm) + " [graph] conclusion, " << ctx.info() << std::endl;
    }
    auto range3 = nvtx3::scoped_range{MEMBER_FUNCTION_NAME(SecondAlgorithm) + " [graph] conclusion" + ctx.info(), nvtxcolor(ctx.eventNumber), nvtx3::payload{ctx.eventNumber}};
    co_return StatusCode::SUCCESS;
}

AlgorithmBase::AlgCoInterface SecondAlgorithm::executeGraphFullyDelegated(AlgorithmContext ctx) const {
    if (m_verbose) {
        std::cout << MEMBER_FUNCTION_NAME(SecondAlgorithm) + " [graph] part1 start, " << ctx.info() << std::endl;
    }
    nvtx3::unique_range range{MEMBER_FUNCTION_NAME(SecondAlgorithm) + " [graph] part1" + ctx.info(), nvtxcolor(ctx.eventNumber), nvtx3::payload{ctx.eventNumber}};
    const int* input = nullptr;
    SC_CHECK_YIELD(ctx.eventStore.retrieve(input, dependencies()[0]));
    auto output = std::make_unique<int>(-1);
    SC_CHECK_YIELD(ctx.eventStore.record(std::move(output), products()[0]));

    if (m_verbose) {
        std::cout << MEMBER_FUNCTION_NAME(SecondAlgorithm) + " [graph] launching CUDA graph, " << ctx.info() << std::endl;
    }

    // Allocate a notification for the host node
    auto* notif = new AlgorithmContext{ctx};
    CUDAThread::post([ctx, notif, this]() {
        m_graphContainer.launchGraph(ctx.stream, notif);
    });

    { auto r = std::move(range); } // End range
    co_yield StatusCode::SUCCESS;

    if (m_verbose) {
        std::cout << MEMBER_FUNCTION_NAME(SecondAlgorithm) + " [graph] conclusion, " << ctx.info() << std::endl;
    }
    auto range3 = nvtx3::scoped_range{MEMBER_FUNCTION_NAME(SecondAlgorithm) + " [graph] conclusion" + ctx.info(), nvtxcolor(ctx.eventNumber), nvtx3::payload{ctx.eventNumber}};
    co_return StatusCode::SUCCESS;
}

AlgorithmBase::AlgCoInterface SecondAlgorithm::executeCachedGraph(AlgorithmContext ctx) const {
    if (m_verbose) {
        std::cout << MEMBER_FUNCTION_NAME(SecondAlgorithm) + " [graph] part1 start, " << ctx.info() << std::endl;
    }
    nvtx3::unique_range range{MEMBER_FUNCTION_NAME(SecondAlgorithm) + " [graph] part1" + ctx.info(), nvtxcolor(ctx.eventNumber), nvtx3::payload{ctx.eventNumber}};
    const int* input = nullptr;
    SC_CHECK_YIELD(ctx.eventStore.retrieve(input, dependencies()[0]));
    auto output = std::make_unique<int>(-1);
    SC_CHECK_YIELD(ctx.eventStore.record(std::move(output), products()[0]));

    if (m_verbose) {
        std::cout << MEMBER_FUNCTION_NAME(SecondAlgorithm) + " [graph] launching CUDA graph, " << ctx.info() << std::endl;
    }

    // Allocate a notification for the host node
    auto* notification = new AlgorithmContext{ctx};
    m_graphContainer.launchGraph(ctx.stream, notification);

    { auto r = std::move(range); } // End range
    co_yield StatusCode::SUCCESS;

    if (m_verbose) {
        std::cout << MEMBER_FUNCTION_NAME(SecondAlgorithm) + " [graph] conclusion, " << ctx.info() << std::endl;
    }
    auto range3 = nvtx3::scoped_range{MEMBER_FUNCTION_NAME(SecondAlgorithm) + " [graph] conclusion" + ctx.info(), nvtxcolor(ctx.eventNumber), nvtx3::payload{ctx.eventNumber}};
    co_return StatusCode::SUCCESS;
}

AlgorithmBase::AlgCoInterface SecondAlgorithm::executeCachedGraphDelegated(AlgorithmContext ctx) const {
    if (m_verbose) {
        std::cout << MEMBER_FUNCTION_NAME(SecondAlgorithm) + " [graph] part1 start, " << ctx.info() << std::endl;
    }
    nvtx3::unique_range range{MEMBER_FUNCTION_NAME(SecondAlgorithm) + " [graph] part1" + ctx.info(), nvtxcolor(ctx.eventNumber), nvtx3::payload{ctx.eventNumber}};
    const int* input = nullptr;
    SC_CHECK_YIELD(ctx.eventStore.retrieve(input, dependencies()[0]));
    auto output = std::make_unique<int>(-1);
    SC_CHECK_YIELD(ctx.eventStore.record(std::move(output), products()[0]));

    if (m_verbose) {
        std::cout << MEMBER_FUNCTION_NAME(SecondAlgorithm) + " [graph] launching CUDA graph, " << ctx.info() << std::endl;
    }

    // Allocate a notification for the host node
    auto* notification = new AlgorithmContext{ctx};
    m_graphContainer.launchGraphDelegated(ctx.stream, notification);

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
