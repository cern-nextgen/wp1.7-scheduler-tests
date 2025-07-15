#pragma once

#include "AlgorithmBase.hpp"
#include "CachingGraphContainer.hpp"

#include <cuda_runtime_api.h>

// Forward declarations
struct Notification;

class ThirdAlgorithmGraph {
public:
    ThirdAlgorithmGraph();
    ~ThirdAlgorithmGraph();
    void launchGraph(cudaStream_t stream, AlgorithmContext* context);
    void launchGraphDelegated(cudaStream_t stream, AlgorithmContext* context);
private:
    cudaGraph_t m_graph{};
    cudaGraphExec_t m_graphExec{};
    cudaGraphNode_t m_kernel5Node{};
    cudaGraphNode_t m_HostFunctionNode{};
    cudaKernelNodeParams m_kernel5Params{};
    cudaHostNodeParams m_hostFunctionParams{};
    std::mutex m_graphMutex;
};

class ThirdAlgorithm : public AlgorithmBase {
public:
    // Constructor with verbose parameter
    explicit ThirdAlgorithm(bool verbose = false);

    StatusCode initialize() override;
    AlgCoInterface execute(AlgorithmContext ctx) const override;
    // Exceute straight is identical to execute (it will fall back to execute())
    AlgCoInterface executeStraightDelegated(AlgorithmContext ctx) const override;
    AlgCoInterface executeStraightMutexed(AlgorithmContext ctx) const override;
    AlgCoInterface executeStraightThreadLocalStreams(AlgorithmContext ctx) const override;
    AlgCoInterface executeStraightThreadLocalContext(AlgorithmContext ctx) const override;
    AlgCoInterface executeGraph(AlgorithmContext ctx) const override;
    AlgCoInterface executeGraphFullyDelegated(AlgorithmContext ctx) const override;
    AlgCoInterface executeCachedGraph(AlgorithmContext ctx) const override;
    AlgCoInterface executeCachedGraphDelegated(AlgorithmContext ctx) const override;
    StatusCode finalize() override;

private:
    bool m_verbose; // Whether verbose output is enabled
    mutable ThirdAlgorithmGraph m_graphImpl; // Graph helper instance
    mutable CachingGraphContainer<ThirdAlgorithmGraph> m_graphContainer; // Container for caching graph instances
};
