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
    void launchGraph(cudaStream_t stream, Notification* notification);
    void launchGraphDelegated(cudaStream_t stream, Notification* notification);
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
    AlgCoInterface execute(EventContext ctx) const override;
    // Exceute straight is identical to execute (it will fall back to execute())
    AlgCoInterface executeStraightDelegated(EventContext ctx) const override;
    AlgCoInterface executeStraightMutexed(EventContext ctx) const override;
    AlgCoInterface executeStraightThreadLocalStreams(EventContext ctx) const override;
    AlgCoInterface executeStraightThreadLocalContext(EventContext ctx) const override;
    AlgCoInterface executeGraph(EventContext ctx) const override;
    AlgCoInterface executeGraphFullyDelegated(EventContext ctx) const override;
    AlgCoInterface executeCachedGraph(EventContext ctx) const override;
    AlgCoInterface executeCachedGraphDelegated(EventContext ctx) const override;
    StatusCode finalize() override;

private:
    bool m_verbose; // Whether verbose output is enabled
    mutable ThirdAlgorithmGraph m_graphImpl; // Graph helper instance
    mutable CachingGraphContainer<ThirdAlgorithmGraph> m_graphContainer; // Container for caching graph instances
};
