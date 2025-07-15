#pragma once

#include "AlgorithmBase.hpp"
#include "CachingGraphContainer.hpp"

#include <cuda_runtime_api.h>

// Forward declarations
struct Notification;

// Helper class for CUDA Graph logic
class SecondAlgorithmGraph {
public:
    SecondAlgorithmGraph();
    ~SecondAlgorithmGraph();

    void launchGraph(cudaStream_t stream, AlgorithmContext* notification);
    void launchGraphDelegated(cudaStream_t stream, AlgorithmContext* notification);

private:
    cudaGraph_t m_graph{};
    cudaGraphExec_t m_graphExec{};
    cudaGraphNode_t m_kernel3Node{};
    cudaGraphNode_t m_kernel4Node{};
    cudaGraphNode_t m_HostFunctionNode{};
    cudaKernelNodeParams m_kernel3Params{}, m_kernel4Params{};
    cudaHostNodeParams m_hostFunctionParams{};
    std::mutex m_graphMutex;
};

class SecondAlgorithm : public AlgorithmBase {
public:
    // Constructor with verbose parameter
    explicit SecondAlgorithm(bool verbose = false);

    StatusCode initialize() override;
    AlgCoInterface execute(AlgorithmContext ctx) const override;
    AlgCoInterface executeStraight(AlgorithmContext ctx) const override;
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
    mutable SecondAlgorithmGraph m_graphImpl; // Graph helper instance
    mutable CachingGraphContainer<SecondAlgorithmGraph> m_graphContainer; // Container for caching graph instances
};
