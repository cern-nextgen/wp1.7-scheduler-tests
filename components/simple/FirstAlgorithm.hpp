#pragma once

#include "AlgorithmBase.hpp"
#include "CachingGraphContainer.hpp"

#include <cuda_runtime_api.h>

// Forward declarations
struct Notification;

// Helper class for CUDA Graph logic
class FirstAlgorithmGraph {
public:
    FirstAlgorithmGraph();
    ~FirstAlgorithmGraph();

    void launchGraph(cudaStream_t stream, Notification* notification);
    void launchGraphDelegated(cudaStream_t stream, Notification* notification);

private:
    cudaGraph_t m_graph{};
    cudaGraphExec_t m_graphExec{};
    cudaGraphNode_t m_kernel1Node{};
    cudaGraphNode_t m_kernel2Node{};
    cudaGraphNode_t m_HostFunctionNode{};
    cudaKernelNodeParams m_kernel1Params{}, m_kernel2Params{};
    cudaHostNodeParams m_hostFunctionParams{};
    std::mutex m_graphMutex;
};

class FirstAlgorithm : public AlgorithmBase {
public:
    // Constructor with verbose and error parameters
    FirstAlgorithm(bool errorEnabled = false, int errorEventId = -1, bool verbose = false);

    StatusCode initialize() override;
    AlgCoInterface execute(EventContext ctx) const override;
    AlgCoInterface executeStraight(EventContext ctx) const override;
    AlgCoInterface executeStraightDelegated(EventContext ctx) const override;
    AlgCoInterface executeGraph(EventContext ctx) const override;
    AlgCoInterface executeCachedGraph(EventContext ctx) const override;
    AlgCoInterface executeCachedGraphDelegated(EventContext ctx) const override;
    StatusCode finalize() override;

private:
    bool m_errorEnabled;  // Whether the error is enabled
    int m_errorEventId;   // Event ID where the error occurs
    bool m_verbose;       // Whether verbose output is enabled
    mutable FirstAlgorithmGraph m_graphImpl; // Graph helper instance
    mutable CachingGraphContainer<FirstAlgorithmGraph> m_graphContainer; // Container for caching graph instances
};


