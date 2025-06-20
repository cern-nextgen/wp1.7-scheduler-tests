#pragma once

#include "AlgorithmBase.hpp"

#include <cuda_runtime_api.h>

// Forward declarations
struct Notification;

class FirstAlgorithm : public AlgorithmBase {
public:
    // Constructor with verbose and error parameters
    FirstAlgorithm(bool errorEnabled = false, int errorEventId = -1, bool verbose = false);

    StatusCode initialize() override;
    AlgCoInterface execute(EventContext ctx) const override;
    AlgCoInterface executeGraph(EventContext ctx) const override;
    StatusCode finalize() override;

    ~FirstAlgorithm() override {
        std::lock_guard<std::mutex> lock(m_graphMutex);
        // Clean up CUDA graph resources
        if (m_graphExec) {
            cudaGraphExecDestroy(m_graphExec);
        }
        if (m_graph) {
            cudaGraphDestroy(m_graph);
        }
    }   

private:
    bool m_errorEnabled;  // Whether the error is enabled
    int m_errorEventId;   // Event ID where the error occurs
    bool m_verbose;       // Whether verbose output is enabled

    void launchGraph(cudaStream_t stream, Notification* notification) const;
    mutable cudaGraph_t m_graph;  // CUDA graph for the algorithm
    mutable cudaGraphExec_t m_graphExec;  // Executable CUDA graph  
    mutable cudaGraphNode_t m_kernel1Node;  // Node for the kernel in the graph
    mutable cudaGraphNode_t m_kernel2Node;  // Node for the second kernel in the graph
    mutable cudaGraphNode_t m_HostFunctionNode;  // Node for the host function in the graph
    mutable cudaKernelNodeParams m_kernel1Params{}, m_kernel2Params{};
    mutable cudaHostNodeParams m_hostFunctionParams{};
    mutable std::mutex m_graphMutex;  // Mutex for thread-safe access to the graph
};
