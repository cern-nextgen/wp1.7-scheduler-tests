#pragma once

#include "AlgorithmBase.hpp"

#include <cuda_runtime_api.h>

// Forward declarations
struct Notification;

class SecondAlgorithm : public AlgorithmBase {
public:
    // Constructor with verbose parameter
    explicit SecondAlgorithm(bool verbose = false);

    StatusCode initialize() override;
    AlgCoInterface execute(EventContext ctx) const override;
    AlgCoInterface executeStraight(EventContext ctx) const override;
    AlgCoInterface executeGraph(EventContext ctx) const override;
    StatusCode finalize() override;

    ~SecondAlgorithm() override {
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
    bool m_verbose; // Whether verbose output is enabled

    // CUDA Graph members for graph launch
    void launchGraph(cudaStream_t stream, Notification* notification) const;
    mutable cudaGraph_t m_graph{};
    mutable cudaGraphExec_t m_graphExec{};
    mutable cudaGraphNode_t m_kernel3Node{};
    mutable cudaGraphNode_t m_kernel4Node{};
    mutable cudaGraphNode_t m_HostFunctionNode{};
    mutable cudaKernelNodeParams m_kernel3Params{}, m_kernel4Params{};
    mutable cudaHostNodeParams m_hostFunctionParams{};
    mutable std::mutex m_graphMutex;
};
