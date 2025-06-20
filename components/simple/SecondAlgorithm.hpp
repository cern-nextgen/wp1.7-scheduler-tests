#pragma once

#include "AlgorithmBase.hpp"

#include <cuda_runtime_api.h>

// Forward declarations
struct Notification;

class SecondAlgorithmGraph {
public:
    SecondAlgorithmGraph();
    ~SecondAlgorithmGraph();
    void launchGraph(cudaStream_t stream, Notification* notification);
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
    AlgCoInterface execute(EventContext ctx) const override;
    AlgCoInterface executeStraight(EventContext ctx) const override;
    AlgCoInterface executeGraph(EventContext ctx) const override;
    StatusCode finalize() override;

private:
    bool m_verbose; // Whether verbose output is enabled
    mutable SecondAlgorithmGraph m_graphImpl; // Graph helper instance
};
