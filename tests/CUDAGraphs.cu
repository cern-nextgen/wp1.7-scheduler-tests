#include <cuda_runtime.h>
#include <cuda.h>
#include <iostream>

// Example kernels copied from CudaKernels.cu
__global__ void kernelA(float* data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) data[idx] += 1.0f;
}

__global__ void kernelB(float* data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) data[idx] *= 2.0f;
}

__global__ void kernelC(float* data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) data[idx] -= 3.0f;
}

int main() {
    // 10MB of floats
    constexpr int N = 10 * 1024 * 1024;

    // Allocate a separate buffer for each graph
    float* d_data1;
    float* d_data2;
    cudaMalloc(&d_data1, N * sizeof(float));
    cudaMalloc(&d_data2, N * sizeof(float));
    cudaMemset(d_data1, 0, N * sizeof(float));
    cudaMemset(d_data2, 0, N * sizeof(float));

    cudaStream_t stream1, stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);

    // CUDA Graph setup for first buffer
    cudaGraph_t graph;
    cudaGraphExec_t graphExec;

    int Nnc = N; // Non-const for kernel params

    void* kernelArgsA[] = { &d_data1, &Nnc };
    void* kernelArgsB[] = { &d_data1, &Nnc };
    void* kernelArgsC[] = { &d_data1, &Nnc };

    cudaGraphCreate(&graph, 0);

    // Kernel A node
    cudaKernelNodeParams kernelNodeParamsA = {};
    kernelNodeParamsA.func = (void*)kernelA;
    kernelNodeParamsA.gridDim = dim3((Nnc + 255) / 256);
    kernelNodeParamsA.blockDim = dim3(256);
    kernelNodeParamsA.sharedMemBytes = 0;
    kernelNodeParamsA.kernelParams = kernelArgsA;
    kernelNodeParamsA.extra = nullptr;

    cudaGraphNode_t nodeA;
    cudaGraphAddKernelNode(&nodeA, graph, nullptr, 0, &kernelNodeParamsA);

    // Kernel B node
    cudaKernelNodeParams kernelNodeParamsB = {};
    kernelNodeParamsB.func = (void*)kernelB;
    kernelNodeParamsB.gridDim = dim3((Nnc + 255) / 256);
    kernelNodeParamsB.blockDim = dim3(256);
    kernelNodeParamsB.sharedMemBytes = 0;
    kernelNodeParamsB.kernelParams = kernelArgsB;
    kernelNodeParamsB.extra = nullptr;

    cudaGraphNode_t nodeB;
    cudaGraphAddKernelNode(&nodeB, graph, &nodeA, 1, &kernelNodeParamsB);

    // Kernel C node
    cudaKernelNodeParams kernelNodeParamsC = {};
    kernelNodeParamsC.func = (void*)kernelC;
    kernelNodeParamsC.gridDim = dim3((Nnc + 255) / 256);
    kernelNodeParamsC.blockDim = dim3(256);
    kernelNodeParamsC.sharedMemBytes = 0;
    kernelNodeParamsC.kernelParams = kernelArgsC;
    kernelNodeParamsC.extra = nullptr;

    cudaGraphNode_t nodeC;
    cudaGraphAddKernelNode(&nodeC, graph, &nodeB, 1, &kernelNodeParamsC);

    // Instantiate and launch the original graph on stream1
    cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0);
    cudaGraphLaunch(graphExec, stream1);

    // Clone the graph for the second buffer
    cudaGraph_t clonedGraph;
    cudaGraphClone(&clonedGraph, graph);

    // Update kernel arguments for the cloned graph to use d_data2
    void* kernelArgsA2[] = { &d_data2, &Nnc };
    void* kernelArgsB2[] = { &d_data2, &Nnc };
    void* kernelArgsC2[] = { &d_data2, &Nnc };

    // Instantiate the cloned graph
    cudaGraphExec_t clonedGraphExec;
    cudaGraphInstantiate(&clonedGraphExec, clonedGraph, nullptr, nullptr, 0);

    // Update kernel node parameters in the cloned graph to use d_data2
    cudaGraphNode_t clonedNodes[3];
    size_t numNodes = 0;
    cudaGraphGetNodes(clonedGraph, clonedNodes, &numNodes);
    // If you know the order, you can update directly:
    cudaKernelNodeParams params;
    cudaGraphKernelNodeGetParams(clonedNodes[0], &params);
    params.kernelParams = kernelArgsA2;
    cudaGraphKernelNodeSetParams(clonedNodes[0], &params);
    cudaGraphKernelNodeGetParams(clonedNodes[1], &params);
    params.kernelParams = kernelArgsB2;
    cudaGraphKernelNodeSetParams(clonedNodes[1], &params);
    cudaGraphKernelNodeGetParams(clonedNodes[2], &params);
    params.kernelParams = kernelArgsC2;
    cudaGraphKernelNodeSetParams(clonedNodes[2], &params);

    // Launch the cloned graph on stream2
    cudaGraphLaunch(clonedGraphExec, stream2);

    // Synchronize both streams at the end
    cudaStreamSynchronize(stream1);
    std::cout << "Original CUDA Graph executed kernels A -> B -> C in succession on stream1." << std::endl;
    cudaStreamSynchronize(stream2);
    std::cout << "Cloned CUDA Graph executed kernels A -> B -> C in succession on stream2." << std::endl;

    // Cleanup
    cudaGraphExecDestroy(graphExec);
    cudaGraphDestroy(graph);
    cudaGraphExecDestroy(clonedGraphExec);
    cudaGraphDestroy(clonedGraph);
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);
    cudaFree(d_data1);
    cudaFree(d_data2);

    return 0;
}