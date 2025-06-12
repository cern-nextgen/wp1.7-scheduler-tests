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
    constexpr int N = 1024;
    constexpr int bytes = N * sizeof(float);

    float* d_data;
    int Nnc = N; // Number of elements in the array, non const for compilation
    cudaMalloc(&d_data, bytes);
    cudaMemset(d_data, 0, bytes);

    cudaStream_t stream1, stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);

    // CUDA Graph setup
    cudaGraph_t graph;
    cudaGraphExec_t graphExec;

    void* kernelArgsA[] = { &d_data, &Nnc };
    void* kernelArgsB[] = { &d_data, &Nnc };
    void* kernelArgsC[] = { &d_data, &Nnc };

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
    kernelNodeParamsB.gridDim = dim3((N + 255) / 256);
    kernelNodeParamsB.blockDim = dim3(256);
    kernelNodeParamsB.sharedMemBytes = 0;
    kernelNodeParamsB.kernelParams = kernelArgsB;
    kernelNodeParamsB.extra = nullptr;

    cudaGraphNode_t nodeB;
    cudaGraphAddKernelNode(&nodeB, graph, &nodeA, 1, &kernelNodeParamsB);

    // Kernel C node
    cudaKernelNodeParams kernelNodeParamsC = {};
    kernelNodeParamsC.func = (void*)kernelC;
    kernelNodeParamsC.gridDim = dim3((N + 255) / 256);
    kernelNodeParamsC.blockDim = dim3(256);
    kernelNodeParamsC.sharedMemBytes = 0;
    kernelNodeParamsC.kernelParams = kernelArgsC;
    kernelNodeParamsC.extra = nullptr;

    cudaGraphNode_t nodeC;
    cudaGraphAddKernelNode(&nodeC, graph, &nodeB, 1, &kernelNodeParamsC);

    // Instantiate and launch the original graph on stream1
    cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0);
    cudaGraphLaunch(graphExec, stream1);

   

    // Clone the graph
    cudaGraph_t clonedGraph;
    cudaGraphClone(&clonedGraph, graph);

    // Instantiate and launch the cloned graph on stream2
    cudaGraphExec_t clonedGraphExec;
    cudaGraphInstantiate(&clonedGraphExec, clonedGraph, nullptr, nullptr, 0);
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
    cudaFree(d_data);

    return 0;
}