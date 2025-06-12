#include <cuda_runtime.h>
#include <cuda.h>
#include <iostream>

// Macro to assert that CUDA calls do not fail
#define CUDA_ASSERT(call) \
    do { \
        cudaError_t err = (call); \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": " \
                      << cudaGetErrorString(err) << std::endl; \
            std::exit(EXIT_FAILURE); \
        } \
    } while (0)

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
    // First buffer: 10MB of floats
    constexpr int N1 = 10 * 1024 * 1024;
    // Second buffer: 100 million floats (~400MB)
    constexpr int N2 = 100 * 1000 * 1000;

    // Allocate a separate buffer for each graph
    float* d_data1;
    float* d_data2;
    CUDA_ASSERT(cudaMalloc(&d_data1, N1 * sizeof(float)));
    CUDA_ASSERT(cudaMalloc(&d_data2, N2 * sizeof(float)));
    CUDA_ASSERT(cudaMemset(d_data1, 0, N1 * sizeof(float)));
    CUDA_ASSERT(cudaMemset(d_data2, 0, N2 * sizeof(float)));

    cudaStream_t stream1, stream2;
    CUDA_ASSERT(cudaStreamCreate(&stream1));
    CUDA_ASSERT(cudaStreamCreate(&stream2));

    // CUDA Graph setup for first buffer
    cudaGraph_t graph;
    cudaGraphExec_t graphExec;

    int Nnc1 = N1; // Non-const for kernel params

    void* kernelArgsA[] = { &d_data1, &Nnc1 };
    void* kernelArgsB[] = { &d_data1, &Nnc1 };
    void* kernelArgsC[] = { &d_data1, &Nnc1 };

    CUDA_ASSERT(cudaGraphCreate(&graph, 0));

    // Kernel A node
    cudaKernelNodeParams kernelNodeParamsA = {};
    kernelNodeParamsA.func = (void*)kernelA;
    kernelNodeParamsA.gridDim = dim3((N1 + 255) / 256);
    kernelNodeParamsA.blockDim = dim3(256);
    kernelNodeParamsA.sharedMemBytes = 0;
    kernelNodeParamsA.kernelParams = kernelArgsA;
    kernelNodeParamsA.extra = nullptr;

    cudaGraphNode_t nodeA;
    CUDA_ASSERT(cudaGraphAddKernelNode(&nodeA, graph, nullptr, 0, &kernelNodeParamsA));

    // Kernel B node
    cudaKernelNodeParams kernelNodeParamsB = {};
    kernelNodeParamsB.func = (void*)kernelB;
    kernelNodeParamsB.gridDim = dim3((N1 + 255) / 256);
    kernelNodeParamsB.blockDim = dim3(256);
    kernelNodeParamsB.sharedMemBytes = 0;
    kernelNodeParamsB.kernelParams = kernelArgsB;
    kernelNodeParamsB.extra = nullptr;

    cudaGraphNode_t nodeB;
    CUDA_ASSERT(cudaGraphAddKernelNode(&nodeB, graph, &nodeA, 1, &kernelNodeParamsB));

    // Kernel C node
    cudaKernelNodeParams kernelNodeParamsC = {};
    kernelNodeParamsC.func = (void*)kernelC;
    kernelNodeParamsC.gridDim = dim3((N1 + 255) / 256);
    kernelNodeParamsC.blockDim = dim3(256);
    kernelNodeParamsC.sharedMemBytes = 0;
    kernelNodeParamsC.kernelParams = kernelArgsC;
    kernelNodeParamsC.extra = nullptr;

    cudaGraphNode_t nodeC;
    CUDA_ASSERT(cudaGraphAddKernelNode(&nodeC, graph, &nodeB, 1, &kernelNodeParamsC));

    // Instantiate and launch the original graph on stream1
    CUDA_ASSERT(cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
    CUDA_ASSERT(cudaGraphLaunch(graphExec, stream1));

    // Clone the graph for the second buffer
    cudaGraph_t clonedGraph;
    CUDA_ASSERT(cudaGraphClone(&clonedGraph, graph));

    // Update kernel arguments for the cloned graph to use d_data2 and N2
    int Nnc2 = N2;
    void* kernelArgsA2[] = { &d_data2, &Nnc2 };
    void* kernelArgsB2[] = { &d_data2, &Nnc2 };
    void* kernelArgsC2[] = { &d_data2, &Nnc2 };

    // Instantiate the cloned graph
    cudaGraphExec_t clonedGraphExec;
    CUDA_ASSERT(cudaGraphInstantiate(&clonedGraphExec, clonedGraph, nullptr, nullptr, 0));

    // Update kernel node parameters in the cloned graph to use d_data2 and N2
    cudaGraphNode_t clonedNodes[3];
    size_t numNodes = 0;
    CUDA_ASSERT(cudaGraphGetNodes(clonedGraph, clonedNodes, &numNodes));
    // If you know the order, you can update directly:
    cudaKernelNodeParams params;
    CUDA_ASSERT(cudaGraphKernelNodeGetParams(clonedNodes[0], &params));
    params.kernelParams = kernelArgsA2;
    params.gridDim = dim3((N2 + 255) / 256);
    CUDA_ASSERT(cudaGraphKernelNodeSetParams(clonedNodes[0], &params));
    CUDA_ASSERT(cudaGraphKernelNodeGetParams(clonedNodes[1], &params));
    params.kernelParams = kernelArgsB2;
    params.gridDim = dim3((N2 + 255) / 256);
    CUDA_ASSERT(cudaGraphKernelNodeSetParams(clonedNodes[1], &params));
    CUDA_ASSERT(cudaGraphKernelNodeGetParams(clonedNodes[2], &params));
    params.kernelParams = kernelArgsC2;
    params.gridDim = dim3((N2 + 255) / 256);
    CUDA_ASSERT(cudaGraphKernelNodeSetParams(clonedNodes[2], &params));

    // Launch the cloned graph on stream2
    CUDA_ASSERT(cudaGraphLaunch(clonedGraphExec, stream2));

    // Print grid dimensions for the original graph
    std::cout << "Original graph grid dimensions:\n";
    std::cout << "  kernelA: " << kernelNodeParamsA.gridDim.x << " blocks\n";
    std::cout << "  kernelB: " << kernelNodeParamsB.gridDim.x << " blocks\n";
    std::cout << "  kernelC: " << kernelNodeParamsC.gridDim.x << " blocks\n";

    // Print grid dimensions for the cloned graph
    std::cout << "Cloned graph expoected grid dimensions:\n";
    std::cout << "  kernelA: " << ((N2 + 255) / 256) << " blocks\n";
    std::cout << "  kernelB: " << ((N2 + 255) / 256) << " blocks\n";
    std::cout << "  kernelC: " << ((N2 + 255) / 256) << " blocks\n";
    
    // Print grid dimensions extracted from the cloned graph
    std::cout << "Cloned graph grid dimensions (extracted from node params):\n";
    CUDA_ASSERT(cudaGraphKernelNodeGetParams(clonedNodes[0], &params));
    std::cout << "  kernelA: " << params.gridDim.x << " blocks\n";
    CUDA_ASSERT(cudaGraphKernelNodeGetParams(clonedNodes[1], &params));
    std::cout << "  kernelB: " << params.gridDim.x << " blocks\n";
    CUDA_ASSERT(cudaGraphKernelNodeGetParams(clonedNodes[2], &params));
    std::cout << "  kernelC: " << params.gridDim.x << " blocks\n";

    // Synchronize both streams at the end
    CUDA_ASSERT(cudaStreamSynchronize(stream1));
    std::cout << "Original CUDA Graph executed kernels A -> B -> C in succession on stream1 (10MB)." << std::endl;
    CUDA_ASSERT(cudaStreamSynchronize(stream2));
    std::cout << "Cloned CUDA Graph executed kernels A -> B -> C in succession on stream2 (100M floats)." << std::endl;

    // Cleanup
    CUDA_ASSERT(cudaGraphExecDestroy(graphExec));
    CUDA_ASSERT(cudaGraphDestroy(graph));
    CUDA_ASSERT(cudaGraphExecDestroy(clonedGraphExec));
    CUDA_ASSERT(cudaGraphDestroy(clonedGraph));
    CUDA_ASSERT(cudaStreamDestroy(stream1));
    CUDA_ASSERT(cudaStreamDestroy(stream2));
    CUDA_ASSERT(cudaFree(d_data1));
    CUDA_ASSERT(cudaFree(d_data2));

    return 0;
}