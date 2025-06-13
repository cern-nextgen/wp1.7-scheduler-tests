#include <cuda_runtime.h>
#include <cuda.h>
#include <iostream>
#include <tbb/task_arena.h>
#include <tbb/task_group.h>
#include <mutex>

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

// Function to allocate memory, clone and run a CUDA graph on a given stream
void run_cloned_graph_on_stream(const cudaGraph_t& refGraph, int N, const char* label, cudaStream_t stream) {
    // Clone the reference graph
    cudaGraph_t clonedGraph;
    CUDA_ASSERT(cudaGraphClone(&clonedGraph, refGraph));

    // Allocate memory node
    cudaMemAllocNodeParams allocParams = {};
    allocParams.poolProps.allocType = cudaMemAllocationTypePinned;
    allocParams.poolProps.location.type = cudaMemLocationTypeDevice;
    allocParams.poolProps.location.id = 0; // Use the default device
    allocParams.bytesize = N * sizeof(float);
    cudaGraphNode_t allocNode;
    CUDA_ASSERT(cudaGraphAddMemAllocNode(&allocNode, clonedGraph, nullptr, 0, &allocParams));

    // Memset node
    cudaMemsetParams memsetParams = {};
    memsetParams.dst = allocParams.dptr;
    memsetParams.value = 0;
    memsetParams.pitch = 0;
    memsetParams.elementSize = sizeof(float);
    memsetParams.width = N;
    memsetParams.height = 1;
    cudaGraphNode_t memsetNode;
    CUDA_ASSERT(cudaGraphAddMemsetNode(&memsetNode, clonedGraph, &allocNode, 1, &memsetParams));

    // Prepare kernel arguments
    int Nnc = N;
    void* kernelArgsA[] = { &allocParams.dptr, &Nnc };
    void* kernelArgsB[] = { &allocParams.dptr, &Nnc };
    void* kernelArgsC[] = { &allocParams.dptr, &Nnc };

    // Update kernel node parameters in the cloned graph to use d_data and N
    constexpr size_t constNumNodes = 5;
    cudaGraphNode_t clonedNodes[constNumNodes];
    size_t numNodes = constNumNodes;
    CUDA_ASSERT(cudaGraphGetNodes(clonedGraph, clonedNodes, &numNodes));

    // Set dependencies: memsetNode -> kernelA
    CUDA_ASSERT(cudaGraphAddDependencies(clonedGraph, &memsetNode, clonedNodes, 1));

    cudaKernelNodeParams params;
    CUDA_ASSERT(cudaGraphKernelNodeGetParams(clonedNodes[0], &params));
    params.kernelParams = kernelArgsA;
    params.gridDim = dim3((N + 255) / 256);
    CUDA_ASSERT(cudaGraphKernelNodeSetParams(clonedNodes[0], &params));
    CUDA_ASSERT(cudaGraphKernelNodeGetParams(clonedNodes[1], &params));
    params.kernelParams = kernelArgsB;
    params.gridDim = dim3((N + 255) / 256);
    CUDA_ASSERT(cudaGraphKernelNodeSetParams(clonedNodes[1], &params));
    CUDA_ASSERT(cudaGraphKernelNodeGetParams(clonedNodes[2], &params));
    params.kernelParams = kernelArgsC;
    params.gridDim = dim3((N + 255) / 256);
    CUDA_ASSERT(cudaGraphKernelNodeSetParams(clonedNodes[2], &params));

    // Instantiate and launch the cloned graph
    cudaGraphExec_t clonedGraphExec;
    CUDA_ASSERT(cudaGraphInstantiate(&clonedGraphExec, clonedGraph, nullptr, nullptr, 0));
    CUDA_ASSERT(cudaGraphLaunch(clonedGraphExec, stream));

    // Print grid dimensions extracted from the cloned graph
    {
        static std::mutex print_mutex;
        std::lock_guard<std::mutex> lock(print_mutex);
        std::cout << label << " grid dimensions (extracted from node params):\n";
        CUDA_ASSERT(cudaGraphKernelNodeGetParams(clonedNodes[0], &params));
        std::cout << "  kernelA: " << params.gridDim.x << " blocks\n";
        CUDA_ASSERT(cudaGraphKernelNodeGetParams(clonedNodes[1], &params));
        std::cout << "  kernelB: " << params.gridDim.x << " blocks\n";
        CUDA_ASSERT(cudaGraphKernelNodeGetParams(clonedNodes[2], &params));
        std::cout << "  kernelC: " << params.gridDim.x << " blocks\n";
    }

    {
        static std::mutex print_mutex;
        std::lock_guard<std::mutex> lock(print_mutex);
        std::cout << label << " executed kernels A -> B -> C in succession." << std::endl;
    }

    // Leak the cloned graph and its execution object for the moment
    // CUDA_ASSERT(cudaGraphExecDestroy(clonedGraphExec));
    // CUDA_ASSERT(cudaGraphDestroy(clonedGraph));
    // Memory will be freed by the graph's memory node
}

int main() {
    // Placeholder for the original graph
    constexpr int N1 = 1;

    // Allocate memory for the reference graph (not executed)
    float* d_data1;
    CUDA_ASSERT(cudaMalloc(&d_data1, N1 * sizeof(float)));
    CUDA_ASSERT(cudaMemset(d_data1, 0, N1 * sizeof(float)));

    // Prepare kernel arguments for the reference graph
    int Nnc1 = N1;
    void* kernelArgsA[] = { &d_data1, &Nnc1 };
    void* kernelArgsB[] = { &d_data1, &Nnc1 };
    void* kernelArgsC[] = { &d_data1, &Nnc1 };

    // Create the reference CUDA graph (but do not instantiate or launch it)
    cudaGraph_t graph;
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

    // Do NOT instantiate or launch the reference/original graph

    // Stream pool setup
    constexpr int NUM_STREAMS = 4;
    cudaStream_t streams[NUM_STREAMS];
    for (int i = 0; i < NUM_STREAMS; ++i) {
        CUDA_ASSERT(cudaStreamCreate(&streams[i]));
    }

    // Now run the cloned/customized graph in a TBB task arena with a task_group
    tbb::task_arena arena(NUM_STREAMS);
    tbb::task_group tg;
    constexpr size_t M = 1000 * 1000; // 1 million
    std::vector<size_t> Ns{ 1 , 5 , 10 , 15 , 18, 20 };

    // Arbitrary number of invocations
    constexpr int num_invocations = 100;
    for (int i = 0; i < num_invocations; ++i) {
        size_t n = Ns[i % Ns.size()];
        int stream_idx = i % NUM_STREAMS;
        arena.execute([&, n, stream_idx, i] {
            tg.run([&, n, stream_idx, i] {
                std::string label = "Cloned graph (N=" + std::to_string(n * M) + ", stream=" + std::to_string(stream_idx) + ", call=" + std::to_string(i) + ")";
                run_cloned_graph_on_stream(graph, n * M, label.c_str(), streams[stream_idx]);
            });
        });
    }
    tg.wait();

    // Cleanup streams
    for (int i = 0; i < NUM_STREAMS; ++i) {
        cudaStreamSynchronize(streams[i]);
        CUDA_ASSERT(cudaStreamDestroy(streams[i]));
    }

    // Cleanup reference graph and buffer
    CUDA_ASSERT(cudaGraphDestroy(graph));
    CUDA_ASSERT(cudaFree(d_data1));

    return 0;
}