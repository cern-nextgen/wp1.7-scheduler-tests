#include <cuda_runtime.h>
#include <cuda.h>
#include <iostream>
#include <tbb/task_arena.h>
#include <tbb/task_group.h>
#include <mutex>
#include <vector>
#include <string>

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

// Class encapsulating the original graph and its instantiation
class OriginalGraph {
public:
    OriginalGraph()
    {

        CUDA_ASSERT(cudaGraphCreate(&graph_, 0));

        int N=1;
        // Use a dummy pointer for initial graph construction (non null)
        float* dummy_ptr;
        CUDA_ASSERT(cudaMalloc(&dummy_ptr, N * sizeof(float)));

        // Memset node params (store as member)
        memsetParams_.dst = dummy_ptr;
        memsetParams_.value = 0;
        memsetParams_.pitch = 0;
        memsetParams_.elementSize = sizeof(float);
        memsetParams_.width = N;
        memsetParams_.height = 1;
        CUDA_ASSERT(cudaGraphAddMemsetNode(&memsetNode_, graph_, nullptr, 0, &memsetParams_));

        void* kernelArgsA[] = { &dummy_ptr, &N };
        void* kernelArgsB[] = { &dummy_ptr, &N };
        void* kernelArgsC[] = { &dummy_ptr, &N };

        // Kernel A node
        kernelNodeParamsA_.func = (void*)kernelA;
        kernelNodeParamsA_.gridDim = dim3((N + 255) / 256);
        kernelNodeParamsA_.blockDim = dim3(256);
        kernelNodeParamsA_.sharedMemBytes = 0;
        kernelNodeParamsA_.kernelParams = kernelArgsA;
        kernelNodeParamsA_.extra = nullptr;
        CUDA_ASSERT(cudaGraphAddKernelNode(&nodeA_, graph_, &memsetNode_, 1, &kernelNodeParamsA_));

        // Kernel B node
        kernelNodeParamsB_.func = (void*)kernelB;
        kernelNodeParamsB_.gridDim = dim3((N + 255) / 256);
        kernelNodeParamsB_.blockDim = dim3(256);
        kernelNodeParamsB_.sharedMemBytes = 0;
        kernelNodeParamsB_.kernelParams = kernelArgsB;
        kernelNodeParamsB_.extra = nullptr;
        CUDA_ASSERT(cudaGraphAddKernelNode(&nodeB_, graph_, &nodeA_, 1, &kernelNodeParamsB_));

        // Kernel C node
        kernelNodeParamsC_.func = (void*)kernelC;
        kernelNodeParamsC_.gridDim = dim3((N + 255) / 256);
        kernelNodeParamsC_.blockDim = dim3(256);
        kernelNodeParamsC_.sharedMemBytes = 0;
        kernelNodeParamsC_.kernelParams = kernelArgsC;
        kernelNodeParamsC_.extra = nullptr;
        CUDA_ASSERT(cudaGraphAddKernelNode(&nodeC_, graph_, &nodeB_, 1, &kernelNodeParamsC_));

        CUDA_ASSERT(cudaGraphInstantiate(&graphExec_, graph_, nullptr, nullptr, 0));
        CUDA_ASSERT(cudaFree(dummy_ptr));
    }

    ~OriginalGraph() {
        std::lock_guard<std::mutex> lock(mutex_);
        CUDA_ASSERT(cudaGraphExecDestroy(graphExec_));
        CUDA_ASSERT(cudaGraphDestroy(graph_));
    }

    // Launch the graph with new kernel arguments (customize with cudaGraphExec* functions)
    void launch(int N, cudaStream_t stream) {
        std::lock_guard<std::mutex> lock(mutex_);

        // Allocate memory asynchronously for this launch
        float* d_data = nullptr;
        CUDA_ASSERT(cudaMallocAsync((void**)&d_data, N * sizeof(float), stream));

        // Update memset params
        memsetParams_.dst = d_data;
        memsetParams_.width = N;
        CUDA_ASSERT(cudaGraphExecMemsetNodeSetParams(graphExec_, memsetNode_, &memsetParams_));

        // Update kernel params
        int Nnc = N;
        void* kernelArgsA[] = { &d_data, &Nnc };
        void* kernelArgsB[] = { &d_data, &Nnc };
        void* kernelArgsC[] = { &d_data, &Nnc };

        cudaKernelNodeParams params = kernelNodeParamsA_;
        params.kernelParams = kernelArgsA;
        params.gridDim = dim3((N + 255) / 256);
        CUDA_ASSERT(cudaGraphExecKernelNodeSetParams(graphExec_, nodeA_, &params));

        params = kernelNodeParamsB_;
        params.kernelParams = kernelArgsB;
        params.gridDim = dim3((N + 255) / 256);
        CUDA_ASSERT(cudaGraphExecKernelNodeSetParams(graphExec_, nodeB_, &params));

        params = kernelNodeParamsC_;
        params.kernelParams = kernelArgsC;
        params.gridDim = dim3((N + 255) / 256);
        CUDA_ASSERT(cudaGraphExecKernelNodeSetParams(graphExec_, nodeC_, &params));

        CUDA_ASSERT(cudaGraphLaunch(graphExec_, stream));

        CUDA_ASSERT(cudaFreeAsync(d_data, stream));
    }

private:
    cudaGraph_t graph_;
    cudaGraphExec_t graphExec_;
    cudaGraphNode_t memsetNode_;
    cudaGraphNode_t nodeA_, nodeB_, nodeC_;
    cudaMemsetParams memsetParams_{};
    cudaKernelNodeParams kernelNodeParamsA_{}, kernelNodeParamsB_{}, kernelNodeParamsC_{};
    std::mutex mutex_;
};

// Function to launch the graph on a given stream
void run_cloned_graph_on_stream(OriginalGraph& refGraph, int N, const char* label, cudaStream_t stream) {
    refGraph.launch(N, stream);
    {
        static std::mutex print_mutex;
        std::lock_guard<std::mutex> lock(print_mutex);
        std::cout << label << " Launched kernels A -> B -> C in succession." << std::endl;
    }
}

int main() {
    OriginalGraph refGraph;

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
                std::string label = "Customized graph (N=" + std::to_string(n * M) + ", stream=" + std::to_string(stream_idx) + ", call=" + std::to_string(i) + ")";
                run_cloned_graph_on_stream(refGraph, n * M, label.c_str(), streams[stream_idx]);
            });
        });
    }
    tg.wait();

    // Cleanup streams
    for (int i = 0; i < NUM_STREAMS; ++i) {
        cudaStreamSynchronize(streams[i]);
        CUDA_ASSERT(cudaStreamDestroy(streams[i]));
    }

    return 0;
}