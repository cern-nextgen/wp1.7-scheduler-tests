#pragma once

#include <cuda_runtime_api.h>

class CUDAThreadLocalStream {
public:
    // Helper class to manage stream lifetime
    class StreamHolder {
    public:
        StreamHolder() {
            cudaStreamCreate(&m_stream);
        }
        ~StreamHolder() {
            if (m_stream) {
                cudaStreamDestroy(m_stream);
            }
        }
        cudaStream_t get() { return m_stream; }
    private:
        cudaStream_t m_stream = nullptr;
    };

    // Returns a reference to the thread-local StreamHolder
    static StreamHolder& getHolder() {
        thread_local StreamHolder holder;
        return holder;
    }

    // Returns the thread-local CUDA stream
    static cudaStream_t get() {
        return getHolder().get();
    }
};