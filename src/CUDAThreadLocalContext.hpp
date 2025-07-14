#pragma once

#include <cuda.h>
#include <stdexcept>

class CUDAThreadLocalContext {
public:
    // Public static function to ensure the primary context is retained for this thread
    static void check() {
        instance().get();
    }

private:
    // Singleton accessor
    static CUDAThreadLocalContext& instance() {
        static CUDAThreadLocalContext s_instance;
        return s_instance;
    }

    // Returns a reference to the thread-local retained context handle
    CUcontext& get() {
        thread_local CUcontext ctx = [] {
            CUcontext local_ctx = nullptr;
            CUresult res = cuDevicePrimaryCtxRetain(&local_ctx, 0);
            if (res != CUDA_SUCCESS) {
                throw std::runtime_error("Failed to retain CUDA primary context");
            }
            return local_ctx;
        }();
        return ctx;
    }

    CUDAThreadLocalContext() = default;
    ~CUDAThreadLocalContext() = default;
    CUDAThreadLocalContext(const CUDAThreadLocalContext&) = delete;
    CUDAThreadLocalContext& operator=(const CUDAThreadLocalContext&) = delete;
  };