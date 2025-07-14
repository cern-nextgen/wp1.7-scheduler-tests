#pragma once

#include <mutex>

class CUDAMutex {
public:
    // Lock and return a unique_lock (moveable)
    static std::unique_lock<std::mutex> lock() {
        return std::unique_lock<std::mutex>(instance().m_mutex);
    }

private:
    // Get the singleton instance
    static CUDAMutex& instance() {
        static CUDAMutex s_instance;
        return s_instance;
    }

    CUDAMutex() = default;
    ~CUDAMutex() = default;
    CUDAMutex(const CUDAMutex&) = delete;
    CUDAMutex& operator=(const CUDAMutex&) = delete;

    std::mutex m_mutex;
};