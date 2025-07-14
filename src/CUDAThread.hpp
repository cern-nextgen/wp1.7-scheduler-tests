#pragma once

#include <tbb/task_arena.h>
#include <tbb/task_group.h>
#include <functional>
#include <mutex>

class CUDAThread {
public:
    // Post a callable to the single-threaded CUDA TBB arena
    template<typename F>
    static void post(F&& f) {
        auto& instance = getInstance();
        instance.m_arena.enqueue([&instance, f](){
            instance.m_group.run(std::move(f));
        });
    }

    ~CUDAThread() {
        m_group.wait();
    }

private:
    CUDAThread() : m_arena(1) {}

    static CUDAThread& getInstance() {
        static CUDAThread instance;
        return instance;
    }

    tbb::task_arena m_arena;
    tbb::task_group m_group;
};