/**
 * @file TestTBBTaskSuspention.cpp
 * @brief Test for TBB task suspension as an alternative to coroutines.
 */

#include <tbb/task.h>
#include <tbb/task_group.h>
#include <tbb/task_arena.h>
#include <iostream>
#include <ranges>
#include <chrono>
using namespace std::chrono_literals;
#include "NVTXUtils.hpp"
using WP17Scheduler::NVTXUtils::nvtxcolor;
using nvtx3::scoped_range;

int main () {
    tbb::task_arena arena(4); // Create an arena with 4 threads

    arena.execute([] {
        tbb::task_group g;

        for (auto i: std::views::iota(0, 100)) {
            g.run([i]() {
                nvtx3::unique_range range{"Task " + std::to_string(i), nvtxcolor(i), nvtx3::payload{i}};
                // std::cout << "Task " << i << " started\n";
                std::this_thread::sleep_for(50ms + (i % 10) * 10ms); // Simulate work
                // std::cout << "Task " << i << " completed\n";
            });
        }

        g.wait(); // Wait for all tasks to complete
    });

    std::cout << "All tasks completed\n";
    return 0;
}

