#pragma once

#include <vector>
#include <mutex>
#include <memory>
#include <forward_list>
#include <utility>

template <typename T>
class CachingGraphContainer {
public:
    CachingGraphContainer() = default;
    ~CachingGraphContainer() = default;

    // Launches a graph: finds a free or creates a new T, then calls launchGraph on it with forwarded args
    template <typename... Args>
    void launchGraph(Args&&... args) {
        size_t idx = 0;
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            if (!m_freeIndices.empty()) {
                idx = m_freeIndices.front();
                m_freeIndices.pop_front();
            } else {
                idx = m_objects.size();
                m_objects.emplace_back(std::make_unique<T>());
            }
        }

        // Call launchGraph outside the lock
        m_objects[idx]->launchGraph(std::forward<Args>(args)...);

        // Mark as free again
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            m_freeIndices.push_front(idx);
        }
    }

private:
    std::vector<std::unique_ptr<T>> m_objects;
    std::forward_list<size_t> m_freeIndices;
    std::mutex m_mutex;
};