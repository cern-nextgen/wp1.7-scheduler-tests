#pragma once
#include "AlgorithmBase.hpp"
#include <vector>
#include <string>
#include <cassert>

/**
 * @brief A mock algorithm for testing purposes. Records algorithm execution. Can inject errors.
 */
class MockAlgorithm : public AlgorithmBase {
friend class MockSuspendingAlgorithm; // Allow the suspending variant to access private members
public:
    MockAlgorithm(const std::vector<std::string>& dependencies,
                 const std::vector<std::string>& products) {
        for (const auto& dep : dependencies) {
            auto s = addDependency<int>(dep);
            assert(s);
        }
        for (const auto& prod : products) {
            auto s = addProduct<int>(prod);
            assert(s);
        }
    }
private:
    bool m_Initialized = false;
    bool m_Finalized = false;
public:
    StatusCode initialize() override { 
        assert (!m_Initialized);
        m_Initialized = true;
        return StatusCode::SUCCESS; 
    }
    AlgCoInterface execute(AlgorithmContext ctx) const override {
        assert (m_Initialized);
        // Get hold of the dependencies
        for (const auto& dep : dependencies()) {
            const int* input = nullptr;
            SC_CHECK_CO_RETURN(ctx.eventStore.retrieve(input, dep));
            // std::cout << "MockAlgorithm event " << ctx.eventNumber << ", alg " << ctx.algorithmNumber << " retrieved dependency " << dep << " with value " << *input << "\n";
            (void)input; // Suppress unused variable warning
        }
        // Produce the products
        for (const auto& prod : products()) {
            auto output = std::make_unique<int>(-1);
            // std::cout << "MockAlgorithm event " << ctx.eventNumber << ", alg " << ctx.algorithmNumber << " recording product " << prod << "\n";
            SC_CHECK_CO_RETURN(ctx.eventStore.record(std::move(output), prod));
        }
        // Record the execution
        getExecutionTracker().tracker.insert({ctx.eventNumber, ctx.algorithmNumber});
        if (ctx.eventNumber == m_injectErrorAtEvent) {
            co_return StatusCode::FAILURE;
        }
        co_return {}; 
    }
    StatusCode finalize() override {
        assert (m_Initialized);
        assert (!m_Finalized);
        m_Finalized = true;
        return StatusCode::SUCCESS;
    }

    ~MockAlgorithm() {
        assert (!m_Initialized || m_Finalized);
    }

    // A record of the executed algorithms
    using ExecutionTracker = std::set<std::tuple<int, std::size_t>>;

    struct LockedTracker {
        std::scoped_lock<std::mutex> lock;
        ExecutionTracker& tracker;
    };

    static LockedTracker getExecutionTracker() {
        static std::mutex mutex;
        static ExecutionTracker executionTracker;
        return {std::scoped_lock<std::mutex>(mutex), executionTracker};
    }

    static void clear() {
        auto lockedTracker = getExecutionTracker();
        lockedTracker.tracker.clear();
    }
    
    // Inject errors,
    private:
    int m_injectErrorAtEvent = std::numeric_limits<int>::max();

    public:
    void setInjectErrorAtEvent(int event) { m_injectErrorAtEvent = event; }

    // TODO: add event store interface.
};

class MockSuspendingAlgorithm : public MockAlgorithm {
public:
    using MockAlgorithm::MockAlgorithm;
    AlgCoInterface execute(AlgorithmContext ctx) const override {
        // Get hold of the dependencies
        for (const auto& dep : dependencies()) {
            const int* input = nullptr;
            SC_CHECK_CO_RETURN(ctx.eventStore.retrieve(input, dep));
            // std::cout << "MockSuspendingAlgorithm event " << ctx.eventNumber << ", alg " << ctx.algorithmNumber << " retrieved dependency " << dep << " with value " << *input << "\n";
            (void)input; // Suppress unused variable warning
        }
        // Already inject resumption here (simulate CUDA callback)
        auto *c = new AlgorithmContext{ctx};
        AlgorithmContext::newScheduleResumeCallback(c);
        // Simulate suspension
        co_yield StatusCode::SUCCESS;
        // Produce the products
        for (const auto& prod : products()) {
            auto output = std::make_unique<int>(-1);
            // std::cout << "MockSuspendingAlgorithm event " << ctx.eventNumber << ", alg " << ctx.algorithmNumber << " recording product " << prod << "\n";
            SC_CHECK_CO_RETURN(ctx.eventStore.record(std::move(output), prod));
        }
        // Record the execution
       getExecutionTracker().tracker.insert({ctx.eventNumber, ctx.algorithmNumber});
       if (ctx.eventNumber == m_injectErrorAtEvent) {
           co_return StatusCode::FAILURE;
       }
       co_return {};
    }

};
