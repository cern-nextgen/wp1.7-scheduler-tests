#include "Scheduler.hpp"
#include "gtest/gtest.h"
#include "MockAlgorithm.hpp"
#include <ranges>

#pragma GCC optimize ("O0")

TEST(SchedulerTest, RegisterFiveAlgorithms) {
    Scheduler sched;
    MockAlgorithm algA{{}, {"prodA"}};
    MockAlgorithm algB{{"prodA"}, {"prodB"}};
    MockAlgorithm algC{{"prodB"}, {"prodC"}};
    MockAlgorithm algD{{"prodC"}, {"prodD"}};
    MockAlgorithm algE{{"prodD"}, {"prodE"}};
    sched.addAlgorithm(algA);
    sched.addAlgorithm(algB);
    sched.addAlgorithm(algC);
    sched.addAlgorithm(algD);
    sched.addAlgorithm(algE);
    // Check that 5 algorithms are registered
    
    ASSERT_EQ(sched.m_algorithms.size(), 5);
}

TEST(SchedulerTest, EventSlot) {
    MockAlgorithm::clear();
    MockAlgorithm algA{{}, {"prodA"}};
    std::vector<std::reference_wrapper<AlgorithmBase>> algorithms{{algA}};
    AlgorithmDependencyMap depMap{algorithms};
    Scheduler::EventSlot evSlot;
    evSlot.initialize(depMap, -1);
    ASSERT_EQ(evSlot.eventNumber, -1); // Default event number
    // evSlot.eventContentManager.dumpContents(depMap);
}

TEST(SchedulerTest, initSchedulerState) {
    MockAlgorithm::clear();
    Scheduler sched;
    MockAlgorithm algA{{}, {"prodA"}};
    MockAlgorithm algB{{"prodA"}, {"prodB"}};
    MockAlgorithm algC{{"prodB"}, {"prodC"}};
    MockAlgorithm algD{{"prodC"}, {"prodD"}};
    MockAlgorithm algE{{"prodD"}, {"prodE"}};
    sched.addAlgorithm(algA);
    sched.addAlgorithm(algB);
    sched.addAlgorithm(algC);
    sched.addAlgorithm(algD);
    sched.addAlgorithm(algE);
    sched.initSchedulerState(); // Initialize the scheduler state
    ASSERT_EQ(sched.m_algorithms.size(), 5);
    ASSERT_EQ(sched.m_eventSlotsNumber, 4); // Should be the default (4)
    ASSERT_EQ(sched.m_eventSlots.size(), 4); // Should be the default (4)
    for (auto i: std::ranges::iota_view(0, sched.m_eventSlotsNumber)) {
        auto & evSlot = sched.m_eventSlots[i];
        ASSERT_EQ(evSlot.algorithms.size(), 5);
        // At initialization, eventNumber should be equal to slot index
        ASSERT_EQ(evSlot.eventNumber, i);
    }
    // We need to finalize the algorithms as they were initialized
    assert (AlgorithmBase::for_all(sched.m_algorithms, &AlgorithmBase::finalize));
}


TEST(SchedulerTest, scheduleEvent) {
    MockAlgorithm::clear();
    Scheduler sched(10,30);
    MockAlgorithm algA{{}, {"prodA"}};
    MockAlgorithm algB{{"prodA"}, {"prodB"}};
    MockAlgorithm algC{{"prodB"}, {"prodC"}};
    MockAlgorithm algD{{"prodC"}, {"prodD"}};
    MockAlgorithm algE{{"prodD"}, {"prodE"}};
    sched.addAlgorithm(algA);
    sched.addAlgorithm(algB);
    sched.addAlgorithm(algC);
    sched.addAlgorithm(algD);
    sched.addAlgorithm(algE);

    Scheduler::RunStats stats;
    int nEvents = 100;
    StatusCode s;
    s = sched.run(nEvents, stats);

    
    // Check that all algorithms have been executed for each event
    auto lockedTracker = MockAlgorithm::getExecutionTracker();
    auto& executionTracker = lockedTracker.tracker;
    for (int eventNum = 0; eventNum < nEvents; ++eventNum) {
        for (std::size_t algNum = 0; algNum < sched.m_algorithms.size(); ++algNum) {
            auto it = executionTracker.find({eventNum, algNum});
            ASSERT_NE(it, executionTracker.end()) << "Algorithm " << algNum 
                << " was not executed for event " << eventNum;
        }
    }
    assert (sched.finalizeAlgorithms());
}

TEST(SchedulerTest, scheduleEventBranchedDependencies) {
    MockAlgorithm::clear();
    Scheduler sched(10,30);
    MockAlgorithm algA{{}, {"prodA"}};
    MockAlgorithm algB{{"prodA", "prodE"}, {"prodB"}};
    MockAlgorithm algC{{"prodA"}, {"prodC"}};
    MockAlgorithm algD{{"prodC"}, {"prodD"}};
    MockAlgorithm algE{{"prodC", "prodD"}, {"prodE"}};
    sched.addAlgorithm(algA);
    sched.addAlgorithm(algB);
    sched.addAlgorithm(algC);
    sched.addAlgorithm(algD);
    sched.addAlgorithm(algE);

    Scheduler::RunStats stats;
    int nEvents = 100;
    StatusCode s;
    s = sched.run(nEvents, stats);

    
    // Check that all algorithms have been executed for each event
    auto lockedTracker = MockAlgorithm::getExecutionTracker();
    auto& executionTracker = lockedTracker.tracker;
    for (int eventNum = 0; eventNum < nEvents; ++eventNum) {
        for (std::size_t algNum = 0; algNum < sched.m_algorithms.size(); ++algNum) {
            auto it = executionTracker.find({eventNum, algNum});
            ASSERT_NE(it, executionTracker.end()) << "Algorithm " << algNum 
                << " was not executed for event " << eventNum;
        }
    }
    assert (sched.finalizeAlgorithms());
}

TEST(SchedulerTest, scheduleSuspendingAlgo) {
    MockAlgorithm::clear();
    Scheduler sched(20,40);
    MockSuspendingAlgorithm algA{{}, {"prodA"}};
    MockAlgorithm algB{{"prodA", "prodE"}, {"prodB"}};
    MockSuspendingAlgorithm algC{{"prodA"}, {"prodC"}};
    MockAlgorithm algD{{"prodC"}, {"prodD"}};
    MockSuspendingAlgorithm algE{{"prodC", "prodD"}, {"prodE"}};
    sched.addAlgorithm(algA);
    sched.addAlgorithm(algB);
    sched.addAlgorithm(algC);
    sched.addAlgorithm(algD);
    sched.addAlgorithm(algE);

    Scheduler::RunStats stats;
    int nEvents = 100;
    StatusCode s;
    s = sched.run(nEvents, stats);
    assert(s);

    // Check that all algorithms have been executed for each event
    auto lockedTracker = MockAlgorithm::getExecutionTracker();
    auto& executionTracker = lockedTracker.tracker;
    for (int eventNum = 0; eventNum < nEvents; ++eventNum) {
        for (std::size_t algNum = 0; algNum < sched.m_algorithms.size(); ++algNum) {
            auto it = executionTracker.find({eventNum, algNum});
            ASSERT_NE(it, executionTracker.end()) << "Algorithm " << algNum 
                << " was not executed for event " << eventNum;
        }
    }
    assert (sched.finalizeAlgorithms());
}

TEST(SchedulerTest, scheduleSuspendingAllStrategies) {
    using ES = Scheduler::ExecutionStrategy;
    for (auto strategy : {ES::SingleLaunch,
                          ES::StraightLaunches,
                          ES::StraightDelegated,
                          ES::StraightMutexed,
                          ES::StraightThreadLocalStreams,
                          ES::StraightThreadLocalContext,
                          ES::Graph,
                          ES::GraphFullyDelegated,
                          ES::CachedGraphs,
                          ES::CachedGraphsDelegated
                        }) {
        // std::cout << "Testing strategy: " << Scheduler::to_string(strategy) << std::endl;
        MockAlgorithm::clear();
        Scheduler sched(20, 40, strategy);
        MockSuspendingAlgorithm algA{{}, {"prodA"}};
        MockAlgorithm algB{{"prodA", "prodE"}, {"prodB"}};
        MockSuspendingAlgorithm algC{{"prodA"}, {"prodC"}};
        MockAlgorithm algD{{"prodC"}, {"prodD"}};
        MockSuspendingAlgorithm algE{{"prodC", "prodD"}, {"prodE"}};
        sched.addAlgorithm(algA);
        sched.addAlgorithm(algB);
        sched.addAlgorithm(algC);
        sched.addAlgorithm(algD);
        sched.addAlgorithm(algE);

        Scheduler::RunStats stats;
        int nEvents = 100;
        StatusCode s;
        s = sched.run(nEvents, stats);
        assert(s);

        // Check that all algorithms have been executed for each event
        auto lockedTracker = MockAlgorithm::getExecutionTracker();
        auto& executionTracker = lockedTracker.tracker;
        for (int eventNum = 0; eventNum < nEvents; ++eventNum) {
            for (std::size_t algNum = 0; algNum < sched.m_algorithms.size(); ++algNum) {
                auto it = executionTracker.find({eventNum, algNum});
                ASSERT_NE(it, executionTracker.end()) << "Algorithm " << algNum 
                    << " was not executed for event " << eventNum;
            }
        }
        assert (sched.finalizeAlgorithms());
    }
}

TEST(SchedulerTest, scheduleEventLinearWithError) {
    MockAlgorithm::clear();
    Scheduler sched(1,1);
    MockAlgorithm algA{{}, {"prodA"}};
    MockAlgorithm algB{{"prodA"}, {"prodB"}};
    MockAlgorithm algC{{"prodB"}, {"prodC"}};
    MockAlgorithm algD{{"prodC"}, {"prodD"}};
    MockAlgorithm algE{{"prodD"}, {"prodE"}};
    std::size_t errorEvent = 42;
    std::size_t errorAlgorithm = 2; // algC
    algC.setInjectErrorAtEvent(errorEvent); // Inject an error at event 42
    sched.addAlgorithm(algA);
    sched.addAlgorithm(algB);
    sched.addAlgorithm(algC);
    sched.addAlgorithm(algD);
    sched.addAlgorithm(algE);

    Scheduler::RunStats stats;
    int nEvents = 100;
    StatusCode s;
    s = sched.run(nEvents, stats);

    
    // Check that all algorithms have been executed for each event
    auto lockedTracker = MockAlgorithm::getExecutionTracker();
    auto& executionTracker = lockedTracker.tracker;
    for (int eventNum = 0; eventNum < errorEvent; ++eventNum) {
        for (std::size_t algNum = 0; algNum < sched.m_algorithms.size(); ++algNum) {
            auto it = executionTracker.find({eventNum, algNum});
            ASSERT_NE(it, executionTracker.end()) << "Algorithm " << algNum 
                << " was not executed for event " << eventNum;
        }
    }
    for(std::size_t algNum = 0; algNum <= errorAlgorithm; ++algNum) {
        auto it = executionTracker.find({errorEvent, algNum});
        ASSERT_NE(it, executionTracker.end()) << "Algorithm " << algNum 
            << " was not executed for event " << errorEvent;
    }
    for(std::size_t algNum = errorAlgorithm + 1; algNum < sched.m_algorithms.size(); ++algNum) {
        auto it = executionTracker.find({errorEvent, algNum});
        ASSERT_EQ(it, executionTracker.end()) << "Algorithm " << algNum 
            << " was executed for event " << errorEvent;
    }
    for (int eventNum = errorEvent + 1; eventNum < nEvents; ++eventNum) {
        for (std::size_t algNum = 0; algNum < sched.m_algorithms.size(); ++algNum) {
            auto it = executionTracker.find({eventNum, algNum});
            ASSERT_EQ(it, executionTracker.end()) << "Algorithm " << algNum 
                << " was executed for event " << eventNum;
        }
    }
    assert (sched.finalizeAlgorithms());
}

