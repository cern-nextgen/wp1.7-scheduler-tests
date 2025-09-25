// Mimic "assert always".
#ifdef NDEBUG
#undef NDEBUG
#endif


#include <cassert>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>

#include <gtest/gtest.h>

#include "AlgorithmDependencyMap.hpp"
#include "EventContentManager.hpp"
#include "EventStore.hpp"
#include "StatusCode.hpp"
#include "MockAlgorithm.hpp"

TEST(EventContentManagerTest, Chain) {
    MockAlgorithm algA{{}, {"prodA"}};
    MockAlgorithm algB{{"prodA"}, {"prodB"}};
    MockAlgorithm algC{{"prodB"}, {"prodC"}};
    std::vector<std::reference_wrapper<AlgorithmBase>> chainAlgs{algA, algB, algC};
    AlgorithmDependencyMap depmap{chainAlgs};
    EventContentManager manager;
    manager.resize(depmap);
    const auto& depMap = manager.getDependentAndReadyAlgs(0, depmap);
    ASSERT_EQ(depMap.size(), 0);
    auto s = manager.setAlgExecuted(0, depmap); // Mark algA as executed
    ASSERT_TRUE(s);
    const auto& depMapAfterA = manager.getDependentAndReadyAlgs(0, depmap);
    //manager.dumpContents();
    ASSERT_EQ(depMapAfterA.size(), 1);
    ASSERT_EQ(depMapAfterA[0], 1); // algB should be ready
    s = manager.setAlgExecuted(1, depmap); // Mark algB as executed
    ASSERT_TRUE(s);
    const auto& depMapAfterB = manager.getDependentAndReadyAlgs(1, depmap);
    ASSERT_EQ(depMapAfterB.size(), 1);
    ASSERT_EQ(depMapAfterB[0], 2); // algC should be ready
    s = manager.setAlgExecuted(2, depmap); // Mark algC as executed
    ASSERT_TRUE(s);
    const auto& depMapAfterC = manager.getDependentAndReadyAlgs(2, depmap);
    ASSERT_EQ(depMapAfterC.size(), 0); // No more algorithms should be ready
}

TEST(EventContentManagerTest, MultipleDependencies) {
    
    // Index: 0=A, 1=B, 2=C, 3=D, 4=EMockAlgorithm algA{{}, {"prodA"}};
    constexpr int AlgAIdx = 0;
    constexpr int AlgBIdx = 1;
    constexpr int AlgCIdx = 2;
    constexpr int AlgDIdx = 3;
    constexpr int AlgEIdx = 4;

    MockAlgorithm algA{{}, {"prodA"}};
    MockAlgorithm algB{{"prodA", "prodE"}, {"prodB"}};
    MockAlgorithm algC{{"prodA"}, {"prodC"}};
    MockAlgorithm algD{{"prodC"}, {"prodD"}};
    MockAlgorithm algE{{"prodC", "prodD"}, {"prodE"}};
    // Note: direct dependants are:
    // A -> B, C
    // B ->
    // C -> D, E
    // D -> E
    // E -> B
    // So the execution order is: A -> C -> D -> E -> B
    std::vector<std::reference_wrapper<AlgorithmBase>> algs{algA, algB, algC, algD, algE};
    AlgorithmDependencyMap depmap{algs};
    EventContentManager ecm{};
    ecm.resize(depmap);

    // Helper lambda to check expected ready dependants, with file/line
    auto expect_ready = [&](int idx, std::vector<size_t> expected, const char* file, int line) {
        auto v = ecm.getDependentAndReadyAlgs(idx, depmap);
        std::sort(v.begin(), v.end());
        std::sort(expected.begin(), expected.end());
        std::ostringstream oss;
        oss << "[" << file << ":" << line << "] getDependantAndReadyAlgs(" << idx << ") expected: ";
        for (auto e : expected) oss << e << ",";
        oss << " got: ";
        for (auto e : v) oss << e << ",";
        ASSERT_EQ(v, expected) << oss.str();
    };
#define EXPECT_READY(idx, expected) expect_ready(idx, expected, __FILE__, __LINE__)

    // At start, only A is ready (no dependencies)
    ASSERT_TRUE(ecm.isAlgExecutable(AlgAIdx, depmap));
    ASSERT_FALSE(ecm.isAlgExecutable(AlgBIdx, depmap));
    ASSERT_FALSE(ecm.isAlgExecutable(AlgCIdx, depmap));
    ASSERT_FALSE(ecm.isAlgExecutable(AlgDIdx, depmap));
    ASSERT_FALSE(ecm.isAlgExecutable(AlgEIdx, depmap));
    EXPECT_READY(AlgAIdx, {});
    EXPECT_READY(AlgBIdx, {});
    EXPECT_READY(AlgCIdx, {});
    EXPECT_READY(AlgDIdx, {});
    EXPECT_READY(AlgEIdx, {});

    // Execute A
    ASSERT_TRUE(ecm.setAlgExecuted(AlgAIdx, depmap));
    // After A: C is ready, B and E still blocked
    ASSERT_TRUE(ecm.isAlgExecutable(AlgCIdx, depmap));
    ASSERT_FALSE(ecm.isAlgExecutable(AlgBIdx, depmap));
    ASSERT_FALSE(ecm.isAlgExecutable(AlgDIdx, depmap));
    ASSERT_FALSE(ecm.isAlgExecutable(AlgEIdx, depmap));
    EXPECT_READY(AlgAIdx, (std::vector<size_t>{AlgCIdx})); // A's dependants: C (B not yet ready)
    EXPECT_READY(AlgBIdx, {});
    EXPECT_READY(AlgCIdx, {});
    EXPECT_READY(AlgDIdx, {});
    EXPECT_READY(AlgEIdx, {});

    // Execute C
    ASSERT_TRUE(ecm.setAlgExecuted(AlgCIdx, depmap));
    // After C: D is ready, E still blocked, B still blocked
    ASSERT_TRUE(ecm.isAlgExecutable(AlgDIdx, depmap));
    ASSERT_FALSE(ecm.isAlgExecutable(AlgBIdx, depmap));
    ASSERT_FALSE(ecm.isAlgExecutable(AlgEIdx, depmap));
    EXPECT_READY(AlgAIdx, (std::vector<size_t>{AlgCIdx}));
    EXPECT_READY(AlgBIdx, {});
    EXPECT_READY(AlgCIdx, (std::vector<size_t>{AlgDIdx})); // C's dependants: D (E not yet ready)
    EXPECT_READY(AlgDIdx, {});
    EXPECT_READY(AlgEIdx, {});

    // Execute D
    ASSERT_TRUE(ecm.setAlgExecuted(AlgDIdx, depmap));
    // After D: E is ready, B still blocked
    ASSERT_TRUE(ecm.isAlgExecutable(AlgEIdx, depmap));
    ASSERT_FALSE(ecm.isAlgExecutable(AlgBIdx, depmap));
    EXPECT_READY(AlgAIdx, (std::vector<size_t>{AlgCIdx}));
    EXPECT_READY(AlgBIdx, {});
    EXPECT_READY(AlgCIdx, (std::vector<size_t>{AlgDIdx, AlgEIdx})); // C's dependants: D, E (now E is ready)
    EXPECT_READY(AlgDIdx, (std::vector<size_t>{AlgEIdx})); // D's dependant: E
    EXPECT_READY(AlgEIdx, {});

    // Execute E
    ASSERT_TRUE(ecm.setAlgExecuted(AlgEIdx, depmap));
    // After E: B is ready
    ASSERT_TRUE(ecm.isAlgExecutable(AlgBIdx, depmap));
    EXPECT_READY(AlgAIdx, (std::vector<size_t>{AlgBIdx, AlgCIdx})); // A's dependants: B, C (B is now ready)
    EXPECT_READY(AlgBIdx, {});
    EXPECT_READY(AlgCIdx, (std::vector<size_t>{AlgDIdx, AlgEIdx}));
    EXPECT_READY(AlgDIdx, (std::vector<size_t>{AlgEIdx}));
    EXPECT_READY(AlgEIdx, (std::vector<size_t>{AlgBIdx})); // E's dependant: B

    // Execute B
    ASSERT_TRUE(ecm.setAlgExecuted(AlgBIdx, depmap));
    // All done, all should be ready
    for (int i = 0; i < 5; ++i) {
        ASSERT_TRUE(ecm.isAlgExecutable(i, depmap));
    }

    // No changes after all executed
    EXPECT_READY(AlgAIdx, (std::vector<size_t>{AlgBIdx, AlgCIdx}));
    EXPECT_READY(AlgBIdx, {});
    EXPECT_READY(AlgCIdx, (std::vector<size_t>{AlgDIdx, AlgEIdx}));
    EXPECT_READY(AlgDIdx, (std::vector<size_t>{AlgEIdx}));
    EXPECT_READY(AlgEIdx, (std::vector<size_t>{AlgBIdx}));
#undef EXPECT_READY
}


int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
