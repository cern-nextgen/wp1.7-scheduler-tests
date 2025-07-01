#include "Scheduler.hpp"

#include <tbb/global_control.h>
#include <chrono> // For timing

#include <algorithm>
#include <cassert>
#include <iostream>

#include "AssertCuda.cuh"
#include "Coroutines.hpp"
#include "EventContext.hpp"
#include "EventStore.hpp"


Scheduler::Scheduler(int threads, int slots, ExecutionStrategy executionStrategy)
    : m_threads{threads},
      m_slots{slots},
      m_nextEvent{},
      m_remainingEvents{},
      m_arena{threads, 0},
      m_executionStrategy(executionStrategy) {
   // Set up a global limit on the number of threads.
   // TODO: explain why + 1
   tbb::global_control global_thread_limit(tbb::global_control::max_allowed_parallelism,
                                           m_threads + 1);
   EventStoreRegistry::initialize(slots);
}


void Scheduler::addAlgorithm(AlgorithmBase& alg) {
   if (m_runStarted) throw RuntimeError("In Scheduler::addAlgorithm(): Algorithms cannot be added after run start");
   m_algorithms.push_back(alg);
}

StatusCode Scheduler::run(int eventsToProcess, RunStats& stats) {
   // Lock and algorithm registration.
   if (!m_runStarted) {
        m_runStarted = true;

        // Initialize all the algorithms.
        if (StatusCode status = AlgorithmBase::for_all(m_algorithms, &AlgorithmBase::initialize);
            !status) {
            return status;
        }

        initSchedulerState();
    }

   // TODO: get rid of this parallel counting?
   m_remainingEvents.store(eventsToProcess);
   m_targetEventId += eventsToProcess;
   auto startTime = std::chrono::high_resolution_clock::now();

   action_type firstAction = [this]() { return update(); };
   if (StatusCode status = executeAction(firstAction); !status) {
      return status;
   }

   action_type action;
   while (m_remainingEvents.load() > 0) {
      m_actionQueue.pop(action);
      if (StatusCode status = executeAction(action); !status) {
         return status;
      }
   }

   tbb::task_group_status taskStatus = m_group.wait();

   auto endTime = std::chrono::high_resolution_clock::now();
   
   assert(taskStatus != tbb::task_group_status::canceled);

   if (StatusCode status = AlgorithmBase::for_all(m_algorithms, &AlgorithmBase::finalize);
       !status) {
      return status;
   }

   auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
   double rate = static_cast<double>(eventsToProcess) / (duration / 1000.0); // Events per second

   // Store run statistics
   stats.events = eventsToProcess;
   stats.rate = rate;
   stats.duration = duration;

   //std::cout << "Processed " << m_events << " events in " << duration << " ms (" << rate << " events/sec)" << std::endl;

   return StatusCode::SUCCESS;
}


Scheduler::~Scheduler() {
   for(auto& stream : m_streams) {
      ASSERT_CUDA(cudaStreamDestroy(stream));
   }
}


void Scheduler::setCudaSlotState(int slot, std::size_t alg, bool state) {
    m_slotStates[slot].algorithms[alg].cudaFinished = state;
}


void Scheduler::actionUpdate() {
   m_actionQueue.push([this]() -> StatusCode { return update(); });
}


void Scheduler::initSchedulerState() {
   m_nextEvent = 0;
   m_remainingEvents = 0;

   m_slotStates.clear();
   for(int i = 0; i < m_slots; ++i) {
      m_slotStates.push_back({m_nextEvent++});
      m_slotStates.back().algorithms.resize(m_algorithms.size());
      m_slotStates.back().eventManager = std::make_unique<EventContentManager>(m_algorithms);
   }

   m_streams.resize(m_slots);
   for(auto& stream : m_streams) {
      ASSERT_CUDA(cudaStreamCreate(&stream));
   }
}


StatusCode Scheduler::executeAction(action_type& f) {

   if(StatusCode status = f(); !status) {
      // Make sure all tasks finished.
      std::cout << "Action failed with status: " << status.what() << " . Dumping the event/algorithm table..." <<  std::endl;
      tbb::task_group_status taskStatus = m_group.wait();
      // TODO: why abort if tasks in the group were canceled?
      assert(taskStatus != tbb::task_group_status::canceled);
      printStatuses();
      return status;
   }
   return StatusCode::SUCCESS;
}


StatusCode Scheduler::update() {
   // Set up actions for launching the next algorithm in each event slot.
   // We iterate over indices as they are used to join slots and CUDA srtreams.
    if (!m_runStarted) {
        throw RuntimeError("In Scheduler::update(): Cannot update before run start");
    }
    for (int slot = 0; slot < m_slots; ++slot) {
        SlotState& slotState = m_slotStates[slot];

        if (std::ranges::all_of(slotState.algorithms,
                                [](const AlgorithmState& algo) { return algo.execState == AlgExecState::FINISHED; })) {
            EventContext ctx{slotState.eventNumber, slot, this, m_streams[slot]};
            EventStoreRegistry::of(ctx).clear();
            slotState.eventNumber = m_nextEvent++;
            std::ranges::for_each(slotState.algorithms, [](AlgorithmState& algo) {
                algo.cudaFinished = true;
                algo.execState = AlgExecState::UNSCHEDULED;
            });
            slotState.eventManager->reset();
        }

        if (slotState.eventNumber >= m_targetEventId) {
            continue;
        }

        for (std::size_t alg = 0; alg < m_algorithms.size(); ++alg) {
            auto& algoSt = slotState.algorithms[alg];
            switch (algoSt.execState.getState()) {
                case AlgExecState::SCHEDULED:
                case AlgExecState::FINISHED:
                    continue;
                case AlgExecState::ERROR:
                    m_actionQueue.push([st=algoSt.execState.getStatus()]() -> StatusCode { return st; });
                    return algoSt.execState.getStatus();
            }
            if (!slotState.eventManager->isAlgExecutable(alg)) {
                continue;
            }

            if (algoSt.execState == AlgExecState::SUSPENDED && !algoSt.cudaFinished) {
                continue;
            }

            assert((algoSt.execState == AlgExecState::UNSCHEDULED) ||
                   (algoSt.execState == AlgExecState::SUSPENDED && algoSt.cudaFinished));

            pushAction(slot, alg, slotState);
        }
    }
    return StatusCode::SUCCESS;
}


void Scheduler::pushAction(int slot, std::size_t ialg, SlotState& slotState) {
    auto& algoSt = slotState.algorithms[ialg];
    algoSt.execState = AlgExecState::SCHEDULED;

    m_arena.execute([this, ialg, slot, &slotState, &alg = m_algorithms[ialg].get()]() {
        m_group.run([this, ialg, slot, &slotState, &alg]() {
            auto& algoSt = slotState.algorithms[ialg];
            if (algoSt.coroutine.empty()) {
                // Do not resume the first time coroutine is launched because initial_suspend never
                // suspends.
                EventContext ctx{slotState.eventNumber, slot, this, m_streams[slot]};
                switch(m_executionStrategy) {
                    case ExecutionStrategy::SingleLaunch:
                        algoSt.coroutine = alg.execute(ctx);
                        break;
                    case ExecutionStrategy::StraightLaunches:
                        algoSt.coroutine = alg.executeStraight(ctx);
                        break;
                    case ExecutionStrategy::StraightDelegated:
                        algoSt.coroutine = alg.executeStraightDelegated(ctx);
                        break;
                    case ExecutionStrategy::StraightMutexed:
                        algoSt.coroutine = alg.executeStraightMutexed(ctx);
                        break;
                    case ExecutionStrategy::Graph:
                        algoSt.coroutine = alg.executeGraph(ctx);
                        break;
                    case ExecutionStrategy::CachedGraphs:
                        algoSt.coroutine = alg.executeCachedGraph(ctx);
                        break;
                    case ExecutionStrategy::CachedGraphsDelegated:
                        algoSt.coroutine = alg.executeCachedGraphDelegated(ctx);
                        break;
                    default:
                        std::cerr << "In Scheduler::pushAction(): Unknown execution strategy:" << to_string(m_executionStrategy) << std::endl;
                        abort();
                }
            } else {
                algoSt.coroutine.resume();
            }

            StatusCode algStatus;
            if (algoSt.coroutine.isResumable()) {
                algStatus = algoSt.coroutine.getYield();
            } else {
                algStatus = algoSt.coroutine.getReturn();
                if (!slotState.eventManager->setAlgExecuted(ialg)) {
                    algoSt.execState = AlgExecState::ERROR;
                }
            }

            // At the last algorithm in the event, decrement the remaining event counter.
            if (ialg == m_algorithms.size() - 1 && !algoSt.coroutine.isResumable()) {
                m_remainingEvents.fetch_sub(1);
            }

            algoSt.status = algStatus;
            if (!algStatus) {
                algoSt.execState = AlgExecState::ERROR;
                m_actionQueue.push([=]() -> StatusCode { return algStatus; });
            } else {
                if (algoSt.coroutine.isResumable()) {
                    algoSt.execState = AlgExecState::SUSPENDED;
                } else {
                    algoSt.execState = AlgExecState::FINISHED;
                    algoSt.coroutine.setEmpty();
                }
                m_actionQueue.push([this]() -> StatusCode { return update(); });
            }
        });
    });
}


void Scheduler::printStatuses() const {
   std::cout << std::endl;
   std::cout << std::endl;
   std::cout << "Printing all statuses" << std::endl;
    for (std::size_t i{0}; const auto& slotState : m_slotStates) {
        std::cout << "Slot number: " << i << ", event number: " << slotState.eventNumber << " -> ";
        for (std::size_t j{0}; const auto& algo : slotState.algorithms) {
            std::cout << "algorithm[" << j++ << "]: " << algo.status.what() << ", ";
        }
        std::cout << std::endl;
        ++i;
    }
   std::cout << "Remaining events: " << m_remainingEvents.load() << std::endl;
   std::cout << std::endl;
}
