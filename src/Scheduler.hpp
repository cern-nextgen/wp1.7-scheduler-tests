#pragma once


#include <cuda_runtime_api.h>
#include <oneapi/tbb/concurrent_queue.h>
#include <oneapi/tbb/task_arena.h>
#include <oneapi/tbb/task_group.h>

#include <atomic>
#include <cstdlib>
#include <functional>
#include <memory>
#include <vector>

#include "AlgExecState.hpp"
#include "AlgorithmBase.hpp"
#include "EventContentManager.hpp"
#include "StatusCode.hpp"


class Scheduler {
public:
   Scheduler(int events = 500, int threads = 4, int slots = 4);
   Scheduler(const Scheduler&) = delete;
   Scheduler& operator=(const Scheduler&) = delete;
   ~Scheduler();

   // Should this be a constant reference?
   void addAlgorithm(AlgorithmBase& alg);
   StatusCode run();
   void setCudaSlotState(int slot, std::size_t alg, bool state);
   void actionUpdate();

private:
   using action_type = std::function<StatusCode()>;

   // Internal state of the scheduler.
   // Note that SlotState is movable but not copyable since AlgCoInterface is not copyable.
   struct SlotState {
      int eventNumber = 0;
      std::vector<bool> cudaFinished{};
      std::vector<StatusCode> algStatuses{};
      std::vector<AlgExecState> algStates{};
      std::vector<AlgorithmBase::AlgCoInterface> coroutines{};
      std::unique_ptr<EventContentManager> eventManager{};
   };

   void initSchedulerState();
   StatusCode executeAction(action_type& f);
   StatusCode update();

   void pushAction(int slot, std::size_t ialg, SlotState& slotState);
   void printStatuses() const;

   int m_events;
   int m_threads;
   int m_slots;
   int m_nextEvent;
   std::atomic_int m_remainingEvents;

   std::vector<std::reference_wrapper<AlgorithmBase>> m_algorithms;

   std::vector<SlotState> m_slotStates;
   std::vector<cudaStream_t> m_streams;

   tbb::task_arena m_arena;
   tbb::task_group m_group;
   tbb::concurrent_bounded_queue<action_type> m_actions;
};
