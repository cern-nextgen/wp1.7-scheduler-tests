#include "Scheduler.hpp"

#include <tbb/global_control.h>

#include <algorithm>
#include <cassert>
#include <iostream>

#include "AssertCuda.cuh"
#include "Coroutines.hpp"
#include "EventContext.hpp"
#include "EventStore.hpp"


Scheduler::Scheduler(int events, int threads, int slots)
    : m_events{events},
      m_threads{threads},
      m_slots{slots},
      m_nextEvent{},
      m_remainingEvents{},
      m_arena{threads, 0} {
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


StatusCode Scheduler::run() {
   // Lock algorithm registration.
   m_runStarted = true;

   // Initialize all the algorithms.
   if(StatusCode status = AlgorithmBase::for_all(m_algorithms, &AlgorithmBase::initialize);
      !status) {
      return status;
   }

   initSchedulerState();

   // Schedule the first set of algorithms.
   action_type firstAction = [this]() { return update(); };
   if(StatusCode status = executeAction(firstAction); !status) {
      return status;
   }

   // Execute "actions" until all events are processed.
   action_type action;
   while(m_remainingEvents.load() > 0) {
      m_actionQueue.pop(action);
      if(StatusCode status = executeAction(action); !status) {
         return status;
      }
   }

   // Make sure all tasks finish.
   tbb::task_group_status taskStatus = m_group.wait();
   assert(taskStatus != tbb::task_group_status::canceled);

   // Finalize all the algorithms.
   if(StatusCode status = AlgorithmBase::for_all(m_algorithms, &AlgorithmBase::finalize);
      !status) {
      return status;
   }

   return StatusCode::SUCCESS;
}


Scheduler::~Scheduler() {
   for(auto& stream : m_streams) {
      ASSERT_CUDA(cudaStreamDestroy(stream));
   }
}


void Scheduler::setCudaSlotState(int slot, std::size_t alg, bool state) {
   m_slotStates[slot].cudaFinished[alg] = state;
}


void Scheduler::actionUpdate() {
   m_actionQueue.push([this]() -> StatusCode { return update(); });
}


void Scheduler::initSchedulerState() {
   m_nextEvent = 0;
   m_remainingEvents = m_events;

   m_slotStates.clear();
   for(int i = 0; i < m_slots; ++i) {
      m_slotStates.push_back({m_nextEvent++});
      m_slotStates.back().cudaFinished.assign(m_algorithms.size(), true);
      m_slotStates.back().algStatuses.assign(m_algorithms.size(), StatusCode::SUCCESS);
      m_slotStates.back().algExecStates.assign(m_algorithms.size(), AlgExecState::UNSCHEDULED);
      m_slotStates.back().coroutines.resize(m_algorithms.size());
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
   if(!m_runStarted) {
      throw RuntimeError("In Scheduler::update(): Cannot update before run start");
   }
   for(int slot = 0; slot < m_slots; ++slot) {
      SlotState& slotState = m_slotStates[slot];

      if(std::ranges::all_of(slotState.algExecStates,
                             [](AlgExecState x) { return x == AlgExecState::FINISHED; })) {
         EventContext ctx{slotState.eventNumber, slot, this, m_streams[slot]};
         EventStoreRegistry::of(ctx).clear();
         slotState.eventNumber = m_nextEvent++;
         // vector<bool> not compatible with std::ranges.
         std::fill(slotState.cudaFinished.begin(), slotState.cudaFinished.end(), true);
         std::ranges::fill(slotState.algExecStates, AlgExecState::UNSCHEDULED);
         slotState.eventManager->reset();
      }

      // Stop processing events when target.
      if(slotState.eventNumber >= m_events) {
         continue;
      }

      for(std::size_t alg = 0; alg < m_algorithms.size(); ++alg) {
         if(slotState.algExecStates[alg] == AlgExecState::SCHEDULED) {
            continue;
         } else if(slotState.algExecStates[alg] == AlgExecState::FINISHED) {
            continue;
         } else if(slotState.algExecStates[alg] == AlgExecState::ERROR) {
            m_actionQueue.push([algStatus = slotState.algStatuses[alg]]() -> StatusCode {
               return algStatus;
            });
            return slotState.algStatuses[alg];
         }

         if(!slotState.eventManager->isAlgExecutable(alg)) {
            continue;
         }

         if(slotState.algExecStates[alg] == AlgExecState::SUSPENDED
            && not slotState.cudaFinished[alg]) {
            continue;
         }

         // This assertion might help keeping track of new additions to AlgExecState.
         // We should only reach this point if the algorithm is either UNSCHEDULED or SUSPENDED with CUDA finished, i.e. ready to be scheduled.
         assert((slotState.algExecStates[alg] == AlgExecState::UNSCHEDULED)
                || (slotState.algExecStates[alg] == AlgExecState::SUSPENDED
                    && slotState.cudaFinished[alg]));

         pushAction(slot, alg, slotState);
      }
   }
   return StatusCode::SUCCESS;
}


void Scheduler::pushAction(int slot, std::size_t ialg, SlotState& slotState) {
   slotState.algExecStates[ialg] = AlgExecState::SCHEDULED;

   // Add the action that would schedule the execution of the algorithm.
   // m_arena.execute() is used to ensure that the action is executed (synchronously) in our task arena.
   // next, m_group.run() launches the asynchronous task in the group to be able to `wait()`.
   m_arena.execute([this, ialg, slot, &slotState, &alg = m_algorithms[ialg].get()]() {
      m_group.run([this, ialg, slot, &slotState, &alg]() {
         if(slotState.coroutines[ialg].empty()) {
            // Do not resume the first time coroutine is launched because initial_suspend never
            // suspends.
            EventContext ctx{slotState.eventNumber, slot, this, m_streams[slot]};
            slotState.coroutines[ialg] = alg.execute(ctx);
         } else {
            slotState.coroutines[ialg].resume();
         }

         StatusCode algStatus;
         if(slotState.coroutines[ialg].isResumable()) {
            algStatus = slotState.coroutines[ialg].getYield();
         } else {
            algStatus = slotState.coroutines[ialg].getReturn();
            if(!slotState.eventManager->setAlgExecuted(ialg)) {
               slotState.algExecStates[ialg] = AlgExecState::ERROR;
            }
         }

         // At the last algorithm in the event, decrement the remaining event counter.
         if(ialg == m_algorithms.size() - 1
            && !slotState.coroutines[ialg].isResumable()) {
            m_remainingEvents.fetch_sub(1);
         }

         slotState.algStatuses[ialg] = algStatus;
         if(!algStatus) {
            slotState.algExecStates[ialg] = AlgExecState::ERROR;
            m_actionQueue.push([algStatus]() -> StatusCode { return algStatus; });
         } else {
            if(slotState.coroutines[ialg].isResumable()) {
               slotState.algExecStates[ialg] = AlgExecState::SUSPENDED;
            } else {
               slotState.algExecStates[ialg] = AlgExecState::FINISHED;
               slotState.coroutines[ialg].setEmpty();
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
   for(std::size_t i{0}; const auto& slotState : m_slotStates) {
      for(std::size_t j{0}; const auto& algStatus : slotState.algStatuses) {
         std::cout << "slot: " << i << ", algorithm number: " << j++ << "\n"
                   << algStatus.what() << std::endl
                   << std::endl;
      }
      ++i;
   }
}
