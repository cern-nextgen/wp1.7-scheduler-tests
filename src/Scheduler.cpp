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
   tbb::global_control global_thread_limit(tbb::global_control::max_allowed_parallelism,
                                           m_threads + 1);
   EventStoreRegistry::instance().data().resize(slots);
}


void Scheduler::addAlgorithm(AlgorithmBase& alg) {
   m_algorithms.push_back(alg);
}


StatusCode Scheduler::run() {
   // Initialize all the algorithms.
   if(StatusCode status = AlgorithmBase::for_all(m_algorithms, &AlgorithmBase::initialize);
      !status) {
      return status;
   }

   this->initSchedulerState();

   // Schedule the first set of algorithms.
   action_type firstAction = [this]() { return this->update(); };
   if(StatusCode status = this->executeAction(firstAction); !status) {
      return status;
   }

   // Execute "actions" until all events are processed.
   action_type action;
   while(m_remainingEvents.load() > 0) {
      m_actions.pop(action);
      if(StatusCode status = this->executeAction(action); !status) {
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
   m_actions.push([this]() -> StatusCode { return this->update(); });
}


void Scheduler::initSchedulerState() {
   m_nextEvent = 0;
   m_remainingEvents = m_events;

   m_slotStates.clear();
   for(int i = 0; i < m_slots; ++i) {
      m_slotStates.push_back({m_nextEvent++});
      m_slotStates.back().cudaFinished = std::vector<bool>(m_algorithms.size(), true);
      m_slotStates.back().algStatuses
          = std::vector<StatusCode>(m_algorithms.size(), StatusCode::SUCCESS);
      m_slotStates.back().algStates
          = std::vector<AlgExecState>(m_algorithms.size(), AlgExecState::UNSCHEDULED);
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
      // Make sure all tasks finish.
      tbb::task_group_status taskStatus = m_group.wait();
      assert(taskStatus != tbb::task_group_status::canceled);
      this->printStatuses();
      return status;
   }
   return StatusCode::SUCCESS;
}


StatusCode Scheduler::update() {
   // Set up actions for launching the next algorithm in each event slot.
   for(int slot = 0; slot < m_slots; ++slot) {
      SlotState& slotState = m_slotStates[slot];

      if(std::ranges::all_of(slotState.algStates,
                             [](AlgExecState x) { return x == AlgExecState::FINISHED; })) {
         EventContext ctx{slotState.eventNumber, slot, this, m_streams[slot]};
         eventStoreOf(ctx).clear();
         slotState.eventNumber = m_nextEvent++;
         // vector<bool> not compatible with std::ranges.
         std::fill(slotState.cudaFinished.begin(), slotState.cudaFinished.end(), true);
         std::ranges::fill(slotState.algStates, AlgExecState::UNSCHEDULED);
         slotState.eventManager->reset();
      }

      // Do not run redundant events.
      if(slotState.eventNumber >= m_events) {
         continue;
      }

      for(std::size_t alg = 0; alg < m_algorithms.size(); ++alg) {
         if(slotState.algStates[alg] == AlgExecState::SCHEDULED) {
            continue;
         } else if(slotState.algStates[alg] == AlgExecState::FINISHED) {
            continue;
         } else if(slotState.algStates[alg] == AlgExecState::ERROR) {
            this->m_actions.push([algStatus = slotState.algStatuses[alg]]() -> StatusCode {
               return algStatus;
            });
            return slotState.algStatuses[alg];
         }

         if(!slotState.eventManager->isAlgExecutable(alg)) {
            continue;
         }

         if(slotState.algStates[alg] == AlgExecState::SUSPENDED
            && not slotState.cudaFinished[alg]) {
            continue;
         }

         this->pushAction(slot, alg, slotState);
      }
   }
   return StatusCode::SUCCESS;
}


void Scheduler::pushAction(int slot, std::size_t ialg, SlotState& slotState) {
   slotState.algStates[ialg] = AlgExecState::SCHEDULED;

   // Add the action that would schedule the execution of the
   // algorithm.
   this->m_arena.execute([this, ialg, slot, &slotState, &alg = m_algorithms[ialg].get()]() {
      this->m_group.run([this, ialg, slot, &slotState, &alg]() {
         if(slotState.coroutines[ialg].empty()) {
            // Do not resume the first time coroutine is launched because
            // initial_suspend never suspends.
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
               slotState.algStates[ialg] = AlgExecState::ERROR;
            }
         }

         // At the last algorithm in the event, decrement the remaining event counter.
         if(ialg == this->m_algorithms.size() - 1
            && !slotState.coroutines[ialg].isResumable()) {
            this->m_remainingEvents.fetch_sub(1);
         }

         slotState.algStatuses[ialg] = algStatus;
         if(algStatus.isFailure()) {
            slotState.algStates[ialg] = AlgExecState::ERROR;
            this->m_actions.push([algStatus]() -> StatusCode { return algStatus; });
         } else {
            if(slotState.coroutines[ialg].isResumable()) {
               slotState.algStates[ialg] = AlgExecState::SUSPENDED;
            } else {
               slotState.algStates[ialg] = AlgExecState::FINISHED;
               slotState.coroutines[ialg].setEmpty();
            }
            this->m_actions.push([this]() -> StatusCode { return this->update(); });
         }
      });
   });
}


void Scheduler::printStatuses() const {
   std::cout << std::endl;
   std::cout << std::endl;
   std::cout << "Printing all statuses" << std::endl;
   for(std::size_t i{}; const auto& slotState : m_slotStates) {
      for(std::size_t j{}; const auto& algStatus : slotState.algStatuses) {
         std::cout << "slot: " << i << ", algorithm number: " << j++ << "\n"
                   << algStatus.what() << std::endl
                   << std::endl;
      }
      ++i;
   }
}
