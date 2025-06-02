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

namespace WP17Scheduler {
   /**
    * @brief Base exception class
    */
   class Exception: public std::exception {
   public:
      Exception(const std::string &w): m_what(w) {}
      const char * what() const noexcept override { return m_what.c_str(); } 
   private:
      const std::string m_what;
   };
}

#define DEFINE_EXCEPTION(name) class name: public WP17Scheduler::Exception { using WP17Scheduler::Exception::Exception; }

/**
 * @brief Scheduler class 
 * This class keeps track of events, threads and algorithms. 
 * The constructor parameters a are:
 * @param events Number of events
 * @param threads number of threads
 * @param slots Number of slots (i.e. concurrent events being processed, aso equal to the number of CUDA streams).
 * @warning This class itself is not thread safe.
 */
class Scheduler {
public:
   /**
    * @brief Scheduler constructor
    * @param events Number of events
    * @param threads number of threads
    * @param slots Number of slots (i.e. concurrent events being processed).
    */
   Scheduler(int events = 500, int threads = 4, int slots = 4);

   // Forbidden constructors
   Scheduler(const Scheduler&) = delete;
   Scheduler& operator=(const Scheduler&) = delete;

   /**
    * @brief Destructor just cleans up the CUDA streams.
    */
   ~Scheduler();

   /** 
    * @brief Adds an algorithm to the algorithm list. This function should be called before running. 
    * @todo: Should this be a constant reference? 
    */
   void addAlgorithm(AlgorithmBase& alg); 

   /** 
    * @brief  Main scheduler loop
    */
   StatusCode run();

   /**
    * @brief Sets the CUDA state to `state` for algorithm number `alg` on slot number `slot`.
    * @param slot Slot index
    * @param alg Algo index
    * @param state `bool` state
    * @todo Document BOOL meaning.
    */
   void setCudaSlotState(int slot, std::size_t alg, bool state);

   /**
    * @brief Pushes the `update()` function in the action queue.
    */
   void actionUpdate();

private:
   using action_type = std::function<StatusCode()>;

   /**
    * @brief Internal state of the scheduler for an individual even being processed.
    * Note that SlotState is movable but not copyable since AlgCoInterface is not copyable.
    * @todo The algorithm states should be a structure instead of a bunch of vectors accessed by the same index.
    */
   struct SlotState {
      /// Event ID being processed in this slot
      int eventNumber = 0;

      /// CUDA status @todo Add details, `bool` meaning.
      std::vector<bool> cudaFinished{};

      /// Algorithm results. Only valid if the algorithm is finished.
      std::vector<StatusCode> algStatuses{};

      /// @todo Execution state of each algorithm in the slot.
      std::vector<AlgExecState> algExecStates{};

      /// @todo Add details.
      std::vector<AlgorithmBase::AlgCoInterface> coroutines{};

      /// @todo Add details. Seems to handles algoritms dependencies and data object collection.
      std::unique_ptr<EventContentManager> eventManager{};
   };

   /**
    * @brief Assigns in initial event ids to each `slot` and creates the CUDA streams.
    */
   void initSchedulerState();

   /**
    * @brief Call functor `f` synchronously. In case of failed status, wait for all tasks in the `m_group` task group to complete. 
    * Asserts if the tasks were cancelled.
    * @param f  `action_type` functor.
    * @return `StatusCode` status.
    * @todo Undestand how the other tasks are signaled to finish
    */
   StatusCode executeAction(action_type& f);

   /**
    * @brief The workhorse of the scheduler. Processes each algo of slot at a time, in individual steps. This is one round of the 
    * scheduling over each algo of each slot.
    * 
    * - If all algorithms in the slot are finished, it resets the slot state and prepares it for the next event (up tp m_events, the objective)
    * 
    * - Idle slots are expressed with an event number greater than or equal to `m_events`.
    *
    * - Then, in all cases errors are checked and reported (all errors are fatal).
    * 
    * - States which do not request action are skipped: SCHEDULED, FINISHED, ERROR, SUSPENDED without CUDA finished.
    * 
    * 
    * @return Status for this round of scheduling.
    */
   StatusCode update();

   /**
    * @brief Bottom half of `update()`. 
    * 
    * @param slot Slot for the currently processed event.
    * @param ialg Algorithm index in the algorithm list.
    * @param slotState Reference to the slot state that is being updated.
    */
   void pushAction(int slot, std::size_t ialg, SlotState& slotState);
   void printStatuses() const;

   int m_events;
   int m_threads;
   int m_slots;
   int m_nextEvent = 0;
   bool m_runStarted = false;
   std::atomic_int m_remainingEvents;

   std::vector<std::reference_wrapper<AlgorithmBase>> m_algorithms;

   std::vector<SlotState> m_slotStates;
   /**
    * @brief Contains CUDA streams for each slot, in a one-to-one relationship.
    * @todo It should simply be a member of SlotState.
    */
   std::vector<cudaStream_t> m_streams;

   tbb::task_arena m_arena;
   tbb::task_group m_group;
   tbb::concurrent_bounded_queue<action_type> m_actionQueue;
public:
   DEFINE_EXCEPTION(RuntimeError);
};
