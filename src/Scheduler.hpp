#pragma once


#include <cuda_runtime_api.h>
#include <oneapi/tbb/concurrent_queue.h>
#include <oneapi/tbb/task_arena.h>
#include <oneapi/tbb/task_group.h>

#include <atomic>
#include <cstdlib>
#include <functional>
#include <memory>
#include <string>
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
    enum class ExecutionStrategy {
        SingleLaunch,
        StraightLaunches,
        StraightDelegated,
        Graph,
        CachedGraphs,
        CachedGraphsDelegated
    };

   // Struct to hold run statistics
   struct RunStats {
      int events = 0;
      double rate = 0.0;      // events/sec
      long long duration = 0; // milliseconds
   };

   /**
    * @brief Scheduler constructor
    * @param events Number of events
    * @param threads number of threads
    * @param slots Number of slots (i.e. concurrent events being processed).
    */
   Scheduler(int threads = 4, int slots = 4, ExecutionStrategy executionStrategy = ExecutionStrategy::SingleLaunch);

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
    * @brief Main scheduler loop with statistics. Re-runable, so we can warm up before measuring performance.
    * @param eventsToProcess Number of events to process.
    * @param stats Reference to RunStats structure to store statistics.
    */
   StatusCode run(int eventsToProcess, RunStats& stats);

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

   static std::string to_string(ExecutionStrategy strategy) {
      switch (strategy) {
         case ExecutionStrategy::SingleLaunch:     return "SingleLaunch";
         case ExecutionStrategy::Graph:            return "Graph";
         case ExecutionStrategy::CachedGraphs:     return "CachedGraphs";
         case ExecutionStrategy::StraightLaunches: return "StraightLaunches";
         case ExecutionStrategy::StraightDelegated: return "StraightDelegated";
         case ExecutionStrategy::CachedGraphsDelegated: return "CachedGraphsDelegated";
         default:                                  return "Unknown";
      }
   }

private:
   using action_type = std::function<StatusCode()>;
   
   /**
    * @brief Internal state of the scheduler for each algorithm.
    */
   struct AlgorithmState {
      /// CUDA status @todo Add details, `bool` meaning.
      bool cudaFinished = false;

      /// Algorithm results. Only valid if the algorithm is finished.
      StatusCode status = StatusCode::SUCCESS;

      /// @todo Execution state of each algorithm in the slot.
      AlgExecState execState = AlgExecState::UNSCHEDULED;

      /// Coroutine interface to algo execution. @todo Add details.
      AlgorithmBase::AlgCoInterface coroutine;
   };

   /**
    * @brief Internal state of the scheduler for an individual even being processed.
    * Note that SlotState is movable but not copyable since AlgCoInterface is not copyable.
    */
   struct SlotState {
      /// Event ID being processed in this slot
      int eventNumber = 0;
      /// States of the individual algorithms in the slot.
      std::vector<AlgorithmState> algorithms;        // State of each algorithm in the slot.
      
      /// @todo Add details. Seems to handles algoritms dependencies and data object collection.
      std::unique_ptr<EventContentManager> eventManager;
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
    * @brief Bottom half of `update()`.  It will launch the asynchronous processing of a given algorithm. 
    * The coroutine will be created or resumed, and the result will be used to update the algorithm state.
    * 
    * @param slot Slot for the currently processed event.
    * @param ialg Algorithm index in the algorithm list.
    * @param slotState Reference to the slot state that is being updated.
    * @todo This function pushes the `update()` function in the action queue once per scheduled algorithm, which is 
    * too much. It should be called only once per full event update loop.
    * @todo This function's code claruty will benefit a lot from turning the collection of algo arrays into a strcuture.
    */
   void pushAction(int slot, std::size_t ialg, SlotState& slotState);

   /**
    * @brief Prints the current status of each algorithm in each slot.
    */
   void printStatuses() const;

   /// @brief Number of threads to use.
   int m_threads;

   /// @brief Number of slots to use (i.e. concurrent events being processed).
   int m_slots;

   /// @brief Id of the next event to process.
   int m_nextEvent = 0;

   /// @brief Flag controling the switchover from configuring (registering algorithms) to running.
   bool m_runStarted = false;

   /// @brief Target event number for the current run.
   int m_targetEventId = 0;

   /// @brief Number of events remaining to be processed.
   std::atomic_int m_remainingEvents;

   /// @brief List of algorithms retistered in the scheduler.
   std::vector<std::reference_wrapper<AlgorithmBase>> m_algorithms;

   /// @brief Vector tracking each slot's state
   std::vector<SlotState> m_slotStates;

   /// @brief CUDA streams for each slot, in a one-to-one relationship.
   /// @todo It should simply be a member of SlotState.
   std::vector<cudaStream_t> m_streams;

   /// @brief TBB task arena representing the thread pool we will run on.
   tbb::task_arena m_arena;

   /// @brief TBB task group to control the tasks.
   tbb::task_group m_group;

   /// @brief TBB concurrent bounded queue for keeping track of actions to be executed.
   tbb::concurrent_bounded_queue<action_type> m_actionQueue;

   /// @brief Execution strategy for the algorithms.
   ExecutionStrategy m_executionStrategy;
public:

   /// @brief Exception class for scheduler errors.
   DEFINE_EXCEPTION(RuntimeError);
};
