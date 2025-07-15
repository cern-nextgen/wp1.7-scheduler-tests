#pragma once
#include <vector>
#include <mutex>
#include <thread>
#include <atomic>
#include <functional>
#include <stdexcept>
#include <ranges>
#include <tbb/concurrent_queue.h>
#include <cuda_runtime_api.h>
#include "AlgorithmBase.hpp"
#include "EventContentManager.hpp"
#include "EventStore.hpp"


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

class AlgorithmContext;

/**
 * @brief A progressive build of the scheduler replacement.
 */
struct Scheduler {
   friend class AlgorithmContext;
public:
    enum class ExecutionStrategy {
        SingleLaunch,
        StraightLaunches,
        StraightDelegated,
        StraightMutexed,
        StraightThreadLocalStreams,
        StraightThreadLocalContext,
        Graph,
        GraphFullyDelegated,
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
    * @param threads number of threads
    * @param slots Number of slots (i.e. concurrent events being processed).
    * @param executionStrategy Execution strategy for the algorithms.
    */
   Scheduler(int threads = 4, int slots = 4, ExecutionStrategy executionStrategy = ExecutionStrategy::SingleLaunch);

   // Forbidden constructors
   Scheduler(const Scheduler&) = delete;
   Scheduler& operator=(const Scheduler&) = delete;

   /**
    * @brief Finalizes all algorithms. This should be called after by user after running, as object destruction
    * order is not guaranteed for scheduler and algorithms.
    * @return StatusCode indicating success or failure.
    */
   StatusCode finalizeAlgorithms() {
      return AlgorithmBase::for_all(m_algorithms, &AlgorithmBase::finalize);
   }

   /**
    * @brief Destructor is empty.
    */
   ~Scheduler() {}


   /** 
    * @brief Adds an algorithm to the algorithm list. This function should be called before running. 
    * @todo: Should this be a constant reference? 
    */
   void addAlgorithm(AlgorithmBase& alg);

private:
   /**
    * @brief Locks the addition of new algorithms and prepares the dependency map.
    * @note Algorithms should be added before this function is called.
    */
   void initSchedulerState();

public:
   /**
    * @brief Starts the processing of events.
    * @param eventsToProcess Number of events to process.
    * @param stats RunStats structure to hold run statistics.
    * @return StatusCode indicating success or failure.
    * @throw RuntimeError if the scheduler is misconfigured or if an error occurs during processing.
    * @note This function is re-runnable, and implicity calls `initSchedulerState()` on first run.
    */
   StatusCode run(int eventsToProcess, RunStats& stats);

public:

   /**
    * @brief Structure holding algorithm state and interface to run it.
    */
   struct AlgorithmSlot {
      /**
       * @brief Mutex to ensure that only one thread processes the algorithm. The processing can queue
       * callbacks to GPU or other async resource, which will add the resuming of this algorithm to the 
       * run queue.
       * @note The locking model is to first hold the mutex on the algo and then briefly get the slot scheduling
       * mutex to update the slot state.
       */
      std::mutex mutex;

      /**
       * @brief Algorithm coroutine.
       */
      AlgorithmBase::AlgCoInterface coroutine; // Coroutine interface for the algorithm
   };

   /**
    * @brief Structure holding all the algos and data for one the processing of one event.
    */
   struct EventSlot {
      // Constructors and destructors will manage the CUDA stream.
      EventSlot();
      ~EventSlot();
      EventSlot(const EventSlot&) = delete;
      EventSlot& operator=(const EventSlot&) = delete;
      EventSlot(EventSlot&&) = delete;
      EventSlot& operator=(EventSlot&&) = delete;
      /**
       * @brief Initializer of event content manager.
       */
      void initialize(const AlgorithmDependencyMap& depMap, int eId) {
         eventContentManager.resize(depMap);
         for (auto aidx: std::ranges::iota_view(std::size_t(0), depMap.algorithmsCount())) {
            std::ignore = algorithms[aidx];
         }
         new (& algorithms) std::vector<AlgorithmSlot>(depMap.algorithmsCount());
         eventNumber = eId;
      }

      /**
       * @brief Event number for the slot.
       */
      int eventNumber = -1;

      /**
       * @brief Mutex to ensure that only one thread schedules the algorithms in the slot.
       */
      std::mutex schedulingMutex;

      /**
       * @brief Algorithms in the slot, each with its own coroutine interface and mutex.
       */
      std::vector<AlgorithmSlot> algorithms; // Algorithms in the slot, each with its own coroutine interface and mutex

      /**
       * @brief Event content manager for the slot, managing data objects and dependencies.
       */
      EventContentManager eventContentManager;

      /**
       * @brief Event store for the slot, storing data products and event information.
       */
      EventStore eventStore;

      /**
       * @brief CUDA stream for the slot
       */
      cudaStream_t stream = nullptr;
   };

   /**
    * @brief Run queue for the scheduler
    */
   struct RunQueue {
      struct ActionRequest {
        enum class ActionType {
          Start,  ///< Request to start the slot
          Resume,  ///< Request to resume the worker thread
          Exit     ///< No action, used to signal thread exit
        } type; ///< Type of action
        int slot = 0; ///< Slot index
        std::size_t alg = 0; ///< Algorithm index
        bool exit = false; ///< Exit flag for the worked thread TODO: duplicate with ActionType::Exit
      };
      tbb::concurrent_queue<ActionRequest> queue; // Queue of action requests
   };
private:

   /**
    * @brief Populates the run queue with requests for each runnable algorithm in each slot.
    * (At this point, only the algorithms with no dependencies will be runnable).
    */
   void populateRunQueue();

   /**
    * @brief Schedules the next event in the given slot, if available, does nothing
    * if all events were scheduled already, unless this is the last event, in which case
    * it will also queue exit requests for the worker threads.
    * @param slot The event slot to schedule the next event in.
    * @note This function assumes that the slot's scheduling mutex is already locked.
    */
   RunQueue::ActionRequest scheduleNextEventInSlot(EventSlot& slot);

   /**
    * @brief Starts the additional worker threads that will process the run queue from 
    * the main thread.
   */
   void startWorkerThreads();

   /**
    * @brief Main loop for the worker threads, processing action requests from the run queue.
    */
   void processRunQueue();

   /**
    * @brief Processes a single action request from the run queue.
    * @param req The action request to process.
    */
   void processActionRequest(RunQueue::ActionRequest& req);

   /**
    * @brief Joins all additional worker threads, ensuring they have completed execution.
    */
   void joinWorkerThreads();

   /// @brief Number of threads to use.
   int m_threadsNumber;

   /// @brief Number of slots to use (i.e. concurrent events being processed).
   int m_eventSlotsNumber;

   /// @brief Id of the next event to process.
   std::atomic_int m_nextEventId = 0;

   /// @brief Flag controling the switchover from configuring (registering algorithms) to running.
   bool m_runStarted = false;

   /// @brief Target event number for the current run.
   int m_targetEventId = 0;


   /// @brief Number of events remaining to be fully processed.
   std::atomic_int m_remainingEventsToComplete = 0;

   /// @brief List of algorithms registered in the scheduler.
   std::vector<std::reference_wrapper<AlgorithmBase>> m_algorithms;

   /// @brief The event content manager
   AlgorithmDependencyMap m_algoDependencyMap;

   /// @brief Vector tracking each slot's state
   std::vector<EventSlot> m_eventSlots;

  //  /// @brief CUDA streams for each slot, in a one-to-one relationship.
  //  /// @todo It should simply be a member of SlotState.
  //  std::vector<cudaStream_t> m_streams;

  //  /// @brief TBB task arena representing the thread pool we will run on.
  //  tbb::task_arena m_arena;

  //  /// @brief TBB task group to control the tasks.
  //  tbb::task_group m_group;

  //  /// @brief TBB concurrent bounded queue for keeping track of actions to be executed.
  //  tbb::concurrent_bounded_queue<action_type> m_actionQueue;

   /// @brief Execution strategy for the algorithms.
   ExecutionStrategy m_executionStrategy;
   /**
    * @brief Run queue for the scheduler
    */
   RunQueue m_runQueue;
   /**
    * @brief Worker threads for processing the slots
    */
   std::vector<std::thread> m_workerThreads;

   /// @brief Abort flag to signal all threads to stop processing.
   std::atomic_bool m_abort = false;

   /// @brief Function to print the status of the scheduler and slots (for debugging).
   void printStatus() const;

public:

   static std::string to_string(ExecutionStrategy strategy) {
      switch (strategy) {
         case ExecutionStrategy::SingleLaunch:               return "SingleLaunch";
         case ExecutionStrategy::Graph:                      return "Graph";
         case ExecutionStrategy::GraphFullyDelegated:        return "GraphFullyDelegated";
         case ExecutionStrategy::CachedGraphs:               return "CachedGraphs";
         case ExecutionStrategy::StraightLaunches:           return "StraightLaunches";
         case ExecutionStrategy::StraightDelegated:          return "StraightDelegated";
         case ExecutionStrategy::StraightMutexed:            return "StraightMutexed";
         case ExecutionStrategy::StraightThreadLocalStreams: return "StraightThreadLocalStreams";
         case ExecutionStrategy::StraightThreadLocalContext: return "StraightThreadLocalContext";
         case ExecutionStrategy::CachedGraphsDelegated:      return "CachedGraphsDelegated";
         default:                                            return "Unknown";
      }
   }

   /// @brief Exception class for scheduler errors.
   DEFINE_EXCEPTION(RuntimeError);
};
