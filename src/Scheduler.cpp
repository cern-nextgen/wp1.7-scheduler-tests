#include "Scheduler.hpp"
#include "AssertCuda.cuh"
#include <exception>
#include <ranges>

#pragma GCC optimize("O0")

Scheduler::Scheduler(int threads, int slots, ExecutionStrategy executionStrategy)
    : m_threadsNumber{threads},
      m_eventSlotsNumber{slots},
      m_nextEventId{},
      m_executionStrategy(executionStrategy) {}

Scheduler::EventSlot::EventSlot() {
  // Create a CUDA stream for the slot.
  ASSERT_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
}

Scheduler::EventSlot::~EventSlot() {
  // Destroy the CUDA stream for the slot.
  if (stream) {
    cudaStreamDestroy(stream);
    stream = nullptr;
  }
}


void Scheduler::addAlgorithm(AlgorithmBase& alg) {
  if (m_runStarted) {
    throw RuntimeError("In Scheduler::addAlgorithm(): Algorithms cannot be added after run start");
  }
  m_algorithms.push_back(alg);
}


void Scheduler::initSchedulerState()  {
  // Ge all to initial state
  m_nextEventId = 0;
  m_remainingEventsToComplete = 0;
  m_eventSlots.clear();
  // First, populate the algorithm dependency map with the algorithms.
  new(&m_algoDependencyMap) AlgorithmDependencyMap(m_algorithms);

  // Initialize all the algorithms.
  if (StatusCode status = AlgorithmBase::for_all(m_algorithms, &AlgorithmBase::initialize);
      !status) {
     throw RuntimeError(std::string("In Scheduler::initSchedulerState(): Algorithm initialization failed: ") + status.what());
  }

  // Then, create the event slots. The event slots are assigned an event number
  // in all cases, even if the event is above target and the slot will be active.
  // It will possibly be in subsequent runs.
  // EventSlot is not movable or copyable so we do a placement new instead of resize.
  new (&m_eventSlots) std::vector<EventSlot>(m_eventSlotsNumber);
  for (auto slotId: std::ranges::iota_view(0, m_eventSlotsNumber)) {
     std::ignore = slotId; // Avoid unused variable warning
     m_eventSlots[slotId].initialize(m_algoDependencyMap, m_nextEventId++);
  //m_eventSlots[slotId].eventContentManager.dumpContents(m_algoDependencyMap, std::cout);
  }
}

StatusCode Scheduler::run(int eventsToProcess, RunStats& stats) {
  // Lock and algorithm registration.
  if (!m_runStarted) {
    initSchedulerState();
    m_runStarted = true;
  }
  m_remainingEventsToComplete.store(eventsToProcess);
  m_targetEventId += eventsToProcess;
  auto startTime = std::chrono::high_resolution_clock::now();
  // Populate the run queue with requests for each runnable algorithm in each slot.
  // (At this point, only the algorithms with no dependencies will be runnable).
  populateRunQueue();
  startWorkerThreads();
  processRunQueue();
  joinWorkerThreads();
  auto endTime = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
  double rate = static_cast<double>(eventsToProcess) / (duration / 1000.0); // Events per second

  // Store run statistics
  stats.events = eventsToProcess;
  stats.rate = rate;
  stats.duration = duration;

  //std::cout << "Processed " << m_events << " events in " << duration << " ms (" << rate << " events/sec)" << std::endl;

  return m_abort?StatusCode::FAILURE: StatusCode::SUCCESS;
}


void Scheduler::populateRunQueue() {
  for (auto& slot: m_eventSlots) {
    if (slot.eventNumber >= m_targetEventId) continue;
    for (auto& alg: slot.algorithms) {
      std::size_t algId = &alg - &slot.algorithms[0];
      if(m_algoDependencyMap.isAlgIndependent(algId)) {
        
        int sId = &slot - &m_eventSlots[0];
        RunQueue::ActionRequest req{RunQueue::ActionRequest::ActionType::Start, sId, algId, false};
        m_runQueue.queue.push(req);
      }
    }
  }
}

Scheduler::RunQueue::ActionRequest Scheduler::scheduleNextEventInSlot(EventSlot& slot) {
  // Clear all coroutines in the slot.
  for (auto& alg: slot.algorithms) {
    std::scoped_lock lock(alg.mutex);
    if (alg.coroutine.empty() || alg.coroutine.isResumable()) {
      std::cerr << "In Scheduler::scheduleNextEventInSlot(): Attempting to schedule a new event in a slot where an algorithm has not completed." << std::endl;
      std::cerr << "Event Slot State:" << std::endl;
      std::cerr << "Event Number: " << slot.eventNumber << std::endl;
      std::cerr << "Algorithms:" << std::endl;
      for (std::size_t i = 0; i < slot.algorithms.size(); ++i) {
        auto& a = slot.algorithms[i];
        std::cerr << "  Algorithm " << i << ": ";
        if (a.coroutine.empty()) {
          std::cerr << "Not started";
        } else if (a.coroutine.isResumable()) {
          std::cerr << "Resumable";
        } else {
          std::cerr << "Completed";
        }
        std::cerr << std::endl;
      }
      throw RuntimeError("In Scheduler::scheduleNextEventInSlot(): Algorithm has not completed for previous event");
    }
    alg.coroutine.setEmpty();
  }
  // Update slot event number and clear the coroutines.
  slot.eventNumber = m_nextEventId++;
  slot.eventStore.clear();
  slot.eventContentManager.reset();
  // If we completed the last event, we need to signal the worker threads to exit.
  if (m_remainingEventsToComplete.fetch_sub(1) == 1) {
    for (int i = 0; i < m_threadsNumber; ++i) {
      // std::cout << "Pushing exit request to run queue for worker thread " << i << std::endl;
      RunQueue::ActionRequest exitReq{RunQueue::ActionRequest::ActionType::Exit, -1,
        std::numeric_limits<std::size_t>::max(), true};
      m_runQueue.queue.push(exitReq);
    }
  }
  // Did we reach the target event number? In this case, we do not need to schedule it.
  // This return value signals to the caller that no scheduling was done.
  if (slot.eventNumber >= m_targetEventId) {
     return RunQueue::ActionRequest{RunQueue::ActionRequest::ActionType::Exit, 0, 0, true};
  }
  // We will schedule this event's first algos.
  RunQueue::ActionRequest firstReq;
  bool first = true;
  for (auto& alg: slot.algorithms) {
    std::size_t algId = &alg - &slot.algorithms[0];
    if(m_algoDependencyMap.isAlgIndependent(algId)) {
      
      int sId = &slot - &m_eventSlots[0];
      RunQueue::ActionRequest req{RunQueue::ActionRequest::ActionType::Start, sId, algId, false};
      if (first) {
        first = false;
        firstReq = req;
      } else {
        m_runQueue.queue.push(req);
      }
    }
  }
  return firstReq;
}

void Scheduler::startWorkerThreads() {
  // Start worker threads if not already started.
  if (m_workerThreads.empty()) {
    for (int i = 0; i < m_threadsNumber - 1; ++i) {
      m_workerThreads.emplace_back(&Scheduler::processRunQueue, this);
    }
  }
}

void Scheduler::joinWorkerThreads() {
  for (auto& thread : m_workerThreads) {
    if (thread.joinable()) {
      thread.join();
    }
  }
  m_workerThreads.clear();
}

void Scheduler::processRunQueue() {
  while (true) {
    RunQueue::ActionRequest req;
    if (m_runQueue.queue.try_pop(req)) {
      if (req.exit) {
        break; // Exit signal received
      }
      // std::cout << "Processing request from queue: slot=" << req.slot << ", alg=" << req.alg << ", type=" 
      //           << (req.type == RunQueue::ActionRequest::ActionType::Start ? "Start" : 
      //               req.type == RunQueue::ActionRequest::ActionType::Resume ? "Resume" : "Exit")
      //           << std::endl;
      processActionRequest(req);
    } else {
      // If no work is available, yield to avoid busy-waiting
      std::this_thread::yield();
    }
  }
}

void Scheduler::processActionRequest(RunQueue::ActionRequest& req) {
  // std::size_t functionRecycles = 0;
  begin:
  if(m_abort) return;
  // if (functionRecycles++) {
  //   std::cout << "Recycle(" << functionRecycles << ") ";
  // }

  // std::cout << "Processing request: slot=" << req.slot << ", alg=" << req.alg << ", type=" 
  //           << (req.type == RunQueue::ActionRequest::ActionType::Start ? "Start" : 
  //               req.type == RunQueue::ActionRequest::ActionType::Resume ? "Resume" : "Exit")
  //           << std::endl;
  // Validate the request.
  if (req.slot < 0 || req.slot >= m_eventSlotsNumber) {
    throw RuntimeError("In Scheduler::processActionRequest(): Invalid slot index");
  }
  auto& slot = m_eventSlots[req.slot];
  //std::cout << ", event=" << slot.eventNumber << std::endl;
  if (req.alg < 0 || req.alg >= slot.algorithms.size()) {
    throw RuntimeError("In Scheduler::processActionRequest(): Invalid algorithm index");
  }
  auto& algSlot = slot.algorithms[req.alg];

  // Lock the algorithm mutex to ensure only one thread runs the algorithm's coroutine at a time.
  std::unique_lock<std::mutex> algLock(algSlot.mutex);

  // If the request is to start, ensure the algorithm has not already run for this event.
  if (req.type == RunQueue::ActionRequest::ActionType::Start) {
    // If the coroutine is not empty, dump the whole slot state for debugging.
    if (!algSlot.coroutine.empty()) {
      std::cerr << "In Scheduler::processActionRequest(): Attempting to start an algorithm that has already started for this event." << std::endl;
      std::cerr << "Request: slot=" << req.slot << ", alg=" << req.alg << std::endl;
      std::cerr << "Event Slot State:" << std::endl;
      std::cerr << "Event Number: " << slot.eventNumber << std::endl;
      std::cerr << "Algorithms:" << std::endl;
      for (std::size_t i = 0; i < slot.algorithms.size(); ++i) {
        auto& a = slot.algorithms[i];
        std::cerr << "  Algorithm " << i << ": ";
        if (a.coroutine.empty()) {
          std::cerr << "Not started";
        } else if (a.coroutine.isResumable()) {
          std::cerr << "Resumable";
        } else {
          std::cerr << "Completed";
        }
        std::cerr << std::endl;
      }
      throw RuntimeError("In Scheduler::processActionRequest(): Algorithm already started for this event");
    }
    assert(algSlot.coroutine.empty());
    auto & alg = m_algorithms[req.alg].get();
    // Create the algorithm context, using the member-by-member constructor.
    AlgorithmContext ctx{
      slot.eventNumber,
      req.slot,
      req.alg,
      *this,
      slot.eventStore,
      slot.stream
    };
    switch (m_executionStrategy)
    {
      case ExecutionStrategy::SingleLaunch:
          algSlot.coroutine = alg.execute(ctx);
          break;
      case ExecutionStrategy::StraightLaunches:
          algSlot.coroutine = alg.executeStraight(ctx);
          break;
      case ExecutionStrategy::StraightDelegated:
          algSlot.coroutine = alg.executeStraightDelegated(ctx);
          break;
      case ExecutionStrategy::StraightMutexed:
          algSlot.coroutine = alg.executeStraightMutexed(ctx);
          break;
      case ExecutionStrategy::StraightThreadLocalStreams:
          algSlot.coroutine = alg.executeStraightThreadLocalStreams(ctx);
          break;
      case ExecutionStrategy::StraightThreadLocalContext:
          algSlot.coroutine = alg.executeStraightThreadLocalContext(ctx);
          break;
      case ExecutionStrategy::Graph:
          algSlot.coroutine = alg.executeGraph(ctx);
          break;
      case ExecutionStrategy::GraphFullyDelegated:
          algSlot.coroutine = alg.executeGraphFullyDelegated(ctx);
          break;
      case ExecutionStrategy::CachedGraphs:
          algSlot.coroutine = alg.executeCachedGraph(ctx);
          break;
      case ExecutionStrategy::CachedGraphsDelegated:
          algSlot.coroutine = alg.executeCachedGraphDelegated(ctx);
          break;
      default:
          std::cerr << "In Scheduler::pushAction(): Unknown execution strategy:" << to_string(m_executionStrategy) << std::endl;
          abort();
    }
  } else if(req.type == RunQueue::ActionRequest::ActionType::Resume) {
    if (algSlot.coroutine.empty()) {
      // std::cerr << "In Scheduler::processActionRequest(): Attempting to resume an algorithm that has not been started for this event." << std::endl;
      // std::cerr << "Request: slot=" << req.slot << ", alg=" << req.alg << std::endl;
      throw RuntimeError("In Scheduler::processActionRequest(): Attempting to resume an algorithm that has not been started for this event");
    }
    if (algSlot.coroutine.empty() || !algSlot.coroutine.isResumable()) {
      // std::cerr << "In Scheduler::processActionRequest(): Attempting to resume an algorithm that is not resumable." << std::endl;
      // std::cerr << "Request: slot=" << req.slot << ", alg=" << req.alg << std::endl;
      throw RuntimeError("In Scheduler::processActionRequest(): Attempting to resume an algorithm that is not resumable");
    }
    algSlot.coroutine.resume();
  } else {
    throw RuntimeError("In Scheduler::processActionRequest(): Unexpected action request type");
  }

  // Let's see the outcome.
  StatusCode algStatus;
  bool done = false;
  if (algSlot.coroutine.isResumable()) {
      algStatus = algSlot.coroutine.getYield();
  } else {
      algStatus = algSlot.coroutine.getReturn();
      done = true;
  }

  // If anything went wrong, set the abort flag and inject termination requests in the run queue.
  if (!algStatus) {
    m_abort = true;
    std::cerr << "In Scheduler::processActionRequest(): Algorithm " << req.alg 
              << " in slot " << req.slot << " for event " << slot.eventNumber 
              << " returned error status. Aborting run." << std::endl;
    for (int i = 0; i < m_threadsNumber; ++i) {
      RunQueue::ActionRequest exitReq{RunQueue::ActionRequest::ActionType::Exit, -1,
        std::numeric_limits<std::size_t>::max(), true};
      m_runQueue.queue.push(exitReq);
    }
    printStatus();
    return;
  }

  // We are done with the algorithm for now.
  algLock.unlock();

  // If an algorithm completed, there might be more to execute.
  if (done) {
    // Algorithm finished processing for this event.
    // Lock the slot mutex to safely check overall slot state.
    std::unique_lock<std::mutex> slotLock(slot.schedulingMutex);
    // TODO: error handling
    std::ignore = slot.eventContentManager.setAlgExecuted(req.alg, m_algoDependencyMap);
    // Get the list of dependent algorithms that might now be ready to run.
    auto dependents = slot.eventContentManager.getDependentAndReadyAlgs(req.alg, m_algoDependencyMap);
    // If there are no dependants, we might have completed the event.
    if (dependents.empty()) {
      // Check if all algorithms are done.
      bool allDone = true;
      for (auto& alg : slot.algorithms) {
        std::scoped_lock lock(alg.mutex);
        if (alg.coroutine.empty() ||  alg.coroutine.isResumable()) {
          allDone = false;
          break;
        }
      }
      if (allDone) {
        // Schedule the next event if available.
        // std::cout << "Event " << slot.eventNumber << " completed." << std::endl;
        auto nextReq = scheduleNextEventInSlot(slot);
        if (nextReq.exit) return;
        slotLock.unlock();
        req = nextReq;
        // std::cout << "GOTO on next event ";
        // std::cout << "Immediately running algorithm " << req.alg << " in slot " << req.slot << " for event " << slot.eventNumber << std::endl;
        goto begin;
      }
    } else {
      slotLock.unlock();
      req = {RunQueue::ActionRequest::ActionType::Start, req.slot, dependents[0], false};
      for (std::size_t i = 1; i < dependents.size(); ++i) {
        RunQueue::ActionRequest depReq{RunQueue::ActionRequest::ActionType::Start, req.slot, dependents[i], false};
        m_runQueue.queue.push(depReq);
      }
      // std::cout << "GOTO on dependent ";
      // std::cout << "Immediately running dependent algorithm " << req.alg << " in slot " << req.slot << " for event " << slot.eventNumber << std::endl;
      goto begin;
    }
  }
}

void Scheduler::printStatus() const {
    std::cout << "Scheduler Status:" << std::endl;
    std::cout << "  Threads: " << m_threadsNumber << std::endl;
    std::cout << "  Slots: " << m_eventSlotsNumber << std::endl;
    std::cout << "  Next Event ID: " << m_nextEventId.load() << std::endl;
    std::cout << "  Target Event ID: " << m_targetEventId << std::endl;
    std::cout << "  Remaining Events to Complete: " << m_remainingEventsToComplete.load() << std::endl;
    std::cout << "  Algorithms: " << m_algorithms.size() << std::endl;
    std::cout << "  Execution Strategy: " << to_string(m_executionStrategy) << std::endl;
    std::cout << "  Abort Flag: " << m_abort.load() << std::endl;
    for (int i = 0; i < m_eventSlotsNumber; ++i) {
        const auto& slot = m_eventSlots[i];
        std::cout << "  Slot " << i << ": Event " << slot.eventNumber << std::endl;
        for (std::size_t j = 0; j < slot.algorithms.size(); ++j) {
            const auto& alg = slot.algorithms[j];
            std::cout << "    Algorithm " << j << ": ";
            if (alg.coroutine.empty()) {
                std::cout << "Not started";
            } else if (alg.coroutine.isResumable()) {
                std::cout << "Resumable";
            } else {
                std::cout << "Completed";
            }
            std::cout << std::endl;
        }
    }
}
