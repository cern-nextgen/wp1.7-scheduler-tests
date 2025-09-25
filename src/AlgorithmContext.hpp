#pragma once


#include <cuda_runtime_api.h>

#include <cstdlib>
#include <string>

// Forward declarations.
class EventStore;
class Scheduler;

/**
 * @brief Indices to event/slot context, plus pointed to global scheduler, they are the sole interface to the algorithms, via `execute()`.
 * Also contains the CUDA stream.
 */
struct AlgorithmContext {
   int eventNumber = 0;
   int slotNumber = 0;
   std::size_t algorithmNumber = 0; // Index of the current algorithm in the scheduler's list.
   Scheduler& scheduler;
   EventStore& eventStore;
   cudaStream_t stream;

   // Member by member constructor
   AlgorithmContext(int eventNumber, int slotNumber, std::size_t algorithmNumber,
                  Scheduler& scheduler, EventStore& eventStore, cudaStream_t stream)
       : eventNumber(eventNumber),
         slotNumber(slotNumber),
         algorithmNumber(algorithmNumber),
         scheduler(scheduler),
         eventStore(eventStore),
         stream(stream) {}

   // Copy constructor
   AlgorithmContext(const AlgorithmContext&) = default;

   std::string info() const {
      return "ctx.eventNumber = " + std::to_string(this->eventNumber)
           + ", ctx.slotNumber = " + std::to_string(this->slotNumber)
           + ", ctx.algorithmNumber = " + std::to_string(this->algorithmNumber);
   }
   // Function to be passed to CUDA callback.
   // it will queue the algorithm resume in the scheduler's run queue.
   static void newScheduleResumeCallback(void* args);
};

