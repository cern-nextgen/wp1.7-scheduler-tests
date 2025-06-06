#pragma once


#include <cuda_runtime_api.h>

#include <cstdlib>
#include <string>


// Forward declarations.
class Scheduler;

/**
 * @brief Indices to event/slot context, plus pointed to global scheduler, they are the sole interface to the algorithms, via `execute()`.
 * Also contains the CUDA stream.
 */
struct EventContext {
   int eventNumber = 0;
   int slotNumber = 0;
   Scheduler* scheduler = nullptr;
   cudaStream_t stream;

   std::string info() const {
      return "ctx.eventNumber = " + std::to_string(this->eventNumber)
           + ", ctx.slotNumber = " + std::to_string(this->slotNumber);
   }
};

/**
 * @brief Parameters passed to the CUDA callback function.
 */
struct Notification {
   EventContext& ctx;
   std::size_t algNumber;
};


// Function to be passed to CUDA callback.
// args points to Notification.
void notifyScheduler(void* args);
