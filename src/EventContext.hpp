#pragma once


#include <cuda_runtime_api.h>

#include <cstdlib>
#include <string>


// Forward declarations.
class Scheduler;


struct EventContext {
   int eventNumber = 0;
   int slotNumber = 0;
   std::size_t algNumber = 0;
   Scheduler* scheduler = nullptr;
   cudaStream_t stream;

   std::string info() const {
      return "ctx.eventNumber = " + std::to_string(this->eventNumber)
           + ", ctx.slotNumber = " + std::to_string(this->slotNumber);
   }
};


// Function to be passed to CUDA callback.
// args points to EventContext.
void notifyScheduler(void* args);
