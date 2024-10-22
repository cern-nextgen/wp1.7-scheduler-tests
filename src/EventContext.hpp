#pragma once


#include <cuda_runtime_api.h>

#include <cstdlib>


// Forward declarations.
class Scheduler;


struct EventContext {
   int eventNumber = 0;
   int slotNumber = 0;
   std::size_t algNumber = 0;
   Scheduler* scheduler = nullptr;
   cudaStream_t stream;
};


// Function to be passed to CUDA callback.
// args points to EventContext.
void notifyScheduler(void* args);
