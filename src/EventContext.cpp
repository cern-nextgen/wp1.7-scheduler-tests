#include "EventContext.hpp"

#include "Scheduler.hpp"


void notifyScheduler(void* args) {
   EventContext* ctx = static_cast<EventContext*>(args);
   ctx->scheduler->setCudaSlotState(ctx->slotNumber, ctx->algNumber, true);
   ctx->scheduler->actionUpdate();
   delete ctx;
}

