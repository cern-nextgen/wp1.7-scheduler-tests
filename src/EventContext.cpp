#include "EventContext.hpp"

#include "Scheduler.hpp"


void notifyScheduler(void* args) {
   Notification* notify = static_cast<Notification*>(args);
   EventContext& ctx = notify->ctx;

   ctx.scheduler->setCudaSlotState(ctx.slotNumber, notify->algNumber, true);
   ctx.scheduler->actionUpdate();

   delete notify;
}
