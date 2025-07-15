#include "Scheduler.hpp"
#include "AlgorithmContext.hpp"

void AlgorithmContext::newScheduleResumeCallback(void* args) {
  using AR = Scheduler::RunQueue::ActionRequest;
  std::unique_ptr<AlgorithmContext> ctx(static_cast<AlgorithmContext*>(args));

  // std::cout << "In AlgorithmContext::newScheduleResumeCallback(): Scheduling resume for event " << ctx->eventNumber
  //           << ", slot " << ctx->slotNumber << ", alg " << ctx->algorithmNumber << std::endl;
  ctx->scheduler.m_runQueue.queue.push(AR{
    .type = AR::ActionType::Resume,
    .slot = ctx->slotNumber,
    .alg = ctx->algorithmNumber
  });
}
