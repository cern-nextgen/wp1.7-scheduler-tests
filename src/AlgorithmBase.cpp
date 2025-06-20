#include "AlgorithmBase.hpp"
#include "EventContext.hpp"

AlgorithmBase::AlgCoInterface AlgorithmBase::executeGraph(EventContext ctx) const {
  auto exec = execute(ctx);
  while (exec.resume()) {
    // Process the coroutine execution
    co_yield exec.getYield();
  }
  co_return exec.getReturn();
}