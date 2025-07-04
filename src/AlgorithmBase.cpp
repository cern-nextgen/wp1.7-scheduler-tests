#include "AlgorithmBase.hpp"
#include "EventContext.hpp"

AlgorithmBase::AlgCoInterface AlgorithmBase::executeStraight(EventContext ctx) const {
  auto exec = execute(ctx);
  while (exec.resume()) {
    // Process the coroutine execution
    co_yield exec.getYield();
  }
  co_return exec.getReturn();
}

AlgorithmBase::AlgCoInterface AlgorithmBase::executeStraightDelegated(EventContext ctx) const {
  auto exec = execute(ctx);
  while (exec.resume()) {
    // Process the coroutine execution
    co_yield exec.getYield();
  }
  co_return exec.getReturn();
}

AlgorithmBase::AlgCoInterface AlgorithmBase::executeStraightMutexed(EventContext ctx) const {
  auto exec = execute(ctx);
  while (exec.resume()) {
    // Process the coroutine execution
    co_yield exec.getYield();
  }
  co_return exec.getReturn();
}

AlgorithmBase::AlgCoInterface AlgorithmBase::executeStraightThreadLocalStreams(EventContext ctx) const {
  auto exec = execute(ctx);
  while (exec.resume()) {
    // Process the coroutine execution
    co_yield exec.getYield();
  }
  co_return exec.getReturn();
}

AlgorithmBase::AlgCoInterface AlgorithmBase::executeStraightThreadLocalContext(EventContext ctx) const {
  auto exec = execute(ctx);
  while (exec.resume()) {
    // Process the coroutine execution
    co_yield exec.getYield();
  }
  co_return exec.getReturn();
}

AlgorithmBase::AlgCoInterface AlgorithmBase::executeGraph(EventContext ctx) const {
  auto exec = execute(ctx);
  while (exec.resume()) {
    // Process the coroutine execution
    co_yield exec.getYield();
  }
  co_return exec.getReturn();
}

AlgorithmBase::AlgCoInterface AlgorithmBase::executeGraphFullyDelegated(EventContext ctx) const {
  auto exec = executeGraph(ctx);
  while (exec.resume()) {
    // Process the coroutine execution
    co_yield exec.getYield();
  }
  co_return exec.getReturn();
}

AlgorithmBase::AlgCoInterface AlgorithmBase::executeCachedGraph(EventContext ctx) const {
  auto exec = executeGraph(ctx);
  while (exec.resume()) {
    // Process the coroutine execution
    co_yield exec.getYield();
  }
  co_return exec.getReturn();
}

AlgorithmBase::AlgCoInterface AlgorithmBase::executeCachedGraphDelegated(EventContext ctx) const {
  auto exec = executeCachedGraph(ctx);
  while (exec.resume()) {
    // Process the coroutine execution
    co_yield exec.getYield();
  }
  co_return exec.getReturn();
}