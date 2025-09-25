#include "AlgorithmBase.hpp"

AlgorithmBase::AlgCoInterface AlgorithmBase::executeStraight(AlgorithmContext ctx) const {
  auto exec = execute(ctx);
  while (exec.isResumable()) {
    // Process the coroutine execution
    co_yield exec.getYield();
    exec.resume();
  }
  co_return exec.getReturn();
}

AlgorithmBase::AlgCoInterface AlgorithmBase::executeStraightDelegated(AlgorithmContext ctx) const {
  auto exec = execute(ctx);
  while (exec.isResumable()) {
    // Process the coroutine execution
    co_yield exec.getYield();
    exec.resume();
  }
  co_return exec.getReturn();
}

AlgorithmBase::AlgCoInterface AlgorithmBase::executeStraightMutexed(AlgorithmContext ctx) const {
  auto exec = execute(ctx);
  while (exec.isResumable()) {
    // Process the coroutine execution
    co_yield exec.getYield();
    exec.resume();
  }
  co_return exec.getReturn();
}

AlgorithmBase::AlgCoInterface AlgorithmBase::executeStraightThreadLocalStreams(AlgorithmContext ctx) const {
  auto exec = execute(ctx);
  while (exec.isResumable()) {
    // Process the coroutine execution
    co_yield exec.getYield();
    exec.resume();
  }
  co_return exec.getReturn();
}

AlgorithmBase::AlgCoInterface AlgorithmBase::executeStraightThreadLocalContext(AlgorithmContext ctx) const {
  auto exec = execute(ctx);
  while (exec.isResumable()) {
    // Process the coroutine execution
    co_yield exec.getYield();
    exec.resume();
  }
  co_return exec.getReturn();
}

AlgorithmBase::AlgCoInterface AlgorithmBase::executeGraph(AlgorithmContext ctx) const {
  auto exec = execute(ctx);
  while (exec.isResumable()) {
    // Process the coroutine execution
    co_yield exec.getYield();
    exec.resume();
  }  
  co_return exec.getReturn();
}

AlgorithmBase::AlgCoInterface AlgorithmBase::executeGraphFullyDelegated(AlgorithmContext ctx) const {
  auto exec = executeGraph(ctx);
  while (exec.isResumable()) {
    // Process the coroutine execution
    co_yield exec.getYield();
    exec.resume();
  }
  co_return exec.getReturn();
}

AlgorithmBase::AlgCoInterface AlgorithmBase::executeCachedGraph(AlgorithmContext ctx) const {
  auto exec = executeGraph(ctx);
  while (exec.isResumable()) {
    // Process the coroutine execution
    co_yield exec.getYield();
    exec.resume();
  }
  co_return exec.getReturn();
}

AlgorithmBase::AlgCoInterface AlgorithmBase::executeCachedGraphDelegated(AlgorithmContext ctx) const {
  auto exec = executeCachedGraph(ctx);
  while (exec.isResumable()) {
    // Process the coroutine execution
    co_yield exec.getYield();
    exec.resume();
  }
  co_return exec.getReturn();
}