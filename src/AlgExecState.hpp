#pragma once


#include <sstream>

#include "StatusCode.hpp"

/**
 * @brief Wrapper fpr algorithms states. With stream operator and reset to default state (UNSCHEDULED).
 */
class AlgExecState {
public:
   /**
    * @brief Enum representing the possible states of an algorithm execution.
    * The states are:
    * - UNSCHEDULED: Initial state of each algorithm. Algorithms will be moved by Scheduler:::update() via `pushAction()` (which will push
    *   the coroutine creation/running proper to the TBB run queue).
    * - SCHEDULED: The algorithm is scheduled for execution, but not yet running. This is set right before sumitting the algorithm to the TBB run queue.
    * - SUSPENDED: The algorithm is currently suspended. The algorithm coroutine is waiting on the CUDA execution. update() will requeue the algorithm
    *   to the TBB run queue (with the misnamed pushAction() function) when the CUDA execution is finished. The CUDA callback (notifyScheduler(), which belongs to
    *   no class nor namespace) will flip the CUDA state to true, and call actionUpdate() to queue an update() execution to the "action" queue (which is schedules 
    *   executions of scheduler::update()). The action queue is processed in Scheduler::run().
    * - FINISHED: The algorithm has finished its execution. The algorithm coroutine is done, and the algorithm results are available.
    * - ERROR: The algorithm has encountered an error during its execution. The algorithm coroutine is done, but the algorithm results are not available.
    *   Once this state is reached, update() will push a lambda returning error to the action queue, which triggers an immediate return from the Scheduler::run() function.
    */

   enum class State { UNSCHEDULED, SCHEDULED, SUSPENDED, FINISHED, ERROR };

   static constexpr State UNSCHEDULED = State::UNSCHEDULED;
   static constexpr State SCHEDULED = State::SCHEDULED;
   static constexpr State SUSPENDED = State::SUSPENDED;
   static constexpr State FINISHED = State::FINISHED;
   static constexpr State ERROR = State::ERROR;

   AlgExecState() = default;
   AlgExecState(const AlgExecState&) = default;
   AlgExecState(AlgExecState&&) = default;
   AlgExecState& operator=(const AlgExecState&) = default;
   AlgExecState& operator=(AlgExecState&&) = default;

   AlgExecState(State state) : m_state{state} {
   }

   void setState(State state) {
      m_state = state;
   }

   State getState() const {
      return m_state;
   }

   /**
    * @brief conversts to `StatusCode` based on the current state.
    * @return SUCCESS or FAILURE
    */
   StatusCode getStatus() const {
      return m_state == ERROR ? StatusCode::FAILURE : StatusCode::SUCCESS;
   }

   /**
    * @brief Resets the state to UNSCHEDULED.
    */
   void reset() {
      new(this)AlgExecState;
   }

   friend bool operator==(const AlgExecState& state1, const AlgExecState& state2) {
      return state1.m_state == state2.m_state;
   }

   friend bool operator!=(const AlgExecState& state1, const AlgExecState& state2) {
      return !(state1 == state2);
   }

   friend std::ostream& operator<<(std::ostream& os, const AlgExecState& state) {
      os << "AlgExecState: ";
      switch(state.getState()) {
         case AlgExecState::State::UNSCHEDULED: return os << "UNSCHEDULED";
         case AlgExecState::State::SCHEDULED: return os << "SCHEDULED";
         case AlgExecState::State::SUSPENDED: return os << "SUSPENDED";
         case AlgExecState::State::FINISHED: return os << "FINISHED";
         case AlgExecState::State::ERROR: return os << "ERROR";
      }
   }

private:
   State m_state{UNSCHEDULED};
};
