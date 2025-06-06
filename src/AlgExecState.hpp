#pragma once


#include <sstream>

#include "StatusCode.hpp"

/**
 * @brief Wrapper fpr algorithms states. With stream operator and reset to default state (UNSCHEDULED).
 */
class AlgExecState {
public:
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
