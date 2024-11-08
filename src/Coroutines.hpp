#pragma once


#include <coroutine>
#include <exception>
#include <utility>


// Coroutine interface.
template <typename PT>
class [[nodiscard]] CoInterface {
public:
   using promise_type = PT;

   using handle_type = std::coroutine_handle<promise_type>;

   CoInterface() : m_handle{} {
   }

   explicit CoInterface(const handle_type& handle) : m_handle{handle} {
   }

   CoInterface(const CoInterface&) = delete;

   CoInterface(CoInterface&& ct) noexcept : m_handle{std::exchange(ct.m_handle, nullptr)} {
   }

   CoInterface& operator=(const CoInterface&) = delete;

   CoInterface& operator=(CoInterface&& ct) {
      if(this != &ct) {
         if(m_handle) {
            m_handle.destroy();
         }
         m_handle = std::exchange(ct.m_handle, nullptr);
      }
      return *this;
   }

   ~CoInterface() {
      if(m_handle) {
         m_handle.destroy();
      }
   }

   bool resume() const {
      if(!m_handle || m_handle.done()) {
         return false;
      }
      m_handle.resume();
      return !m_handle.done();
   }

   bool isResumable() const {
      return m_handle && (!m_handle.done());
   }

   auto getYield() const {
      return m_handle.promise().m_yieldVal;
   }

   auto getReturn() const {
      return m_handle.promise().m_retVal;
   }

   bool empty() const {
      return m_handle.address() == nullptr;
   }

   void setEmpty() {
      *this = {};
   }

private:
   handle_type m_handle;
};


class PromiseBase {
public:
   // Do not suspend coroutine immediately.
   auto initial_suspend() {
      return std::suspend_never{};
   }

   // Should be suspended at the end and guarantee not to throw.
   auto final_suspend() noexcept {
      return std::suspend_always{};
   }

   // Deal with exceptions not handled locally inside coroutine.
   [[noreturn]] void unhandled_exception() {
      std::terminate();
   }
};


using EmptyCoYield = void;
using EmptyCoReturn = void;


template <typename T = void, typename U = void>
class Promise : protected PromiseBase {
   friend CoInterface<Promise>;

public:
   using PromiseBase::final_suspend;
   using PromiseBase::initial_suspend;
   using PromiseBase::unhandled_exception;

   // Creates coroutine object returned to the caller of the coroutine.
   auto get_return_object() {
      return CoInterface<Promise>{CoInterface<Promise>::handle_type::from_promise(*this)};
   }

   auto yield_value(const T& value) {
      m_yieldVal = value;
      return std::suspend_always{};
   }

   void return_value(const U& value) {
      m_retVal = value;
   }

private:
   T m_yieldVal;
   U m_retVal;
};


template <>
class Promise<EmptyCoYield, EmptyCoReturn> : protected PromiseBase {
public:
   using PromiseBase::final_suspend;
   using PromiseBase::initial_suspend;
   using PromiseBase::unhandled_exception;

   auto get_return_object() {
      return CoInterface<Promise>{CoInterface<Promise>::handle_type::from_promise(*this)};
   }

   void return_void() {
   }
};


template <typename T>
class Promise<T, EmptyCoReturn> : protected PromiseBase {
   friend CoInterface<Promise>;

public:
   using PromiseBase::final_suspend;
   using PromiseBase::initial_suspend;
   using PromiseBase::unhandled_exception;

   auto get_return_object() {
      return CoInterface<Promise>{CoInterface<Promise>::handle_type::from_promise(*this)};
   }

   auto yield_value(const T& value) {
      m_yieldVal = value;
      return std::suspend_always{};
   }

   void return_void() {
   }

private:
   T m_yieldVal;
};


template <typename U>
class Promise<EmptyCoYield, U> : protected PromiseBase {
   friend CoInterface<Promise>;

public:
   using PromiseBase::final_suspend;
   using PromiseBase::initial_suspend;
   using PromiseBase::unhandled_exception;

   auto get_return_object() {
      return CoInterface<Promise>{CoInterface<Promise>::handle_type::from_promise(*this)};
   }

   void return_value(const U& value) {
      m_retVal = value;
   }

private:
   U m_retVal;
};
