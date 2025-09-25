#pragma once

#include <iostream>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <utility>

#define SC_CHECK(EXP)                                             \
   do {                                                           \
      const StatusCode sc = EXP;                                  \
      if(!sc) {                                                   \
         std::cout << "Failed to execute: " << #EXP << std::endl; \
         return StatusCode::FAILURE;                              \
      }                                                           \
   } while(false)

#define SC_CHECK_YIELD(EXP)                                       \
   do {                                                           \
      const StatusCode sc = EXP;                                  \
      if(!sc) {                                                   \
         std::cout << "Failed to execute: " << #EXP << std::endl; \
         co_yield StatusCode::FAILURE;                            \
      }                                                           \
   } while(false)

#define SC_CHECK_CO_RETURN(EXP)                                   \
   do {                                                           \
      const StatusCode sc = EXP;                                  \
      if(!sc) {                                                   \
         std::cout << "Failed to execute: " << #EXP << std::endl; \
         co_return StatusCode::FAILURE;                            \
      }                                                           \
   } while(false)
class [[nodiscard]] StatusCode {
public:
   enum class ErrorCode : int { SUCCESS = 0, FAILURE = 1 };

   static constexpr ErrorCode SUCCESS = ErrorCode::SUCCESS;
   static constexpr ErrorCode FAILURE = ErrorCode::FAILURE;

   // Default constructor
   StatusCode() = default;

   // Copy constructor
   StatusCode(const StatusCode& other) {
      std::shared_lock<std::shared_mutex> lock(other.m_mutex); // Shared lock for thread-safe read
      m_code = other.m_code;
      m_msg = other.m_msg;
   }

   // Move constructor
   StatusCode(StatusCode&& other) noexcept {
      std::unique_lock<std::shared_mutex> lock(other.m_mutex); // Exclusive lock for thread-safe move
      m_code = std::move(other.m_code);
      m_msg = std::move(other.m_msg);
   }

   // Copy assignment operator
   StatusCode& operator=(const StatusCode& other) {
      if (this != &other) {
         std::unique_lock<std::shared_mutex> lockThis(m_mutex, std::defer_lock);
         std::shared_lock<std::shared_mutex> lockOther(other.m_mutex, std::defer_lock);
         std::lock(lockThis, lockOther); // Lock both mutexes
         m_code = other.m_code;
         m_msg = other.m_msg;
      }
      return *this;
   }

   // Move assignment operator
   StatusCode& operator=(StatusCode&& other) noexcept {
      if (this != &other) {
         std::unique_lock<std::shared_mutex> lockThis(m_mutex, std::defer_lock);
         std::unique_lock<std::shared_mutex> lockOther(other.m_mutex, std::defer_lock);
         std::lock(lockThis, lockOther); // Lock both mutexes
         m_code = std::move(other.m_code);
         m_msg = std::move(other.m_msg);
      }
      return *this;
   }

   StatusCode(ErrorCode code, std::string msg)
       : m_code{code},
         m_msg{"<<<<< " + std::move(msg) + " >>>>>"} {
   }

   StatusCode(ErrorCode code) : m_code{code}, m_msg{""} {
   }

   ~StatusCode() {
      // Take the mutex exclusively during destruction
      std::unique_lock<std::shared_mutex> lock(m_mutex);
   }

   explicit operator bool() const {
      // Shared lock for read access
      std::shared_lock<std::shared_mutex> lock(m_mutex);
      return m_code == SUCCESS;
   }

   friend bool operator==(const StatusCode& status1, const StatusCode& status2) {
      // Shared lock for read access
      std::shared_lock<std::shared_mutex> lock1(status1.m_mutex);
      std::shared_lock<std::shared_mutex> lock2(status2.m_mutex);
      return status1.m_code == status2.m_code;
   }

   friend bool operator!=(const StatusCode& status1, const StatusCode& status2) {
      // Shared lock for read access
      std::shared_lock<std::shared_mutex> lock1(status1.m_mutex);
      std::shared_lock<std::shared_mutex> lock2(status2.m_mutex);
      return !(status1 == status2);
   }

   const std::string& appendMsg(const std::string& s) {
      // Exclusive lock for modifying the message
      std::unique_lock<std::shared_mutex> lock(m_mutex);
      return m_msg += std::string{"\n"} += "***** " + s + " *****";
   }

   std::string what() const {
      // Shared lock for read access
      std::shared_lock<std::shared_mutex> lock(m_mutex);
      if (m_code == SUCCESS && m_msg.empty()) {
         return "<<<<< SUCCESS >>>>>";
      } else if (m_code == FAILURE && m_msg.empty()) {
         return "<<<<< FAILURE >>>>>";
      }
      return m_msg;
   }

private:
   mutable std::shared_mutex m_mutex; // Mutex for thread safety
   ErrorCode m_code{SUCCESS};
   std::string m_msg{};
};
