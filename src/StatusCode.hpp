#pragma once


#include <string>
#include <utility>


class [[nodiscard]] StatusCode {
public:
   enum class ErrorCode : int { SUCCESS = 0, FAILURE = 1 };

   static constexpr ErrorCode SUCCESS = ErrorCode::SUCCESS;
   static constexpr ErrorCode FAILURE = ErrorCode::FAILURE;

   StatusCode() = default;
   StatusCode(const StatusCode&) = default;
   StatusCode(StatusCode&&) = default;
   StatusCode& operator=(const StatusCode&) = default;
   StatusCode& operator=(StatusCode&&) = default;

   StatusCode(ErrorCode code, std::string msg)
       : m_code{code},
         m_msg{"***** " + std::move(msg) + " *****"} {
   }

   StatusCode(ErrorCode code) : m_code{code}, m_msg{""} {
   }

   explicit operator bool() const {
      return m_code == SUCCESS;
   }

   bool isSuccess() const {
      return m_code == SUCCESS;
   }

   bool isFailure() const {
      return m_code == FAILURE;
   }

   friend bool operator==(const StatusCode& status1, const StatusCode& status2) {
      return status1.m_code == status2.m_code;
   }

   friend bool operator!=(const StatusCode& status1, const StatusCode& status2) {
      return !(status1 == status2);
   }

   auto& appendMsg(std::string s) {
      return m_msg += std::string{"\n"} += "***** " + s + " *****";
   }

   std::string what() const {
      if(m_code == SUCCESS && m_msg.empty()) {
         return "**** SUCCESS ****";
      } else if(m_code == FAILURE && m_msg.empty()) {
         return "**** FAILURE ****";
      }
      return m_msg;
   }

private:
   ErrorCode m_code{SUCCESS};
   std::string m_msg{};
};
