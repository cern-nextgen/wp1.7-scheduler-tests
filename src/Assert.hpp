#pragma once


#ifndef NDEBUG


#include <cstdlib>
#include <iostream>
#include <string>


namespace detail {


inline std::string trace_err(const char* file, const char* func, int line) {
   return std::string{"File: "} + std::string{file} + std::string{", function: "}
        + std::string{func} + std::string{", line: "} + std::to_string(line) + std::string{":"}
        + std::string{"\n"};
}


}  // namespace detail


#define ASSERT_MSG(condition, msg)                                                           \
   do {                                                                                      \
      if(!(condition)) {                                                                     \
         std::cerr << msg << detail::trace_err(__FILE__, __func__, __LINE__) << (#condition) \
                   << " failed." << std::endl;                                               \
         std::abort();                                                                       \
      }                                                                                      \
   } while(false)


#define ASSERT(condition) ASSERT_MSG(condition, "")


#else


#define ASSERT_MSG(condition, msg) ((void)0)
#define ASSERT(condition) ((void)0)


#endif
