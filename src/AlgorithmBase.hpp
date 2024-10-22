#pragma once


#include <functional>
#include <string>
#include <utility>
#include <vector>

#include "Coroutines.hpp"
#include "StatusCode.hpp"


// Forward declarations.
struct EventContext;


class AlgorithmBase {
public:
   using AlgCoInterface = CoInterface<Promise<StatusCode, StatusCode>>;

   virtual ~AlgorithmBase() = default;

   virtual StatusCode initialize() = 0;

   // EventContext must be passed by value. Otherwise when coroutine is resumed reference
   // might be out of scope and no longer be valid. This is exactly the case in the Scheduler.
   virtual AlgCoInterface execute(EventContext ctx) const = 0;

   virtual StatusCode finalize() = 0;

   virtual const std::vector<std::string>& dependencies() const = 0;
   virtual const std::vector<std::string>& products() const = 0;

   static StatusCode for_all(const std::vector<std::reference_wrapper<AlgorithmBase>>& algs,
                             auto F, auto&&... args) {
      for(auto& alg : algs) {
         if(StatusCode status = (alg.get().*F)(std::forward<decltype(args)>(args)...);
            !status) {
            return status;
         }
      }
      return StatusCode::SUCCESS;
   }
};

