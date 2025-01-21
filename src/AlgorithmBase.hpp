#pragma once


#include <functional>
#include <string>
#include <string_view>
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

   const std::vector<std::string>& dependencies() const {
      return m_dependencies;
   }

   const std::vector<std::string>& products() const {
      return m_products;
   }

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

protected:
   template <typename T>
   StatusCode addDependency(std::string_view name) {
      m_dependencies.push_back(std::string{name} + " " + typeid(T).name());
      return StatusCode::SUCCESS;
   }

   template <typename T>
   StatusCode addProduct(std::string_view name) {
      m_products.push_back(std::string{name} + " " + typeid(T).name());
      return StatusCode::SUCCESS;
   }

private:
   std::vector<std::string> m_dependencies, m_products;
};
