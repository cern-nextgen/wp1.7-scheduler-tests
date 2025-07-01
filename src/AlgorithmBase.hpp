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

/**
 * @brief Base class for algorithms. Algorithms express dependencies on products
 */
class AlgorithmBase {
public:
   using AlgCoInterface = CoInterface<Promise<StatusCode, StatusCode>>;

   virtual ~AlgorithmBase() = default;

   virtual StatusCode initialize() = 0;

   /**
    * @brief Execute the algorithm kernel by kernel, with callback in between.
    * @param ctx The event context.
    * @return A coroutine interface for the execution.
    */
   virtual AlgCoInterface execute(EventContext ctx) const = 0;

   /**
    * @brief Execute the algorithm with all kernels launched without delay and just a final synchronization.
    * @param ctx The event context.
    * @return A coroutine interface for the execution.
    */
   virtual AlgCoInterface executeStraight(EventContext ctx) const;

   /**
    * @brief Execute the algorithm with all kernels launched without delay and just a final synchronization in a task
    * All such task will be executed in a single thread to work around CUDA RT performance drop when using multiple
    * threads.
    * @param ctx The event context.
    * @return A coroutine interface for the execution.
    */
   virtual AlgCoInterface executeStraightDelegated(EventContext ctx) const;

   /**
    * @brief Execute the algorithm with all kernels launched without delay and just a final synchronization in a task
    * All such tasks will be mutexed to work around CUDA RT performance drop when using multiple.
    * @param ctx The event context.
    * @return A coroutine interface for the execution.
    */
   virtual AlgCoInterface executeStraightMutexed(EventContext ctx) const;

   /**
    * @brief Execute the algorithm as a CUDA graph with just a final synchronization.
    * @param ctx The event context.
    * @return A coroutine interface for the execution.
    */
   virtual AlgCoInterface executeGraph(EventContext ctx) const;

   /**
    * @brief Execute the algorithm as a CUDA graph with cached graphs to minimize contention.
    * @param ctx The event context.
    * @return A coroutine interface for the execution.
    */
   virtual AlgCoInterface executeCachedGraph(EventContext ctx) const;

   /**
    * @brief Execute the algorithm as a CUDA graph with cached graphs to minimize contention. CUDA calls are delegated to a single thread.
    * @param ctx The event context.
    * @return A coroutine interface for the execution.
    */
   virtual AlgCoInterface executeCachedGraphDelegated(EventContext ctx) const;

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
