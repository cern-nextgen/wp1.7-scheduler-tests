#pragma once


#include <boost/dynamic_bitset/dynamic_bitset.hpp>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <mutex>
#include <vector>
#include "AlgorithmDependencyMap.hpp"


// Forward declarations.
class AlgorithmBase;
class StatusCode;


/**
 * @brief Manager for the content of the event store, handling algorithm execution and dependencies.
 * @note It is intended to be used inside the event slot scheduling and protected by the slot mutex.
 */

class EventContentManager {
public:

   EventContentManager() = default;

    /**
      * @brief Default constructor, only used at initialization of the scheduler slot.
      */
   void resize(const AlgorithmDependencyMap& depMap) {
      m_algContent.clear();
      m_algContent.resize(depMap.m_algDependencies[0].size());
    }

   EventContentManager(const EventContentManager& E) = delete;
   EventContentManager& operator=(const EventContentManager& E) = delete;


   /**
    * @brief Set one of the algorithms as having finished its execution.
    * All the products that the algorithm
    */
   StatusCode setAlgExecuted(std::size_t alg, 
       const AlgorithmDependencyMap& depMap);

   /**
    * @brief Get the list of algorithms that are ready to be executed following the completion of the given algorithm.
    * @param algIdx The index of the algorithm that has completed.
    * @param depMap Reference to the AlgorithmDependencyMap containing the algorithm dependencies.
    * @return A vector of indices of the algorithms that are ready to be executed.
    */
   std::vector<std::size_t> getDependentAndReadyAlgs (
       std::size_t algIdx, const AlgorithmDependencyMap& depMap) const;

   /**
    * @brief Check if all algorithms have completed execution.
    * @return true if all algorithms have completed, false otherwise.
    */
   bool isEventComplete() const;

   /**
    * @brief Check if an algorithm's data dependencies are availble.
    * @param algIdx The index of the algorithm to check.
    * @param depMap Reference to the AlgorithmDependencyMap containing the algorithm dependencies.
    * @todo Could be renamed to `areAlgoDependenciesMet()`.
    */
   bool isAlgExecutable(std::size_t algIdx, const AlgorithmDependencyMap& depMap) const;

   /**
    * @brief Reset the content manager to no algorithm executed.
    */
   void reset();

   /**
    * @brief Dump the contents of the AlgorithmDependencyMap to std::ostream.
    * @param os Output stream to write to.
    */
   void dumpContents(const AlgorithmDependencyMap& depMap, std::ostream& os = std::cout) const;

  private:
   /// Current content of the event store.
   AlgorithmDependencyMap::DataObjColl_t m_algContent;
};
