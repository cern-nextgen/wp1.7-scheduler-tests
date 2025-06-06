#pragma once


#include <boost/dynamic_bitset/dynamic_bitset.hpp>
#include <cstdlib>
#include <functional>
#include <mutex>
#include <vector>


// Forward declarations.
class AlgorithmBase;
class StatusCode;

/**
 * @brief Scheduler slot level manager for the content of the event. It manages the dependencies and statuses of the algorithms,
 * as well as an array of references to the algorithms themselves.
 * The algorithms dependencies and products are stored as bitsets. The results of an algorithm is recorded in an all-or-nothing manner,
 * at the full completion of the algorithm execution.
 * @see `setBits()` in `EventContentManager.cpp`.
 */
class EventContentManager {
public:
   /**
    * @brief Default constructor, only used at initialization of the scheduler slot.
    */
   EventContentManager() = default;


   /**
    * @brief Constructs the EventContentManager with a list of algorithms.
    * It initializes the dependencies and products bitsets for each algorithm. The string to bitset mapping
    * only exists during the execution of the algorithm.
    * @param algs A vector of references to AlgorithmBase objects.
    */
   explicit EventContentManager(
       const std::vector<std::reference_wrapper<AlgorithmBase>>& algs);
   EventContentManager(const EventContentManager& E);
   EventContentManager& operator=(const EventContentManager& E);

   /**
    * @brief Set one of the algorithms as having finished its execution.
    * All the products that the algorithm
    * @note Thread safe 
    */
   StatusCode setAlgExecuted(std::size_t alg);

   // Check if an algorithm is ready to be run.
   /**
    * @brief Check if an algorithm's data dependencies are availble.
    * @note Thread safe
    * @todo Could be renamed to `areAlgoDependenciesMet()`.
    */
   bool isAlgExecutable(std::size_t alg) const;

   // Reset the event store content.
   /**
    * @brief Reset the event store content.s
    */
   void reset();

private:
   /// Type used for bitset.
   typedef boost::dynamic_bitset<> DataObjColl_t;

   /// Per-algorithm dependencies
   std::vector<DataObjColl_t> m_algDependencies;

   /// Per-algorithm products
   std::vector<DataObjColl_t> m_algProducts;

   /// Current content of the event store.
   DataObjColl_t m_algContent;

   /// Mutex: this object is not thread-safe.
   mutable std::mutex m_storeContentMutex;
};
