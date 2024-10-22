#pragma once


#include <boost/dynamic_bitset/dynamic_bitset.hpp>
#include <cstdlib>
#include <functional>
#include <mutex>
#include <vector>


// Forward declarations.
class AlgorithmBase;
class StatusCode;


class EventContentManager {
public:
   EventContentManager() = default;

   explicit EventContentManager(
       const std::vector<std::reference_wrapper<AlgorithmBase>>& algs);
   EventContentManager(const EventContentManager& E);
   EventContentManager& operator=(const EventContentManager& E);

   // Set one of the algorithms as having finished its execution.
   StatusCode setAlgExecuted(std::size_t alg);

   // Check if an algorithm is ready to be run.
   bool isAlgExecutable(std::size_t alg) const;

   // Reset the event store content.
   void reset();

private:
   typedef boost::dynamic_bitset<> DataObjColl_t;

   std::vector<DataObjColl_t> m_algDependencies;
   std::vector<DataObjColl_t> m_algProducts;
   DataObjColl_t m_algContent;
   mutable std::mutex m_storeContentMutex;
};

