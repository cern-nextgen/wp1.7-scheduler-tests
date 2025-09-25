#include "EventContentManager.hpp"

#include <algorithm>
#include <cassert>
#include <iterator>
#include <ostream>
#include <set>

#include "AlgorithmBase.hpp"
#include "StatusCode.hpp"

StatusCode EventContentManager::setAlgExecuted(std::size_t alg, 
    const AlgorithmDependencyMap & depMap) {
   assert(alg < depMap.m_algDependencies.size());
   m_algContent |= depMap.m_algProducts[alg];
   return StatusCode::SUCCESS;
}

std::vector<std::size_t> EventContentManager::getDependentAndReadyAlgs(std::size_t algIdx, 
  const AlgorithmDependencyMap & depMap) const {
   assert(algIdx < depMap.m_algDependents.size());
   std::vector<std::size_t> readyAlgs;

   auto &deps = depMap.m_algDependents[algIdx];
   std::size_t i = deps.find_first();
   while (i != boost::dynamic_bitset<>::npos) {
      if (depMap.m_algDependencies[i].is_subset_of(m_algContent)) {
         readyAlgs.push_back(i);
      }
      i = deps.find_next(i);
   }
   return readyAlgs;
}

bool EventContentManager::isAlgExecutable(std::size_t algIdx, 
    const AlgorithmDependencyMap& depMap) const {
   assert(algIdx < depMap.m_algDependencies.size());
   return depMap.m_algDependencies[algIdx].is_subset_of(m_algContent);
}


void EventContentManager::reset() {
   m_algContent.reset();
}

void EventContentManager::dumpContents(const AlgorithmDependencyMap& depMap, std::ostream& os) const {
    os << "EventContentManager dump:\n";
    os << "Dependencies per algorithm:\n";
    for (size_t i = 0; i < depMap.m_algDependencies.size(); ++i) {
        os << "  Alg " << i << ": ";
        bool first = true;
        for (size_t j = depMap.m_algDependencies[i].find_first(); j != boost::dynamic_bitset<>::npos; j = depMap.m_algDependencies[i].find_next(j)) {
            if (!first) os << ", ";
            os << j;
            first = false;
        }
        os << "\n";
    }
    os << "Products per algorithm:\n";
    for (size_t i = 0; i < depMap.m_algProducts.size(); ++i) {
        os << "  Alg " << i << ": ";
        bool first = true;
        for (size_t j = depMap.m_algProducts[i].find_first(); j != boost::dynamic_bitset<>::npos; j = depMap.m_algProducts[i].find_next(j)) {
            if (!first) os << ", ";
            os << j;
            first = false;
        }
        os << "\n";
    }
    os << "Dependants per algorithm:\n";
    for (size_t i = 0; i < depMap.m_algDependents.size(); ++i) {
        os << "  Alg " << i << ": ";
        bool first = true;
        for (size_t j = depMap.m_algDependents[i].find_first(); j != boost::dynamic_bitset<>::npos; j = depMap.m_algDependents[i].find_next(j)) {
            if (!first) os << ", ";
            os << j;
            first = false;
        }
        os << "\n";
    }
    os << "Current event content bitset:\n  ";
    bool first = true;
    for (size_t j = m_algContent.find_first(); j != boost::dynamic_bitset<>::npos; j = m_algContent.find_next(j)) {
        if (!first) os << ", ";
        os << j;
        first = false;
    }
    os << "\n";
}
