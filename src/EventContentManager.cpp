#include "EventContentManager.hpp"

#include <algorithm>
#include <cassert>
#include <iterator>
#include <set>

#include "AlgorithmBase.hpp"
#include "Assert.hpp"
#include "StatusCode.hpp"


namespace {
void setBits(const std::vector<std::string>& allObjectsVec,
             const std::vector<std::string>& objects, boost::dynamic_bitset<>& objBitset) {
   objBitset.reset();
   assert(objBitset.size() == allObjectsVec.size());
   for(const auto& obj : objects) {
      auto it = std::ranges::find(allObjectsVec, obj);
      std::size_t index = it - allObjectsVec.begin();
      assert(index < allObjectsVec.size());
      objBitset.set(index);
   }
}
}  // namespace


EventContentManager::EventContentManager(
    const std::vector<std::reference_wrapper<AlgorithmBase>>& algs)
    : m_algDependencies(algs.size()),
      m_algProducts(algs.size()),
      m_algContent(),
      m_storeContentMutex() {
   std::set<std::string> allObjectsSet;
   for(const auto& alg : algs) {
      const auto& deps = alg.get().dependencies();
      const auto& prods = alg.get().products();

      for(const auto& dep : deps) {
         allObjectsSet.insert(dep);
      }
      for(const auto& prod : prods) {
         allObjectsSet.insert(prod);
      }
   }
   std::vector<std::string> allObjectsVec;
   std::ranges::copy(allObjectsSet, std::back_inserter(allObjectsVec));

   for(std::size_t i = 0; i < algs.size(); ++i) {
      m_algDependencies[i].resize(allObjectsSet.size());
      m_algProducts[i].resize(allObjectsSet.size());
      setBits(allObjectsVec, algs[i].get().dependencies(), m_algDependencies[i]);
      setBits(allObjectsVec, algs[i].get().products(), m_algProducts[i]);
   }

   m_algContent.resize(allObjectsSet.size());
}


EventContentManager::EventContentManager(const EventContentManager& parent)
    : m_algDependencies(parent.m_algDependencies),
      m_algProducts(parent.m_algProducts),
      m_algContent(parent.m_algContent),
      m_storeContentMutex() {
}


EventContentManager& EventContentManager::operator=(const EventContentManager& E) {
   if(this == &E) {
      return *this;
   }

   m_algDependencies = E.m_algDependencies;
   m_algProducts = E.m_algProducts;
   m_algContent = E.m_algContent;
   return *this;
}


StatusCode EventContentManager::setAlgExecuted(std::size_t alg) {
   assert(alg < m_algDependencies.size());
   std::lock_guard<std::mutex> guard(m_storeContentMutex);
   m_algContent |= m_algProducts[alg];
   return StatusCode::SUCCESS;
}


bool EventContentManager::isAlgExecutable(std::size_t alg) const {
   assert(alg < m_algDependencies.size());
   std::lock_guard<std::mutex> guard(m_storeContentMutex);
   return m_algDependencies[alg].is_subset_of(m_algContent);
}


void EventContentManager::reset() {
   m_algContent.reset();
}
