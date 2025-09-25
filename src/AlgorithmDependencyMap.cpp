#include "AlgorithmDependencyMap.hpp"

#include <algorithm>
#include <cassert>
#include <iterator>
#include <ostream>
#include <set>

#include "AlgorithmBase.hpp"
#include "StatusCode.hpp"


/**
 * @brief Helper function setting the bits in the bitset corresponding to the objects (names), with a mapping of
 * name to index based on the position of the string in allObjectsVec.
 * @param allObjectsVec vector of string mapping string position to bit position (reset at startup).
 * @param objects list of string to be added to the bitset
 * @param objBitset the bitset.
 */
void AlgorithmDependencyMap::DataObjColl_t::setBits(const std::vector<std::string>& allObjectsVec,
             const std::vector<std::string>& objects) {
   reset();
   assert(size() == allObjectsVec.size());
   for(const auto& obj : objects) {
      auto it = std::ranges::find(allObjectsVec, obj);
      std::size_t index = it - allObjectsVec.begin();
      assert(index < allObjectsVec.size());
      set(index);
   }
}


AlgorithmDependencyMap::AlgorithmDependencyMap(
    const std::vector<std::reference_wrapper<AlgorithmBase>>& algs)
    : m_algDependencies(algs.size()),
      m_algDependents(algs.size()),
      m_algProducts(algs.size()) {
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
      m_algDependencies[i].setBits(allObjectsVec, algs[i].get().dependencies());
      m_algProducts[i].setBits(allObjectsVec, algs[i].get().products());
   }

   // Build the algorithm dependancy map. This is done via the products,
   // so it's a 2 step process. We need the complete products and dependencies
   // maps for this.
   for (std::size_t i = 0; i < algs.size(); ++i) {
      m_algDependents[i].resize(algs.size());
   }
   for (std::size_t i = 0; i< algs.size(); ++i) {
      // All the (j) products this algorithm depends on.
      for (auto j=m_algDependencies[i].find_first();
           j != boost::dynamic_bitset<>::npos;
           j = m_algDependencies[i].find_next(j)) {
         // Find the producer for this dependency. Hopefully only one.
         if (std::ranges::count_if(m_algProducts, [&](const DataObjColl_t& prod) {
            return prod.test(j);
         }) != 1) {
            std::cerr << "Error: Multiple or no producers for the same product found." << std::endl;
            std::cerr << "Product index: " << j << "(" << allObjectsVec[j] << ")" << std::endl;
            std::cerr << "Dependant algorithm index: " << i << std::endl;
            std::cerr << "Algorithm(s) producing this product: ";
            for (std::size_t k = 0; k < m_algProducts.size(); ++k) {
               if (m_algProducts[k].test(j)) {
                  std::cerr << k << " (" << algs[k].get().products()[j] << ") ";
               }
            }
            assert(false);
         }
         // There is one and only one producer for this dependency.
         auto producer = std::ranges::find_if(m_algProducts, [&](const DataObjColl_t& prod) {
            return prod.test(j);
         });
         m_algDependents[producer - m_algProducts.begin()].set(i);
      }
   }
}

bool AlgorithmDependencyMap::isAlgIndependent(std::size_t algIdx) const {
   assert(algIdx < m_algDependencies.size());
   return m_algDependencies[algIdx].none();
}

