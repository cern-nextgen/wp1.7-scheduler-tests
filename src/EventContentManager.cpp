#include "EventContentManager.hpp"

#include <algorithm>
#include <cassert>
#include <iterator>
#include <ostream>
#include <set>

#include "AlgorithmBase.hpp"
#include "StatusCode.hpp"


// /**
//  * @brief Helper function setting the bits in the bitset corresponding to the objects (names), with a mapping of
//  * name to index based on the position of the string in allObjectsVec.
//  * @param allObjectsVec vector of string mapping string position to bit position (reset at startup).
//  * @param objects list of string to be added to the bitset
//  * @param objBitset the bitset.
//  */
// void EventContentManager::DataObjColl_t::setBits(const std::vector<std::string>& allObjectsVec,
//              const std::vector<std::string>& objects) {
//    reset();
//    assert(size() == allObjectsVec.size());
//    for(const auto& obj : objects) {
//       auto it = std::ranges::find(allObjectsVec, obj);
//       std::size_t index = it - allObjectsVec.begin();
//       assert(index < allObjectsVec.size());
//       set(index);
//    }
// }


// EventContentManager::EventContentManager(
//     const std::vector<std::reference_wrapper<AlgorithmBase>>& algs)
//     : m_algDependencies(algs.size()),
//       m_algDependants(algs.size()),
//       m_algProducts(algs.size()),
//       m_algContent(),
//       m_storeContentMutex() {
//    std::set<std::string> allObjectsSet;
//    for(const auto& alg : algs) {
//       const auto& deps = alg.get().dependencies();
//       const auto& prods = alg.get().products();

//       for(const auto& dep : deps) {
//          allObjectsSet.insert(dep);
//       }
//       for(const auto& prod : prods) {
//          allObjectsSet.insert(prod);
//       }
//    }
//    std::vector<std::string> allObjectsVec;
//    std::ranges::copy(allObjectsSet, std::back_inserter(allObjectsVec));

//    for(std::size_t i = 0; i < algs.size(); ++i) {
//       m_algDependencies[i].resize(allObjectsSet.size());
//       m_algProducts[i].resize(allObjectsSet.size());
//       m_algDependencies[i].setBits(allObjectsVec, algs[i].get().dependencies());
//       m_algProducts[i].setBits(allObjectsVec, algs[i].get().products());
//    }

//    // Build the algorithm dependancy map. This is done via the products,
//    // so it's a 2 step process. We need the complete products and dependencies
//    // maps for this.
//    for (std::size_t i = 0; i < algs.size(); ++i) {
//       m_algDependants[i].resize(algs.size());
//    }
//    for (auto i = 0; i< algs.size(); ++i) {
//       // All the (j) products this algorithm depends on.
//       for (auto j=m_algDependencies[i].find_first();
//            j != boost::dynamic_bitset<>::npos;
//            j = m_algDependencies[i].find_next(j)) {
//          // Find the producer for this dependency. Hopefully only one.
//          if (std::ranges::count_if(m_algProducts, [&](const DataObjColl_t& prod) {
//             return prod.test(j);
//          }) != 1) {
//             std::cerr << "Error: Multiple or no producers for the same product found." << std::endl;
//             std::cerr << "Product index: " << j << "(" << allObjectsVec[j] << ")" << std::endl;
//             std::cerr << "Dependant algorithm index: " << i << std::endl;
//             std::cerr << "Algorithm(s) producing this product: ";
//             for (std::size_t k = 0; k < m_algProducts.size(); ++k) {
//                if (m_algProducts[k].test(j)) {
//                   std::cerr << k << " (" << algs[k].get().products()[j] << ") ";
//                }
//             }
//             assert(false);
//          }
//          // There is one and only one producer for this dependency.
//          auto producer = std::ranges::find_if(m_algProducts, [&](const DataObjColl_t& prod) {
//             return prod.test(j);
//          });
//          m_algDependants[producer - m_algProducts.begin()].set(i);
//       }
//    }

//    m_algContent.resize(allObjectsSet.size());
// }


// EventContentManager::EventContentManager(const EventContentManager& parent)
//     : m_algDependencies(parent.m_algDependencies),
//       m_algProducts(parent.m_algProducts),
//       m_algContent(parent.m_algContent),
//       m_storeContentMutex() {
// }


// EventContentManager& EventContentManager::operator=(const EventContentManager& E) {
//    if(this == &E) {
//       return *this;
//    }

//    m_algDependencies = E.m_algDependencies;
//    m_algProducts = E.m_algProducts;
//    m_algContent = E.m_algContent;
//    return *this;
// }


// StatusCode EventContentManager::setAlgExecuted(std::size_t alg) {
//    assert(alg < m_algDependencies.size());
//    std::lock_guard<std::mutex> guard(m_storeContentMutex);
//    m_algContent |= m_algProducts[alg];
//    return StatusCode::SUCCESS;
// }

StatusCode EventContentManager::setAlgExecuted(std::size_t alg, 
    const AlgorithmDependencyMap & depMap) {
   assert(alg < depMap.m_algDependencies.size());
   m_algContent |= depMap.m_algProducts[alg];
   return StatusCode::SUCCESS;
}

// std::vector<std::size_t> EventContentManager::getDependantAndReadyAlgs(
//     std::size_t algIdx) const {
//    assert(algIdx < m_algDependants.size());
//    std::lock_guard<std::mutex> guard(m_storeContentMutex);
//    std::vector<std::size_t> readyAlgs;

//    auto &deps = m_algDependants[algIdx];
//    std::size_t i = deps.find_first();
//    while (i != boost::dynamic_bitset<>::npos) {
//       if (m_algDependencies[i].is_subset_of(m_algContent)) {
//          readyAlgs.push_back(i);
//       }
//       i = deps.find_next(i);
//    }
//    return readyAlgs;
// }

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

// void EventContentManager::dumpContents(std::ostream& os) const {
//     std::lock_guard<std::mutex> guard(m_storeContentMutex);
//     os << "EventContentManager dump:\n";
//     os << "Dependencies per algorithm:\n";
//     for (size_t i = 0; i < m_algDependencies.size(); ++i) {
//         os << "  Alg " << i << ": ";
//         bool first = true;
//         for (size_t j = m_algDependencies[i].find_first(); j != boost::dynamic_bitset<>::npos; j = m_algDependencies[i].find_next(j)) {
//             if (!first) os << ", ";
//             os << j;
//             first = false;
//         }
//         os << "\n";
//     }
//     os << "Products per algorithm:\n";
//     for (size_t i = 0; i < m_algProducts.size(); ++i) {
//         os << "  Alg " << i << ": ";
//         bool first = true;
//         for (size_t j = m_algProducts[i].find_first(); j != boost::dynamic_bitset<>::npos; j = m_algProducts[i].find_next(j)) {
//             if (!first) os << ", ";
//             os << j;
//             first = false;
//         }
//         os << "\n";
//     }
//     os << "Dependants per algorithm:\n";
//     for (size_t i = 0; i < m_algDependants.size(); ++i) {
//         os << "  Alg " << i << ": ";
//         bool first = true;
//         for (size_t j = m_algDependants[i].find_first(); j != boost::dynamic_bitset<>::npos; j = m_algDependants[i].find_next(j)) {
//             if (!first) os << ", ";
//             os << j;
//             first = false;
//         }
//         os << "\n";
//     }
//     os << "Current event content bitset:\n  ";
//     bool first = true;
//     for (size_t j = m_algContent.find_first(); j != boost::dynamic_bitset<>::npos; j = m_algContent.find_next(j)) {
//         if (!first) os << ", ";
//         os << j;
//         first = false;
//     }
//     os << "\n";
// }

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
