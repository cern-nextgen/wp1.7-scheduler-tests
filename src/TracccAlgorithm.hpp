
#pragma once

// Local include(s).
#include "AlgorithmBase.hpp"
#include "EventContext.hpp"

// traccc include(s).
#include <traccc/clusterization/clusterization_algorithm.hpp>
#include <vecmem/memory/host_memory_resource.hpp>

/// Algorithm making use of traccc
class TracccAlgorithm : public AlgorithmBase
{

public:
   /// @name Function(s) inherited from @c AlgorithmBase
   /// @{

   /// Initialize the algorithm
   virtual StatusCode initialize() override;
   /// Execute the algorithm
   virtual AlgCoInterface execute(EventContext ctx) const override;
   /// Finalize the algorithm
   virtual StatusCode finalize() override;

   /// List the dependencies of the algorithm
   virtual const std::vector<std::string> &dependencies() const = 0;
   /// List the products of the algorithm
   virtual const std::vector<std::string> &products() const = 0;

   /// @}

private:
   /// (Host) memory resource to use in the clusterization algorithm
   vecmem::host_memory_resource m_mr;
   /// The traccc clusterization algorithm
   traccc::host::clusterization_algorithm m_clusterization{m_mr};

}; // class TracccAlgorithm
