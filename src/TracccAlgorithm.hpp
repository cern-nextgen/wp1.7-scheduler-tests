
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
   StatusCode initialize() override;
   /// Execute the algorithm
   AlgCoInterface execute(EventContext ctx) const override;
   /// Finalize the algorithm
   StatusCode finalize() override;

   /// List the dependencies of the algorithm
   virtual const std::vector<std::string> &dependencies() const override;
   /// List the products of the algorithm
   virtual const std::vector<std::string> &products() const override;

   /// @}

private:
   /// (Host) memory resource to use in the clusterization algorithm
   mutable vecmem::host_memory_resource m_mr;
   /// The traccc clusterization algorithm
   traccc::host::clusterization_algorithm m_clusterization{m_mr};

}; // class TracccAlgorithm
