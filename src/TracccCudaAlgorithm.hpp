#pragma once

// Local include(s).
#include "AlgorithmBase.hpp"
#include "EventContext.hpp"

// traccc include(s).
#include <traccc/cuda/clusterization/clusterization_algorithm.hpp>
#include <traccc/cuda/utils/stream.hpp>
#include <traccc/geometry/detector.hpp>
#include <vecmem/memory/cuda/device_memory_resource.hpp>
#include <vecmem/memory/cuda/host_memory_resource.hpp>
#include <vecmem/memory/host_memory_resource.hpp>
#include <vecmem/utils/cuda/async_copy.hpp>

/// Algorithm making use of traccc
class TracccCudaAlgorithm : public AlgorithmBase {
public:
   TracccCudaAlgorithm();

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
   mutable vecmem::host_memory_resource m_host_mr;
   mutable vecmem::cuda::host_memory_resource m_cuda_host_mr;
   mutable vecmem::cuda::device_memory_resource m_device_mr;
   mutable traccc::memory_resource m_mr;

   traccc::silicon_detector_description::host m_host_det_descr;
   traccc::default_detector::host m_host_detector;
   traccc::silicon_detector_description::buffer m_device_det_descr;
   mutable traccc::default_detector::buffer m_device_detector;

   mutable traccc::cuda::stream m_stream;
   mutable vecmem::cuda::async_copy m_copy;

   traccc::cuda::clusterization_algorithm m_ca_cuda;
};  // class TracccCudaAlgorithm

