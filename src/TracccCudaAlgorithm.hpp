#pragma once


#include <traccc/cuda/clusterization/clusterization_algorithm.hpp>
#include <traccc/cuda/utils/stream.hpp>
#include <traccc/geometry/detector.hpp>
#include <vecmem/memory/cuda/device_memory_resource.hpp>
#include <vecmem/memory/cuda/host_memory_resource.hpp>
#include <vecmem/memory/host_memory_resource.hpp>
#include <vecmem/utils/cuda/async_copy.hpp>

#include "AlgorithmBase.hpp"
#include "EventContext.hpp"


class TracccCudaAlgorithm : public AlgorithmBase {
public:
   TracccCudaAlgorithm();

   StatusCode initialize() override;
   AlgCoInterface execute(EventContext ctx) const override;
   StatusCode finalize() override;

private:
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
};

