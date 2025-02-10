#pragma once


#include <detray/detectors/bfield.hpp>
#include <detray/navigation/navigator.hpp>
#include <detray/propagator/rk_stepper.hpp>
#include <traccc/cuda/clusterization/clusterization_algorithm.hpp>
#include <traccc/cuda/utils/stream.hpp>
#include <traccc/finding/finding_algorithm.hpp>
#include <traccc/fitting/fitting_algorithm.hpp>
#include <traccc/geometry/detector.hpp>
#include <traccc/io/read_detector_description.hpp>
#include <traccc/seeding/seeding_algorithm.hpp>
#include <traccc/seeding/spacepoint_formation_algorithm.hpp>
#include <traccc/seeding/track_params_estimation.hpp>
#include <vecmem/memory/cuda/device_memory_resource.hpp>
#include <vecmem/memory/cuda/host_memory_resource.hpp>
#include <vecmem/memory/host_memory_resource.hpp>
#include <vecmem/utils/cuda/async_copy.hpp>

#include "AlgorithmBase.hpp"
#include "EventContext.hpp"
#include "traccc/cuda/clusterization/measurement_sorting_algorithm.hpp"
#include "traccc/cuda/finding/finding_algorithm.hpp"
#include "traccc/cuda/fitting/fitting_algorithm.hpp"
#include "traccc/cuda/seeding/seeding_algorithm.hpp"
#include "traccc/cuda/seeding/spacepoint_formation_algorithm.hpp"
#include "traccc/cuda/seeding/track_params_estimation.hpp"


class TracccCudaAlgorithm : public AlgorithmBase {
public:
   TracccCudaAlgorithm(int numEvents);

   StatusCode initialize() override;
   AlgCoInterface execute(EventContext ctx) const override;
   StatusCode finalize() override;

private:
   using device_spacepoint_formation_algorithm
       = traccc::cuda::spacepoint_formation_algorithm<traccc::default_detector::device>;

   using stepper_type = detray::rk_stepper<detray::bfield::const_field_t::view_t,
                                           traccc::default_detector::host::algebra_type,
                                           detray::constrained_step<>>;

   using device_navigator_type = detray::navigator<const traccc::default_detector::device>;

   using device_finding_algorithm
       = traccc::cuda::finding_algorithm<stepper_type, device_navigator_type>;

   using device_fitting_algorithm = traccc::cuda::fitting_algorithm<
       traccc::kalman_fitter<stepper_type, device_navigator_type>>;

private:
   mutable vecmem::host_memory_resource m_host_mr;
   mutable vecmem::cuda::host_memory_resource m_cuda_host_mr;
   mutable vecmem::cuda::device_memory_resource m_device_mr;
   mutable traccc::memory_resource m_mr;

   traccc::silicon_detector_description::host m_host_det_descr;
   traccc::default_detector::host m_host_detector;
   traccc::silicon_detector_description::buffer m_device_det_descr;
   traccc::default_detector::buffer m_device_detector;
   traccc::default_detector::view m_device_detector_view;

   mutable traccc::cuda::stream m_stream;
   mutable vecmem::cuda::async_copy m_copy;

   traccc::cuda::clusterization_algorithm m_ca_cuda;
   traccc::cuda::measurement_sorting_algorithm m_ms_cuda;
   device_spacepoint_formation_algorithm m_sf_cuda;
   traccc::cuda::seeding_algorithm m_sa_cuda;
   traccc::cuda::track_params_estimation m_tp_cuda;
   device_finding_algorithm m_finding_alg_cuda;
   device_fitting_algorithm m_fitting_alg_cuda;

   std::vector<traccc::edm::silicon_cell_collection::host> m_cells;
   int m_numEvents;
};

