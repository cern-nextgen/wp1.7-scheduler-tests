// Local include(s).
#include "TracccCudaAlgorithm.hpp"

#include <detray/detectors/bfield.hpp>
#include <detray/navigation/navigator.hpp>
#include <detray/propagator/rk_stepper.hpp>
#include <iostream>
#include <traccc/finding/finding_algorithm.hpp>
#include <traccc/fitting/fitting_algorithm.hpp>
#include <traccc/io/read_cells.hpp>
#include <traccc/io/read_detector.hpp>
#include <traccc/io/read_detector_description.hpp>
#include <traccc/seeding/seeding_algorithm.hpp>
#include <traccc/seeding/spacepoint_formation_algorithm.hpp>
#include <traccc/seeding/track_params_estimation.hpp>

#include "Assert.hpp"
#include "traccc/clusterization/clustering_config.hpp"
#include "traccc/cuda/clusterization/measurement_sorting_algorithm.hpp"
#include "traccc/cuda/finding/finding_algorithm.hpp"
#include "traccc/cuda/fitting/fitting_algorithm.hpp"
#include "traccc/cuda/seeding/seeding_algorithm.hpp"
#include "traccc/cuda/seeding/spacepoint_formation_algorithm.hpp"
#include "traccc/cuda/seeding/track_params_estimation.hpp"

TracccCudaAlgorithm::TracccCudaAlgorithm()
    : m_host_mr{},
      m_cuda_host_mr{},
      m_device_mr{},
      m_mr{m_device_mr, &m_cuda_host_mr},
      m_host_det_descr{m_host_mr},
      m_host_detector{m_host_mr},
      m_device_det_descr{},
      m_device_detector{},
      m_stream{},
      m_copy{m_stream.cudaStream()},
      m_ca_cuda{m_mr, m_copy, m_stream, traccc::clustering_config{256, 16, 8, 256}} {
}

StatusCode TracccCudaAlgorithm::initialize() {
   traccc::io::read_detector_description(m_host_det_descr,
                                         "geometries/odd/odd-detray_geometry_detray.json",
                                         "geometries/odd/odd-digi-geometric-config.json");
   // detector file, material file, grid file
   traccc::io::read_detector(m_host_detector,
                             m_host_mr,
                             "geometries/odd/odd-detray_geometry_detray.json",
                             "geometries/odd/odd-detray_material_detray.json",
                             "geometries/odd/odd-detray_surface_grids_detray.json");
   m_device_detector
       = detray::get_buffer(detray::get_data(m_host_detector), m_device_mr, m_copy);
   m_stream.synchronize();  // Synchronize needed here?
                            //
   // Initialize here instead of constructor so that m_device_det_descr gets the correct size.
   m_device_det_descr = traccc::silicon_detector_description::buffer{
       static_cast<traccc::silicon_detector_description::buffer::size_type>(
           m_host_det_descr.size()),
       m_device_mr};
   traccc::silicon_detector_description::data host_det_descr_data{
       vecmem::get_data(m_host_det_descr)};
   m_copy(host_det_descr_data, m_device_det_descr);


   return StatusCode::SUCCESS;
}

AlgorithmBase::AlgCoInterface TracccCudaAlgorithm::execute(EventContext ctx) const {
   traccc::edm::silicon_cell_collection::host cells{m_host_mr};

   traccc::io::read_cells(
       cells, ctx.eventNumber, "odd/geant4_10muon_10GeV/", &m_host_det_descr);
   traccc::edm::silicon_cell_collection::buffer cells_buffer{
       static_cast<unsigned int>(cells.size()), m_mr.main};
   m_copy(vecmem::get_data(cells), cells_buffer)->ignore();

   traccc::spacepoint_collection_types::buffer spacepoints_cuda_buffer{0, *m_mr.host};
   traccc::seed_collection_types::buffer seeds_cuda_buffer{0, *m_mr.host};
   traccc::measurement_collection_types::buffer measurements_cuda_buffer{0, *m_mr.host};
   measurements_cuda_buffer = m_ca_cuda(cells_buffer, m_device_det_descr);
   traccc::cuda::measurement_sorting_algorithm ms_cuda{m_copy, m_stream};
   ms_cuda(measurements_cuda_buffer);
   cudaLaunchHostFunc(ctx.stream, notifyScheduler, new EventContext{ctx});
   co_yield StatusCode::SUCCESS;

   // Should this happen inside initialize?
   // m_device_detector is also declared mutable otherwise assignment fails.
   traccc::default_detector::view device_detector_view = detray::get_data(m_device_detector);

   using device_spacepoint_formation_algorithm
       = traccc::cuda::spacepoint_formation_algorithm<traccc::default_detector::device>;
   device_spacepoint_formation_algorithm sf_cuda(m_mr, m_copy, m_stream);
   traccc::cuda::track_params_estimation tp_cuda{m_mr, m_copy, m_stream};

   spacepoints_cuda_buffer = sf_cuda(device_detector_view, measurements_cuda_buffer);
   cudaLaunchHostFunc(ctx.stream, notifyScheduler, new EventContext{ctx});
   co_yield StatusCode::SUCCESS;

   using stepper_type = detray::rk_stepper<detray::bfield::const_field_t::view_t,
                                           traccc::default_detector::host::algebra_type,
                                           detray::constrained_step<>>;
   using device_navigator_type = detray::navigator<const traccc::default_detector::device>;
   using device_finding_algorithm
       = traccc::cuda::finding_algorithm<stepper_type, device_navigator_type>;
   using device_fitting_algorithm = traccc::cuda::fitting_algorithm<
       traccc::kalman_fitter<stepper_type, device_navigator_type>>;

   device_finding_algorithm finding_alg_cuda{
       device_finding_algorithm::config_type{}, m_mr, m_copy, m_stream};
   device_fitting_algorithm fitting_alg_cuda{
       device_fitting_algorithm::config_type{}, m_mr, m_copy, m_stream};

   // Constant B field for the track finding and fitting
   const traccc::vector3 field_vec = {0.f, 0.f, traccc::seedfinder_config{}.bFieldInZ};
   const detray::bfield::const_field_t field = detray::bfield::create_const_field(field_vec);
   traccc::bound_track_parameters_collection_types::buffer params_cuda_buffer{0, *m_mr.host};

   traccc::cuda::seeding_algorithm sa_cuda{traccc::seedfinder_config{},
                                           {traccc::seedfinder_config{}},
                                           traccc::seedfilter_config{},
                                           m_mr,
                                           m_copy,
                                           m_stream};
   seeds_cuda_buffer = sa_cuda(spacepoints_cuda_buffer);
   cudaLaunchHostFunc(ctx.stream, notifyScheduler, new EventContext{ctx});
   co_yield StatusCode::SUCCESS;
   params_cuda_buffer = tp_cuda(spacepoints_cuda_buffer, seeds_cuda_buffer, field_vec);
   cudaLaunchHostFunc(ctx.stream, notifyScheduler, new EventContext{ctx});
   co_yield StatusCode::SUCCESS;

   traccc::track_candidate_container_types::buffer track_candidates_buffer = finding_alg_cuda(
       device_detector_view, field, measurements_cuda_buffer, params_cuda_buffer);
   cudaLaunchHostFunc(ctx.stream, notifyScheduler, new EventContext{ctx});
   co_yield StatusCode::SUCCESS;

   traccc::track_state_container_types::buffer track_states_buffer
       = fitting_alg_cuda(device_detector_view, field, track_candidates_buffer);

   co_return StatusCode::SUCCESS;
}

StatusCode TracccCudaAlgorithm::finalize() {
   return StatusCode::SUCCESS;
}

const std::vector<std::string> &TracccCudaAlgorithm::dependencies() const {
   static std::vector<std::string> deps = {};
   return deps;
}

const std::vector<std::string> &TracccCudaAlgorithm::products() const {
   static std::vector<std::string> prods = {};
   return prods;
}

